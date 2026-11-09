#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TorchBackend: torch-generic intermediate layer.

Holds the shared torch implementation (numpy<->torch conversion, device move via
``getattr(torch, self.torch_lib)``). Hardware-specific subclasses override
``is_npu`` only; device_type() comes from Backend (config-driven _segment_name).

``device_name`` is implemented here (hardware model via
``torch.<torch_lib>.get_device_name``) — the Backend default is the
framework-neutral degrade to device_type(). CpuTorchBackend keeps its
device_name override (no ``torch.cpu.get_device_name`` exists).
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from ....utilities import get, numpy_to_torch_tensor, torch_to_numpy_tensor
from .base import Backend


def np_to_torch_inputs(testcase, raw_inputs):
    """Convert to torch tensors preserving non-contiguous views via torch.as_strided.

    Uses testcase.np_storages (contiguous) for from_numpy, then rebuilds
    non-contiguous views from testcase metadata (view_shape/view_stride/view_offset),
    same approach as aclnn input_generation.
    """
    np_storages = getattr(testcase, "np_storages", None)
    if np_storages is None:
        return [torch.from_numpy(np.ascontiguousarray(arr)) if arr is not None else None for arr in raw_inputs]
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = testcase.flat_tensor_dtypes
    result = []
    for idx, raw in enumerate(raw_inputs):
        if raw is None:
            result.append(None)
            continue
        storage = np_storages[idx]
        dtype = get(flat_dtypes, idx) if flat_dtypes else None
        t = numpy_to_torch_tensor(storage, is_complex32="complex32" in str(dtype or ""))
        v_shape = flat_shapes[idx]
        v_stride = testcase.flat_view_stride(idx)
        v_offset = testcase.flat_view_offset(idx)
        if t.dim() == 0 and v_shape == ():
            result.append(t)
        else:
            result.append(torch.as_strided(t, v_shape, v_stride, v_offset))
    return result


class TorchBackend(Backend):
    """Shared torch implementation; subclass per hardware."""

    # torch module attribute name ('npu'/'mlu'/...); injected by _build
    # (config-driven). CpuTorchBackend pins 'cpu' as a class attribute.
    torch_lib: str

    def device_name(self, dev_id: int = 0) -> str:
        """Hardware MODEL name via torch.<torch_lib>.get_device_name."""
        return getattr(torch, self.torch_lib).get_device_name(dev_id)

    def is_available(self) -> bool:
        return bool(getattr(torch, self.torch_lib).is_available())

    def device_count(self) -> int:
        return getattr(torch, self.torch_lib).device_count()

    def to_device(self, tensor, dev_id: int = 0, preserve_stride: bool = False):
        if preserve_stride:
            return self._to_device_preserving_stride(tensor, dev_id)
        t = self.from_numpy(tensor)
        return getattr(t, self.torch_lib)(dev_id)

    def _to_device_preserving_stride(self, tensor, dev_id):
        """Move a (possibly non-contiguous) tensor to device preserving stride."""
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            tensor = self.from_numpy(tensor)
        if self.torch_lib == "cpu":
            return tensor
        if tensor.is_contiguous():
            return getattr(tensor, self.torch_lib)(dev_id)
        return self._clone_full_storage(tensor, f"{self.torch_lib}:{dev_id}")

    def synchronize(self, dev_id: int = 0):
        getattr(torch, self.torch_lib).synchronize(dev_id)

    def clear_cache(self, dev_id: int = 0):
        # torch.<lib>.empty_cache 无参且各 lib 可用性不一（torch.cpu 早期版本缺失），
        # 缺 API 静默跳过：清理是尽力而为，不得让用例失败。
        empty = getattr(getattr(torch, self.torch_lib, None), "empty_cache", None)
        if callable(empty):
            empty()

    def reset(self, dev_id: int = 0):
        """用例收尾缓存回收（编排钩子，worker 每 case 后调用）。

        dynamo cache 按代码对象累计且无淘汰：GraphNetwork.forward 为所有通用
        图模式用例共享（worker 进程跨用例复用），触顶 cache_size_limit（默认
        8）后新用例静默回退 eager 单算子执行——TTK 每 case 全新 torch.compile，
        无跨 case 复用诉求，收尾统一 reset。gc 后 clear_cache 顺带回收
        AclGraph 私有显存池。"""
        import torch._dynamo

        torch._dynamo.reset()
        gc.collect()
        self.clear_cache(dev_id)

    def from_numpy(self, arr):
        return numpy_to_torch_tensor(arr)

    def to_numpy(self, tensor, safe: bool = False):
        tensor = tensor.detach()
        if safe:
            tensor = tensor.clone()
        return torch_to_numpy_tensor(tensor.cpu().contiguous())

    def has_device(self) -> bool:
        return True

    def _clone_full_storage(self, tensor, dst_device=None):
        """Clone a non-contiguous tensor preserving stride AND stride-gap data.

        ``empty_strided`` + ``copy_`` only allocates storage for the visible
        elements of the view, leaving stride-gap memory uninitialized.  An
        operator that reads into those gaps sees garbage that differs between
        clones and the original tensor — which is why the last profiling round
        (using the original) can diverge from earlier rounds (using clones).

        To fix this we clone the *entire* underlying storage (gaps included)
        and rebuild the non-contiguous view via ``as_strided``.
        """
        src_storage = tensor.untyped_storage()
        elem_size = tensor.element_size()
        storage_numel = src_storage.size() // elem_size
        src_flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(src_storage, 0, (storage_numel,), (1,))
        new_flat = src_flat.to(dst_device) if dst_device else src_flat.clone()
        result = torch.as_strided(new_flat, tensor.shape, tensor.stride(), tensor.storage_offset())
        # set_ 只恢复物理存储；共轭和取负标记决定视图的逻辑数值。
        if tensor.is_conj():
            result = result.conj()
        if tensor.is_neg():
            result = torch._neg_view(result)
        # src_flat 由 torch.empty 起底，requires_grad 默认 False，重建视图会丢标记；
        # 非连续输入经此路径到设备执行时需把源 tensor 的 requires_grad 带过去。
        if tensor.requires_grad:
            result.requires_grad_(True)
        return result

    def clone(self, tensor):
        """Clone a tensor preserving its non-contiguous stride and gap data.

        Unlike ``torch.Tensor.clone()``, which materializes a non-contiguous
        view into a contiguous tensor, this clones the full underlying storage
        (so stride-gap data is identical to the original) and rebuilds the
        non-contiguous view via ``as_strided``.
        """
        if tensor is None:
            return None
        if tensor.is_contiguous():
            return tensor.clone()
        return self._clone_full_storage(tensor)

    def is_npu_only(self, api_name: str) -> bool:
        return api_name.startswith(("torch_npu.", "torch.npu"))

    def supports_graph_mode(self) -> bool:
        return True

    def supports_format_cast(self) -> bool:
        return self.is_npu()

    def needs_numpy_fallback(self, testcase) -> bool:
        return not testcase.is_torch_dtype_support()

    def inputs_from_numpy(self, testcase, raw_inputs):
        if self.needs_numpy_fallback(testcase):
            return list(raw_inputs)
        return np_to_torch_inputs(testcase, raw_inputs)
