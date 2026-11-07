#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for non-contiguous stride preservation via Backend.clone / to_device.

clone() and to_device(preserve_stride=True) must keep the original (possibly
non-contiguous) stride of a tensor, unlike .clone() / .to(device) / .npu()
which materialize non-contiguous views into contiguous tensors.
"""

import numpy as np
import pytest
import torch

from ttk.core_modules.common.backend.torch_backend import TorchBackend


def _make_non_contiguous(shape=(1, 128, 8, 4), stride=(16384, 64, 4, 1), dtype=torch.float32, device="cpu"):
    """Build a non-contiguous view via as_strided over a larger storage."""
    storage = torch.arange(
        max(stride[0] * shape[0], 1) if len(stride) == len(shape) else np.prod(shape),
        dtype=dtype,
        device=device,
    ).reshape(-1)
    return torch.as_strided(storage, shape, stride, 0)


class TestClonePreservingStride:
    """clone() 必须保留非连续张量的 stride（不同于 .clone() 会连续化）。"""

    def _backend(self):
        """构造 CPU TorchBackend 实例。"""
        tb = TorchBackend()
        tb.torch_lib = "cpu"
        tb.profile = {}
        return tb

    def test_clone_contiguous_falls_back_to_clone(self):
        """clone(连续张量) 回退到 .clone()。"""
        backend = self._backend()
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        out = backend.clone(t)
        assert out is not t
        assert torch.equal(out, t)
        assert out.is_contiguous()

    def test_clone_non_contiguous_stride_preserved(self):
        """非连续张量 clone：步长保留、数据一致。"""
        backend = self._backend()
        t = _make_non_contiguous()
        assert not t.is_contiguous()
        out = backend.clone(t)
        assert out.stride() == t.stride()
        assert not out.is_contiguous()
        assert torch.equal(out, t)

    def test_clone_bfloat16_stride_preserved(self):
        """bf16 非连续张量 clone 后 stride/data 不变。"""
        backend = self._backend()
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.bfloat16)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        out = backend.clone(t)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()
        assert torch.equal(out, t)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.parametrize("view_kind", ["conjugate", "imaginary"])
    def test_clone_preserves_lazy_view_values_and_storage(self, dtype, view_kind):
        """共轭及虚部取负视图的克隆保留数值、布局和独立存储。"""
        backend = self._backend()
        values = torch.arange(24, dtype=torch.float64)
        storage = torch.complex(values, values + 1).to(dtype)
        source = storage.as_strided((3, 2), (7, 2), 3).conj()
        if view_kind == "imaginary":
            source = source.imag
        expected = source.clone()

        out = backend.clone(source)

        assert torch.equal(out, expected)
        assert out.shape == source.shape
        assert out.stride() == source.stride()
        assert out.storage_offset() == source.storage_offset()
        assert out.is_conj() == source.is_conj()
        assert out.is_neg() == source.is_neg()
        assert out.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        assert list(out.untyped_storage()) == list(source.untyped_storage())
        out.add_(2)
        assert torch.equal(out, expected + 2)
        assert torch.equal(source, expected)

    @pytest.mark.parametrize("view_kind", ["conjugate", "imaginary"])
    @pytest.mark.parametrize("key", [0, "input"])
    def test_mutable_rounds_keep_lazy_view_inputs(self, view_kind, key):
        """每轮原地运算均从同一逻辑输入开始，源张量保持原值。"""
        backend = self._backend()
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        source = torch.complex(values, values + 1).T.conj()
        if view_kind == "imaginary":
            source = source.imag
        expected = source.clone()
        args = [source] if isinstance(key, int) else []
        kwargs = {key: source} if isinstance(key, str) else {}
        load = backend.prepare_mutable_rounds(args, kwargs, [key], total=3)

        for i in range(3):
            load(i)
            current = args[key] if isinstance(key, int) else kwargs[key]
            current.add_(2)
            assert torch.equal(current, expected + 2)
        assert torch.equal(source, expected)
