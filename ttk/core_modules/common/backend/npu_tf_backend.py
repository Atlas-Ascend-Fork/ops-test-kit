#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""NPU TF backend — uses npu_device plugin for Ascend NPU.

Corresponds to NpuTorchBackend (torch_npu). open() swaps the global TF
context to _ContextWithDefaultDevice with NPU as default device. It is
deferred to ``set_device(dev_id)`` — called at the execution phases, AFTER
input generation and golden (both CPU-scoped; TF device tensors are rebuilt
from raw_inputs after open) — rather than __init__ or input generation:
E2E workers are forked from the parent process, and an open() in the parent
leaves the children with dead GE threads inherited across fork (aclprofStart
segfaults). The parent only queries metadata (has_device / device_type) and
must not touch the device.

NPU dispatch: ops inside tf.function compile through GE and run on NPU;
bare eager calls run on CPU; tensor/device strings report CPU:0 regardless
of actual execution. Execution consumes NPU-born tensors only — the
CPU-scoped input/golden phase materializes its own tensors, and device
tensors are rebuilt from raw_inputs after open.

as_default() monkey-patches ops.device to _device_consistent_with_context,
which ignores the argument device path and always uses ctx.default_device
(NPU:0).  Therefore to_device / device_scope need not (and cannot) place
tensors on a specific NPU via tf.device — all ops auto-dispatch to NPU.
CPU-side placement (e.g. golden generation) must go through
tensorflow.python.eager.context.device, which the patch does not intercept
(see cpu_tf_backend.cpu_device_context).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

from .tf_backend import TfBackend


class NpuTfBackend(TfBackend):
    """NPU TF backend via npu_device plugin."""

    tf_device_type = "NPU"
    _segment_name = "npu"

    _opened_device = None

    def __init__(self):
        # open 延迟到 set_device(dev_id)（_run_profile 在输入生成前统一调用，
        # 与 torch 的 torch_npu.npu.set_device 同一时机、同一语义）。此处不能
        # open：E2E worker 经 fork 拉起，父进程 open 后 C++ GE 线程不随 fork
        # 存活，子进程继承的句柄已坏（aclprofStart 段错误）。父进程仅做元数据
        # 查询（has_device/device_type），不需要设备。
        pass

    def is_npu(self) -> bool:
        return True

    def is_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("npu_device") is not None
        except Exception:
            return False

    def device_count(self) -> int:
        return 1

    def to_device(self, tensor, dev_id=0, preserve_stride=False):
        import tensorflow as tf

        if isinstance(tensor, (tf.Tensor, tf.Variable)):
            # np_to_tf_inputs 已在 set_device 之后的 NPU 默认设备上下文创建,
            # 透传即可; from_numpy 往返是纯 D2H+H2D 开销。fallback 路径的
            # numpy 入参仍走 from_numpy 转换
            return tensor
        return self.from_numpy(tensor)

    def synchronize(self, dev_id=0):
        pass

    def device_scope(self, dev_id=0):
        return nullcontext()

    def set_device(self, dev_id: int = 0):
        self._ensure_npu_opened(dev_id)

    def _ensure_npu_opened(self, dev_id):
        if NpuTfBackend._opened_device is None:
            import npu_device

            handle = npu_device.open(dev_id)
            handle.as_default()
            NpuTfBackend._opened_device = dev_id
        elif NpuTfBackend._opened_device != dev_id:
            raise RuntimeError(
                f"npu_device only supports one device; already opened "
                f"{NpuTfBackend._opened_device}, cannot open {dev_id}"
            )

    def device_name(self, dev_id=0):
        try:
            from ttk.core_modules.common.device import get_npu_hw_info

            from ..device import DSMIInterface

            platform = DSMIInterface().get_chip_info(dev_id).get_complete_platform()
            return get_npu_hw_info(platform).get("short_soc_version", platform)
        except Exception:
            return self._segment_name or "NPU"

    def soc_series(self):
        return self.device_name()

    def supports_graph_mode(self) -> bool:
        return True

    def wrap_eager_callable(self, resolved, api_name=None):
        """eager 包装已迁通路层（framework_api/tf/eager_runner，经 adapter.wrap_eager）。

        原实现（tf.function 包装——npu_device 只对 tf.function 内的 op 触发 GE
        编译，裸 eager 静默回落 CPU）在 framework_api/tf/eager_runner._wrap_eager_npu。
        """
        return resolved

    def set_deterministic_level(self, level):
        try:
            import npu_device

            cfg = npu_device.global_options()
            cfg.deterministic = level
            npu_device.global_options()
        except Exception as e:
            logging.warning(f"Failed to set TF deterministic: {e}")
