#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU TF backend (for golden baseline or no-device testing).

When npu_device.open() has been called (NPU TF backend active), NPU becomes
the TF default device via _ContextWithDefaultDevice.  CPU golden generation
must explicitly place tensors and ops on CPU to avoid silently running on NPU.

Placement goes through tensorflow.python.eager.context.device — the eager
primitive beneath tf.device — NOT tf.device itself: npu_device.as_default()
overwrites ops.device so that tf.device's argument is ignored (everything
routes to NPU), while that very patch is implemented on top of
context.device, which therefore stays usable.  This keeps the CPU-side
placement independent of the process-global npu_device open scope.
"""

from __future__ import annotations

from .tf_backend import TfBackend


class CpuTfBackend(TfBackend):
    """CPU TF backend (for golden baseline or no-device testing)."""

    tf_device_type = "cpu"
    profile = {"profiler": {"activities": ["CPU"]}}
    _segment_name = "cpu"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def from_numpy(self, arr):
        import numpy as np
        import tensorflow as tf

        from ttk.utilities.dtypes import normalize_to_tf_dtype

        if arr is None:
            return None
        # 0-D 本就连续；ascontiguousarray 会把标量提升为 (1,)，破坏 TF 0-D 参数校验
        arr = np.ascontiguousarray(arr) if arr.ndim else arr
        arr = normalize_to_tf_dtype(arr)
        with cpu_device_context():
            return tf.convert_to_tensor(arr)

    def device_scope(self, dev_id=0):
        return cpu_device_context()

    def synchronize(self, dev_id=0):
        pass

    def has_device(self) -> bool:
        return False


def cpu_device_context():
    """CPU 设备上下文：CPU 侧张量物化的放置原语（golden / 输入生成阶段）。

    npu_device.as_default() 以 ops.device 覆写吃掉 tf.device 的设备参数
    （恒路由 NPU），而该 patch 的实现自身建立在 context.device 之上——
    故本原语在 patch 后仍可用，CPU 物化不随 npu_device.open 的进程级
    作用域漂移。
    """
    from tensorflow.python.eager import context as tf_context

    return tf_context.device("/CPU:0")
