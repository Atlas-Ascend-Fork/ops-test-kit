#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TfBackend: TF-generic intermediate layer.

Holds the shared TF implementation (numpy<->tf conversion, device move via
``tf.device``). Hardware-specific subclasses override ``is_npu`` only.
"""

from __future__ import annotations

import numpy as np

from .base import Backend
from .framework_detector import TF_NPU_ONLY_PREFIXES


class TfBackend(Backend):
    """Shared TF implementation; subclass per hardware."""

    tf_device_type: str = ""

    def is_available(self) -> bool:
        import tensorflow as tf

        return bool(tf.config.list_physical_devices(self.tf_device_type))

    def device_count(self) -> int:
        import tensorflow as tf

        return len(tf.config.list_physical_devices(self.tf_device_type))

    def to_device(self, tensor, dev_id: int = 0, preserve_stride: bool = False):
        import tensorflow as tf

        if isinstance(tensor, (tf.Tensor, tf.Variable)):
            # 张量由 np_to_tf_inputs 在当前默认设备上下文创建, 透传即可;
            # from_numpy 往返无收益(NPU 路径更是纯 D2H+H2D 开销), 且对
            # Variable 会丢失变量身份。fallback 路径的 numpy 入参仍走转换
            return tensor
        t = self.from_numpy(tensor)
        if self.tf_device_type and self.tf_device_type != "cpu":
            with tf.device(f"/{self.tf_device_type}:{dev_id}"):
                return tf.identity(t)
        return t

    def synchronize(self, dev_id: int = 0):
        pass

    def from_numpy(self, arr):
        import tensorflow as tf

        from ttk.utilities.dtypes import normalize_to_tf_dtype

        if arr is None:
            return None
        # 0-D 本就连续；ascontiguousarray 会把标量提升为 (1,)，破坏 TF 0-D 参数校验
        arr = np.ascontiguousarray(arr) if arr.ndim else arr
        arr = normalize_to_tf_dtype(arr)
        return tf.convert_to_tensor(arr)

    def to_numpy(self, tensor, safe: bool = False):
        if tensor is None:
            return None
        return tensor.numpy()

    def has_device(self) -> bool:
        return self.tf_device_type != "cpu"

    def device_name(self, dev_id: int = 0) -> str:
        return self.device_type()

    def is_npu(self) -> bool:
        return False

    def soc_series(self) -> str:
        return self.device_name()

    def clone(self, tensor):
        import tensorflow as tf

        if isinstance(tensor, tf.Variable):
            # 变量含状态, 克隆需复制当前值到新变量(随默认设备放置)
            return tf.Variable(tensor.read_value())
        return tf.identity(tensor)

    def stage_mutable_rounds(self, args, kwargs, keys, total):  # noqa: ARG002
        """TF 覆写：同一 tf.Variable 逐轮 assign 复位（语义等价于换新克隆）。

        npu_device 的 tf.function 对新 Variable 触发重追踪，状态算子（scatter
        族）的副作用会跨 concrete function 累积——逐轮换新变量时第 N 轮实际
        应用 N 次更新（纯 CPU/TF 无此行为，同变量复用亦无）。同变量 + 逐轮
        assign 复位不触发重追踪，每轮从备份初值起恰好一次更新；与主线 graph
        腿 reset_mutable_refs 同一策略（闭包捕获/trace 固化下克隆不可换）。
        """
        backups = {key: self.clone(args[key] if isinstance(key, int) else kwargs[key]) for key in keys}

        def stage(_round):
            for key, backup in backups.items():
                var = args[key] if isinstance(key, int) else kwargs[key]
                var.assign(backup)

        return stage

    def is_npu_only(self, api_name: str) -> bool:
        return api_name.startswith(TF_NPU_ONLY_PREFIXES)

    def wrap_eager_callable(self, resolved, api_name=None):
        """eager 包装已迁通路层（framework_api/tf/eager_runner，经 adapter.wrap_eager）。

        保留基类契约（外部 backend 子类可覆写）；本类不再覆写——TfBackend 侧
        原实现（Resource* handle 转换）与 NpuTfBackend 侧（tf.function NPU 分发）
        均在 framework_api/tf/eager_runner.wrap_eager_callable 按 backend 设备形态分发。
        """
        return resolved

    def supports_graph_mode(self) -> bool:
        return True

    def supports_format_cast(self) -> bool:
        return False

    def needs_numpy_fallback(self, testcase) -> bool:
        return not testcase.is_tf_dtype_support()

    def inputs_from_numpy(self, testcase, raw_inputs):
        # L1 张量构造经 framework_hooks 注册表注入（tf/tensor_binding.np_to_tf_inputs，
        # 依赖通路层 tf/stateful 的 OpDef 元数据——common 不反向 import 通路）。
        # 未注册（独立使用 backend）时退基础转换。
        from .framework_hooks import get_input_factory

        fn = get_input_factory("tf")
        if fn is not None:
            return fn(testcase, raw_inputs)
        return [self.from_numpy(x) if x is not None else None for x in raw_inputs]
