#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
onnx 域 backend：numpy 直通域（无框架 tensor）。

- OnnxBackend 基类：from_numpy/to_numpy/to_device 全 passthrough——H2D 延迟到
  OmSession（按 om desc size 分配）；supports_format_cast=False；
  needs_numpy_fallback=False；
- CpuOnnxBackend：golden 与输入生成侧占位（golden 实际走 ReferenceEvaluator，
  不经 backend 执行）；
- NpuOnnxBackend：**零 ACL**——设备探测/身份全走 DSMI（common/device 既有设施），
  libascendcl 的加载与运行时归 framework_api/onnx/runtime.py。
  依赖方向约束（framework_hooks.py）：common 不得 import 通路层——backend 层
  若 import AclModelInterface 即反向依赖，故设备侧只做「记录」，执行由通路层完成。
"""

__all__ = ["CpuOnnxBackend", "NpuOnnxBackend", "OnnxBackend"]

import logging

import numpy as np

from .base import Backend

_log = logging.getLogger(__name__)


class OnnxBackend(Backend):
    """onnx 域基类：numpy 直通域（无框架 tensor）。

    result_to_numpy 覆写为 ndarray 直通（基类对 ndarray 走 np.array 拷贝）；
    stage_mutable_rounds 沿用基类默认（ONNX 无原地写，keys 恒空 → no-op）。
    """

    profile: dict = {}

    def from_numpy(self, arr):
        return arr

    def to_numpy(self, tensor, safe=False):  # noqa: ARG002  # safe 仅为 torch detach 语义
        return tensor

    def to_device(self, tensor, dev_id=0, preserve_stride=False):  # noqa: ARG002
        return tensor  # 延迟上板：真正 H2D 在 OmSession（需按 om desc size 分配）

    def result_to_numpy(self, result, copy=False):
        if result is None:
            return [None]
        if isinstance(result, np.ndarray):
            return [result.copy() if copy else result]
        if isinstance(result, (tuple, list)):
            out = []
            for item in result:
                if item is None:
                    out.append(None)
                elif isinstance(item, np.ndarray):
                    out.append(item.copy() if copy else item)
                elif hasattr(item, "numpy") and callable(item.numpy):
                    arr = self.to_numpy(item)
                    out.append(arr.copy() if copy else arr)
                else:
                    out.append(np.array(item))
            return out
        return [np.array(result)]

    def supports_graph_mode(self):
        return False  # CPU 侧不可执行 om（仅 golden 占位）

    def supports_format_cast(self):
        return False

    def needs_numpy_fallback(self, testcase):  # noqa: ARG002
        return False


class CpuOnnxBackend(OnnxBackend):
    """CPU 占位 backend：golden/输入生成侧（golden 走 ReferenceEvaluator）。"""

    _segment_name = "cpu"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def synchronize(self, dev_id=0):  # noqa: ARG002
        return

    def has_device(self) -> bool:
        return False


class NpuOnnxBackend(OnnxBackend):
    """NPU onnx 后端：设备探测/身份全走 DSMI，零 libascendcl 触碰。

    set_device 仅记录 dev_id——真正 aclrtSetDevice 由 onnx/runtime 在 OmSession
    建立时惰性执行（execute.py 保证 set_device 先于 wrap_eager 注入）。
    """

    _segment_name = "npu"

    def __init__(self):
        self._dev_id = None

    def supports_graph_mode(self):
        return True  # 静态图（cst）支持（= om 执行）；dyn/aclgraph 由 run_graph 快速失败

    def is_available(self) -> bool:
        return self.device_count() > 0

    def device_count(self) -> int:
        from ..device.dsmi.interface import DSMIInterface

        try:
            return DSMIInterface().get_device_count()
        except Exception as e:  # noqa: BLE001  # 探测失败按无设备处理（get_backend 回落 CPU）
            _log.warning("onnx backend DSMI device probe failed: %s", e)
            return 0

    def device_name(self, dev_id: int = 0) -> str:
        """DSMI 完整芯片名（如 Ascend910_9362）——atc --soc_version 直传值。"""
        from ..device.dsmi.interface import DSMIInterface

        try:
            info = DSMIInterface().get_chip_info(dev_id)
        except Exception as e:  # noqa: BLE001  # DSMI 异常按未知芯片处理（调用方报错可见）
            _log.warning("onnx backend DSMI chip info failed on device %s: %s", dev_id, e)
            return f"npu{dev_id}"
        if info is None:
            return f"npu{dev_id}"
        return info.get_complete_platform()

    def soc_series(self):
        from ..device import get_npu_hw_info

        dev_id = 0 if self._dev_id is None else self._dev_id
        try:
            return get_npu_hw_info(self.device_name(dev_id)).get("short_soc_version", self.device_name(dev_id))
        except Exception as e:  # noqa: BLE001  # platform ini 缺失时退化为完整芯片名
            _log.warning("onnx backend soc series resolve failed: %s", e)
            return self.device_name(dev_id)

    def is_npu(self) -> bool:
        return True

    def set_device(self, dev_id: int = 0):
        self._dev_id = dev_id  # 仅记录：ACL aclrtSetDevice 由 onnx/runtime 惰性执行

    def synchronize(self, dev_id=0):  # noqa: ARG002
        return  # aclmdlExecute 同步语义

    def has_device(self) -> bool:
        return True

    # peak-memory 钩子走基类默认（reset no-op / max 返回 None，
    # base.py 的 B027 豁免即为此语义）——零 ACL 的本 backend 无 allocator 峰值
    # 可报，np.memory 列不采集、不误报 0，无需覆写。
