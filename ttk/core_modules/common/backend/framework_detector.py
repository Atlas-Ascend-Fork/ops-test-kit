#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""框架探测：api_name 前缀 → torch/tf/onnx（backend 与 api_param 的 dtype 谓词共用）。"""

from typing import Optional

# TF 通路的 api_name 前缀。npu_device/npu_bridge 是 TF Adapter 暴露的 NPU 自定义算子
# (如 npu_device.compat.v1.tbe.npu_vector_ops.lamb_apply_optimizer_assign) —— 它们本质是
# TF 图算子, 必须按 TF 路由, 否则会落进 torch 分支解析失败。
TF_API_PREFIXES = ("tf.", "tensorflow.", "npu_device.", "npu_bridge.")

# NPU-only 的 TF 侧前缀：tf.npu_* 扩展与 TF Adapter 暴露的自定义算子
# （npu_device./npu_bridge.）——无 CPU kernel，golden 路径（纯 CPU 语义）不可直跑，
# 须走插件 golden 或 golden_api（e2e golden 三级回退的快速失败判据）。
TF_NPU_ONLY_PREFIXES = ("tf.npu_", "npu_device.", "npu_bridge.")

# onnx 通路的 api_name 前缀（onnx.<OpType>，不带 opset 版本——环境注册表最新即权威）。
# torch.onnx.* 以 "torch." 开头不在此列。
ONNX_API_PREFIXES = ("onnx.",)


def detect_framework(api_name: str) -> str:
    """Detect framework ('torch'/'tf'/'onnx') from api_name prefix.

    Routing rule:
        TF_API_PREFIXES 前缀 -> "tf"
        ONNX_API_PREFIXES 前缀 -> "onnx"
        everything else (torch., torch_npu., torch.ops., torch.Tensor.) -> "torch"
    """
    if not api_name:
        return "torch"
    if api_name.startswith(TF_API_PREFIXES):
        return "tf"
    if api_name.startswith(ONNX_API_PREFIXES):
        return "onnx"
    return "torch"


def is_inplace_tensor_method(api_name: str, framework: Optional[str] = None) -> bool:
    """Check if api_name is an inplace tensor method.

    torch: torch.Tensor.xxx_ (trailing underscore)
    tf / onnx: never (无原地 Tensor 方法)
    """
    if framework is None:
        framework = detect_framework(api_name)
    if framework != "torch":
        return False
    if not api_name:
        return False
    parts = api_name.split(".")
    return len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor" and parts[-1].endswith("_")
