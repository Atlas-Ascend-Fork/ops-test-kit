#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
torch 域 CubeMathType 精度模式配置，经 TorchAdapter.setup_device_state 触达。

进程级 torch_npu 状态、幂等。基线复位按需执行：KEEP_DTYPE +
allow_hf32=False 基线与 torch_npu 进程默认并不恒同（conv.allow_hf32
默认 True；910A 上 fp32 matmul 默认内部 USE_FP16），且 option 只写不清
——未携带该键的用例若也写基线会改动其默认数值通路。故仅当本用例携带
该键、或本 worker 前序用例已改写过（storage 标记，防跨用例状态泄漏）
时才写基线再按值覆盖；否则不触碰 torch_npu（亦不导入）：
  0: KEEP_DTYPE（保持原始 dtype）/ 1: ALLOW_FP32_DOWN_PRECISION /
  2: USE_FP16 / 3: USE_HF32（同时置 matmul/conv allow_hf32=True）/
  4: 高精度档（FORCE_GRP_ACC_FOR_FP32，新版本为 USE_FP32_ADD——按枚举值
  取成员，对成员改名版本兼容）。
conv.cube_math_type 在无此配置项的 torch_npu 版本为静默 no-op，赋值保留
以兼容支持版本。仅 NPU + torch 框架生效（TF/xpu 不涉及，亦不导入
torch_npu）。
"""

import logging

from ...common.backend import detect_framework


def cube_math_type_value(testcase):
    """读取用例 attributes 的 CubeMathType 值；非 torch 框架恒 None。"""
    if detect_framework(testcase.api_name) != "torch":
        return None
    attributes = getattr(testcase, "attributes", None) or {}
    return attributes.get("CubeMathType")


def setup_cube_math_type(backend, testcase, process_ctx):
    """按用例 CubeMathType 配置 torch_npu 立方计算精度模式（matmul/conv 类算子）。

    非 NPU 或非 torch 直接返回；未携带该键且前序用例未改写过时不导入、
    不写入 torch_npu。process_ctx.storage 的 ``_cube_math_type_dirty``
    标记记录本进程是否已改写过，跨用例防状态泄漏。
    """
    if not backend.is_npu() or detect_framework(testcase.api_name) != "torch":
        return

    cube_math_type_val = cube_math_type_value(testcase)
    # 未携带该键且前序用例未改写过：保持 torch_npu 进程默认（不导入、不写入）
    if cube_math_type_val is None and not process_ctx.storage.get("_cube_math_type_dirty", False):
        return

    import torch_npu

    cmt_value = torch_npu.npu.CubeMathType.KEEP_DTYPE
    torch_npu.npu.matmul.cube_math_type = cmt_value
    torch_npu.npu.conv.cube_math_type = cmt_value
    torch_npu.npu.matmul.allow_hf32 = False
    torch_npu.npu.conv.allow_hf32 = False
    # 一旦写入即无法恢复 torch_npu 未设置态（option 只写不清），标记此后恒须复位
    process_ctx.storage["_cube_math_type_dirty"] = True

    # 复位路径：前序改写过、本用例未携带该键——基线已就位，静默返回
    if cube_math_type_val is None:
        return

    try:
        cmt = torch_npu.npu.CubeMathType(int(cube_math_type_val))
    except (ValueError, TypeError):
        logging.warning(
            "[CubeMathType] invalid value %r, expect int in 0-4; keep KEEP_DTYPE baseline",
            cube_math_type_val,
        )
        return
    torch_npu.npu.matmul.cube_math_type = cmt
    torch_npu.npu.conv.cube_math_type = cmt
    if cmt == torch_npu.npu.CubeMathType.USE_HF32:
        torch_npu.npu.matmul.allow_hf32 = True
        torch_npu.npu.conv.allow_hf32 = True
