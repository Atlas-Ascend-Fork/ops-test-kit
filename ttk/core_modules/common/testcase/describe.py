#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""spec.describe 参数收集：未识别 CSV 列 → params dict（模式无关）。

多输入算子（如 sparse_flash_mla 的 18 输入）在 CSV 里用 B/S1/N1/D 等
抽象参数列替代逐张量形状声明。这些列不在任何 testcase 的
complete_headers 里（"未识别列"），框架把它们收集为 params 交给
spec.describe(params)，由算子侧组装成 {shapes, dtypes} 回填 testcase。
收集只发生在行内的通路形状/类型字段（kernel/geir 的 input_shapes/
input_dtypes，aclmn/e2e 的 tensor_view_shapes/tensor_dtypes）任一缺省时
——两者均已显式声明的 CSV 行不触发。
"""

__all__ = ["collect_describe_params"]


from .field_parser import process_eval


def collect_describe_params(testcase) -> dict:
    """未识别 CSV 列 → params dict（仅非空单元格；键名原样透传）。

    值解析用受限 eval（字面量容器 + 白名单函数，见 field_parser._eval）；
    解析失败（裸词如 ``BSND``/``bfloat16``）退回原始字符串——数值/容器
    参数拿到类型化值，枚举/布局参数拿到字符串。
    """
    legit = set(testcase.get_all_legit_headers())
    params = {}
    for key, raw in (testcase.original_dict or {}).items():
        if key in legit or raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        params[key] = _param_value(text)
    return params


def _param_value(text: str):
    try:
        return process_eval(text)
    except Exception:  # noqa: BLE001 受限求值尽力而为（裸词退回字符串）
        return text
