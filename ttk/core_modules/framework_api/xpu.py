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
E2E XPU 三方采集的输入派生（worker._dispatch_xpu / _collect_xpu 消费）。

由用例声明（tensor_list_dist / output_tensor_indexes / param plan）把 flat
numpy 输入还原为三方分发所需的 per-slot 输入列表、dtype 与参数名。
"""

from ttk.utilities.container import nest_by_distribution


def e2e_xpu_inputs(testcase, raw_inputs):
    """raw_inputs (flat numpy) re-nested, pure outputs filtered → top-level input slots."""
    dist = testcase.tensor_list_dist
    nested = nest_by_distribution(list(raw_inputs), dist) if dist else list(raw_inputs)
    out_indices = set(testcase.output_tensor_indexes or ())
    # output_tensor_indexes 是 top-level 槽位索引（对齐 _check_output_slots
    # 与 e2e_xpu_input_names）：TensorList 槽也只占一个槽位
    return [slot for real_idx, slot in enumerate(nested) if real_idx not in out_indices]


def e2e_xpu_input_dtypes(testcase):
    """Logical tensor dtypes (pure outputs filtered) for XPU schema.

    过滤规则与 e2e_xpu_inputs 一致（输出槽剔除、TensorList 保持嵌套），dtype
    为 CSV 声明值（如 complex32），与 raw_inputs 的 tensor_list_dist 重排一致。
    """
    dist = testcase.tensor_list_dist
    flat_dtypes = testcase.flat_tensor_dtypes or ()
    nested = nest_by_distribution(list(flat_dtypes), dist) if dist else list(flat_dtypes)
    out_indices = set(testcase.output_tensor_indexes or ())
    return [slot for real_idx, slot in enumerate(nested) if real_idx not in out_indices]


def e2e_xpu_input_names(testcase):
    """Input tensor param names (pure outputs filtered) for XPU schema."""
    plan = testcase.get_param_plan()
    if plan is None:
        return []
    out_indices = set(testcase.output_tensor_indexes or ())
    names = []
    real_idx = 0
    for p in plan.overload_params:
        if p.is_tensor_like and p.name != "out":
            if real_idx not in out_indices:
                names.append(p.name)
            real_idx += 1
    return names
