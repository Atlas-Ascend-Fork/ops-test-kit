# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""e2e XPU 采集输入的输出槽过滤回归（framework_api/xpu.py）。

看护：output_tensor_indexes 是 top-level 槽位索引（对齐 _check_output_slots
与 e2e_xpu_input_names），TensorList 槽只占一个槽位。修复前 inputs/dtypes 按
子张量数递增 real_idx，TensorList 之后的输出槽过滤失效——输出张量漏进 XPU
输入列表，且与 names 列表长度错位（193da3e 引入，重构搬家未改）。"""

from types import SimpleNamespace

import numpy as np

from ttk.core_modules.framework_api.xpu import (
    e2e_xpu_input_dtypes,
    e2e_xpu_input_names,
    e2e_xpu_inputs,
)


def _testcase(dist, output_tensor_indexes):
    tc = SimpleNamespace(
        tensor_list_dist=dist,
        output_tensor_indexes=output_tensor_indexes,
        flat_tensor_dtypes=("float32", "float32", "int32"),
    )
    # plan 参数序与 top-level 槽位一一对应（name 仅示意）
    tc.get_param_plan = lambda: SimpleNamespace(
        overload_params=[
            SimpleNamespace(name="tensors", is_tensor_like=True),
            SimpleNamespace(name="extra", is_tensor_like=True),
            SimpleNamespace(name="out", is_tensor_like=True),
        ]
    )
    return tc


def test_tensorlist_then_output_slot_filtered():
    """TensorList(2) 槽 0 + 纯输出槽 1：输出张量不得漏进 XPU 输入，
    且 inputs/dtypes/names 三列表长度一致（按名绑定不错位）。"""
    tc = _testcase((2, 0), (1,))
    raw = [np.zeros(2), np.zeros(2), np.ones(3)]  # [TL 子张量×2, 输出张量]

    inputs = e2e_xpu_inputs(tc, raw)
    assert len(inputs) == 1
    assert isinstance(inputs[0], list)  # 仅 TensorList 输入
    assert len(inputs[0]) == 2

    dtypes = e2e_xpu_input_dtypes(tc)
    assert dtypes == [["float32", "float32"]]

    names = e2e_xpu_input_names(tc)
    assert names == ["tensors"]
    assert len(inputs) == len(dtypes) == len(names)


def test_output_before_tensorlist_unchanged():
    """输出槽过滤。纯输出照常滤（含 TensorList 之前的槽 0）；in-place 方法式的
    self 双身份槽和 CSV 声明的被写输入位要保留，漏掉会让三方派发零张量。"""
    # 语义 1：纯输出槽 0 在 TensorList 之前，过滤行为保持
    tc = _testcase((0, 2), (0,))
    raw = [np.ones(3), np.zeros(2), np.zeros(2)]  # [输出张量, TL 子张量×2]

    inputs = e2e_xpu_inputs(tc, raw)
    assert len(inputs) == 1
    assert isinstance(inputs[0], list)
    assert len(inputs[0]) == 2
    assert e2e_xpu_input_dtypes(tc) == [["float32", "int32"]]
    assert e2e_xpu_input_names(tc) == ["extra"]

    # 语义 2：in-place 方法式（自动填输出槽 (0,)）的 self 槽是输入，不得滤
    from ttk.core_modules.framework_api.testcase import TestcaseE2e

    tc_inplace = TestcaseE2e()
    tc_inplace.testcase_name = "issue537_xlogy"
    tc_inplace.api_name = "torch.Tensor.xlogy_"
    tc_inplace.tensor_view_shapes = ((4,),)
    tc_inplace.tensor_dtypes = ("float32",)
    tc_inplace.tensor_formats = ("ND",)
    tc_inplace.attributes = {"other": 87}
    tc_inplace._validate_csv_inputs()
    assert tc_inplace.output_tensor_indexes == (0,), "方法式 in-place 未声明输出槽须自动填 (0,)"

    raw_inplace = [np.zeros(4, dtype=np.float32)]
    assert [a.shape for a in e2e_xpu_inputs(tc_inplace, raw_inplace)] == [(4,)]
    assert e2e_xpu_input_dtypes(tc_inplace) == ["float32"]
    assert "self" in e2e_xpu_input_names(tc_inplace)

    # 语义 3：CSV 声明的被写输入位与输出槽重叠（双身份）保留，其余纯输出照滤
    tc_declared = SimpleNamespace(
        api_name="torch.some_func",
        tensor_list_dist=(0, 0, 0),
        output_tensor_indexes=(1, 2),
        inplace_input_indexes=(1,),
        flat_tensor_dtypes=("float32", "float32", "float32"),
    )
    tc_declared.get_param_plan = lambda: SimpleNamespace(
        overload_params=[
            SimpleNamespace(name="a", is_tensor_like=True),
            SimpleNamespace(name="b", is_tensor_like=True),
            SimpleNamespace(name="c", is_tensor_like=True),
        ]
    )
    raw_declared = [np.zeros(2), np.zeros(2), np.ones(2)]

    inputs_declared = e2e_xpu_inputs(tc_declared, raw_declared)
    assert len(inputs_declared) == 2  # 槽 0 与双身份槽 1 保留，纯输出槽 2 滤掉
    assert e2e_xpu_input_dtypes(tc_declared) == ["float32", "float32"]
    assert e2e_xpu_input_names(tc_declared) == ["a", "b"]


def test_no_output_indexes_passthrough():
    """未声明输出槽（按返回值取输出，最常见形态）：全量透传。"""
    tc = _testcase((2, 0), ())
    raw = [np.zeros(2), np.zeros(2), np.ones(3)]

    inputs = e2e_xpu_inputs(tc, raw)
    assert len(inputs) == 2
    assert e2e_xpu_input_dtypes(tc) == [["float32", "float32"], "int32"]
