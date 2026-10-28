# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""onnx golden 层单测（纯 CPU，ReferenceEvaluator 参照）。

_run_onnx_reference 的参数位重建（含 Variadic 嵌套 / 输出槽剔除）与
generate_golden 的同 API 分发——golden 与被测共用同一 model_builder 产物
（同构参照）。

跨环境：全部用例经 extract_onnx_params（conftest.onnx_registry 处理 onnx
缺席与 opset 门禁）；Split 的 num_outputs 语义经 split_with_num_outputs
能力探针。
"""

import numpy as np
import pytest

from ttk.core_modules.common.backend import detect_framework
from ttk.core_modules.framework_api.golden import _run_onnx_reference, generate_golden
from ttk.core_modules.framework_api.testcase import TestcaseE2e

pytestmark = pytest.mark.usefixtures("onnx_registry")


def _case(api_name, shapes, dtypes, attrs=None, output_tensor_indexes=None):
    tc = TestcaseE2e()
    tc.api_name = api_name
    tc.testcase_name = f"golden_{api_name.rsplit('.', 1)[-1].lower()}"
    tc.tensor_view_shapes = shapes
    tc.tensor_dtypes = dtypes
    tc.attributes = attrs or {}
    tc.output_tensor_indexes = output_tensor_indexes or ()
    return tc


def test_generate_golden_priority3_dispatch():
    """无 golden_api / 无插件 → 同 API 分发（ReferenceEvaluator）；flat 双输入
    Add 数值对拍 numpy 手算（ReferenceEvaluator 路径即 _run_onnx_reference）。"""
    tc = _case("onnx.Add", ((2, 3), (2, 3)), ("float32", "float32"))
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = (a + 10).astype(np.float32)
    goldens = generate_golden(tc, [a, b])
    assert len(goldens) == 1
    np.testing.assert_array_equal(goldens[0], a + b)
    assert detect_framework(tc.api_name) == "onnx"


def test_reference_concat_variadic_nested_dist():
    """Variadic 嵌套（dist=(2,)）：raw_inputs flat 2 张量 + axis 属性。"""
    tc = _case("onnx.Concat", (((2, 3), (2, 4)),), (("int32", "int32"),), attrs={"axis": 1})
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = np.arange(8, dtype=np.int32).reshape(2, 4)
    (golden,) = _run_onnx_reference("onnx.Concat", [a, b], tc)
    np.testing.assert_array_equal(golden, np.concatenate([a, b], axis=1))


def test_reference_split_output_slots_excluded(split_with_num_outputs):
    """Split：输出槽（output_tensor_indexes）不入输入位，None 占位剔除后单输入。"""
    tc = _case(
        "onnx.Split",
        ((6, 4), None, None, None),
        ("float32", "float32", "float32", "float32"),
        attrs={"axis": 0, "num_outputs": 3},
        output_tensor_indexes=(1, 2, 3),
    )
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    goldens = _run_onnx_reference("onnx.Split", [x, None, None, None], tc)
    assert len(goldens) == 3
    for got, exp in zip(goldens, np.split(x, 3, axis=0)):
        np.testing.assert_array_equal(got, exp)


def test_reference_unresolvable_op_raises():
    """schema 不在注册表 → 上抛（Invoke.golden 收敛为 GOLDEN_FAILURE 占位）。"""
    tc = _case("onnx.NoSuchOpXYZ", ((2, 3),), ("float32",))
    with pytest.raises(RuntimeError, match="cannot parse onnx.NoSuchOpXYZ"):
        _run_onnx_reference("onnx.NoSuchOpXYZ", [np.zeros((2, 3), np.float32)], tc)
