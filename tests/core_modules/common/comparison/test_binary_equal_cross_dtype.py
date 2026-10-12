# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""binary_equal 跨 dtype 比对测试：整数跨宽度相等、bool/int 互认、int4 二进制一致、浮点跨 dtype 拒绝、空数组。"""

import numpy as np
import pytest

# 触发各比对类的注册（装饰器在 import 时执行）
import ttk.core_modules.common.comparison.binary_equal  # noqa: F401
import ttk.core_modules.common.comparison.is_close  # noqa: F401
from ttk.core_modules.common.comparison.registry import ComparisonRegister


def _cls(token):
    """从注册表取比对类。"""
    return ComparisonRegister.registry[token]


def _arr(vals, dtype_str):
    """按 dtype 字符串构造 numpy 数组（int4 需要 ml_dtypes，缺失则 skip）。"""
    if dtype_str == "int4":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        return np.array(vals, dtype=ml_dtypes.int4)
    if dtype_str == "bool":
        return np.array(vals, dtype=bool)
    return np.array(vals, dtype=np.dtype(dtype_str))


def test_binary_equal_alias_registered():
    """binary_equal / bin / binary 三个别名指向同一比对类。"""
    assert _cls("binary_equal") is _cls("bin") is _cls("binary")


def test_isclose_alias_registered():
    """isclose / close 别名指向同一比对类。"""
    assert _cls("isclose") is _cls("close")


@pytest.mark.parametrize(
    ("actual_vals", "actual_dtype", "golden_vals", "golden_dtype", "check_metrics"),
    [
        pytest.param([1, 2, 3], "int32", [1, 2, 3], "int64", True, id="int32_int64_equal"),
        pytest.param([False, True], "bool", [0, 1], "int32", False, id="bool_vs_int"),
    ],
)
def test_cross_dtype_pass(actual_vals, actual_dtype, golden_vals, golden_dtype, check_metrics):
    """跨 dtype 一致 → PASS（int32/int64、bool/int32、int4/int4 二进制一致）。"""
    actual = _arr(actual_vals, actual_dtype)
    golden = _arr(golden_vals, golden_dtype)
    c = _cls("binary_equal")(actual, golden, 0, actual_dtype, {})
    p, _l, is_pass, metrics = c.compare()
    assert is_pass is True
    if check_metrics:
        assert p == "100%"
        assert metrics["standard"] == "binary_equal"
        assert metrics["pass"] is True


@pytest.mark.parametrize(
    ("actual_vals", "actual_dtype", "golden_vals", "golden_dtype"),
    [
        pytest.param([1, 2], "int32", [1, 3], "int64", id="int32_int64_different"),
        pytest.param([1, 2, 3], "int4", [1, 9, 3], "int4", id="int4_different"),
        pytest.param([1.0], "float32", [1.0], "float64", id="float_cross_dtype"),
        pytest.param([1], "int32", [1.0], "float32", id="int_vs_float"),
        pytest.param([1], "uint64", [1], "int32", id="uint64_int32_precision_loss"),
    ],
)
def test_cross_dtype_fail(actual_vals, actual_dtype, golden_vals, golden_dtype):
    """跨 dtype 不一致/不可比 → FAIL（值不同、浮点跨 dtype、int/float 混合、uint64/int32 精度损失）。"""
    actual = _arr(actual_vals, actual_dtype)
    golden = _arr(golden_vals, golden_dtype)
    c = _cls("binary_equal")(actual, golden, 0, actual_dtype, {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is False


@pytest.mark.parametrize(
    ("actual", "golden", "dtype", "expected_pass"),
    [
        pytest.param(np.array([]), np.array([]), "int32", True, id="empty_both"),
        pytest.param(np.array([1], np.int32), np.array([], np.int32), "int32", False, id="empty_one"),
    ],
)
def test_empty_cases(actual, golden, dtype, expected_pass):
    """空数组场景：双方空 → PASS，一方空 → FAIL。"""
    c = _cls("binary_equal")(actual, golden, 0, dtype, {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is expected_pass


def test_isclose_populates_metrics():
    """isclose 比对填充 standard/pass/precision metrics。"""
    c = _cls("isclose")(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0, "float32", {"rtol": [1e-3], "atol": [1e-8], "ptol": [1e-3]}
    )
    _p, _l, is_pass, metrics = c.compare()
    assert metrics["standard"] == "isclose"
    assert metrics["pass"] is True
    assert "precision" in metrics
