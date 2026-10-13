# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""registry 基础设施单元测试：EachCompareResult 默认值、torch 输入规整、
ComparisonBase.compare() 4-tuple + 空数组短路。"""

import logging

import numpy as np
import pytest

from ttk.core_modules.common.comparison.registry import (
    ComparisonBase,
    ComparisonRegister,
    EachCompareResult,
    register_comparison,
)


def test_each_compare_result_defaults():
    """EachCompareResult 默认值：precision/diff_index/is_pass/log/standard/metrics/error_info 及 metrics 独立 dict。"""
    r = EachCompareResult(1)
    assert r.precision == 1
    assert r.diff_index is None
    assert r.is_pass is False
    assert r.log == ""
    assert r.standard == ""
    assert r.metrics == {}
    assert r.error_info is None
    # distinct default dicts (not shared)
    assert EachCompareResult(1).metrics is not EachCompareResult(1).metrics


def test_torch_input_normalized_to_numpy_dtype():
    """torch 输入经 compare() 规整为 numpy 且 dtype 字节保持：fp32 tensor 与 fp32 numpy 逐位相等。"""
    torch = pytest.importorskip("torch")
    from ttk.core_modules.common.comparison.registry import get_comparison

    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    g = np.array([1.0, 2.0], dtype=np.float32)
    _p, _l, is_pass, _m = get_comparison("bin")(t, g, 0, "float32", {}).compare()
    assert is_pass is True  # 转换走样（如抬到 fp64）则字节不等
    g64 = np.array([1.0, 2.0], dtype=np.float64)
    _p, _l, is_pass64, _m = get_comparison("bin")(t, g64, 0, "float32", {}).compare()
    assert is_pass64 is False


# 一个最小 ComparisonBase 子类用于测 _precheck
@register_comparison("__test_dummy")
class _Dummy(ComparisonBase):
    STANDARD_NAME = "dummy"

    def compare_impl(self):
        return EachCompareResult(0.5, is_pass=True, standard="dummy", metrics={"k": 1})


@pytest.mark.parametrize(
    ("actual", "golden", "expected"),
    [
        pytest.param(np.array([]), np.array([]), (True, "100%"), id="both_empty"),
    ],
)
def test_compare_both_empty_short_circuits_to_pass(actual, golden, expected):
    """compare() 公开入口：双方空数组短路 PASS（precision=100%），不进 compare_impl。"""
    c = _Dummy(actual, golden, 0, "float32", {})
    precision, _log, is_pass, _m = c.compare()
    assert is_pass is expected[0]
    assert precision == expected[1]


def test_compare_size_mismatch_short_circuits_to_fail():
    """compare() 公开入口：元素数不一致统一短路 FAIL（#257 上提基类）——"N vs M"
    precision + 大小说明，不进 compare_impl（此前仅 isclose 有私有检查，binary
    异长崩溃、stat_rel_err/cosine 长侧静默截断）。"""
    c = _Dummy(np.ones(4, np.float32), np.ones(2, np.float32), 0, "float32", {})
    precision, log, is_pass, _m = c.compare()
    assert is_pass is False
    assert precision == "4 vs 2"
    assert "size is different with golden size" in log


def test_log_diff_output_respects_max():
    """diff 行打印数 = MAX_DIFF_OUTPUT（此前 off-by-one 打 101 行）。"""
    from ttk.core_modules.common.comparison.registry import get_comparison

    cls = get_comparison("bin")
    c = cls(np.zeros(500, np.float32), np.ones(500, np.float32), 0, "float32", {})
    _p, log, _ok, _m = c.compare()
    n = sum(1 for line in log.splitlines() if line.startswith("Index:"))
    assert n == ComparisonBase.MAX_DIFF_OUTPUT


def test_register_comparison_rejects_bad_arg():
    """register_comparison 入参须为 list/tuple/str，其余 TypeError。"""
    with pytest.raises(TypeError, match="list / tuple / str"):
        register_comparison(42)


def test_register_comparison_duplicate_warns_and_overrides(caplog):
    """重复注册同名 token：告警且后者覆盖。"""
    key = "_dup_test_token"
    try:
        with caplog.at_level(logging.WARNING):
            register_comparison([key])(ComparisonBase)
            register_comparison([key])(EachCompareResult)  # 占位覆盖者
        assert any("has already been registered" in r.message for r in caplog.records)
        assert ComparisonRegister.registry[key] is EachCompareResult
    finally:
        ComparisonRegister.registry.pop(key, None)
