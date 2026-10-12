# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""resolve_tolerance 收敛测试：legacy 注入、cross_check level 预设/override、isclose/cosine legacy 读取。"""

import numpy as np
import pytest

from ttk.core_modules.common.comparison.resolve import resolve_tolerance


def test_legacy_injection():
    """legacy（precision_tolerances/absolute_precision）注入 params["legacy"]。"""
    standards = resolve_tolerance(None, [(0.1, 0.01)], 1e-8, ["float32"], None)
    assert standards[0].params["legacy"]["rtol"] == 0.1
    assert standards[0].params["legacy"]["ptol"] == 0.01
    assert standards[0].params["legacy"]["atol"] == 1e-8


@pytest.mark.parametrize(
    ("tol_spec", "expected_level", "expected_mare", "expected_mere", "expected_rmse", "check_extras"),
    [
        pytest.param({"level": "L1"}, "L1", 5.0, 1.5, 1.5, True, id="L1_preset"),
        pytest.param({"level": "L1", "mare_ratio": 3.0}, "L1", 3.0, 1.5, 1.5, False, id="explicit_ratio_override"),
    ],
)
def test_cross_check_level_preset(tol_spec, expected_level, expected_mare, expected_mere, expected_rmse, check_extras):
    """cross_check level → ratio 预设 + override 组合(level 矩阵)：
    L0/L1/L2 预设、显式 ratio 覆盖、无 level 全 ratio。"""
    tol = {"float32": {"standard": "cross_check", **tol_spec}}
    s = resolve_tolerance(tol, None, 1e-8, ["float32"], None)[0]
    assert s.params["level"] == expected_level
    assert s.params["mare_ratio"] == expected_mare
    assert s.params["mere_ratio"] == expected_mere
    assert s.params["rmse_ratio"] == expected_rmse
    if check_extras:
        assert s.token == "cross_check"  # noqa: S105 判据名变量，非凭据
        assert "small_value" in s.params
        assert "small_value_atol" in s.params


def test_isclose_reads_legacy_rtol():
    """C1: isclose 从 legacy 子 dict 读 rtol（非顶层）。"""
    import ttk.core_modules.common.comparison.is_close  # noqa: F401 — 触发 @register_comparison
    from ttk.core_modules.common.comparison.registry import ComparisonRegister
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    standards = resolve_tolerance(None, [(0.001, 0.001)], 1e-9, ["float32"], None)
    cls = ComparisonRegister.registry["isclose"]
    out = np.array([1.0, 2.0])
    gold = np.array([1.0, 2.0])
    c = cls(out, gold, 0, "float32", standards[0].params)
    c.compare()  # 容差在 compare() 的 _finalize_tolerances 终值化（torch→numpy 转换后）
    # rtol/ptol/atol 应来自 legacy 子 dict（per-output 取位后的终值）
    assert c.rtol == 0.001
    assert c.atol == 1e-9


def test_cosine_reads_legacy_rtol():
    """C1: cosine 从 legacy 子 dict 读 rtol（cosine 只读 rtol，无 ptol/atol）。"""
    import ttk.core_modules.common.comparison.cosine_similarity  # noqa: F401 — 触发 @register_comparison
    from ttk.core_modules.common.comparison.registry import ComparisonRegister
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    standards = resolve_tolerance(None, [(0.01, 0.001)], 1e-9, ["float32"], "cosine")
    cls = ComparisonRegister.registry["cosine"]
    out = np.array([1.0, 2.0])
    gold = np.array([1.0, 2.0])
    c = cls(out, gold, 0, "float32", standards[0].params)
    c.compare()  # 容差终值化后 cosine 只有 rtol 属性（不消费 ptol/atol）
    assert c.rtol == 0.01
