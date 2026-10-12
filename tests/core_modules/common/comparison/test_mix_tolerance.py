# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MixToleranceComparison 单元测试：逐元素条件、matched_ratio 边界、max_abs_error 硬上限、NaN/Inf 真值表。"""

import numpy as np
import pytest

from ttk.core_modules.common.comparison.mix_tolerance import MixToleranceComparison

# float32 表值（resolve_tolerance 解析的最终值）——显式 limit 场景（完全替代，不叠加 ULP）
FP32 = {"rtol": 2**-10, "atol": 2**-16, "required_matched_ratio": 0.99, "max_abs_error_limit": 1e-2}
# 未配置 limit 场景（None 哨兵 + floor → mix 内走 max(floor, 32*ULP(g_low)) 动态式）
FP32_DYN = {
    "rtol": 2**-10,
    "atol": 2**-16,
    "required_matched_ratio": 0.99,
    "max_abs_error_limit": None,
    "max_abs_error_floor": 1e-2,
}


def _impl(actual, golden, dtype="float32", options=None):
    """返回 EachCompareResult。options 缺省用 float32 表值（resolve_tolerance 解析好的最终值）。"""
    c = MixToleranceComparison(np.asarray(actual), np.asarray(golden), 0, dtype, options or FP32)
    return c.compare_impl()


def _run(actual, golden, dtype="float32", options=None):
    """返回 compare() 的 4-tuple。"""
    c = MixToleranceComparison(np.asarray(actual), np.asarray(golden), 0, dtype, options or FP32)
    return c.compare()


# —— 逐元素通过条件：|actual - golden| <= atol + rtol * |golden| ——
def test_exact_match_passes():
    r = _impl([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r.is_pass is True
    assert r.metrics["matched_ratio"] == 1.0
    assert r.metrics["max_abs_error"] == 0.0


def test_atol_covers_small_golden_no_divzero():
    """golden=0 时 atol 兜底（天然避免除零）：err=atol 过、err=2*atol 不过。"""
    atol = FP32["atol"]
    ok = _impl([atol], [0.0])
    assert ok.is_pass is True
    bad = _impl([2 * atol], [0.0])
    assert bad.is_pass is False  # 单元素 ratio=0 < 0.99


def test_rtol_covers_large_golden():
    """大值场景走相对容差：golden=100、err=5e-3（> atol，< 硬上限）在 rtol 预算内通过；rtol=0 则失败。"""
    r = _impl([100.005], [100.0])  # 预算 = atol + rtol*100 ≈ 0.098
    assert r.is_pass is True
    r0 = _impl([100.005], [100.0], options=dict(FP32, rtol=0.0))
    assert r0.is_pass is False  # atol=2^-16 兜不住 5e-3


# —— 整体通过条件：matched_ratio >= required 且 max_abs_error <= 硬上限 ——
def test_ratio_boundary_99_percent():
    """100 元素 1 个超元素容差（err=5e-3 < 硬上限）→ ratio=0.99 恰好达标（>=）；2 个 → 0.98 FAIL。"""
    g = np.ones(100)
    a1 = g.copy()
    a1[0] = 1.0 + 5e-3  # 超元素预算（≈9.9e-4）但低于硬上限 1e-2
    r1 = _impl(a1, g)
    assert r1.metrics["matched_ratio"] == 0.99
    assert r1.is_pass is True

    a2 = g.copy()
    a2[:2] = 1.0 + 5e-3
    r2 = _impl(a2, g)
    assert r2.metrics["matched_ratio"] == 0.98
    assert r2.is_pass is False


def test_max_abs_error_hard_limit_beats_ratio():
    """ratio 达标但单点绝对误差超硬上限 → FAIL（硬上限拦灾难性离群点）。"""
    g = np.ones(100)
    a = g.copy()
    a[0] = 11.0  # err=10 > 1e-2，ratio=0.99 达标
    r = _impl(a, g)
    assert r.is_pass is False
    assert r.metrics["max_abs_error"] == pytest.approx(10.0)
    assert "max_abs_error" in r.metrics["reason"]


# —— 大值场景下 rtol 预算 > 硬上限，全部元素过逐元素容差但绝对误差超限 ——
def test_hard_limit_exceeded_not_shown_as_100_percent():
    """全部元素在 atol+rtol*|g| 预算内但 err > 硬上限 → FAIL 且 matched_ratio < 1（不误导性显示 100%）。"""
    g = np.array([100.0, -100.0, 200.0, -200.0])
    a = g + np.array([1.2, -1.1, 1.4, -1.3])  # 预算 ≈0.098*|g| 全过，err 均超 1e-2 硬上限
    r = _impl(a, g, options=dict(FP32, rtol=1e-3, atol=1e-3))
    assert r.is_pass is False
    assert r.metrics["matched_ratio"] < 1.0
    assert r.precision == r.metrics["matched_ratio"]


def test_hard_limit_exceeded_diff_index_has_details():
    """同场景 → 超限元素全部进入 diff_index（ttk-compare.log 可展示出错部分）。"""
    g = np.array([100.0, -100.0, 200.0, -200.0])
    a = g + np.array([1.2, -1.1, 1.4, -1.3])
    r = _impl(a, g, options=dict(FP32, rtol=1e-3, atol=1e-3))
    assert r.diff_index is not None
    assert r.diff_index.size == 4


def test_hard_limit_not_exceeded_ratio_unchanged():
    """err <= 硬上限时并入条件恒真 → matched_ratio 与判定不变（判定等价性）。"""
    g = np.ones(100)
    a = g.copy()
    a[0] = 1.0 + 5e-3  # 超元素预算（≈9.9e-4）、低于硬上限 1e-2
    r = _impl(a, g)
    assert r.metrics["matched_ratio"] == 0.99
    assert r.is_pass is True


# —— NaN/Inf 真值表 ——
@pytest.mark.parametrize(
    ("a", "g", "expect_pass"),
    [
        (np.nan, np.nan, True),  # 都 NaN → 一致
        (np.inf, np.inf, True),  # 都 +Inf → 一致
        (np.inf, -np.inf, False),  # Inf 异号 → 无界误差
        (np.nan, 1.0, False),  # NaN vs 有限 → 无界误差
        (1.0, np.inf, False),  # 有限 vs Inf → 无界误差
    ],
)
def test_nan_inf_truth_table(a, g, expect_pass):
    r = _impl([a], [g])
    assert r.is_pass == expect_pass


def test_mismatch_sets_max_abs_error_none():
    """NaN/Inf 不一致 → max_abs_error 无界，metrics 中以 None 表示（保 CSV eval 往返）。"""
    r = _impl([np.nan, 1.0], [1.0, 1.0])
    assert r.is_pass is False
    assert r.metrics["max_abs_error"] is None
    assert "NaN/Inf mismatch" in r.metrics["reason"]


def test_all_nan_consistent_passes():
    """全 NaN 且一致 → matched（max_abs_error=0）。"""
    r = _impl([np.nan, np.nan], [np.nan, np.nan])
    assert r.is_pass is True
    assert r.metrics["matched_ratio"] == 1.0


# —— diff_idx 排序：误差降序（NaN 视作 +inf 最前）——
def test_diff_idx_worst_first():
    g = np.ones(4)
    a = np.array([1.001, np.nan, 1.02, 1.004])  # err: 1e-3, nan, 2e-2, 4e-3
    r = _impl(a, g)
    assert r.is_pass is False
    assert list(r.diff_index) == [1, 2, 3, 0]


# —— 结构 / 边界 ——
def test_empty_arrays_pass():
    precision, _l, is_pass, _m = _run([], [])
    assert precision == "100%"
    assert is_pass is True


def test_size_mismatch_fails():
    r = _impl([1.0, 2.0], [1.0])
    assert r.is_pass is False
    assert r.precision == "2 vs 1"


def test_metrics_are_plain_literals():
    """metrics 全 Python 字面量（CSV eval 往返）。"""
    r = _impl([1.0, 2.0], [1.0, 2.0])
    assert eval(repr(r.metrics)) == r.metrics  # noqa: S307


def test_spec_override_params_take_effect():
    """resolve 解析后的 override 参数生效：rtol=0.5 时 err=4e-3（> 默认预算 ≈9.9e-4，< 硬上限）可通过。"""
    r = _impl([1.004], [1.0], options=dict(FP32, rtol=0.5))
    assert r.is_pass is True
    assert r.metrics["rtol"] == 0.5
    assert _impl([1.004], [1.0]).is_pass is False


def test_fp8_output_vs_high_precision_golden():
    """fp8 输出走混合容差：ml_dtypes fp8 promote 到 float32 后与高精度 golden 比对。"""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    # float8_e5m2 表值：rtol=2^-1, atol=2^-3, max_abs_error_limit=max(1e-1, 32*2^-2)=8.0
    opts = {"rtol": 2**-1, "atol": 2**-3, "required_matched_ratio": 0.99, "max_abs_error_limit": 8.0}
    a = np.array([1.0, 2.0, 4.0], ml_dtypes.float8_e5m2)
    g = np.array([1.0, 2.1, 4.0], np.float32)  # 单标杆：高精度 golden
    r = MixToleranceComparison(a, g, 0, "float8_e5m2", opts).compare_impl()
    assert r.is_pass is True
    assert r.metrics["matched_ratio"] == 1.0
    assert r.metrics["max_abs_error"] == pytest.approx(0.1, abs=1e-6)


def test_dynamic_ulp_limit_large_value_domain():
    """未配置 limit + golden 高精度 + 大值域：上限 = max(floor, 32*ULP(g_low)) 自适应放宽。"""
    a = np.asarray([60000.0, 60000.0], dtype=np.float16)
    g = np.asarray([60005.0, 60000.0], dtype=np.float32)  # err=5，静态 1e-2 会误杀
    r = _impl(a, g, "float16", FP32_DYN)
    assert r.metrics["pass"] is True  # 32*ULP(60000)=1024 ≥ err=5


def test_dynamic_ulp_limit_small_value_domain_keeps_floor():
    """未配置 limit + 小值域：32*ULP 极小，floor 兜底不收紧。"""
    a = np.asarray([0.001, 0.001], dtype=np.float16)
    g = np.asarray([0.001, 0.0015], dtype=np.float32)
    fp16_dyn = {
        "rtol": 2**-9,
        "atol": 2**-9,
        "required_matched_ratio": 0.99,
        "max_abs_error_limit": None,
        "max_abs_error_floor": 1e-1,
    }
    r = _impl(a, g, "float16", fp16_dyn)
    # err=5e-4 ≤ floor(1e-1) → 放行；若错误地用 32*ULP(0.001)≈3e-5 收紧会 FAIL
    assert r.metrics["pass"] is True


def test_explicit_limit_fully_replaces_dynamic():
    """显式配置 limit：完全替代动态 ULP，不叠加（兼容存量 Spec 语义）。"""
    a = np.asarray([60000.0, 60000.0], dtype=np.float16)
    g = np.asarray([60005.0, 60000.0], dtype=np.float32)
    # 显式 1e-2：若仍叠加动态 ULP(1024) 会错误放行 err=5
    r = _impl(
        a, g, "float16", {"rtol": 2**-9, "atol": 2**-9, "required_matched_ratio": 0.99, "max_abs_error_limit": 1e-2}
    )
    assert r.metrics["pass"] is False


def test_inf_cast_same_sign_inf_passes():
    """actual=+inf 且 golden 超 fp16 范围：golden RNE 收窄为 +inf，同号一致。"""
    a = np.asarray([np.inf, 1.0], dtype=np.float16)
    g = np.asarray([70000.0, 1.0], dtype=np.float32)
    r = _impl(a, g, "float16", FP32_DYN)
    assert r.metrics["pass"] is True


def test_dynamic_ulp_same_dtype_enable_mode():
    """同 dtype（--golden-mode Enable 未 Promote）动态锚定同样生效——
    锚定不应有 dtype 前提（收窄对同 dtype 是恒等变换）。"""
    a = np.asarray([65472.0, 65472.0], dtype=np.float16)
    g = np.asarray([65504.0, 65472.0], dtype=np.float16)  # err=32（fp16 大 binade 1ULP）
    fp16_dyn = {
        "rtol": 2**-9,
        "atol": 2**-9,
        "required_matched_ratio": 0.99,
        "max_abs_error_limit": None,
        "max_abs_error_floor": 1e-1,
    }
    r = _impl(a, g, "float16", fp16_dyn)
    # 32*ULP(65472)=1024 ≥ err=32 → 放行（此前被 dtype 门错位退回 floor=0.1 误杀）
    assert r.metrics["pass"] is True


def test_dynamic_ulp_max_finite_anchor_binade_width():
    """锚点为 dtype 最大有限值（65504）时：spacing 返回 inf，特判取同 binade 前驱的
    格宽（32）——limit=1024 而非 inf（标准 ULP 语义为区间格宽，非全放行）。"""
    a = np.asarray([65472.0, 65504.0], dtype=np.float16)
    g = np.asarray([65504.0, 65504.0], dtype=np.float16)
    fp16_dyn = {
        "rtol": 2**-9,
        "atol": 2**-9,
        "required_matched_ratio": 0.99,
        "max_abs_error_limit": None,
        "max_abs_error_floor": 1e-1,
    }
    r = _impl(a, g, "float16", fp16_dyn)
    assert r.metrics["max_abs_error_limit"] == 32.0 * 32.0


def test_all_inf_same_sign_passes():
    """全元素 ±inf 同号（finite0 全空）：same_inf 一致，PASS；limit 恒为 floor。"""
    a = np.asarray([np.inf, np.inf], dtype=np.float16)
    g = np.asarray([np.inf, np.inf], dtype=np.float32)
    fp16_dyn = {
        "rtol": 2**-9,
        "atol": 2**-9,
        "required_matched_ratio": 0.99,
        "max_abs_error_limit": None,
        "max_abs_error_floor": 1e-1,
    }
    r = _impl(a, g, "float16", fp16_dyn)
    assert r.metrics["pass"] is True
    assert r.metrics["max_abs_error"] == 0.0
