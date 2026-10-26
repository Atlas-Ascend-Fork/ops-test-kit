# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for CrossCheckComparison: pass/fail/ratio/small_value/nan + safe_div."""

import numpy as np

from ttk.core_modules.common.comparison.cross_check import CrossCheckComparison, safe_div


def _make(output, golden, third_party, params, output_dtype="float32"):
    return CrossCheckComparison(output, golden, 0, output_dtype, params, third_party=third_party)


def test_pass_when_all_good():
    """三方误差一致 → PASS + metrics 结构。"""
    params = {
        "level": "L1",
        "mare_ratio": 5.0,
        "mere_ratio": 1.5,
        "rmse_ratio": 1.5,
        "small_value": 2**-14,
        "small_value_atol": 2**-30,
        "legacy": {"rtol": None, "ptol": None, "atol": 1e-8},
    }
    g = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c = _make(g.copy(), g.copy(), g.copy(), params)
    precision, log, is_pass, metrics = c.compare()
    assert is_pass is True
    assert metrics["standard"] == "cross_check"
    assert "config" in metrics  # 单元测试定位 metrics 形状（不只靠 Task 9 E2E）
    assert "result" in metrics


def test_ratio_exceeded():
    """NPU 误差远大于 third_party → ratio_exceeded FAIL。"""
    params = {
        "level": "L1",
        "mare_ratio": 5.0,
        "mere_ratio": 1.5,
        "rmse_ratio": 1.5,
        "small_value": 2**-14,
        "small_value_atol": 2**-30,
        "legacy": {"rtol": None, "ptol": None, "atol": 1e-8},
    }
    g = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = g + 1.0  # NPU 误差大
    third = g + 0.001  # third_party 误差小
    c = _make(out, g, third, params)
    precision, log, is_pass, metrics = c.compare()
    assert not is_pass
    assert "ratio exceeded" in metrics["reason"]


def test_safe_div_branches():
    """safe_div: 分母夹按 dtype 取的小值域阈值 err（精度标准）；nan/inf 分母 -> inf。"""
    err = 2**-14  # fp32 的 err
    assert safe_div(0, 0, err) == 1.0  # 0/0 -> 1（两边都完美，一致）
    assert safe_div(1.0, 0, err) == 1.0 / err  # 分母为 0 -> 夹到 err（非 inf）
    assert safe_div(6, 3, err) == 2.0  # 分母 > err，照常相除
    assert safe_div(0.1, float("nan"), err) == float("inf")  # nan 分母 -> inf
    assert safe_div(float("inf"), float("inf"), err) == 1.0  # 两侧均溢出 -> 一致
    assert safe_div(float("inf"), 3.0, err) == float("inf")  # NPU 溢出、竞品未溢出
    assert safe_div(3.0, float("inf"), err) == 0.0  # 竞品溢出、NPU 未溢出 -> NPU 严格更优
    assert isinstance(safe_div(6, 3, err), float)  # 守护 float() 强转


def test_small_value_partition_pass():
    """防回归：golden 全小（<small_value）+ third_party 误差大 + NPU 精确 → small_ratio 小 → PASS。"""
    params = {
        "level": "L1",
        "mare_ratio": 5.0,
        "mere_ratio": 1.5,
        "rmse_ratio": 1.5,
        "small_value": 2**-14,
        "small_value_atol": 2**-30,
        "legacy": {"rtol": None, "ptol": None, "atol": 1e-8},
    }
    g = np.full(100, 1e-15, dtype=np.float32)  # 全 < small_value（2**-14≈6e-5）→ 全 small 分区，large 空
    out = g.copy()  # NPU 精确 → err_target=0
    third = g + 1e-8  # third_party 误差大(>>small_value_atol=2**-30≈9.3e-10)→err_third=100
    c = _make(out, g, third, params)
    precision, log, is_pass, metrics = c.compare()
    assert is_pass
    assert metrics["result"]["small_err_cnt_target"] == 0
    assert metrics["result"]["small_err_cnt_third"] > 0
    assert metrics["result"]["mare"] is None  # 全小值域 large-empty → mare N/A（非 0.0 误导）


_NAN_PARAMS = {
    "level": "L1",
    "mare_ratio": 5.0,
    "mere_ratio": 1.5,
    "rmse_ratio": 1.5,
    "small_value": 2**-14,
    "small_value_atol": 2**-30,
    "legacy": {"rtol": None, "ptol": None, "atol": 1e-8},
}


def test_nan_inf_mismatch():
    """golden 与 third_party 一致(NaN)，但 NPU 不一致 → FAIL（规则3）。"""
    g = np.array([1.0, np.nan], dtype=np.float32)
    out = np.array([1.0, 1.0], dtype=np.float32)  # golden[1]=nan 但 NPU 非 nan
    third = np.array([1.0, np.nan], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    precision, log, is_pass, metrics = c.compare()
    assert not is_pass
    assert metrics["reason"] == "NaN/Inf mismatch"


def test_nan_inf_pass_npu_matches_third():
    """规则1：NPU 与 third_party 一致 → 通过（不论 golden 为何值）。"""
    # golden=NaN, NPU=+Inf, third=+Inf → NPU 与 third 一致 → 通过
    g = np.array([np.nan], dtype=np.float32)
    out = np.array([np.inf], dtype=np.float32)
    third = np.array([np.inf], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass

    # golden=1.0, NPU=NaN, third=NaN → NPU 与 third 一致 → 通过
    g = np.array([1.0], dtype=np.float32)
    out = np.array([np.nan], dtype=np.float32)
    third = np.array([np.nan], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass


def test_nan_inf_pass_npu_matches_golden():
    """规则2：NPU 与 golden 一致 → 通过（不论 third_party、不论 golden 是否有限）。"""
    # golden=NaN, NPU=NaN, third=1.0 → NPU 与 golden 一致 → 通过
    g = np.array([np.nan], dtype=np.float32)
    out = np.array([np.nan], dtype=np.float32)
    third = np.array([1.0], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass

    # golden=+Inf, NPU=+Inf, third=-Inf → NPU 与 golden 一致 → 通过
    g = np.array([np.inf], dtype=np.float32)
    out = np.array([np.inf], dtype=np.float32)
    third = np.array([-np.inf], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass


def test_nan_inf_pass_npu_matches_finite_golden_third_special():
    """规则2（golden 有限）：NPU 与有限 golden 一致、third_party 为 NaN/Inf → PASS。

    竞品单方面特殊值（如 fp16 中间量溢出后 0*inf=NaN）属竞品缺陷，不应归因为
    NPU 的测试失败；失败语义与 Path B 比值哲学对齐：只有 NPU 偏离 golden 且
    竞品能对上 golden 才判 NPU 失败。
    """
    # golden=0, NPU=0, third=NaN → 通过
    g = np.array([0.0], dtype=np.float32)
    out = np.array([0.0], dtype=np.float32)
    third = np.array([np.nan], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass

    # golden=39840, NPU=39840, third=+Inf → 通过
    g = np.array([39840.0], dtype=np.float32)
    out = np.array([39840.0], dtype=np.float32)
    third = np.array([np.inf], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass


def test_nan_inf_rule2_across_promoted_golden_dtype():
    """规则2 的「数值相等」必须按**声明的输出 dtype** 比，而不是裸 ==。

    上面那组用例把 NPU 与 golden 都构造成同一 dtype，恒能逐位相等；真实流水不是这样：
    golden_mode=Promote 会把 golden 的输出 dtype 一并抬档（bf16→fp32、fp32→fp64）且此后
    无人降回，于是「有限且相等」在裸 == 下结构性恒不成立，规则2 从未生效。
    数据取自 GroupNormSiluQuant 实测（n=1 且输入含 ±inf，rstd=1/sqrt(eps)）：
    NPU(bf16)=1016，golden(fp32)=1016.26996，竞品 torch 两遍法 (inf-inf)^2 得 NaN。
    """
    # bf16 输出档：NPU 落在 bf16 可表示值上，golden 是被抬档的 fp32 真值
    out = np.array([1016.0], dtype=np.float32)
    g = np.array([1016.26996], dtype=np.float32)
    third = np.array([np.nan], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS, output_dtype="bfloat16")
    _, _, is_pass, _ = c.compare()
    assert is_pass

    # fp32 输出档同理：golden 被抬到 fp64
    out = np.array([np.float32(1016.2699556378086)], dtype=np.float64)
    g = np.array([1016.2699556378086], dtype=np.float64)
    third = np.array([np.nan], dtype=np.float64)
    c = _make(out, g, third, _NAN_PARAMS, output_dtype="float32")
    _, _, is_pass, _ = c.compare()
    assert is_pass


def test_nan_inf_rule2_still_fails_beyond_rounding():
    """折叠不等于放水：NPU 偏离 golden 超过窄档一格舍入时仍判 FAIL。"""
    # bf16 在 1016 附近的 1 ULP = 4（网格 …1012/1016/1020/1024…）；NPU 给 1032 已偏 4 格
    out = np.array([1032.0], dtype=np.float32)
    g = np.array([1016.26996], dtype=np.float32)
    third = np.array([np.nan], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS, output_dtype="bfloat16")
    _, _, is_pass, metrics = c.compare()
    assert not is_pass
    assert metrics["reason"] == "NaN/Inf mismatch"


def test_nan_inf_fail_golden_matches_third_finite():
    """规则3（有限值）：golden 与 third_party 均为有限且相等，但 NPU 为 NaN → FAIL。"""
    g = np.array([1.0], dtype=np.float32)
    out = np.array([np.nan], dtype=np.float32)  # NPU 异常
    third = np.array([1.0], dtype=np.float32)  # golden 与 third 一致
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, metrics = c.compare()
    assert not is_pass
    assert metrics["reason"] == "NaN/Inf mismatch"


def test_nan_inf_fail_golden_third_disagree():
    """golden 与 third_party 不一致、NPU 与双方都不一致 → FAIL（规则1和规则2都不满足）。"""
    # golden=NaN, NPU=1.0, third=+Inf → t≠b 且 t≠g → FAIL
    g = np.array([np.nan], dtype=np.float32)
    out = np.array([1.0], dtype=np.float32)
    third = np.array([np.inf], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert not is_pass

    # golden=+Inf, NPU=1.0, third=-Inf → t≠b 且 t≠g → FAIL
    g = np.array([np.inf], dtype=np.float32)
    out = np.array([1.0], dtype=np.float32)
    third = np.array([-np.inf], dtype=np.float32)
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert not is_pass


def test_rmse_no_overflow_in_float32():
    """float32 大差值（**2 溢出量级）不应让 rmse_ratio 误判 Inf→FAIL。

    golden=1e19, NPU=-1e19（差 2e19，平方=4e38 > float32 max≈3.4e38→Inf），
    third_party 同款差值 → 两侧 rmse 均溢出 → safe_div 返回 1.0（一致）→ PASS。
    """
    g = np.array([1e19, 2e19], dtype=np.float32)
    out = np.array([-1e19, -2e19], dtype=np.float32)  # 差值 2e19/4e19，平方溢出 float32
    third = np.array([-1e19, -2e19], dtype=np.float32)  # 竞品同款 → ratio=1.0
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, metrics = c.compare()
    assert is_pass
    assert metrics["result"]["rmse"] == 1.0


def test_rmse_overflow_npu_only():
    """NPU 差值溢出但竞品未溢出 → rmse_ratio=Inf → FAIL（真实异常，非误判）。"""
    g = np.array([1e19], dtype=np.float32)
    out = np.array([-1e19], dtype=np.float32)  # 平方溢出 float32 → rmse_npu=Inf
    third = np.array([1e19 + 1e10], dtype=np.float32)  # 差值小 → rmse_party 有限
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, metrics = c.compare()
    assert not is_pass


def test_golden_inf_but_target_equals_third_party():
    """golden 非有限（inf），t/b 均有限且 t==b → PASS。

    golden 在该位溢出不可用作真值，退化为 t vs b 两方一致性判定。
    """
    g = np.array([1.0, np.inf], dtype=np.float32)
    out = np.array([1.0, 42.0], dtype=np.float32)
    third = np.array([1.0, 42.0], dtype=np.float32)  # t==b
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, _ = c.compare()
    assert is_pass


def test_golden_inf_and_target_differs_from_third_party():
    """golden 非有限（inf），t/b 均有限但 t≠b → FAIL。

    golden 不可用且两方有分歧，不放行。
    """
    g = np.array([1.0, np.inf], dtype=np.float32)
    out = np.array([1.0, 42.0], dtype=np.float32)
    third = np.array([1.0, 99.0], dtype=np.float32)  # t≠b
    c = _make(out, g, third, _NAN_PARAMS)
    _, _, is_pass, metrics = c.compare()
    assert not is_pass
    assert metrics["reason"] == "NaN/Inf mismatch"


def test_xpu_golden_mode_degenerate_two_party():
    """golden-mode XPU：b == g（同份三方数据双槽）→ 等效容差 = ratio_limit × small_value。

    rel_party 恒 0，safe_div 分母夹到 small_value（fp32: 2^-14）：L1 mare=5 →
    NPU 相对 XPU 误差 < 5×2^-14≈3e-4 过、超过即挂。钉住退化语义——该行为
    由 safe_div 夹底涌现，改动 safe_div 前以此为准。"""
    params = {
        "level": "L1",
        "mare_ratio": 5.0,
        "mere_ratio": 1.5,
        "rmse_ratio": 1.5,
        "small_value": 2**-14,
        "small_value_atol": 2**-30,
    }
    golden = np.array([1.0, 1.0], dtype=np.float32)
    c = _make(golden.copy(), golden, golden.copy(), params)
    assert c.compare()[2] is True  # 完全一致 → 比值 1.0

    c = _make(np.array([1.0, 1.0 + 1e-4], dtype=np.float32), golden, golden.copy(), params)
    assert c.compare()[2] is True  # rel_err=1e-4 < 3e-4

    c = _make(np.array([1.0, 1.01], dtype=np.float32), golden, golden.copy(), params)
    assert c.compare()[2] is False  # rel_err=1e-2 远超


_PARAMS = {
    "level": "L1",
    "mare_ratio": 5.0,
    "mere_ratio": 1.5,
    "rmse_ratio": 1.5,
    "small_value": 2**-14,
    "small_value_atol": 2**-30,
    "legacy": {"rtol": None, "ptol": None, "atol": 1e-8},
}


def test_element_count_mismatch_fails_not_raises():
    """#292：三方元素数不一致直接 FAIL（给出一元大小串），不再 IndexError/ValueError。"""
    r = _make(np.ones(2, np.float32), np.ones(2, np.float32), np.ones(1, np.float32), _PARAMS).compare()
    assert r[2] is False
    assert r[0] == "2 vs 2 vs 1"
    r2 = _make(np.ones(2, np.float32), np.ones(2, np.float32), np.ones(3, np.float32), _PARAMS).compare()
    assert r2[2] is False


def test_empty_outputs_vs_nonempty_third_party_fails():
    """#292：output/golden 都空而 third_party 非空 → FAIL（此前"都空→PASS"短路误通过）。"""
    r = _make(np.ones(0, np.float32), np.ones(0, np.float32), np.ones(3, np.float32), _PARAMS).compare()
    assert r[2] is False


def test_empty_outputs_missing_third_party_golden_failure():
    """#292：都空且三方缺失 → GOLDEN_FAILURE FAIL（不再被"都空→PASS"绕过既有语义）。"""
    r = _make(np.ones(0, np.float32), np.ones(0, np.float32), None, _PARAMS).compare()
    assert r[2] is False
    assert r[0] == "GOLDEN_FAILURE"


def test_all_empty_consistent_still_passes():
    """三方全空且一致 → PASS 语义保留。"""
    r = _make(np.ones(0, np.float32), np.ones(0, np.float32), np.ones(0, np.float32), _PARAMS).compare()
    assert r[2] is True
