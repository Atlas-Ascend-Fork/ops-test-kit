# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""cross_check: 三方交叉校验（NPU / golden / third_party）。

分块实现：三方 astype 提升与全部掩码/误差数组均在块内（O(CHUNK_SIZE)），
mare 为块间 running max（精确）、mere/rmse 为 float64 分块累加（与全量
fp32 mean 有 ~1e-7 相对漂移，远小于比值阈值）、diff 只保留各路径 top-K
（仅 _log_diff_output 消费）。
"""

import math

import numpy as np

from .registry import CHUNK_SIZE, FAIL_REASONS, ComparisonBase, EachCompareResult, _TopK, register_comparison


def safe_div(num, den, err):
    """三比值的除法:分母一律夹小值域阈值 err。

    精度标准:正常值域的 mare/mere/rmse 三比值,分母按 dtype 取的 small_value
    (见 resolve.py)夹底。err 既防除零,也保证竞品误差落在噪声地板以下时不把比值
    放大——固定常数会让判定随输出量级漂移。
    """
    if np.isnan(num) or np.isnan(den):
        return float("inf")
    if np.isinf(num) and np.isinf(den):
        return 1.0  # both overflow -> ratio=1 (consistent)
    if np.isinf(num):
        return float("inf")  # NPU overflow, third_party finite -> NPU worse
    if num == 0 and den == 0:
        return 1.0  # both perfect match -> ratio=1 (consistent)
    return float(num / max(den, err))


def _topk_list(topk: _TopK) -> list:
    r = topk.result()
    return r.tolist() if r is not None else []


class _ScanState:
    """分块扫描累计量（special/large/small 三路径）。"""

    __slots__ = (
        "special_ok",
        "n_large",
        "rn_sum",
        "rp_sum",
        "rn_max",
        "rp_max",
        "sq_n",
        "sq_p",
        "party_all_invalid",
        "err_target",
        "err_third",
        "top_large",
        "top_small",
    )

    def __init__(self, k: int):
        self.special_ok = True
        self.n_large = 0
        self.rn_sum = 0.0
        self.rp_sum = 0.0
        self.rn_max = float("-inf")
        self.rp_max = float("-inf")
        self.sq_n = 0.0
        self.sq_p = 0.0
        self.party_all_invalid = True
        self.err_target = 0
        self.err_third = 0
        self.top_large = _TopK(k)
        self.top_small = _TopK(k)


@register_comparison(["cross_check"])
class CrossCheckComparison(ComparisonBase):
    STANDARD_NAME = "cross_check"

    def _precheck(self):
        """三方前置校验：third_party 元素数与 output 不一致 → FAIL（异长会
        IndexError/ValueError，空输出 vs 非空三方是位数不匹配）；third_party
        缺失时不短路，空输出也走 compare_impl 的 GOLDEN_FAILURE。output 与
        golden 的元素数校验由基类统一处理。"""
        if self.third_party is None:
            return None
        if self.third_party.size != self.output.size:
            precision = f"{self.output.size} vs {self.golden.size} vs {self.third_party.size}"
            log = f"Output {self.output_idx} size is different among output/golden/third_party: {precision}\n"
            return EachCompareResult(precision, None, False, log)
        return super()._precheck()

    def compare_impl(self) -> EachCompareResult:
        t, g, b = self.output, self.golden, self.third_party
        if b is None:
            return EachCompareResult(
                "GOLDEN_FAILURE",
                is_pass=False,
                standard=self.STANDARD_NAME,
                metrics={
                    "standard": self.STANDARD_NAME,
                    "pass": False,
                    "level": self.tol_options.get("level"),
                    "reason": FAIL_REASONS["third_party_unavailable"],
                },
            )

        T = np.promote_types(np.float32, np.promote_types(t.dtype, np.promote_types(g.dtype, b.dtype)))
        sv = {"small_value": self.tol_options["small_value"], "small_value_atol": self.tol_options["small_value_atol"]}
        limits = {
            "mare": self.tol_options["mare_ratio"],
            "mere": self.tol_options["mere_ratio"],
            "rmse": self.tol_options["rmse_ratio"],
        }

        state = _ScanState(self.MAX_DIFF_OUTPUT)
        with np.errstate(invalid="ignore", divide="ignore"):
            for s in range(0, t.size, CHUNK_SIZE):
                self._scan_chunk(
                    t[s : s + CHUNK_SIZE].astype(T, copy=False),
                    g[s : s + CHUNK_SIZE].astype(T, copy=False),
                    b[s : s + CHUNK_SIZE].astype(T, copy=False),
                    s,
                    sv,
                    state,
                )

        large_result = self._assemble_large(state, sv, limits)
        small_result = self._assemble_small(state)

        passed = large_result.ratio_ok and state.special_ok and small_result.ok
        diff_idx = large_result.diff_idx + small_result.diff_idx

        reason = self._build_reason(passed, state.special_ok, small_result, large_result)
        metrics = self._build_metrics(limits, sv, large_result, small_result, passed, reason)

        _diff_arr = np.array(diff_idx, dtype=int) if diff_idx else None
        return EachCompareResult(
            "PASS" if passed else "FAIL",
            is_pass=passed,
            diff_index=_diff_arr if not passed else None,
            standard=self.STANDARD_NAME,
            metrics=metrics,
        )

    def _scan_chunk(self, tc, gc, bc, s, sv, state: _ScanState):
        """单块三路径扫描：special 交叉校验 / large 比值累计 / small 误差计数。"""
        special = np.isnan(tc) | np.isinf(tc) | np.isnan(gc) | np.isinf(gc) | np.isnan(bc) | np.isinf(bc)
        finite = ~special
        if state.special_ok and special.any() and self._special_fail(tc, gc, bc, special).any():
            state.special_ok = False
        large = finite & (np.abs(gc) >= sv["small_value"])
        small = finite & (np.abs(gc) < sv["small_value"])

        if large.any():
            tn, gn, bn = tc[large], gc[large], bc[large]
            rel_n = np.abs(tn - gn) / (np.abs(gn) + 1e-7)
            rel_p = np.abs(bn - gn) / (np.abs(gn) + 1e-7)
            state.n_large += int(tn.size)
            state.rn_sum += float(rel_n.sum(dtype=np.float64))
            state.rp_sum += float(rel_p.sum(dtype=np.float64))
            state.rn_max = max(state.rn_max, float(rel_n.max()))
            state.rp_max = max(state.rp_max, float(rel_p.max()))
            state.sq_n += float(np.square((tn - gn).astype(np.float64)).sum())
            state.sq_p += float(np.square((bn - gn).astype(np.float64)).sum())
            # third_party 在本块 large 位全 inf/nan（算子数学溢出，如 npu_quant_matmul
            # 用负 scale）——与全量 np.all 语义一致：逐块求全、块间求与
            state.party_all_invalid = state.party_all_invalid and bool(np.all(np.isinf(rel_p) | np.isnan(rel_p)))
            d = np.abs(tn - gn)
            mis = d > 0
            if mis.any():
                state.top_large.add(d[mis], (np.flatnonzero(large) + s)[mis])

        if small.any():
            ts, gs, bs = tc[small], gc[small], bc[small]
            state.err_target += int(np.sum(np.abs(ts - gs) > sv["small_value_atol"]))
            state.err_third += int(np.sum(np.abs(bs - gs) > sv["small_value_atol"]))
            d = np.abs(ts - gs)
            mis = d > 0
            if mis.any():
                state.top_small.add(d[mis], (np.flatnonzero(small) + s)[mis])

    @staticmethod
    def _special_fail(t, g, b, special) -> np.ndarray:
        """Path A: 特殊位（NaN/Inf）三方交叉校验——失败位掩码。

        规则（满足任一即通过）：
          1. NPU 与 third_party 一致（同为 nan/+inf/-inf，或同为有限且数值相等）→ 通过
          2. NPU 与 golden 一致（同为 nan/+inf/-inf，或同为有限且数值相等）→ 通过
          以上都不满足（NPU 与 golden、third_party 双方均不一致）→ 不通过

        规则2 不要求 golden 非有限：golden 有限时真值明确，NPU 与其一致即无偏差，
        third_party 单方面的 NaN/Inf 属竞品缺陷（如 fp16 中间量溢出后 0*inf），
        不应归因为 NPU 的测试失败——失败语义与 Path B 的比值哲学对齐：
        只有 NPU 偏离 golden 且竞品能对上 golden 才判 NPU 失败。
        """
        t_nan, g_nan, b_nan = np.isnan(t), np.isnan(g), np.isnan(b)
        t_pinf, g_pinf, b_pinf = np.isposinf(t), np.isposinf(g), np.isposinf(b)
        t_ninf, g_ninf, b_ninf = np.isneginf(t), np.isneginf(g), np.isneginf(b)
        t_fin, g_fin, b_fin = np.isfinite(t), np.isfinite(g), np.isfinite(b)

        # 规则1：NPU 与 third_party 一致（同为 nan/+inf/-inf，或同为有限且数值相等）
        tb_match = (t_nan & b_nan) | (t_pinf & b_pinf) | (t_ninf & b_ninf) | (t_fin & b_fin & (t == b))
        # 规则2：NPU 与 golden 一致（同为 nan/+inf/-inf，或同为有限且数值相等）
        tg_match = (t_nan & g_nan) | (t_pinf & g_pinf) | (t_ninf & g_ninf) | (t_fin & g_fin & (t == g))

        return special & ~(tb_match | tg_match)

    @staticmethod
    def _assemble_large(state: _ScanState, sv, limits) -> "_LargeResult":
        """Path B-1: 大值域 mare/mere/rmse 三比值判定（分块累计量 → 三比值）。"""
        if state.n_large == 0:
            return _LargeResult(ratio_ok=True, diff_idx=[])
        if state.party_all_invalid:
            # ratio 不参与判定，只看 target vs golden
            return _LargeResult(ratio_ok=True, diff_idx=_topk_list(state.top_large))

        mare_ratio = safe_div(state.rn_max, state.rp_max, sv["small_value"])
        mere_ratio = safe_div(state.rn_sum / state.n_large, state.rp_sum / state.n_large, sv["small_value"])
        rmse_npu = math.sqrt(state.sq_n / state.n_large)
        rmse_party = math.sqrt(state.sq_p / state.n_large)
        rmse_ratio = safe_div(rmse_npu, rmse_party, sv["small_value"])

        exceeded = []
        if mare_ratio > limits["mare"]:
            exceeded.append(f"mare({mare_ratio:.2f}>{limits['mare']})")
        if mere_ratio > limits["mere"]:
            exceeded.append(f"mere({mere_ratio:.2f}>{limits['mere']})")
        if rmse_ratio > limits["rmse"]:
            exceeded.append(f"rmse({rmse_ratio:.2f}>{limits['rmse']})")

        return _LargeResult(
            ratio_ok=not exceeded,
            mare=mare_ratio,
            mere=mere_ratio,
            rmse=rmse_ratio,
            exceeded=exceeded,
            diff_idx=_topk_list(state.top_large),
        )

    @staticmethod
    def _assemble_small(state: _ScanState) -> "_SmallResult":
        """Path B-2: 小值域 ErrorCount 比值判定。"""
        small_ratio = state.err_target / max(state.err_third, 1)
        return _SmallResult(
            ok=small_ratio <= 2.0,
            err_target=state.err_target,
            err_third=state.err_third,
            ratio=small_ratio,
            diff_idx=_topk_list(state.top_small),
        )

    @staticmethod
    def _build_reason(passed, special_ok, small_result, large_result):
        if passed:
            return None
        if not special_ok:
            return FAIL_REASONS["nan_inf_mismatch"]
        if not small_result.ok:
            return FAIL_REASONS["small_value_exceeded"].format(ratio=f"{small_result.ratio:.2f}>2.0")
        return FAIL_REASONS["ratio_exceeded"].format(exceeded=", ".join(large_result.exceeded))

    def _build_metrics(self, limits, sv, large_result, small_result, passed, reason):
        metrics = {
            "standard": self.STANDARD_NAME,
            "level": self.tol_options.get("level"),
            "config": {
                "mare": limits["mare"],
                "mere": limits["mere"],
                "rmse": limits["rmse"],
                "small_value": sv["small_value"],
                "small_value_atol": sv["small_value_atol"],
            },
            "result": {
                "mare": float(large_result.mare) if large_result.mare is not None else None,
                "mere": float(large_result.mere) if large_result.mere is not None else None,
                "rmse": float(large_result.rmse) if large_result.rmse is not None else None,
                "small_err_cnt_target": small_result.err_target,
                "small_err_cnt_third": small_result.err_third,
                "small_err_ratio": float(small_result.ratio),
            },
            "pass": passed,
        }
        if reason:
            metrics["reason"] = reason
        return metrics

    def _log_diff_output(self, diff_index, diff_order=None):
        """Three-party diff (NPU/golden/third_party + |t-g| + |b-g|), worst-first.
        large positions use %.6e, small positions use %.10e (small values need
        more digits). [L]/[S] tag distinguishes large/small in each row."""
        log = ""
        if diff_index is None or len(diff_index) == 0:
            return log
        t, g, b = self.output, self.golden, self.third_party
        small_value = self.tol_options.get("small_value", 1e-7)
        n = len(diff_index)
        log += f"Output {self.output_idx} Compare Difference length {n} ({diff_order or 'worst-first'})\n"
        for idx in range(n):
            i = int(diff_index[idx])
            is_small = abs(float(g[i])) < small_value
            fmt = "%.10e" if is_small else "%.6e"
            tag = "S" if is_small else "L"
            log += (
                f"Index: {idx:03d} RealIndex: {i:06d} [{tag}] "
                f"NPU(t)={fmt % float(t[i])} golden(g)={fmt % float(g[i])} third_party(b)={fmt % float(b[i])} "
                f"|t-g|={fmt % float(abs(t[i] - g[i]))} |b-g|={fmt % float(abs(b[i] - g[i]))}\n"
            )
        return log


class _LargeResult:
    __slots__ = ("ratio_ok", "mare", "mere", "rmse", "exceeded", "diff_idx")

    def __init__(self, ratio_ok, mare=None, mere=None, rmse=None, exceeded=None, diff_idx=None):
        self.ratio_ok = ratio_ok
        self.mare = mare
        self.mere = mere
        self.rmse = rmse
        self.exceeded = exceeded or []
        self.diff_idx = diff_idx or []


class _SmallResult:
    __slots__ = ("ok", "err_target", "err_third", "ratio", "diff_idx")

    def __init__(self, ok, err_target, err_third, ratio, diff_idx):
        self.ok = ok
        self.err_target = err_target
        self.err_third = err_third
        self.ratio = ratio
        self.diff_idx = diff_idx or []
