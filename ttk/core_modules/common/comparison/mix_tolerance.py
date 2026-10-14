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
mix_tolerance comparison — 生态算子开源精度标准（混合容差）

分块实现：所有中间数组 O(CHUNK_SIZE) 不随元素数增长。matched_ratio 为精确
整数计数；max_abs_error 为块间 running max（精确）；
diff_index 只保留 top-K（worst-first，仅 _log_diff_output 消费）。
"""

# Standard Packages
import numpy as np

# Third-party Packages
from .registry import CHUNK_SIZE, FAIL_REASONS, ComparisonBase, EachCompareResult, _TopK, register_comparison


class _MixScan:
    """分块扫描累计量（compare_impl 判定输入）。"""

    __slots__ = ("matched", "has_mismatch", "has_finite", "max_abs_error", "topk")

    def __init__(self, k: int):
        self.matched = 0
        self.has_mismatch = False
        self.has_finite = False
        self.max_abs_error = 0.0
        self.topk = _TopK(k)


@register_comparison(["mixed", "mix_tolerance"])  # mixed=CLI 简写；mix_tolerance=Spec.tolerance 官方标准名
class MixToleranceComparison(ComparisonBase):
    STANDARD_NAME = "mix_tolerance"

    def compare_impl(self) -> EachCompareResult:
        # 空数组与元素数已在 compare() 入口 _precheck 统一短路，这里只处理非空同数
        # rtol/atol/required_matched_ratio/max_abs_error_limit 均为 resolve_tolerance 解析好的最终值（不查表）
        actual, golden = self.output, self.golden
        if actual.size != golden.size:
            return EachCompareResult(
                f"{actual.size} vs {golden.size}",
                is_pass=False,
                standard=self.STANDARD_NAME,
                metrics={"standard": self.STANDARD_NAME, "pass": False, "reason": "output size != golden size"},
            )
        T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
        rtol = self.tol_options["rtol"]
        atol = self.tol_options["atol"]
        required_ratio = self.tol_options["required_matched_ratio"]
        # （对齐 cann/opbase mixed_tolerance_standard）上限解析：显式配置 → 完全替代（不叠加 ULP）；未配置 →
        # 动态式 max(floor, 32*ULP(g_low))，g_low 为最大有限误差点 golden 收窄值
        # ——上限随被测 dtype 与值域自适应缩放。inf-cast（actual=±inf 处 golden
        # RNE 收窄，同号 ±inf 一致）在 _scan_chunks / _ulp_dynamic_limit 块内执行。
        max_err_limit = self._resolve_err_limit(actual, golden, T)
        scan = self._scan_chunks(actual, golden, T, rtol, atol, max_err_limit)
        matched_ratio = scan.matched / actual.size
        # max_abs_error 取有限元素对的最大绝对误差；NaN/Inf 不一致视为无界误差 → 必超硬上限
        if scan.has_mismatch:
            max_abs_error = float("inf")
        elif not scan.has_finite:  # 全非有限且一致
            max_abs_error = 0.0
        else:
            max_abs_error = scan.max_abs_error
        # 整体通过条件：matched_ratio 达标 且 max_abs_error 不超硬上限
        passed = matched_ratio >= required_ratio and max_abs_error <= max_err_limit

        if passed:
            diff_idx, reason = None, None
        else:  # 未通过元素按误差降序（NaN 视作 +inf 排最前）；top-K，log 控制打印数
            diff_idx = scan.topk.result()
            parts = []
            if scan.has_mismatch:
                parts.append(FAIL_REASONS["nan_inf_mismatch"])
            exceeded = []
            if matched_ratio < required_ratio:
                exceeded.append(f"matched_ratio({matched_ratio:.4f}<{required_ratio})")
            if max_abs_error > max_err_limit:
                exceeded.append(f"max_abs_error({max_abs_error:.2e}>{max_err_limit:.2e})")
            if exceeded:
                parts.append(FAIL_REASONS["threshold_exceeded"].format(metrics=", ".join(exceeded)))
            reason = "; ".join(parts)

        return EachCompareResult(
            matched_ratio,
            diff_index=diff_idx,
            is_pass=passed,
            standard=self.STANDARD_NAME,
            diff_order="worst-first by error" if diff_idx is not None else None,
            metrics=_metrics(rtol, atol, matched_ratio, required_ratio, max_abs_error, max_err_limit, passed, reason),
        )

    def _resolve_err_limit(self, actual, golden, T):
        """!292 上限终值：显式配置 max_abs_error_limit → 原值（完全替代，不叠加
        ULP）；未配置 → 动态式 max(floor, 32*ULP(g_low))。"""
        limit = self.tol_options["max_abs_error_limit"]
        if limit is not None:
            return limit
        return self._ulp_dynamic_limit(actual, golden, T, self.tol_options["max_abs_error_floor"])

    def _ulp_dynamic_limit(self, actual, golden, T, floor):
        """!292 动态上限预扫描：max(floor, 32*ULP(g_low))——g_low 为最大有限误差点
        的 golden 收窄值（RNE 到被测 dtype）。分块追踪 argmax，内存 O(CHUNK_SIZE)。

        最大有限值处 spacing=inf，退一格取同 binade 前驱（binade 内格宽恒定，如
        fp16 最大 binade 格宽=32）。锚点超 dtype 范围时 ulp=nan，max() 依赖 floor
        在前回退（勿调换参数顺序），且该场景 rtol 分支先行拦截，limit 取值无影响。"""
        best_err = -np.inf
        best_idx = -1
        with np.errstate(invalid="ignore", divide="ignore"):
            for s in range(0, actual.size, CHUNK_SIZE):
                a = actual[s : s + CHUNK_SIZE].astype(T, copy=False)
                g = golden[s : s + CHUNK_SIZE].astype(T, copy=False)
                if np.isinf(a).any():
                    g = np.where(np.isinf(a), golden[s : s + CHUNK_SIZE].astype(actual.dtype).astype(T), g)
                finite = np.isfinite(a) & np.isfinite(g)
                if finite.any():
                    err = np.abs(a - g)
                    masked = np.where(finite, err, -np.inf)
                    i = int(np.argmax(masked))
                    if float(masked[i]) > best_err:
                        best_err = float(masked[i])
                        best_idx = s + i
        if best_idx < 0:
            return floor
        g_low = np.asarray(golden).ravel()[best_idx].astype(actual.dtype)
        ulp = np.spacing(np.abs(g_low))
        if np.isinf(ulp):
            ulp = np.spacing(np.nextafter(np.abs(g_low), g_low.dtype.type(0), dtype=g_low.dtype))
        return max(floor, 32.0 * float(np.asarray(ulp).astype(T)))

    def _scan_chunks(self, actual, golden, T, rtol, atol, max_err_limit) -> _MixScan:
        """逐块扫描：matched 计数 + mismatch/finite 标记 + max_abs_error running max + top-K 候选。

        所有中间数组 O(CHUNK_SIZE) 不随元素数增长；dtype 提升在块内完成
        （同 dtype astype(copy=False) 零拷贝）。"""
        scan = _MixScan(self.MAX_DIFF_OUTPUT)
        with np.errstate(invalid="ignore", divide="ignore"):
            for s in range(0, actual.size, CHUNK_SIZE):
                a = actual[s : s + CHUNK_SIZE].astype(T, copy=False)
                g = golden[s : s + CHUNK_SIZE].astype(T, copy=False)
                # inf-cast（!292）：actual=±inf 处 golden RNE 收窄到被测 dtype
                # （超范围→±inf），同号 ±inf 视为一致；同 dtype 收窄恒等
                if np.isinf(a).any():
                    g = np.where(
                        np.isinf(a),
                        golden[s : s + CHUNK_SIZE].astype(actual.dtype).astype(T),
                        g,
                    )
                err = np.abs(a - g)
                a_nan, g_nan = np.isnan(a), np.isnan(g)
                a_inf, g_inf = np.isinf(a), np.isinf(g)
                same_nan = a_nan & g_nan  # 都 NaN（NaN 无符号）
                same_inf = a_inf & g_inf & (np.sign(a) == np.sign(g))  # 都 Inf 且同号
                mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
                finite = np.isfinite(a) & np.isfinite(g)
                # 逐元素通过条件：|actual - golden| <= atol + rtol * |golden|（atol 天然避免除零）
                # 且 <= max_abs_error_limit（硬上限并入逐元素判定——不并入时
                # 全部元素通过 rtol/atol 而绝对误差超硬上限的场景会得到 matched_ratio=100%
                # 却 FAIL 且 diff_index 为空，precision 误导、日志无出错明细。并入后判定等价：
                # 硬上限未超时该条件恒真（ratio 不变）；超限时新旧判定均 FAIL）。
                elem_ok = same_nan | same_inf | (finite & (err <= atol + rtol * np.abs(g)) & (err <= max_err_limit))
                scan.matched += int(elem_ok.sum())
                scan.has_mismatch = scan.has_mismatch or bool(mismatch.any())
                if finite.any():
                    scan.has_finite = True
                    scan.max_abs_error = max(scan.max_abs_error, float(err[finite].max()))
                bad = np.flatnonzero(~elem_ok)
                if bad.size:  # 排序键与全量实现一致：NaN 误差视作 +inf 排最前
                    scan.topk.add(np.where(np.isnan(err[bad]), np.inf, err[bad]), bad + s)
        return scan


def _metrics(rtol, atol, matched_ratio, required_ratio, max_abs_error, max_err_limit, passed, reason=None):
    # 全 Python 字面量，保 CSV eval 往返；max_abs_error=None（inf）当 NaN/Inf mismatch
    m = {
        "standard": "mix_tolerance",  # 模块级 _metrics，标准名固定
        "rtol": rtol,
        "atol": atol,
        "matched_ratio": matched_ratio,
        "required_matched_ratio": required_ratio,
        "max_abs_error": None if np.isinf(max_abs_error) else max_abs_error,
        "max_abs_error_limit": max_err_limit,
        "pass": bool(passed),
    }
    if reason:
        m["reason"] = reason
    return m
