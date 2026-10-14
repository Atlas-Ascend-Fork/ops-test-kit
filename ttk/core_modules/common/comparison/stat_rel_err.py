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
stat_rel_err comparison — 相对误差统计精度标准（MERE/MARE）
"""

import numpy as np

from ....utilities.dtypes import dtype_to_name
from .registry import CHUNK_SIZE, FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison

# 算法常量（不是配置——配置由 resolve_tolerance 解析后经 params 传入）
MARE_RATIO = 10
# 相消噪声地板（rel_err 分母下限）= 输出 dtype 的 1 ulp（@1.0），表外 dtype 回落
# FLOOR。固定 1e-7 与 dtype 无关：近零 golden 处中间量残留被放大成 O(1) 的
# "相对误差"，mare 单点即挂整张用例。fp16/bf16 档与 SMALL_VALUE 同值；恰好
# 1 ulp 的偏差 re=1 仍判 FAIL。
NOISE_FLOOR = {
    "float16": 2.0**-11,
    "bfloat16": 2.0**-8,
    "float32": 2.0**-23,
}
FLOOR = 1e-7
# CHUNK_SIZE 见 registry（判据层共用）；分块降内存动机与 top-K 语义见彼处注释。


class _ChunkScan:
    """分块扫描累计量（compare_impl 判定输入）。"""

    __slots__ = ("mism", "n_finite", "err_sum", "mare", "cand_err", "cand_idx")

    def __init__(self):
        self.mism = []  # 前 K 个 NaN/Inf 不一致位（全局升序），够 K 个即停
        self.n_finite = 0
        self.err_sum = 0.0  # float64 累加各块有限元素 rel_err 之和
        self.mare = 0.0
        self.cand_err, self.cand_idx = [], []  # 各块局部 top-K 候选，循环后归并取全局 top-K


@register_comparison("stat_rel_err")
class StatRelErrComparison(ComparisonBase):
    STANDARD_NAME = "stat_rel_err"

    def compare_impl(self) -> EachCompareResult:
        # 空数组与元素数已在 compare() 入口 _precheck 统一短路，这里只处理非空同数
        # threshold 是 resolve_tolerance 解析好的最终值（经 params → options 传入），不查表
        actual, golden = self.output, self.golden
        T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
        th = self.tol_options["threshold"]
        floor = NOISE_FLOOR.get(dtype_to_name(self.output_dtype), FLOOR)
        scan = self._scan_chunks(actual, golden, T, floor)

        if scan.mism:  # NaN/Inf 不一致：污染，不算 mere/mare
            return EachCompareResult(
                "FAIL",
                diff_index=np.asarray(scan.mism, dtype=np.int64),
                is_pass=False,
                standard=self.STANDARD_NAME,
                metrics=_metrics(th, None, None, False, FAIL_REASONS["nan_inf_mismatch"]),
            )
        if scan.n_finite == 0:  # 全非有限且一致
            return EachCompareResult(
                "PASS", is_pass=True, standard=self.STANDARD_NAME, metrics=_metrics(th, None, None, True)
            )
        mere = scan.err_sum / scan.n_finite
        passed = mere < th and scan.mare < MARE_RATIO * th
        if passed:
            diff_idx = None
            reason = None
        else:  # 数值 FAIL：top-K 按 rel_err 降序(worst first)
            err = np.concatenate(scan.cand_err)
            idx = np.concatenate(scan.cand_idx)
            diff_idx = idx[np.argsort(-err)[: self.MAX_DIFF_OUTPUT]]
            exceeded = []
            if mere >= th:
                exceeded.append(f"mere({mere:.2e}>={th})")
            if scan.mare >= MARE_RATIO * th:
                exceeded.append(f"mare({scan.mare:.2e}>={MARE_RATIO * th})")
            reason = FAIL_REASONS["threshold_exceeded"].format(metrics=", ".join(exceeded))

        return EachCompareResult(
            "PASS" if passed else "FAIL",
            diff_index=diff_idx,
            is_pass=passed,
            standard=self.STANDARD_NAME,
            diff_order="worst-first by error" if diff_idx is not None else None,
            metrics=_metrics(th, mere, scan.mare, passed, reason),
        )

    def _scan_chunks(self, actual, golden, T, floor) -> _ChunkScan:
        """按 CHUNK_SIZE 分块扫描：mismatch 位（前 K 个）+ 有限元素 mere/mare 累计 + top-K 候选。

        所有中间数组均为 O(CHUNK_SIZE) 不随元素数增长；dtype 提升在块内做免全量拷贝。
        一旦发现 mismatch，结论已定（FAIL），后续块只为补齐前 K 个 mismatch 位置。"""
        k = self.MAX_DIFF_OUTPUT  # _log_diff_output 实际打印的行数
        scan = _ChunkScan()
        with np.errstate(invalid="ignore", divide="ignore"):
            for s in range(0, actual.size, CHUNK_SIZE):
                # 切片上标自动截断到 n，最后一块可小于 CHUNK_SIZE
                ac = actual[s : s + CHUNK_SIZE].astype(T, copy=False)
                gc = golden[s : s + CHUNK_SIZE].astype(T, copy=False)
                a_nan, g_nan = np.isnan(ac), np.isnan(gc)
                a_inf, g_inf = np.isinf(ac), np.isinf(gc)
                same_nan = a_nan & g_nan  # 都 NaN（NaN 无符号）
                same_inf = a_inf & g_inf & (np.sign(ac) == np.sign(gc))  # 都 Inf 且同号
                mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
                if mismatch.any():
                    need = k - len(scan.mism)
                    scan.mism.extend((np.flatnonzero(mismatch)[:need] + s).tolist())
                    if len(scan.mism) >= k:
                        break
                    continue
                if scan.mism:  # 结论已定（mismatch FAIL），后续块只为补齐前 K 个位置
                    continue
                fin = np.isfinite(ac) & np.isfinite(gc)
                nf = int(fin.sum())
                if not nf:
                    continue
                scan.n_finite += nf
                re = np.abs(ac - gc) / (np.abs(gc) + floor)
                re_f = re[fin]
                scan.err_sum += float(re_f.sum(dtype=np.float64))
                scan.mare = max(scan.mare, float(re_f.max()))
                idx_f = np.flatnonzero(fin) + s
                if re_f.size > k:  # 块内只留 top-K：全局 top-K 必在诸块 top-K 的并集内
                    part = np.argpartition(re_f, re_f.size - k)[re_f.size - k :]
                    re_f, idx_f = re_f[part], idx_f[part]
                scan.cand_err.append(re_f)
                scan.cand_idx.append(idx_f)
        return scan


def _metrics(th, mere, mare, passed, reason=None):
    # 全 Python 字面量，保 CSV eval 往返；mere/mare=None（非 NaN）当 mismatch 或全 match
    m = {
        "standard": "stat_rel_err",
        "mere": mere,
        "mare": mare,
        "threshold": th,
        "pass": bool(passed),
    }  # 模块级 _metrics，标准名固定
    if reason:
        m["reason"] = reason
    return m
