# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""复检（precision recheck）统计引擎。

--rerun 多轮逐轮采集三方交叉校验（cross_check）的 mare/mere/rmse 三比值作为
样本；对每路样本做 Bootstrap 有放回重采样（1000 次）取中位数；95% 置信区间
取中位数集合的第 25 小 / 第 976 小（顺序统计量）。判定：任一比值 CI 下限
> 1.0 → FAIL（NPU 系统性精度劣于三方）。单轮已确认的 NaN/Inf 不匹配也保留为 FAIL。

纯 numpy、无 IO / 设备依赖，供 common/instance 统计复检聚合与单测共用。
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy

from .registry import FAIL_REASONS

__all__ = [
    "BOOTSTRAP_TIMES",
    "CI_LOWER_ORDER",
    "CI_UPPER_ORDER",
    "RATIO_KEYS",
    "RATIO_THRESHOLD",
    "RecheckCollector",
    "RecheckVerdict",
]

# 三方报告三比值（cross_check metrics["result"] 的键）
RATIO_KEYS = ("mare", "mere", "rmse")
# Bootstrap 重采样次数
BOOTSTRAP_TIMES = 1000
# 置信区间顺序统计量（1-based）：第 25 小 / 第 976 小
CI_LOWER_ORDER = 25
CI_UPPER_ORDER = 976
# 判定阈值：CI 下限大于 1.0 = 有统计学证据表明 NPU 系统性精度劣化
RATIO_THRESHOLD = 1.0

# 单轮 bootstrap 分块重采样的块大小（内存上界 256 × n 样本 × 8B）
_BOOTSTRAP_CHUNK = 256


def _derive_bootstrap_seed(base_seed, testcase_name: str, mode: str, output_idx: int, ratio: str) -> int:
    """单路 (mode, output, ratio) Bootstrap 重采样 RNG 的确定性种子——
    同输入复跑得到同区间（复检结论可复现）。"""
    digest = hashlib.blake2b(
        f"ttk-recheck-bootstrap:v1:{base_seed or 0}:{testcase_name}:{mode}:{output_idx}:{ratio}".encode(),
        digest_size=4,
    ).digest()
    return int.from_bytes(digest, "big")


def _bootstrap_median_ci(samples, n_boot: int = BOOTSTRAP_TIMES, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap 中位数 95% 置信区间（顺序统计量口径）。

    有放回重采样 n_boot 次（每次取 len(samples) 个），每次算中位数；
    升序排序后取第 CI_LOWER_ORDER / CI_UPPER_ORDER 小（1-based）。
    分块采样限制峰值内存（n_boot=1000、n=1000 时约 2MB/块）。
    样本含 inf 时中位数语义不变（safe_div 的比值可能为 inf）；
    NaN 样本应在采集侧剔除，此处再兜底过滤。
    """
    arr = numpy.asarray(
        [v for v in samples if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=numpy.float64
    )
    n = arr.size
    if n <= 0:
        raise ValueError("bootstrap_median_ci requires at least one valid sample")
    if n_boot < CI_UPPER_ORDER:
        raise ValueError(f"n_boot({n_boot}) must be >= CI_UPPER_ORDER({CI_UPPER_ORDER})")
    rng = numpy.random.default_rng(seed)
    medians = numpy.empty(n_boot, dtype=numpy.float64)
    filled = 0
    while filled < n_boot:
        chunk = min(_BOOTSTRAP_CHUNK, n_boot - filled)
        idx = rng.integers(0, n, size=(chunk, n))
        medians[filled : filled + chunk] = numpy.median(arr[idx], axis=1)
        filled += chunk
    medians.sort()
    return float(medians[CI_LOWER_ORDER - 1]), float(medians[CI_UPPER_ORDER - 1])


def _is_metrics_entry(payload) -> bool:
    """metrics 条目判定：比对层产物恒带 standard 键。"""
    return isinstance(payload, dict) and "standard" in payload


def _collect_ratio_samples(samples: list, mode: str, out_idx, metrics) -> None:
    """单个 cross_check 条目 → 三比值样本（跳过 None / NaN）。"""
    result = metrics.get("result") or {}
    for ratio in RATIO_KEYS:
        value = result.get(ratio)
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        samples.append((str(mode), int(out_idx), ratio, value))


def _iter_output_metrics(precision_metrics):
    """遍历单轮比对的模式、输出序号和 metrics 条目。

    兼容两种嵌套形状（按值是否为 metrics 条目分层判定）：
    - 带模式层（kernel/geir/e2e）：{"dyn": {out_idx: {...}}, "eager": ...}；
    - 无模式层（aclnn 直接落比对层产物）：{out_idx: {...}}——mode 记 ""。
    """
    for key, payload in (precision_metrics or {}).items():
        if _is_metrics_entry(payload):
            yield "", key, payload
        elif isinstance(payload, dict):
            for out_idx, metrics in payload.items():
                if _is_metrics_entry(metrics):
                    yield str(key), out_idx, metrics


class RecheckCollector:
    """N 轮复检的样本采集与判定。

    - add_run：一轮比对 metrics 入账（至少采到一个比值计为有效轮）；
    - add_failed_run：一轮执行失败（生成/golden/设备/三方异常）入账；
    - finish：逐路 bootstrap 置信区间 + 终判。
    """

    def __init__(self):
        self._samples: Dict[Tuple[str, int, str], List[float]] = {}
        self._nan_inf_mismatches = set()
        self.total_rounds = 0
        self.valid_rounds = 0
        self.failed_rounds = 0
        self.fail_markers: Dict[str, int] = {}

    def add_run(self, precision_metrics) -> None:
        self.total_rounds += 1
        samples = []
        for mode, out_idx, metrics in _iter_output_metrics(precision_metrics):
            if metrics["standard"] != "cross_check":
                continue
            # 特殊值位不参与比值统计，保留三方校验已确认的不匹配。
            if metrics.get("reason") == FAIL_REASONS["nan_inf_mismatch"]:
                self._nan_inf_mismatches.add((mode, int(out_idx)))
            _collect_ratio_samples(samples, mode, out_idx, metrics)
        if not samples:
            self.failed_rounds += 1
            self._bump_marker("NO_RATIO_SAMPLES")
            return
        self.valid_rounds += 1
        for mode, out_idx, ratio, value in samples:
            self._samples.setdefault((mode, out_idx, ratio), []).append(value)

    def add_failed_run(self, marker: Optional[str] = None) -> None:
        self.total_rounds += 1
        self.failed_rounds += 1
        self._bump_marker(marker or "RUN_FAILURE")

    def _bump_marker(self, marker: str) -> None:
        self.fail_markers[marker] = self.fail_markers.get(marker, 0) + 1

    def finish(self, base_seed, testcase_name: str) -> "RecheckVerdict":
        if not self._samples and not self._nan_inf_mismatches:
            return RecheckVerdict(
                "SKIP",
                reason="no cross_check ratio samples collected (standard is not cross_check / third-party unavailable)",
                total_rounds=self.total_rounds,
                valid_rounds=self.valid_rounds,
                failed_rounds=self.failed_rounds,
                fail_markers=dict(self.fail_markers),
                detail={},
            )
        status = "PASS"
        detail: Dict[Tuple[str, int, str], dict] = {}
        for key in sorted(self._samples):
            samples = self._samples[key]
            if not samples:
                continue
            mode, out_idx, ratio = key
            median = float(numpy.median(numpy.asarray(samples, dtype=numpy.float64)))
            ci_lower, ci_upper = _bootstrap_median_ci(
                samples,
                seed=_derive_bootstrap_seed(base_seed, testcase_name, mode, out_idx, ratio),
            )
            verdict = "FAIL" if ci_lower > RATIO_THRESHOLD else "PASS"
            if verdict == "FAIL":
                status = "FAIL"
            detail[key] = {
                "median": median,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(samples),
                "verdict": verdict,
            }
        reason = None
        if status == "FAIL":
            failed_paths = [
                f"{_path_label(mode, out_idx)}/{ratio}={entry['ci_lower']:.4g}"
                for (mode, out_idx, ratio), entry in sorted(detail.items())
                if entry["verdict"] == "FAIL"
            ]
            reason = "systematic precision degradation (bootstrap ci_lower>1.0): " + ", ".join(failed_paths)
        if self._nan_inf_mismatches:
            status = "FAIL"
            paths = ", ".join(_path_label(mode, out_idx) for mode, out_idx in sorted(self._nan_inf_mismatches))
            mismatch_reason = f"{FAIL_REASONS['nan_inf_mismatch']}: {paths}"
            reason = f"{mismatch_reason}. {reason}" if reason else mismatch_reason
        return RecheckVerdict(
            status,
            reason=reason,
            total_rounds=self.total_rounds,
            valid_rounds=self.valid_rounds,
            failed_rounds=self.failed_rounds,
            fail_markers=dict(self.fail_markers),
            detail=detail,
        )


def _path_label(mode: str, out_idx: int) -> str:
    """失败路展示段：mode 有值带层（dyn/out0），aclnn 单层无 mode（out0）。"""
    return f"{mode}/out{out_idx}" if mode else f"out{out_idx}"


@dataclass
class RecheckVerdict:
    """复检终判与 CSV 行组装（metrics_payload 嵌套 dict 可经 str() 落列）。"""

    status: str
    reason: Optional[str] = None
    total_rounds: int = 0
    valid_rounds: int = 0
    failed_rounds: int = 0
    fail_markers: Dict[str, int] = field(default_factory=dict)
    detail: Dict[Tuple[str, int, str], dict] = field(default_factory=dict)

    def metrics_payload(self) -> dict:
        """detail → 嵌套 dict {mode: {out_idx: {ratio: {median/ci/verdict}}}}。"""
        nested: Dict[str, Dict[str, Dict[str, dict]]] = {}
        for (mode, out_idx, ratio), entry in sorted(self.detail.items()):
            nested.setdefault(mode, {}).setdefault(str(out_idx), {})[ratio] = dict(entry)
        return nested

    def summary_line(self) -> str:
        return (
            f"{self.status} (rounds={self.valid_rounds}/{self.total_rounds}"
            + (f", failed={self.failed_rounds}" if self.failed_rounds else "")
            + (f", reason={self.reason}" if self.reason else "")
            + ")"
        )
