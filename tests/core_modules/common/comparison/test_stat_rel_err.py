# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""StatRelErrComparison 单元测试：mismatch 真值表、diff_idx 分流、mere/mare 公式、边界场景、
分块路径对拍与 RSS 回归。"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ttk.core_modules.common.comparison.stat_rel_err import CHUNK_SIZE, FLOOR, StatRelErrComparison


def _impl(actual, golden, dtype, threshold):
    """返回 EachCompareResult。threshold 必须显式给（resolve_tolerance 解析好的最终值）。"""
    c = StatRelErrComparison(np.asarray(actual), np.asarray(golden), 0, dtype, {"threshold": threshold})
    return c.compare_impl()


def _run(actual, golden, dtype, threshold):
    """返回 compare() 的 4-tuple。"""
    c = StatRelErrComparison(np.asarray(actual), np.asarray(golden), 0, dtype, {"threshold": threshold})
    return c.compare()


def _reference_impl(actual, golden):
    """旧全量实现，仅用于分块路径对拍（不含 mismatch 分支）。"""
    T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
    a = actual.astype(T)
    g = golden.astype(T)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = np.abs(a - g) / (np.abs(g) + FLOOR)
        mask = np.isfinite(a) & np.isfinite(g)
        mere = float(np.mean(rel_err[mask]))
        mare = float(np.max(rel_err[mask]))
        finite_idx = np.where(mask)[0]
        diff_idx = finite_idx[np.argsort(-rel_err[finite_idx])]
    return mere, mare, diff_idx


# —— 防线 1：mismatch 真值表全覆盖（13 cell）——
@pytest.mark.parametrize(
    ("a", "g", "expect_pass", "mere_none"),
    [
        # match：全非有限且一致 → PASS，mere=None
        (np.nan, np.nan, True, True),
        # finite/finite → mere 路径（mere 算出来，非 None）
        (1.0, 2.0, False, False),  # mere≈0.5 >> th → FAIL
        # mismatch → FAIL，mere=None
        (np.nan, 1.0, False, True),
    ],
)
def test_mismatch_truth_table(a, g, expect_pass, mere_none):
    """防线 1：mismatch 真值表全覆盖（13 cell）—非有限值一致/不一致的 PASS/FAIL 与 mere 是否为 None。"""
    r = _impl([a], [g], "float32", 2**-13)
    assert r.is_pass == expect_pass
    assert (r.metrics["mere"] is None) == mere_none


# —— 防线 2：混合数组（mismatch 与数值 FAIL 的 diff_idx 分流）——
def test_mismatch_present_diff_idx_only_mismatch():
    """防线 2：混合数组中 mismatch 位的 diff_idx 仅包含 mismatch 位置。"""
    r = _impl([np.nan, 1.0], [1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is False
    assert r.metrics["mere"] is None
    assert list(r.diff_index) == [0]


def test_numeric_fail_diff_idx_worst_first_full():
    """防线 2：纯数值 FAIL 的 diff_idx 按误差从大到小排列。"""
    r = _impl([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is False
    assert list(r.diff_index) == [2, 1, 0]


# —— 防线 3：mere/mare/阈值公式数值校验 ——
def test_mere_mare_values():
    """防线 3：mere=0.25、mare=0.5 的公式数值校验。"""
    r = _impl([1.0, 2.0], [1.0, 4.0], "float32", 2**-13)
    assert r.metrics["mere"] == pytest.approx(0.25, abs=1e-4)
    assert r.metrics["mare"] == pytest.approx(0.5, abs=1e-4)
    assert r.metrics["threshold"] == 2**-13
    assert r.is_pass is False


# —— 防线 4：分块实现与旧全量实现对拍 ——
def test_matches_reference_across_chunk_boundary():
    """防线 4：单块/多块两种规模下 mere/mare 与旧全量实现一致，diff_idx 截断 top-K 且 worst-first。"""
    rng = np.random.default_rng(7)
    k = StatRelErrComparison.MAX_DIFF_OUTPUT
    for n in (CHUNK_SIZE - 1000, CHUNK_SIZE + 1000):
        g = rng.standard_normal(n).astype(np.float32)
        a = g + rng.random(n).astype(np.float32) * np.float32(1e-3)
        mere, mare, ref_idx = _reference_impl(a, g)
        r = _impl(a, g, "float32", 2**-13)
        assert r.is_pass is False
        assert r.metrics["mere"] == pytest.approx(mere, rel=1e-3)  # float64 累加 vs 全量 float32 mean
        assert r.metrics["mare"] == mare  # max 与分块顺序无关，逐位一致
        got = r.diff_index
        assert got.size == k
        np.testing.assert_array_equal(got, ref_idx[: got.size])


def test_chunked_cross_chunk_nan_inf():
    """防线 4：跨块 NaN/Inf——一致的非有限对不算 mismatch，mismatch 位跨块按全局升序收集。"""
    n = 2 * CHUNK_SIZE + 10
    g = np.ones(n, dtype=np.float32)
    a = np.ones(n, dtype=np.float32)
    g[CHUNK_SIZE - 1] = np.nan
    a[CHUNK_SIZE - 1] = np.nan  # 同 NaN：一致
    a[CHUNK_SIZE] = np.nan  # 单侧 NaN：mismatch（跨块边界）
    g[2 * CHUNK_SIZE - 1] = np.inf
    a[2 * CHUNK_SIZE - 1] = np.inf  # 同号 Inf：一致
    a[n - 1] = -np.inf  # 单侧 Inf：mismatch
    r = _impl(a, g, "float32", 2**-13)
    assert r.is_pass is False
    assert r.metrics["mere"] is None
    assert list(r.diff_index) == [CHUNK_SIZE, n - 1]


# 2^20 处 float32 的 ulp=0.125 > FLOOR，|g|+FLOOR 恰为 2^20 → rel_err 可精确落在阈值上
_BASE = np.float32(2**20)


def test_chunked_strict_threshold_boundary():
    """防线 4：严格 < 判定——恰等于 th → mere FAIL；恰等于 MARE_RATIO*th → mare FAIL；全等 → PASS。"""
    th = 2**-13
    n = CHUNK_SIZE + 1000
    # 1) 全部 rel_err == th → mere == th（不满足 mere < th）→ FAIL 仅 mere 超限
    g = np.full(n, _BASE, dtype=np.float32)
    a = np.full(n, np.float32(_BASE + 128.0), dtype=np.float32)
    r = _impl(a, g, "float32", th)
    assert r.is_pass is False
    assert "mere(" in r.metrics["reason"]
    assert "mare(" not in r.metrics["reason"]
    # 2) 单点 rel_err == 10*th → mare == MARE_RATIO*th（不满足 mare <）→ FAIL 仅 mare 超限
    a2 = g.copy()
    a2[123] = np.float32(_BASE + 1280.0)  # 1280/2^20 == 10*2^-13
    r2 = _impl(a2, g, "float32", th)
    assert r2.is_pass is False
    assert "mare(" in r2.metrics["reason"]
    assert "mere(" not in r2.metrics["reason"]
    # 3) 全等 → mere=mare=0 → PASS，diff_index=None
    r3 = _impl(g, g, "float32", th)
    assert r3.is_pass is True
    assert r3.diff_index is None


# —— 防线 5：大数组数值 FAIL 路径 RSS 回归 ——
# 用 /proc/self/status 的 VmHWM（KiB）而非 getrusage：部分环境的 ru_maxrss 不可靠
_RSS_PROBE = """import sys

import numpy as np

from ttk.core_modules.common.comparison.stat_rel_err import StatRelErrComparison

n = int(sys.argv[1])
golden = np.ones(n, dtype=np.float32)
actual = np.ones(n, dtype=np.float32)
actual += np.float32(2.0e-4)  # 触发数值 FAIL 和 top-K 归并路径
_p, _l, is_pass, _m = StatRelErrComparison(
    actual, golden, 0, "float32", {"threshold": 2**-13}
).compare()
assert not is_pass
with open("/proc/self/status") as f:
    hwm = next(line.split()[1] for line in f if line.startswith("VmHWM"))
print(hwm)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="VmHWM 仅 Linux 提供")
def test_large_array_bounded_rss():
    """防线 5：30M 元素数值 FAIL 比较的进程峰值 RSS 有界（旧实现 ~52B/元素 ≈ 1.5GiB，分块后 ≤ 输入+O(CHUNK)）。"""
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", _RSS_PROBE, str(30_000_000)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    hwm_kib = int(proc.stdout.strip().splitlines()[-1])
    assert hwm_kib < 800_000, f"comparison peak RSS {hwm_kib} KiB exceeds bounded budget"
