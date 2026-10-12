# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""分块实现对拍（分块防护改写的功能等价性验收）。

每个判据保留**改写前的旧全量实现**作为基准，在 CHUNK_SIZE±1000 两侧与分块
实现对拍：判定/precision/计数精确相等，diff 前 K 位逐位一致；cross_check 的
mere/rmse 为 float64 分块累加（与全量 fp32 mean 有 ~1e-7 相对漂移，approx 对比）。
"""

import numpy as np
import pytest

from ttk.core_modules.common.comparison.registry import CHUNK_SIZE, get_comparison

K = 100  # ComparisonBase.MAX_DIFF_OUTPUT


def _make(token, output, golden, params, third_party=None):
    cls = get_comparison(token)
    c = cls(np.asarray(output), np.asarray(golden), 0, str(np.asarray(output).dtype), params, third_party=third_party)
    # 对拍直接调 compare_impl（绕开 compare 模板的 log/gc）——容差终值化须显式补
    c._finalize_tolerances()
    return c


# ---------------------------------------------------------------------------
# 参考实现：改写前的旧全量代码（逐字保留语义，仅去 self 化）
# ---------------------------------------------------------------------------
def _ref_mix(actual, golden, rtol, atol, required_matched_ratio, max_abs_error_limit):
    T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
    a = actual.astype(T)
    g = golden.astype(T)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.abs(a - g)
        a_nan, g_nan = np.isnan(a), np.isnan(g)
        a_inf, g_inf = np.isinf(a), np.isinf(g)
        same_nan = a_nan & g_nan
        same_inf = a_inf & g_inf & (np.sign(a) == np.sign(g))
        mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
        finite = np.isfinite(a) & np.isfinite(g)
        elem_ok = same_nan | same_inf | (finite & (err <= atol + rtol * np.abs(g)))
        matched_ratio = float(elem_ok.sum()) / a.size
        if mismatch.any():
            max_abs_error = float("inf")
        elif finite.any():
            max_abs_error = float(err[finite].max())
        else:
            max_abs_error = 0.0
        passed = matched_ratio >= required_matched_ratio and max_abs_error <= max_abs_error_limit
        diff_idx = None
        if not passed:
            err_for_sort = np.where(np.isnan(err), np.inf, err)
            bad_idx = np.where(~elem_ok)[0]
            diff_idx = bad_idx[np.argsort(-err_for_sort[bad_idx], kind="stable")]
    return matched_ratio, max_abs_error, passed, diff_idx


def _ref_close(output, golden, rtol, atol):
    if rtol == 0 and atol == 0:
        ok = (output == golden) | (np.isnan(output) & np.isnan(golden))
    else:
        ok = np.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
    bad = np.where(~ok)[0]
    return (golden.size - bad.size) / golden.size, bad


def _ref_quant(output, golden):
    out = output.astype(np.int32, copy=False)
    gold = golden.astype(np.int32, copy=False)
    bad = np.where(np.abs(out - gold) > 1)[0]
    return (golden.size - bad.size) / golden.size, bad


def _ref_requant(output, golden, view_ulp):
    o, g = output, golden
    if view_ulp:
        o, g = o.view(np.int8), g.view(np.int8)
        diff = np.abs(np.subtract(o, g, dtype=np.int16))  # 修复语义（int8 回绕见独立用例）
    else:
        diff = np.abs(np.subtract(o, g))
    bad = np.where(diff > 1)[0]
    both_nan = np.where(np.isnan(output) & np.isnan(golden))[0]
    bad = np.setdiff1d(bad, both_nan)
    return (golden.size - bad.size) / golden.size, bad


def _ref_binary(output, golden):
    import hashlib

    o, g = output, golden
    if o.dtype.name in ("bfloat16", "int4", "float8_e5m2", "float8_e4m3fn", "hifloat8"):
        ho = hashlib.sha256(o.tobytes()).hexdigest()
        hg = hashlib.sha256(g.tobytes()).hexdigest()
    else:
        ho = hashlib.sha256(o.data).hexdigest()
        hg = hashlib.sha256(g.data).hexdigest()
    if ho == hg:
        return None, 0
    o8, g8 = o.view(np.uint8), g.view(np.uint8)
    itemsize = o.dtype.itemsize
    diff_idx = np.unique(np.floor_divide(np.where(o8 != g8)[0], itemsize))
    diff_idx = np.setdiff1d(diff_idx, np.where(np.isnan(o) & np.isnan(g))[0])
    return diff_idx, diff_idx.size


def _safe_div(num, den, err):
    if np.isnan(num) or np.isnan(den):
        return float("inf")
    if np.isinf(num) and np.isinf(den):
        return 1.0
    if np.isinf(num):
        return float("inf")
    if num == 0 and den == 0:
        return 1.0
    return float(num / max(den, err))


def _top_diff(t, g, mask):
    d = np.abs(t[mask] - g[mask])
    li = np.where(mask)[0]
    mis = d > 0
    if not mis.any():
        return []
    md, ml = d[mis], li[mis]
    k = min(K, md.size)
    top = np.argpartition(-md, k - 1)[:k]
    return ml[top[np.argsort(-md[top])]].tolist()


def _ref_cross(t, g, b, sv, limits):
    T = np.promote_types(np.float32, np.promote_types(t.dtype, np.promote_types(g.dtype, b.dtype)))
    t, g, b = t.astype(T), g.astype(T), b.astype(T)
    special = np.isnan(t) | np.isinf(t) | np.isnan(g) | np.isinf(g) | np.isnan(b) | np.isinf(b)
    finite = ~special
    large = finite & (np.abs(g) >= sv["small_value"])
    small = finite & (np.abs(g) < sv["small_value"])

    special_ok = True
    if special.any():
        t_nan, g_nan, b_nan = np.isnan(t), np.isnan(g), np.isnan(b)
        t_pinf, g_pinf, b_pinf = np.isposinf(t), np.isposinf(g), np.isposinf(b)
        t_ninf, g_ninf, b_ninf = np.isneginf(t), np.isneginf(g), np.isneginf(b)
        t_fin, g_fin, b_fin = np.isfinite(t), np.isfinite(g), np.isfinite(b)
        tb = (t_nan & b_nan) | (t_pinf & b_pinf) | (t_ninf & b_ninf) | (t_fin & b_fin & (t == b))
        tg = (t_nan & g_nan) | (t_pinf & g_pinf) | (t_ninf & g_ninf) | (t_fin & g_fin & (t == g))
        special_ok = not (special & ~(tb | tg)).any()

    ratio_ok, mare, mere, rmse = True, None, None, None
    if large.any():
        rel_n = np.abs(t[large] - g[large]) / (np.abs(g[large]) + 1e-7)
        rel_p = np.abs(b[large] - g[large]) / (np.abs(g[large]) + 1e-7)
        if not np.all(np.isinf(rel_p) | np.isnan(rel_p)):
            mare = _safe_div(rel_n.max(), rel_p.max(), sv["small_value"])
            mere = _safe_div(rel_n.mean(), rel_p.mean(), sv["small_value"])
            rn = np.sqrt(np.mean((t[large] - g[large]) ** 2))
            rp = np.sqrt(np.mean((b[large] - g[large]) ** 2))
            if np.isinf(rn) or np.isinf(rp):
                rn = np.sqrt(np.mean(((t[large] - g[large]).astype(np.float64)) ** 2))
                rp = np.sqrt(np.mean(((b[large] - g[large]).astype(np.float64)) ** 2))
            rmse = _safe_div(rn, rp, sv["small_value"])
            ratio_ok = not (mare > limits["mare"] or mere > limits["mere"] or rmse > limits["rmse"])
    err_t = int(np.sum(small & (np.abs(t - g) > sv["small_value_atol"])))
    err_3 = int(np.sum(small & (np.abs(b - g) > sv["small_value_atol"])))
    small_ok = err_t / max(err_3, 1) <= 2.0
    return {
        "passed": ratio_ok and special_ok and small_ok,
        "mare": mare,
        "mere": mere,
        "rmse": rmse,
        "special_ok": special_ok,
        "err_target": err_t,
        "err_third": err_3,
        "diff": _top_diff(t, g, large) + _top_diff(t, g, small),
    }


# ---------------------------------------------------------------------------
# mix_tolerance
# ---------------------------------------------------------------------------
_MIX = {"rtol": 2**-10, "atol": 2**-16, "required_matched_ratio": 0.99, "max_abs_error_limit": 1e-2}
_MIX_SIZES = (CHUNK_SIZE - 1000, CHUNK_SIZE + 1000)


@pytest.mark.parametrize("n", _MIX_SIZES)
def test_mix_parity_partial_fail(n):
    rng = np.random.default_rng(11)
    g = rng.standard_normal(n).astype(np.float32)
    a = g + rng.random(n).astype(np.float32) * np.float32(0.05)  # 部分超差
    ref_ratio, ref_max, ref_pass, ref_idx = _ref_mix(a, g, **_MIX)
    r = _make("mix_tolerance", a, g, _MIX).compare_impl()
    assert r.is_pass == ref_pass
    assert r.metrics["matched_ratio"] == ref_ratio  # 整数计数，精确
    assert r.metrics["max_abs_error"] == ref_max  # running max，精确
    assert r.diff_index.size == min(K, ref_idx.size)
    np.testing.assert_array_equal(r.diff_index, ref_idx[: r.diff_index.size])  # worst-first 逐位一致


@pytest.mark.parametrize("n", _MIX_SIZES)
def test_mix_parity_all_fail_and_special(n):
    g = np.ones(n, dtype=np.float32)
    a = np.full(n, 1.5, dtype=np.float32)  # 全超差 → top-K 归并路径
    _, _, ref_pass, ref_idx = _ref_mix(a, g, **_MIX)
    r = _make("mix_tolerance", a, g, _MIX).compare_impl()
    assert ref_pass is False
    assert r.is_pass is False
    assert r.diff_index.size == K
    np.testing.assert_array_equal(r.diff_index, ref_idx[:K])

    # 跨块 NaN/Inf：一致非有限对通过、单侧 mismatch 必挂（max_abs_error=inf）
    p0, p1 = min(CHUNK_SIZE - 1, n - 2), min(CHUNK_SIZE, n - 1)  # 块尾/块首（n<CHUNK 时退化）
    a2 = g.copy()
    a2[p0] = np.nan
    g2 = g.copy()
    g2[p0] = np.nan  # 同 NaN：一致
    a2[p1] = np.nan  # 单侧 NaN：mismatch（块边界两侧）
    r2 = _make("mix_tolerance", a2, g2, _MIX).compare_impl()
    assert r2.is_pass is False
    assert r2.metrics["max_abs_error"] is None  # inf → None
    assert r2.metrics["reason"].startswith("NaN/Inf mismatch")


# ---------------------------------------------------------------------------
# is_close / quant / requant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["close", "quant"])
@pytest.mark.parametrize("n", [CHUNK_SIZE - 1000, CHUNK_SIZE + 1000])
def test_close_quant_parity(method, n):  # noqa: S105 method 为比对标准名，非口令
    rng = np.random.default_rng(22)
    if method == "close":
        g = rng.standard_normal(n).astype(np.float32)
        a = g + rng.random(n).astype(np.float32) * np.float32(0.01)
        params = {"legacy": {"rtol": None, "ptol": None, "atol": None}}
        ref_prec, ref_bad = _ref_close(a, g, 0.0001, 1e-8)  # float32 默认 rtol/atol，与实现一致
    else:
        g = rng.integers(0, 4, n, dtype=np.int8)
        a = g.copy()
        a[::97] = (g.astype(np.int16)[::97] + 7).astype(np.int8)  # 稀疏超差
        params = {}
        ref_prec, ref_bad = _ref_quant(a, g)
    r = _make(method, a, g, params).compare_impl()
    if method == "close":
        assert r.precision == ref_prec  # 计数精确
    else:
        assert r.precision == ref_prec
        assert r.metrics["exceed_count"] == ref_bad.size
    assert r.diff_index.size == min(K, ref_bad.size)
    np.testing.assert_array_equal(r.diff_index, ref_bad[: r.diff_index.size])  # 索引升序一致


def test_close_exact_zero_tolerance_parity():
    """rtol=0/atol=0 分支 + 双 NaN 视为相等。"""
    n = 3 * CHUNK_SIZE
    g = np.ones(n, dtype=np.float32)
    a = g.copy()
    a[min(CHUNK_SIZE, n - 1)] = 2.0  # 失败位跨块
    a[2 * CHUNK_SIZE - 1] = np.nan
    g[2 * CHUNK_SIZE - 1] = np.nan  # 双 NaN → 相等
    ref_prec, ref_bad = _ref_close(a, g, 0, 0)
    r = _make("close", a, g, {"legacy": {"rtol": [0], "ptol": None, "atol": [0]}}).compare_impl()
    assert list(r.diff_index) == [CHUNK_SIZE]
    assert r.precision == ref_prec == (n - 1) / n


def test_requant_parity_and_int8_wraparound_fix():
    """requant 对拍（含双 NaN 排除）+ int8 回绕修复：255 ULP 偏差旧实现误判 PASS。"""
    from ttk.utilities.dtypes import resolve_numpy_dtype

    f8 = resolve_numpy_dtype("float8_e5m2")
    n = CHUNK_SIZE + 1000
    raw_g = np.zeros(n, dtype=np.uint8)
    raw_a = np.zeros(n, dtype=np.uint8)
    raw_g[::53] = 0x3C  # 1.0 (e5m2)
    raw_a[::53] = 0x3D  # 1.25 → ULP 差 1 → OK
    raw_g[7] = 0x30  # 0.5
    raw_a[7] = 0x40  # 2.0 → ULP 差 16 → 失败
    raw_g[11] = 0x7F  # NaN
    raw_a[11] = 0x7F  # 同 NaN → 排除
    a = raw_a.view(f8)
    g = raw_g.view(f8)
    ref_prec, ref_bad = _ref_requant(a, g, view_ulp=True)
    r = _make("requant", a, g, {}).compare_impl()
    assert r.precision == ref_prec
    assert 7 in r.diff_index
    assert 11 not in r.diff_index

    # 回绕修复：0x7F(NaN) vs 0x80(-0.0)——int8 视图差 255，旧实现回绕成 -1 → |1|≤1 误判 PASS
    a2 = np.array([0x7F], dtype=np.uint8).view(f8)
    g2 = np.array([0x80], dtype=np.uint8).view(f8)
    r2 = _make("requant", a2, g2, {}).compare_impl()
    assert r2.is_pass is False  # 修复后 255 ULP → FAIL


# ---------------------------------------------------------------------------
# binary_equal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [CHUNK_SIZE - 1000, CHUNK_SIZE + 1000])
def test_binary_parity_sparse_and_dense(n):
    rng = np.random.default_rng(33)
    g = rng.standard_normal(n).astype(np.float32)
    a = g.copy()
    a[::101] = g[::101] + 1.0  # 稀疏差异（含块边界两侧）
    pos = min(CHUNK_SIZE, n - 1)
    a[pos] = g[pos] + 2.0
    ref_idx, ref_cnt = _ref_binary(a, g)
    r = _make("bin", a, g, {}).compare_impl()
    assert r.metrics["pass"] is False
    got = r.diff_index
    assert got.size == min(K, ref_cnt)
    np.testing.assert_array_equal(got, ref_idx[: got.size])  # 索引升序 + 计数精确

    a2 = rng.standard_normal(n).astype(np.float32)  # 全差异（多字节/元素命中）
    ref_idx2, ref_cnt2 = _ref_binary(a2, g)
    r2 = _make("bin", a2, g, {}).compare_impl()
    assert r2.diff_index.size == min(K, ref_cnt2)
    np.testing.assert_array_equal(r2.diff_index, ref_idx2[: r2.diff_index.size])


def test_binary_parity_nan_payload():
    """双 NaN 不同 payload：字节差异存在但元素级被排除（对拍旧 setdiff1d 语义）。"""
    n = 4 * CHUNK_SIZE
    g = np.ones(n, dtype=np.float32)
    a = g.copy()
    g[CHUNK_SIZE] = np.nan
    a[CHUNK_SIZE] = np.nan
    a.view(np.uint32)[CHUNK_SIZE] ^= 1  # 双 NaN 但 payload 不同 → 字节差、元素级排除
    a[n - 1] = 5.0  # 真差异
    ref_idx, ref_cnt = _ref_binary(a, g)
    r = _make("bin", a, g, {}).compare_impl()
    assert ref_cnt == 1
    assert list(r.diff_index) == [n - 1]


# ---------------------------------------------------------------------------
# cross_check
# ---------------------------------------------------------------------------
_CC_SV = {"small_value": 2**-14, "small_value_atol": 2**-30}
_CC_LIMITS = {"mare": 5.0, "mere": 1.5, "rmse": 1.5}
_CC_PARAMS = {"level": "L1", "mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5, **_CC_SV}


@pytest.mark.parametrize("n", [CHUNK_SIZE - 1000, CHUNK_SIZE + 1000])
def test_cross_parity_mixed(n):
    rng = np.random.default_rng(44)
    g = rng.standard_normal(n).astype(np.float32)
    t = g + rng.random(n).astype(np.float32) * np.float32(0.01)
    b = g + rng.random(n).astype(np.float32) * np.float32(0.005)
    p0 = min(CHUNK_SIZE - 1, n - 2)
    t[p0] = np.nan
    g[p0] = np.nan  # 同 NaN（块边界）
    b[p0] = 1.0
    t[n - 1] = np.inf
    g[n - 1] = np.inf  # 同 Inf
    b[n - 1] = 2.0
    ref = _ref_cross(t, g, b, _CC_SV, _CC_LIMITS)
    r = _make("cross_check", t, g, _CC_PARAMS, third_party=b).compare_impl()
    assert r.is_pass == ref["passed"]
    m = r.metrics["result"]
    assert m["small_err_cnt_target"] == ref["err_target"]
    assert m["small_err_cnt_third"] == ref["err_third"]
    for key in ("mare", "mere", "rmse"):
        rv, mv = ref[key], m[key]
        if rv is None:
            assert mv is None
        else:
            assert mv == pytest.approx(rv, rel=1e-5)  # mere/rmse f64 累加漂移；mare 精确
    got = list(r.diff_index) if r.diff_index is not None else []
    # |t-g| 键序列必须逐位一致；参考实现 argpartition 的并列序任意（fp32 舍入
    # 会产生并列组），分块实现按并列组内索引升序——两种都是合法 top-K 序
    d_all = np.abs(t.astype(np.float32) - g.astype(np.float32))
    assert [d_all[i] for i in got] == [d_all[i] for i in ref["diff"]]
    for i in range(1, len(got)):
        if d_all[got[i]] == d_all[got[i - 1]]:
            assert got[i] > got[i - 1]  # 并列组内索引升序（确定性）


def test_cross_parity_party_all_invalid():
    """third_party 误差在 large 位全溢出（有限大值反号相减）→ ratio 不参与判定。

    注：不能用 b=inf 构造——b 非有限会让位置进 special 路径（Path A 判 FAIL）；
    party_all_invalid 要求 b 有限但 |b-g| 溢出 fp32。"""
    n = 2 * CHUNK_SIZE + 7
    g = np.full(n, 3e38, dtype=np.float32)
    t = np.full(n, np.float32(3e38 * 1.000001), dtype=np.float32)  # NPU 微小偏差（有限）
    b = np.full(n, -3e38, dtype=np.float32)  # |b-g|=6e38 溢出 → rel_party 全 inf
    ref = _ref_cross(t, g, b, _CC_SV, _CC_LIMITS)
    r = _make("cross_check", t, g, _CC_PARAMS, third_party=b).compare_impl()
    assert r.is_pass == ref["passed"] is True
    assert r.metrics["result"]["mare"] is None  # ratio 未参与


# ---------------------------------------------------------------------------
# 内存回归：默认判据 mix_tolerance 的 FAIL 路径 RSS 有界
# ---------------------------------------------------------------------------
_RSS_PROBE = """import sys

import numpy as np

from ttk.core_modules.common.comparison.registry import get_comparison

n = int(sys.argv[1])
golden = np.ones(n, dtype=np.float32)
actual = np.full(n, 1.1, dtype=np.float32)  # 全量 mismatch → top-K 归并路径
params = {"rtol": 2 ** -10, "atol": 2 ** -16, "required_matched_ratio": 0.99, "max_abs_error_limit": 1e-2}
_p, _l, is_pass, _m = get_comparison("mix_tolerance")(actual, golden, 0, "float32", params).compare()
assert not is_pass
with open("/proc/self/status") as f:
    hwm = next(line.split()[1] for line in f if line.startswith("VmHWM"))
print(hwm)
"""


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="VmHWM 仅 Linux 提供")
def test_mix_large_array_bounded_rss():
    """30M 元素全量 FAIL：分块后峰值 RSS ≤ 输入 + O(CHUNK)（旧实现 ~49B/元素 ≈ 1.4GiB）。"""
    import subprocess
    import sys
    from pathlib import Path

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


def test_requant_int8_extremes_no_wraparound():
    """requant 整型提升：int8 极值差 255（127 vs -128）不得回绕成 -1 误判 PASS。"""
    out = np.array([127, 0, 5], dtype=np.int8)
    gold = np.array([-128, 0, 5], dtype=np.int8)
    r = _make("requant", out, gold, {}).compare_impl()
    assert r.is_pass is False
    assert list(r.diff_index) == [0]
