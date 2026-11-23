#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for comparison/ulp.py：单调键 / 分块等价 / 折叠口径 / 特殊值 / XPU golden 标注 / 自定义插件兼容 / cross_check 集成。"""

import numpy as np
import pytest

import ttk.core_modules.common.comparison.ulp as ulp_mod
from ttk.core_modules.common.comparison.cross_check import CrossCheckComparison
from ttk.core_modules.common.comparison.ulp import (
    COMPARE_DETAIL_HEADER,
    iter_detail_rows,
    pop_ulp_payloads,
    scan_ulp_triplet,
    ulp_monotone_key,
)


def fp16_from_hex(*values):
    """十六进制位模式 → float16 数组（构造已知 ULP 距离的用例）。"""
    return np.array(values, dtype=np.uint16).view(np.float16)


def fp32_from_hex(*values):
    return np.array(values, dtype=np.uint32).view(np.float32)


# ---------------------------------------------------------------------------
# 单调键（独立数值表：手算期望值，交叉验证实现）
# ---------------------------------------------------------------------------
def test_monotone_key_basics():
    """fp16 关键位点的键值：±Inf/±0（相邻 1）/subnormal 连续/NaN 落极值外。"""
    bits = np.array([0xFC00, 0xFBFF, 0x8000, 0x0000, 0x0001, 0x7BFF, 0x7C00], dtype=np.uint16)
    #        -inf     最大负有限  -0.0   +0.0  最小正次正规  最大正有限  +inf
    expect = [0x03FF, 0x0400, 0x7FFF, 0x8000, 0x8001, 0xFBFF, 0xFC00]
    key = ulp_monotone_key(bits, np.uint16(0x8000), np.uint16(0xFFFF))
    assert key.tolist() == expect


def test_monotone_key_signed_zero_one_ulp():
    """+0.0 与 -0.0 相距 1 ULP（位级语义）。"""
    bits = np.array([0x0000, 0x8000], dtype=np.uint16)
    key = ulp_monotone_key(bits, np.uint16(0x8000), np.uint16(0xFFFF))
    assert abs(int(key[0]) - int(key[1])) == 1


# ---------------------------------------------------------------------------
# scan_ulp_triplet：已知距离 / 特殊值 / XPU golden 标注
# ---------------------------------------------------------------------------
def test_scan_known_distances():
    """fp16 三方：逐对 max_ulp（手算：npu=[1,1,1,1], cpu=[1,1.001,1,1], xpu=[1,1.001,1.002,1.001]）。"""
    t = fp16_from_hex(0x3C00, 0x3C00, 0x3C00, 0x3C00)  # npu：1.0 ×4
    g = fp16_from_hex(0x3C00, 0x3C01, 0x3C03, 0x3C00)  # cpu：1.0 / +1 / +3 / 1.0
    b = fp16_from_hex(0x3C00, 0x3C01, 0x3C02, 0x3C02)  # xpu：1.0 / +1 / +2 / +2
    payload = scan_ulp_triplet(t, g, b, "float16")

    assert payload["dtype"] == "float16"
    assert payload["golden_is_third_party"] is False
    assert tuple(payload["pairs"]) == ("npu_cpu", "npu_xpu", "xpu_cpu")
    assert payload["pairs"]["npu_cpu"]["max_ulp"] == 3
    assert payload["pairs"]["npu_xpu"]["max_ulp"] == 2
    assert payload["pairs"]["xpu_cpu"]["max_ulp"] == 2


def test_scan_special_values_counted_not_maxed():
    """NaN/±Inf 一律不参与 max；全 NaN → max_ulp=None。"""
    t = fp16_from_hex(0x3C00, 0x7C00, 0xFC00, 0x7E00)  # 1.0 / +inf / -inf / NaN
    g = fp16_from_hex(0x3C01, 0x3C00, 0x3C00, 0x3C00)
    b = fp16_from_hex(0x3C00, 0x3C00, 0x3C00, 0x3C00)
    payload = scan_ulp_triplet(t, g, b, "float16")

    assert payload["pairs"]["npu_cpu"]["max_ulp"] == 1  # 有限对仅 idx0

    # 全 NaN：三对均无 finite 对 → None（渲染 N/A）
    nan_arr = fp16_from_hex(0x7E00, 0x7E01)
    empty = scan_ulp_triplet(nan_arr, nan_arr.copy(), nan_arr.copy(), "float16")
    for stat in empty["pairs"].values():
        assert stat["max_ulp"] is None


def test_scan_signed_zero():
    """±0 相差 1 ULP。"""
    t = fp16_from_hex(0x0000)  # +0.0
    g = fp16_from_hex(0x8000)  # -0.0
    payload = scan_ulp_triplet(t, g, t.copy(), "float16")
    assert payload["pairs"]["npu_cpu"]["max_ulp"] == 1
    assert payload["pairs"]["npu_xpu"]["max_ulp"] == 0  # t 与 b 同为 +0.0
    assert payload["pairs"]["xpu_cpu"]["max_ulp"] == 1


def test_scan_subnormal_staircase():
    """次正规区逐台阶精确：0x0000→0x0005 距离 5。"""
    t = fp16_from_hex(0x0000)
    g = fp16_from_hex(0x0005)
    payload = scan_ulp_triplet(t, g, t.copy(), "float16")
    assert payload["pairs"]["npu_cpu"]["max_ulp"] == 5


def test_xpu_golden_flag():
    """golden_is_third_party=True 时载荷如实标注（检视意见 P2）。"""
    t = fp16_from_hex(0x3C00, 0x3C02)
    g = fp16_from_hex(0x3C00, 0x3C03)
    b = g.copy()  # XPU golden 模式：third_party 与 golden 同一份数据
    payload = scan_ulp_triplet(t, g, b, "float16", golden_is_third_party=True)
    assert payload["golden_is_third_party"] is True
    # 实际数值仍计算（渲染层决定是否展示）
    assert payload["pairs"]["xpu_cpu"]["max_ulp"] == 0  # g≡b 自比恒 0
    assert payload["pairs"]["npu_cpu"]["max_ulp"] == payload["pairs"]["npu_xpu"]["max_ulp"]


def test_promoted_golden_fold():
    """Promote golden（fp32）RNE 折叠到 fp16 网格：同格 0 ULP、跨格计差。"""
    t = fp16_from_hex(0x3C00, 0x3C00)  # npu：1.0, 1.0
    g = np.array([1.0, 1.0009765625], dtype=np.float32)  # cpu：恰好落在 fp16 格点
    b = fp16_from_hex(0x3C00, 0x3C00)
    payload = scan_ulp_triplet(t, g, b, "float16")
    assert payload["pairs"]["npu_cpu"]["max_ulp"] == 1
    # 折叠后同格（fp32 值恰为 fp16 格点）→ 0 ULP
    t2 = fp16_from_hex(0x3C01)
    g2 = np.array([1.0009765625], dtype=np.float32)
    assert scan_ulp_triplet(t2, g2, t2.copy(), "float16")["pairs"]["npu_cpu"]["max_ulp"] == 0


def test_xpu_dtype_fold():
    """XPU 输出 dtype 异于 ruler（fp32→fp16）时同样折叠计算。"""
    t = fp16_from_hex(0x3C00, 0x3C02)
    g = fp16_from_hex(0x3C00, 0x3C02)
    b = np.array([1.0, 1.001953125], dtype=np.float32)  # 折叠后 0x3C00 / 0x3C02
    payload = scan_ulp_triplet(t, g, b, "float16")
    assert payload["pairs"]["npu_xpu"]["max_ulp"] == 0


def test_bfloat16_ruler():
    """bf16 ruler（依赖 ml_dtypes）：1.0(0x3F80) vs 1.0078125(0x3F82) 差 2。"""
    pytest.importorskip("ml_dtypes")
    import ml_dtypes

    def bf16_from_hex(*values):
        return np.array(values, dtype=np.uint16).view(ml_dtypes.bfloat16)

    t = bf16_from_hex(0x3F80)
    g = bf16_from_hex(0x3F82)
    payload = scan_ulp_triplet(t, g, t.copy(), "bfloat16")
    assert payload["pairs"]["npu_cpu"]["max_ulp"] == 2


def test_unsupported_dtype_placeholder():
    """非 fp16/bf16/fp32：pairs 空的显式占位（不静默）。"""
    t = np.array([1, 2], dtype=np.int32)
    payload = scan_ulp_triplet(t, t.copy(), t.copy(), "int32")
    assert payload["pairs"] == {}
    assert payload["dtype"] == "int32"
    (row,) = list(iter_detail_rows(payload, {"standard": "cross_check"}, "c", "dyn", 0))
    assert (row[5], row[6], row[7]) == ("N/A", "N/A", "N/A")  # 三对列均 N/A
    assert row[4] is False  # golden_is_third_party
    # 别名归一：fp16 → float16 正常计算
    t16 = fp16_from_hex(0x3C00)
    assert scan_ulp_triplet(t16, t16.copy(), t16.copy(), "fp16")["dtype"] == "float16"


# ---------------------------------------------------------------------------
# 分块等价性（对照全量朴素实现；模式照抄 test_chunked_parity.py）
# ---------------------------------------------------------------------------
def _naive_pair_max(a, b, ruler_np_dtype):
    """全量朴素实现：折叠 → 位模式 → 键 → 逐元素距离 → max。"""

    def keys_of(arr):
        a2 = arr.astype(ruler_np_dtype) if arr.dtype != ruler_np_dtype else np.ascontiguousarray(arr)
        if ruler_np_dtype.itemsize == 2:
            bits = a2.view(np.uint16).astype(np.int64)
            key = np.where(bits >= 0x8000, 0xFFFF - bits, bits + 0x8000)
        else:
            bits = a2.view(np.uint32).astype(np.int64)
            key = np.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
        return key

    ka, kb = keys_of(a), keys_of(b)
    finite = np.isfinite(a) & np.isfinite(b)
    pos = np.flatnonzero(finite)
    dist = np.abs(ka[pos] - kb[pos]) if pos.size else np.array([], dtype=np.int64)
    return int(dist.max()) if dist.size else None


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_chunk_parity(dtype, monkeypatch):
    """分块（CHUNK_SIZE=3）结果 ≡ 全量朴素实现（含 NaN/Inf/±0 注入）。"""
    monkeypatch.setattr(ulp_mod, "CHUNK_SIZE", 3)
    rng = np.random.default_rng(2026)
    n = 41
    if dtype == "float16":
        t = rng.standard_normal(n).astype(np.float16)
        g = rng.standard_normal(n).astype(np.float16)
        b = rng.standard_normal(n).astype(np.float16)
    else:
        t = rng.standard_normal(n).astype(np.float32)
        g = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
    for arr in (t, g, b):
        arr[3] = np.nan
        arr[7] = np.inf
        arr[11] = -np.inf
        arr[13] = 0.0  # 与他方 -0.0 组成 ±0 对
    g[13] = -0.0
    b[17] = -0.0

    np_dtype = np.dtype(dtype)
    payload = scan_ulp_triplet(t, g, b, dtype)
    naive = {
        "npu_cpu": _naive_pair_max(t, g, np_dtype),
        "npu_xpu": _naive_pair_max(t, b, np_dtype),
        "xpu_cpu": _naive_pair_max(b, g, np_dtype),
    }
    for name, expect in naive.items():
        assert payload["pairs"][name]["max_ulp"] == expect, name


# ---------------------------------------------------------------------------
# 载荷契约：字面量往返 / XPU golden 渲染 / 自定义插件兼容 / cross_check 集成
# ---------------------------------------------------------------------------
def test_literal_roundtrip():
    """payload 全 Python 字面量：eval(repr) 逐层相等（CSV 列契约）。"""
    t = fp16_from_hex(0x3C00, 0x3C02, 0x7C00)
    g = fp16_from_hex(0x3C00, 0x3C05, 0x3C00)
    payload = scan_ulp_triplet(t, g, t.copy(), "float16")
    assert eval(repr(payload)) == payload  # noqa: S307


def test_pop_ignores_custom_plugin_ulp():
    """自定义插件的 metrics={"ulp": 7} 不被摘出（检视意见 P2：双 CSV 均丢失）。"""
    custom = {0: {"standard": "custom", "pass": False, "reason": "x", "ulp": 7}}
    assert pop_ulp_payloads(custom) == {}
    assert custom[0]["ulp"] == 7  # 主表保留

    mix = {0: {"standard": "mix_tolerance", "pass": True, "ulp": 7}}
    assert pop_ulp_payloads(mix) == {}
    assert mix[0]["ulp"] == 7

    # cross_check 但 ulp 非本模块载荷结构（标量）→ 同样不动
    cc_scalar = {0: {"standard": "cross_check", "pass": True, "ulp": 7}}
    assert pop_ulp_payloads(cc_scalar) == {}
    assert cc_scalar[0]["ulp"] == 7


def _cross_check_params():
    return {
        "level": "L1",
        "mare_ratio": 5.0,
        "mere_ratio": 1.5,
        "rmse_ratio": 1.5,
        "small_value": 2**-11,
        "small_value_atol": 2**-16,
        "legacy": {"rtol": None, "ptol": None, "atol": 1e-8},
    }


def test_cross_check_integration_attaches_ulp():
    """cross_check.compare 出口：metrics['ulp'] 结构齐全且字面量安全。"""
    g = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    c = CrossCheckComparison(g.copy(), g.copy(), 0, "float16", _cross_check_params(), third_party=g.copy())
    _, _, is_pass, metrics = c.compare()
    assert is_pass is True
    payload = metrics["ulp"]
    assert payload["dtype"] == "float16"
    assert payload["golden_is_third_party"] is False
    assert tuple(payload["pairs"]) == ("npu_cpu", "npu_xpu", "xpu_cpu")
    for stat in payload["pairs"].values():
        assert stat["max_ulp"] == 0


def test_cross_check_xpu_golden_detection():
    """golden≡third_party（同底层数据）时 cross_check 自动标注（检视意见 P2）。"""
    t = np.array([1.0, 2.0], dtype=np.float16)
    golden_source = np.array([1.0, 2.1], dtype=np.float16)  # 同一份 ndarray 作 golden 与 third_party
    c = CrossCheckComparison(t, golden_source, 0, "float16", _cross_check_params(), third_party=golden_source)
    _, _, _, metrics = c.compare()
    assert metrics["ulp"]["golden_is_third_party"] is True  # np.shares_memory 探测

    # 渲染层：npu_cpu/xpu_cpu 两列 N/A（无 CPU 参与与自比，检视意见 P2）
    (row,) = list(iter_detail_rows(metrics["ulp"], metrics, "c", "", 0))
    assert row[4] is True
    assert (row[5], row[6], row[7]) == ("N/A", row[6], "N/A")  # npu_xpu 保留真实值


def test_cross_check_missing_third_party_no_ulp():
    """third_party 缺失 → GOLDEN_FAILURE：无 ulp 键（fail-closed，明细文件无行）。"""
    g = np.array([1.0, 2.0], dtype=np.float32)
    params = _cross_check_params()
    params.update({"small_value": 2**-14, "small_value_atol": 2**-30})
    c = CrossCheckComparison(g.copy(), g.copy(), 0, "float32", params)
    _, _, is_pass, metrics = c.compare()
    assert not is_pass
    assert "ulp" not in metrics


def test_pop_ulp_payloads_on_cross_check_metrics():
    """metrics 经 pop_ulp_payloads 摘出：互斥（leaf 无 ulp）+ 载荷无损。"""
    g = np.array([1.0, 2.0], dtype=np.float16)
    c = CrossCheckComparison(g.copy(), g.copy(), 0, "float16", _cross_check_params(), third_party=g.copy())
    _, _, _, metrics = c.compare()
    expected = metrics["ulp"]  # 摘出前先捕获（pop 后原位不可再读）
    nested = {"dyn": {0: metrics}}
    payloads = pop_ulp_payloads(nested)
    assert set(payloads) == {("dyn", 0)}
    payload, leaf = payloads[("dyn", 0)]
    assert payload == expected
    assert leaf is metrics  # 宽表参数列的数据源
    assert "standard" in leaf
    # 摘出后：leaf 与嵌套均无 ulp 残留
    assert "ulp" not in nested["dyn"][0]
    assert not pop_ulp_payloads(nested)


def test_detail_header_width():
    """宽表表头 23 列且唯一（消费方按列名解析的前提）。"""
    assert len(COMPARE_DETAIL_HEADER) == 23
    assert len(set(COMPARE_DETAIL_HEADER)) == 23
    rows = list(
        iter_detail_rows(
            scan_ulp_triplet(fp16_from_hex(0x3C00), fp16_from_hex(0x3C00), fp16_from_hex(0x3C00), "float16"),
            {"standard": "cross_check"},
            "c",
            "dyn",
            0,
        )
    )
    assert len(rows) == 1  # 宽表：一输出一行
    for row in rows:
        assert len(row) == len(COMPARE_DETAIL_HEADER)
