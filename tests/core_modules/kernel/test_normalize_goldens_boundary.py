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
_normalize_goldens 溢出语义测试。

各 dtype 的溢出阈值/tie 归属/带外替换值须与其顶 binade 结构一致，全部语义经
昇腾硬件实测标定：
  fp16/bf16/fp32   Ascend910B3（numpy/torch CPU、NPU .to、npu_dtype_cast 五路一致）
  e5m2/e4m3fn/hifloat8/e8m0   Ascend950PR（含 MX 量化算子通路）
"""

import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.kernel.golden import _normalize_goldens
from ttk.utilities.dtypes import get_dtype_overflow_boundary, resolve_numpy_dtypes


def _missing(pkg):
    return importlib.util.find_spec(pkg) is None


# 注册 fp8/hifloat8 自定义 numpy dtype（golden.py 的 numpy.dtype() 依赖注册）；
# 缺包的 dtype 不注册，对应测试类 skipif 跳过，避免 pytest 收集阶段崩溃
_TO_REGISTER = []
if not _missing("ml_dtypes"):
    _TO_REGISTER += ["float8_e5m2", "float8_e4m3fn"]
if not _missing("en_dtypes"):
    _TO_REGISTER += ["float8_e8m0", "hifloat8"]
if _TO_REGISTER:
    resolve_numpy_dtypes(_TO_REGISTER)

_normalize = _normalize_goldens


def _run(values, dtype, overflow_mode=1):
    golden = np.array(values, dtype=np.float64 if dtype == "float32" else np.float32)
    sw = MagicMock()
    sw.overflow_mode = overflow_mode
    with patch("ttk.core_modules.kernel.golden.get_global_storage", return_value=sw):
        _normalize([golden], [dtype])
    return golden


def _expect(dtype, values):
    """期望值按与 golden 相同的 dtype 构造，避免 fp32/fp64 字面量精度差。"""
    return np.array(values, dtype=np.float64 if dtype == "float32" else np.float32)


def _assert_same(actual, expect):
    for a, e in zip(actual, expect):
        assert (np.isnan(a) and np.isnan(e)) or a == e, f"{a!r} != {e!r}"


class TestOverflowBoundaryFunction:
    """get_dtype_overflow_boundary：RNE 边界 + 各舍入模式。"""

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            ("float16", 65520.0),  # (65504 + 65536) / 2
            ("bfloat16", (float.fromhex("0x1.FEp127") + 2.0**128) / 2),
            ("float32", (3.4028234663852886e38 + 2.0**128) / 2),
            ("float8_e5m2", 61440.0),  # (57344 + 65536) / 2
            ("float8_e4m3fn", 464.0),  # 448 + 32/2（顶格 NaN，frexp 不适用）
            ("hifloat8", 40960.0),  # 32768 + 16384/2（1-bit 顶 binade）
        ],
    )
    def test_rne_boundary(self, dtype, expected):
        lo, hi = get_dtype_overflow_boundary(dtype)
        assert hi == pytest.approx(expected)
        assert lo == pytest.approx(-expected)

    def test_rna_equals_rne(self):
        # rne/rna 溢出边界相同：tie 的偶候选与远离零候选同为 2^(emax+1) → inf
        for dtype in ("float16", "bfloat16", "float32", "float8_e5m2", "float8_e4m3fn", "hifloat8"):
            assert get_dtype_overflow_boundary(dtype, "rna") == get_dtype_overflow_boundary(dtype, "rne")

    def test_directed_modes(self):
        hi_max = float.fromhex("0x1.FEp127")
        assert get_dtype_overflow_boundary("bfloat16", "rtp") == (float("-inf"), np.nextafter(hi_max, np.inf))
        assert get_dtype_overflow_boundary("bfloat16", "rtn") == (np.nextafter(-hi_max, -np.inf), float("inf"))
        assert get_dtype_overflow_boundary("bfloat16", "rtz") == (float("-inf"), float("inf"))

    def test_case_insensitive_and_invalid(self):
        assert get_dtype_overflow_boundary("float16", "RNE") == get_dtype_overflow_boundary("float16")
        with pytest.raises(ValueError, match="unsupported rounding mode"):
            get_dtype_overflow_boundary("float16", "rnz")


@pytest.mark.skipif(_missing("ml_dtypes"), reason="ml_dtypes is not installed")
class TestRneOverflowDtypes:
    """IEEE 满顶 binade 格式：边界带保真，tie 归 inf（Ascend910B3/950 实测）。"""

    def test_float16_band_and_tie(self):
        out = _run([65504, 65510, 65519.9, 65520, 70000, -65519.9, -65520], "float16")
        _assert_same(out, _expect("float16", [65504, 65510, 65519.9, np.inf, np.inf, -65519.9, -np.inf]))

    def test_bfloat16_band_and_tie(self):
        boundary = (float.fromhex("0x1.FEp127") + 2.0**128) / 2
        out = _run([3.3929e38, boundary, 3.4e38, -boundary], "bfloat16")
        _assert_same(out, _expect("bfloat16", [3.3929e38, np.inf, np.inf, -np.inf]))

    def test_float32_band_and_tie(self):
        boundary = (3.4028234663852886e38 + 2.0**128) / 2
        out = _run([3.40282352e38, boundary, 3.41e38], "float32")
        _assert_same(out, _expect("float32", [3.40282352e38, np.inf, np.inf]))

    def test_e5m2_band_and_tie(self):
        out = _run([57344, 58000, 61439.9, 61440, 65000, -61440], "float8_e5m2")
        _assert_same(out, _expect("float8_e5m2", [57344, 58000, 61439.9, np.inf, np.inf, -np.inf]))

    def test_nan_passthrough(self):
        out = _run([np.nan, 1.0, np.inf], "float16")
        _assert_same(out, _expect("float16", [np.nan, 1.0, np.inf]))


@pytest.mark.skipif(_missing("ml_dtypes"), reason="ml_dtypes is not installed")
class TestE4M3Fn:
    """e4m3fn：顶格 NaN、无 inf。边界 464，tie 归有限值，溢出 → NaN（950 实测）。"""

    def test_band_tie_and_overflow(self):
        out = _run([448, 460, 464, 465, 479.9, 500, -464, -465], "float8_e4m3fn")
        _assert_same(out, _expect("float8_e4m3fn", [448, 460, 464, np.nan, np.nan, np.nan, -464, np.nan]))

    def test_inf_source_becomes_nan(self):
        out = _run([np.inf, -np.inf, 1.0], "float8_e4m3fn")
        _assert_same(out, _expect("float8_e4m3fn", [np.nan, np.nan, 1.0]))


@pytest.mark.skipif(_missing("en_dtypes"), reason="en_dtypes is not installed")
class TestHiFloat8:
    """hifloat8：1-bit 顶 binade。边界 40960，tie 归 inf（950 实测）。"""

    def test_band_tie_and_overflow(self):
        out = _run([32768, 40000, 40959.9, 40960, 41000, 49152, -40960], "hifloat8")
        _assert_same(out, _expect("hifloat8", [32768, 40000, 40959.9, np.inf, np.inf, np.inf, -np.inf]))


@pytest.mark.skipif(_missing("en_dtypes"), reason="en_dtypes is not installed")
class TestE8M0:
    """e8m0：无 inf/无符号/无零。双侧饱和，负值 NaN（950 MX 通路实测）。"""

    def test_saturation_and_nan(self):
        min_v, max_v = 2.0**-127, 2.0**127
        out = _run([-5.0, 0.0, -0.0, 1e-50, min_v, 1.0, max_v, 2e38], "float8_e8m0")
        _assert_same(out, _expect("float8_e8m0", [np.nan, min_v, min_v, min_v, min_v, 1.0, max_v, max_v]))


class TestSaturationModeUnchanged:
    """overflow_mode=0（休眠饱和通路）：clip 到 ±max_finite，行为维持不变。"""

    def test_clip_path(self):
        out = _run([65510.0, 70000.0, -70000.0], "float16", overflow_mode=0)
        _assert_same(out, _expect("float16", [65504.0, 65504.0, -65504.0]))
