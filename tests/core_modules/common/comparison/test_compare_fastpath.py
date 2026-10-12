# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""大数组比对快路径：isclose 抬 fp32、binary_equal 逐字节 —— 结果必须与原实现逐位一致。

分块比对快速路径回归测试；分块驱动方式
按分支 API（CloseComparison._isclose / registry.CHUNK_SIZE）适配。"""

import hashlib

import numpy as np
import pytest

from ttk.core_modules.common.comparison.binary_equal import BinaryComparison
from ttk.core_modules.common.comparison.is_close import CloseComparison
from ttk.core_modules.common.comparison.registry import CHUNK_SIZE

SPECIALS = [np.inf, -np.inf, np.nan, 0.0, -0.0]


def _mixed(dtype, n, seed):
    """随机数组 + 注入特殊值（±inf / nan / ±0 / dtype 极值）。"""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3, 3, n).astype(dtype)
    for v in SPECIALS + [np.finfo(dtype).tiny, np.finfo(dtype).max]:
        a[rng.choice(n, max(n // 200, 1), replace=False)] = v
    return a


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(("rtol", "atol"), [(1e-3, 1e-4), (1e-4, 1e-5), (0.0, 1e-8), (1e-2, 0.0)])
def test_close_mask_matches_numpy_isclose(dtype, rtol, atol):
    """抬 fp32 的掩码与 np.isclose(equal_nan=True) 必须逐位相同（含 nan/inf/±0/denormal）。"""
    a = _mixed(dtype, 50000, seed=11)
    b = a.copy()
    rng = np.random.default_rng(12)
    idx = rng.choice(a.size, a.size // 20, replace=False)
    b[idx] = rng.uniform(-3, 3, idx.size)
    ref = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    got = CloseComparison._close_mask(a, b, rtol, atol)
    assert np.array_equal(ref, got)


def _close_cmp_on(a, b, rtol, atol):
    """按分支 API 驱动分块 isclose（绕过 base.compare() 的输入包装）。"""
    cmp_obj = CloseComparison.__new__(CloseComparison)
    cmp_obj.output, cmp_obj.golden = a, b
    cmp_obj.rtol, cmp_obj.atol = rtol, atol
    return cmp_obj._isclose()


def test_close_chunked_matches_single_shot_across_boundary():
    """分块路径与单次全量 np.isclose 给出相同的不一致下标（含跨块边界的差异）。"""
    n = 2 * CHUNK_SIZE + 1000
    a = np.full(n, 1.0, dtype="float16")
    b = a.copy()
    for pos in (0, CHUNK_SIZE - 1, CHUNK_SIZE, n - 1):
        b[pos] = 9.0
    result = _close_cmp_on(a, b, 1e-3, 1e-4)
    ref = np.where(~np.isclose(a, b, rtol=1e-3, atol=1e-4, equal_nan=True))[0]
    assert np.array_equal(np.sort(result.diff_index), ref)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "int64", "float16", "float32"])
def test_bytes_equal_matches_sha256(dtype):
    """逐字节判定必须与原 SHA256 字节哈希同结论（浮点 ±0 / NaN 一并覆盖）。"""
    rng = np.random.default_rng(3)
    n = 20000
    for ndiff in (0, 1, 7):
        if "int" in dtype:
            a = rng.integers(-100, 100, n).astype(dtype)
        else:
            a = rng.uniform(-3, 3, n).astype(dtype)
            a[0], a[1] = 0.0, np.nan
        b = a.copy()
        if "float" in dtype and ndiff:
            b[0] = -0.0  # 数值相等但字节不同 -> 必须判不等
        if ndiff:
            pos = rng.choice(np.arange(2, n), ndiff, replace=False)
            b[pos] = (b[pos] + 1).astype(dtype)
        a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        ref = hashlib.sha256(a.data).hexdigest() == hashlib.sha256(b.data).hexdigest()
        assert BinaryComparison._bytes_equal(a, b) is ref


def test_bytes_equal_detects_diff_across_chunk_boundary():
    """跨块边界的单字节差异不能被漏判。"""
    step = CHUNK_SIZE
    a = np.zeros(2 * step + 100, dtype=np.int8)
    for pos in (step - 1, step, 2 * step + 50):
        b = a.copy()
        b[pos] = 1
        assert BinaryComparison._bytes_equal(a, b) is False
    assert BinaryComparison._bytes_equal(a, a.copy()) is True
