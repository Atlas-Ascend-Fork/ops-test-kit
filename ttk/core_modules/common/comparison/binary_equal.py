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
binary comparison

分块实现：sha256 分块喂入（digest 与全量一致）、字节级差异定位
O(CHUNK_SIZE)。diff_index 只保留前 K 个差异元素（索引升序，仅
_log_diff_output 消费）；差异元素计数为精确值（precision 不变）。
"""

# Standard Packages
import hashlib

import numpy as np

from ....utilities.dtypes import is_4bit_dtype

# Third-party Packages
from .registry import CHUNK_SIZE, FAIL_REASONS, ComparisonBase, EachCompareResult, _FirstK, register_comparison

# 自定义 dtype（无 .data 直读，哈希需 tobytes 拷贝——分块后拷贝同样 O(CHUNK)）
_HASH_COPY_DTYPES = ("bfloat16", "int4", "float8_e5m2", "float8_e4m3fn", "hifloat8")


def _hash_array(arr) -> str:
    """分块 sha256：对 arr 的字节序列增量哈希，digest 与一次性哈希完全一致。"""
    h = hashlib.sha256()
    copy_bytes = arr.dtype.name in _HASH_COPY_DTYPES
    for s in range(0, arr.size, CHUNK_SIZE):
        piece = arr[s : s + CHUNK_SIZE]
        h.update(piece.tobytes() if copy_bytes else piece.data)
    return h.hexdigest()


@register_comparison(["bin", "binary", "binary_equal"])
class BinaryComparison(ComparisonBase):
    STANDARD_NAME = "binary_equal"

    def compare_impl(self) -> EachCompareResult:
        # 空数组已在 compare() 入口 _check_empty 统一短路，这里只处理非空
        if str(self.output.dtype) != str(self.golden.dtype):
            return self._cross_dtype_compare()
        diff_idx, golden_size, diff_idx_size = self._numpy_binary_compare(self.output, self.golden)
        return self._result(diff_idx, golden_size, diff_idx_size)

    def _result(self, diff_idx, golden_size, diff_idx_size):
        if diff_idx_size == 0:
            return EachCompareResult(
                1, is_pass=True, standard=self.STANDARD_NAME, metrics={"standard": self.STANDARD_NAME, "pass": True}
            )
        precision = (golden_size - diff_idx_size) / golden_size
        return EachCompareResult(
            precision,
            diff_idx,
            is_pass=False,
            standard=self.STANDARD_NAME,
            metrics={"standard": self.STANDARD_NAME, "pass": False, "reason": FAIL_REASONS["bitwise_mismatch"]},
        )

    def _cross_dtype_compare(self) -> EachCompareResult:
        od, gd = self.output.dtype, self.golden.dtype
        # int4/float4 自定义 dtype 会让 np.issubdtype/promote_types 崩，短路拒
        if is_4bit_dtype(od) or is_4bit_dtype(gd):
            return self._reject(od, gd)
        out_int = np.issubdtype(od, np.integer) or od.kind == "b"
        gold_int = np.issubdtype(gd, np.integer) or gd.kind == "b"
        if out_int and gold_int:
            promoted = np.promote_types(od, gd)
            if np.issubdtype(promoted, np.integer):
                diff_idx, golden_size, diff_idx_size = self._numpy_binary_compare(
                    self.output.astype(promoted), self.golden.astype(promoted)
                )
                return self._result(diff_idx, golden_size, diff_idx_size)
        return self._reject(od, gd)

    def _reject(self, od, gd) -> EachCompareResult:
        return EachCompareResult(
            0,
            is_pass=False,
            standard=self.STANDARD_NAME,
            log=f"Dtype not bitwise-comparable: {od} vs {gd}",
            metrics={"standard": self.STANDARD_NAME, "pass": False, "reason": FAIL_REASONS["cross_dtype_uncomparable"]},
        )

    @staticmethod
    def _bytes_equal(output: np.ndarray, golden: np.ndarray) -> bool:
        """逐字节相等判定（与 SHA256 字节哈希同语义，但走 SIMD 且首块不等即短路）。

        仅对可 .data 直读的常规 dtype 成立（调用方以 _HASH_COPY_DTYPES 闸门）；
        判"两块内存是否逐字节相同"不需要密码学哈希，实测 2GiB 一对：
        SHA256 x2 = 1.97s，本法 = 0.09s。浮点 ±0 / NaN 的字节级差异同样判不等。"""
        if output.nbytes != golden.nbytes:
            return False
        ou = output.view(np.uint8).reshape(-1)
        gu = golden.view(np.uint8).reshape(-1)
        for start in range(0, ou.size, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, ou.size)
            if not np.array_equal(ou[start:end], gu[start:end]):
                return False
        return True

    @staticmethod
    def _numpy_binary_compare(output: np.ndarray, golden: np.ndarray):
        if not output.flags["C_CONTIGUOUS"]:
            output = np.ascontiguousarray(output)
        if not golden.flags["C_CONTIGUOUS"]:
            golden = np.ascontiguousarray(golden)
        if "float8_e8m0" in str(output.dtype):
            output = output.view(np.uint8)
            golden = golden.view(np.uint8)
        if "float4" in str(output.dtype):
            output = output.view(np.int8)
            golden = golden.view(np.int8)
        # 逐字节快路径（SIMD + 首块不等即短路）替代整对 SHA256——语义与字节哈希
        # 完全一致。view(uint8) 对无 .data 直读的自定义窄 dtype（bfloat16/int4/
        # fp8 等）不成立，那几类仍走分块哈希。
        if output.dtype.name not in _HASH_COPY_DTYPES:
            if BinaryComparison._bytes_equal(output, golden):
                return None, golden.size, 0
        elif _hash_array(output) == _hash_array(golden):
            return None, golden.size, 0

        first_k, diff_count = BinaryComparison._scan_byte_diff(
            output,
            golden,
            output.dtype.itemsize,  # view 后为 1（e8m0/float4 字节位即元素位）
        )
        if not diff_count:
            return None, golden.size, 0
        return np.asarray(first_k, dtype=np.int64), golden.size, diff_count

    @staticmethod
    def _scan_byte_diff(output, golden, itemsize):
        """字节级分块差异定位 → (前 K 个差异元素位升序, 差异元素总数)。

        CHUNK_SIZE 为 2^20，被任意 2 的幂 itemsize 整除——字节块边界不切断元素，
        元素字节必落同块（块内 unique 即全局去重）。"""
        output_int = output.view(np.uint8)
        golden_int = golden.view(np.uint8)
        first_k = _FirstK(BinaryComparison.MAX_DIFF_OUTPUT)
        diff_count = 0
        for s in range(0, output_int.size, CHUNK_SIZE):
            same = output_int[s : s + CHUNK_SIZE] == golden_int[s : s + CHUNK_SIZE]
            if same.all():
                continue
            bad_bytes = np.flatnonzero(~same)  # 块内字节位
            elem_local = np.unique(bad_bytes // itemsize)  # 块内元素位（升序去重）
            # 双方 NaN 的元素位视为相等（isnan 作用于 view 后数组——e8m0/float4
            # 的 uint8/int8 视图无 NaN 概念，恒 False）
            elem_chunk_o = output[s // itemsize : (s + CHUNK_SIZE) // itemsize]
            elem_chunk_g = golden[s // itemsize : (s + CHUNK_SIZE) // itemsize]
            both_nan = np.isnan(elem_chunk_o) & np.isnan(elem_chunk_g)
            if both_nan.any():
                elem_local = elem_local[~both_nan[elem_local]]
            diff_count += int(elem_local.size)
            first_k.add(elem_local + s // itemsize)
        return first_k.result(), diff_count
