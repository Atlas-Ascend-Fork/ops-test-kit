#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GEIR pipe 二进制流解析。

协议 = 逐输出 [8B num_outputs][8B byte_count][data]*——chunk 读取/reshape
（动态维取实际字节数推）/逐输出解析；stdout/stderr 留给 GE 运行时日志。
"""

import io
import logging

import numpy as np

from ...utilities import get, is_4bit_dtype, resolve_numpy_dtypes, unpack_4bits


def _read_output_chunk(buf):
    byte_count_raw = buf.read(8)
    if len(byte_count_raw) < 8:
        return None
    byte_count = int(np.frombuffer(byte_count_raw, dtype=np.int64)[0])
    raw = buf.read(byte_count)
    if len(raw) < byte_count:
        return None
    return byte_count, raw


def _reshape_output(arr, expect_shape, idx, case_name):
    if expect_shape is None:
        return arr
    if arr.size == 0:
        expect_numel = int(np.prod(expect_shape)) if expect_shape else 0
        logging.warning(
            "[%s] output[%d] empty from GE; expect_shape=%s, expect_numel=%d; skipping reshape",
            case_name,
            idx,
            expect_shape,
            expect_numel,
        )
        return arr
    try:
        return arr.reshape(expect_shape)
    except ValueError as e:
        logging.warning(
            "[%s] output[%d] reshape failed (%s); arr.size=%d, expect_shape=%s; keeping flat array",
            case_name,
            idx,
            e,
            arr.size,
            expect_shape,
        )
        return arr


def _parse_single_output(buf, idx, output_dtypes, output_shapes, case_name):
    chunk = _read_output_chunk(buf)
    if chunk is None:
        return None, True
    byte_count, raw = chunk

    dtype_str = get(output_dtypes, idx) if output_dtypes else ""
    is_complex32 = "complex32" in str(dtype_str)
    is_4bit = is_4bit_dtype(dtype_str)
    if is_4bit:
        np_dtype = np.dtype("uint8")
    else:
        np_dtype = np.dtype("float16") if is_complex32 else np.dtype(resolve_numpy_dtypes([dtype_str])[0])
    try:
        arr = np.frombuffer(raw, dtype=np_dtype)
    except ValueError as e:
        itemsize = np_dtype.itemsize
        usable = byte_count - (byte_count % itemsize) if itemsize else 0
        logging.warning(
            "[%s] output[%d] frombuffer failed (%s); byte_count=%d not multiple of "
            "itemsize=%d (dtype=%s); truncating to %d usable bytes",
            case_name,
            idx,
            e,
            byte_count,
            itemsize,
            dtype_str,
            usable,
        )
        arr = np.frombuffer(raw[:usable], dtype=np_dtype) if usable else np.array([], dtype=np_dtype)

    if is_4bit:
        arr = unpack_4bits(arr, resolve_numpy_dtypes([dtype_str])[0])

    expect_shape = get(output_shapes, idx) if output_shapes else None
    if expect_shape is not None:
        expect_shape = list(expect_shape) + ([2] if is_complex32 else [])
    arr = _reshape_output(arr, expect_shape, idx, case_name)
    return arr, False


def _parse_stdout(data, output_dtypes, output_shapes, case_name=""):
    if not data:
        return []

    buf = io.BytesIO(data)
    num_outputs_raw = buf.read(8)
    if len(num_outputs_raw) < 8:
        return []
    num_outputs = int(np.frombuffer(num_outputs_raw, dtype=np.int64)[0])

    results = []
    for i in range(num_outputs):
        arr, stop = _parse_single_output(buf, i, output_dtypes, output_shapes, case_name)
        if stop:
            break
        results.append(arr)

    return results
