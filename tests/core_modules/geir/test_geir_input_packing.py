#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for GEIR 4-bit input packing.

NumPy 4-bit dtypes are unpacked (1 byte/element) while the generated C++
program reads the device packed representation ((num_elements + 1) / 2 bytes,
two 4-bit elements per byte) — see LoadInputFromFile in geir_op_template.cpp.j2.
"""

import numpy as np
import pytest

from ttk.core_modules.geir.execute import _write_input_bin
from ttk.utilities import is_4bit_dtype, unpack_4bits
from ttk.utilities.dtypes import numpy_float4_e2m1, numpy_int4


def _missing_en_dtypes():
    try:
        import en_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


def _missing_ml_dtypes():
    try:
        import ml_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


def _expected_bytes(num_elements):
    # mirrors C++ LoadInputFromFile: expected = (num_elements + 1) / 2
    return (num_elements + 1) // 2


@pytest.mark.parametrize(
    "make_dtype",
    [
        pytest.param(
            numpy_float4_e2m1, marks=pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
        ),
        pytest.param(numpy_int4, marks=pytest.mark.skipif(_missing_ml_dtypes(), reason="ml_dtypes is not installed")),
    ],
    ids=["float4_e2m1", "int4"],
)
def test_write_input_bin_packs_4bit(tmp_path, make_dtype):
    dtype = make_dtype()
    # en_dtypes/ml_dtypes store the 4-bit code (0-15) in each byte
    arr = np.arange(8, dtype=np.uint8).reshape(2, 4).view(dtype)
    path = str(tmp_path / "input.bin")

    _write_input_bin(arr, path)

    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size == _expected_bytes(arr.size)
    # low nibble first, matching pack_4bits / C++ packed representation
    np.testing.assert_array_equal(raw, [0x10, 0x32, 0x54, 0x76])
    unpacked = unpack_4bits(raw, dtype)
    np.testing.assert_array_equal(unpacked.view(np.uint8), np.arange(8, dtype=np.uint8))


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
@pytest.mark.parametrize("num_elements", [7, 8], ids=["odd", "even"])
def test_write_input_bin_4bit_file_size_matches_cpp(tmp_path, num_elements):
    # odd element counts must still yield ceil(n/2) bytes: pack_4bits pads to even
    dtype = numpy_float4_e2m1()
    arr = np.arange(num_elements, dtype=np.uint8).view(dtype)
    path = str(tmp_path / "input.bin")

    _write_input_bin(arr, path)

    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size == _expected_bytes(num_elements)
    assert is_4bit_dtype(arr.dtype)


def test_write_input_bin_non_4bit_passthrough(tmp_path):
    arr = np.arange(12, dtype=np.float16).reshape(3, 4)
    path = str(tmp_path / "input.bin")

    _write_input_bin(arr, path)

    loaded = np.fromfile(path, dtype=np.float16)
    np.testing.assert_array_equal(loaded, arr.reshape(-1))
