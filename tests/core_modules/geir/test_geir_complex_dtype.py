# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for GEIR complex32/64/128 dtype support."""

import struct

import numpy as np

from ttk.core_modules.geir.graph_builder import _DTYPE_TO_GE_ENUM, _render_template, _resolve_dtype
from ttk.core_modules.geir.output_parse import _parse_stdout


def _make_geir_output(arrays):
    """Build C++ output protocol: [8B num_outputs] [8B byte_count][data]..."""
    parts = [struct.pack("<q", len(arrays))]
    for arr in arrays:
        data = arr.tobytes()
        parts.append(struct.pack("<q", len(data)))
        parts.append(data)
    return b"".join(parts)


def test_resolve_dtype_complex():
    assert _resolve_dtype("complex32") == "DT_COMPLEX32"
    assert _resolve_dtype("complex64") == "DT_COMPLEX64"
    assert _resolve_dtype("complex128") == "DT_COMPLEX128"
    assert "complex32" in _DTYPE_TO_GE_ENUM
    assert "complex64" in _DTYPE_TO_GE_ENUM
    assert "complex128" in _DTYPE_TO_GE_ENUM


def test_parse_stdout_complex32():
    arr = np.arange(12, dtype=np.float16).reshape(2, 3, 2)
    data = _make_geir_output([arr])
    result = _parse_stdout(data, ["complex32"], [[2, 3]])
    assert len(result) == 1
    assert result[0].dtype == np.float16
    assert result[0].shape == (2, 3, 2)
    np.testing.assert_array_equal(result[0], arr)


def test_parse_stdout_complex64():
    arr = np.array([[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]], dtype=np.complex64)
    data = _make_geir_output([arr])
    result = _parse_stdout(data, ["complex64"], [[2, 3]])
    assert len(result) == 1
    assert result[0].dtype == np.complex64
    assert result[0].shape == (2, 3)
    np.testing.assert_array_equal(result[0], arr)


def test_parse_stdout_complex128():
    arr = np.array([[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]], dtype=np.complex128)
    data = _make_geir_output([arr])
    result = _parse_stdout(data, ["complex128"], [[2, 3]])
    assert len(result) == 1
    assert result[0].dtype == np.complex128
    assert result[0].shape == (2, 3)
    np.testing.assert_array_equal(result[0], arr)


def test_parse_stdout_mixed_complex_and_float():
    c32_arr = np.arange(12, dtype=np.float16).reshape(2, 3, 2)
    f32_arr = np.arange(6, dtype=np.float32)
    data = _make_geir_output([c32_arr, f32_arr])
    result = _parse_stdout(data, ["complex32", "float32"], [[2, 3], [6]])
    assert len(result) == 2
    assert result[0].dtype == np.float16
    assert result[0].shape == (2, 3, 2)
    assert result[1].dtype == np.float32
    assert result[1].shape == (6,)


def test_template_get_data_type_size_has_complex():
    dt_enums = ["DT_FLOAT", "DT_FLOAT16", "DT_COMPLEX32", "DT_COMPLEX64", "DT_COMPLEX128"]
    src = _render_template(
        "geir_op_template.cpp.j2",
        proto_file="dummy.proto",
        op_class="Dummy",
        input_names=[],
        output_names=[],
        dynamic_input_names=[],
        attr_entries=[],
        dtype_map=[(v, v) for v in dt_enums],
        format_map=[("FORMAT_ND", "FORMAT_ND")],
        dtype_size_cases=[
            ("DT_FLOAT", 4),
            ("DT_FLOAT16", 2),
            ("DT_COMPLEX32", 4),
            ("DT_COMPLEX64", 8),
            ("DT_COMPLEX128", 16),
        ],
        packed_float4_enums=[],
        ge_prof_ok=True,
    )
    assert "case DT_COMPLEX32: return 4;" in src
    assert "case DT_COMPLEX64: return 8;" in src
    assert "case DT_COMPLEX128: return 16;" in src
    assert "DT_COMPLEX32" in src
    assert "DT_COMPLEX64" in src
    assert "DT_COMPLEX128" in src
