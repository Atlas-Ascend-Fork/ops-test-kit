# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GEIR 静态空输出：声明形状必须还原，动态/未知槽与真实异常仍保留可见性。"""

import struct

import numpy as np

from ttk.core_modules.geir.output_parse import _parse_stdout


def _make_geir_output(blobs):
    """Build C++ output protocol: [8B num_outputs] [8B byte_count][data]..."""
    parts = [struct.pack("<q", len(blobs))]
    for blob in blobs:
        parts.append(struct.pack("<q", len(blob)))
        parts.append(blob)
    return b"".join(parts)


def test_parse_stdout_restores_static_empty_shape():
    """声明静态空形状时必须还原，不能退化成一维 (0,)。"""
    expected = [(0, 3, 4, 5), (0, 3, 1, 1), (0, 3, 1, 1)]
    data = _make_geir_output([b""] * len(expected))

    result = _parse_stdout(data, ["float32"] * len(expected), expected, case_name="static_empty")

    assert [tuple(arr.shape) for arr in result] == expected
    assert all(arr.dtype == np.float32 and arr.size == 0 for arr in result)


def test_parse_stdout_restores_static_empty_shape_for_other_layouts():
    """空维可出现在任意轴：N/C/H/W 置零都应还原。"""
    expected = [[(2, 3, 4, 0), (2, 0, 4, 5)], [(0,), (0, 7)]]
    for shapes in expected:
        data = _make_geir_output([b""] * len(shapes))
        result = _parse_stdout(data, ["float16"] * len(shapes), shapes, case_name="empty_any_axis")
        assert [tuple(arr.shape) for arr in result] == shapes


def test_parse_stdout_dynamic_dim_slots_keep_flat_array():
    """-1 动态维/-2 未知 rank 无法确定形状，仍应保留扁平数组而非按声明猜测。"""
    for shapes in ([[-1, 3, 4, 5]], [[-2]], [[2, -1]]):
        data = _make_geir_output([b""] * len(shapes))
        result = _parse_stdout(data, ["float32"] * len(shapes), shapes, case_name="dyn_shape")
        assert [arr.size for arr in result] == [0]
        assert all(arr.shape == (0,) for arr in result)


def test_parse_stdout_nonempty_declaration_without_bytes_stays_flat():
    """声明非空却收到 0 字节是真实异常，不能被 reshape 掩盖成一张空张量。"""
    shapes = [[2, 3], [0, 3, 4, 5]]
    data = _make_geir_output([b"", b""])

    result = _parse_stdout(data, ["float32", "float32"], shapes, case_name="anomaly")

    assert result[0].shape == (0,)
    assert tuple(result[1].shape) == (0, 3, 4, 5)


def test_parse_stdout_empty_output_does_not_affect_following_outputs():
    """前一输出为空不能影响后续非空输出的字节流解析。"""
    shapes = [[0, 3, 4, 5], [2, 3]]
    payload = np.arange(6, dtype=np.float32).tobytes()
    data = _make_geir_output([b"", payload])

    result = _parse_stdout(data, ["float32", "float32"], shapes, case_name="mixed")

    assert tuple(result[0].shape) == (0, 3, 4, 5)
    assert tuple(result[1].shape) == (2, 3)
    np.testing.assert_array_equal(result[1], np.arange(6, dtype=np.float32).reshape(2, 3))
