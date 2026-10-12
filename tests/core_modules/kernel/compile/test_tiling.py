# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""tiling 解析契约：bytes↔int32 tuple 互转、奇数字节哨兵、str 化返回恒 str。"""

import numpy as np
import pytest

from ttk.core_modules.kernel.compile.tiling import parse_tiling_data, render_tiling_data


def test_parse_tiling_data_round_trip():
    data_bytes, data_tuple = parse_tiling_data((1, 2, 3))
    assert data_bytes == np.array([1, 2, 3], dtype="<i4").tobytes()
    assert data_tuple == (1, 2, 3)
    assert parse_tiling_data(None) == (None, None)
    with pytest.raises(TypeError):
        parse_tiling_data(42)


def test_parse_tiling_data_odd_bytes_sentinel():
    _bytes, data_tuple = parse_tiling_data(b"\x01\x02\x03")
    assert data_tuple == ("TILING_NOT_4BYTE_ALIGNED",)


def test_render_tiling_data_contract():
    """返回值恒为 str——None 输入曾误返 (key, False) 元组。"""
    assert render_tiling_data(None, {}, 3) == "Tiling data is None"
    assert render_tiling_data((), {}, 3) == "Tiling data is empty"
    assert render_tiling_data((4, 5), {}, 3) == "(4, 5)"
    assert render_tiling_data((4, 5), {"_vars": {"3": ["a", "b"]}}, 3) == "{'a': 4, 'b': 5}"
