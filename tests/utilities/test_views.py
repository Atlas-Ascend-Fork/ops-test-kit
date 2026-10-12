# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""views 公开面：numpy_as_strided_safe 的视图语义与第三方 dtype 兜底。"""

import numpy as np
import pytest

from ttk.utilities.views import numpy_as_strided_safe


def test_view_semantics_native_dtype():
    """offset + 字节 stride 寻址正确（view[i,j,k] == flat[offset + Σ stride]）。"""
    storage = np.arange(48, dtype=np.float32)
    view = numpy_as_strided_safe(storage[3:], shape=(2, 2, 3), strides=(24, 12, 4))
    assert view.shape == (2, 2, 3)
    assert view.strides == (24, 12, 4)
    assert view[0, 0, 0] == storage[3]
    assert view[1, 1, 2] == storage[3 + (1 * 24 + 1 * 12 + 2 * 4) // 4]


def test_view_survives_third_party_float8():
    """ml_dtypes float8（kind='f' 非标准 type code）——函数存在的理由。"""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    src = np.linspace(-1.0, 1.0, 24).astype(ml_dtypes.float8_e5m2)
    view = numpy_as_strided_safe(src[2:], shape=(2, 7), strides=(8, 1))
    assert view.dtype == src.dtype
    expected = np.array([[src[2 + i * 8 + j] for j in range(7)] for i in range(2)])
    assert np.array_equal(view, expected)
