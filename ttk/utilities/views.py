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
numpy array view utils（槽模型 storage/view 分离的低层引擎）
"""

import numpy


def numpy_as_strided_safe(base, shape, strides):
    """numpy as_strided that handles non-native dtypes (e.g. ml_dtypes.float8_e5m2).

    numpy.as_strided internally calls np.asarray which fails with
    ``TypeError: data type '<f1' not understood`` for certain third-party
    dtypes (kind='f' but internal type code conflicts with numpy reserved codes).
    Workaround: view the array as numpy.void with same itemsize, apply as_strided,
    then view back to original dtype.

    Native numpy dtypes (kind in 'biufcmMOSUV') skip the try/except overhead.
    Third-party dtypes with kind='V' work natively and also skip the fallback.
    Only the problematic case (kind='f' but non-standard, like ml_dtypes.float8_e5m2)
    triggers the void-proxy workaround.

    上层入口：common.testcase.tensor.make_storage_view（按槽声明的完整视图构造）。
    """
    from numpy.lib.stride_tricks import as_strided as _np_as_strided

    dtype = base.dtype
    if dtype.kind != "f" or dtype.char in ("f", "d", "e", "g"):
        return _np_as_strided(base, shape=shape, strides=strides)
    try:
        return _np_as_strided(base, shape=shape, strides=strides)
    except (TypeError, ValueError):
        proxy = numpy.dtype((numpy.void, dtype.itemsize))
        base_proxy = base.view(proxy)
        view_proxy = _np_as_strided(base_proxy, shape=shape, strides=strides)
        return view_proxy.view(dtype)
