# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Tests for golden_flatten (common/invoke.py, kernel通路 deep 语义):
- Flat ndarray output
- Nested (TensorList) output
- Edge case: empty tuple
"""

import numpy as np

from ttk.core_modules.common.invoke import golden_flatten


def _golden_reshape(results):
    return golden_flatten(results, deep=True)


class TestGoldenReshapeFlat:
    def test_flat_ndarrays(self):
        a = np.ones((3,), dtype="float16")
        b = np.ones((4,), dtype="float32")
        result = _golden_reshape((a, b))
        assert len(result) == 2
        assert result[0] is a
        assert result[1] is b


class TestGoldenReshapeNested:
    def test_nested_tensorlist(self):
        a1 = np.ones((3,), dtype="float16")
        a2 = np.ones((4,), dtype="float16")
        a3 = np.ones((5,), dtype="float32")
        result = _golden_reshape(((a1, a2), a3))
        assert len(result) == 3
        assert result[0] is a1
        assert result[1] is a2
        assert result[2] is a3


class TestGoldenReshapeEdgeCases:
    def test_empty_tuple(self):
        result = _golden_reshape(())
        assert result == []
