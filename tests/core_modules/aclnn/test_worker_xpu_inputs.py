# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_aclnn_xpu_inputs 逻辑视图装配测试。

三方参考须按逻辑视图（view shape/stride/offset）取数，而非存储形态
np_storages。覆盖 transpose/broadcast/slice 三种非连续构造与 complex32 直通。
"""

import numpy
from numpy.lib.stride_tricks import as_strided

from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.aclnn.worker import _aclnn_xpu_inputs, _xpu_view_arrays


def _make_case(view_shapes, dtypes, storages, strides=None, offsets=None, storage_shapes=None, outputs=()):
    case = TestcaseAclnn()
    case.tensor_view_shapes = view_shapes
    case.tensor_dtypes = dtypes
    case.np_storages = list(storages)
    case.tensor_view_strides = strides
    case.tensor_view_offsets = offsets
    case.tensor_storage_shapes = storage_shapes
    case.output_tensor_indexes = outputs
    return case


def test_transpose_input_sent_as_logical_view():
    storage0 = numpy.arange(18, dtype=numpy.float32).reshape(3, 6)
    storage1 = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    case = _make_case(
        view_shapes=((2, 3), (2, 3), (2, 3)),
        dtypes=("float32", "float32", "float32"),
        storages=(storage0, storage1, numpy.ones((2, 3), dtype=numpy.float32)),
        strides=((4, 1), None, None),
        offsets=(5, None, None),
        storage_shapes=((3, 6), None, None),
        outputs=(2,),
    )
    inputs = _aclnn_xpu_inputs(case)
    assert len(inputs) == 2  # 纯输出槽（slot2）被剔除
    assert inputs[0].shape == (2, 3)
    numpy.testing.assert_array_equal(inputs[0], [[5, 6, 7], [9, 10, 11]])
    numpy.testing.assert_array_equal(inputs[1], storage1)


def test_broadcast_input_sent_as_logical_view():
    storage0 = numpy.array([42.0], dtype=numpy.float32)
    case = _make_case(
        view_shapes=((2, 3), (2, 3)),
        dtypes=("float32", "float32"),
        storages=(storage0, numpy.zeros((2, 3), dtype=numpy.float32)),
        strides=((0, 0), None),
        offsets=(0, None),
        storage_shapes=((1,), None),
    )
    inputs = _aclnn_xpu_inputs(case)
    assert inputs[0].shape == (2, 3)
    numpy.testing.assert_array_equal(inputs[0], numpy.full((2, 3), 42.0, dtype=numpy.float32))


def test_slice_input_sent_as_logical_view():
    storage0 = numpy.arange(468, dtype=numpy.float32).reshape(26, 6, 3)
    case = _make_case(
        view_shapes=((7, 6, 3), (7, 6, 3)),
        dtypes=("float32", "float32"),
        storages=(storage0, numpy.zeros((7, 6, 3), dtype=numpy.float32)),
        strides=((36, 3, 1), None),
        offsets=(36, None),
        storage_shapes=((26, 6, 3), None),
    )
    inputs = _aclnn_xpu_inputs(case)
    assert inputs[0].shape == (7, 6, 3)
    expected = as_strided(storage0.ravel()[36:], shape=(7, 6, 3), strides=(36 * 4, 3 * 4, 4))
    numpy.testing.assert_array_equal(inputs[0], numpy.array(expected, copy=True))


def test_view_rebuild_failure_falls_back_to_storage():
    storage0 = numpy.zeros((2, 3), dtype=numpy.float32)
    case = _make_case(
        view_shapes=((2, 3),),
        dtypes=("float32",),
        storages=(storage0,),
        strides=((9, 9),),
    )
    inputs = _xpu_view_arrays(case)
    assert inputs[0] is storage0


def test_complex32_storage_passthrough_unchanged():
    storage0 = numpy.arange(12, dtype=numpy.float16).reshape(2, 3, 2)
    case = _make_case(
        view_shapes=((2, 3),),
        dtypes=("complex32",),
        storages=(storage0,),
    )
    inputs = _aclnn_xpu_inputs(case)
    assert inputs[0] is storage0
