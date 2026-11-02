# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""customize_inputs 返回值回写：None = 原地写法；返回序列按槽位写回 raw_inputs
视图；长度/形状不匹配、dynamic 槽非 None 报错；框架张量统一转 numpy。"""

from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.framework_api.input_generate import (
    _apply_input_plugin_result,
    _plugin_value_to_numpy,
)


def _nb():
    return SimpleNamespace(testcase_name="case_x")


def test_none_keeps_inplace_semantics():
    """整体 None 或槽位 None：不回写，保持生成值。"""
    raw = [np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.int32)]
    _apply_input_plugin_result(None, raw, _nb(), None)
    _apply_input_plugin_result((np.array([3.0, 4.0], dtype=np.float32), None), raw, _nb(), None)
    assert raw[0].tolist() == [3.0, 4.0]
    assert raw[1].tolist() == [0, 0]


def test_return_writes_back():
    """返回序列按槽位写回：视图写穿 storage；TensorList 嵌套展平；单张量可裸返回。"""
    storage = np.zeros(8, dtype=np.float32)
    _apply_input_plugin_result((np.arange(4, dtype=np.float32),), [storage[::2]], _nb(), None)
    assert storage.tolist() == [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]

    raw = [np.zeros(2), np.zeros(2), np.zeros(2)]
    _apply_input_plugin_result(([np.ones(2), np.full(2, 2.0)], np.full(2, 3.0)), raw, _nb(), None)
    assert [r.tolist() for r in raw] == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    single = [np.zeros(3, dtype=np.float32)]
    _apply_input_plugin_result(np.array([1.0, 2.0, 3.0], dtype=np.float32), single, _nb(), None)
    assert single[0].tolist() == [1.0, 2.0, 3.0]


def test_mismatch_raises():
    """长度/shape 不符、dynamic 槽返回非 None：显式报错。"""
    raw2 = [np.zeros(2), np.zeros(2)]
    with pytest.raises(RuntimeError, match="returned 1 tensor"):
        _apply_input_plugin_result((np.zeros(2),), raw2, _nb(), None)

    with pytest.raises(RuntimeError, match=r"shape \(3, 2\) for flat slot 0, expected \(2, 3\)"):
        _apply_input_plugin_result((np.zeros((3, 2)),), [np.zeros((2, 3))], _nb(), None)

    with pytest.raises(RuntimeError, match="no generated tensor"):
        _apply_input_plugin_result((np.zeros(2), None), [None, np.zeros(2)], _nb(), None)


def test_plugin_value_to_numpy():
    """框架张量/普通对象统一转 numpy。"""
    tf = pytest.importorskip("tensorflow")

    from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend

    backend = CpuTfBackend()
    t = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    arr = _plugin_value_to_numpy(t, backend)
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert _plugin_value_to_numpy(tf.Variable([5.0, 6.0]), backend).tolist() == [5.0, 6.0]
    assert _plugin_value_to_numpy(np.array([1.5]), backend).tolist() == [1.5]
    assert _plugin_value_to_numpy([1, 2, 3], backend).tolist() == [1, 2, 3]
