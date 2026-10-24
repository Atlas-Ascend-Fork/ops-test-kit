# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Regression test: custom input plugin must sync original_input_arrays.

Bug: In Kernel/GEIR gen_input, the custom ``input`` plugin updated
``context.input_arrays`` (run-format) but left ``context.original_input_arrays``
(ori-format) stale.  XPU cross_check reads ``original_input_arrays`` first
(``_xpu_inputs`` in profiling.py), so the third-party provider received
pre-custom-input random data instead of the plugin's data.

Fix: call ``_transform_to_original_format`` after the plugin returns, same
as the stored_inputs and manual_input_binaries branches already do.
Additionally, ``_transform_to_original_format`` no longer short-circuits on
``golden_mode == "Disable"``: cross_check sets ``golden_mode_override`` to
``"Promote"`` *after* ``gen_input``, so the override is not visible during
input generation.  The function now always computes ``original_input_arrays``
(only ``manual_golden_binaries`` skips).
"""

from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.kernel import input_generate as _in_gen
from ttk.core_modules.kernel.testcase import TestcaseKernel
from ttk.switches import SWITCHES

_gen_input = _in_gen.gen_input


def _kernel_case(name="custom_input_sync"):
    case = TestcaseKernel()
    case.testcase_name = name
    case.op_name = "add"
    case.input_shapes = ((2,), None)
    case.input_dtypes = ("float32", "float32")
    case.input_formats = ("ND", "ND")
    case.input_ori_shapes = case.input_shapes
    case.input_ori_formats = case.input_formats
    case.output_shapes = ((2,),)
    case.output_dtypes = ("float32",)
    case.output_formats = ("ND",)
    case.output_ori_shapes = case.output_shapes
    case.output_ori_formats = case.output_formats
    case.output_inplace_indexes = ()
    case.output_shape_unknown_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), None)
    case.precision_tolerances = ((0.001, 0.001),)
    case.absolute_precision = (0.0001,)
    case._input_distribution = (0, 0)
    case._output_distribution = (0,)
    case.is_valid = True
    case.fail_reason = None
    return case


def _setup_mocks(monkeypatch, input_func, golden_mode="Enable"):
    sw = SWITCHES()
    sw.golden_mode = golden_mode
    sw.plugin_path = ()
    monkeypatch.setattr(_in_gen, "get_global_storage", lambda: sw)

    mock_keeper = MagicMock()
    mock_keeper.return_value.info_of.return_value = {"inputs": []}
    monkeypatch.setattr(_in_gen, "OpInfoKeeper", mock_keeper)

    monkeypatch.setattr(_in_gen, "get_plugin_function", lambda *_a, **_k: input_func)


def test_custom_input_syncs_original_input_arrays(monkeypatch):
    """Plugin returns NEW arrays → original_input_arrays must reflect them."""
    case = _kernel_case()
    sentinel = 42.0

    def custom_input(*arrays, **kwargs):
        arr0 = arrays[0]
        return (np.full(arr0.shape, sentinel, dtype=arr0.dtype), arrays[1])

    _setup_mocks(monkeypatch, custom_input)

    _gen_input(case)

    expected = np.array([42.0, 42.0], np.float32)
    np.testing.assert_array_equal(case.input_arrays[0], expected)
    assert case.original_input_arrays is not None
    np.testing.assert_array_equal(case.original_input_arrays[0], expected)
