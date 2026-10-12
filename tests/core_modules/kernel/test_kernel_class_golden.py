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
Integration coverage for the KERNEL **class-form** golden dispatch.

The partial-split (``_NegScale``) and no-custom-``__init__`` (``_Double``)
shapes are covered by ``test_aclnn_class_golden.py`` (same pattern, ACLNN
path). This file keeps only the KERNEL-specific **dispatch routing** guard:
a class is ``Callable`` but MUST route to ``__invoke_class`` (not the
custom-callable branch that would do ``golden_func(*input_arrays)``).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.kernel import golden as _mod
from ttk.core_modules.kernel.testcase import TestcaseKernel

# Access module-level private functions via getattr to dodge class-style name mangling.
_generate_golden = getattr(_mod, "__generate_golden")


# --------------------------------------------------------------------------- #
# Golden classes — emulate what a real plugin author would ship.
# --------------------------------------------------------------------------- #
class _NegScale:
    """Partial split: ``__init__`` takes a config attr, ``__call__`` takes the input."""

    def __init__(self, *, scale):
        self.scale = scale

    def __call__(self, x):  # noqa: N803 (short param mirrors real plugins)
        return [x * self.scale]


# --------------------------------------------------------------------------- #
# Helpers — mirror tests/test_kernel_builtin_golden.py::_make_testcase.
# --------------------------------------------------------------------------- #
def _make_testcase(op_name, input_shapes, input_dtypes, output_shapes, output_dtypes, attributes):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name}_class_golden"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = input_shapes
    case.output_ori_shapes = output_shapes
    n_in = len(input_shapes)
    n_out = len(output_shapes or ())
    case.input_formats = ("ND",) * n_in
    case.input_ori_formats = ("ND",) * n_in
    case.output_formats = ("ND",) * n_out
    case.output_ori_formats = ("ND",) * n_out
    case.input_data_ranges = (None,) * n_in
    case.attributes = attributes
    case.input_arrays = tuple(np.ones(s, dtype=d) for s, d in zip(input_shapes, input_dtypes))
    case.original_input_arrays = None
    return case


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Enable"  # not Promote → __golden_mode is a no-op
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


def _mock_op_info(input_names):
    """OpInfoKeeper mock whose info_of() yields inputs named per ``input_names``."""
    mock = MagicMock()
    mock.return_value.info_of.return_value = {"inputs": [{"name": n} for n in input_names]}
    return mock


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    # Avoid real ASCEND toolkit lookups during construction/validate.
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch("ttk.core_modules.kernel.golden.OpInfoKeeper")
@patch("ttk.core_modules.kernel.golden.get_global_storage")
@patch("ttk.core_modules.common.plugin.resolver.get_plugin_function")
class TestKernelClassGoldenDispatch:
    """class-form golden (isinstance(golden_func, type)) → __invoke_class end-to-end."""

    def test_dispatch_routes_class_to_invoke_class_not_callable_path(self, mock_get_plugin, mock_sw, mock_op_info):
        """A class is a ``Callable`` but MUST route to ``__invoke_class`` (not the custom-callable
        path that would do ``golden_func(*input_arrays)``). Guards the ``isinstance(type)`` check
        ordering in ``__invoke_golden``: if a class leaked to the custom branch it would be called
        as ``_NegScale(array)`` → TypeError → ``['GOLDEN_FAILURE']``."""
        mock_get_plugin.return_value = _NegScale
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": [{"name": "x"}]}

        inp = np.array([1.0, 2.0, 3.0], dtype="float32")
        case = _make_testcase(
            op_name="neg_scale",
            input_shapes=((3,),),
            input_dtypes=("float32",),
            output_shapes=((3,),),
            output_dtypes=("float32",),
            attributes={"scale": 5},
        )
        case.input_arrays = (inp,)

        golden = _generate_golden(case, ["float32"])

        # If mis-routed to the custom-callable branch, golden[0] would be the
        # string "GOLDEN_FAILURE"; the class path yields a numeric ndarray.
        assert isinstance(golden[0], np.ndarray), (
            f"class golden must hit __invoke_class, not the custom-callable branch: {golden[0]!r}"
        )
        np.testing.assert_array_equal(golden[0], np.array([5.0, 10.0, 15.0], dtype="float32"))
