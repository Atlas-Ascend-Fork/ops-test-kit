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
Tests for kwargs collection in golden/input generation:
- output_generation.__collect_dynamic_golden_kwargs
- input_generation.__collect_dynamic_kwargs

Verifies that kwargs use nested fields directly after normalize.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Access module-level private functions via getattr to avoid name mangling in classes
from ttk.core_modules.kernel import golden as _out_gen_mod
from ttk.core_modules.kernel import input_generate as _in_gen_mod
from ttk.core_modules.kernel.testcase import TestcaseKernel

_collect_golden_kwargs = getattr(_out_gen_mod, "__collect_dynamic_golden_kwargs")
_collect_input_kwargs = getattr(_in_gen_mod, "__collect_dynamic_kwargs")


def _make_testcase(
    op_name="Add",
    input_shapes=((8,), (8,)),
    input_dtypes=("float16", "float16"),
    output_shapes=((8,),),
    output_dtypes=("float16",),
    **kwargs,
):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name or 'None'}"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = kwargs.pop("input_ori_shapes", input_shapes)
    case.output_ori_shapes = kwargs.pop("output_ori_shapes", output_shapes)
    case.attributes = kwargs.pop("attributes", {})
    n_in = len(input_shapes)
    n_out = len(output_shapes or ())
    case.input_formats = kwargs.pop("input_formats", ("ND",) * n_in)
    case.input_ori_formats = kwargs.pop("input_ori_formats", ("ND",) * n_in)
    case.output_formats = kwargs.pop("output_formats", ("ND",) * n_out)
    case.output_ori_formats = kwargs.pop("output_ori_formats", ("ND",) * n_out)
    case.input_data_ranges = kwargs.pop("input_data_ranges", (None,) * n_in)
    for k, v in kwargs.items():
        setattr(case, k, v)
    return case


def _validate(case):
    n_in = len(case.input_shapes) if case.input_shapes else 0
    n_out = len(case.output_shapes) if case.output_shapes and not isinstance(case.output_shapes, str) else 0
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock:
        mock.return_value.info_of.return_value = {
            "coreType.value": "AiCore",
            "inputs": [{"name": f"i{i}"} for i in range(n_in)],
            "outputs": [{"name": f"o{i}"} for i in range(n_out)],
        }
        case.validate()


def _make_arrays(shapes, dtypes):
    arrays = []
    for shape, dtype in zip(shapes, dtypes):
        if shape is None:
            arrays.append(None)
        else:
            arrays.append(np.ones(shape, dtype=dtype))
    return arrays


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Enable"
    sw.plugin_path = None
    return sw


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


# =====================================================================
# Tests for output_generation.__collect_dynamic_golden_kwargs
# =====================================================================


class TestGoldenKwargsNonTensorList:
    @patch("ttk.core_modules.kernel.golden.OpInfoKeeper")
    @patch("ttk.core_modules.kernel.golden.get_global_storage")
    def test_input_dtypes_matches_stc(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,), (8,)),
            input_dtypes=("float16", "float32", "int32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_dtypes"] == case.input_dtypes
        assert kwargs["input_dtypes"] == ("float16", "float32", "int32")

    @patch("ttk.core_modules.kernel.golden.OpInfoKeeper")
    @patch("ttk.core_modules.kernel.golden.get_global_storage")
    def test_all_format_fields_match(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_formats=("ND", "NCHW"),
            input_ori_formats=("ND", "ND"),
            output_formats=("ND",),
            output_ori_formats=("ND",),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_formats"] == case.input_formats
        assert kwargs["input_ori_formats"] == case.input_ori_formats
        assert kwargs["output_formats"] == case.output_formats
        assert kwargs["output_ori_formats"] == case.output_ori_formats


# =====================================================================
# Tests for input_generation.__collect_dynamic_kwargs
# =====================================================================


class TestInputKwargsNonTensorList:
    @patch("ttk.core_modules.kernel.input_generate.OpInfoKeeper")
    @patch("ttk.core_modules.kernel.input_generate.get_global_storage")
    def test_input_ranges_matches_stc(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_data_ranges=((-1.0, 1.0), (-2.0, 2.0)),
        )
        _validate(case)
        _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)

        kwargs = _collect_input_kwargs(case)
        assert kwargs["input_ranges"] == case.input_data_ranges
