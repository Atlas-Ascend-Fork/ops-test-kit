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
Tests for ttk.core_modules.kernel.testcase:
- validate() flow
- dyn @property auto-derivation from stc fields
- ready_for_profile / compile_failed / apply_compile_result
- tensor dict construction
"""

from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.kernel.testcase import TestcaseKernel


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
    n_out = len(output_shapes) if isinstance(output_shapes, (tuple, list)) else 1
    case.input_formats = kwargs.pop("input_formats", ("ND",) * n_in)
    case.input_ori_formats = kwargs.pop("input_ori_formats", ("ND",) * n_in)
    case.output_formats = kwargs.pop("output_formats", ("ND",) * n_out)
    case.output_ori_formats = kwargs.pop("output_ori_formats", ("ND",) * n_out)
    case.input_data_ranges = kwargs.pop("input_data_ranges", (None,) * n_in)
    for k, v in kwargs.items():
        setattr(case, k, v)
    return case


@pytest.fixture(autouse=True)
def _mock_op_info(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


_DEFAULT_OP_INFO = {
    "coreType.value": "AiCore",
    "inputs": [{"name": "x"}, {"name": "y"}],
    "outputs": [{"name": "z"}],
}


def _validate(case, op_info=_DEFAULT_OP_INFO):
    if op_info is not None and "coreType.value" not in op_info:
        op_info = {**op_info, "coreType.value": "AiCore"}
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock:
        mock.return_value.info_of.return_value = op_info
        case.validate()


class TestValidateBasic:
    def test_valid_minimal_case(self):
        case = _make_testcase()
        _validate(case)
        assert case.is_valid
        assert case.fail_reason is None

    def test_missing_op_name(self):
        case = _make_testcase()
        case.op_name = None
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "OP_NAME_MISSING"

    def test_missing_output_shapes(self):
        case = _make_testcase(output_shapes=None)
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "STC_OUTPUT_NOT_SPECIFIED"

    def test_output_shapes_elewise_inference(self):
        case = _make_testcase(output_shapes="ELEWISE")
        _validate(case)
        assert case.is_valid
        assert case.output_shapes == ((8,),)

    def test_invalid_input_dtypes(self):
        case = _make_testcase(input_dtypes=("badtype",))
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "STC_INPUT_DTYPES_INVALID"

    def test_invalid_output_dtypes(self):
        case = _make_testcase(output_dtypes=("badtype",))
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "OUTPUT_DTYPES_INVALID"


class TestDynPropertyDerivation:
    def test_dyn_non_shape_fields_same_as_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_input_dtypes == case.input_dtypes
        assert case.dyn_input_formats == case.input_formats
        assert case.dyn_input_ori_formats == case.input_ori_formats

    def test_is_valid_after_validate(self):
        case = _make_testcase()
        _validate(case)
        assert case.is_valid


class TestReadyForProfile:
    def test_not_ready_initially(self):
        case = _make_testcase()
        assert not case.ready_for_profile()

    def test_ready_after_3_compiles(self):
        case = _make_testcase()
        case.compile_done = 3
        assert case.ready_for_profile()

    def test_not_ready_after_2_compiles(self):
        case = _make_testcase()
        case.compile_done = 2
        assert not case.ready_for_profile()


class TestCompileFailed:
    def _make_result(self, status):
        r = MagicMock()
        r.compile_result = status
        return r

    def test_all_succ(self):
        case = _make_testcase()
        for mode in ("dyn", "cst", "bin"):
            setattr(case, f"{mode}_compile_result", self._make_result("SUCC"))
        assert not case.compile_failed()

    def test_all_fail_means_failed(self):
        # compile_failed() = 所有编译结果非 SUCC（全失败）; 任一 SUCC 即不算失败（有可用编译）
        case = _make_testcase()
        for mode in ("dyn", "cst", "bin"):
            setattr(case, f"{mode}_compile_result", self._make_result("FAIL"))
        assert case.compile_failed() is True

    def test_compile_dynamic_op_success_with_cst(self):
        case = _make_testcase()
        case.dyn_compile_result = self._make_result("FAIL")
        case.cst_compile_result = self._make_result("SUCC")
        case.bin_compile_result = self._make_result("FAIL")
        assert case.compile_dynamic_op_success()


class TestApplyCompileResult:
    def test_apply_dynamic(self):
        from ttk.core_modules.kernel.compile.result import DynamicCompilationResult

        case = _make_testcase()
        result = DynamicCompilationResult()
        result.compile_result = "SUCC"
        result.func_params = ("x", "y")
        result.tiling_result = None
        case.apply_compile_result(result)
        assert case.dyn_compile_result is result
        assert case.compile_done == 1

    def test_apply_const(self):
        from ttk.core_modules.kernel.compile.result import ConstCompilationResult

        case = _make_testcase()
        result = ConstCompilationResult()
        result.compile_result = "SUCC"
        result.func_params = ("x", "y")
        case.apply_compile_result(result)
        assert case.cst_compile_result is result
        assert case.compile_done == 1


class TestTensorDict:
    def test_bin_tensor_dict_with_const(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y", "valueDepend": "required"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=((3, 4), (2, 5)),
            output_shapes=((3, 5),),
        )
        _validate(case, op_info=op_info)
        with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        assert len(inputs) == 2
        assert inputs[0]["shape"] == (-2,)
        assert inputs[1]["shape"] == (-1, -1)

    def test_bin_tensor_dict_none_input(self):
        case = _make_testcase(input_shapes=(None, (3, 4)))
        _validate(case)
        with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = None
            inputs, outputs = case.bin_tensor_dict
        assert inputs[0] is None
        assert inputs[1]["shape"] == (-2,)

    def test_bin_tensor_dict_const_and_tensor_list_mixed(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"}, {"name": "y"}, {"name": "z"}],
            "outputs": [{"name": "out"}],
        }
        case = _make_testcase(
            input_shapes=((2, 3), ((3, 4), (5, 6)), (7, 8)),
            input_dtypes=("float16", ("float16", "float16"), "float16"),
        )
        _validate(case, op_info=op_info)
        with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        assert inputs[0]["shape"] == (-1, -1)
        assert isinstance(inputs[1], tuple)
        assert len(inputs[1]) == 2
        assert inputs[1][0]["shape"] == (-2,)
        assert inputs[2]["shape"] == (-2,)

    def test_bin_tensor_dict_cached(self):
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            output_shapes=((3, 6),),
        )
        _validate(case)
        with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = None
            result1 = case.bin_tensor_dict
            result2 = case.bin_tensor_dict
        assert result1 is result2


class TestConstInputIndexes:
    def test_no_const_indexes_no_change(self):
        case = _make_testcase()
        _validate(case)
        assert case.input_shapes == ((8,), (8,))

    def test_const_indexes_preserves_stc(self):
        case = _make_testcase(
            input_shapes=((1,), (1,), (1,)),
            input_dtypes=("float16", "float16", "float16"),
            output_shapes=((1,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_shapes == ((1,), (1,), (1,))
        assert len(case.input_dtypes) == 3


class TestInputBytes:
    def test_input_bytes_calculation(self):
        case = _make_testcase()  # 默认 2× float16 (8,) = 2*(8*2) = 32 bytes
        _validate(case)
        assert case.input_bytes == 32

    def test_output_bytes_calculation(self):
        case = _make_testcase()  # 默认 1× float16 (8,) = 8*2 = 16 bytes
        _validate(case)
        assert case.output_bytes == 16


class TestGetCompilationHash:
    def test_same_params_same_hash(self):
        case1 = _make_testcase()
        case2 = _make_testcase()
        _validate(case1)
        _validate(case2)
        assert case1.get_compilation_hash() == case2.get_compilation_hash()

    def test_different_params_different_hash(self):
        case1 = _make_testcase()
        case2 = _make_testcase(op_name="Sub")
        _validate(case1)
        _validate(case2)
        assert case1.get_compilation_hash() != case2.get_compilation_hash()


class TestDynamicize:
    """Tests for TestcaseKernel._dynamicize static method.

    每行参数：input_shapes 与 dynamicize 后的期望结果。
    规则：>0 的维度替换为 -1；-1/-2/0 保持；None/空保持；scalar (1,) → (-1,)。
    """

    @pytest.mark.parametrize(
        ("input_shapes", "expected"),
        [
            (((8, 16), (3, 224, 224)), ((-1, -1), (-1, -1, -1))),
            (((-1,), (-1, -1)), ((-1,), (-1, -1))),
            (((0,),), ((0,),)),
            ((None,), (None,)),
            (((8, -1, 0),), ((-1, -1, 0),)),
            (((8,), (16, 32), (-1,)), ((-1,), (-1, -1), (-1,))),
            ((), ()),
        ],
        ids=[
            "positive-dims-become-minus1",
            "already-minus1-stays",
            "zero-stays",
            "none-stays",
            "mixed-dims",
            "multiple-tensors",
            "empty-input",
        ],
    )
    def test_dynamicize(self, input_shapes, expected):
        """验证 _dynamicize 在各类 shape 输入下的替换结果。"""
        assert TestcaseKernel._dynamicize(input_shapes) == expected


class TestNestedShapes:
    def test_flat_backward_compat(self):
        case = _make_testcase()
        _validate(case)
        assert case.tensor_list_distribution == (0, 0, 0)
        assert case.input_shapes == ((8,), (8,))

    def test_nested_inputs_flattened(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_formats=(("ND", "ND"), "ND"),
            input_ori_formats=(("ND", "ND"), "ND"),
            input_ori_shapes=(((3, 4), (5, 4)), (8,)),
        )
        _validate(case)
        assert case.tensor_list_distribution == (2, 0, 0)
        assert case.input_shapes == (((3, 4), (5, 4)), (8,))
        assert case.flat_input_shapes == ((3, 4), (5, 4), (8,))
        assert case.flat_input_dtypes == ("float16", "float16", "float16")

    def test_nested_inputs_leading_none_slot(self):
        case = _make_testcase(
            input_shapes=((None, (3, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_ori_shapes=((None, (3, 4)), (8,)),
        )
        _validate(case)
        # 组内首位 None 槽仍识别为 TensorList，分布/展平元数据与顺序无关
        assert case.tensor_list_distribution == (2, 0, 0)
        assert case.flat_input_shapes == (None, (3, 4), (8,))

    def test_nested_inputs_compressed_dtypes(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float16")
        assert case.flat_input_dtypes == ("float16", "float16", "float16")

    def test_nested_outputs_and_elewise(self):
        case = _make_testcase(
            input_shapes=((6, 4),),
            input_dtypes=("float16",),
            output_shapes=(((3, 4), (3, 4)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.tensor_list_distribution == (0, 2)
        assert case.flat_output_shapes == ((3, 4), (3, 4))

        case2 = _make_testcase(
            input_shapes=(((3, 1), (1, 4)), (3, 4)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes="ELEWISE",
            output_dtypes=("float16",),
        )
        _validate(case2)
        assert case2.output_shapes == ((3, 4),)

        incompatible = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes="ELEWISE",
            output_dtypes=("float16",),
        )
        _validate(incompatible)
        assert not incompatible.is_valid
        assert incompatible.fail_reason == "STC_OUTPUT_INFERENCE_FAILED"
        assert incompatible.output_shapes == "ELEWISE"

    def test_precision_tolerances_and_ranges_broadcast(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        case.precision_tolerances = ((0.01, 0.02),)
        case.input_data_ranges = ((None, 1.0),)
        _validate(case)
        assert case.precision_tolerances == (((0.01, 0.02), (0.01, 0.02)),)
        assert case.input_data_ranges == (((None, 1.0), (None, 1.0)), (None, 1.0))


class TestTensorApiFlatPrecision:
    """Tests for flat_precision_tolerances and flat_absolute_precision on aclnn/e2e base."""

    def test_flat_precision_tolerances_nested(self):
        from ttk.core_modules.common.testcase.api_base import ApiTestcaseBase

        case = ApiTestcaseBase()
        case.tensor_view_shapes = (((3, 4), (5, 4)), (8,))
        case.output_tensor_indexes = (0, 1)
        # output_dist = (2, 0), first entry gets extended by _flatten_by_distribution
        case.precision_tolerances = (((0.01, 0.02), (0.03, 0.04)), (0.05, 0.06))
        case._tensor_list_dist = None
        assert case.flat_precision_tolerances == ((0.01, 0.02), (0.03, 0.04), (0.05, 0.06))

    def test_flat_absolute_precision_single_float(self):
        from ttk.core_modules.common.testcase.api_base import ApiTestcaseBase

        case = ApiTestcaseBase()
        case.absolute_precision = 1e-5
        assert case.flat_absolute_precision == 1e-5
        assert isinstance(case.flat_absolute_precision, float)

    def test_flat_absolute_precision_nested_tuple(self):
        from ttk.core_modules.common.testcase.api_base import ApiTestcaseBase

        case = ApiTestcaseBase()
        case.tensor_view_shapes = (((3, 4), (5, 4)), (8,))
        case.output_tensor_indexes = (0, 1)
        case.absolute_precision = ((1e-5, 1e-6), 1e-7)
        case._tensor_list_dist = None
        assert case.flat_absolute_precision == (1e-5, 1e-6, 1e-7)


# =====================================================================
# TDD tests for flat property refactoring:
# Original fields must be preserved; flat_* properties return flattened.
# These tests FAIL until the refactoring is complete.
# =====================================================================


class TestFlatPropertiesPreserveOriginal:
    """After validate(), original nested fields must be preserved as-is."""

    def test_shapes_and_dtypes_preserved(self):
        nested_shapes = (((3, 4), (5, 4)), (8,))
        nested_dtypes = (("float16", "float16"), "float16")
        case = _make_testcase(input_shapes=nested_shapes, input_dtypes=nested_dtypes)
        _validate(case)
        assert case.input_shapes == nested_shapes
        assert case.input_dtypes == nested_dtypes
        assert case.dyn_input_dtypes == case.input_dtypes

    def test_compressed_dtypes_normalized(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float16")


class TestDynPropertiesUseFlat:
    """flat_dyn_* properties provide flat backward-compatible access."""

    def test_flat_dyn_inputs_and_outputs_from_nested(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.flat_dyn_inputs == ((-1, -1), (-1, -1), (-1,))
        assert case.flat_dyn_input_dtypes == ("float16", "float16", "float32")
        assert case.flat_dyn_outputs == ((-1,), (-1,))

    def test_input_bytes_uses_flat(self):
        """input_bytes correctly computes from flat inputs."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        assert case.input_bytes > 0

    def test_output_bytes_uses_flat(self):
        """output_bytes correctly computes from flat outputs."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.output_bytes > 0


class TestSplitDistribution:
    """Input and output distributions are computed separately."""

    def test_input_distribution_nested(self):
        """input_distribution is inferred from nested input_shapes."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        assert case.input_distribution == (2, 0)

    def test_output_distribution_nested(self):
        """output_distribution is inferred from nested output_shapes."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.output_distribution == (2,)

    def test_combined_distribution(self):
        """tensor_list_distribution = input_dist + output_dist."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.tensor_list_distribution == (2, 0, 2)

    def test_flat_inputs_no_distribution(self):
        """Flat inputs produce (0, 0) input_distribution."""
        case = _make_testcase(input_shapes=((8,), (8,)))
        _validate(case)
        assert case.input_distribution == (0, 0)

    def test_no_nested_outputs_no_distribution(self):
        """Flat outputs produce (0,...) output_distribution matching len."""
        case = _make_testcase(output_shapes=((8,),))
        _validate(case)
        assert case.output_distribution == (0,)


def _mock_compile_result(has_kernel_json=True, clear_atomic=False):
    r = MagicMock()
    r.compile_result = "SUCC"
    r.kernel_name = "test_kernel"
    r.kernel_dir = "/tmp"
    r.tiling_key = 0
    r.block_dim = 1
    r.func_params = ("x",)
    r.tiling_result = MagicMock()
    r.tiling_result.tiling_time = (1.0,)
    r.tiling_result.all_set = MagicMock()
    if has_kernel_json:
        r.kernel_json_info = MagicMock()
        r.kernel_json_info.clear_atomic = clear_atomic
        r.kernel_json_info.optional_input_gen_placeholder.return_value = True
        r.kernel_json_info.optional_output_gen_placeholder.return_value = True
        r.kernel_json_info.dynamic_param_is_folded.return_value = False
        r.kernel_json_info.parameters = []
        r.kernel_json_info.workspaces = ()
        r.kernel_json_info.global_workspace_size = 0
        r.kernel_json_info.oom_enabled.return_value = False
    else:
        r.kernel_json_info = None
    r.workspaces = ()
    r.compile_time = 0.1
    return r


class TestClearAtomicProperties:
    def test_dyn_clear_atomic_from_compile_result(self):
        case = _make_testcase()
        case.dyn_compile_result = _mock_compile_result(clear_atomic=True)
        assert case.dyn_clear_atomic is True

    def test_dyn_clear_atomic_force_ops(self):
        case = _make_testcase(op_name="ones_like")
        case.dyn_compile_result = _mock_compile_result(clear_atomic=False)
        assert case.dyn_clear_atomic is True

    def test_clear_atomic_no_compile_result(self):
        case = _make_testcase()
        assert case.dyn_clear_atomic is False
        assert case.cst_clear_atomic is False
        assert case.bin_clear_atomic is False


class TestShapeInference:
    """Tests for _do_shape_inference.

    分两组：
    - test_inference_returns_value: 各类 REDUCE/ELEWISE 推导返回非 None 结果
      （ELEWISE 还断言精确值 ((8,),)；其余仅断言非 None）
    - test_inference_raises: None 输入或非法推导类型 → 抛 ValueError
    """

    _EXACT = object()  # sentinel: 精确匹配 expected；否则仅断言非 None

    @pytest.mark.parametrize(
        ("input_shapes", "output_shapes", "attributes", "expected"),
        [
            (((8,), (8,)), "ELEWISE", {}, ((8,),)),
            (((8, 16),), "REDUCE", {"axis": (1,)}, _EXACT),
            (((8, 16, 32),), "REDUCE", {"axes": (1, 2)}, _EXACT),
            (((8, 16),), "REDUCE", {}, _EXACT),
            (((8,), (8,)), "ELEWISE(1, None)", {}, _EXACT),
        ],
        ids=["elewise", "reduce-with-axis", "reduce-with-axes", "reduce-no-axis", "elewise-with-args"],
    )
    def test_inference_returns_value(self, input_shapes, output_shapes, attributes, expected):
        """验证 ELEWISE/REDUCE 各场景推导出非 None 结果（ELEWISE 还检查精确值）。"""
        result = TestcaseKernel._do_shape_inference(input_shapes, output_shapes, attributes)
        if expected is self._EXACT:
            assert result is not None
        else:
            assert result == expected

    @pytest.mark.parametrize(
        ("input_shapes", "output_shapes", "match"),
        [
            ((None,), "ELEWISE", "None input"),
            (((8,),), "INVALID_TYPE", "Invalid"),
        ],
        ids=["none-in-inputs-raises", "invalid-inference-raises"],
    )
    def test_inference_raises(self, input_shapes, output_shapes, match):
        """验证 None 输入或非法类型 → ValueError。"""
        with pytest.raises(ValueError, match=match):
            TestcaseKernel._do_shape_inference(input_shapes, output_shapes, {})


class TestExpandIndices:
    @pytest.mark.parametrize(
        ("count", "distribution", "indices", "expected"),
        [
            (3, (), [0, 1, 2], [0, 1, 2]),
            (3, (), [0, None, 2], [0, None, 2]),
        ],
        ids=["simple-expand", "with-none"],
    )
    def test_expand_indices_returns_list(self, count, distribution, indices, expected):
        """验证无 distribution 时 _expand_indices 原样返回列表。"""
        assert TestcaseKernel._expand_indices(count, distribution, indices) == expected

    def test_expand_indices_with_list_distribution(self):
        """带 distribution 时 _expand_indices 返回长度符合 TensorList 展开。"""
        result = TestcaseKernel._expand_indices(5, (2, 0, 0), [0, 1])
        assert len(result) == 3


class TestAutoSetInplaceIndexes:
    def test_no_inplace(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.output_inplace_indexes is None or case.output_inplace_indexes == ()

    def test_with_inplace(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "paramType": "default"}, {"name": "y", "paramType": "default"}],
            "outputs": [{"name": "x", "paramType": "default"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.output_inplace_indexes == (0,)

    def test_param_type_mismatch(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "paramType": "dynamic"}, {"name": "y", "paramType": "default"}],
            "outputs": [{"name": "x", "paramType": "default"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.fail_reason == "OUTPUT_INPLACE_INDEXES_ERROR"


class TestConstInputIndexesOpInfo:
    def test_value_depend_required(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = op_info
            assert case.const_input_indexes == (0,)

    def test_no_value_depend(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "ignore"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = op_info
            assert case.const_input_indexes == ()


class TestSupportedRerunTitle:
    def test_returns_expected_titles(self):
        titles = TestcaseKernel.supported_rerun_title()
        assert "dyn_perf_us" in titles
        assert "precision_status" in titles


class TestSetCaseCoreType:
    def test_core_type_from_global_storage(self):
        case = _make_testcase()
        _validate(case)
        assert case.core_type == "AiCore"

    def test_core_type_from_op_info(self):
        op_info = {
            "coreType.value": "VectorCore",
            "inputs": [{"name": "x"}],
            "outputs": [{"name": "y"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.core_type == "VectorCore"

    def test_core_type_default_when_no_op_info(self):
        case = _make_testcase()
        _validate(case, op_info=None)
        assert case.core_type == "AiCore"


class TestStcShapeSizeCheck:
    def test_shape_out_of_bound(self):
        case = _make_testcase(input_shapes=((999999999,),), input_dtypes=("float32",))
        mock_gs = MagicMock()
        mock_gs.core_type = None
        mock_gs.hbm_size_limit = 0
        mock_gs.op_impl_mode = None
        mock_gs.kernel_meta = "/tmp"
        mock_gs.short_soc_version = "Ascend910B2"
        with patch("ttk.core_modules.kernel.testcase.get_global_storage", return_value=mock_gs), patch(
            "ttk.core_modules.kernel.info_keeper.OpInfoKeeper"
        ) as mock_oi:
            mock_oi.return_value.info_of.return_value = {
                "coreType.value": "AiCore",
                "inputs": [{"name": "x"}],
                "outputs": [{"name": "z"}],
            }
            case.validate()
        assert not case.is_valid
        assert case.fail_reason == "STC_SHAPE_OUT_OF_BOUND"


class TestGetDynFuncParamName:
    def test_basic(self):
        case = _make_testcase()
        case.dyn_func_params = ("x", "y", "axis")
        result = case.get_dyn_func_param_name(2)
        assert result == "axis"

    def test_with_list_distribution(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        case.dyn_func_params = ("inputs", "axis")
        result = case.get_dyn_func_param_name(1)
        assert result == "inputs"
