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
Tests for ttk.core_modules.framework_api.testcase:
nested tensor structure, flat properties, compression, normalize, and validate.
"""

import copy

import numpy
import pytest

from ttk.core_modules.framework_api.api_model import APIParamInfo, ParamInfo
from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.core_modules.framework_api.torch.param_extractor import _MANUAL_OVERRIDES

NESTED_SHAPES = (((3, 3), (3, 2)), (3, 5))
FLAT_SHAPES = ((3, 3), (3, 5))
TENSORLIST_SHAPES = (((2, 3), (2, 5), (2, 1)),)


def _make(api_name="torch.dummy", **kwargs):
    case = TestcaseE2e()
    case.api_name = api_name
    case.is_valid = True
    case.fail_reason = None
    case.attributes = kwargs.pop("attributes", {})
    for k, v in kwargs.items():
        setattr(case, k, v)
    return case


@pytest.fixture
def make_testcase():
    """Local override: TestcaseE2e (overrides conftest's TestcaseAclnn)."""
    return _make


class TestFlatTensorDtypesCompression:
    def test_compressed_broadcast_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=TENSORLIST_SHAPES, tensor_dtypes=("float32",))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")
        assert all(isinstance(x, str) for x in case.flat_tensor_dtypes)

    def test_per_param_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_dtypes=("float32", "int8"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "int8")

    def test_fully_nested(self, make_testcase):
        case = make_testcase(tensor_view_shapes=TENSORLIST_SHAPES, tensor_dtypes=(("float32", "float16", "int8"),))
        assert case.flat_tensor_dtypes == ("float32", "float16", "int8")

    def test_flat_no_nesting(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES, tensor_dtypes=("float32", "float32"))
        assert case.flat_tensor_dtypes == ("float32", "float32")


class TestPureOutputIndexes:
    def test_flat_output(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((2, 3), (2, 3), (2, 3)), output_tensor_indexes=(2,))
        assert case.pure_output_indexes == [2]

    def test_tensorlist_output(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, output_tensor_indexes=(0,))
        assert case.pure_output_indexes == [0, 1]

    def test_multi_output(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((2, 3), (2, 3), (2, 3), (2, 3)), output_tensor_indexes=(1, 3))
        assert case.pure_output_indexes == [1, 3]


class TestFlatOutputDtypes:
    def test_flat_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32", "float16", "int8"),
            output_tensor_indexes=(2,),
        )
        assert case.flat_output_dtypes == ("int8",)

    def test_tensorlist_output_expanded(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "int8"), "float16"),
            output_tensor_indexes=(0,),
        )
        assert case.flat_output_dtypes == ("float32", "int8")

    def test_no_output_indexes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES, tensor_dtypes=("float32", "float32"))
        assert case.flat_output_dtypes == ()

    def test_out_of_range_index_skipped(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=("float32",),
            output_tensor_indexes=(5,),
        )
        assert case.flat_output_dtypes == ()

    def test_none_shape_output_slots_fall_back_to_declaration(self, make_testcase):
        """onnx 输出槽 None-shape 槽视图为 None → 逐一回落 CSV 声明 dtype（声明序
        对齐，Split 多输出形态同适用）；对照真实 shape 输出槽不受回落影响。"""
        case = make_testcase(
            api_name="onnx.Add",
            tensor_view_shapes=((128, 256), (128, 256), None),
            tensor_dtypes=("float16", "float16", "float16"),
            output_tensor_indexes=(2,),
        )
        _ = case.tensor_slots  # 生产访问序：槽表先物化（None-shape 槽 → 槽视图 dtype None）
        assert case.flat_tensor_dtypes == ("float16", "float16", None)
        assert case.flat_output_dtypes == ("float16",)

        case = make_testcase(
            api_name="onnx.Split",
            tensor_view_shapes=((6, 4), None, None, None),
            tensor_dtypes=("float32", "float32", "float32", "float32"),
            output_tensor_indexes=(1, 2, 3),
        )
        _ = case.tensor_slots
        assert case.flat_output_dtypes == ("float32", "float32", "float32")

        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float16", "int32"),
            output_tensor_indexes=(1,),
        )
        _ = case.tensor_slots
        assert case.flat_output_dtypes == ("int32",)

    def test_feeds_unpack_4bit_outputs(self, make_testcase):
        pytest.importorskip("ml_dtypes")
        from ttk.core_modules.framework_api.utils import unpack_4bit_outputs

        case = make_testcase(
            tensor_view_shapes=((2, 2), (2, 2)),
            tensor_dtypes=("float32", "int4"),
            output_tensor_indexes=(1,),
        )
        packed = numpy.array([[0x21], [0x43]], dtype=numpy.uint8)
        result = unpack_4bit_outputs(case, [packed])
        assert str(result[0].dtype) == "int4"
        numpy.testing.assert_array_equal(numpy.asarray(result[0], dtype=numpy.int32), [1, 2, 3, 4])


class TestMultiDeviceRankIsolation:
    def test_rank_generation_on_template_copy_keeps_shared_state(self, make_testcase):
        """worker 多卡循环同款隔离模式：per-rank 生成走声明态模板深拷贝。"""
        case = make_testcase(
            tensor_view_shapes=FLAT_SHAPES,
            tensor_dtypes=("float32", "float32"),
        )
        template = copy.deepcopy(case)
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 5))]
        case.tensors = [numpy.ones((3, 3)), numpy.ones((3, 5))]
        shared_first = case.np_storages[0]

        rank = copy.deepcopy(template)
        rank.np_storages = [numpy.zeros((3, 3)), numpy.zeros((3, 5))]
        rank.tensors = [numpy.zeros((3, 3)), numpy.zeros((3, 5))]

        assert case.np_storages[0] is shared_first
        numpy.testing.assert_array_equal(case.np_storages[0], numpy.ones((3, 3)))
        numpy.testing.assert_array_equal(rank.np_storages[0], numpy.zeros((3, 3)))


class TestValidateFrameworkApi:
    @pytest.fixture(autouse=True)
    def mock_api_info(self):
        from unittest.mock import patch

        info = APIParamInfo(
            api_name="torch.dummy",
            overloads=[
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="tensors", type="List[Tensor]")],
            ],
        )
        with patch.object(TestcaseE2e, "get_api_info", return_value=info):
            yield

    def test_valid_flat(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32"))
        case.validate()
        assert case.is_valid

    def test_valid_nested_compressed(self, make_testcase):
        case = make_testcase(tensor_view_shapes=TENSORLIST_SHAPES, tensor_dtypes=("float32",))
        case.validate()
        assert case.is_valid
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_missing_api_name(self, make_testcase):
        case = make_testcase(api_name="", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",))
        case.validate()
        assert not case.is_valid

    def test_missing_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=(), tensor_dtypes=("float32",))
        case.validate()
        assert not case.is_valid

    def test_output_index_out_of_range(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32"), output_tensor_indexes=(5,)
        )
        case.validate()
        assert not case.is_valid


class TestCheckTensorConfiguration:
    """Tests for _check_tensor_configuration — hard validation against API signature."""

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def _register_api(self, api_name, params):
        FrameworkApiInfoKeeper().register(api_name, params, source="test")

    def test_tensor_count_mismatch_too_many_invalidates(self, make_testcase):
        self._register_api(
            "torch.test_count",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        case = make_testcase(
            api_name="torch.test_count",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32", "float32"),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "INPUT_COUNT_EXCEEDED"

    def test_tensor_count_too_few_required_invalidates(self, make_testcase):
        self._register_api(
            "torch.test_req",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
            ],
        )
        case = make_testcase(api_name="torch.test_req", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "TENSOR_COUNT_MISMATCH"

    def test_optional_out_omitted_and_provided_valid(self, make_testcase):
        self._register_api(
            "torch.test_opt",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
            ],
        )
        case_omit = make_testcase(
            api_name="torch.test_opt", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case_omit.validate()
        assert case_omit.is_valid

        case_provide = make_testcase(
            api_name="torch.test_opt",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32", "float32"),
            output_tensor_indexes=(2,),
        )
        case_provide.validate()
        assert case_provide.is_valid

    def test_tensorlist_type_mismatch_and_match(self, make_testcase):
        self._register_api(
            "torch.test_tl",
            [
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="dim", type="int"),
            ],
        )
        case_bad = make_testcase(api_name="torch.test_tl", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",))
        case_bad.validate()
        assert not case_bad.is_valid
        assert case_bad.fail_reason == "PARAM_TYPE_MISMATCH"

        self._register_api(
            "torch.test_tl_ok",
            [
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
            ],
        )
        case_ok = make_testcase(
            api_name="torch.test_tl_ok",
            tensor_view_shapes=(((2, 3), (2, 5)), (3, 3)),
            tensor_dtypes=("float32", "float32"),
            output_tensor_indexes=(1,),
        )
        case_ok.validate()
        assert case_ok.is_valid

    def test_single_tensor_nested_for_tensor_param_invalid(self, make_testcase):
        self._register_api(
            "torch.test_nested_mismatch",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        case = make_testcase(
            api_name="torch.test_nested_mismatch",
            tensor_view_shapes=(((2, 3), (2, 5)), (3, 3)),
            tensor_dtypes=("float32", "float32"),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "PARAM_TYPE_MISMATCH"

    def test_api_not_found_skips_validation(self, make_testcase):
        case = make_testcase(
            api_name="nonexistent.api", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "API_PARSE_FAIL"

    def test_onnx_attr_validation(self, make_testcase):
        """onnx 域属性校验：未知属性校验期拒绝；声明属性放行。"""
        self._register_api(
            "onnx.TestAttr",
            [
                ParamInfo(name="X", type="Tensor"),
                ParamInfo(name="Y", type="Tensor"),
                ParamInfo(name="alpha", type="float", is_keyword_only=True, is_optional=True),
            ],
        )
        case = make_testcase(
            api_name="onnx.TestAttr",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
            attributes={"alphaX": 1.0},
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "UNKNOWN_ATTRIBUTE"

        self._register_api(
            "onnx.TestAttrOk",
            [
                ParamInfo(name="X", type="Tensor"),
                ParamInfo(name="Y", type="Tensor"),
                ParamInfo(name="alpha", type="float", is_keyword_only=True, is_optional=True),
            ],
        )
        case = make_testcase(
            api_name="onnx.TestAttrOk",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
            attributes={"alpha": 1.5},
        )
        case.validate()
        assert case.is_valid

    def test_onnx_dynamic_input_gate(self, make_testcase):
        """onnx 域 -1 动态槽校验期拒绝（不得落通用 npu_preprocess 门）；
        静态 shape + None 输出槽声明不误伤。"""
        self._register_api(
            "onnx.TestDyn",
            [
                ParamInfo(name="X", type="Tensor"),
                ParamInfo(name="Y", type="Tensor"),
            ],
        )
        case = make_testcase(
            api_name="onnx.TestDyn",
            tensor_view_shapes=((-1, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "ONNX_DYNAMIC_INPUT_UNSUPPORTED"

        case = make_testcase(
            api_name="onnx.TestDyn",
            tensor_view_shapes=((2, 3), (2, 3), None),
            tensor_dtypes=("float32", "float32", "float32"),
            output_tensor_indexes=(2,),
        )
        case.validate()
        assert case.is_valid

    def test_onnx_dynamic_in_nested_tensorlist_rejected_by_generic_gate(self, make_testcase):
        """TensorList 内 -1 由通用校验层先行拒绝（DYN_SHAPE_IN_TENSORLIST，
        全域语义非 onnx 特有）；onnx 门禁只接管普通输入位的 -1 残余场景。"""
        self._register_api(
            "onnx.TestDynTl",
            [ParamInfo(name="tensors", type="tuple of Tensors")],
        )
        case = make_testcase(
            api_name="onnx.TestDynTl",
            tensor_view_shapes=(((-1, 3), (3, 4)),),
            tensor_dtypes=("float32",),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "DYN_SHAPE_IN_TENSORLIST"

    def test_torch_domain_unaffected_by_onnx_gates(self, make_testcase):
        """torch 域对照：-1 槽不受 onnx 门禁影响（npu_preprocess 通道保留）；
        extra_attrs 有插件 kwargs 消费方，未知属性不做校验期拒绝。"""
        self._register_api(
            "torch.test_dyn_torch",
            [
                ParamInfo(name="X", type="Tensor"),
                ParamInfo(name="Y", type="Tensor"),
            ],
        )
        case = make_testcase(
            api_name="torch.test_dyn_torch",
            tensor_view_shapes=((-1, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
        )
        case.validate()
        assert case.is_valid

        self._register_api(
            "torch.test_extra_attr",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        case = make_testcase(
            api_name="torch.test_extra_attr",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
            attributes={"plugin_kwarg": 1.0},
        )
        case.validate()
        assert case.is_valid

    def test_none_element_for_tensorlist_skipped(self, make_testcase):
        self._register_api(
            "torch.test_none_tl",
            [
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
            ],
        )
        case = make_testcase(
            api_name="torch.test_none_tl",
            tensor_view_shapes=(None, (3, 3)),
            tensor_dtypes=("float32", "float32"),
            output_tensor_indexes=(1,),
        )
        case.validate()
        assert case.is_valid

    def test_live_torch_add_count_valid_and_invalid(self, make_testcase):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.add", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case.validate()
        assert case.is_valid

        case_bad = make_testcase(api_name="torch.add", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",))
        case_bad.validate()
        assert not case_bad.is_valid
        assert case_bad.fail_reason == "TENSOR_COUNT_MISMATCH"


class TestCheckMultiOverload:
    """Tests for multi-overload matching in _check_tensor_configuration."""

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_multi_overload_matches_second(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.test_multi",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="input", type="Tensor")],
            ],
            source="test",
        )
        case = make_testcase(api_name="torch.test_multi", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",))
        case.validate()
        assert case.is_valid

    def test_multi_overload_matches_first(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.test_multi2",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="input", type="Tensor")],
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.test_multi2", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case.validate()
        assert case.is_valid

    def test_multi_overload_no_match(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.test_multi3",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="input", type="Tensor")],
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.test_multi3",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32", "float32"),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "INPUT_COUNT_EXCEEDED"

    def test_multi_overload_tensorlist_vs_tensor(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.test_multi_tl",
            [
                [ParamInfo(name="tensors", type="tuple of Tensors")],
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            ],
            source="test",
        )
        case_nested = make_testcase(
            api_name="torch.test_multi_tl", tensor_view_shapes=(((2, 3), (2, 5)),), tensor_dtypes=("float32",)
        )
        case_nested.validate()
        assert case_nested.is_valid

        case_flat = make_testcase(
            api_name="torch.test_multi_tl", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case_flat.validate()
        assert case_flat.is_valid


class TestOutputTensorExcludesFromInputCount:
    """Tests for _check_tensor_configuration excluding output_tensor_indexes."""

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_all_tensors_marked_output_invalidates(self, make_testcase):
        """1 tensor with output_tensor_indexes=(0,) → 0 input tensors → invalid."""
        FrameworkApiInfoKeeper().register(
            "torch.abs",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.abs", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",), output_tensor_indexes=(0,)
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "ALL_TENSORS_MARKED_OUTPUT"

    def test_output_tensor_excluded_from_count(self, make_testcase):
        """2 tensors, 1 output → 1 input tensor, API needs 1 input → valid."""
        FrameworkApiInfoKeeper().register(
            "torch.abs_valid",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.abs_valid",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
            output_tensor_indexes=(1,),
        )
        case.validate()
        assert case.is_valid

    def test_two_outputs_one_input_valid(self, make_testcase):
        """3 tensors, 2 outputs → 1 input. API with 1 required tensor + 2 optional → valid."""
        FrameworkApiInfoKeeper().register(
            "torch.sort_like",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.sort_like",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32", "float32"),
            output_tensor_indexes=(1, 2),
        )
        case.validate()
        assert case.is_valid

    def test_live_torch_abs_all_output_invalid(self, make_testcase):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.abs", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",), output_tensor_indexes=(0,)
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "ALL_TENSORS_MARKED_OUTPUT"


class TestInplaceTensorMethod:
    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_is_inplace_tensor_method_static(self):
        assert TestcaseE2e._is_inplace_tensor_method("torch.Tensor.add_")
        assert TestcaseE2e._is_inplace_tensor_method("torch.Tensor.relu_")
        assert not TestcaseE2e._is_inplace_tensor_method("torch.Tensor.add")
        assert not TestcaseE2e._is_inplace_tensor_method("torch.add")
        assert not TestcaseE2e._is_inplace_tensor_method("")

    def test_auto_fill_output_for_inplace(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.Tensor.fake_add_",
            [
                ParamInfo(name="self", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
                ParamInfo(name="alpha", type="Number", default="1"),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.Tensor.fake_add_",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
            attributes={"alpha": "2"},
        )
        case.validate()
        assert case.is_valid
        assert case.output_tensor_indexes == (0,)

    def test_inplace_autofill_before_normalize(self, make_testcase):
        """auto-fill 先于 normalize：odist 以补位后的输出位计算，容差按输出展开。"""
        FrameworkApiInfoKeeper().register(
            "torch.Tensor.fake_stack_mul_",
            [
                ParamInfo(name="self", type="List[Tensor]"),
                ParamInfo(name="other", type="Tensor"),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.Tensor.fake_stack_mul_",
            tensor_view_shapes=(((2, 3), (2, 3)), (2, 3)),
            tensor_dtypes=(("float32", "float32"), "float32"),
            precision_tolerances=("1e-3",),
        )
        case.validate()
        assert case.is_valid, case.fail_reason
        assert case.output_tensor_indexes == (0,)
        assert case.output_dist == (2,)
        assert case.flat_precision_tolerances == ("1e-3", "1e-3")

    def test_inplace_self_not_excluded_from_input_count(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.Tensor.fake_mul_",
            [
                ParamInfo(name="self", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.Tensor.fake_mul_", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case.validate()
        assert case.is_valid

    def test_inplace_pure_output_excludes_self(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.Tensor.fake_sub_",
            [
                ParamInfo(name="self", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.Tensor.fake_sub_", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case.validate()
        assert case.pure_output_indexes == []

    def test_non_inplace_not_auto_filled(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.Tensor.fake_add",
            [
                ParamInfo(name="self", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.Tensor.fake_add", tensor_view_shapes=((2, 3), (2, 3)), tensor_dtypes=("float32", "float32")
        )
        case.validate()
        assert case.output_tensor_indexes is None or case.output_tensor_indexes == ()


class TestOutputConfigurationValidation:
    def test_required_out_missing_fails(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch_npu.test_req_out",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", is_keyword_only=True),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch_npu.test_req_out", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",)
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "MISSING_REQUIRED_OUTPUT"

    def test_required_tensor_list_out_wrong_and_correct_count(self, make_testcase):
        info = APIParamInfo(
            api_name="torch_npu.test_tl_out",
            overloads=[
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
                ]
            ],
            _return_counts=[4],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch_npu.test_tl_out", info)
        case_bad = make_testcase(
            api_name="torch_npu.test_tl_out",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32", "float32"),
            output_tensor_indexes=(1, 2),
        )
        case_bad.validate()
        assert not case_bad.is_valid
        assert case_bad.fail_reason == "OUTPUT_COUNT_MISMATCH"

        case_ok = make_testcase(
            api_name="torch_npu.test_tl_out",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3), (2, 3), (2, 3)),
            tensor_dtypes=("float32",) * 5,
            output_tensor_indexes=(1, 2, 3, 4),
        )
        case_ok.validate()
        assert case_ok.is_valid

    def test_optional_out_passes_with_and_without_output(self, make_testcase):
        FrameworkApiInfoKeeper().register(
            "torch.opt_out",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", is_optional=True, is_keyword_only=True),
            ],
            source="test",
        )
        case_no_out = make_testcase(api_name="torch.opt_out", tensor_view_shapes=((2, 3),), tensor_dtypes=("float32",))
        case_no_out.validate()
        assert case_no_out.is_valid

        case_with_out = make_testcase(
            api_name="torch.opt_out",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
            output_tensor_indexes=(1,),
        )
        case_with_out.validate()
        assert case_with_out.is_valid


class TestOutputConfigUnknownTensorListCount:
    """Test _check_output_configuration when out_expected_count=0 with is_tensor_list=True.

    This happens for APIs parsed from TypeError multi-overload (e.g. torch.sort)
    where we know out is TensorList but don't know the exact count.
    """

    def test_optional_tensor_list_unknown_count_no_out_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like",
            overloads=[
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="out", type="tuple of Tensors", is_optional=True, is_keyword_only=True),
                ]
            ],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like", info)
        case = make_testcase(api_name="torch.sort_like", tensor_view_shapes=((4, 4),), tensor_dtypes=("float32",))
        case.validate()
        assert case.is_valid

    def test_optional_tensor_list_unknown_count_with_out_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like2",
            overloads=[
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="out", type="tuple of Tensors", is_optional=True, is_keyword_only=True),
                ]
            ],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like2", info)
        case = make_testcase(
            api_name="torch.sort_like2",
            tensor_view_shapes=((4, 4), (4, 4), (4, 4)),
            tensor_dtypes=("float32", "float32", "float32"),
            output_tensor_indexes=(1, 2),
        )
        case.validate()
        assert case.is_valid

    def test_required_tensor_list_unknown_count_with_out_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like3",
            overloads=[
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="out", type="tuple of Tensors", is_keyword_only=True),
                ]
            ],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like3", info)
        case = make_testcase(
            api_name="torch.sort_like3",
            tensor_view_shapes=((4, 4), (4, 4)),
            tensor_dtypes=("float32", "float32"),
            output_tensor_indexes=(1,),
        )
        case.validate()
        assert case.is_valid

    def test_required_tensor_list_unknown_count_no_out_fails(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like4",
            overloads=[
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="out", type="tuple of Tensors", is_keyword_only=True),
                ]
            ],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like4", info)
        case = make_testcase(api_name="torch.sort_like4", tensor_view_shapes=((4, 4),), tensor_dtypes=("float32",))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "MISSING_REQUIRED_OUTPUT"

    def test_known_count_exact_validation_still_works(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.exact_tl",
            overloads=[
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="out", type="tuple of Tensors", is_keyword_only=True),
                ]
            ],
            _return_counts=[2],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.exact_tl", info)
        case = make_testcase(
            api_name="torch.exact_tl",
            tensor_view_shapes=((4, 4), (4, 4)),
            tensor_dtypes=("float32", "float32"),
            output_tensor_indexes=(1,),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "OUTPUT_COUNT_MISMATCH"


class TestRuntimeInjectedParamsExemption:
    """hcom/world_size 等运行时注入位的必选属性豁免。

    npu._schemas 权威签名里注入位是必选参数，但其值由多卡 worker 在通信域
    初始化后物化（非 CSV 声明）——校验层不得按 MISSING_REQUIRED_ATTR 误报。
    """

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_injected_params_exempt_from_required_attrs(self, make_testcase):
        """必选 hcom/world_size 不在 attributes → 校验通过（值源在通信域）。"""
        FrameworkApiInfoKeeper().register(
            "torch.test_hccl_inject",
            [
                ParamInfo(name="x1", type="Tensor"),
                ParamInfo(name="x2", type="Tensor"),
                ParamInfo(name="hcom", type="str"),
                ParamInfo(name="world_size", type="int"),
                ParamInfo(name="reduce_op", type="str", is_optional=True, default="sum", is_keyword_only=True),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.test_hccl_inject",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
        )
        case.validate()
        assert case.is_valid, f"注入位误报: {case.fail_reason}"

    def test_non_injected_required_param_still_enforced(self, make_testcase):
        """豁免不外溢：非注入位的必选参数缺失仍报 MISSING_REQUIRED_ATTR。"""
        FrameworkApiInfoKeeper().register(
            "torch.test_hccl_strict",
            [
                ParamInfo(name="x1", type="Tensor"),
                ParamInfo(name="x2", type="Tensor"),
                ParamInfo(name="hcom", type="str"),
                ParamInfo(name="reduce_op", type="str"),
            ],
            source="test",
        )
        case = make_testcase(
            api_name="torch.test_hccl_strict",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=("float32", "float32"),
        )
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "MISSING_REQUIRED_ATTR"
