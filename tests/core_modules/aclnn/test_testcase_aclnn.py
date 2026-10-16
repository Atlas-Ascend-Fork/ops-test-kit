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
Tests for ttk.core_modules.aclnn.testcase: nested tensor structure,
flat properties, compression, and parameter validation.
"""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ttk.core_modules.aclnn.info_keeper import OpApiInfo
from ttk.core_modules.aclnn.testcase import TestcaseAclnn

NESTED_SHAPES = (((3, 3), (3, 2)), (3, 5))
FLAT_SHAPES = ((3, 3), (3, 5))


class TestGetTensorListDistribution:
    """Tests for _get_tensor_list_distribution.

    每行参数：tensor_view_shapes 设置值与对应的 tensor_list_dist 期望。
    """

    @pytest.mark.parametrize(
        ("tensor_view_shapes", "expected"),
        [
            (FLAT_SHAPES, (0, 0)),
            (NESTED_SHAPES, (2, 0)),
            ((((1,), (2,)), ((3,),)), (2, 1)),
            ((((2, 3), (2, 5), (2, 1)),), (3,)),
            (None, ()),
        ],
        ids=["flat-shapes", "nested-shapes", "all-nested", "tensorlist-only", "none"],
    )
    def test_tensor_list_dist(self, make_testcase, tensor_view_shapes, expected):
        """验证 tensor_view_shapes 推导出的 tensor_list_dist。"""
        case = make_testcase(tensor_view_shapes=tensor_view_shapes)
        assert case.tensor_list_dist == expected


class TestFlatTensorViewShapes:
    """Tests for flat_tensor_view_shapes property.

    每行参数：tensor_view_shapes 输入与 flat_tensor_view_shapes 期望输出。
    """

    @pytest.mark.parametrize(
        ("tensor_view_shapes", "expected"),
        [
            (FLAT_SHAPES, FLAT_SHAPES),
            (NESTED_SHAPES, ((3, 3), (3, 2), (3, 5))),
            ((((2, 3), (2, 5), (2, 1)),), ((2, 3), (2, 5), (2, 1))),
            (None, None),
            ((), ()),
        ],
        ids=["flat", "nested", "tensorlist-only", "none", "empty"],
    )
    def test_flat_tensor_view_shapes(self, make_testcase, tensor_view_shapes, expected):
        """验证 flat_tensor_view_shapes 在 flat/nested/None/empty 下的返回值。"""
        case = make_testcase(tensor_view_shapes=tensor_view_shapes)
        assert case.flat_tensor_view_shapes == expected


class TestFlatTensorDtypesCompression:
    """Tests for flat_tensor_dtypes: flatten nested TensorList structure.

    In production, _normalize_compressed_fields runs first, expanding
    compressed forms like ('float32',) to (('float32','float32'),'float32').
    The flat property then just flattens nesting.
    """

    def test_already_flat(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_dtypes=("float32", "float32", "float32"))
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_top_level_compression_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_dtypes=("float32",))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_per_param_compression_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_dtypes=("float32", "float32"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_per_tensor_list_explicit(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_dtypes=(("float32", "int8"), "float32"))
        assert case.flat_tensor_dtypes == ("float32", "int8", "float32")

    def test_per_tensor_list_broadcast_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_dtypes=(("float32",), "float32"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_flat_no_nesting(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES, tensor_dtypes=("float32", "float32"))
        assert case.flat_tensor_dtypes == ("float32", "float32")

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), tensor_dtypes=None)
        assert case.flat_tensor_dtypes is None


class TestFlatTensorViewOffsetsCompression:
    """Tests for flat_tensor_view_offsets: flatten nested TensorList structure.

    After _normalize_compressed_fields, offsets are fully expanded; the flat
    property just needs to flatten nesting.  Without normalization the raw
    compressed form is preserved (broadcast is normalize's job).
    """

    def test_nested_offsets_flattened(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_view_offsets=((5, 10), 3))
        assert case.flat_tensor_view_offsets == (5, 10, 3)

    def test_flat_offsets_passthrough(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES, tensor_view_offsets=(0, 5))
        assert case.flat_tensor_view_offsets == (0, 5)

    def test_compressed_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_view_offsets=(0,))
        case._normalize_compressed_fields()
        assert case.flat_tensor_view_offsets == (0, 0, 0)

    def test_per_param_after_normalize(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, tensor_view_offsets=(5, 10))
        case._normalize_compressed_fields()
        assert case.flat_tensor_view_offsets == (5, 5, 10)

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), tensor_view_offsets=None)
        assert case.flat_tensor_view_offsets is None

    def test_empty(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), tensor_view_offsets=())
        assert case.flat_tensor_view_offsets == ()


class TestIsTensorListElement:
    """Tests for _is_tensor_list_element static method.

    每行参数：value 为待判定值，expected 为是否被识别为 TensorList 元素
    （顶层为 tuple-of-tuple 才算 TensorList）。
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (((3, 3), (3, 2)), True),
            ((3, 5), False),
            (None, False),
            ((), False),
        ],
        ids=["tensor-list", "single-tensor", "none", "empty"],
    )
    def test_is_tensor_list_element(self, value, expected):
        """验证 _is_tensor_list_element 对各类输入的判定结果。"""
        assert TestcaseAclnn._is_tensor_list_element(value) is expected


class TestOutputProperties:
    def test_output_dtypes_tensor_list_and_single(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "int8"), "float32"),
            output_tensor_indexes=(0, 1),
        )
        assert case.output_dtypes == (("float32", "int8"), "float32")
        assert case.flat_output_dtypes == ("float32", "int8", "float32")

    def test_output_view_shapes_tensor_list_and_flat(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES, output_tensor_indexes=(0, 1))
        assert case.output_view_shapes == (((3, 3), (3, 2)), (3, 5))
        assert case.flat_output_view_shapes == ((3, 3), (3, 2), (3, 5))

    def test_flat_output_storage_shapes(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_storage_shapes=(((10, 10), (10, 10)), (3, 5)),
            output_tensor_indexes=(0,),
        )
        assert case.flat_output_storage_shapes == ((10, 10), (10, 10))

    def test_output_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=None, output_tensor_indexes=())
        assert case.output_dtypes == ()
        assert case.flat_output_dtypes == ()


class TestIsScalarListElement:
    """Tests for _is_scalar_list_element static method.

    每行参数：value 为待判定值，expected 为是否被识别为 ScalarList 元素
    （顶层 tuple-of-tuple 才算 ScalarList）。
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (("float32", "int8"), False),
            ((("float32", "int8"),), True),
            ("float32", False),
            (None, False),
            ((), False),
        ],
        ids=["scalar-list-not-nested", "scalar-list-nested", "single-scalar", "none", "empty"],
    )
    def test_is_scalar_list_element(self, value, expected):
        """验证 _is_scalar_list_element 对各类输入的判定结果。"""
        assert TestcaseAclnn._is_scalar_list_element(value) is expected


class TestNormalizeInputDataRanges:
    def _make_case(self, shapes, ranges=None):
        case = TestcaseAclnn()
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = tuple(
            tuple("float32" for _ in s)
            if isinstance(s, (tuple, list)) and s and isinstance(s[0], (tuple, list))
            else "float32"
            for s in shapes
        )
        if ranges is not None:
            case.input_data_ranges = ranges
        case._normalize_compressed_fields()
        return case

    def test_single_range_broadcast_to_all_params(self):
        case = self._make_case(shapes=((3, 4), ((5, 6), (7, 8))), ranges=((0.0, 1.0),))
        assert case.input_data_ranges == ((0.0, 1.0), ((0.0, 1.0), (0.0, 1.0)))

    def test_tensorlist_range_list_short_padded(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6), (7, 8)), (9, 10)), ranges=(((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0))
        )
        assert case.is_valid is False

    def test_already_nested_unchanged(self):
        case = self._make_case(shapes=(((3, 4), (5, 6)), (7, 8)), ranges=(((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0)))
        assert case.input_data_ranges == (((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0))


class TestNonePlaceholderInStrideOffsetStorage:
    """Tests for None placeholder in tensor_view_strides, tensor_view_offsets, tensor_storage_shapes.

    When a field value is None, the accessor derives from tensor_view_shapes (contiguous).
    """

    def _make_case(self, strides=None, offsets=None, storage_shapes=None):
        case = TestcaseAclnn()
        case.tensor_view_shapes = ((3, 4), (5, 6), (7, 8))
        case.tensor_dtypes = ("float32", "float32", "float32")
        if strides is not None:
            case.tensor_view_strides = strides
        if offsets is not None:
            case.tensor_view_offsets = offsets
        if storage_shapes is not None:
            case.tensor_storage_shapes = storage_shapes
        return case

    def test_stride_none_falls_back_to_contiguous(self):
        case = self._make_case(strides=(None, (12, 2), None))
        assert case.flat_view_stride(0) == (4, 1)
        assert case.flat_view_stride(1) == (12, 2)
        assert case.flat_view_stride(2) == (8, 1)

    def test_offset_none_falls_back_to_zero(self):
        case = self._make_case(offsets=(None, 10, None))
        assert case.flat_view_offset(0) == 0
        assert case.flat_view_offset(1) == 10
        assert case.flat_view_offset(2) == 0

    def test_storage_shape_none_falls_back_to_view(self):
        case = self._make_case(storage_shapes=((6, 8), None, (10, 12)))
        assert case.flat_storage_shape(0) == (6, 8)
        assert case.flat_storage_shape(1) == (5, 6)
        assert case.flat_storage_shape(2) == (10, 12)

    def test_all_none_uses_defaults(self):
        case = self._make_case(
            strides=(None, None, None), offsets=(None, None, None), storage_shapes=(None, None, None)
        )
        assert case.flat_view_stride(0) == (4, 1)
        assert case.flat_view_offset(1) == 0
        assert case.flat_storage_shape(2) == (7, 8)

    def test_no_fields_set_defaults(self):
        case = self._make_case()
        from ttk.utilities.shape import shape_stride

        for i in range(3):
            assert case.flat_view_stride(i) == shape_stride(case.flat_tensor_view_shapes[i])
            assert case.flat_view_offset(i) == 0
            assert case.flat_storage_shape(i) == case.flat_tensor_view_shapes[i]

    def test_mixed_none_and_values(self):
        case = self._make_case(
            strides=(None, (12, 2), None), offsets=(5, None, 20), storage_shapes=((4, 6), None, None)
        )
        assert case.flat_view_stride(0) == (4, 1)
        assert case.flat_view_stride(1) == (12, 2)
        assert case.flat_view_stride(2) == (8, 1)
        assert case.flat_view_offset(0) == 5
        assert case.flat_view_offset(1) == 0
        assert case.flat_view_offset(2) == 20
        assert case.flat_storage_shape(0) == (4, 6)
        assert case.flat_storage_shape(1) == (5, 6)
        assert case.flat_storage_shape(2) == (7, 8)


class TestFlatInputDataRangesPadding:
    @pytest.mark.parametrize(
        ("tensor_view_shapes", "input_data_ranges", "output_tensor_indexes", "expected_pure_output", "expected_flat"),
        [
            (((3, 3), (3, 2), (3, 5)), ((None, 1.0), (-1.0, 1.0)), (2,), [2], ((None, 1.0), (-1.0, 1.0), (None, None))),
            (
                ((3, 3), (3, 2), (3, 5)),
                ((None, 1.0), (-1.0, 1.0), (0.0, 5.0)),
                (2,),
                None,
                ((None, 1.0), (-1.0, 1.0), (0.0, 5.0)),
            ),
            (((3, 3), (3, 2)), ((None, 1.0), (-1.0, 1.0)), (), None, ((None, 1.0), (-1.0, 1.0))),
            (
                ((2, 3), (3, 4), (4, 5), (5, 6)),
                ((None, 1.0),),
                (),
                None,
                ((None, 1.0), (None, 1.0), (None, 1.0), (None, 1.0)),
            ),
        ],
        ids=[
            "short-by-one-pads-none",
            "exact-count-no-padding",
            "no-short-no-padding",
            "broadcast-fills-all-even-with-gap",
        ],
    )
    def test_normalize_compressed_pads(
        self,
        make_testcase,
        tensor_view_shapes,
        input_data_ranges,
        output_tensor_indexes,
        expected_pure_output,
        expected_flat,
    ):
        case = make_testcase(
            tensor_view_shapes=tensor_view_shapes,
            input_data_ranges=input_data_ranges,
            output_tensor_indexes=output_tensor_indexes,
        )
        case._normalize_compressed_fields()
        if expected_pure_output is not None:
            assert case.pure_output_indexes == expected_pure_output
        assert case.flat_input_data_ranges == expected_flat

    def test_expand_by_dist_short_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,),
        )
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0), (None, None))


# ========== TensorList flattening edge-case tests ==========

# Two TensorList params, each with 1 tensor inside.
_TL2_SHAPES = (((3, 2, 4),), ((3, 2, 4),))


class TestTensorListFlatFields:
    def test_flat_dtypes_formats_and_offsets_nested(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=_TL2_SHAPES,
            tensor_dtypes=(("float32",), ("float32",)),
            tensor_formats=(("ND",), ("ND",)),
            tensor_view_offsets=((0,), (0,)),
        )
        assert case.flat_tensor_dtypes == ("float32", "float32")
        assert case.flat_tensor_formats == ("ND", "ND")
        assert case.flat_tensor_view_offsets == (0, 0)

    def test_flat_strides_and_storage_shapes_compressed(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            tensor_view_strides=((4, 1),),
            tensor_storage_shapes=(((3, 2, 4),),),
        )
        case._normalize_compressed_fields()
        assert case.flat_tensor_view_strides == ((4, 1), (4, 1), (4, 1))
        assert case.flat_tensor_storage_shapes == ((3, 2, 4), (3, 2, 4), (3, 2, 4))

    def test_flat_passthrough_no_tensorlist(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((2, 3), (4, 5)), tensor_dtypes=("float32", "float16"))
        assert case.flat_tensor_dtypes == ("float32", "float16")


# ========== Comprehensive tests for _normalize_range_field_by_dist paths ==========


class TestIsRangeFieldAlreadyNested:
    _ALREADY_NESTED = [
        ((((0.0, 1.0), (-1.0, 1.0)), (0.5, 2.0)), (2, 0)),
        ((((0.0, 1.0),),), (1,)),
    ]
    _NOT_ALREADY_NESTED = [
        (((0.0, 1.0), (-1.0, 1.0)), (2, 0)),
        ((((0.0, 1.0),), (0.5, 2.0)), (2, 0)),
    ]

    def _assert_already_nested(self, field, dist):
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = field
        case._normalize_field_by_dist("input_data_ranges", dist, TestcaseAclnn._is_range_group)
        assert case.is_valid is True
        assert case.input_data_ranges == field

    def _assert_not_already_nested(self, field, dist):
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = field
        case._normalize_field_by_dist("input_data_ranges", dist, TestcaseAclnn._is_range_group)
        assert case.input_data_ranges != field or case.is_valid is False

    @pytest.mark.parametrize(
        ("field", "dist"), _ALREADY_NESTED, ids=["fully-nested-range-list", "single-element-tensorlist-nested"]
    )
    def test_already_nested(self, field, dist):
        self._assert_already_nested(field, dist)

    @pytest.mark.parametrize(
        ("field", "dist"), _NOT_ALREADY_NESTED, ids=["len-mismatch-returns-false", "inner-len-mismatch-returns-false"]
    )
    def test_not_already_nested(self, field, dist):
        self._assert_not_already_nested(field, dist)

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ((), ()),
            (None, None),
        ],
        ids=["empty-field", "none-field"],
    )
    def test_empty_and_none_field(self, field, expected):
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = field
        case._normalize_field_by_dist("input_data_ranges", (0,), TestcaseAclnn._is_range_group)
        assert case.input_data_ranges == expected


class TestResolvePadValue:
    def test_explicit_pad_overrides_default(self):
        assert TestcaseAclnn._resolve_pad_value("input_data_ranges", explicit_pad=(0, 0)) == (0, 0)

    def test_auto_resolve_from_header_defaults(self):
        assert TestcaseAclnn._resolve_pad_value("input_data_ranges") == (None, None)
        assert TestcaseAclnn._resolve_pad_value("absolute_precision") == 1e-8

    def test_auto_resolve_unknown_field_returns_no_pad(self):
        from ttk.core_modules.common.testcase.base import _NO_PAD

        assert TestcaseAclnn._resolve_pad_value("nonexistent_field") is _NO_PAD


class TestWriteBackNormalized:
    def test_writes_back_as_tuple_and_clears_cache(self, make_testcase):
        case = make_testcase(tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)), input_data_ranges=((0.0, 1.0),))
        case._flat_input_data_ranges = ((0.0, 1.0),)
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert isinstance(case.input_data_ranges, tuple)
        assert case._flat_input_data_ranges is None


class TestNormalizeRangeFieldEdgeCases:
    def test_non_tuple_and_empty_and_none_unchanged(self, make_testcase):
        for val in ("not_a_tuple", (), None):
            case = make_testcase(tensor_view_shapes=((3, 4), (5, 6)), input_data_ranges=val)
            dist = case.tensor_list_dist
            case._normalize_range_field_by_dist("input_data_ranges", dist)
            assert case.input_data_ranges == val

    def test_len_gt_dist_invalid(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)), input_data_ranges=((0.0, 1.0), (2.0, 3.0), (4.0, 5.0))
        )
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_range_expr_at_tensorlist_broadcast(self, make_testcase):
        case = make_testcase(tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)), input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == (((0.0, 1.0), (0.0, 1.0)), (-1.0, 1.0))

    def test_explicit_pad_value_used(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 4), (5, 6), (7, 8)), input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist, pad_value=(5.0, 10.0))
        assert case.input_data_ranges == ((0.0, 1.0), (-1.0, 1.0), (5.0, 10.0))

    def test_all_tensorlist_positions_broadcast(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), ((7, 8), (9, 10), (11, 12))),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)),
        )
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == (((0.0, 1.0), (0.0, 1.0)), ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))

    def test_partial_range_list_at_tensorlist_invalid(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=(((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)), (-1.0, 1.0)),
        )
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"


# ========== Tests for _auto_fill_output_tensor_indexes ==========


def _make_op_api_info(tensor_names, tensor_types=None):
    """Helper to create a mock OpApiInfo with given tensor param names."""
    if tensor_types is None:
        tensor_types = ["aclTensor*"] * len(tensor_names)
    params = OrderedDict()
    for name, typ in zip(tensor_names, tensor_types):
        params[name] = {"type": typ}
    info = OpApiInfo(params=params)
    return info


class TestAutoFillOutputTensorIndexes:
    """Tests for _auto_fill_output_tensor_indexes naming-convention-based detection."""

    def setup_method(self):
        self._patcher = None

    def _make_case(self, api_name="aclnnDummy", tensor_names=None, tensor_types=None, tensor_view_shapes=None):
        """Create a TestcaseAclnn with mocked OpApiInfoKeeper.info_of."""
        if tensor_names is None:
            tensor_names = ["self", "other", "out"]
        if tensor_view_shapes is None:
            tensor_view_shapes = tuple((2, 3) for _ in tensor_names)
        info = _make_op_api_info(tensor_names, tensor_types)
        case = TestcaseAclnn()
        case.api_name = api_name
        case.is_valid = True
        case.fail_reason = None
        case.tensor_view_shapes = tensor_view_shapes
        case.tensor_dtypes = tuple("float32" for _ in tensor_names)
        case.attributes = {}
        case.output_tensor_indexes = ()
        case.output_inplace_indexes = ()
        self._patcher = patch("ttk.core_modules.aclnn.testcase.OpApiInfoKeeper")
        MockCls = self._patcher.start()
        MockCls.return_value.info_of.return_value = info
        return case

    def teardown_method(self):
        if self._patcher is not None:
            self._patcher.stop()

    # --- Basic output detection by *Out suffix ---

    @pytest.mark.parametrize(
        ("api_name", "tensor_names", "tensor_view_shapes", "expected"),
        [
            ("aclnnDummy", ["self", "other", "yOut"], None, (2,)),
            ("aclnnDummy", ["input", "other", "output"], None, (2,)),
            ("aclnnDummy", ["input", "meanOutOptional", "rstdOutOptional"], None, (1, 2)),
            # --- Ref suffix (inplace) ---
            ("aclnnDummy", ["selfRef", "other", "yOut"], None, (0, 2)),
            # --- Fallback to last tensor ---
            ("aclnnDummy", ["self", "weight", "bias"], None, (-1,)),
            # --- Backward/Grad exclusions ---
            ("aclnnSoftmaxBackward", ["gradOutput", "output", "yOut"], None, (2,)),
            ("aclnnLayerNormBackward", ["input", "gradOut", "gradInputOut", "gradWeightOut"], None, (2, 3)),
            ("aclnnNsaSelectedAttentionGrad", ["query", "attentionOut", "dqOut"], None, (2,)),
            ("aclnnLogSoftmaxBackward", ["gradOutput", "output", "yOut"], None, (2,)),
            # --- Non-backward: no exclusions ---
            ("aclnnAdd", ["self", "other", "out"], None, (-1,)),
            ("aclnnAminmax", ["self", "minOut", "maxOut"], None, (1, 2)),
            # --- None (nullptr) tensor handling ---
            ("aclnnDummy", ["input", "resultOut", "reserveOutOptional"], ((3, 4), (3, 4), None), (1, 2)),
        ],
        ids=[
            "out-suffix-detected",
            "output-literal-detected",
            "out-optional-detected",
            "mixed-ref-and-out",
            "fallback-no-matching-names",
            "backward-excludes-grad-output",
            "backward-excludes-grad-out",
            "backward-excludes-attention-out",
            "backward-output-not-last-excluded",
            "non-backward-out-fallback",
            "non-backward-all-out",
            "none-out-optional-in-output-indexes",
        ],
    )
    def test_auto_fill_output_tensor_indexes(self, api_name, tensor_names, tensor_view_shapes, expected):
        """验证命名约定驱动的 output_tensor_indexes 自动填充。"""
        case = self._make_case(api_name=api_name, tensor_names=tensor_names, tensor_view_shapes=tensor_view_shapes)
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == expected

    # --- Already set by user ---

    def test_user_set_not_overridden(self):
        """用户已显式设置 output_tensor_indexes → 不被自动填充覆盖。"""
        case = self._make_case(tensor_names=["self", "other", "yOut"])
        case.output_tensor_indexes = (0,)
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (0,)

    # --- Not valid ---

    def test_invalid_skipped(self):
        """is_valid 为 False 时 → 不执行自动填充，output_tensor_indexes 保持空。"""
        case = self._make_case(tensor_names=["self", "out"])
        case.is_valid = False
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == ()

    # --- None (nullptr) inplace handling ---

    def test_none_ref_skipped_in_inplace(self):
        """None 输入在 inplace 命名约定下被跳过，仅填充非 None 的 Ref 位置。"""
        case = self._make_case(
            tensor_names=["varRef", "mRef", "maxGradNormOptionalRef", "grad"],
            tensor_view_shapes=((512,), (512,), None, (512,)),
        )
        case._auto_fill_output_inplace_indices()
        assert case.output_inplace_indexes == (0, 1)


def test_pure_attrs_unregistered_api_raises(monkeypatch):
    """info_of 未注册返回 None → 显式 RuntimeError。

    原实现直接解引用 None.params 抛 AttributeError，正确性依赖 worker 前置
    has_api 检查的隐式时序（pure_attrs 还被 _profiling_print / xpu_attrs 访问）。
    """
    from types import SimpleNamespace

    from ttk.core_modules.aclnn import testcase as aclnn_testcase

    monkeypatch.setattr(aclnn_testcase, "OpApiInfoKeeper", lambda: SimpleNamespace(info_of=lambda api: None))
    case = TestcaseAclnn()
    case.testcase_name = "unregistered_case"
    case.api_name = "aclnnNotRegistered"
    case.attributes = {"alpha": 1}

    with pytest.raises(RuntimeError, match=r"\[unregistered_case\] API aclnnNotRegistered not registered"):
        _ = case.pure_attrs


class TestTensorParamValid:
    """_tensor_param_valid：view/stride/offset/storage 一致性校验。

    per-slot None 声明值（flat_view_offset/flat_view_stride 契约中的默认回退：
    None → 0 / shape_stride）必须被校验按默认值处理，不得抛 TypeError。
    回归：tensor_view_offsets 声明含 None 时曾直接 view_offset < 0 崩溃。
    """

    @pytest.mark.parametrize(
        ("view_shape", "view_stride", "view_offset", "storage_shape", "expected"),
        [
            ((3, 2), None, None, (4, 2), True),
            ((3, 2), (2, 1), None, (4, 2), True),
            ((3, 2), (2, 1), 0, (4, 2), True),
            ((3, 2), (2, 1), 2, (4, 2), True),
            ((3, 2), (2, 1), -1, (4, 2), False),
            ((3, 2), (2, 1), 0, (2, 2), False),
            ((3, 2), (2, -1), 0, (4, 2), False),
        ],
        ids=[
            "none-offset-defaults",
            "none-offset-explicit-stride",
            "zero-offset",
            "tail-offset",
            "negative-offset",
            "span-exceeds-storage",
            "negative-stride",
        ],
    )
    def test_param_valid(self, view_shape, view_stride, view_offset, storage_shape, expected):
        assert TestcaseAclnn._tensor_param_valid(view_shape, view_stride, view_offset, storage_shape) is expected


class TestIsTorchDtypeSupportScalarList:
    """is_torch_dtype_support：嵌套 ScalarList 声明不得误判为非 torch 原生。

    回归：曾遍历嵌套 scalar_dtypes 结构，ScalarList 的 tuple 元素被当成
    dtype 传入 is_torch_native_dtype 恒为 False，导致全原生 dtype 用例被
    错误路由到 numpy 通道（use_numpy=True）。
    """

    def test_nested_scalar_list_all_native_is_supported(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 8), (3, 5)), ((2, 8), (3, 5))),
            tensor_dtypes=(("float16", "float16"), ("float16", "float16")),
            scalar_dtypes=(("float32", "float32"),),
        )
        case._scalar_list_dist = (2,)
        assert case.flat_scalar_dtypes == ("float32", "float32")
        assert case.is_torch_dtype_support() is True

    def test_flat_native_scalars_still_supported(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((4, 6),), tensor_dtypes=("float16",), scalar_dtypes=("float32",))
        assert case.is_torch_dtype_support() is True

    def test_non_native_tensor_dtype_unsupported(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((4, 6),), tensor_dtypes=("int4",))
        assert case.is_torch_dtype_support() is False

    def test_non_native_scalar_dtype_unsupported(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((4, 6),), tensor_dtypes=("float16",), scalar_dtypes=("hifloat8",))
        assert case.is_torch_dtype_support() is False


class TestXpuAttrsScalarList:
    """xpu_attrs：ScalarList 参数聚合为列表整体下发，普通标量单值下发，None 槽跳过。"""

    @pytest.mark.parametrize(
        ("params", "scalar_dtypes", "scalars_value", "expected"),
        [
            (
                (("x", "aclTensorList*"), ("scalars", "aclScalarList*")),
                (("float32", "float32", "float32"),),
                [[-0.5, 1.0, 0.0]],
                {"scalars": [-0.5, 1.0, 0.0]},
            ),
            (
                (("x", "aclTensorList*"), ("scalars", "aclScalarList*"), ("alpha", "aclScalar*")),
                (("float32", "float32"), "float64"),
                [[0.5, 1.5], 2.0],
                {"scalars": [0.5, 1.5], "alpha": 2.0},
            ),
            (
                (("x", "aclTensorList*"), ("alpha", "aclScalar*"), ("beta", "aclScalar*")),
                ("float32", "int64"),
                [1.0, 2],
                {"alpha": 1.0, "beta": 2},
            ),
            (
                (("x", "aclTensorList*"), ("alpha", "aclScalar*"), ("beta", "aclScalar*")),
                ("float32", "int64"),
                [None, 2],
                {"beta": 2},
            ),
        ],
        ids=["scalar-list-whole", "mixed-list-and-single", "plain-scalars", "none-scalar-skipped"],
    )
    def test_xpu_attrs(self, make_testcase, monkeypatch, params, scalar_dtypes, scalars_value, expected):
        """验证 scalars 按签名类型（ScalarList 整列表 / Scalar 单值 / None 跳过）写入 attrs。"""
        info = _make_op_api_info(*zip(*params))
        monkeypatch.setattr(
            "ttk.core_modules.aclnn.testcase.OpApiInfoKeeper",
            lambda: SimpleNamespace(info_of=lambda api: info),
        )
        case = make_testcase(scalar_dtypes=scalar_dtypes, scalars=scalars_value)
        assert case.xpu_attrs == expected

    def test_numpy_scalar_items_converted(self, make_testcase, monkeypatch):
        """numpy 标量经 item() 转为 Python 原生值，attrs 可 JSON 序列化。"""
        import json

        import numpy

        info = _make_op_api_info(["x", "scalars"], ["aclTensorList*", "aclScalarList*"])
        monkeypatch.setattr(
            "ttk.core_modules.aclnn.testcase.OpApiInfoKeeper",
            lambda: SimpleNamespace(info_of=lambda api: info),
        )
        case = make_testcase(scalar_dtypes=(("float32",),), scalars=[[numpy.float32(0.5)]])
        attrs = case.xpu_attrs
        assert attrs == {"scalars": [0.5]}
        assert isinstance(attrs["scalars"][0], float)
        json.dumps(attrs)
