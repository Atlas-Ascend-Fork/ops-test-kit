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
Tests for normalize refactoring in testcase_op.py:
- _normalize_scalar_field_by_dist (string/scalar fields)
- _normalize_shape_field_by_dist (shape fields)
- _normalize_range_field_by_dist (range/pair fields)
- Distribution computation (_compute_input/output_distribution)
- flat_* properties after normalize
- ELEWISE/REDUCE output resolution before output_distribution
- Edge cases: None, empty, single float, compressed, already nested, length mismatch
"""

from unittest.mock import patch

import pytest

from ttk.core_modules.kernel.testcase import TestcaseKernel


def _make_testcase(
    op_name="Add",
    input_shapes=((8,), (8,)),
    input_dtypes=("float16", "float16"),
    output_shapes=((8,),),
    output_dtypes=("float16",),
    const_input_indexes=None,
    **kwargs,
):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name or 'None'}"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    if const_input_indexes is not None:
        case.const_input_indexes = const_input_indexes
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


@pytest.fixture(autouse=True)
def _mock_op_info(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


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


# =====================================================================
# Distribution computation
# =====================================================================


class TestDistributionComputation:
    def test_input_dist_nested(self):
        case = _make_testcase(input_shapes=(((3, 4), (5, 4)), (8,)))
        _validate(case)
        assert case.input_distribution == (2, 0)

    def test_output_dist_nested(self):
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.output_distribution == (2,)

    def test_combined_dist(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.tensor_list_distribution == (2, 0, 2)


# =====================================================================
# _normalize_scalar_field_by_dist — string/scalar fields
# =====================================================================


class TestNormalizeFieldByDist:
    def test_compressed_broadcast(self):
        """('float16',) broadcasts to match distribution."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float16")

    def test_compressed_single_float(self):
        """Single float absolute_precision broadcasts."""
        case = _make_testcase()
        case.absolute_precision = 1e-5
        _validate(case)
        assert case.absolute_precision == (1e-05,)

    def test_compressed_output_dtypes(self):
        """('float16',) output_dtypes broadcasts with output dist."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.output_dtypes == (("float16", "float16"),)

    def test_already_nested_skip(self):
        """Already nested field is not modified."""
        dtypes = (("float16", "float16"), "float16")
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=dtypes,
        )
        _validate(case)
        assert case.input_dtypes == dtypes

    def test_length_mismatch_invalid(self):
        """len(field) > len(dist) → invalid."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float32", "int32"),
        )
        _validate(case)
        assert case.is_valid is False

    def test_empty_field_skip(self):
        """Empty field is skipped, no normalize applied."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float16"),
            input_formats=(),
        )
        _validate(case)
        # input_formats is empty — normalize skips it
        assert case.input_formats == ()


# =====================================================================
# _normalize_shape_field_by_dist — shape fields
# =====================================================================


class TestNormalizeShapeField:
    def test_compressed_shape_broadcast(self):
        """Single shape broadcasts to all positions."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_ori_shapes=((3, 4),),
        )
        _validate(case)
        # (3,4) looks like a single shape, len=1, dist=(2,0), flat_count=3
        # broadcast to (3,4),(3,4),(3,4) then re-nest
        assert case.input_ori_shapes == (((3, 4), (3, 4)), (3, 4))

    def test_already_nested_shape_skip(self):
        """Already nested shape field is not modified."""
        ori = (((3, 4), (5, 4)), (8,))
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_ori_shapes=ori,
        )
        _validate(case)
        assert case.input_ori_shapes == ori

    def test_none_element_in_shape(self):
        """Mixed None and shape elements."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_ori_shapes=(None, None),
        )
        _validate(case)
        # len=2, dist=(2,0), flat_count=3.
        # _is_shape_field_already_nested: dist[0]=2, val=None → not tuple → False
        # len=2 < flat_count=3, len==len(dist)=2 → flatten_by_distribution
        # flatten_by_distribution((None, None), (2,0)) → (None, None)
        # re-nest: ((None, None), None) — but None is not a valid shape for TensorList
        # This is an edge case; just verify no crash
        assert case.is_valid


# =====================================================================
# _normalize_range_field_by_dist — range/pair fields
# =====================================================================


class TestNormalizeRangeField:
    def test_compressed_range_broadcast(self):
        """Single range broadcasts to all tensors in TensorList."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_data_ranges=((0, 1),),
        )
        _validate(case)
        assert case.input_data_ranges == (((0, 1), (0, 1)), (0, 1))

    def test_already_nested_range_skip(self):
        """Already nested ranges are not modified."""
        ranges = (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_data_ranges=ranges,
        )
        _validate(case)
        assert case.input_data_ranges == ranges

    def test_precision_tolerances_compressed(self):
        """Compressed precision_tolerances broadcast."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            precision_tolerances=((0.004, 0.004),),
        )
        _validate(case)
        assert case.precision_tolerances == (((0.004, 0.004), (0.004, 0.004)),)

    def test_precision_tolerances_pair_not_destructed(self):
        """(rtol, ptol) pairs must not be destructed during normalize."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            precision_tolerances=(((0.001, 0.002), (0.003, 0.004)),),
        )
        _validate(case)
        # Already nested — preserved
        assert case.precision_tolerances == (((0.001, 0.002), (0.003, 0.004)),)

    def test_non_tuple_field_skip(self):
        """Non-tuple field (e.g. float absolute_precision) is wrapped."""
        case = _make_testcase()
        case.precision_tolerances = None
        _validate(case)
        assert case.precision_tolerances is None

    def test_bare_scalar_broadcast(self):
        """1e-5 bare scalar → broadcast to all positions."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
        )
        case.absolute_precision = 1e-5
        _validate(case)
        assert case.absolute_precision == ((1e-5, 1e-5),)


# =====================================================================
# flat_* properties after normalize
# =====================================================================


class TestFlatPropertiesAfterNormalize:
    def test_flat_input_dtypes_compressed(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
        )
        _validate(case)
        assert case.flat_input_dtypes == ("float16", "float16", "float16")

    def test_flat_input_data_ranges_nested(self):
        ranges = (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_data_ranges=ranges,
        )
        _validate(case)
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_flat_absolute_precision_per_output_no_double_wrap(self):
        case = _make_testcase(output_shapes=((8,), (9,), (10,)))
        case.absolute_precision = (1e-5, 1e-5, 1e-5)
        _validate(case)
        assert case.flat_absolute_precision == (1e-5, 1e-5, 1e-5)
        for val in case.flat_absolute_precision:
            assert isinstance(val, float), f"Expected float, got {type(val): {val}}"


# =====================================================================
# ELEWISE/REDUCE resolution order
# =====================================================================


class TestInferenceResolutionOrder:
    def test_elewise_resolves_before_output_dist(self):
        """output_shapes='ELEWISE' resolved before output_distribution computed."""
        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes="ELEWISE",
            output_dtypes=("float16",),
        )
        _validate(case)
        assert isinstance(case.output_shapes, tuple)
        assert len(case.output_shapes) == 1
        assert case.output_distribution == (0,)

    def test_elewise_output_dtypes_normalized(self):
        """After ELEWISE resolves, output_dtypes gets normalized."""
        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes="ELEWISE",
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.output_dtypes == ("float16",)

    def test_reduce_resolves(self):
        """REDUCE resolves correctly."""
        case = _make_testcase(
            input_shapes=((8, 4),),
            input_dtypes=("float16",),
            output_shapes="REDUCE",
            output_dtypes=("float16",),
            attributes={"axis": (1,)},
        )
        _validate(case)
        assert isinstance(case.output_shapes, tuple)
        assert case.output_distribution == (0,)


# =====================================================================
# Edge cases
# =====================================================================


class TestNormalizeEdgeCases:
    def test_empty_input_shapes(self):
        case = _make_testcase(input_shapes=(), input_dtypes=(), input_ori_shapes=())
        _validate(case)
        assert case.input_distribution == ()

    def test_output_shapes_none(self):
        case = _make_testcase()
        case.output_shapes = None
        _validate(case)
        assert case.is_valid is False

    def test_multiple_tensorlists(self):
        """Multiple TensorList positions in input."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), ((8,), (9,), (10,))),
            input_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_distribution == (2, 3)
        assert case.input_dtypes == (("float16", "float16"), ("float16", "float16", "float16"))
        assert case.flat_input_dtypes == ("float16", "float16", "float16", "float16", "float16")

    def test_short_field_no_pad_invalid(self):
        """input_dtypes has no pad value — fewer values than dist → invalid."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,), (9,)),
            input_dtypes=("float16", "float32"),
        )
        _validate(case)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_precision_tolerances_flat_no_output_dist(self):
        """precision_tolerances with no output_distribution stays as-is."""
        case = _make_testcase(output_shapes=((8,),))
        case.precision_tolerances = ((0.004, 0.004),)
        _validate(case)
        # output_dist = (0,), len matches → _is_range_field_already_nested checks...
        # dist[0]=0, so no TensorList check → returns True → skip
        assert case.precision_tolerances == ((0.004, 0.004),)

    def test_all_none_fields(self):
        """Optional fields are None/empty — no crash during normalize."""
        case = _make_testcase()
        case.input_ori_shapes = None
        case.input_data_ranges = None
        case.precision_tolerances = None
        case.absolute_precision = None
        _validate(case)
        assert case.is_valid


# =====================================================================
# Per-TensorList compressed broadcast
# =====================================================================


class TestPerTensorListCompressed:
    """TensorList position has compressed (len-1) value that needs broadcast."""

    def test_input_dtype_per_tl_compressed(self):
        """('float32',) at TensorList(2) position → ('float32','float32')."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16",), "float32"),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float32")
        assert case.flat_input_dtypes == ("float16", "float16", "float32")

    def test_input_format_per_tl_compressed(self):
        """('ND',) at TensorList(2) position → ('ND','ND')."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_formats=(("ND",), "NZ"),
        )
        _validate(case)
        assert case.input_formats == (("ND", "ND"), "NZ")

    def test_output_dtype_per_tl_compressed(self):
        """('float16',) at output TensorList(2) position → ('float16','float16')."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16",),),
        )
        _validate(case)
        assert case.output_dtypes == (("float16", "float16"),)
        assert case.flat_output_dtypes == ("float16", "float16")

    def test_mixed_compressed_and_expanded(self):
        """One position compressed, another fully expanded."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float32"), "int32"),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float32"), "int32")

    def test_multiple_tensorlists_both_compressed(self):
        """Two TensorList positions, both with compressed dtype."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), ((8,), (9,), (10,))),
            input_dtypes=(("float16",), ("int32",)),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), ("int32", "int32", "int32"))
        assert case.flat_input_dtypes == ("float16", "float16", "int32", "int32", "int32")

    def test_scalar_at_tensorlist_position(self):
        """Scalar value at TensorList position broadcast to match count."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float32"),
        )
        _validate(case)
        # len=2, dist=(2,0), len(field)==len(dist) → flatten_by_distribution
        # "float16" at dist=2 → broadcast to ["float16","float16"]
        # "float32" at dist=0 → ["float32"]
        # flat = ("float16","float16","float32") → re-nest → (("float16","float16"),"float32")
        assert case.input_dtypes == (("float16", "float16"), "float32")


# =====================================================================
# Non-TensorList multi-output compressed broadcast
# =====================================================================


class TestNonTensorListMultiOutputCompressed:
    """Flat (non-nested) outputs with compressed single dtype/value."""

    def test_dtype_broadcast_5_outputs(self):
        """('uint16',) with 5 flat outputs → 5-element tuple."""
        case = _make_testcase(
            output_shapes=((8,), (8,), (8,), (8,), (8,)),
            output_dtypes=("uint16",),
        )
        _validate(case)
        assert case.output_dtypes == ("uint16", "uint16", "uint16", "uint16", "uint16")
        assert case.flat_output_dtypes == ("uint16", "uint16", "uint16", "uint16", "uint16")

    def test_format_broadcast_3_outputs(self):
        """('ND',) with 3 flat outputs → 3-element tuple."""
        case = _make_testcase(
            output_shapes=((8,), (16,), (32,)),
            output_formats=("ND",),
        )
        _validate(case)
        assert case.output_formats == ("ND", "ND", "ND")

    def test_single_output_no_broadcast(self):
        """Single output with single dtype — no broadcast needed."""
        case = _make_testcase(
            output_shapes=((8, 1716),),
            output_dtypes=("int32",),
        )
        _validate(case)
        assert case.output_dtypes == ("int32",)

    def test_per_param_compressed_3_outputs_invalid(self):
        """output_dtypes has no pad value — 2 dtypes for 3 outputs → invalid."""
        case = _make_testcase(
            output_shapes=((8,), (16,), (32,)),
            output_dtypes=("float16", "float32"),
        )
        _validate(case)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_input_dtype_broadcast_flat(self):
        """('float32',) with 3 flat inputs → 3-element tuple."""
        case = _make_testcase(
            input_shapes=((8,), (16,), (32,)),
            input_dtypes=("float32",),
        )
        _validate(case)
        assert case.input_dtypes == ("float32", "float32", "float32")


# =====================================================================
# _is_field_already_nested — prevent double-normalize
# =====================================================================


class TestIsFieldAlreadyNested:
    def _assert_already_nested_scalar(self, field, dist):
        case = TestcaseKernel()
        case.is_valid = True
        case.input_dtypes = field
        case._normalize_field_by_dist("input_dtypes", dist, TestcaseKernel._is_scalar_group)
        assert case.is_valid is True
        assert case.input_dtypes == field

    def _assert_not_already_nested_scalar(self, field, dist):
        case = TestcaseKernel()
        case.is_valid = True
        case.input_dtypes = field
        case._normalize_field_by_dist("input_dtypes", dist, TestcaseKernel._is_scalar_group)
        assert case.input_dtypes != field or case.is_valid is False

    def _assert_already_nested_range(self, field, dist):
        case = TestcaseKernel()
        case.is_valid = True
        case.input_data_ranges = field
        case._normalize_field_by_dist("input_data_ranges", dist, TestcaseKernel._is_range_group)
        assert case.is_valid is True
        assert case.input_data_ranges == field

    def test_fully_nested_matches_dist(self):
        self._assert_already_nested_scalar((("float16", "float16"), "float16"), (2, 0))

    def test_compressed_not_nested(self):
        self._assert_not_already_nested_scalar(("float16",), (2, 0))

    def test_range_fully_nested(self):
        self._assert_already_nested_range((((0.0, 1.0), (0.5, 1.0)), (0.0, 5.0)), (2, 0))


# =====================================================================
# Normalize is idempotent (no double-normalize)
# =====================================================================


class TestNormalizeIdempotent:
    """Verify that normalize on already-normalized fields is a no-op."""

    def test_fully_nested_dtypes_not_double_expanded(self):
        dtypes = (("float16", "float16"), "float16")
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=dtypes,
        )
        _validate(case)
        assert case.input_dtypes == dtypes

    def test_normalized_then_normalize_again(self):
        """Calling _expand_compressed_fields twice produces same result."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
        )
        _validate(case)
        first = case.input_dtypes
        # Normalize again
        case._expand_compressed_fields()
        assert case.input_dtypes == first


# =====================================================================
# append_output_dtype — interface test
# =====================================================================


class TestAppendOutputMetadata:
    """Tests for append_output_metadata: encapsulates metadata append with cache consistency."""

    def test_flat_output_no_dist(self):
        """1 output, no TensorList → all flat properties include appended metadata."""
        case = _make_testcase(
            output_shapes=((8, 1716),),
            output_dtypes=("int32",),
        )
        _validate(case)
        assert case.flat_output_dtypes == ("int32",)
        case.append_output_metadata("uint64", "ND", (9,))
        assert case.output_dtypes == ("int32", "uint64")
        assert case.flat_output_dtypes == ("int32", "uint64")
        assert case.flat_output_formats[-1] == "ND"
        assert case.flat_output_shapes[-1] == (9,)

    def test_flat_5_outputs_no_dist(self):
        """5 outputs, no TensorList → flat properties include appended metadata."""
        case = _make_testcase(
            output_shapes=((8,), (8,), (8,), (8,), (8,)),
            output_dtypes=("uint16",),
        )
        _validate(case)
        case.append_output_metadata("uint64", "ND", (9,))
        assert case.flat_output_dtypes == ("uint16",) * 5 + ("uint64",)

    def test_tensorlist_output(self):
        """TensorList output → flat properties flatten then append metadata."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16",),),
        )
        _validate(case)
        assert case.flat_output_dtypes == ("float16", "float16")
        case.append_output_metadata("uint64", "ND", (9,))
        assert case.output_dtypes == (("float16", "float16"), "uint64")
        assert case.flat_output_dtypes == ("float16", "float16", "uint64")
        assert case.flat_output_shapes == ((8,), (8,), (9,))

    def test_mixed_tensorlist_and_flat(self):
        """TensorList + flat output → flat correct after append."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            output_shapes=(((8,), (8,)), (4,)),
            output_dtypes=("float16",),
        )
        _validate(case)
        case.append_output_metadata("uint64", "ND", (9,))
        assert case.flat_output_dtypes == ("float16", "float16", "float16", "uint64")

    def test_cache_hit_refreshed(self):
        """flat properties accessed before append, then after — cache refreshed."""
        case = _make_testcase(
            output_shapes=((8,),),
            output_dtypes=("int32",),
        )
        _validate(case)
        _ = case.flat_output_dtypes  # populate cache
        case.append_output_metadata("uint64", "ND", (9,))
        assert case.flat_output_dtypes == ("int32", "uint64")


# =====================================================================
# _normalize_manual_binaries
# =====================================================================


class TestNormalizeManualBinaries:
    """Tests for _normalize_manual_binaries.

    分为三组参数化测试：
    - test_normalize_preserves_value: validate 后字段被规范化为 expected（包括 None/空、
      flat/nested、'None' 字符串转换、reshape 到嵌套、padding 缺失尾部等场景）
    - test_list_converted_to_tuple: 列表（含嵌套列表）被转换为 tuple，保留 isinstance 断言
    - test_normalize_rejects_invalid: 非法结构 → MANUAL_BINARIES_INVALID
    """

    _DEFAULT_SHAPES = ((8,), (8,))
    _DEFAULT_DTYPES = ("float16", "float16")
    _TL_SHAPES = (((3, 4), (5, 4)), (8,))
    _TL_DTYPES = (("float16", "float16"), "float32")
    _TL_NONE_SHAPES = (((3, 4), None), (8,))

    @pytest.mark.parametrize(
        ("input_shapes", "input_dtypes", "value", "expected"),
        [
            # empty / None / single string / flat / nested / 'None' string conversion
            (_DEFAULT_SHAPES, _DEFAULT_DTYPES, (), ()),
            (_DEFAULT_SHAPES, _DEFAULT_DTYPES, None, None),
            (((8,),), ("float16",), "ax.csv", ("ax.csv",)),
            (
                ((8,), (8,), (8,)),
                ("float16", "float16", "float16"),
                ("ax.csv", "ay.csv", "az.csv"),
                ("ax.csv", "ay.csv", "az.csv"),
            ),
            # None unquoted preserved (TensorList scenario via dtypes)
            (
                ((8,), None, (8,)),
                (("float16", "float16"), "float32"),
                ("ax.csv", None, "az.csv"),
                ("ax.csv", None, "az.csv"),
            ),
            # 'None' quoted string converted to None
            (
                ((8,), None, (8,)),
                ("float16", "float16", "float32"),
                ("ax.csv", "None", "az.csv"),
                ("ax.csv", None, "az.csv"),
            ),
            # nested tuple preserved
            (_TL_SHAPES, _TL_DTYPES, (("ax.csv", "ay.csv"), "az.csv"), (("ax.csv", "ay.csv"), "az.csv")),
            # nested with 'None' quoted → converted
            (_TL_NONE_SHAPES, _TL_DTYPES, (("ax.csv", "None"), "az.csv"), (("ax.csv", None), "az.csv")),
            # nested with None unquoted preserved
            (_TL_NONE_SHAPES, _TL_DTYPES, (("ax.csv", None), "az.csv"), (("ax.csv", None), "az.csv")),
            # flat with None input + explicit None binary ok
            (
                ((8,), None, (8,)),
                (("float16", "float16"), "float32"),
                ("ax.csv", None, "az.csv"),
                ("ax.csv", None, "az.csv"),
            ),
            # nested with optional None input ok
            (_TL_NONE_SHAPES, _TL_DTYPES, (("ax.csv", None), "az.csv"), (("ax.csv", None), "az.csv")),
            # flat binaries for TensorList inputs → reshaped to nested
            (_TL_SHAPES, _TL_DTYPES, ("ax.csv", "ay.csv", "az.csv"), (("ax.csv", "ay.csv"), "az.csv")),
            # flat binaries with None → reshaped and padded to nested
            (_TL_NONE_SHAPES, _TL_DTYPES, ("ax.csv", None, "az.csv"), (("ax.csv", None), "az.csv")),
            # no TensorList → stays flat
            (((8,), (8,)), ("float16", "float16"), ("ax.csv", "ay.csv"), ("ax.csv", "ay.csv")),
            # single input stays single
            (((8,),), ("float16",), "ax.csv", ("ax.csv",)),
            # all-None inputs + all-None binaries
            ((None, None), ("float16", "float16"), (None, None), (None, None)),
            # flat trailing None input preserved
            (
                ((8,), (8,), None),
                ("float16", "float16", "float16"),
                ("ax.csv", "ay.csv", None),
                ("ax.csv", "ay.csv", None),
            ),
            # TensorList group None mixed nested preserved
            (_TL_NONE_SHAPES, _TL_DTYPES, (("ax.csv", None), "az.csv"), (("ax.csv", None), "az.csv")),
            # trailing None inputs → binaries padded with None
            (((8,), None, None), ("float16", "float16", "float16"), ("ax.csv",), ("ax.csv", None, None)),
        ],
        ids=[
            "empty-not-modified",
            "none-not-modified",
            "single-string-wrapped",
            "flat-tuple-preserved",
            "none-unquoted-preserved",
            "none-quoted-converted",
            "nested-tuple-preserved",
            "nested-with-none-quoted",
            "nested-with-none-unquoted",
            "flat-with-none-input-explicit-none-ok",
            "nested-with-optional-input-none-ok",
            "flat-input-reshaped-to-nested",
            "flat-with-none-padded-to-nested",
            "no-tensor-list-stays-flat",
            "single-input-stays-single",
            "flat-all-none-inputs-all-none-binaries",
            "flat-trailing-none-input",
            "tensor-list-group-none-mixed-nested",
            "trailing-none-inputs-implicit-none-binaries",
        ],
    )
    def test_normalize_preserves_value(self, input_shapes, input_dtypes, value, expected):
        """验证 validate 后 manual_input_binaries 等于 expected。"""
        case = _make_testcase(input_shapes=input_shapes, input_dtypes=input_dtypes)
        case.manual_input_binaries = value
        _validate(case)
        assert case.manual_input_binaries == expected

    @pytest.mark.parametrize(
        ("input_shapes", "input_dtypes", "value", "expected", "inner_index"),
        [
            # top-level list → tuple
            (_DEFAULT_SHAPES, _DEFAULT_DTYPES, ["ax.csv", "ay.csv"], ("ax.csv", "ay.csv"), None),
            # nested list → tuple (inner element converted)
            (_TL_SHAPES, _TL_DTYPES, (["ax.csv", "ay.csv"], "az.csv"), (("ax.csv", "ay.csv"), "az.csv"), 0),
        ],
        ids=["list-converted-to-tuple", "nested-list-converted-to-tuple"],
    )
    def test_list_converted_to_tuple(self, input_shapes, input_dtypes, value, expected, inner_index):
        """验证 list 被 normalize 转换为 tuple（包括嵌套场景的内层转换）。"""
        case = _make_testcase(input_shapes=input_shapes, input_dtypes=input_dtypes)
        case.manual_input_binaries = value
        _validate(case)
        assert case.manual_input_binaries == expected
        assert isinstance(case.manual_input_binaries, tuple)
        if inner_index is not None:
            assert isinstance(case.manual_input_binaries[inner_index], tuple)

    @pytest.mark.parametrize(
        ("input_shapes", "input_dtypes", "value"),
        [
            # nested rejected without TensorList
            (((8,), (8,)), ("float16", "float16"), (("ax.csv", "ay.csv"), "az.csv")),
            # mixed nested/flat rejected
            (_TL_SHAPES, _TL_DTYPES, ("ax.csv", ("az.csv",))),
            # flat count exceeds inputs
            (((8,), (8,)), ("float16", "float16"), ("ax.csv", "ay.csv", "az.csv")),
            # flat missing required input
            (((8,), (8,), (8,)), ("float16", "float16", "float16"), ("ax.csv", "ay.csv")),
            # flat file for None input rejected
            (((8,), None, (8,)), ("float16", "float16", "float32"), ("ax.csv", "unexpected.csv", "az.csv")),
            # nested group count exceeds inputs
            (_TL_SHAPES, _TL_DTYPES, (("ax.csv", "ay.csv", "extra.csv"), "az.csv")),
            # nested missing required input (1 file for 2 non-None inputs)
            (_TL_SHAPES, _TL_DTYPES, (("ax.csv",), "az.csv")),
            # flat binaries exceed non-TensorList inputs (None, None not TensorList)
            (((None, None), (8,)), (("float16", "float16"), "float32"), (None, None, "az.csv")),
            # nested top-level count mismatch
            (_TL_SHAPES, _TL_DTYPES, (("ax.csv", "ay.csv"),)),
            # nested TensorList position is str instead of tuple
            (_TL_SHAPES, _TL_DTYPES, ("ax.csv", "az.csv")),
            # nested non-TensorList position is tuple
            (_TL_SHAPES, _TL_DTYPES, (("ax.csv", "ay.csv"), ("az.csv",))),
            # invalid type
            (((8,),), ("float16",), 123),
            # file for None in TensorList group rejected
            (_TL_NONE_SHAPES, _TL_DTYPES, (("ax.csv", "unexpected.csv"), "az.csv")),
            # missing file for non-None in TensorList group rejected
            (_TL_SHAPES, _TL_DTYPES, (("ax.csv", None), "az.csv")),
        ],
        ids=[
            "nested-rejected-without-tensor-list",
            "mixed-nested-flat-rejected",
            "flat-count-exceeds-inputs-rejected",
            "flat-missing-required-input-rejected",
            "flat-file-for-none-input-rejected",
            "nested-group-count-exceeds-inputs-rejected",
            "nested-missing-required-input-rejected",
            "flat-binaries-exceed-non-tensor-list-inputs-rejected",
            "nested-top-level-count-mismatch-rejected",
            "nested-tensor-list-position-is-str-rejected",
            "nested-non-tensor-list-position-is-tuple-rejected",
            "invalid-type-rejected",
            "file-for-none-in-tensor-list-group-rejected",
            "missing-file-for-non-none-in-tensor-list-group-rejected",
        ],
    )
    def test_normalize_rejects_invalid(self, input_shapes, input_dtypes, value):
        """验证非法结构被标记为 MANUAL_BINARIES_INVALID。"""
        case = _make_testcase(input_shapes=input_shapes, input_dtypes=input_dtypes)
        case.manual_input_binaries = value
        _validate(case)
        assert case.is_valid is False
        assert case.fail_reason == "MANUAL_BINARIES_INVALID"


# =====================================================================
# flat_manual_input_binaries property
# =====================================================================


class TestFlatManualInputBinaries:
    """Tests for flat_manual_input_binaries — uses deep_flatten."""

    def test_flat_no_dist(self):
        case = _make_testcase(input_shapes=((8,), (8,)), input_dtypes=("float16", "float16"))
        case.manual_input_binaries = ("a.csv", "b.csv")
        _validate(case)
        assert case.flat_manual_input_binaries == ("a.csv", "b.csv")

    def test_nested_tensorlist(self):
        case = _make_testcase(input_shapes=(((3, 4), (5, 4)), (8,)), input_dtypes=(("float16", "float16"), "float32"))
        case.manual_input_binaries = (("a.csv", "b.csv"), "c.csv")
        _validate(case)
        assert case.flat_manual_input_binaries == ("a.csv", "b.csv", "c.csv")

    def test_nested_with_none(self):
        case = _make_testcase(input_shapes=(((3, 4), None), (8,)), input_dtypes=(("float16", "float16"), "float32"))
        case.manual_input_binaries = (("a.csv", None), "c.csv")
        _validate(case)
        assert case.flat_manual_input_binaries == ("a.csv", None, "c.csv")

    def test_none_when_not_set(self):
        case = _make_testcase()
        _validate(case)
        assert case.flat_manual_input_binaries is None

    def test_empty_tuple(self):
        case = _make_testcase()
        case.manual_input_binaries = ()
        _validate(case)
        assert case.flat_manual_input_binaries == ()

    def test_flat_reshaped_to_nested_then_flat(self):
        """Flat binaries → reshape to nested → flat property returns flat."""
        case = _make_testcase(input_shapes=(((3, 4), (5, 4)), (8,)), input_dtypes=(("float16", "float16"), "float32"))
        case.manual_input_binaries = ("a.csv", "b.csv", "c.csv")
        _validate(case)
        assert case.manual_input_binaries == (("a.csv", "b.csv"), "c.csv")
        assert case.flat_manual_input_binaries == ("a.csv", "b.csv", "c.csv")


# =====================================================================
# Unified _normalize_field_by_dist — TDD tests
# =====================================================================


class TestIsScalarGroup:
    """Tests for _is_scalar_group classmethod.

    每行参数：value 为待判定值，expected 为是否被识别为标量组（tuple/list 视为组）。
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (("NCHW", "NHWC"), True),
            (["NCHW", "NHWC"], True),
            ("NCHW", False),
            (42, False),
            (1e-5, False),
            ((), True),
        ],
        ids=["tuple", "list", "string", "int", "float", "empty-tuple"],
    )
    def test_is_scalar_group(self, value, expected):
        """验证 _is_scalar_group 对各类输入的判定结果。"""
        assert TestcaseKernel._is_scalar_group(value) is expected


class TestIsRangeGroup:
    """Tests for _is_range_group classmethod.

    每行参数：value 为待判定值，expected 为是否被识别为 range 组
    （元素必须为二元 range 元组才算组；单个 (low, high) 不算组）。
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (((0.0, 1.0), (0.5, 1.0)), True),
            ((0.0, 1.0), False),
            (("NCHW", "NHWC"), False),
            ((), False),
            (((0.0, 1.0),), True),
            ((1, 2), False),
        ],
        ids=["range-list", "single-range", "string-tuple", "empty", "single-element-range-list", "int-tuple"],
    )
    def test_is_range_group(self, value, expected):
        """验证 _is_range_group 对各类输入的判定结果。"""
        assert TestcaseKernel._is_range_group(value) is expected


class TestNormalizeFieldAlreadyNested:
    def test_scalar_and_range_fully_nested_skip(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float32"), "int32"),
            input_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float32"), "int32")
        assert case.input_data_ranges == (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))

    def test_flat_dist_zero_already_nested(self):
        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float32"),
        )
        _validate(case)
        assert case.input_dtypes == ("float16", "float32")


class TestNormalizeFieldLenEqOne:
    def test_scalar_compressed_broadcast(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float16")

    def test_scalar_multi_element_group_rejected(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
        )
        case.input_dtypes = (("float16", "float32"),)
        case._normalize_field_by_dist("input_dtypes", (2, 0), TestcaseKernel._is_scalar_group)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_range_compressed_broadcast(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_data_ranges=((0, 1),),
        )
        _validate(case)
        assert case.input_data_ranges == (((0, 1), (0, 1)), (0, 1))


class TestNormalizeFieldPerPositionExpand:
    def test_scalar_group_single_broadcast(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16",), "float32"),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float32")

    def test_range_group_mismatch_rejected(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
        )
        case.input_data_ranges = (((0, 1), (0, 1), (0, 1)), (0, 5))
        case._normalize_field_by_dist("input_data_ranges", (2, 0), TestcaseKernel._is_range_group)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"


class TestNormalizeFieldGuardChecks:
    def test_guard_checks_skip_normalize(self):
        case = _make_testcase()
        case.is_valid = False
        case.input_dtypes = ("float16",)
        original = case.input_dtypes
        case._normalize_field_by_dist("input_dtypes", (2, 0), TestcaseKernel._is_scalar_group)
        assert case.input_dtypes == original

        case2 = _make_testcase(input_shapes=(((3, 4), (5, 4)), (8,)), input_formats=())
        _validate(case2)
        assert case2.input_formats == ()

        case3 = _make_testcase()
        case3.precision_tolerances = None
        _validate(case3)
        assert case3.precision_tolerances is None
