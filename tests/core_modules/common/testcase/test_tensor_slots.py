#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for the AoS Tensor slot model.

Covers:
- Tensor descriptor: dtype normalization, ori fallback, derived predicates
- materialize/values_skeleton/mirror/write primitives
- ApiTestcaseBase substrate: tensors/np_storages write-through views,
  set_flat_* runtime writes, flat_tensors cache identity, pickle/deepcopy
"""

import copy
import pickle

import numpy
import pytest

from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.testcase.api_base import ApiTestcaseBase
from ttk.core_modules.common.testcase.tensor import (
    Tensor,
    flat_slots,
    make_storage_view,
    materialize_slots,
    slot_data_mirror,
    slot_storage_mirror,
    values_match_slots,
    values_skeleton,
    write_slot_data,
    write_slot_storage,
)
from ttk.utilities.container import nest_by_distribution

NESTED_SHAPES = (((3, 3), (3, 2)), (3, 5))


class TestTensorDescriptor:
    def test_dtype_normalization(self):
        assert Tensor(dtype="fp16").dtype == "float16"
        assert Tensor(dtype="float").dtype == "float32"
        assert Tensor(dtype="float32").dtype == "float32"
        assert Tensor(dtype=None).dtype is None

    def test_dtype_unknown_passthrough(self):
        assert Tensor(dtype="*").dtype == "*"

    def test_ori_fallback(self):
        t = Tensor(shape=(2, 3), dtype="fp16", format="NC1HWC0")
        assert t.ori_shape == (2, 3)
        assert t.ori_format == "NC1HWC0"
        t2 = Tensor(shape=(2, 3), ori_shape=(3, 2), ori_format="NCHW")
        assert t2.ori_shape == (3, 2)
        assert t2.ori_format == "NCHW"

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [((2, 3), False), ((), True)],
        ids=["tensor", "scalar"],
    )
    def test_is_scalar(self, shape, expected):
        assert Tensor(shape=shape).is_scalar is expected

    def test_is_contiguous(self):
        assert Tensor(shape=(2, 3)).is_contiguous
        assert Tensor(shape=(2, 3), view_stride=(3, 1)).is_contiguous
        assert not Tensor(shape=(2, 3), view_stride=(1, 3)).is_contiguous
        assert not Tensor(shape=(2, 3), view_stride=(3, 1), view_offset=4).is_contiguous
        assert Tensor(shape=(2, 3), view_stride=(3, 1), view_offset=0).is_contiguous

    def test_is_const(self):
        assert not Tensor().is_const
        assert Tensor(value=1.5).is_const

    def test_nbytes(self):
        assert Tensor(shape=(2, 3), dtype="float32").nbytes == 24
        assert Tensor(shape=(), dtype="float32").nbytes == 4
        assert Tensor(dtype=None).nbytes == 0


class TestMaterializeSlots:
    def test_dist_walk(self):
        slots = materialize_slots(
            {
                "view_shapes": ((3, 3), (3, 2), (3, 5)),
                "dtypes": ("fp16", "fp16", "float"),
                "formats": ("ND", "ND", "ND"),
            },
            (2, 0),
        )
        assert len(slots) == 2
        assert isinstance(slots[0], tuple)
        assert len(slots[0]) == 2
        assert slots[0][0].shape == (3, 3)
        assert slots[0][0].dtype == "float16"
        assert slots[0][1].shape == (3, 2)
        assert slots[1].shape == (3, 5)
        assert slots[1].dtype == "float32"

    def test_none_slot(self):
        slots = materialize_slots({"view_shapes": (None, (3, 5))}, (0, 0))
        assert slots[0] is None
        assert slots[1].shape == (3, 5)

    def test_short_columns_pad_none(self):
        slots = materialize_slots({"view_shapes": ((3, 3), (3, 2)), "dtypes": ("fp16",)}, (0, 0))
        assert slots[0].dtype == "float16"
        assert slots[1].dtype is None

    def test_flat_slots(self):
        slots = materialize_slots({"view_shapes": ((3, 3), (3, 2), (3, 5))}, (2, 0))
        flat = flat_slots(slots)
        assert len(flat) == 3
        assert all(isinstance(t, Tensor) for t in flat)


class TestSlotPrimitives:
    def test_values_skeleton_and_mirror(self):
        a, b, c = numpy.ones(3), numpy.ones(4), numpy.ones(5)
        slots = values_skeleton(((a, b), c))
        assert isinstance(slots[0], tuple)
        assert len(slots[0]) == 2
        assert slots[0][0].data is a
        assert slots[1].data is c
        mirror = slot_data_mirror(slots)
        assert mirror[0] == [a, b]
        assert mirror[1] is c

    def test_values_skeleton_none_leaves(self):
        a = numpy.ones(3)
        slots = values_skeleton((a, None))
        assert isinstance(slots[0], Tensor)
        assert slots[1] is None

    def test_values_match_slots(self):
        a, b, c = numpy.ones(3), numpy.ones(4), numpy.ones(5)
        slots = values_skeleton(((a, b), c))
        assert values_match_slots(slots, ((a, b), c))
        assert not values_match_slots(slots, (a, b, c))
        assert not values_match_slots(slots, ((a, b),))
        assert not values_match_slots(slots, ((a, b, c), c))

    def test_write_slot_data(self):
        a, b = numpy.ones(3), numpy.ones(4)
        slots = values_skeleton((a, (b, b)))
        new_a, new_b1, new_b2 = numpy.zeros(3), numpy.zeros(4), numpy.zeros(4)
        write_slot_data(slots, (new_a, (new_b1, new_b2)))
        assert slots[0].data is new_a
        assert slots[1][0].data is new_b1
        assert slots[1][1].data is new_b2

    def test_write_slot_data_none_in_tensor_list(self):
        a = numpy.ones(3)
        slots = values_skeleton(((a, None), a))
        new_a = numpy.zeros(3)
        write_slot_data(slots, ((new_a, None), new_a))
        assert slots[0][0].data is new_a
        assert slots[0][1] is None

    def test_write_slot_storage_none_in_tensor_list(self):
        a = numpy.ones(3)
        slots = values_skeleton(((a, None), a))
        new_a = numpy.zeros(3)
        write_slot_storage(slots, ((new_a, None), new_a))
        assert slots[0][0].storage is new_a
        assert slots[0][1] is None

    def test_mirror_none_in_tensor_list(self):
        a = numpy.ones(3)
        slots = values_skeleton(((a, None), a))
        mirror = slot_data_mirror(slots)
        assert mirror[0] == [a, None]
        assert mirror[1] is a
        storage_mirror = slot_storage_mirror(slots)
        assert storage_mirror[0] == [None, None]


class TestMakeStorageView:
    def test_view_write_through(self):
        storage = numpy.arange(6)
        view = make_storage_view(storage, (2, 2), (2, 1), 0)
        view[:] = -1
        assert (storage[:4] == -1).all()
        assert (storage[4:] == [4, 5]).all()

    def test_rejects_non_contiguous_storage(self):
        storage = numpy.arange(6).reshape((2, 3)).T
        with pytest.raises(ValueError, match="C-contiguous"):
            make_storage_view(storage, (2, 3), (3, 1), 0)

    def test_rejects_negative_offset(self):
        with pytest.raises(ValueError, match="non-negative"):
            make_storage_view(numpy.arange(6), (2,), (1,), -1)

    def test_rejects_out_of_bounds_view(self):
        with pytest.raises(ValueError, match="beyond storage"):
            make_storage_view(numpy.zeros(4), (1000,), (1,), 0)

    def test_rejects_strided_out_of_bounds_view(self):
        with pytest.raises(ValueError, match="beyond storage"):
            make_storage_view(numpy.zeros(8), (4,), (3,), 0)

    def test_default_strides_when_none(self):
        storage = numpy.arange(6)
        view = make_storage_view(storage, (2, 3), None, 0)
        assert view.shape == (2, 3)
        assert (view == storage.reshape((2, 3))).all()

    def test_empty_view_bypasses_bounds_check(self):
        storage = numpy.zeros(4)
        view = make_storage_view(storage, (0, 100), (100, 1), 0)
        assert view.size == 0


class TestSlotSubstrateSafety:
    """np_storages 长度不匹配不得静默丢弃运行时数据。"""

    def _declared_case(self):
        case = TestcaseAclnn()
        case.tensor_view_shapes = NESTED_SHAPES
        return case

    def test_np_storages_mismatch_raises_and_keeps_data(self):
        case = self._declared_case()
        case.tensors = ([numpy.ones((3, 3)), numpy.ones((3, 2))], numpy.ones((3, 5)))
        with pytest.raises(ValueError, match="does not match existing"):
            case.np_storages = [numpy.ones((3, 3))]
        assert case.tensors[1] is not None

    def test_np_storages_without_runtime_data_allows_rebuild(self):
        case = self._declared_case()
        case.np_storages = [numpy.ones((3, 3))]  # 未写过运行时数据，允许裸槽重建
        assert case.np_storages[0].shape == (3, 3)

    def test_output_tensor_indexes_out_of_range_marks_invalid(self):
        case = self._declared_case()
        case.output_tensor_indexes = (5,)
        case._normalize_compressed_fields()
        assert case.is_valid is False
        assert "output_tensor_indexes" in case.fail_reason

    def test_output_tensor_indexes_negative_marks_invalid(self):
        case = self._declared_case()
        case.output_tensor_indexes = (-1,)
        case._normalize_compressed_fields()
        assert case.is_valid is False

    def test_output_accessors_safe_when_out_of_range(self):
        case = self._declared_case()
        case.output_tensor_indexes = (5,)
        assert case.output_dist == ()
        assert case.pure_output_indexes == []

    def test_rebuild_slots_preserves_data_kind_and_name(self):
        case = self._declared_case()
        case.tensors = (numpy.ones((3, 3)),)  # 结构不一致 → skeleton 重建
        slots = case.tensor_slots
        flat = flat_slots(slots)
        flat[0].data_kind = "const"
        flat[0].name = "t0"
        case.tensors = ([numpy.ones((3, 3)), numpy.ones((3, 2))], numpy.ones((3, 5)))
        new_flat = flat_slots(case.tensor_slots)
        assert new_flat[0].data_kind == "const"
        assert new_flat[0].name == "t0"


def _make_declared_case():
    case = TestcaseAclnn()
    case.tensor_view_shapes = NESTED_SHAPES
    case.tensor_dtypes = ("float32", "float32", "float32")
    return case


class TestSubstrateViews:
    def test_tensors_none_until_set(self):
        case = _make_declared_case()
        assert case.tensors is None
        assert case.np_storages is None
        assert case.flat_tensors is None

    def test_tensors_roundtrip_nested(self):
        case = _make_declared_case()
        t1, t2, t3 = numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))
        case.tensors = ([t1, t2], t3)
        assert case.tensors[0] == [t1, t2]
        assert case.tensors[1] is t3
        assert list(case.flat_tensors) == [t1, t2, t3]

    def test_tensors_set_none_resets_without_wiping_storages(self):
        case = _make_declared_case()
        storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        case.np_storages = storages
        case.tensors = None
        assert case.tensors is None
        assert case.np_storages is not None

    def test_np_storages_set_then_tensors_keeps_storages(self):
        case = _make_declared_case()
        s0, s1, s2 = numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))
        case.np_storages = [s0, s1, s2]
        case.tensors = ([s0.reshape((3, 3)), s1], s2)
        assert case.np_storages[0] is s0
        assert case.np_storages[2] is s2

    def test_flat_tensors_identity_cache(self):
        case = _make_declared_case()
        case.tensors = (numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5)))
        flat = case.flat_tensors
        assert case.flat_tensors is flat

    def test_set_flat_tensor_write_through(self):
        case = _make_declared_case()
        case.tensors = ([numpy.ones((3, 3)), numpy.ones((3, 2))], numpy.ones((3, 5)))
        new_view = numpy.zeros((3, 5))
        case.set_flat_tensor(2, new_view)
        assert case.tensors[1] is new_view
        assert case.flat_tensors[2] is new_view

    def test_set_flat_storage_write_through(self):
        case = _make_declared_case()
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        new_storage = numpy.zeros((3, 5))
        case.set_flat_storage(2, new_storage)
        assert case.np_storages[2] is new_storage

    def test_set_flat_tensor_on_none_slot_raises(self):
        case = TestcaseAclnn()
        case.tensor_view_shapes = (None,)
        with pytest.raises(IndexError):
            case.set_flat_tensor(0, numpy.ones(1))

    def test_np_storages_none_value_resets_flag(self):
        case = _make_declared_case()
        case.np_storages = [numpy.ones((3, 3))]
        case.np_storages = None
        assert case.np_storages is None

    def test_declaration_materialization(self):
        case = _make_declared_case()
        case.tensor_formats = (("ND", "ND"), "NCHW")
        case.tensor_view_strides = (((3, 1), (2, 1)), (5, 1))
        slots = case.tensor_slots
        assert slots[0][0].format == "ND"
        assert slots[0][0].view_stride == (3, 1)
        assert slots[0][1].view_stride == (2, 1)
        assert slots[1].format == "NCHW"
        assert slots[1].view_stride == (5, 1)
        assert slots[1].ori_shape == (3, 5)

    def test_storage_view_shares_object_identity(self):
        case = _make_declared_case()
        s = numpy.ones((3, 3))
        case.np_storages = [s, numpy.ones((3, 2)), numpy.ones((3, 5))]
        case.np_storages[0][:] = -1
        assert (s == -1).all()

    def test_pickle_roundtrip(self):
        case = _make_declared_case()
        s0 = numpy.arange(9, dtype=numpy.float32).reshape((3, 3))
        case.np_storages = [s0, numpy.ones((3, 2)), numpy.ones((3, 5))]
        case.tensors = ([s0, numpy.ones((3, 2))], numpy.ones((3, 5)))
        restored = pickle.loads(pickle.dumps(case))
        assert restored.np_storages[0] is not s0
        numpy.testing.assert_array_equal(restored.np_storages[0], s0)
        assert list(restored.flat_tensors)[0].shape == (3, 3)
        assert restored.tensor_slots[0][0].shape == (3, 3)

    def test_deepcopy_roundtrip(self):
        case = _make_declared_case()
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        case.tensors = ([numpy.ones((3, 3)), numpy.ones((3, 2))], numpy.ones((3, 5)))
        cloned = copy.deepcopy(case)
        cloned.set_flat_tensor(2, numpy.zeros((3, 5)))
        assert case.tensors[1].shape == (3, 5)
        assert (cloned.tensors[1] == 0).all()

    def test_data_only_skeleton_without_declaration(self):
        case = TestcaseAclnn()
        t1, t2, t3 = numpy.ones(1), numpy.ones(2), numpy.ones(3)
        case.tensors = ((t1, t2), t3)
        assert list(case.flat_tensors) == [t1, t2, t3]
        assert case.tensors[0] == [t1, t2]


class TestDeclarationFlip:
    """声明面翻转：物化后 flat/嵌套视图从槽表派生，运行期声明写点自动同步。"""

    def test_flat_derived_from_slots_after_materialization(self):
        case = _make_declared_case()
        case.tensor_dtypes = (("float32", "int8"), "float32")
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        assert case.flat_tensor_dtypes == ("float32", "int8", "float32")
        assert case.tensor_dtypes == (("float32", "int8"), "float32")

    def test_flat_legacy_before_materialization(self):
        case = _make_declared_case()
        case.tensor_dtypes = (("float32", "int8"), "float32")
        assert case.flat_tensor_dtypes == ("float32", "int8", "float32")
        assert case.tensor_slots[0][0].dtype == "float32"

    def test_runtime_dtype_write_syncs_flat_and_nested(self):
        case = _make_declared_case()
        case.tensor_dtypes = ("float32", "float32", "float32")
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        case.tensors = ([numpy.ones((3, 3)), numpy.ones((3, 2))], numpy.ones((3, 5)))
        promoted = ["float32", "float32", "float64"]
        dist = case.tensor_list_dist
        case.tensor_dtypes = tuple(nest_by_distribution(promoted, dist))
        assert case.flat_tensor_dtypes == ("float32", "float32", "float64")
        assert case.tensor_dtypes == (("float32", "float32"), "float64")

    def test_declaration_write_preserves_runtime_data(self):
        case = _make_declared_case()
        case.tensor_dtypes = ("float32", "float32", "float32")
        s0 = numpy.ones((3, 3))
        case.np_storages = [s0, numpy.ones((3, 2)), numpy.ones((3, 5))]
        d2 = numpy.zeros((3, 5))
        case.tensors = ([numpy.ones((3, 3)), numpy.ones((3, 2))], d2)
        case.tensor_dtypes = ("float16", "float16", "float16")
        assert case.np_storages[0] is s0
        assert case.tensors[1] is d2
        assert case.flat_tensor_dtypes == ("float16", "float16", "float16")

    def test_skeleton_slots_fall_back_to_raw_declaration_views(self):
        case = TestcaseAclnn()
        case.tensor_dtypes = ("float32", "int32")
        case.tensors = (numpy.ones(2), numpy.ones(3))
        assert case.tensor_dtypes == ("float32", "int32")
        assert case.flat_tensor_dtypes == ("float32", "int32")

    def test_empty_column_stays_empty_after_materialization(self):
        case = _make_declared_case()
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        assert case.tensor_formats == ()
        assert case.flat_tensor_formats == ()

    def test_dist_ambiguity_dead_after_materialization(self):
        case = _make_declared_case()
        case.tensor_dtypes = (("float32", "int8"), "float32")
        case.np_storages = [numpy.ones((3, 3)), numpy.ones((3, 2)), numpy.ones((3, 5))]
        flat = case.flat_tensor_dtypes
        assert len(flat) == 3
        assert flat == ("float32", "int8", "float32")
        assert case.tensor_list_dist == (2, 0)


class TestScalarSlots:
    """标量合并：scalar = shape=() 的 Tensor，scalars 为槽表视图。"""

    def test_scalar_materialization_shape(self):
        case = TestcaseAclnn()
        case.scalar_dtypes = ("fp16",)
        slots = case.scalar_slots
        assert len(slots) == 1
        assert slots[0].is_scalar
        assert slots[0].shape == ()
        assert slots[0].dtype == "float16"

    def test_none_dtype_none_slot(self):
        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64", None)
        slots = case.scalar_slots
        assert slots[0].dtype == "int64"
        assert slots[1] is None

    def test_data_range_materialized(self):
        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64",)
        case.scalar_data_ranges = ((0, 10),)
        assert case.scalar_slots[0].data_range == (0, 10)

    def test_scalars_none_until_written(self):
        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64",)
        assert case.scalars is None
        assert case.flat_scalars is None

    def test_scalar_list_nesting_write_through(self):
        import torch

        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64", ("float32", "float32"))
        case._scalar_list_dist = (0, 2)
        s0 = torch.tensor(3)
        s1, s2 = torch.tensor(1.0), torch.tensor(2.0)
        case.scalars = (s0, [s1, s2])
        assert case.scalars[0] is s0
        assert case.scalars[1] == [s1, s2]
        assert list(case.flat_scalars) == [s0, s1, s2]
        assert case.flat_scalar_dtypes == ("int64", "float32", "float32")
        assert all(t.is_scalar for t in flat_slots(case.scalar_slots))

    def test_runtime_scalar_dtype_write_syncs_and_preserves_values(self):
        import torch

        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64", ("float32", "float32"))
        case._scalar_list_dist = (0, 2)
        s0 = torch.tensor(3)
        s1, s2 = torch.tensor(1.0), torch.tensor(2.0)
        case.scalars = (s0, [s1, s2])
        case.scalar_dtypes = ("int64", ("float64", "float64"))
        assert case.flat_scalar_dtypes == ("int64", "float64", "float64")
        assert case.scalar_dtypes == ("int64", ("float64", "float64"))
        assert case.scalars[0] is s0
        assert case.scalars[1] == [s1, s2]

    def test_scalars_none_reset_keeps_declaration(self):
        import torch

        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64",)
        case.scalars = (torch.tensor(3),)
        case.scalars = None
        assert case.scalars is None
        assert case.flat_scalar_dtypes == ("int64",)

    def test_skeleton_fallback_reads_raw(self):
        import torch

        case = TestcaseAclnn()
        case.scalar_dtypes = ("int64",)
        case.scalars = (torch.tensor(3), torch.tensor(4))
        assert case.scalar_dtypes == ("int64",)
        assert case.flat_scalar_dtypes == ("int64",)


class TestKernelInputSlots:
    """Kernel 双 numpy 槽位映射：original_input_arrays→storage、input_arrays→data。"""

    def _kernel_case(self, shapes=((2, 2),), dtypes=("float32",), formats=("ND",)):
        from ttk.core_modules.kernel.testcase import TestcaseKernel

        case = TestcaseKernel()
        case.op_name = "add"
        case.input_shapes = shapes
        case.input_dtypes = dtypes
        case.input_formats = formats
        case.output_shapes = ((2, 3),)
        case.output_dtypes = ("float32",)
        case.output_formats = ("ND",)
        case.output_ori_shapes = ((2, 3),)
        case.output_ori_formats = ("ND",)
        return case

    def test_no_transform_same_object(self):
        case = self._kernel_case()
        arr = numpy.ones((2, 2))
        case.input_arrays = (arr,)
        case.original_input_arrays = (arr,)
        slots = flat_slots(case.input_tensor_slots)
        assert slots[0].data is slots[0].storage
        assert case.input_arrays[0] is case.original_input_arrays[0]

    def test_dual_data_write_through(self):
        case = self._kernel_case()
        physical = numpy.zeros((1, 2, 2, 1, 16))
        logical = numpy.ones((2, 2))
        case.input_arrays = (physical,)
        case.original_input_arrays = (logical,)
        assert case.input_arrays[0] is physical
        assert case.original_input_arrays[0] is logical
        slots = flat_slots(case.input_tensor_slots)
        assert slots[0].data is physical
        assert slots[0].storage is logical

    def test_ori_clear_keeps_data(self):
        case = self._kernel_case()
        physical = numpy.zeros((2, 2))
        logical = numpy.ones((2, 2))
        case.input_arrays = (physical,)
        case.original_input_arrays = (logical,)
        case.original_input_arrays = None
        assert case.original_input_arrays is None
        assert case.input_arrays[0] is physical

    def test_none_until_written(self):
        case = self._kernel_case()
        assert case.input_arrays is None
        assert case.original_input_arrays is None
        assert case.flat_input_arrays == ()

    def test_declaration_materialization_with_ori(self):
        case = self._kernel_case(shapes=((2, 2), (2, 3)), dtypes=("float32", "float32"), formats=("ND", "ND"))
        case.input_ori_shapes = ((2, 2), (2, 3))
        case.input_ori_formats = ("NCHW", "NCHW")
        case.input_arrays = (numpy.ones((2, 2)), numpy.ones((2, 3)))
        slots = flat_slots(case.input_tensor_slots)
        assert slots[0].shape == (2, 2)
        assert slots[0].dtype == "float32"
        assert slots[0].ori_shape == (2, 2)
        assert slots[0].format == "ND"
        assert slots[0].ori_format == "NCHW"
        assert slots[1].shape == (2, 3)

    def test_promote_swap_and_restore(self):
        case = self._kernel_case()
        physical = numpy.ones((2, 2), dtype=numpy.float16)
        logical = numpy.ones((2, 2), dtype=numpy.float16)
        case.input_arrays = (physical,)
        case.original_input_arrays = (logical,)
        bak = (case.input_arrays, case.original_input_arrays)
        promoted = physical.astype(numpy.float32)
        ori_promoted = logical.astype(numpy.float32)
        case.input_arrays = (promoted,)
        case.original_input_arrays = (ori_promoted,)
        assert case.input_arrays[0] is promoted
        case.input_arrays, case.original_input_arrays = bak
        assert case.input_arrays[0] is physical
        assert case.original_input_arrays[0] is logical

    def test_flat_derives_from_slots(self):
        case = self._kernel_case(shapes=((2, 2), (2, 3)), dtypes=("float32", "float32"), formats=("ND", "ND"))
        a, b = numpy.ones((2, 2)), numpy.ones((2, 3))
        case.input_arrays = (a, b)
        flat = case.flat_input_arrays
        assert list(flat) == [a, b]
        assert case.flat_input_arrays is flat

    def test_skeleton_without_declaration(self):
        case = self._kernel_case()
        case.input_shapes = None
        a = numpy.ones(3)
        case.input_arrays = ((a, a), a)
        assert list(case.flat_input_arrays) == [a, a, a]

    def test_pickle_roundtrip(self):
        case = self._kernel_case()
        physical = numpy.zeros((2, 2))
        logical = numpy.ones((2, 2))
        case.input_arrays = (physical,)
        case.original_input_arrays = (logical,)
        restored = pickle.loads(pickle.dumps(case))
        numpy.testing.assert_array_equal(restored.input_arrays[0], physical)
        numpy.testing.assert_array_equal(restored.original_input_arrays[0], logical)
        assert restored.input_tensor_slots[0].shape == (2, 2)


# ══════════════════════════════════════════════════════════════════════════
# manual_xpu_binaries 槽位（ApiTestcaseBase 共享字段：flat 展开 + 校验）
# ══════════════════════════════════════════════════════════════════════════
class _ManualXpuCase(ApiTestcaseBase):
    """最小可实例化载体：覆写输出槽访问器（基类无 flat_output_view_shapes）。"""

    __slots__ = ("_out_shapes",)

    def __init__(self, out_shapes=()):
        super().__init__()
        self._out_shapes = out_shapes

    @property
    def flat_output_view_shapes(self):
        return self._out_shapes


class TestApiBaseManualXpuField:
    """ApiTestcaseBase 共享字段：flat 展开（分布缓存直填）+ 校验（不成套）。"""

    def test_flat_undeclared_is_none(self):
        case = _ManualXpuCase()
        assert case.flat_manual_xpu_binaries is None

    def test_flat_passthrough_without_distribution(self):
        case = _ManualXpuCase()
        case.manual_xpu_binaries = ("gpu.bin",)
        assert case.flat_manual_xpu_binaries == ("gpu.bin",)

    def test_flat_deep_flattens_with_distribution(self):
        case = _ManualXpuCase()
        case._output_dist = (2,)  # 输出分布缓存直填（property 契约）
        case.manual_xpu_binaries = (("gpu0.bin", "gpu1.bin"),)
        assert case.flat_manual_xpu_binaries == ("gpu0.bin", "gpu1.bin")

    def test_check_passes_without_manual_golden_pairing(self, tmp_path):
        """aclnn/e2e 与 kernel 的差异：真跑设备，不要求 manual_input/golden 成套。"""
        path = tmp_path / "gpu.bin"
        numpy.arange(8, dtype=numpy.float32).tofile(path)

        case = _ManualXpuCase(((8,),))
        case.is_valid = True
        case.manual_xpu_binaries = (str(path),)
        case._check_manual_xpu_binaries()

        assert case.is_valid is True

    @pytest.mark.parametrize("paths_count", [1, 2])  # 1=缺文件 2=数量超槽
    def test_check_rejects_invalid_declaration(self, tmp_path, paths_count):
        path = tmp_path / "gpu.bin"
        numpy.arange(8, dtype=numpy.float32).tofile(path)
        target = str(path) if paths_count == 2 else str(tmp_path / "nope.bin")
        declared = (target, str(path))[:paths_count] if paths_count == 2 else (target,)

        case = _ManualXpuCase(((8,),))
        case.is_valid = True
        case.manual_xpu_binaries = declared if paths_count == 2 else (str(tmp_path / "nope.bin"),)
        case._check_manual_xpu_binaries()

        assert case.is_valid is False
        assert case.fail_reason == "MANUAL_XPU_BINARIES_INVALID"
