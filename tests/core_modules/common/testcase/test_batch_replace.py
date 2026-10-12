#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""batch 片段替换单测。

契约：同 (sample_seed, 区段 shape, dtype, data_range, distribution) → 逐位
一致；全局随机态零漂移；relation 不完整 / 指令非法 / 布局错位均跳过不阻断。
"""

from types import SimpleNamespace

import numpy
import pytest

from ttk.core_modules.common.testcase.base import TestcaseBase
from ttk.core_modules.common.testcase.batch_replace import (
    apply_batch_replacement,
    apply_slot_batch_replacement,
)
from ttk.core_modules.common.testcase.input_data import generate_np_storages
from ttk.utilities import RandomData, shape_stride

_SWITCHES = SimpleNamespace(random_seed=42, input_distribution="uniform")


def _batch_fields(axis_samples):
    """构造三字段嵌套（slot → 轴组 → 样本，与 test_testcase_base 契约同构）。

    axis_samples: [(axis_idx, [((start, stop, step), seed), ...]), ...]
    返回 (batch_axis, batch_slice_info, batch_seed)。
    """
    axes = tuple(axis for axis, _ in axis_samples)
    slice_groups = tuple(tuple(sl for sl, _ in samples) for _, samples in axis_samples)
    seed_groups = tuple(tuple(seed for _, seed in samples) for _, samples in axis_samples)
    return (axes,), (slice_groups,), (seed_groups,)


def _case(fields, name="t"):
    case = TestcaseBase()
    case.testcase_name = name
    case.is_valid = True
    case.batch_axis, case.batch_slice_info, case.batch_seed = fields
    return case


def _single_axis_relation(slices, seeds):
    """单槽单轴组（axis 0）便捷构造：slices/seeds 为样本列表。"""
    return _batch_fields([(0, list(zip(slices, seeds)))])


class _StubCase:
    """generate_np_storages 消费的最小 testcase 契约（aclnn/e2e 槽访问器）。"""

    def __init__(self, shapes, batch=None, dtypes="float32"):
        self.testcase_name = "stub"
        self.api_name = "stub_api"
        self.flat_tensor_view_shapes = tuple(shapes)
        self.flat_tensor_dtypes = (dtypes,) * len(shapes)
        self.flat_input_data_ranges = ((None, None),) * len(shapes)
        self.pure_output_indexes = ()
        self.batch_axis, self.batch_slice_info, self.batch_seed = batch or (None, None, None)

    def flat_storage_shape(self, idx):
        return self.flat_tensor_view_shapes[idx]

    def flat_view_stride(self, idx):
        return shape_stride(self.flat_tensor_view_shapes[idx])

    def flat_view_offset(self, idx):
        return 0


class TestApplySlotBatchReplacement:
    def test_same_seed_same_region_identical(self):
        """核心契约：同 seed + 同区段 shape → 逐位一致（背景不同不影响区段）。"""
        fields = _single_axis_relation([(0, 2, 1)], [101])
        arr_a = numpy.random.RandomState(1).uniform(-1, 1, (4, 3)).astype("float32")
        arr_b = numpy.random.RandomState(2).uniform(-1, 1, (4, 3)).astype("float32")
        for arr in (arr_a, arr_b):
            apply_slot_batch_replacement(
                _case(fields),
                0,
                arr,
                dtype="float32",
                data_range=(None, None),
                distribution="uniform",
            )
        numpy.testing.assert_array_equal(arr_a[:2], arr_b[:2])
        # 背景不同：区段外保持各自随机值
        assert not numpy.array_equal(arr_a[2:], arr_b[2:])

    def test_cross_case_different_total_size_same_sample(self):
        """整批 (4,3) vs 分批 (2,3)：同 seed 声明样本逐位一致。"""
        fields = _single_axis_relation([(0, 2, 1)], [101])
        big = numpy.zeros((4, 3), dtype="float32")
        small = numpy.ones((2, 3), dtype="float32")
        apply_slot_batch_replacement(
            _case(fields),
            0,
            big,
            dtype="float32",
            data_range=(None, None),
            distribution="uniform",
        )
        apply_slot_batch_replacement(
            _case(fields),
            0,
            small,
            dtype="float32",
            data_range=(None, None),
            distribution="uniform",
        )
        numpy.testing.assert_array_equal(big[:2], small)

    def test_global_random_state_not_drifted(self):
        """替换后全局随机态零漂移（get_state/set_state 包裹播种）。"""
        fields = _single_axis_relation([(0, 2, 1)], [101])
        arr = numpy.zeros((4, 3), dtype="float32")
        numpy.random.seed(7)
        apply_slot_batch_replacement(
            _case(fields),
            0,
            arr,
            dtype="float32",
            data_range=(None, None),
            distribution="uniform",
        )
        after = numpy.random.get_state()
        numpy.random.seed(7)
        expected = numpy.random.get_state()
        numpy.random.set_state(after)
        draw_replaced = numpy.random.uniform(size=5)
        numpy.random.set_state(expected)
        numpy.testing.assert_array_equal(draw_replaced, numpy.random.uniform(size=5))

    def test_multiple_samples_distinct_regions(self):
        fields = _single_axis_relation([(0, 1, 1), (1, 3, 1), (3, 4, 1)], [11, 22, 33])
        arr = numpy.zeros((4, 3), dtype="float32")
        apply_slot_batch_replacement(
            _case(fields),
            0,
            arr,
            dtype="float32",
            data_range=(None, None),
            distribution="uniform",
        )
        assert not numpy.array_equal(arr[0], arr[1])
        assert not numpy.array_equal(arr[1], arr[2])

    @pytest.mark.parametrize(
        "fields",
        [
            (None, None, None),
            (((0,),), None, ((101,),)),
            (((0,),), (((0, 2, 1),),), None),
        ],
        ids=["none", "missing-slices", "missing-seed"],
    )
    def test_incomplete_relation_noop(self, fields):
        arr = numpy.arange(12, dtype="float32").reshape(4, 3).copy()
        case = TestcaseBase()
        case.testcase_name = "t"
        case.batch_axis, case.batch_slice_info, case.batch_seed = fields
        apply_slot_batch_replacement(
            case,
            0,
            arr,
            dtype="float32",
            data_range=(None, None),
            distribution="uniform",
        )
        numpy.testing.assert_array_equal(arr, numpy.arange(12, dtype="float32").reshape(4, 3))

    def test_invalid_axis_or_slice_skipped(self, caplog):
        original = numpy.arange(12, dtype="float32").reshape(4, 3).copy()
        with caplog.at_level("WARNING"):
            # axis 越界
            fields = _batch_fields([(9, [((0, 2, 1), 101)])])
            arr = original.copy()
            apply_slot_batch_replacement(
                _case(fields),
                0,
                arr,
                dtype="float32",
                data_range=(None, None),
                distribution="uniform",
            )
            numpy.testing.assert_array_equal(arr, original)
            # 空切片（start==stop）
            fields2 = _batch_fields([(0, [((2, 2, 1), 101)])])
            apply_slot_batch_replacement(
                _case(fields2),
                0,
                arr,
                dtype="float32",
                data_range=(None, None),
                distribution="uniform",
            )
            numpy.testing.assert_array_equal(arr, original)
        assert "invalid axis" in caplog.text

    def test_none_elements_skipped(self):
        """指令中的 None 元素（对齐 bcid 派生语义）跳过。"""
        # slot → 轴组 → 样本：首样本 None（slice+seed 双 None），次样本有效
        case = _case(
            (
                ((0,),),
                (((None, (0, 2, 1)),),),
                (((None, 101),),),
            )
        )
        arr = numpy.zeros((4, 3), dtype="float32")
        apply_slot_batch_replacement(case, 0, arr, dtype="float32", data_range=(None, None), distribution="uniform")
        assert not numpy.allclose(arr[:2], 0)

    def test_none_array_noop(self):
        fields = _single_axis_relation([(0, 2, 1)], [101])
        apply_slot_batch_replacement(
            _case(fields),
            0,
            None,
            dtype="float32",
            data_range=(None, None),
            distribution="uniform",
        )

    def test_apply_batch_replacement_flat(self):
        fields = _single_axis_relation([(0, 2, 1)], [101])
        arrays = [numpy.zeros((4, 3), dtype="float32"), None]
        apply_batch_replacement(
            _case(fields),
            arrays,
            flat_dtypes=("float32", "float32"),
            flat_ranges=((None, None), (None, None)),
            distribution="uniform",
        )
        assert not numpy.allclose(arrays[0][:2], 0)


class TestGenerateNpStoragesIntegration:
    def test_batch_relation_cross_case_consistency(self):
        """aclnn/e2e 入口：整批 vs 分批，同 bcid 声明样本输入一致。"""
        batch = _single_axis_relation([(0, 2, 1)], [101])
        big, _ = generate_np_storages(_StubCase(((4, 3),), batch=batch), _SWITCHES)
        small, _ = generate_np_storages(_StubCase(((2, 3),), batch=batch), _SWITCHES)
        numpy.testing.assert_array_equal(big[0][:2], small[0])

    def test_no_relation_unchanged_behavior(self):
        """无 batch 字段：播种与替换均不介入（行为零变化）。"""
        storages, _ = generate_np_storages(_StubCase(((4, 3),)), _SWITCHES)
        assert storages[0].shape == (4, 3)

    def test_layout_guard_skips_replacement(self, caplog):
        """非连续 stride（view/storage 错位）跳过替换，仅保留播种背景。"""

        class _StridedCase(_StubCase):
            def flat_view_stride(self, idx):
                return (1, 4)  # 非连续（转置语义），≠ shape_stride((4,3))=(3,1)

        numpy.random.seed(42 + 0)
        expected = RandomData("float32", (4, 3), (None, None)).generate("uniform")
        with caplog.at_level("WARNING"):
            storages, _ = generate_np_storages(
                _StridedCase(((4, 3),), batch=_single_axis_relation([(0, 2, 1)], [101])), _SWITCHES
            )
        numpy.testing.assert_array_equal(storages[0], expected)
        assert "non-contiguous view stride" in caplog.text

    def test_storage_shape_mismatch_skips_replacement(self, caplog):
        class _Fp8LikeCase(_StubCase):
            def flat_storage_shape(self, idx):
                return (12,)  # 展平存储 ≠ view shape

        numpy.random.seed(42 + 0)
        expected = RandomData("float32", (12,), (None, None)).generate("uniform")
        with caplog.at_level("WARNING"):
            storages, _ = generate_np_storages(
                _Fp8LikeCase(((4, 3),), batch=_single_axis_relation([(0, 2, 1)], [101])), _SWITCHES
            )
        numpy.testing.assert_array_equal(storages[0], expected)
        assert "storage shape" in caplog.text
