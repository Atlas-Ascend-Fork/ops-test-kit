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
"""kernel/GEIR 输入生成 batch 接入单测：播种 + ori 态片段替换。"""

from types import SimpleNamespace

import numpy
import pytest

import ttk.core_modules.kernel.input_generate as kernel_input_generate
from ttk.core_modules.common.testcase.batch import has_complete_batch_relation


def _kernel_case(shapes, batch=None, dtypes="float32"):
    return SimpleNamespace(
        flat_input_shapes=tuple(shapes),
        flat_input_ori_shapes=tuple(shapes),
        flat_input_dtypes=(dtypes,) * len(shapes),
        flat_input_ori_formats=("ND",) * len(shapes),
        flat_input_formats=("ND",) * len(shapes),
        flat_input_data_ranges=((None, None),) * len(shapes),
        input_distribution=(0,) * len(shapes),
        attributes={},
        batch_axis=batch[0] if batch else None,
        batch_slice_info=batch[1] if batch else None,
        batch_seed=batch[2] if batch else None,
    )


def _batch_fields(slices, seeds):
    return ((0,),), ((tuple(slices),),), ((tuple(seeds),),)


@pytest.fixture
def patched_switches(monkeypatch):
    switches = SimpleNamespace(random_seed=42, input_distribution="uniform")
    monkeypatch.setattr(kernel_input_generate, "get_global_storage", lambda: switches)
    return switches


class TestKernelRealtimeRandomInput:
    def test_batch_relation_seeds_and_replaces(self, patched_switches):
        """relation 完整：整张量播种 + 样本区段按 seed 重生成（ori 态，
        format=ND 时 input/original 两份同引用）。"""
        batch = _batch_fields([(0, 2, 1)], [101])
        context = _kernel_case(((4, 3),), batch=batch)
        kernel_input_generate._realtime_random_input(context)
        arr = context.input_arrays[0]
        # 复算期望：seed(42+0) 背景 + seed(101) 区段
        numpy.random.seed(42)
        from ttk.utilities import RandomData

        expected = RandomData("float32", (4, 3), (None, None)).generate("uniform")
        state = numpy.random.get_state()
        try:
            numpy.random.seed(101)
            region = RandomData("float32", (2, 3), (None, None)).generate("uniform")
        finally:
            numpy.random.set_state(state)
        expected[:2] = region
        numpy.testing.assert_array_equal(arr, expected)
        assert context.original_input_arrays[0] is arr  # ND 无变换，两份同引用

    def test_cross_case_consistency(self, patched_switches):
        """整批 (4,3) vs 分批 (2,3)：同 seed 样本逐位一致。"""
        batch = _batch_fields([(0, 2, 1)], [101])
        big = _kernel_case(((4, 3),), batch=batch)
        small = _kernel_case(((2, 3),), batch=batch)
        kernel_input_generate._realtime_random_input(big)
        kernel_input_generate._realtime_random_input(small)
        numpy.testing.assert_array_equal(big.input_arrays[0][:2], small.input_arrays[0])

    def test_no_relation_no_seeding(self, patched_switches):
        """无 batch 字段：不播种（全局流连续），行为与迁移前一致。"""
        context = _kernel_case(((4, 3),))
        kernel_input_generate._realtime_random_input(context)
        assert context.input_arrays[0].shape == (4, 3)
        assert not has_complete_batch_relation(context)
