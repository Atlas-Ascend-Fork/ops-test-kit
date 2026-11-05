# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""e2e 插件 golden Promote 升精度单测（numpy 载体；torch 分支在 CI 环境经
test_golden_generation 的集成用例覆盖）。
"""

from types import SimpleNamespace

import numpy as np

from ttk.core_modules.framework_api.golden import _promoted_golden_inputs


def _make_testcase(tensors, dtypes, shapes=None, pure_output_indexes=(), dist=None, override=None):
    return SimpleNamespace(
        tensors=tensors,
        flat_tensors=list(tensors),
        flat_tensor_dtypes=list(dtypes),
        tensor_view_shapes=shapes or [arr.shape if arr is not None else None for arr in tensors],
        pure_output_indexes=tuple(pure_output_indexes),
        tensor_list_dist=dist,
        golden_mode_override=override,
    )


class TestPromotedGoldenInputs:
    def test_mode_gate(self):
        tensors = [np.array([1.5], dtype=np.float16)]
        case = _make_testcase(tensors, ["float16"])
        assert _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Enable")) is None
        assert _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Promote")) is not None

    def test_override_wins_over_global(self):
        tensors = [np.array([1.5], dtype=np.float16)]
        case = _make_testcase(tensors, ["float16"], override="Promote")
        assert _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Enable")) is not None

    def test_float_promoted_int_untouched_none_kept(self):
        tensors = [
            np.array([1.5, 2.5], dtype=np.float16),
            np.array([1, 2], dtype=np.int32),
            None,
            np.array([3.0], dtype=np.float32),
        ]
        case = _make_testcase(tensors, ["float16", "int32", "float16", "float32"])
        promoted = _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Promote"))
        assert promoted[0].dtype == np.float32
        np.testing.assert_array_equal(promoted[0], np.array([1.5, 2.5], dtype=np.float32))
        assert promoted[1].dtype == np.int32  # 整型不动
        assert promoted[2] is None  # None 槽保持
        assert promoted[3].dtype == np.float64  # fp32 → fp64

    def test_shared_tensors_not_mutated(self):
        tensors = [np.array([1.5], dtype=np.float16)]
        case = _make_testcase(tensors, ["float16"])
        promoted = _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Promote"))
        assert promoted[0] is not tensors[0]
        assert tensors[0].dtype == np.float16  # 原对象不动

    def test_output_slots_excluded(self):
        tensors = [
            np.array([1.5], dtype=np.float16),
            np.array([2.5], dtype=np.float16),
        ]
        # flat 槽 1 为纯输出槽（pure_output_indexes）
        case = _make_testcase(tensors, ["float16", "float16"], pure_output_indexes=(1,))
        promoted = _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Promote"))
        assert promoted[0].dtype == np.float32
        assert promoted[1].dtype == np.float16  # 纯输出槽不升

    def test_tensorlist_renests_by_distribution(self):
        # flat 视角：TensorList 槽 0 含两个叶子（dist=[2, 0]：分组 2 + 单槽）
        flat = [
            np.array([1.0], dtype=np.float16),
            np.array([2.0], dtype=np.float16),
            np.array([3.0], dtype=np.float32),
        ]
        case = _make_testcase(flat, ["float16", "float16", "float32"], shapes=[[(1,), (1,)], (1,)], dist=[2, 0])
        promoted = _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Promote"))
        assert isinstance(promoted[0], list)  # 嵌套结构保持
        assert all(t.dtype == np.float32 for t in promoted[0])
        assert promoted[1].dtype == np.float64

    def test_nothing_to_promote_returns_none(self):
        tensors = [np.array([1, 2], dtype=np.int32)]
        case = _make_testcase(tensors, ["int32"])
        assert _promoted_golden_inputs(case, SimpleNamespace(golden_mode="Promote")) is None
