# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""e2e 插件 golden Promote 升精度单测（numpy/tf 载体；torch 分支在 CI 环境经
test_golden_generation 的集成用例覆盖）。
"""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.framework_api.golden import _promoted_golden_inputs

# tf 载体用例环境门禁（两段探针）：find_spec 只验"包存在"，不验"可导入"——
# CI 常见形态是 wheel 在位但 import 崩溃（protobuf 版本不兼容 / C 扩展缺 so），
# 仅 find_spec 门禁会放行、测试体 import tensorflow 处 FAIL/ERROR 而非 SKIP。
# 故取 test_tf_graph_runner.py importorskip 的"不可导入也 skip"语义，但保留
# 包缺席时 find_spec 短路的零开销（importorskip 缺席同样快速失败、在位即真实
# 导入）；存在时 try-import 一次，经 sys.modules 缓存供测试体复用。仅作用于
# tf 载体类，numpy 载体用例保持恒跑；tf 可用时用例原样全保真执行。


def _tf_importable() -> bool:
    """包存在（find_spec）+ 可导入（try-import 实探）的联合判定。"""
    if importlib.util.find_spec("tensorflow") is None:
        return False
    try:
        import tensorflow  # noqa: F401  实探即缓存：测试体的 import 复用同一份
    except Exception:  # noqa: BLE001  # 崩溃形态不定（ImportError/AttributeError/...），一律按不可用落 SKIP
        return False
    return True


_TF_AVAILABLE = _tf_importable()


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


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed（tf 载体 promote 用例仅在 tf 环境运行）")
class TestTfTensorPromote:
    """tf EagerTensor 载体：tf.cast 升精度、失败原样回落。"""

    def test_tf_eager_tensor_promoted_by_cast(self):
        import tensorflow as tf

        from ttk.core_modules.framework_api.golden import _promote_tensor_value

        t = tf.constant([1.5, 2.5], dtype=tf.float32)
        promoted = _promote_tensor_value(t, "float64", "float32")
        assert promoted.dtype == tf.float64
        np.testing.assert_allclose(promoted.numpy(), [1.5, 2.5])
        assert t.dtype == tf.float32

    def test_tf_variable_promote_keeps_carrier(self):
        """tf.Variable 升精度须保持 Variable 载体：handle 转换/回读按 isinstance 定位，
        cast 产物降级为普通 Tensor 会使 mutable 位静默失效。"""
        import tensorflow as tf

        from ttk.core_modules.framework_api.golden import _promote_tensor_value

        v = tf.Variable([1.5, 2.5], dtype=tf.float32)
        promoted = _promote_tensor_value(v, "float64", "float32")
        assert isinstance(promoted, tf.Variable)
        assert promoted.dtype == tf.float64
        np.testing.assert_allclose(promoted.numpy(), [1.5, 2.5])

    def test_tf_degrade_paths_keep_original(self):
        """目标 dtype 不可解析、resource/variant 不可 cast（getter 抛 KeyError、
        cast 抛 OpError）均告警后原样回落。"""
        import tensorflow as tf

        from ttk.core_modules.framework_api.golden import _promote_tensor_value

        t = tf.constant([1.5], dtype=tf.float32)
        assert _promote_tensor_value(t, "no_such_dtype_xyz", "float32") is t
        handle = tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=[])
        assert _promote_tensor_value(handle, "float64", "resource") is handle
