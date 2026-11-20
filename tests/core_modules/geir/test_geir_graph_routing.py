# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""图模式 worker 路由单元测试。

覆盖（执行层分支点）：
- ``_compile_stage`` 按例路由：``_resolved_graph`` 非空 → generate_graph_source /
  write_graph_case_config + 组合签名编译键；空 → 单算子路径原样
- ``_reconcile_inputs`` / ``_geir_param_order`` 图模式跳过（dN 扁平槽无锚点
  参数序/槽位对账语义）
- 图模式常量定值：平铺 ``dN`` 键复用单算子链（spec_tensors 按 dN 切分 →
  共享覆盖 → ``is_const`` → .bin 烤 Const）
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import ttk.core_modules.geir.worker as geir_worker
from ttk.core_modules.geir.graph_builder import graph_signature


def _graph_tc(**kw):
    base = {
        "testcase_name": "g_case",
        "op_name": "fused_mul_add_n",
        "_resolved_graph": {
            "ops": ("mul", "add_n"),
            "edges": (),
            "outputs": (),
            "dyn_in_counts": {},
            "dyn_out_counts": {},
            "node_attrs": {},
        },
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _op_tc(**kw):
    base = {"testcase_name": "op_case", "op_name": "add", "_resolved_graph": None}
    base.update(kw)
    return SimpleNamespace(**base)


class TestCompileStageRouting:
    """_compile_stage：判定依据是 _resolved_graph（非 graph_ops）。"""

    def test_graph_mode_routes_to_graph_builder(self):
        builder_calls = {}

        class FakeBuilder:
            work_dir = "/fake/work"
            op_dir = "/fake/ops"

            def __init__(self, switches):
                pass

            def generate_graph_source(self, ops, mode="const"):
                builder_calls["graph_source"] = (ops, mode)
                return "/fake/ops/graph__3_mul5_add_n.cpp"

            def write_graph_case_config(self, testcase, mode="const"):
                builder_calls["graph_config"] = mode
                return "/fake/work/g_case.json"

            def generate_op_source(self, op_name, mode="const"):
                builder_calls["op_source"] = op_name

            def write_case_config(self, testcase, mode="const"):
                builder_calls["op_config"] = mode

        def _fake_compile(source_path, key, op_dir):
            builder_calls["compile"] = (source_path, key, op_dir)
            return "/fake/binary"

        fake_compiler = SimpleNamespace(build_dir="/fake/work", compile_op=_fake_compile)

        def _fake_compiler_factory(switches, build_dir=None):
            return fake_compiler

        with mock.patch.object(geir_worker, "GeirGraphBuilder", FakeBuilder), mock.patch.object(
            geir_worker, "GeirCompiler", _fake_compiler_factory
        ):
            compiler, binary, config = geir_worker._compile_stage(_graph_tc(), SimpleNamespace(), "const")

        assert builder_calls["graph_source"] == (("mul", "add_n"), "const")
        assert "op_source" not in builder_calls
        assert builder_calls["compile"][1] == graph_signature(("mul", "add_n"))
        assert builder_calls["compile"][1] == "graph__3_mul5_add_n"
        assert binary == "/fake/binary"
        assert config == "/fake/work/g_case.json"

    def test_op_mode_unchanged(self):
        builder_calls = {}

        class FakeBuilder:
            work_dir = "/fake/work"
            op_dir = "/fake/ops"

            def __init__(self, switches):
                pass

            def generate_op_source(self, op_name, mode="const"):
                builder_calls["op_source"] = op_name
                return "/fake/ops/add.cpp"

            def write_case_config(self, testcase, mode="const"):
                builder_calls["op_config"] = mode
                return "/fake/work/op_case.json"

            def generate_graph_source(self, ops, mode="const"):
                builder_calls["graph_source"] = ops

        def _fake_compile(source_path, key, op_dir):
            builder_calls["compile"] = (source_path, key, op_dir)
            return "/fake/bin"

        fake_compiler = SimpleNamespace(build_dir="/fake/work", compile_op=_fake_compile)

        def _fake_compiler_factory(switches, build_dir=None):
            return fake_compiler

        with mock.patch.object(geir_worker, "GeirGraphBuilder", FakeBuilder), mock.patch.object(
            geir_worker, "GeirCompiler", _fake_compiler_factory
        ):
            _compiler, binary, _config = geir_worker._compile_stage(_op_tc(), SimpleNamespace(), "const")

        assert builder_calls["op_source"] == "add"
        assert "graph_source" not in builder_calls
        assert builder_calls["compile"][1] == "add"  # 单算子编译键 = op_name
        assert binary == "/fake/bin"

    def test_graph_source_failure_raises(self):
        class FakeBuilder:
            work_dir = "/fake/work"
            op_dir = "/fake/ops"

            def __init__(self, switches):
                pass

            def generate_graph_source(self, ops, mode="const"):
                return None

        def _fake_compiler_factory(switches, build_dir=None):
            return SimpleNamespace(build_dir="/fake")

        with mock.patch.object(geir_worker, "GeirGraphBuilder", FakeBuilder), mock.patch.object(
            geir_worker, "GeirCompiler", _fake_compiler_factory
        ), pytest.raises(RuntimeError, match="graph source generation failed"):
            geir_worker._compile_stage(_graph_tc(), SimpleNamespace(), "const")


class TestGraphModeGating:
    def test_reconcile_inputs_skipped_for_graph(self):
        # 图模式：input_tensor_slots 即便可物化也不得触碰（锚点端口基准不适用）
        tc = _graph_tc(
            input_tensor_slots=[object()],
            input_arrays=(np.zeros(3),),
            op_name="fused_mul_add_n",
        )
        geir_worker._reconcile_inputs(tc)  # 不应抛错/不应访问 slots

    def test_param_order_empty_for_graph(self):
        assert geir_worker._geir_param_order(_graph_tc()) == []

    def test_param_order_op_mode_queries_keeper(self):
        tc = _op_tc()
        fake_info = {"inputs": [{"name": "x1"}, {"name": "x2"}], "attr": [{"name": "alpha"}]}
        with mock.patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as keeper:
            keeper.return_value.info_of.return_value = fake_info
            order = geir_worker._geir_param_order(tc)
        assert order == ["x1", "x2", "alpha"]


class TestConstSlotConfig:
    """图模式常量定值：平铺 dN 键复用单算子链（切分 → 覆盖 → is_const）。"""

    def test_is_const_by_d_name_in_spec_tensors(self):
        """is_const 按图输入名（dN）对 spec_tensors 同名——与单算子"输入名 ∈ 属性键"同构。"""
        from ttk.core_modules.geir.graph_builder import _build_flat_inputs_json

        tc = SimpleNamespace(
            input_shapes=((2, 2), (1,)),
            input_dtypes=("float32", "float32"),
            input_formats=(),
            input_ori_formats=(),
            input_ori_shapes=(),
            dyn_input_shapes=None,
            spec_tensors={"d1": (1.5,)},  # 嵌套同名定值经校验落位
        )
        inputs = _build_flat_inputs_json(tc, is_dynamic=False)
        assert inputs[1]["is_const"] is True
        assert inputs[0]["is_const"] is False

    def test_cache_split_nested_form(self):
        """嵌套 attributes：顶层键全归节点容器（spec_attrs），spec_tensors 由校验按同名补位。"""
        from ttk.core_modules.geir.testcase import GeirTestcase

        tc = GeirTestcase()
        tc.graph_ops = ("mul", "add_n")
        tc.input_shapes = ((2, 2), (1,))
        tc.attributes = {"o0": {"x2": (1.5,)}, "o1": {"N": 2}}
        tc._cache_split_attributes()
        assert tc.spec_tensors == {}
        assert tc.spec_attrs == {"o0": {"x2": (1.5,)}, "o1": {"N": 2}}

    def test_graph_override_applies_d_const(self):
        """图模式经共享覆盖链按 dN 覆盖输入（与单算子同一条代码路径）。"""
        arr0 = np.zeros((2, 2), dtype="float32")
        arr1 = np.zeros((1,), dtype="float32")
        tc = SimpleNamespace(
            attributes={"o0": {"x2": (1.5,)}},
            spec_tensors={"d1": (1.5,)},
            input_arrays=(arr0, arr1),
            original_input_arrays=None,
            input_distribution=(),
            op_name="fused_mul_add_n",
            _resolved_graph={"ops": ("mul", "add_n")},
        )
        from ttk.core_modules.kernel.input_generate import _override_inputs_from_attributes

        with mock.patch("ttk.core_modules.kernel.input_generate.OpInfoKeeper") as keeper:
            keeper.return_value.info_of.return_value = None  # 图模式不得触达锚点 info
            _override_inputs_from_attributes(tc)
        assert arr1[0] == pytest.approx(1.5)
        assert arr0.max() == 0  # 非定值槽不受影响
