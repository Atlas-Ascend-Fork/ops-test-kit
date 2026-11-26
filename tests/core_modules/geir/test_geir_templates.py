# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GEIR 模板渲染测试：单算子/图模式结构正确。

- 单算子：Type 属性按值形态运行时分流（字符串 → ge::DataType 存储，整数 →
  int64 存储，选择权在用例）、attr_entries 构建路由。
- 图模式结构断言：N 实例声明烘焙、动态口展开、attr setter 逐节点注入、
  通用接线循环、SetOutputs 双形态、共享骨架存在。

夹具 ProtoLoader / detect_cann_features 全 mock——离线可跑，不依赖 CANN。
"""

from types import SimpleNamespace

import pytest

import ttk.core_modules.geir.graph_builder as gb
from ttk.core_modules.geir.graph_builder import GeirGraphBuilder, _render_template


def _proto(op_class, inputs, outputs, dynamic_inputs=(), dynamic_outputs=(), attrs=(), optional_inputs=()):
    return SimpleNamespace(
        op_class=op_class,
        proto_file=f"fixture_{op_class.lower()}_proto.h",
        inputs=list(inputs),
        outputs=list(outputs),
        dynamic_inputs=list(dynamic_inputs),
        dynamic_outputs=list(dynamic_outputs),
        attrs=list(attrs),
        optional_inputs=list(optional_inputs),
        required_attrs=[a for a in attrs if a in ()],  # 由下方 _REQ 显式给出
    )


# 夹具算子（对齐 CANN proto 口名；REQUIRED_ATTR 以 required_attrs 给出）
_MUL = _proto("Mul", ["x1", "x2"], ["y"], attrs=[("alpha", "Float")])
_ADDN = SimpleNamespace(
    op_class="AddN",
    proto_file="fixture_addn_proto.h",
    inputs=["x"],
    outputs=["y"],
    dynamic_inputs=["x"],
    dynamic_outputs=[],
    attrs=[("N", "Int")],
    optional_inputs=[],
    required_attrs=[("N", "Int")],
)
_SPLITV = SimpleNamespace(
    op_class="SplitV",
    proto_file="fixture_splitv_proto.h",
    inputs=["x", "split_dim"],
    outputs=["y"],
    dynamic_inputs=[],
    dynamic_outputs=["y"],
    attrs=[("num_split", "Int")],
    optional_inputs=[],
    required_attrs=[("num_split", "Int")],
)
_FOREACH = SimpleNamespace(
    op_class="ForeachList",
    proto_file="fixture_foreach_proto.h",
    inputs=["x1", "x2"],
    outputs=["y"],
    dynamic_inputs=["x1", "x2"],
    dynamic_outputs=["y"],
    attrs=[("alpha", "Float"), ("mode", "String")],
    optional_inputs=[],
    required_attrs=[],
)
_CONVB = _proto("ConvBias", ["x", "bias"], ["y"], optional_inputs=["bias"], attrs=[("group", "Int")])
_CAST = SimpleNamespace(
    op_class="Cast",
    proto_file="fixture_cast_proto.h",
    inputs=["x"],
    outputs=["y"],
    dynamic_inputs=[],
    dynamic_outputs=[],
    attrs=[("dtype", "Type")],
    optional_inputs=[],
    required_attrs=[("dst_type", "Int")],
)

_PROTO_FIXTURES = {
    "mul": _MUL,
    "add_n": _ADDN,
    "split_v": _SPLITV,
    "foreach_list": _FOREACH,
    "conv_bias": _CONVB,
    "cast": _CAST,
}


class _FixtureLoader:
    def __init__(self, protos):
        self._protos = protos

    def get_op_info(self, op_name):
        return self._protos.get(op_name)


def _attr_entries(proto):
    from ttk.core_modules.geir.graph_builder import _PROTO_ATTR_TYPE_TO_CPP

    entries = []
    input_names = set(proto.inputs)
    for attr_name, proto_type in proto.attrs:
        if attr_name in input_names or attr_name[0] in ("!", "#", "@"):
            continue
        entries.append({"name": attr_name, "proto_type": proto_type, "cpp_type": _PROTO_ATTR_TYPE_TO_CPP[proto_type]})
    return entries


def _op_ctx(op_name):
    proto = _PROTO_FIXTURES[op_name]
    return {
        "proto_file": proto.proto_file,
        "op_class": proto.op_class,
        "input_names": proto.inputs[:],
        "output_names": proto.outputs[:],
        "dynamic_input_names": (proto.dynamic_inputs or []),
        "dynamic_output_names": (proto.dynamic_outputs or []),
        "attr_entries": _attr_entries(proto),
        "dtype_map": [("DT_UNDEFINED", "DT_UNDEFINED"), ("DT_FLOAT", "DT_FLOAT"), ("DT_INT64", "DT_INT64")],
        "format_map": [("FORMAT_ND", "FORMAT_ND")],
        "dtype_size_cases": [("DT_FLOAT", 4), ("DT_INT64", 8)],
        "packed_float4_enums": [],
        "ge_prof_ok": False,
    }


def test_op_template_dtype_attr_value_form_dual_branch():
    """Type 属性按值形态运行时分流：同一渲染同时含两种 SetAttr 分支。"""
    ctx = dict(_op_ctx("cast"))
    ctx["attr_entries"] = [{"name": "dtype", "proto_type": "Type", "cpp_type": "ge_dtype"}]
    out = _render_template("geir_op_template.cpp.j2", **ctx)
    assert 'op_inst.Operator::SetAttr("dtype", StringToDataType(__v.getString()));' in out
    assert 'op_inst.Operator::SetAttr("dtype", __v.getInt());' in out


class TestBuildAttrEntries:
    def test_type_attr_routes_ge_dtype(self, monkeypatch):
        monkeypatch.setattr(gb, "ProtoLoader", lambda: None)
        proto = SimpleNamespace(
            op_class="FixtureOp",
            inputs=["x"],
            attrs=[("dtype", "Type"), ("size", "Int")],
        )
        entries = {e["name"]: e["cpp_type"] for e in GeirGraphBuilder(None)._build_attr_entries(proto)}
        assert entries == {"dtype": "ge_dtype", "size": "int64_t"}

    def test_unknown_proto_type_raises(self, monkeypatch):
        monkeypatch.setattr(gb, "ProtoLoader", lambda: None)
        proto = SimpleNamespace(op_class="FixtureOp", inputs=["x"], attrs=[("weird", "Blob")])
        with pytest.raises(RuntimeError, match="Unsupported proto attr type 'Blob'"):
            GeirGraphBuilder(None)._build_attr_entries(proto)


def _graph_ctx(ops):
    nodes, proto_files = [], []
    for i, op in enumerate(ops):
        proto = _PROTO_FIXTURES[op]
        nodes.append(
            {
                "idx": i,
                "op_class": proto.op_class,
                "proto_file": proto.proto_file,
                "input_names": proto.inputs[:],
                "dynamic_input_names": (proto.dynamic_inputs or []),
                "output_names": proto.outputs[:],
                "dynamic_output_names": (proto.dynamic_outputs or []),
                "attr_entries": _attr_entries(proto),
            }
        )
        if proto.proto_file not in proto_files:
            proto_files.append(proto.proto_file)
    return {
        "proto_files": proto_files,
        "nodes": nodes,
        "dtype_map": [("DT_UNDEFINED", "DT_UNDEFINED"), ("DT_FLOAT", "DT_FLOAT"), ("DT_INT64", "DT_INT64")],
        "format_map": [("FORMAT_ND", "FORMAT_ND")],
        "dtype_size_cases": [("DT_FLOAT", 4), ("DT_INT64", 8)],
        "packed_float4_enums": [],
        "ge_prof_ok": False,
    }


class TestGraphTemplateRender:
    def test_mul_addn_structure(self):
        src = _render_template("geir_graph_template.cpp.j2", **_graph_ctx(("mul", "add_n")))
        # N 实例烘焙（类型化变量防 vector<Operator> 切片）+ 基类句柄向量
        assert 'auto op_0 = op::Mul("o0");' in src
        assert 'auto op_1 = op::AddN("o1");' in src
        assert "ops.push_back(op_0);" in src
        # 动态输入口展开（AddN.x）+ in_flat 游标
        assert "op_1.create_dynamic_input_byindex_x(dyn_n, in_flat);" in src
        assert "in_flat += 1;" in src
        # attr setter 逐节点注入（共享 partial + node_attrs/op_1）
        assert 'if (node_attrs.has("N")) {' in src
        assert 'op_1.set_attr_N(node_attrs.at("N").getInt());' in src
        # 通用接线循环（const char_t* 公开形态）
        assert "dst_op.SetInput(dst_port.c_str(), src_op, src_port.c_str());" in src
        assert 'dst_op.SetInput((uint32_t)dst.at("in_idx").getInt(), src_op, src_out_idx);' in src
        assert "dst_op.SetInput(dst_port.c_str(), dst_elem, src_op);" in src
        # dN Data/Const 二态
        assert 'op::Const("d" + to_string(data_idx))' in src
        assert 'op::Data("d" + to_string(data_idx))' in src
        # 图输出纯索引形态 + SetInputs
        assert "graph.SetOutputs(ttk_out_idx);" in src
        assert "graph.SetInputs(data_ops);" in src
        # 无单算子段落泄漏
        assert "op_inst" not in src

    def test_dynamic_output_combination(self):
        src = _render_template("geir_graph_template.cpp.j2", **_graph_ctx(("split_v", "cast")))
        assert "op_0.create_dynamic_output_y(dyn_out_n);" in src
        assert "graph.SetOutputs(ttk_out_idx);" in src
        assert "UpdateDynamicOutputDesc(ref_port.c_str()" in src
        # 纯索引形态：不再有端口名形态分支
        assert "ttk_out_name" not in src

    def test_proto_include_dedup(self):
        ctx = _graph_ctx(("mul", "cast", "conv_bias"))
        src = _render_template("geir_graph_template.cpp.j2", **ctx)
        assert src.count('#include "fixture_cast_proto.h"') == 1

    def test_shared_skeleton_present(self):
        """图模式与单算子共享同一 main 骨架（serve 循环/帧协议/profiling）。"""
        src = _render_template("geir_graph_template.cpp.j2", **_graph_ctx(("mul", "add_n")))
        op_src = _render_template("geir_op_template.cpp.j2", **_op_ctx("mul"))
        for skeleton in (
            "ttk_serve_mode",
            "write_fail_frame",
            "session->AddGraph(graph_id, graph, graph_options)",
            "session->RunGraph(graph_id, input_tensors, output_tensors)",
            "shape_unknown_indexes",
            "finalize_all();",
        ):
            assert skeleton in src, f"shared skeleton missing: {skeleton}"
            assert skeleton in op_src

    def test_graph_dtype_attr_value_form_dual_branch(self, monkeypatch):
        """图模板 Type 属性渲染运行时双分支：字符串值 → ge::DataType 存储，整数值 → int64。"""
        monkeypatch.setattr(gb, "ProtoLoader", lambda: None)
        ctx = _graph_ctx(("cast",))
        src = _render_template("geir_graph_template.cpp.j2", **ctx)
        assert 'op_0.Operator::SetAttr("dtype", StringToDataType(__v.getString()));' in src
        assert 'op_0.Operator::SetAttr("dtype", __v.getInt());' in src

    def test_format_map_conv3d_dhwcn(self):
        """conv3d filter 的 DHWCN 映射到 FORMAT_DHWCN，并渲染进 StringToFormat。"""
        from ttk.core_modules.geir.graph_builder import _FORMAT_TO_GE_ENUM, _resolve_format

        assert _FORMAT_TO_GE_ENUM["DHWCN"] == "FORMAT_DHWCN"
        assert _resolve_format("DHWCN") == "FORMAT_DHWCN"
        ctx = dict(_op_ctx("cast"))
        ctx["format_map"] = [(v, v) for v in _FORMAT_TO_GE_ENUM.values()]
        out = _render_template("geir_op_template.cpp.j2", **ctx)
        assert 'if (s == "FORMAT_DHWCN") return FORMAT_DHWCN;' in out
