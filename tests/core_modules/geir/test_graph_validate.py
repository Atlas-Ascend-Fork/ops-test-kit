# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""图模式校验（graph_validate）单元测试。

覆盖：逐规则违例、四组合法用例（含负向对照）、缺省边展开
（逐边断言）、嵌套 attributes（别名糖 / REQUIRED_ATTR / 平铺 dN 常量）、
图级扁平输入、首版边界、``_resolved_graph`` pickle 往返。

ProtoLoader 经 mock 注入夹具实例（graph_validate 模块级引用可替换）；
OpInfoKeeper 软查替换为受控 info_map，保证跨环境确定性。
"""

import pickle
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

import ttk.core_modules.geir.graph_validate as gv
from ttk.core_modules.geir.graph_validate import GraphRef, graph_ref_str, validate_graph_case
from ttk.core_modules.geir.proto_loader import ProtoLoader

_PROTO_HEADERS = """
REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Mul)

REG_OP(AddN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(N, Int)
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddN)

REG_OP(Cast)
    .INPUT(x, TensorType({DT_FLOAT}))
    .ATTR(dtype, Type)
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Cast)

REG_OP(ReduceMean)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(axes, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ReduceMean)

REG_OP(SplitV)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(split_dim, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

REG_OP(ConvBias)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ConvBias)
"""


@pytest.fixture
def loader(tmp_path):
    inc = tmp_path / "opp" / "built-in" / "op_graph" / "inc"
    inc.mkdir(parents=True)
    (inc / "ops_proto_graphmode_ut.h").write_text(_PROTO_HEADERS, encoding="utf-8")
    return ProtoLoader(ascend_path=str(tmp_path))


@contextmanager
def _graph_env(loader, info_map=None):
    """graph_validate 的 ProtoLoader/_safe_info_of 双替换（受控、跨环境确定）。"""

    def fake_info_of(op_name):
        return (info_map or {}).get(op_name)

    with mock.patch.object(gv, "ProtoLoader", return_value=loader), mock.patch.object(
        gv, "_safe_info_of", fake_info_of
    ):
        yield


def _tc(**kw):
    """轻量 tc 桩：validate_graph_case 消费的字段子集。"""
    defaults = {
        "testcase_name": "ut_case",
        "graph_ops": (),
        "graph_edges": (),
        "graph_outputs": None,
        "input_shapes": (),
        "input_dtypes": (),
        "output_shapes": (),
        "output_dtypes": (),
        "attributes": None,
        "dyn_input_shapes": None,
        "input_data_ranges": None,
        "output_inplace_indexes": (),
        "batch_axis": None,
        "batch_slice_info": None,
        "batch_seed": None,
        "_graph_raw_out_str": False,
        "_resolved_graph": None,
    }
    defaults.update(kw)
    tc = SimpleNamespace(**defaults)
    for i in range(1, 10):
        setattr(tc, f"attributes{i}", None)
    return tc


def _errs(loader, tc, info_map=None):
    with _graph_env(loader, info_map):
        return validate_graph_case(tc)


# ═══════════════ 合法用例（四例） ═══════════════


def test_design_example2_mul_addn(loader):
    """例 2：Mul + AddN（动态口 + REQUIRED_ATTR）——合法且固化正确。"""
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(("d0", "o0:x1"), ("d2", "o0:x2"), ("o0:y", "o1:x:0"), ("d1", "o1:x:1")),
        input_shapes=((8, 1024), (8, 1024), (1,)),
        input_dtypes=("float32", "float32", "float32"),
        output_shapes=((8, 1024),),
        output_dtypes=("float32",),
        attributes={"o1": {"N": 2}},
    )
    assert _errs(loader, tc) == []
    rg = tc._resolved_graph
    assert rg["ops"] == ("mul", "add_n")
    assert (GraphRef("d", 0, None, None), GraphRef("o", 0, "x1", None)) in rg["edges"]
    assert (GraphRef("o", 0, "y", None), GraphRef("o", 1, "x", 0)) in rg["edges"]
    assert rg["outputs"] == (GraphRef("o", 1, "y", None),)
    assert rg["dyn_in_counts"] == {1: {"x": 2}}


def test_design_example3_const_slot(loader):
    """例 3：常量定值（嵌套同名：x2 对上 o0 输入口名 → d1 定值槽）→ spec_tensors 按 dN 落位。"""
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o0:y", "o1:x:0"), ("d2", "o1:x:1")),
        input_shapes=((8, 1024), (1,), (8, 1024)),
        input_dtypes=("float32",) * 3,
        output_shapes=((8, 1024),),
        output_dtypes=("float32",),
        attributes={"o0": {"x2": (1.5,)}, "o1": {"N": 2}},
    )
    assert _errs(loader, tc) == []
    assert tc._resolved_graph["node_attrs"][1] == {"N": 2}
    assert tc._spec_tensors_cache == {"d1": (1.5,)}  # 同名定值按图输入名 dN 并入 spec_tensors
    # 负向对照：删掉 x2 定值键 → d1 变回 Data，图仍合法（pass 拒绝融合由执行期验证）
    tc2 = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=tc.graph_edges,
        input_shapes=tc.input_shapes,
        input_dtypes=tc.input_dtypes,
        output_shapes=tc.output_shapes,
        output_dtypes=tc.output_dtypes,
        attributes={"o1": {"N": 2}},
    )
    assert _errs(loader, tc2) == []
    assert tc2._spec_tensors_cache == {}


def test_design_example4_fanout(loader):
    """例 4：输入扇出 d0 → x1/x2（扇出合法）。"""
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(("d0", "o0:x1"), ("d0", "o0:x2"), ("o0:y", "o1:x:0"), ("d1", "o1:x:1")),
        input_shapes=((8, 1024), (8, 1024)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 1024),),
        output_dtypes=("float32",),
        attributes={"o1": {"N": 2}},
    )
    assert _errs(loader, tc) == []


def test_design_example1_single_node(loader):
    """例 1（同构）：单节点 cast + 嵌套 attributes ——REQUIRED_ATTR 无、ATTR(dtype) 给出。"""
    tc = _tc(
        graph_ops=("cast",),
        graph_edges=(("d0", "o0:x"),),
        input_shapes=((8, 16),),
        input_dtypes=("float32",),
        output_shapes=((8, 16),),
        output_dtypes=("float32",),
        attributes={"o0": {"dtype": "float32"}},
    )
    assert _errs(loader, tc) == []
    assert tc._resolved_graph["outputs"] == (GraphRef("o", 0, "y", None),)


# ═══════════════ 自动直链（§3.4 逐边断言） ═══════════════


def test_auto_chain_single_node_expansion(loader):
    """单节点图 graph_edges 留空：各必选口按声明序接入 d0, d1, …。"""
    tc = _tc(
        graph_ops=("reduce_mean",),
        graph_edges=(),
        input_shapes=((8, 16), (1,)),
        input_dtypes=("float32", "int64"),
        output_shapes=((8,),),
        output_dtypes=("float32",),
    )
    assert _errs(loader, tc) == []
    rg = tc._resolved_graph
    assert set(rg["edges"]) == {
        (GraphRef("d", 0, None, None), GraphRef("o", 0, "x", None)),
        (GraphRef("d", 1, None, None), GraphRef("o", 0, "axes", None)),
    }
    assert rg["outputs"] == (GraphRef("o", 0, "y", None),)
    assert rg["dyn_in_counts"] == {}


def test_auto_chain_multi_node_rejected(loader):
    """多节点图留空报错，要求显式声明全部边。"""
    tc = _tc(
        graph_ops=("cast", "reduce_mean"),
        graph_edges=(),
        input_shapes=((8, 16), (1,)),
        input_dtypes=("float32", "int64"),
        output_shapes=((8, 16),),
        output_dtypes=("float32",),
    )
    errs = _errs(loader, tc)
    assert any("single-node" in e and "2 nodes" in e for e in errs), errs
    assert tc._resolved_graph is None


def test_auto_chain_input_count_mismatch(loader):
    tc = _tc(
        graph_ops=("reduce_mean",),
        graph_edges=(),
        input_shapes=((8, 16),),  # 需要 2 个图输入
        input_dtypes=("float32",),
        output_shapes=((8,),),
        output_dtypes=("float32",),
    )
    errs = _errs(loader, tc)
    assert any("edge derivation" in e for e in errs)
    assert tc._resolved_graph is None


def test_auto_chain_rejected_for_dynamic_or_optional(loader):
    # DYNAMIC_INPUT（add_n）→ 要求显式写边
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(),
        input_shapes=((8, 8), (8, 8), (8, 8), (8, 8)),
        input_dtypes=("float32",) * 4,
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o1": {"N": 2}},
    )
    errs = _errs(loader, tc)
    assert any("empty graph_edges" in e for e in errs)
    # OPTIONAL_INPUT（conv_bias）同样要求显式
    tc2 = _tc(
        graph_ops=("conv_bias",),
        graph_edges=(),
        input_shapes=((8, 8),),
        input_dtypes=("float32",),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    errs2 = _errs(loader, tc2)
    assert any("empty graph_edges" in e for e in errs2)


# ═══════════════ 逐规则违例 ═══════════════


def _mul_conv_case(**kw):
    base = {
        "graph_ops": ("conv_bias",),
        "graph_edges": (("d0", "o0:x"),),
        "input_shapes": ((8, 8),),
        "input_dtypes": ("float32",),
        "output_shapes": ((8, 8),),
        "output_dtypes": ("float32",),
    }
    base.update(kw)
    return _tc(**base)


def test_r1_missing_required_input(loader):
    """必选口缺入边（bias 有边、x 未喂）。"""
    tc = _mul_conv_case(graph_edges=(("d0", "o0:bias"),))
    errs = _errs(loader, tc)
    assert any("required input port 'x' has no incoming edge" in e for e in errs)


def test_r2_optional_port_no_edge_ok(loader):
    """OPTIONAL_INPUT 无边合法（= 槽位缺省）。"""
    assert _errs(loader, _mul_conv_case()) == []


def test_r3_duplicate_dst_slot(loader):
    tc = _mul_conv_case(graph_edges=(("d0", "o0:x"), ("d1", "o0:x")), input_shapes=((8, 8), (8, 8)))
    errs = _errs(loader, tc)
    assert any("has duplicate edges" in e for e in errs)


def test_r4_bad_forms(loader):
    # 目的为 dN
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("o0:y", "d0"), ("d0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    assert any("destination must be an operator input port" in e for e in _errs(loader, tc))
    # 源算子引用缺端口名
    tc = _tc(
        graph_ops=("mul", "cast"),
        graph_edges=(("o0", "o1:x"), ("d0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    assert any("missing output port name" in e for e in _errs(loader, tc))
    # 固定口带 :K
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("d0", "o0:x1:0"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    assert any("does not allow element index" in e for e in _errs(loader, tc))
    # 动态输入口缺 :K
    tc = _tc(
        graph_ops=("add_n",),
        graph_edges=(("d0", "o0:x"),),
        input_shapes=((8, 8),),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"N": 1}},
    )
    assert any("requires element index" in e for e in _errs(loader, tc))


def test_r5_topology_violation(loader):
    tc = _tc(
        graph_ops=("mul", "cast"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o1:y", "o0:x1")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    errs = _errs(loader, tc)
    assert any("violates topological order" in e for e in errs)


def test_r7_dangling_input(loader):
    tc = _mul_conv_case(
        graph_edges=(("d0", "o0:x"), ("d1", "o0:bias")),
        input_shapes=((8, 8), (8, 8), (8, 8)),
    )
    errs = _errs(loader, tc)
    assert any("d2 is not referenced" in e for e in errs)


def test_r8_const_edge_violations(loader):
    # 定值口源为算子输出口 → 语义矛盾
    tc = _tc(
        graph_ops=("mul", "conv_bias"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o0:y", "o1:x"), ("o0:y", "o1:bias")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o1": {"bias": (1.0,)}},
    )
    errs = _errs(loader, tc)
    assert any("must take its source from dN" in e for e in errs)
    # 定值悬空：键有定值但该口无边（conv_bias.bias 无边引用）
    tc = _tc(
        graph_ops=("conv_bias",),
        graph_edges=(("d0", "o0:x"),),
        input_shapes=((8, 8),),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"bias": (1.0,)}},
    )
    errs = _errs(loader, tc)
    assert any("dangling const" in e for e in errs)


def test_r9_unknown_port(loader):
    tc = _mul_conv_case(graph_edges=(("d0", "o0:zzz"),))
    errs = _errs(loader, tc)
    assert any("not in input declaration" in e for e in errs)


def test_r10_out_of_range(loader):
    # dN 越界
    tc = _mul_conv_case(graph_edges=(("d5", "o0:x"),))
    assert any("out of range" in e for e in _errs(loader, tc))
    # oN 越界
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o9:y", "o0:x1")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    assert any("out of range" in e for e in _errs(loader, tc))
    # 引用文法非法
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("dd0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    errs = _errs(loader, tc)
    # 解析错误必须进返回列表：只记日志返回空错误会让执行期退回单算子路径
    assert errs
    assert any("invalid reference" in e for e in errs)
    assert tc._resolved_graph is None


def test_r11_dynamic_output_rules(loader):
    # 动态输出口未被引用（缺省输出推导无元素数信息）
    tc = _tc(
        graph_ops=("split_v",),
        graph_edges=(("d0", "o0:x"), ("d1", "o0:split_dim")),
        input_shapes=((12, 24), (1,)),
        input_dtypes=("float32", "int64"),
        output_shapes=((12, 4), (12, 4)),
        output_dtypes=("float32", "float32"),
        attributes={"o0": {"num_split": 2}},
    )
    errs = _errs(loader, tc)
    assert any("is not referenced by any edge or graph_outputs" in e for e in errs)
    # 显式 outputs 引用元素 0/1 → 合法，展开两项
    tc2 = _tc(
        graph_ops=("split_v",),
        graph_edges=(("d0", "o0:x"), ("d1", "o0:split_dim")),
        graph_outputs=(("o0:y:0",), ("o0:y:1",)),
        input_shapes=((12, 24), (1,)),
        input_dtypes=("float32", "int64"),
        output_shapes=((12, 4), (12, 4)),
        output_dtypes=("float32", "float32"),
        attributes={"o0": {"num_split": 2}},
    )
    assert _errs(loader, tc2) == []
    assert tc2._resolved_graph["outputs"] == (
        GraphRef("o", 0, "y", 0),
        GraphRef("o", 0, "y", 1),
    )
    # 元素序号不连续（0 与 2）
    tc3 = _tc(
        graph_ops=("split_v",),
        graph_edges=(("d0", "o0:x"), ("d1", "o0:split_dim")),
        graph_outputs=(("o0:y:0",), ("o0:y:2",)),
        input_shapes=((12, 24), (1,)),
        input_dtypes=("float32", "int64"),
        output_shapes=((12, 4), (12, 4)),
        output_dtypes=("float32", "float32"),
        attributes={"o0": {"num_split": 2}},
    )
    errs = _errs(loader, tc3)
    assert any("contiguous" in e for e in errs)


def test_dynamic_input_contiguity(loader):
    """动态口元素序号从 0 连续（1/2 有洞）。"""
    tc = _tc(
        graph_ops=("add_n",),
        graph_edges=(("d0", "o0:x:1"), ("d1", "o0:x:2")),
        input_shapes=((8, 8), (8, 8)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"N": 2}},
    )
    errs = _errs(loader, tc)
    assert any("contiguous" in e for e in errs)


# ═══════════════ 嵌套 attributes 规则（§5） ═══════════════


def test_attrs_alias_and_errors(loader):
    # 唯一算子名别名（add_n → o1）
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o0:y", "o1:x:0"), ("d2", "o1:x:1")),
        input_shapes=((8, 8), (8, 8), (8, 8)),
        input_dtypes=("float32",) * 3,
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"add_n": {"N": 2}},
    )
    assert _errs(loader, tc) == []
    assert tc._resolved_graph["node_attrs"] == {1: {"N": 2}}
    # 未知顶层键
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"zzz": {}},
    )
    assert any("top-level key" in e for e in _errs(loader, tc))
    # 值非 dict
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": 5},
    )
    assert any("must be a dict" in e for e in _errs(loader, tc))
    # 节点内未知键（既非 ATTR 也非输入口名）
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"foo": 1}},
    )
    assert any("neither a proto ATTR" in e for e in _errs(loader, tc))
    # REQUIRED_ATTR 缺失（add_n 的 N）
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o0:y", "o1:x:0"), ("d2", "o1:x:1")),
        input_shapes=((8, 8), (8, 8), (8, 8)),
        input_dtypes=("float32",) * 3,
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={},
    )
    assert any("REQUIRED_ATTR" in e and "'N'" in e for e in _errs(loader, tc))


# ═══════════════ 图级扁平输入 + 首版边界 ═══════════════


def test_flat_inputs_rejects_tensorlist_nesting(loader):
    tc = _tc(
        graph_ops=("add_n",),
        graph_edges=(("d0", "o0:x:0"), ("d1", "o0:x:1")),
        input_shapes=(((8, 8), (8, 8)), (8, 8)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"N": 2}},
    )
    assert any(e.startswith("flat:") for e in _errs(loader, tc))


def test_mode_restrictions(loader):
    base = {
        "graph_ops": ("mul",),
        "graph_edges": (("d0", "o0:x1"), ("d1", "o0:x2")),
        "input_shapes": ((8, 8), (8, 8)),
        "input_dtypes": ("float32", "float32"),
        "output_shapes": ((8, 8),),
        "output_dtypes": ("float32",),
    }
    # output_shapes 推理字符串（快照）
    tc = _tc(**{**base, "_graph_raw_out_str": True})
    assert any("inference strings" in e for e in _errs(loader, tc))
    # inplace 禁用
    tc = _tc(**{**base, "output_inplace_indexes": (0,)})
    assert any("output_inplace_indexes" in e for e in _errs(loader, tc))
    # batch 三字段
    tc = _tc(**{**base, "batch_axis": (0,)})
    assert any("batch_axis" in e for e in _errs(loader, tc))
    # attributesN 填写报错
    tc = _tc(**base)
    tc.attributes1 = {"a": 1}
    assert any("attributes1" in e for e in _errs(loader, tc))


def test_output_count_reconciliation(loader):
    tc = _tc(
        graph_ops=("mul",),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2")),
        input_shapes=((8, 8), (8, 8)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 8), (8, 8)),  # 2 项，图输出仅 1
        output_dtypes=("float32", "float32"),
    )
    errs = _errs(loader, tc)
    assert any("count mismatch" in e for e in errs)


def test_graph_outputs_explicit_subset(loader):
    """显式图输出（子集导出）：只导 o0:y，引用顺序即输出序。"""
    tc = _tc(
        graph_ops=("mul", "cast"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o0:y", "o1:x")),
        graph_outputs=(("o0:y",),),
        input_shapes=((8, 8), (8, 8)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    assert _errs(loader, tc) == []
    assert tc._resolved_graph["outputs"] == (GraphRef("o", 0, "y", None),)


# ═══════════════ 数量可扩展性（任意 dN / 任意节点数） ═══════════════


def test_many_data_nodes_and_ops(loader):
    """6 个图输入 × 3 节点链（mul→mul→add_n 4 元素动态口）+ 中间扇出。

    dN 建图为运行期 config 循环、算子块按组合渲染——数量均不受限（例：
    mul_chain_addn_multi_in_01）。"""
    tc = _tc(
        graph_ops=("mul", "mul", "add_n"),
        graph_edges=(
            ("d0", "o0:x1"),
            ("d1", "o0:x2"),
            ("o0:y", "o1:x1"),
            ("d2", "o1:x2"),
            ("o1:y", "o2:x:0"),
            ("d3", "o2:x:1"),
            ("d4", "o2:x:2"),
            ("d5", "o2:x:3"),
        ),
        input_shapes=((8, 64),) * 6,
        input_dtypes=("float32",) * 6,
        output_shapes=((8, 64),),
        output_dtypes=("float32",),
        attributes={"o2": {"N": 4}},
    )
    assert _errs(loader, tc) == []
    rg = tc._resolved_graph
    assert len(rg["edges"]) == 8
    assert rg["dyn_in_counts"] == {2: {"x": 4}}
    assert rg["outputs"] == (GraphRef("o", 2, "y", None),)


def test_single_op_with_many_inputs(loader):
    """单节点多输入：conv_bias 型 1 节点 + 2 输入（可选口缺省）不受数量限制。"""
    tc = _tc(
        graph_ops=("conv_bias",),
        graph_edges=(("d0", "o0:x"),),
        input_shapes=((8, 8),),
        input_dtypes=("float32",),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
    )
    assert _errs(loader, tc) == []


def test_graph_outputs_count_scales(loader):
    """图输出数量同样可扩展：split_v 导出 6 个动态口元素。"""
    tc = _tc(
        graph_ops=("split_v",),
        graph_edges=(("d0", "o0:x"), ("d1", "o0:split_dim")),
        graph_outputs=tuple((f"o0:y:{k}",) for k in range(6)),
        input_shapes=((12, 24), (1,)),
        input_dtypes=("float32", "int64"),
        output_shapes=((12, 4),) * 6,
        output_dtypes=("float32",) * 6,
        attributes={"o0": {"num_split": 6}},
    )
    assert _errs(loader, tc) == []
    assert tc._resolved_graph["outputs"] == tuple(GraphRef("o", 0, "y", k) for k in range(6))
    assert tc._resolved_graph["dyn_out_counts"] == {0: {"y": 6}}


# ═══════════════ picklability ═══════════════


def test_resolved_graph_pickle_roundtrip(loader):
    tc = _tc(
        graph_ops=("mul", "add_n"),
        graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2"), ("o0:y", "o1:x:0"), ("d2", "o1:x:1")),
        input_shapes=((8, 1024), (1,), (8, 1024)),
        input_dtypes=("float32",) * 3,
        output_shapes=((8, 1024),),
        output_dtypes=("float32",),
        attributes={"o0": {"x2": (1.5,)}, "o1": {"N": 2}},
    )
    assert _errs(loader, tc) == []
    restored = pickle.loads(pickle.dumps(tc._resolved_graph))
    assert restored == tc._resolved_graph


def test_graph_ref_str_roundtrip(loader):
    assert graph_ref_str(GraphRef("d", 0, None, None)) == "d0"
    assert graph_ref_str(GraphRef("o", 1, "x", 0)) == "o1:x:0"
    assert graph_ref_str(GraphRef("o", 0, "y", None)) == "o0:y"


# ═══════════════ 解析健壮性回归 ═══════════════


def test_invalid_ref_errors_returned(loader):
    """非法引用（graph_edges=(('bad', 'o0:x:0'),)）必须返回错误列表。

    _parse_edges 若只记日志返回 None，调用方得到空错误列表——用例被
    当作校验成功（_resolved_graph 未落），执行期退回单算子路径。"""
    tc = _tc(
        graph_ops=("add_n",),
        graph_edges=(("bad", "o0:x:0"), ("d1", "o0:x:1")),
        input_shapes=((8, 8), (8, 8)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"N": 2}},
    )
    errs = _errs(loader, tc)
    assert errs, "invalid references must yield validation errors (non-empty list)"
    assert any("invalid reference" in e for e in errs)
    assert tc._resolved_graph is None


def test_invalid_ref_marks_case_invalid_end_to_end(loader):
    """端到端：走完整 GeirTestcase.validate() 链路，用例最终
    GRAPH_DESC_INVALID（而非被当作合法用例进入执行期）。"""
    from unittest import mock as _mock

    from ttk.core_modules.geir.testcase import GeirTestcase

    with _graph_env(loader):
        tc = GeirTestcase()
        tc.testcase_name = "rev_p1_bad_ref"
        tc.op_name = "mul"
        tc.graph_ops = ("mul",)
        tc.graph_edges = (("bad", "o0:x1"), ("d1", "o0:x2"))
        tc.input_shapes = ((8, 8), (8, 8))
        tc.input_dtypes = ("float32", "float32")
        tc.output_shapes = ((8, 8),)
        tc.output_ori_shapes = ((8, 8),)  # 直构用例无等价头回落，显式给出
        tc.output_dtypes = ("float32",)
        with _mock.patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as keeper:
            keeper.return_value.info_of.return_value = None
            tc.validate()
    assert tc.is_valid is False
    assert tc.fail_reason == "GRAPH_DESC_INVALID"
    assert tc._resolved_graph is None


def test_mixed_dynamic_refs_no_crash(loader):
    """o0:x 与 o0:x:0 混合（None/int 混排）不得 TypeError 崩掉
    解析表，须收敛为结构错误列表（动态口缺 :K 已由 R4 报告并短路）。"""
    tc = _tc(
        graph_ops=("add_n",),
        graph_edges=(("d0", "o0:x"), ("d1", "o0:x:0")),
        input_shapes=((8, 8), (8, 8)),
        input_dtypes=("float32", "float32"),
        output_shapes=((8, 8),),
        output_dtypes=("float32",),
        attributes={"o0": {"N": 2}},
    )
    errs = _errs(loader, tc)  # None/int 混排若无防御会 sorted() TypeError
    assert isinstance(errs, list)
    assert any("requires element index" in e for e in errs)
    assert tc._resolved_graph is None


def test_scalar_free_eval_fields_fail_not_crash(loader):
    """graph_edges/graph_outputs 经 FREE_EVAL 解出标量（1/True/字符串）时不得
    TypeError 崩掉校验（中断整份 CSV），须收敛为该行校验错误（用例标 GRAPH_DESC_INVALID）。"""
    for bad in (1, True, "d0"):
        tc = _tc(  # graph_edges 标量
            graph_ops=("add_n",),
            graph_edges=bad,
            input_shapes=((8, 8),),
            input_dtypes=("float32",),
            output_shapes=((8, 8),),
            output_dtypes=("float32",),
            attributes={"o0": {"N": 1}},
        )
        errs = _errs(loader, tc)
        assert any("graph_edges" in e for e in errs), errs
        assert tc._resolved_graph is None
        tc = _tc(  # graph_outputs 标量
            graph_ops=("mul",),
            graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2")),
            input_shapes=((8, 8), (8, 8)),
            output_shapes=((8, 8),),
            output_dtypes=("float32",),
            graph_outputs=bad,
        )
        errs = _errs(loader, tc)
        assert any(e.startswith("outputs:") and "must be a tuple/list" in e for e in errs), (bad, errs)
        assert tc._resolved_graph is None


def test_invalid_attr_key_reported(loader):
    """嵌套 attributes 的空串/非字符串键不得 IndexError
    （str(key)[0]），按无效属性键报校验错误。"""
    for bad_attrs in ({"o0": {"": 1}}, {"o0": {1: 2}}):
        tc = _tc(
            graph_ops=("mul",),
            graph_edges=(("d0", "o0:x1"), ("d1", "o0:x2")),
            input_shapes=((8, 8), (8, 8)),
            input_dtypes=("float32", "float32"),
            output_shapes=((8, 8),),
            output_dtypes=("float32",),
            attributes=bad_attrs,
        )
        errs = _errs(loader, tc)  # {'': 1} 的空键会触发 str(key)[0] IndexError
        assert any("is invalid (must be a non-empty string)" in e for e in errs), errs
        assert tc._resolved_graph is None
