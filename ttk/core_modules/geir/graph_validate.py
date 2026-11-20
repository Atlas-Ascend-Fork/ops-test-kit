#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir/graph_validate.py — 图模式用例校验与 `_resolved_graph` 固化。

纯函数、不依赖设备：``--validate`` 期全部可跑。入口 ``validate_graph_case(tc)``：
- proto 可达性（每节点 REG_OP 能被 ProtoLoader 找到）
- 边文法与结构、拓扑序、目的槽唯一
- 输入完整性、图输入使用
- 嵌套节点属性（ATTR 路由 / REQUIRED_ATTR / 同名常量定值）
- 图输出与数量对账、图级扁平输入（禁 TensorList 嵌套）
- 首版边界（``_check_mode_restrictions``）
- 缺省值推导：graph_edges 缺省边展开、缺省 graph_outputs

全过后把规范形态落 ``tc._resolved_graph``（可 pickle 纯数据，testcase 经 pickle 跨
worker 进程边界传递，``OpProtoInfo`` 不随之序列化，worker 进程内由 ProtoLoader
（Singleton）重查）。错误格式统一 ``"R{n}: <描述>"``。
"""

__all__ = ["GraphRef", "validate_graph_case", "graph_ref_str"]

import logging
import re
from collections import namedtuple
from typing import Dict, List, Optional

from .proto_loader import ProtoLoader

GraphRef = namedtuple("GraphRef", ["kind", "idx", "port", "elem"])
# GraphRef 字段语义：
#   kind  'd' = 图输入槽（Data/Const 二态，port/elem 恒 None）；'o' = graph_ops 节点
#   idx   'd'：input_shapes 图级扁平序（0 起）；'o'：graph_ops 声明序（0 起）
#   port  端口名（该节点 proto 声明；'d' 恒为 None，Data/Const 唯一输出）
#   elem  DYNAMIC_INPUT/DYNAMIC_OUTPUT 的元素序号（0 起）；固定口恒为 None
# namedtuple 不可变且可哈希：可 pickle 跨 worker；(idx, port, elem) 直接作
# 去重与 dst-key 查找表键。序列化（config JSON）为结构化四元组
# {"kind","idx","port","elem"}，ref 字符串文法只存在于 CSV 层。

_REF_RE = re.compile(r"^(d|o)(\d+)(?::(\w+)(?::(\d+))?)?$")
_NODE_KEY_RE = re.compile(r"^o(\d+)$")


def graph_ref_str(ref: GraphRef) -> str:
    """GraphRef → CSV 层文法字符串（日志/调试用；序列化走结构化四元组）。"""
    s = f"{ref.kind}{ref.idx}"
    if ref.port is not None:
        s += f":{ref.port}"
    if ref.elem is not None:
        s += f":{ref.elem}"
    return s


def _parse_ref(ref) -> GraphRef:
    """文法解析 ``dN`` / ``oN:port`` / ``oN:port:K`` → GraphRef（只管文法，范围校验归结构检查）。

    只管文法；索引范围校验需要上下文（节点数/输入数），交给 ``_check_edge_structure``。
    """
    m = _REF_RE.match(ref) if isinstance(ref, str) else None
    if not m:
        raise ValueError(f"invalid reference {ref!r} (expected 'dN' / 'oN:port' / 'oN:port:K')")
    kind, idx, port, elem = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    if kind == "d" and (port is not None or elem is not None):
        raise ValueError(f"dN takes no port (Data/Const has a single output): {ref!r}")
    return GraphRef(kind, idx, port, int(elem) if elem is not None else None)


def _safe_info_of(op_name):
    """OpInfoKeeper 软查：纯 CPU 制备环境可能缺 CANN 配置，降级 None。"""
    try:
        from ttk.core_modules.kernel.info_keeper import OpInfoKeeper

        return OpInfoKeeper().info_of(op_name)
    except Exception as exc:  # noqa: BLE001
        logging.debugc("[graph] op info unavailable for %s: %s", op_name, exc)
        return None


# ----------------------------------------------------------------------------
# proto 可达性
# ----------------------------------------------------------------------------
def _resolve_node_protos(ops):
    """逐节点 ``ProtoLoader().get_op_info``。返回 (protos, alias_table, errors)。

    protos 与 ops 等长（缺失位置为 None）；alias_table = {op_name: node_idx}，
    同名出现多次的算子名禁用作 attributes 嵌套键别名（指称不唯一）。
    """
    loader = ProtoLoader()
    protos: List[Optional[object]] = []
    errors: List[str] = []
    name_nodes: Dict[str, list] = {}
    for i, op in enumerate(ops):
        proto = loader.get_op_info(op)
        protos.append(proto)
        if proto is None:
            errors.append(f"proto: node o{i} operator '{op}' has no REG_OP declaration (op_proto header not reachable)")
            continue
        name_nodes.setdefault(op, []).append(i)
        # SOC 软查：查不到降 info 日志不 fail，纯 proto 无部署信息的节点合法，
        # 执行期编译兜底（平台 gating 由 soc_series 列 + pass 的 MeetRequirements 承担）
        if not _safe_info_of(op):
            logging.info("[graph] 节点 o%d (%s) 无 ops-info 条目，按纯 proto IR 建图", i, op)
    alias_table = {name: idxs[0] for name, idxs in name_nodes.items() if len(idxs) == 1}
    return protos, alias_table, errors


# ----------------------------------------------------------------------------
# 边解析与自动直链
# ----------------------------------------------------------------------------
def _parse_edges(edges_raw):
    """原始边（二元字符串组）→ [(GraphRef, GraphRef), ...]。

    返回 (edges, errors)：任一条边非法时 edges 为 None，errors 收集全部文法
    错误（含边序号，文法错误由 _parse_ref 给出）。错误随
    validate_graph_case 的返回列表上抛，用例据此标 GRAPH_DESC_INVALID；
    只记日志不返回会让 GeirTestcase.validate 视为校验成功（_resolved_graph
    未落），执行期退回单算子路径。graph_edges 经 FREE_EVAL
    可解出标量（如 ``1``/``True``）：迭代前先做顶层类型检查，标量形态报
    结构错误而不是 TypeError 崩掉解析表。
    """
    if not isinstance(edges_raw, (tuple, list)):
        return None, [
            f"graph_edges must be a sequence of (src, dst) pairs (got {edges_raw!r}, type {type(edges_raw).__name__})"
        ]
    edges = []
    errors: List[str] = []
    for i, edge in enumerate(edges_raw):
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            errors.append(f"edge[{i}] must be a (src, dst) pair: {edge!r}")
            continue
        try:
            src = _parse_ref(edge[0])
            dst = _parse_ref(edge[1])
        except ValueError as exc:
            errors.append(f"edge[{i}]: {exc}")
            continue
        edges.append((src, dst))
    if errors:
        return None, errors
    return edges, errors


def _expand_auto_chain(tc, protos):
    """graph_edges 缺省（仅单节点图）：各必选口按声明序接入 d0, d1, …。

    多节点图连线不可唯一推导，须显式声明全部边；节点含
    DYNAMIC_INPUT / OPTIONAL_INPUT 时同样要求显式。
    返回 (edges, errors)；errors 非空时 edges 为 None。
    """
    ops = tc.graph_ops
    errors: List[str] = []
    if len(protos) != 1:
        errors.append(
            f"edges: empty graph_edges is only supported for single-node graphs "
            f"(got {len(protos)} nodes); declare all edges explicitly"
        )
        return None, errors
    proto = protos[0]
    if proto.dynamic_inputs or proto.optional_inputs:
        errors.append(
            f"edges: empty graph_edges requires no DYNAMIC_INPUT/OPTIONAL_INPUT on any node, "
            f"node o0 ({ops[0]}) has one; declare graph_edges explicitly"
        )
        return None, errors

    edges = []
    # dN 分配：按口声明序接入 d0, d1, …
    for port in proto.inputs:
        edges.append((GraphRef("d", len(edges), None, None), GraphRef("o", 0, port, None)))
    n_inputs = len(tc.input_shapes or ())
    if len(edges) != n_inputs:
        errors.append(f"edge derivation requires {len(edges)} graph inputs, input_shapes declares {n_inputs}")
        return None, errors
    return edges, errors


# ----------------------------------------------------------------------------
# 规则检查
# ----------------------------------------------------------------------------
def _check_edge_structure(edges, protos, ops, n_inputs):
    """源/目的形态合法性、端口名存在于 proto 对应方向、索引范围。"""
    errors: List[str] = []
    n_ops = len(ops)
    for i, (src, dst) in enumerate(edges):
        tag = f"edge[{i}] ({graph_ref_str(src)}->{graph_ref_str(dst)})"
        # 目的侧仅算子输入口
        if dst.kind == "d" or dst.port is None:
            errors.append(f"{tag} destination must be an operator input port 'oN:port[:K]'")
            continue
        if dst.idx >= n_ops:
            errors.append(f"{tag} destination node o{dst.idx} out of range ({n_ops} nodes)")
            continue
        dst_proto = protos[dst.idx]
        if dst.port not in dst_proto.inputs:
            errors.append(f"{tag} destination port '{dst.port}' not in input declaration of {ops[dst.idx]}")
        elif dst.port in dst_proto.dynamic_inputs and dst.elem is None:
            errors.append(f"{tag} dynamic input port '{dst.port}' requires element index :K")
        elif dst.port not in dst_proto.dynamic_inputs and dst.elem is not None:
            errors.append(f"{tag} fixed input port '{dst.port}' does not allow element index :K")

        # 源侧 dN 或算子输出口
        if src.kind == "d":
            if src.idx >= n_inputs:
                errors.append(f"{tag} source d{src.idx} out of range ({n_inputs} graph inputs)")
            continue
        if src.port is None:
            errors.append(f"{tag} operator reference missing output port name")
            continue
        if src.idx >= n_ops:
            errors.append(f"{tag} source node o{src.idx} out of range ({n_ops} nodes)")
            continue
        src_proto = protos[src.idx]
        if src.port not in src_proto.outputs:
            errors.append(f"{tag} source port '{src.port}' not in output declaration of {ops[src.idx]}")
        elif src.port in src_proto.dynamic_outputs and src.elem is None:
            errors.append(f"{tag} dynamic output port '{src.port}' requires element index :K")
        elif src.port not in src_proto.dynamic_outputs and src.elem is not None:
            errors.append(f"{tag} fixed output port '{src.port}' does not allow element index :K")
    return errors


def _check_topology(edges):
    """拓扑序：边只允许低序→高序（dN 视作 -1 恒过）；相等即自环。"""
    errors = []
    for i, (src, dst) in enumerate(edges):
        if src.kind != "o" or dst.kind != "o":
            continue
        if src.idx >= dst.idx:
            errors.append(
                f"edge[{i}] ({graph_ref_str(src)}->{graph_ref_str(dst)}) violates topological order "
                f"(graph_ops must be topologically sorted, edges go from lower to higher index; equal forms a cycle)"
            )
    return errors


def _check_dst_uniqueness(edges):
    """同一 (节点, 口, 元素) 目的槽至多一条入边。"""
    errors = []
    seen = set()
    for _src, dst in edges:
        key = (dst.idx, dst.port, dst.elem)
        if key in seen:
            errors.append(
                f"destination slot o{dst.idx}:{dst.port}"
                + (f":{dst.elem}" if dst.elem is not None else "")
                + " has duplicate edges"
            )
            continue
        seen.add(key)
    return errors


def _check_input_completeness(edges, protos, ops):
    """必选口恰一条、可选口至多一条（重复入边单独报错）、动态口元素从 0 连续。

    返回 (dyn_in_counts, errors)；dyn_in_counts = {node_idx: {port: 元素数}}。
    """
    errors: List[str] = []
    by_port: Dict[tuple, list] = {}
    for _src, dst in edges:
        by_port.setdefault((dst.idx, dst.port), []).append(dst.elem)
    dyn_in_counts: Dict[int, Dict[str, int]] = {}
    for i, proto in enumerate(protos):
        optional = set(proto.optional_inputs)
        dynamic = set(proto.dynamic_inputs)
        for port in proto.inputs:
            elems = by_port.get((i, port), [])
            if port in dynamic:
                # 过滤 None 元素（缺 :K 的动态口引用）：结构检查已报告该错误，
                # 此处防御 sorted(None/int) 混排 TypeError，非法输入不应让直接
                # 调用方崩溃
                ks = sorted(e for e in set(elems) if e is not None)
                if not ks:
                    errors.append(f"node o{i} ({ops[i]}) dynamic input port '{port}' needs at least 1 element")
                    continue
                if ks != list(range(len(ks))):
                    errors.append(
                        f"node o{i} dynamic input port '{port}' element indices must be contiguous from 0, count = max index + 1: {ks}"
                    )
                else:
                    dyn_in_counts.setdefault(i, {})[port] = len(ks)
            elif port not in optional and not elems:
                errors.append(f"node o{i} ({ops[i]}) required input port '{port}' has no incoming edge")
            # OPTIONAL_INPUT 无边合法（= 槽位缺省）；多于一条由目的槽唯一性捕获
    return dyn_in_counts, errors


def _check_input_usage(edges, n_inputs):
    """每个 dN 至少作为一条边的源（悬空图输入报错）。"""
    errors = []
    used = {src.idx for src, _dst in edges if src.kind == "d"}
    for i in range(n_inputs):
        if i not in used:
            errors.append(f"graph input d{i} is not referenced by any edge (dangling)")
    return errors


def _resolve_node_attr_cfg(attributes, protos, alias_table):
    """嵌套 attributes → {node_idx: {键: 值}}。

    顶层键须为 oN 或唯一算子名别名；值必须是 dict。返回 (resolved, errors)。
    """
    resolved: Dict[int, dict] = {}
    errors: List[str] = []
    if not attributes:
        return resolved, errors
    if not isinstance(attributes, dict):
        errors.append("attrs: graph-mode attributes must be a nested dict (top-level keys oN)")
        return resolved, errors
    for key, val in attributes.items():
        m = _NODE_KEY_RE.match(str(key))
        if m:
            idx = int(m.group(1))
        elif str(key) in alias_table:
            idx = alias_table[str(key)]
        else:
            errors.append(f"attrs: top-level key {key!r} is not a node reference oN (nor a unique operator name alias)")
            continue
        if idx >= len(protos):
            errors.append(f"top-level key {key!r} node index out of range ({len(protos)} nodes)")
            continue
        if not isinstance(val, dict):
            errors.append(f"attrs: value of key {key!r} must be a dict (node attribute/const container)")
            continue
        resolved.setdefault(idx, {}).update(val)
    return resolved, errors


def _check_const_edges(edges, node_cfgs, protos):
    """同名定值：源必须是 dN；定值悬空报错；同一 dN 冲突定值报错。

    返回 (const_slots, errors)；const_slots = {dN: 定值}。dN 扇出到多个定值口
    时声明必须一致，静默"后写覆盖"会丢失前一处声明（golden/设备输入同源下
    存在假通过风险）。
    """
    errors: List[str] = []
    const_slots: Dict[int, object] = {}
    for i, proto in enumerate(protos):
        cfg = node_cfgs.get(i)
        if not cfg:
            continue
        input_names = set(proto.inputs)
        for key, val in cfg.items():
            if key not in input_names:
                continue  # ATTR 名 → _check_node_attrs 管
            if key in proto.dynamic_inputs:
                errors.append(
                    f"const on dynamic input port '{key}' of node o{i} is not supported (declare consts on fixed ports via dN + explicit edges)"
                )
                continue
            hits = [(src, dst) for src, dst in edges if dst.idx == i and dst.port == key]
            if not hits:
                errors.append(f"input port '{key}' of node o{i} has a const but no incoming edge (dangling const)")
                continue
            for src, _dst in hits:
                if src.kind != "d":
                    errors.append(
                        f"const port o{i}:{key} must take its source from dN (got {graph_ref_str(src)}); "
                        f"the slot is baked as Const, an operator output source is contradictory"
                    )
                    continue
                if src.idx in const_slots and const_slots[src.idx] != val:
                    errors.append(
                        f"graph input d{src.idx} fans out to multiple const ports with conflicting values: "
                        f"{const_slots[src.idx]!r} vs {val!r} (one const per dN)"
                    )
                    continue
                const_slots[src.idx] = val
    return const_slots, errors


def _check_node_attrs(node_cfgs, protos, ops):
    """逐节点键路由（ATTR 名）与 REQUIRED_ATTR 必给。"""
    errors: List[str] = []
    for i, proto in enumerate(protos):
        cfg = node_cfgs.get(i) or {}
        input_names = set(proto.inputs)
        attr_names = {name for name, _t in proto.attrs}
        for key in cfg:
            if not isinstance(key, str) or not key:
                # 空串/非字符串键（FREE_EVAL 可解出 {'': 1} / {1: 2}）：str(key)[0]
                # 对空串 IndexError，按无效属性键报校验错误
                errors.append(f"attrs: key {key!r} of node o{i} ({ops[i]}) is invalid (must be a non-empty string)")
                continue
            if key[0] in ("!", "#", "@"):
                continue  # 特殊前缀键沿用单算子内部属性语义
            if key in input_names:
                continue  # 输入口名同名 = 常量定值键，_check_const_edges 管
            if key not in attr_names:
                errors.append(
                    f"attrs: key {key!r} of node o{i} ({ops[i]}) is neither a proto ATTR nor an input port name"
                )
        for name, _t in proto.required_attrs:
            if name not in cfg:
                errors.append(f"attrs: REQUIRED_ATTR '{name}' of node o{i} ({ops[i]}) is missing")
    return errors


def _check_graph_outputs(tc, protos, edges):
    """图输出解析/推导、动态输出口引用完整性、数量对账。

    返回 (outputs, dyn_out_counts, errors)。
    """
    ops = tc.graph_ops
    errors: List[str] = []
    # 动态输出口被引用元素（边源先行收集；显式 outputs 解析后并入）
    dyn_out_refs: Dict[tuple, set] = {}
    for src, _dst in edges:
        if src.kind != "o" or src.idx >= len(protos) or src.port is None:
            continue  # 越界/缺口的错误由 _check_edge_structure 报，此处防越界访问
        if src.port in (protos[src.idx].dynamic_outputs or []):
            dyn_out_refs.setdefault((src.idx, src.port), set()).add(src.elem)

    outputs: List[GraphRef] = []
    raw = tc.graph_outputs
    if raw is not None and raw != () and not isinstance(raw, (tuple, list)):
        # graph_outputs 经 FREE_EVAL 可解出标量（如 1/True）：迭代前先做顶层
        # 类型检查，收敛为该行 outputs 错误而不是 TypeError 崩掉逐例校验
        errors.append(
            f"outputs: graph_outputs must be a tuple/list of reference strings (got {raw!r}, type {type(raw).__name__})"
        )
        return outputs, {}, errors
    if raw is None or raw == ():
        # 缺省：末节点全部输出（固定口按声明序、动态口按边引用的元素序展开）
        last = len(protos) - 1
        proto = protos[last]
        for port in proto.outputs:
            if port in (proto.dynamic_outputs or []):
                elems = sorted(dyn_out_refs.get((last, port), set()))
                if not elems:
                    errors.append(
                        f"dynamic output port '{port}' of the last node is not referenced by any edge; default output derivation cannot infer element count, declare graph_outputs explicitly"
                    )
                    continue
                for k in elems:
                    outputs.append(GraphRef("o", last, port, k))
            else:
                outputs.append(GraphRef("o", last, port, None))
    else:
        for item in raw:
            # 容忍 (('o1:y',),) 一层包装：元素为 ref 字符串或其单元素元组
            refs = item if isinstance(item, (tuple, list)) else (item,)
            for r in refs:
                try:
                    ref = _parse_ref(r)
                except ValueError as exc:
                    errors.append(f"outputs: {exc}")
                    continue
                if ref.kind != "o" or ref.port is None:
                    errors.append(f"graph output reference {r!r} must be an operator output port 'oN:port[:K]'")
                    continue
                if ref.idx >= len(protos):
                    errors.append(f"graph output reference {r!r} node index out of range ({len(protos)} nodes)")
                    continue
                proto = protos[ref.idx]
                if ref.port not in proto.outputs:
                    errors.append(f"graph output reference {r!r} port not in output declaration of {ops[ref.idx]}")
                    continue
                if ref.port in (proto.dynamic_outputs or []):
                    if ref.elem is None:
                        errors.append(f"graph output reference {r!r} on a dynamic port requires element index :K")
                        continue
                    dyn_out_refs.setdefault((ref.idx, ref.port), set()).add(ref.elem)
                elif ref.elem is not None:
                    errors.append(f"graph output reference {r!r} on a fixed port does not allow element index :K")
                    continue
                outputs.append(ref)

    # 动态输出口必须被引用（边或图输出）、元素序号从 0 连续
    dyn_out_counts: Dict[int, Dict[str, int]] = {}
    for i, proto in enumerate(protos):
        for port in proto.dynamic_outputs or []:
            elems = dyn_out_refs.get((i, port), set())
            if not elems:
                errors.append(
                    f"dynamic output port '{port}' of node o{i} ({ops[i]}) is not referenced by any edge or graph_outputs"
                )
                continue
            ks = sorted(elems)
            if ks != list(range(len(ks))):
                errors.append(
                    f"dynamic output port '{port}' of node o{i} element indices must be contiguous from 0: {ks}"
                )
            else:
                dyn_out_counts.setdefault(i, {})[port] = len(ks)

    # 数量对账（输出引用顺序即 output_shapes/output_dtypes/精度对齐的索引）
    if not errors:
        n_shapes = len(tc.output_shapes or ())
        n_dtypes = len(tc.output_dtypes or ())
        if len(outputs) != n_shapes or n_shapes != n_dtypes:
            errors.append(
                f"outputs: count mismatch: graph_outputs expands to {len(outputs)}, "
                f"output_shapes {n_shapes}, output_dtypes {n_dtypes}"
            )
    return outputs, dyn_out_counts, errors


def _check_flat_inputs(input_shapes, dyn_input_shapes):
    """图级扁平（禁 TensorList 嵌套），变参结构由"动态口多条边"表达。"""
    errors: List[str] = []
    shapes = input_shapes or ()
    for i, s in enumerate(shapes):
        if s is None:
            errors.append(
                f"flat: graph input d{i} is None; None placeholder is not supported in graph mode (optional input = no edge)"
            )
            continue
        if not _is_flat_int_seq(s):
            errors.append(f"flat: graph input d{i} must be a flat int tuple (TensorList nesting not supported): {s!r}")
    if dyn_input_shapes is not None and len(dyn_input_shapes) != len(shapes):
        errors.append(
            f"flat: dyn_input_shapes count {len(dyn_input_shapes)} does not match input_shapes count {len(shapes)}"
        )
    if dyn_input_shapes:
        for i, d in enumerate(dyn_input_shapes):
            if d is None:
                continue
            if not _is_flat_int_seq(d):
                errors.append(
                    f"flat: dyn_input_shapes[{i}] must be a flat int tuple (TensorList nesting not supported): {d!r}"
                )
    return errors


def _is_flat_int_seq(value) -> bool:
    if not isinstance(value, (tuple, list)):
        return False
    return all(isinstance(d, int) and not isinstance(d, bool) for d in value)


def _check_mode_restrictions(tc):
    """图模式使用边界集中一处。"""
    errors: List[str] = []
    if getattr(tc, "_graph_raw_out_str", False):
        errors.append(
            "mode: graph mode requires explicit output_shapes/output_ori_shapes (inference strings are single-operator semantics)"
        )
    if getattr(tc, "output_inplace_indexes", None):
        errors.append("mode: output_inplace_indexes is disabled in graph mode (GE inplace is node-level semantics)")
    for field in ("batch_axis", "batch_slice_info", "batch_seed"):
        if getattr(tc, field, None):
            errors.append(
                f"mode: {field} is not supported in graph mode (batch consistency semantics undefined for graphs)"
            )
    for i in range(1, 10):
        if getattr(tc, f"attributes{i}", None):
            errors.append(
                f"mode: graph mode ignores attributes{i} (flat merge is skipped in graph mode, the column has no effect)"
            )
            break
    return errors


# ----------------------------------------------------------------------------
# 编排入口
# ----------------------------------------------------------------------------
def validate_graph_case(tc) -> List[str]:
    """图模式校验编排：门控由调用方（GeirTestcase.validate 尾部）保证。

    全过后拼装 ``tc._resolved_graph``（单一真源：直链边展开、缺省输出推导均已
    完成的规范形态）；有错时返回错误列表（格式 "R{n}: <描述>"），不落位。
    """
    errors: List[str] = []
    ops = tuple(tc.graph_ops)
    if not ops:
        return errors

    protos, alias_table, proto_errors = _resolve_node_protos(ops)
    errors.extend(proto_errors)
    if any(p is None for p in protos):
        return errors  # proto 不可达短路：后续检查全部依赖 proto

    n_inputs = len(tc.input_shapes or ())

    # ① 边：显式解析或缺省推导（展开后的边原样过全部校验规则，推导结果不被特殊信任）
    if tc.graph_edges:
        edges, parse_errors = _parse_edges(tc.graph_edges)
        errors.extend(parse_errors)
        if edges is None:
            return errors  # 文法解析失败：错误已入返回列表，用例标 GRAPH_DESC_INVALID
    else:
        edges, chain_errors = _expand_auto_chain(tc, protos)
        errors.extend(chain_errors)
        if edges is None:
            return errors

    # ② 结构检查
    errors.extend(_check_edge_structure(edges, protos, ops, n_inputs))
    errors.extend(_check_topology(edges))
    errors.extend(_check_dst_uniqueness(edges))
    if errors:
        # 结构错误短路：后续聚合检查（元素序号排序等）假定
        # 边形态合法，缺 :K 的动态口引用混入 None 元素会 TypeError 崩掉整表
        # 解析，单用例校验失败必须收敛为错误列表而非异常
        return errors
    dyn_in_counts, complete_errors = _check_input_completeness(edges, protos, ops)
    errors.extend(complete_errors)
    errors.extend(_check_input_usage(edges, n_inputs))

    # ③ 嵌套 attributes：顶层键解析（oN/别名）→ 常量定值（同名）+ 节点属性
    node_cfgs, attr_cfg_errors = _resolve_node_attr_cfg(tc.attributes, protos, alias_table)
    errors.extend(attr_cfg_errors)
    const_slots, const_errors = _check_const_edges(edges, node_cfgs, protos)
    errors.extend(const_errors)
    errors.extend(_check_node_attrs(node_cfgs, protos, ops))
    # 节点 ATTR 值（输入口名同名定值键已解析走），builder 消费
    node_attrs = {
        idx: {k: v for k, v in cfg.items() if k not in set(protos[idx].inputs)} for idx, cfg in node_cfgs.items()
    }

    # ④ 图输出与动态口引用完整性
    outputs, dyn_out_counts, output_errors = _check_graph_outputs(tc, protos, edges)
    errors.extend(output_errors)

    # ⑤ 图级扁平输入 + 首版边界
    errors.extend(_check_flat_inputs(tc.input_shapes, getattr(tc, "dyn_input_shapes", None)))
    errors.extend(_check_mode_restrictions(tc))

    if errors:
        return errors

    # 全过 → 固化（可 pickle 纯数据；自动直链已展开为显式边、缺省输出已推导）
    tc._resolved_graph = {
        "ops": ops,
        "edges": tuple((src, dst) for src, dst in edges),
        "outputs": tuple(outputs),
        "dyn_in_counts": dyn_in_counts,
        "dyn_out_counts": dyn_out_counts,
        "node_attrs": node_attrs,
    }
    # 同名常量定值按图输入名（dN）并入 spec_tensors：输入覆盖与 is_const
    # 均按 dN 同名消费（与单算子输入名定值同一条链），const_slots 不外泄
    merged_spec = dict(getattr(tc, "_spec_tensors_cache", None) or {})
    for dN, val in const_slots.items():
        merged_spec[f"d{dN}"] = val
    tc._spec_tensors_cache = merged_spec
    edge_strs = ", ".join(f"{graph_ref_str(s)}->{graph_ref_str(d)}" for s, d in tc._resolved_graph["edges"])
    out_strs = ", ".join(graph_ref_str(o) for o in tc._resolved_graph["outputs"])
    logging.info(
        "[%s] graph resolved: ops=%s edges=[%s] outputs=[%s] dyn_in=%s",
        tc.testcase_name,
        ops,
        edge_strs,
        out_strs,
        dyn_in_counts,
    )
    return errors
