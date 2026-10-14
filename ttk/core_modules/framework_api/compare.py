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
E2E 比对编排 —— golden 生成（三级回退编排）、精度比对、XPU 三方采集输入、
dump 落盘。
"""

import functools
import logging

from ttk.core_modules.common.dump import (
    DataDumper,
)
from ttk.utilities.container import nest_by_distribution
from ttk.utilities.dtypes import is_float_dtype

from ..common.comparison import CompareHooks, compare_with_hooks
from ..common.comparison.resolve import resolve_tolerance
from ..common.invoke import Invoke, golden_mode
from ..common.plugin import SpecResolver, get_plugin_function
from ..common.process import process_context
from .result import FrameworkApiReturnStructure


def e2e_xpu_inputs(testcase, raw_inputs):
    """raw_inputs (flat numpy) re-nested, pure outputs filtered → top-level input slots."""
    dist = testcase.tensor_list_dist
    nested = nest_by_distribution(list(raw_inputs), dist) if dist else list(raw_inputs)
    out_indices = set(testcase.output_tensor_indexes or ())
    # output_tensor_indexes 是 top-level 槽位索引（对齐 _check_output_configuration
    # 与 e2e_xpu_input_names）：TensorList 槽也只占一个槽位
    return [slot for real_idx, slot in enumerate(nested) if real_idx not in out_indices]


def e2e_xpu_input_dtypes(testcase):
    """Logical tensor dtypes (pure outputs filtered) for XPU schema.

    与 e2e_xpu_inputs 同款过滤（输出槽剔除、TensorList 保持嵌套），dtype 为
    CSV 声明值（如 complex32），与 raw_inputs 的 tensor_list_dist 重排一致。
    """
    dist = testcase.tensor_list_dist
    flat_dtypes = testcase.flat_tensor_dtypes or ()
    nested = nest_by_distribution(list(flat_dtypes), dist) if dist else list(flat_dtypes)
    out_indices = set(testcase.output_tensor_indexes or ())
    return [slot for real_idx, slot in enumerate(nested) if real_idx not in out_indices]


def e2e_xpu_input_names(testcase):
    """Input tensor param names (pure outputs filtered) for XPU schema."""
    plan = testcase.get_param_plan()
    if plan is None:
        return []
    out_indices = set(testcase.output_tensor_indexes or ())
    names = []
    real_idx = 0
    for p in plan.overload_params:
        if p.is_tensor_like and p.name != "out":
            if real_idx not in out_indices:
                names.append(p.name)
            real_idx += 1
    return names


def _make_dumper(testcase, switches) -> "DataDumper":
    return DataDumper(switches, testcase.testcase_name)


def dump_inputs(testcase, raw_inputs, switches):
    # 占位槽位（None/str）由 DataDumper 跳过但保留原 idx——不在此预过滤，
    # 否则文件名索引与 CSV 输出槽位错位
    _make_dumper(testcase, switches).dump_inputs(raw_inputs)


def dump_outputs(testcase, result_nps, switches, segment=None):
    """output 逐形态 dump（segment 区分，避免各形态同名互相覆盖）。"""
    _make_dumper(testcase, switches).dump_outputs(result_nps, segment=segment)


def dump_goldens(testcase, golden_nps, switches):
    _make_dumper(testcase, switches).dump_goldens(golden_nps)


def _dump_on_fail(testcase, raw_inputs, result_nps, golden_nps, switches, segment=None):
    """Force dump all data on precision failure (per-form segment)."""
    _make_dumper(testcase, switches).dump_on_fail(raw_inputs, result_nps, golden_nps, segment=segment)


def _extend_backward_dtypes(testcase, output_dtypes):
    """backward：梯度位 dtype = 被微输入 dtype（与输出位同长追加——判据/
    容差按位路由零改动；output_tensor_indexes 坐标系不受影响——梯度是追加位）。"""
    if not getattr(testcase, "backward_mode", False):
        return output_dtypes
    input_dtypes = testcase.flat_tensor_dtypes or ()
    grad_count = sum(1 for d in input_dtypes if is_float_dtype(d))
    return tuple(output_dtypes) + tuple(_matching_grad_dtypes(testcase, grad_count))


def _matching_grad_dtypes(testcase, grad_count):
    """梯度 dtype 逐位匹配浮点输入的声明 dtype（requires_grad 位的声明值）。"""
    floating = [d for d in (testcase.flat_tensor_dtypes or ()) if is_float_dtype(d)]
    return (floating * ((grad_count // max(len(floating), 1)) + 1))[:grad_count]


def generate_golden_data(testcase, raw_inputs, switches, backend, dump=True):
    """Generate golden outputs and dump them. Returns golden_nps list."""
    process_ctx = process_context()
    process_ctx.notify_status("OnGenGolden")
    if golden_mode(testcase, switches) == "XPU":
        # golden 由三方执行产出（XPU 旁路——CPU 参照不生成）；
        # 编排层在三方采集后以 third_parties 回填，失败 → GOLDEN_FAILURE 整例 FAIL。
        golden_nps = ["GOLDEN_FAILURE"]
    elif switches.golden_mode == "Disable":
        # switch 级预检（与 kernel/aclnn 的通路侧预检同款）：override=Promote
        # 叠加全局 Disable 时仍抑制——旧行为保持。
        golden_nps = ["SUPPRESSED"]
    else:
        from .golden import generate_golden

        # e2e 的"func"= 绑定好上下文的 runner（三级回退在 generate_golden 内，
        # 含 golden_api="disable"→SUPPRESSED）；占位/异常语义经 Invoke.golden 收敛。
        golden_nps = Invoke.golden(
            testcase,
            functools.partial(
                generate_golden, testcase, raw_inputs, switches.plugin_path, switches, backend.device_type()
            ),
            switches,
        )
    if dump:
        process_ctx.notify_status("OnDumpGolden")
        dump_goldens(testcase, golden_nps, switches)
    return golden_nps


def has_dynamic_golden_source(testcase, switches, manual_case=None):
    """Return whether a dynamic-slot case can avoid the CPU default Golden."""
    if switches.golden_mode == "Disable" or str(testcase.golden_api or "").lower() == "disable":
        return True
    if manual_case is not None and manual_case.has_goldens:
        return True
    # A non-disabled golden_api takes priority over plugins and receives the
    # unresolved None slot, so it is not a valid dynamic-slot Golden source.
    if testcase.golden_api:
        return False
    return get_plugin_function(testcase.api_name, "golden", "e2e", switches.plugin_path) is not None


def evaluate_eager_precision(
    testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct, third_parties=None
):
    """Compare eager mode results against golden, set return_struct. Returns True if pass."""
    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in result_nps)
    output_dtypes = _extend_backward_dtypes(testcase, output_dtypes)
    resolver = SpecResolver(testcase.api_name, "e2e", switches.plugin_path)
    standards = resolve_tolerance(
        resolver.resolve_tolerance(),
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    # 统一比对入口：SpecResolver 钩子 + compare_with_hooks——与
    # kernel/geir/aclnn 同一份实现。pre_compare 不再编排层原地改写（变换不跨
    # eager→graph 泄漏），golden 每比对独立复制，混合占位跳过钩子。
    try:
        precision_str, log_str, is_pass, metrics = compare_with_hooks(
            testcase,
            result_nps,
            golden_nps,
            standards,
            CompareHooks(
                output_dtypes=output_dtypes,
                third_parties=third_parties,
                pre_compare=resolver.resolve_pre_compare(),
                custom_compare=resolver.resolve_compare(),
            ),
        )
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Eager comparison failure")
        return_struct.construct("COMPARE_FAILURE", "FAIL", None, metrics={})
        return False

    if log_str:
        logging.debugc(f"\nComparing eager with golden\n{log_str}")

    precision_status = "SKIP" if is_pass is None else "PASS" if is_pass else "FAIL"
    if is_pass is False and switches.dump_config.dump_on_fail:  # SKIP 不触发 on-fail dump
        _dump_on_fail(testcase, raw_inputs, result_nps, golden_nps, switches, segment="eager")
    return_struct.construct(precision_str, precision_status, perf, metrics=metrics)
    return True


def evaluate_graph_precision(
    testcase, raw_inputs, graph_nps, golden_nps, switches, return_struct, mode, perf=None, third_parties=None
):
    """Compare graph mode results against golden.

    Args:
        mode: "static" or "dynamic"
        perf: ProfileResult from graph execution (may be None)
    """
    if not graph_nps:
        return_struct.construct("GRAPH_EXEC_FAILURE", "FAIL", None, mode=mode)
        return

    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in graph_nps)
    resolver = SpecResolver(testcase.api_name, "e2e", switches.plugin_path)
    standards = resolve_tolerance(
        resolver.resolve_tolerance(),
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    log_str, metrics = "", {}  # try 前初始化（防 except 路径 NameError）
    try:
        precision_str, log_str, is_pass, metrics = compare_with_hooks(
            testcase,
            graph_nps,
            golden_nps,
            standards,
            CompareHooks(
                output_dtypes=output_dtypes,
                third_parties=third_parties,
                pre_compare=resolver.resolve_pre_compare(),
                custom_compare=resolver.resolve_compare(),
            ),
        )
    except Exception:
        logging.exception(f"Graph {mode} comparison failure")
        precision_str, log_str, is_pass = "COMPARE_FAILURE", "", False

    if log_str:
        logging.debugc(f"\nComparing graph {mode} with golden\n{log_str}")

    status = "SKIP" if is_pass is None else "PASS" if is_pass else "FAIL"
    if is_pass is False and switches.dump_config.dump_on_fail:  # SKIP 不触发 on-fail dump
        # on-fail 补漏 segment 与常规 dump 词表对齐（graph_cst/dyn/aclgraph），
        # 与 result 列前缀共用 _MODE_PREFIX 映射
        segment = f"graph_{FrameworkApiReturnStructure._MODE_PREFIX[mode]}"
        _dump_on_fail(testcase, raw_inputs, graph_nps, golden_nps, switches, segment=segment)
    return_struct.construct(precision_str, status, perf, mode=mode, metrics=metrics)
