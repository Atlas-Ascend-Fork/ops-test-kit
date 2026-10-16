#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""kernel 通路 worker 入口：run（执行池，单用例全序列）+ run_compile（编译池，
dyn/cst/bin 分派）——阶段序经 notify_status 上报。"""

# Standard Packages
import logging
import os
from types import SimpleNamespace
from typing import Tuple

# Third-Party Packages
import numpy

from ttk.remote import DATA
from ttk.remote.client import dispatch_xpu
from ttk.switches import get_global_storage

from ...utilities import deep_flatten, get, render_table, resolve_numpy_dtypes
from ..common.comparison.batch_consistency import assemble_batch_summary
from ..common.dump import (
    DataDumper,
    manual_data_dump_store,
    resolve_manual_case,
    snapshot_manual_values,
)
from ..common.logging import build_single_log_dir, default_logging_config, release_case_stdout
from ..common.plugin import SpecResolver
from ..common.process import DeviceLock, gate_host_memory, process_context
from ..common.testcase.batch import resolve_deterministic_level
from ..common.testcase.input_data import reseed_per_testcase
from .compare import comparing, handle_profiling_result
from .compile.dynamic import dynamic_compilation
from .compile.static import static_compilation
from .compile.tiling import parse_tiling_data, render_tiling_data
from .execute import execute_on_device
from .golden import GoldenShapeRequiredError, gen_output
from .info_keeper import OpInfoKeeper
from .input_generate import gen_input
from .result import ComparisonResult, KernelExecuteResult, KernelReturnStructure
from .testcase import TestcaseKernel


def __gen_workspaces(workspaces: Tuple[int], debug_buf_size: int) -> Tuple[numpy.ndarray, ...]:
    workspaces = workspaces or ()
    workspace_byte_arrays = []
    for workspace_size in workspaces:
        if workspace_size:
            workspace_byte_arrays.append(numpy.random.randint(0, 255, size=(int(workspace_size),), dtype=numpy.uint8))
        else:
            workspace_byte_arrays.append(None)
    if debug_buf_size > 0:
        if not workspace_byte_arrays:
            workspace_byte_arrays.append(None)
        first_ws_size = 0 if workspace_byte_arrays[0] is None else workspace_byte_arrays[0].size
        workspace_byte_arrays[0] = numpy.random.randint(
            0, 255, size=(first_ws_size + debug_buf_size,), dtype=numpy.uint8
        )
    return tuple(workspace_byte_arrays)


def init_prof_results(context):
    """以三模式编译结果字符串初始化 prof_result 占位（未执行，cycle=编译结论）。"""
    dyn_compile_result = context.dyn_compile_result.compile_result
    cst_compile_result = context.cst_compile_result.compile_result
    bin_compile_result = context.bin_compile_result.compile_result

    context.dyn_prof_result = KernelExecuteResult(dyn_compile_result)
    context.cst_prof_result = KernelExecuteResult(cst_compile_result)
    context.bin_prof_result = KernelExecuteResult(bin_compile_result)


def early_end(context, print_content):
    """通用提前结束出口：以指定状态产出一行结果（异常/短路路径共用）。

    与 compile_fail_end 同构：perf 列为各模式编译结论（未启用模式经
    init_prof_results 占位为 CST_OFF/BIN_OFF/DYN_OFF）；precision 列
    OFF 模式透传 OFF 标记、其余模式标早退原因——避免"只跑 dyn 三列全
    失败"的误读，与正常结束行的 OFF 语义一致。
    """
    init_prof_results(context)
    compare_result = ComparisonResult(print_content)
    for typ in ("dyn", "cst", "bin"):
        value = getattr(context, f"{typ}_compile_result").compile_result
        if isinstance(value, str) and value.endswith("_OFF"):
            setattr(compare_result, f"{typ}_precision", value)
    return_structure = KernelReturnStructure()
    return_structure.construct(context, compare_result, print_content)
    __profiling_end_print(context, compare_result, print_content)
    return return_structure


def _dump_only_end(context: TestcaseKernel):
    """dump-only（--no-prof）数据落盘完成后的提前结束（MANUAL_DATA_PREPARED）。"""
    compare_result = ComparisonResult(None).set(
        "MANUAL_DATA_PREPARED",
        "MANUAL_DATA_PREPARED",
        "MANUAL_DATA_PREPARED",
        "PASS",
    )
    return_structure = KernelReturnStructure()
    return_structure.construct(context, compare_result, "PASS")
    __profiling_end_print(context, compare_result, "PASS")
    return return_structure


def compile_fail_end(context: TestcaseKernel):
    """编译失败提前结束：perf fake-pass 判定 + 编译结论入结果行。"""
    init_prof_results(context)
    passed = handle_profiling_result(context)
    compare_result = ComparisonResult(passed).set(
        context.dyn_compile_result.compile_result,
        context.cst_compile_result.compile_result,
        context.bin_compile_result.compile_result,
        passed,
    )
    return_structure = KernelReturnStructure()
    return_structure.construct(context, compare_result, passed)
    __profiling_end_print(context, compare_result, passed)
    return return_structure


def compile_only_end(context: TestcaseKernel):
    """--compile-only 提前结束：只出编译耗时/tiling 行，不执行不比对。"""
    init_prof_results(context)
    compare_result = ComparisonResult("COMPILE_ONLY")
    return_structure = KernelReturnStructure()
    return_structure.construct(context, compare_result, "COMPILE_ONLY")
    __compile_only_end_print(context)
    return return_structure


def _xpu_inputs(context):
    """XPU (torch/tf) needs the logical ori-shape arrays; NPU run-format
    (input_arrays, e.g. NC1HWC0) is unusable on non-NPU accelerators. Fallback if unset."""
    ori = getattr(context, "original_input_arrays", None)
    return ori or context.input_arrays


def _extract_third_party(xpu_results, priority):
    """从 priority provider 取 outputs。委托 ttk.remote.client。"""
    from ..common.xpu import extract_third_party

    return extract_third_party(xpu_results, priority)


def _do_xpu_profiling(context, xpu_mode, deadline_s=None, inputs=None):
    """Run XPU dispatch，return priority provider name（specs[0].provider）。
    resolve_providers 失败 → xpu_results={} + return None（→ _extract_third_party None → GOLDEN_FAILURE）。

    Kernel 适配层：从 context + OpInfoKeeper 提取参数，委托 client.dispatch_xpu。
    """
    try:
        op_info = OpInfoKeeper().info_of(context.op_name)
        input_names = [ipt["name"] for ipt in op_info["inputs"]] if op_info else []
    except Exception:
        input_names = []

    op_type = OpInfoKeeper().op_type_of(context.op_name)
    sw = get_global_storage()
    need_data = bool(xpu_mode & DATA)
    xpu_results, priority = dispatch_xpu(
        op_name=context.op_name,
        inputs=inputs if inputs is not None else _xpu_inputs(context),
        input_names=input_names,
        op_type=op_type,
        attributes=context.attributes if hasattr(context, "attributes") else {},
        input_formats=getattr(context, "input_formats", None),
        input_dtypes=getattr(context, "input_dtypes", None),
        testcase_name=getattr(context, "testcase_name", context.op_name),
        switches=sw,
        need_data=need_data,
        param_order=_kernel_param_order(op_info),
        deadline_s=deadline_s,
    )
    context.xpu_results = xpu_results
    return priority


def _kernel_param_order(op_info) -> list:
    """def.cpp param order (inputs then attrs) for server-side pool merge."""
    if not op_info:
        return []
    return [inp["name"] for inp in op_info["inputs"]] + [attr["name"] for attr in op_info["attr"]]


def _setup_profiling_env(context: TestcaseKernel):
    from .error_cleaner import clear_error_manager

    clear_error_manager()
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.change_name(context.testcase_name)
    _log_dir = build_single_log_dir(switches.test_mode, context.testcase_name, switches.root_path)
    default_logging_config(file_handler=switches.logging_to_file, testcase_name=context.testcase_name, log_dir=_log_dir)
    return switches, process_ctx


def _create_dump_store(context: TestcaseKernel):
    """dump 方向 store（--no-prof 落盘 / 全空目录首跑落盘）；异常 → PREPARE_FAILURE。"""
    switches = get_global_storage()
    try:
        return manual_data_dump_store(context, switches), None
    except Exception as exc:
        logging.exception("Manual Kernel data dump store failure")
        return None, early_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")


def _check_preconditions(context: TestcaseKernel, switches, process_ctx):
    process_ctx.notify_status("OnParseParameters")
    __apply_force_overrides(context)
    if context.is_valid:
        __parse_dynamic_tiling_data(context)
        __parse_binary_tiling_data(context)
    if not context.is_valid:
        return early_end(context, context.fail_reason)
    if switches.compile_only:
        return compile_only_end(context)
    if context.compile_failed():
        return compile_fail_end(context)
    return None


def _open_dump_only_store(context: TestcaseKernel, switches):
    """no-prof 写方向开仓；full-flow 返回 None（load/首跑落盘在加载侧决出）。"""
    if not switches.no_prof:
        return None, None
    return _create_dump_store(context)


def _resolve_manual_stores(context: TestcaseKernel, switches, process_ctx, dump_store):
    """load 方向：加载已存用例；目录全空首跑 → 返回落盘 store（生成后写入）。

    返回 (manual_case, dump_store, early_result)；early_result 非 None = 提前结束。
    """
    if switches.no_prof or not switches.manual_data_dirs:
        return None, dump_store, None
    try:
        manual_case = resolve_manual_case(
            context,
            "kernel",
            switches,
            before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
            require_goldens=switches.golden_mode != "XPU",
        )
    except Exception as exc:
        logging.exception("Manual Kernel data loading failure")
        return None, dump_store, early_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
    if manual_case is not None or not switches.dump_config.dump_input:
        return manual_case, dump_store, None
    # 目录全空的首跑：正常生成并落盘（后续运行即可加载）
    dump_store, early_result = _create_dump_store(context)
    return manual_case, dump_store, early_result


def _manual_data_dump(context: TestcaseKernel, switches, dump_store, prepared_inputs):
    """dump 方向收尾：落盘，no-prof 时提前结束。返回值非 None = 提前返回。"""
    if dump_store is None:
        return None
    if not switches.no_prof and switches.golden_mode == "XPU":
        # golden 由三方执行产出，CPU 侧无可落盘真值
        logging.warning(
            "[%s] golden-mode XPU: skip manual-data output (golden produced by third party)",
            context.testcase_name,
        )
        return None
    early_result = _write_manual_dump(context, dump_store, prepared_inputs)
    if early_result is not None:
        return early_result
    if switches.no_prof:
        return _dump_only_end(context)
    return None


def _generate_input(context: TestcaseKernel, switches, process_ctx, manual_case, dump_store):
    logging.debug(f"Expecting {context.input_bytes + context.output_bytes} bytes memory usage")
    process_ctx.notify_status("OnGenInput")
    try:
        if manual_case is None:
            gen_input(context)
        else:
            gen_input(context, stored_inputs=manual_case.inputs)
    except Exception as exc:
        logging.exception("Input data generation failure:")
        if manual_case is not None:
            return None, early_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
        return None, early_end(context, "INPUT_GEN_FAILURE")
    prepared_inputs = None
    if dump_store is not None and manual_case is None:
        # 生成态数据将写入 store：先快照（设备执行可能原地改动 backing storage）
        try:
            prepared_inputs = snapshot_manual_values(tuple(deep_flatten(context.input_arrays or ())), "input")
        except Exception as exc:
            logging.exception("Manual Kernel input snapshot failure")
            return None, early_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    return prepared_inputs, None


def _resolve_tolerance(context: TestcaseKernel):
    from ..common.comparison.resolve import needs_promote_golden as _needs_promote_golden
    from ..common.comparison.resolve import resolve_tolerance as _resolve_tolerance

    switches = get_global_storage()
    try:
        tolerance = SpecResolver(context.op_name, "kernel", getattr(switches, "plugin_path", None)).resolve_tolerance()
        pre_compare = SpecResolver(
            context.op_name, "kernel", getattr(switches, "plugin_path", None)
        ).resolve_pre_compare()
        custom_compare = SpecResolver(
            context.op_name, "kernel", getattr(switches, "plugin_path", None)
        ).resolve_compare()
        output_dtypes = resolve_numpy_dtypes(context.flat_output_dtypes)
        input_dtypes = resolve_numpy_dtypes(context.flat_input_dtypes)
        standards = _resolve_tolerance(
            tolerance,
            context.flat_precision_tolerances,
            context.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
            input_dtypes=input_dtypes,
        )
        need_3party = any(s.token == "cross_check" for s in standards)  # noqa: S105  # token 为比对标准名，非口令
        # cross_check（三方）与 mixed/mix_tolerance（单标杆，2.3 更高精度标杆）均要求 golden
        # 升精度；golden_mode=XPU 与 Promote 互斥（golden 由三方执行产出）；
        # Disable 是显式豁免（golden 不生成），升精度无从谈起，不得覆盖
        if _needs_promote_golden(standards) and switches.golden_mode not in ("XPU", "Disable"):
            context.golden_mode_override = "Promote"
        return SimpleNamespace(
            tolerance=tolerance,
            pre_compare=pre_compare,
            custom_compare=custom_compare,
            standards=standards,
            need_3party=need_3party,
        ), None
    except ValueError as e:
        logging.error("[%s] tolerance invalid: %s", context.op_name, e)
        return None, early_end(context, "TOLERANCE_INVALID")


def _generate_golden(context: TestcaseKernel, manual_case, tol_state):
    from ..common.comparison.resolve import resolve_tolerance as _resolve_tolerance

    switches = get_global_storage()
    process_ctx = process_context()
    stored_goldens = None
    if manual_case is not None:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            # 槽位声明经 testcase 协议（manual_data_golden_shapes/dtypes）——
            # 含 shape-unknown 元数据槽，读侧 CSV 可推导
            stored_goldens = manual_case.load_goldens(
                shapes=context.manual_data_golden_shapes,
                dtypes=context.manual_data_golden_dtypes,
            )
        except Exception as exc:
            logging.exception("Manual Kernel golden loading failure")
            return early_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
    try:
        if manual_case is None:
            if switches.golden_mode == "XPU":
                # golden 由三方执行产出（本侧参照不生成）；采集后回填
                context.golden_arrays = ["GOLDEN_FAILURE"]
            else:
                gen_output(context)
        else:
            gen_output(context, stored_goldens=stored_goldens)
    except Exception as exc:
        if isinstance(exc, GoldenShapeRequiredError):
            logging.error(
                "[%s] output shape of %s depends on input values (output_shape_unknown_indexes); "
                "cannot allocate the output buffer without a golden: %s. "
                "Provide a golden via --plugin and rerun.",
                context.testcase_name,
                context.op_name,
                exc,
            )
        else:
            logging.exception("Output buffer initialization data or golden data generation failure:")
        if manual_case is not None:
            return early_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
        return early_end(context, "OUTPUT_GEN_FAILURE")
    finally:
        if hasattr(context, "golden_mode_override"):
            del context.golden_mode_override
    if getattr(context, "output_shape_unknown_indexes", None):
        output_dtypes = resolve_numpy_dtypes(context.flat_output_dtypes)
        input_dtypes = resolve_numpy_dtypes(context.flat_input_dtypes)
        tol_state.standards = _resolve_tolerance(
            tol_state.tolerance,
            context.flat_precision_tolerances,
            context.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
            input_dtypes=input_dtypes,
        )
    return None


def _write_manual_dump(context: TestcaseKernel, dump_store, prepared_inputs):
    switches = get_global_storage()
    process_ctx = process_context()
    try:
        process_ctx.notify_status("OnWriteManualData")
        case_dir = dump_store.write_case(
            context,
            "kernel",
            prepared_inputs,
            context.golden_arrays,
            file_format=switches.dump_config.file_format,
            write_goldens=switches.dump_config.dump_golden,
        )
    except Exception as exc:
        logging.exception("Manual Kernel data dump failure")
        return early_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    logging.info("[%s] manual Kernel data dumped: %s", context.testcase_name, case_dir)
    return None


def _dispatch_xpu(context: TestcaseKernel, need_3party, golden_is_xpu, manual_case=None):
    """kernel 三方采集统一入口（dispatch/collect 两段式）。

    分流：manual 源 / golden-XPU / 自动降级 / 总闸关闭 → ready（内联同步完成，
    collect 即取）；否则 cross_check / PERF / dump_xpu → background（NPU 执行期
    重叠）。collect 统一返回 (third_parties, xpu_results, provider)。
    """
    from ..common.xpu import (
        MANUAL_XPU_PROVIDER,
        XpuDispatch,
        dump_manual_xpu_archive,
        load_manual_third_parties,
        manual_xpu_results,
        overlap_deadline_s,
        snapshot_xpu_inputs,
        xpu_join_timeout,
        xpu_mode,
        xpu_overlap_enabled,
    )

    switches = get_global_storage()
    process_ctx = process_context()
    need_data = need_3party or golden_is_xpu

    manual = load_manual_third_parties(context, manual_case) if need_data else None
    if manual is not None:
        # 双来源短路：CSV manual_xpu_binaries 显式路径 / manual-data case 目录槽位
        process_ctx.notify_status("OnLoadManualXpu")
        context.xpu_results = manual_xpu_results()
        dump_manual_xpu_archive(context, manual)
        return XpuDispatch.ready((manual, context.xpu_results, MANUAL_XPU_PROVIDER))

    mode = xpu_mode(switches, need_data)
    if mode == 0:
        return XpuDispatch.ready((None, None, None))
    if golden_is_xpu or not xpu_overlap_enabled():
        process_ctx.notify_status("OnXpuProfiling")
        return XpuDispatch.ready(_collect_xpu_remote(context, mode, None, None, need_3party))
    # 重叠：主线程快照输入（防 NPU 执行原地改写 vs 线程序列化读竞争），线程跑采集
    budget = overlap_deadline_s(switches)
    _xpu_inputs_snapshot = snapshot_xpu_inputs(_xpu_inputs(context))
    process_ctx.notify_status("OnXpuDispatch")
    return XpuDispatch.background(
        _collect_xpu_remote,
        context,
        mode,
        budget,
        _xpu_inputs_snapshot,
        need_3party,
        join_timeout=xpu_join_timeout(budget),
    )


def _collect_xpu_remote(context: TestcaseKernel, mode, deadline_s=None, inputs=None, need_3party=False):
    """kernel 远端采集（同步形态 = 现状语义；重叠形态由 _dispatch_xpu 以
    deadline_s/快照 inputs 调起）。返回 (third_parties, xpu_results, provider)。"""
    xpu_priority = _do_xpu_profiling(context, mode, deadline_s=deadline_s, inputs=inputs)
    third_parties_nested = _extract_third_party(getattr(context, "xpu_results", None), xpu_priority)
    if need_3party and third_parties_nested is None:
        logging.warning(
            "[%s] cross_check configured but no third_party output (no XPU / endpoint down); "
            "cross_check outputs will GOLDEN_FAILURE",
            context.op_name,
        )
    third_parties = list(deep_flatten(third_parties_nested)) if third_parties_nested is not None else None
    return third_parties, getattr(context, "xpu_results", None), (xpu_priority or "xpu")


def _gen_workspaces(context: TestcaseKernel):
    process_ctx = process_context()
    process_ctx.notify_status("OnGenWorkspace")
    context.dyn_workspace_arrays = __gen_workspaces(
        context.dyn_compile_result.workspaces, context.dyn_compile_result.debug_buf_size
    )
    context.cst_workspace_arrays = __gen_workspaces(
        context.cst_compile_result.workspaces, context.cst_compile_result.debug_buf_size
    )
    context.bin_workspace_arrays = __gen_workspaces(
        context.bin_compile_result.workspaces, context.bin_compile_result.debug_buf_size
    )


def _dump_prepared_data(context: TestcaseKernel):
    """落盘 input（含 tiling_data）与 golden；须在 golden-XPU 回填 golden_arrays 之前调用。"""
    process_ctx = process_context()
    process_ctx.notify_status("OnDumpInputDataIfRequired")
    __dump_input(context)
    process_ctx.notify_status("OnDumpGoldenDataIfRequired")
    __dump_golden(context)


def _assemble_xpu_golden(context: TestcaseKernel, third_parties, golden_is_xpu: bool):
    """golden-XPU 装配：三方输出即 golden；无输出 → 整例 FAIL（GOLDEN_FAILURE）。

    非 golden-XPU（flag False）直接放行；须在 _dump_prepared_data 之后、
    _device_profiling 之前调用（FAIL 不空跑设备）。
    """
    if not golden_is_xpu:
        return None
    if third_parties is None:
        logging.error(
            "[%s] golden-mode XPU: no third_party output (no third_party declaration / XPU down); case FAIL",
            context.op_name,
        )
        compare_result = ComparisonResult(None).set("GOLDEN_FAILURE", "GOLDEN_FAILURE", "GOLDEN_FAILURE", "FAIL")
        return_structure = KernelReturnStructure()
        return_structure.construct(context, compare_result, "GOLDEN_FAILURE")
        __profiling_end_print(context, compare_result, "GOLDEN_FAILURE")
        return return_structure
    context.golden_arrays = list(third_parties)
    return None


def _device_profiling(context: TestcaseKernel, dev_id, device_grant_events, device_granted_indices):
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.notify_status("OnAcquireLock")
    device_id = [dev_id]
    use_device = switches.mode.has_device()
    with DeviceLock(
        process_ctx,
        dev_id,
        use_device=use_device,
        grant_event=device_grant_events.get(dev_id),
        granted_idx=device_granted_indices.get(dev_id),
    ):
        context.device_id = device_id[0]
        process_ctx.notify_status("OnProfilingPrint")
        __profiling_print(context)
        process_ctx.notify_status("OnDynProfiling")
        if switches.backend == "npusim":
            from ..common.simulator import run_kernel_sim

            (context.dyn_prof_result, context.cst_prof_result, context.bin_prof_result) = run_kernel_sim(context)
        else:
            context.dyn_prof_result = execute_on_device(context, "dynamic")
            process_ctx.notify_status("OnCstProfiling")
            context.cst_prof_result = execute_on_device(context, "const")
            process_ctx.notify_status("OnBinProfiling")
            context.bin_prof_result = execute_on_device(context, "binary")


def _compare_and_finalize(context: TestcaseKernel, standards, third_parties, pre_compare, custom_compare):
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.notify_status("PostProfiling")
    passed = handle_profiling_result(context)
    process_ctx.notify_status("OnDumpOutputDataIfRequired")
    __dump_output(context)
    process_ctx.notify_status("OnComparison")
    __adapt_output_shape_unknown(context)
    goldens = context.golden_arrays
    dyn_outputs = context.dyn_prof_result.output_bytes
    cst_outputs = context.cst_prof_result.output_bytes
    bin_outputs = context.bin_prof_result.output_bytes
    output_dtypes = context.flat_output_dtypes
    if context.output_shape_unknown_indexes and goldens:
        # 尾接的 shape 元数据槽是 TTK/NPU 运行时协议（kernel 上报实际输出 shape），
        # 不是算子输出——不进精度比对。数据比对按 golden shape 分配的全量 buffer
        # 进行，shape 欠报/错报会以残留初始值不匹配或越界形式暴露
        goldens = goldens[:-1]
        dyn_outputs = dyn_outputs[:-1] if len(dyn_outputs) > 1 else dyn_outputs
        cst_outputs = cst_outputs[:-1] if len(cst_outputs) > 1 else cst_outputs
        bin_outputs = bin_outputs[:-1] if len(bin_outputs) > 1 else bin_outputs
        output_dtypes = output_dtypes[:-1]
    compare_result = comparing(
        context.dyn_compile_result.kernel_name,
        context.cst_compile_result.kernel_name,
        context.bin_compile_result.kernel_name,
        dyn_outputs,
        cst_outputs,
        bin_outputs,
        goldens,
        output_dtypes,
        standards=standards,
        third_parties=third_parties,
        testcase=context,
        pre_compare=pre_compare,
        custom_compare=custom_compare,
    )
    if compare_result.passed != "PASS" and switches.dump_config.dump_on_fail:
        __dump_on_fail(context)
    process_ctx.notify_status("OnReturning")
    return_structure = KernelReturnStructure()
    return_structure.construct(context, compare_result, passed)
    if resolve_deterministic_level(switches, context) == 3:
        # dl=3 batch 一致性：比对后切 output 0 逐样本 MD5 → batch_summary
        # （output_bytes 的 shape-tensor 附加段在尾部，output 0 即首个真实
        # 数据输出；dyn 模式为主比对轴）
        return_structure.batch_summary = assemble_batch_summary(
            context.dyn_prof_result.output_bytes, context, context.batch_consistency_id
        )
    __profiling_end_print(context, compare_result, passed)
    return return_structure


def run(
    context: TestcaseKernel, device_grant_events: dict, device_granted_indices: dict, dev_id: int
) -> KernelReturnStructure:
    """
    Universal Testcase Profiling Entrance
    """
    try:
        return _run_profile(context, device_grant_events, device_granted_indices, dev_id)
    finally:
        release_case_stdout()


def _run_profile(
    context: TestcaseKernel, device_grant_events: dict, device_granted_indices: dict, dev_id: int
) -> KernelReturnStructure:
    switches, process_ctx = _setup_profiling_env(context)
    # --seed 用例级播种先于一切数据生成（输入/golden/workspace），与派发解耦
    reseed_per_testcase(context, switches)

    # dump 方向：--no-prof 落盘（bridge 仅在写方向激活时设置 manual_data_dirs）
    dump_store, early_result = _open_dump_only_store(context, switches)
    if early_result is not None:
        return early_result

    early_result = _check_preconditions(context, switches, process_ctx)
    if early_result is not None:
        return early_result

    # host 内存闸门：入口位（校验/编译早退之后的首个重负载前）——manual 落盘
    # 加载、输入生成、golden、设备执行全在闸门之后
    gate_host_memory(switches.no_memory_check, process_ctx)

    manual_case, dump_store, early_result = _resolve_manual_stores(context, switches, process_ctx, dump_store)
    if early_result is not None:
        return early_result

    prepared_inputs, early_result = _generate_input(context, switches, process_ctx, manual_case, dump_store)
    if early_result is not None:
        return early_result

    process_ctx.notify_status("OnGenGolden")
    tol_state, early_result = _resolve_tolerance(context)
    if early_result is not None:
        return early_result

    early_result = _generate_golden(context, manual_case, tol_state)
    if early_result is not None:
        return early_result

    early_result = _manual_data_dump(context, switches, dump_store, prepared_inputs)
    if early_result is not None:
        return early_result

    golden_is_xpu = switches.golden_mode == "XPU"
    xpu = _dispatch_xpu(context, tol_state.need_3party, golden_is_xpu, manual_case)  # 三方 dispatch
    _gen_workspaces(context)
    _dump_prepared_data(context)
    if golden_is_xpu:
        # golden-XPU：三方即 golden，设备前收割（FAIL 不空跑设备 + golden 落盘契约）
        early_result = _xpu_promote_golden(context, xpu)
        if early_result is not None:
            return early_result
    _device_profiling(context, dev_id, device_grant_events, device_granted_indices)
    third_parties = _collect_xpu(context, switches, xpu)
    return _compare_and_finalize(
        context, tol_state.standards, third_parties, tol_state.pre_compare, tol_state.custom_compare
    )


def _xpu_promote_golden(context: TestcaseKernel, xpu):
    """golden-XPU 装配（设备前收割）：三方即 golden，缺失则整例 FAIL 提前返回。"""
    third_parties, _results, _provider = xpu.collect()
    early_result = _assemble_xpu_golden(context, third_parties, golden_is_xpu=True)
    if early_result is not None:
        xpu.discard()
    return early_result


def _collect_xpu(context: TestcaseKernel, switches, xpu):
    """NPU 执行完毕的三方收割（ready 即取/幂等；重叠形态 join）+ case 目录槽位落盘。"""
    process_ctx = process_context()
    if xpu.pending:
        process_ctx.notify_status("OnXpuCollect")
    third_parties, xpu_results, provider = xpu.collect()
    from ..common.xpu import dump_xpu_slot_files

    dump_xpu_slot_files(context, switches, third_parties, provider)
    return third_parties


def __profiling_end_print(context: TestcaseKernel, compare_result: ComparisonResult, passed: str):
    c = compare_result
    memory_oob = (
        "PASS"
        if all(getattr(context, f"{x}_prof_result").oob_status == "PASS" for x in ("dyn", "cst", "bin"))
        else "FAIL"
    )
    lines = [
        ("MODE", "Performance", "Comparison", "Memory check"),
        ("DYN", str(context.dyn_prof_result.cycle), str(c.dyn_precision), str(context.dyn_prof_result.oob)),
        ("CST", str(context.cst_prof_result.cycle), str(c.cst_precision), str(context.cst_prof_result.oob)),
        ("BIN", str(context.bin_prof_result.cycle), str(c.bin_precision), str(context.bin_prof_result.oob)),
        ("STATUS", str(passed), str(c.passed), memory_oob),
    ]
    logging.info("\n" + render_table(lines))


def __compile_only_end_print(context: TestcaseKernel):
    def __normalize(compile_time, tiling_time, tiling_key, block_dim, local_memory, kernel_name):
        if isinstance(tiling_time, (tuple, list)):
            tiling_time = numpy.median(tiling_time[1:]) if len(tiling_time) > 1 else tiling_time[0]
        return (
            f"{compile_time:.3f}" if isinstance(compile_time, float) else compile_time,
            tiling_time,
            tiling_key,
            block_dim,
            local_memory,
            kernel_name,
        )

    dc = context.dyn_compile_result
    cc, bc = context.cst_compile_result, context.bin_compile_result
    lines = [
        ("COMPILE", "Compile/s", "Tiling/s", "TilingKey", "BlockDim", "SimtUB/B", "KernelName"),
        (
            "DYN",
            *__normalize(
                dc.compile_time,
                dc.tiling_result.tiling_time,
                dc.tiling_key,
                dc.block_dim,
                dc.simt_ub_size,
                dc.kernel_name,
            ),
        ),
        (
            "CST",
            *__normalize(
                cc.compile_time, "\\\\\\\\\\\\", "\\\\\\\\\\\\", cc.block_dim, cc.simt_ub_size, cc.kernel_name
            ),
        ),
        (
            "BIN",
            *__normalize(
                bc.compile_time,
                bc.tiling_result.tiling_time,
                bc.tiling_key,
                bc.block_dim,
                bc.simt_ub_size,
                bc.kernel_name,
            ),
        ),
    ]
    logging.info("\n" + render_table(lines))


def __parse_binary_tiling_data(context: TestcaseKernel):
    # noinspection PyBroadException
    try:
        context.bin_tiling_data_bytes, context.bin_tuple_tiling_data = parse_tiling_data(
            context.bin_compile_result.tiling_data
        )
        context.bin_str_tiling_data = render_tiling_data(
            context.bin_tuple_tiling_data,
            context.bin_compile_result.compile_info,
            context.bin_compile_result.tiling_key,
        )
    except Exception:
        logging.exception("Binary tiling data parsing failure")
        context.bin_compile_result.compile_result = "TILING_PARSE_FAILURE"


def __parse_dynamic_tiling_data(context: TestcaseKernel):
    # noinspection PyBroadException
    try:
        context.dyn_tiling_data_bytes, context.dyn_tuple_tiling_data = parse_tiling_data(
            context.dyn_compile_result.tiling_data
        )
        context.dyn_str_tiling_data = render_tiling_data(
            context.dyn_tuple_tiling_data,
            context.dyn_compile_result.compile_info,
            context.dyn_compile_result.tiling_key,
        )
    except Exception:
        logging.exception("Dynamic tiling data parsing failure")
        context.dyn_compile_result.compile_result = "TILING_PARSE_FAILURE"


def __apply_force_overrides(context: TestcaseKernel):
    switches = get_global_storage()

    def __apply_manual_block_dim(container):
        if container:
            context.dyn_compile_result.block_dim = get(container, 0) or context.dyn_compile_result.block_dim
            context.cst_compile_result.block_dim = get(container, 1) or context.cst_compile_result.block_dim
            context.bin_compile_result.block_dim = get(container, 2) or context.bin_compile_result.block_dim

    def __clear_atomic_if_not_none(idx, if_none):
        if switches.force_clear_atomic[idx] is not None:
            return switches.force_clear_atomic[idx]
        return if_none

    def __set_manual_simt_share_memory_size(idx: int, mode: str):
        if switches.force_simt_ub_size[idx] is not None:
            compile_result = getattr(context, f"{mode}_compile_result")
            compile_result.simt_ub_size = switches.force_simt_ub_size[idx]

    # command line should have the higher priority
    __apply_manual_block_dim(switches.force_block_dim)
    context.dyn_clear_atomic = __clear_atomic_if_not_none(0, context.dyn_clear_atomic)
    context.cst_clear_atomic = __clear_atomic_if_not_none(1, context.cst_clear_atomic)
    context.bin_clear_atomic = __clear_atomic_if_not_none(2, context.bin_clear_atomic)
    __set_manual_simt_share_memory_size(0, "dyn")
    __set_manual_simt_share_memory_size(1, "cst")
    __set_manual_simt_share_memory_size(2, "bin")


def __print_get_shape(golden):
    return golden.shape if hasattr(golden, "shape") else golden


def __print_get_dtype(golden):
    return golden.dtype if hasattr(golden, "dtype") else golden


def __profiling_print(context: TestcaseKernel):
    if not (context.input_arrays or context.output_arrays):
        raise TypeError("Input or Output Data is not available, please check your custom function")
    flat_input_arrays = tuple(deep_flatten(context.input_arrays)) if context.input_arrays else ()
    logging.info(
        "\n====================================================================\n"
        "=======================================================\n"
        "==================================\n"
        f"Op Name: {context.op_name}\n"
        f"//////////// {get_global_storage().dyn_switches} //////////\n"
        f"Input Shape: {context.dyn_inputs}\n"
        f"Input Dtype: {tuple(str(ipt.dtype) if ipt is not None else None for ipt in flat_input_arrays)}\n"
        f"Input Actual Shape: {tuple(ipt.shape if ipt is not None else None for ipt in flat_input_arrays)}\n"
        f"Kernel Name: {context.dyn_compile_result.kernel_name}\n"
        f"Compilation Result: {context.dyn_compile_result.compile_result}\n"
        f"BlockDim: {context.dyn_compile_result.block_dim}\n"
        f"Workspace Bytes: {context.dyn_compile_result.workspaces}\n"
        f"Tiling Data Parsed Dict: {context.dyn_str_tiling_data}\n"
        f"Tiling Data Parsed Tuple(int32): {context.dyn_tuple_tiling_data}\n"
        f"Tiling Key: {context.dyn_compile_result.tiling_key} ({context.str_tiling_key()})\n"
        f"Clear Atomic: {context.dyn_clear_atomic}\n"
        f"Simt UB: {context.dyn_compile_result.simt_ub_size} Bytes\n"
        f"//////////// {get_global_storage().cst_switches} //////////\n"
        f"Kernel Name: {context.cst_compile_result.kernel_name}\n"
        f"Compilation Result: {context.cst_compile_result.compile_result}\n"
        f"BlockDim: {context.cst_compile_result.block_dim}\n"
        f"Workspace Bytes: {context.cst_compile_result.workspaces}\n"
        f"Clear Atomic: {context.cst_clear_atomic}\n"
        f"Simt UB: {context.cst_compile_result.simt_ub_size} Bytes\n"
        f"//////////// {get_global_storage().bin_switches} //////////\n"
        f"Kernel Name: {context.bin_compile_result.kernel_name}\n"
        f"Compilation Result: {context.bin_compile_result.compile_result}\n"
        f"BlockDim: {context.bin_compile_result.block_dim}\n"
        f"Workspace Bytes: {context.bin_compile_result.workspaces}\n"
        f"Tiling Data Parsed Dict: {context.bin_str_tiling_data}\n"
        f"Tiling Data Parsed Tuple(int32): {context.bin_tuple_tiling_data}\n"
        f"Tiling Key: {context.bin_compile_result.tiling_key} ({context.str_tiling_key(True)})\n"
        f"Clear Atomic: {context.bin_clear_atomic}\n"
        f"Simt UB: {context.bin_compile_result.simt_ub_size} Bytes\n"
        "////////////////////////////\n"
        f"Golden Shape: {tuple(__print_get_shape(ga) for ga in context.golden_arrays)}\n"
        f"Golden Dtype: {tuple(str(__print_get_dtype(ga)) for ga in context.golden_arrays)}\n"
        f"Output Shape: {context.output_shapes}\n"
        f"Output Dtype: {context.output_dtypes}\n"
        f"Input Data Range: {context.actual_input_data_ranges}\n"
        f"Precision Tolerance: {context.precision_tolerances}\n"
        f"Mode: {get_global_storage().mode.name}\n"
        f"PID: {os.getpid()}\n"
        f"Device: {context.device_id}\n"
        "==================================\n"
        "=======================================================\n"
        "===================================================================="
    )


def __make_dumper(context: TestcaseKernel) -> "DataDumper":
    return DataDumper(get_global_storage(), context.dump_file_prefix or context.testcase_name)


def __dump_input(context: TestcaseKernel) -> None:
    # 门控在此而非全交给 dumper：tiling_data 两条目不走 dump_input 开关，须随 input 槽一并门控
    dumper = __make_dumper(context)
    if dumper.cfg.dump_input:
        dumper.dump_inputs(tuple(deep_flatten(context.input_arrays or ())), segment="dyn")
        dumper.dump_named("tiling_data_dyn", context.dyn_tiling_data_bytes)
        dumper.dump_named("tiling_data_bin", context.bin_tiling_data_bytes)


def __dump_output(context: TestcaseKernel) -> None:
    dumper = __make_dumper(context)
    output_dtypes = resolve_numpy_dtypes(context.flat_output_dtypes)
    for typ in ("dyn", "cst", "bin"):
        output_bytes = getattr(context, f"{typ}_prof_result").output_bytes
        dumper.dump_outputs(output_bytes, dtypes=output_dtypes, segment=typ)


def __dump_golden(context: TestcaseKernel) -> None:
    __make_dumper(context).dump_goldens(context.golden_arrays)


def __dump_on_fail(context: TestcaseKernel) -> None:
    dumper = __make_dumper(context)
    inputs = tuple(deep_flatten(context.input_arrays or ()))
    outputs = {typ: getattr(context, f"{typ}_prof_result").output_bytes for typ in ("dyn", "cst", "bin")}
    dumper.dump_on_fail(
        inputs,
        outputs,
        context.golden_arrays,
        dtypes=resolve_numpy_dtypes(context.flat_output_dtypes),
        segment="dyn",
    )
    if not dumper.cfg.dump_input:
        dumper.dump_named("tiling_data_dyn", context.dyn_tiling_data_bytes)
        dumper.dump_named("tiling_data_bin", context.bin_tiling_data_bytes)


def __adapt_output_shape_unknown(context: TestcaseKernel):
    """
    Detect and transfer npu output shape tensor from uint32 to uint64 encoding.
    """
    if not context.output_shape_unknown_indexes:
        return
    for x in ("dyn", "cst", "bin"):
        # translate to uint64 for all.
        # considering stc tbe & dyn asc different implement.
        x_output_bytes = getattr(context, f"{x}_prof_result").output_bytes
        if not x_output_bytes:
            continue
        last_array = x_output_bytes[-1]
        if isinstance(last_array, str):
            continue
        np_array = numpy.frombuffer(last_array, dtype=numpy.uint8)
        if np_array[3] == 128:
            # uint64 implement.
            pass
        else:
            # uint32 implement. use half bytes.
            np_array = np_array[: np_array.size // 2]
            np_array = np_array.view(numpy.uint32).astype(numpy.uint64, copy=False)
            for i in range(np_array.size // 9):
                effective_num = np_array[i * 9]
                if effective_num <= 8:
                    for j in range(int(effective_num + 1), 9):
                        # uint64 cast view as uint32 will generate 0.
                        if np_array[i * 9 + j] == 0:
                            np_array[i * 9 + j] = 1
            # mark it as uint64 encoding. set uint64 highest bit as 1
            # SE says only the first one needs to be set.
            np_array.view(numpy.uint8)[3] = 128
            x_output_bytes[-1] = np_array


def run_compile(testcase: TestcaseKernel, mode: str, **_):
    """编译池 worker 入口：错误态清理 + 用例合法性检查 + dyn/cst/bin 分派。

    每任务重配到用例日志目录（追加写）——编译日志与 CANN tbe/ccec 打屏
    输出随任务归入该用例 ttk.log；任务结束释放镜像跟随（防与执行 worker
    的镜像双转发同一档案）。"""
    from .error_cleaner import clear_error_manager

    clear_error_manager()
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.change_name(testcase.testcase_name)
    _log_dir = build_single_log_dir(switches.test_mode, testcase.testcase_name, switches.root_path)
    default_logging_config(
        file_handler=switches.logging_to_file, testcase_name=testcase.testcase_name, log_dir=_log_dir
    )
    process_ctx.notify_status("InitCompilation")
    try:
        if not testcase.is_valid:
            return None

        if mode in ("Dyn", "Bin"):
            return dynamic_compilation(testcase, mode)
        if mode == "Cst":
            return static_compilation(testcase, mode)
        raise RuntimeError(f"Unknown mode {mode}")
    finally:
        release_case_stdout()
