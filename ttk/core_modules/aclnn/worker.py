#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn worker: per-testcase orchestration of the staged pipeline.

Stage order for the single-device path: input generation -> tolerance
resolution -> golden generation (ahead of execution; the tolerance anchor
is the CSV-declared output dtype, so nothing depends on execution
results) -> npu preprocess hook + two-phase execution under the device
lock -> dynamic replay golden reload when the API returned shapes the CSV
could not declare -> output dump -> third-party collection -> comparison.

Multi-device (HCCL) cases keep golden after execution: the multi-device
golden consumes each rank's execution result by construction.
"""

__all__ = ["run"]


# Standard Packages
import logging
import os

# Third-Party Packages
from ttk.switches import get_global_storage

from ...utilities import (
    nest_by_distribution,
    render_table,
    resolve_numpy_dtypes,
)
from ..common.comparison.batch_consistency import assemble_batch_summary
from ..common.comparison.resolve import needs_promote_golden
from ..common.comparison.resolve import resolve_tolerance as _resolve_tolerance
from ..common.dump import (
    DataDumper,
    manual_data_dump_store,
    resolve_manual_case,
    snapshot_manual_values,
)
from ..common.invoke import Invoke
from ..common.logging import build_single_log_dir, default_logging_config
from ..common.plugin import SpecResolver
from ..common.process import DeviceLock, MultiDeviceLock, gate_host_memory, process_context
from ..common.result import format_xpu_metrics
from ..common.testcase.batch import resolve_deterministic_level
from ..common.testcase.input_data import reseed_per_testcase
from .compare import Comparator
from .execute import execute_on_device
from .golden import GoldenGenerator
from .hccl_multi_device import compare_multi_device, profile_multi_device, release_retained_multi_device_resources
from .info_keeper import OpApiInfoKeeper
from .input_generate import InputGenerator
from .result import AclnnReturnStructure, ApiComparisonResult, ApiProfilingResult
from .testcase import TestcaseAclnn


def _profiling_end_print(context: TestcaseAclnn, return_structure: AclnnReturnStructure):
    prof_result = getattr(context, "prof_result", None)
    oob = getattr(prof_result, "oob", None)
    lines = [
        ("MODE", "Performance", "Comparison", "Memory check"),
        ("API", str(return_structure.op_perf_us), str(return_structure.precision), str(oob)),
        (
            "STATUS",
            return_structure.perf_status,
            str(return_structure.precision_status),
            "PASS" if oob == "PASS" else "FAIL",
        ),
    ]
    logging.info("\n" + render_table(lines))


def _profiling_print(context: TestcaseAclnn, dev_id: int):
    attr_list = []
    for k, v in context.pure_attrs.items():
        attr_list.append(f"{k}: {v}")
    attrs = "\n".join(attr_list)
    scalars = context.scalars or ()
    scalar_values = []
    for group in scalars:
        if isinstance(group, (list, tuple)):
            scalar_values.append(tuple(s.item() if s is not None else None for s in group))
        else:
            scalar_values.append(group.item() if group is not None else None)
    scalar_values = tuple(scalar_values)
    logging.info(
        "\n"
        "====================================================================\n"
        "=======================================================\n"
        "==================================\n"
        f"Op Api Name: {context.api_name}\n"
        f"////////////////// Tensors //////////////\n"
        f"View Shape: {context.tensor_view_shapes}\n"
        f"View Stride: {context.tensor_view_strides}\n"
        f"View Offset: {context.tensor_view_offsets}\n"
        f"Storage Shape: {context.tensor_storage_shapes}\n"
        f"Format: {context.tensor_formats}\n"
        f"Dtype: {context.tensor_dtypes}\n"
        f"////////////////// Scalars //////////////\n"
        f"Value: {scalar_values}\n"
        f"Dtype: {context.scalar_dtypes}\n"
        f"//////////// Attributes/Arrays //////////\n"
        f"{attrs}\n"
        f"/////////////////////////////////////////\n"
        f"Output Shape: {context.flat_output_view_shapes}\n"
        f"Output Dtype: {context.flat_output_dtypes}\n"
        f"Tensor Data Range: {context.actual_input_data_ranges}\n"
        f"Scalar Data Range: {context.full_scalar_data_ranges}\n"
        f"Precision Tolerance: {context.precision_tolerances}\n"
        f"Mode: {get_global_storage().mode.name}\n"
        f"PID: {os.getpid()}\n"
        f"Device: {dev_id}\n"
        "==================================\n"
        "=======================================================\n"
        "===================================================================="
    )


def prof_end(context, print_content):
    compare_result = ApiComparisonResult(print_content)
    return_structure = AclnnReturnStructure()
    return_structure.construct(context, compare_result)
    _profiling_end_print(context, return_structure)
    return return_structure


def _aclnn_xpu_inputs(context: TestcaseAclnn) -> list:
    """np_storages re-nested, pure outputs filtered → top-level input slots for XPU.

    np_storages is flat (one per flat tensor index). Re-nest by tensor_list_dist
    to get top-level slots, then skip pure_output_indexes (inplace outputs kept).
    """
    dist = context.tensor_list_dist
    nested = nest_by_distribution(list(context.np_storages), dist) if dist else list(context.np_storages)
    inputs = []
    real_idx = 0
    for slot in nested:
        if real_idx not in context.pure_output_indexes:
            inputs.append(slot)
        real_idx += len(slot) if isinstance(slot, (list, tuple)) else 1
    return inputs


def _aclnn_xpu_input_dtypes(context: TestcaseAclnn) -> list:
    """Logical tensor dtypes (pure outputs filtered) for XPU schema.

    与 _aclnn_xpu_inputs 同款过滤（pure outputs 剔除、TensorList 保持嵌套），
    保证 dtype 列表与 inputs 槽位一一对应。dtype 为 CSV 声明值（如 complex32），
    嵌套结构与 np_storages 的 tensor_list_dist 重排一致。
    """
    dist = context.tensor_list_dist
    flat_dtypes = context.flat_tensor_dtypes or ()
    nested = nest_by_distribution(list(flat_dtypes), dist) if dist else list(flat_dtypes)
    dtypes = []
    real_idx = 0
    for slot in nested:
        if real_idx not in context.pure_output_indexes:
            dtypes.append(slot)
        real_idx += len(slot) if isinstance(slot, (list, tuple)) else 1
    return dtypes


def _aclnn_xpu_input_names(context: TestcaseAclnn) -> list:
    """Input tensor param names (pure outputs filtered) for XPU schema."""
    op_api_info = OpApiInfoKeeper().info_of(context.api_name)
    if op_api_info is None:
        return []
    return [n for i, n in enumerate(op_api_info.tensors) if i not in context.pure_output_indexes]


def _aclnn_param_order(context: TestcaseAclnn) -> list:
    """C header param order (excluding pure outputs) for server-side pool merge."""
    op_api_info = OpApiInfoKeeper().info_of(context.api_name)
    if op_api_info is None:
        return []
    pure_output_names = {op_api_info.tensors[i] for i in context.pure_output_indexes if i < len(op_api_info.tensors)}
    return [
        name
        for name in op_api_info.params
        if name not in pure_output_names and name not in ("workspaceSize", "executor")
    ]


def _make_dumper(context: TestcaseAclnn) -> "DataDumper":
    return DataDumper(get_global_storage(), context.dump_file_prefix or context.testcase_name)


def _dump_input(context: TestcaseAclnn):
    dumper = _make_dumper(context)
    # 纯输出位不是真输入，不落 input 槽；动态 -1 槽是 NPU 侧值，
    # CPU/golden/dump 路径保持 None 标记（upstream !281）
    inputs = [
        t for idx, t in enumerate(context.flat_tensors) if t is not None and idx not in context.pure_output_indexes
    ]
    dumper.dump_inputs(inputs)


def _dump_output(context: TestcaseAclnn):
    dumper = _make_dumper(context)
    output_dtypes = resolve_numpy_dtypes(context.flat_output_dtypes)
    dumper.dump_outputs(context.prof_result.output_bytes, dtypes=output_dtypes)


def _dump_golden(context: TestcaseAclnn):
    _make_dumper(context).dump_goldens(context.golden_tensors)


def _dump_on_fail(context: TestcaseAclnn):
    dumper = _make_dumper(context)
    inputs = [t for idx, t in enumerate(context.flat_tensors) if idx not in context.pure_output_indexes]
    dumper.dump_on_fail(
        inputs,
        context.prof_result.output_bytes,
        context.golden_tensors,
        dtypes=resolve_numpy_dtypes(context.flat_output_dtypes),
    )


def _invoke_aclnn_npu_preprocess(context, switches, process_ctx, dev_id):
    """Run the optional hook inside the device lock before the main API.

    invoke 只管执行与形态校验；物化结果经 testcase.absorb_preprocess 落地
    （显式通道回填 / 隐式通道核验）。"""
    func = SpecResolver(context.api_name, "aclnn", switches.plugin_path).resolve_npu_preprocess()
    if func is None:
        return None
    if getattr(switches, "backend", None) != "npusim":
        import torch_npu

        torch_npu.npu.set_device(dev_id)
    process_ctx.notify_status("OnNpuPreprocess")
    result = Invoke.npu_preprocess(context, func, switches)
    return context.absorb_preprocess(result, context.get_param_plan())


def _run_aclnn_npu_preprocess(context, switches, process_ctx, dev_id):
    try:
        _invoke_aclnn_npu_preprocess(context, switches, process_ctx, dev_id)
    except RuntimeError as exc:
        if not str(exc).startswith(Invoke.NPU_PREPROCESS_FAILURE):
            raise
        logging.exception("[%s] NPU preprocess failure", context.testcase_name)
        return str(exc)
    return None


def _resolve_case_standards(context, switches):
    """Tolerance standards for the case, anchored on CSV-declared output dtypes.

    The anchor never depends on execution results, so resolution can run
    before device execution. cross_check standards additionally request a
    third-party reference and promote golden input precision.
    """
    plugin_path = getattr(switches, "plugin_path", None)
    tolerance = SpecResolver(context.api_name, "aclnn", plugin_path).resolve_tolerance()
    output_dtypes = resolve_numpy_dtypes(context.flat_output_dtypes)
    standards = _resolve_tolerance(
        tolerance,
        context.flat_precision_tolerances,
        context.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    need_3party = any(s.token == "cross_check" for s in standards)  # noqa: S105  # token 为精度标准类型名，非凭据
    return standards, need_3party


def _generate_golden_ahead(context, process_ctx, manual_case, switches) -> bool:
    """Generate (or statically load) golden before device execution.

    Returns True when manual replay goldens were loaded using the declared
    output shapes. Dynamic-shape cases — declared shapes unknown or the
    static load failing — defer the load until execution returned the
    actual shapes. Golden generation failure degrades to a placeholder
    that surfaces in the comparison row; execution still runs.
    """
    if manual_case is not None and manual_case.has_goldens:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            context.golden_tensors = manual_case.load_goldens(
                shapes=context.flat_output_view_shapes,
                dtypes=context.flat_output_dtypes,
            )
            _dump_golden(context)
            return True
        except Exception:
            logging.info(
                "[%s] static replay golden load failed; deferring to execution-returned shapes",
                context.testcase_name,
            )
            return False
    if switches.golden_mode == "XPU":
        # Golden is assembled from the third-party output after execution.
        return False
    process_ctx.notify_status("OnGenGolden")
    # noinspection PyBroadException
    try:
        GoldenGenerator(context).gen()
        process_ctx.notify_status("OnDumpGoldenDataIfRequired")
        _dump_golden(context)
    except Exception:
        logging.exception("Golden data generation failure")
        context.golden_tensors = ["GOLDEN_FAILURE"]
    return False


def _shapes_match(declared, actual) -> bool:
    """Whether declared output shapes equal the execution-returned shapes.

    Unknown declarations (None entries) or non-shape returns count as a
    mismatch so the replay golden gets reloaded with actual shapes.
    """
    if not isinstance(actual, (list, tuple)) or not isinstance(declared, (list, tuple)):
        return False
    if len(actual) != len(declared):
        return False
    for d, a in zip(declared, actual):
        if not isinstance(a, (list, tuple)) or d is None:
            return False
        try:
            if tuple(int(x) for x in a) != tuple(d):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _return_row(context, process_ctx, compare_result):
    process_ctx.notify_status("OnReturning")
    return_structure = AclnnReturnStructure()
    return_structure.construct(context, compare_result)
    _profiling_end_print(context, return_structure)
    return return_structure


def _dump_only_end(context, process_ctx):
    """dump-only（--no-prof）数据落盘完成后的提前结束（MANUAL_DATA_PREPARED）。"""
    compare_result = ApiComparisonResult(None).set("MANUAL_DATA_PREPARED", "PASS")
    return _return_row(context, process_ctx, compare_result)


def _setup_profiling_env(context: TestcaseAclnn):
    """日志环境 + 进程命名 + 首个状态点。"""
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.change_name(context.testcase_name)
    _log_dir = build_single_log_dir(switches.test_mode, context.testcase_name, switches.root_path)
    default_logging_config(file_handler=switches.logging_to_file, testcase_name=context.testcase_name, log_dir=_log_dir)
    process_ctx.notify_status("OnParseParameters")
    return switches, process_ctx


def _check_preconditions(context: TestcaseAclnn, switches, is_multi_device):
    """前置校验：用例合法性 / API 登记 / 动态 shape 约束。返回值非 None = 整行早退。"""
    if not context.is_valid:
        return prof_end(context, context.fail_reason)
    if not OpApiInfoKeeper().has_api(context.api_name):
        return prof_end(context, "OP_API_NOT_FOUND")
    dyn_indexes = context.dyn_input_slot_indexes
    # dump/prepare（--no-prof）方向允许动态槽记录 None（钩子留待 NPU 重放执行）；
    # 执行方向仍要求已注册 npu_preprocess 钩子。
    if (
        dyn_indexes
        and not switches.no_prof
        and SpecResolver(context.api_name, "aclnn", switches.plugin_path).resolve_npu_preprocess() is None
    ):
        return prof_end(context, "DYN_SHAPE_REQUIRES_NPU_PREPROCESS")
    if dyn_indexes and is_multi_device:
        return prof_end(context, "MULTI_DEVICE_UNSUPPORTED_FOR_DYN_SHAPE")
    return None


def _open_dump_only_store(context: TestcaseAclnn, switches):
    """dump 方向（--no-prof）开仓；full-flow 返回 None（load/首跑落盘在加载侧决出）。"""
    if not switches.no_prof:
        return None, None
    # -1 动态槽以 None 空标记落盘（input_N_none.bin），物化钩子留待重放执行
    # dump 方向：--no-prof 落盘（bridge 已保证写方向激活时设置 manual_data_dirs）
    try:
        return manual_data_dump_store(context, switches), None
    except Exception as exc:
        logging.exception("Manual data dump store failure")
        return None, prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")


def _resolve_manual_stores(context: TestcaseAclnn, switches, process_ctx, dump_store):
    """load 方向：加载已存用例；目录全空首跑 → 返回落盘 store（生成后写入）。

    返回 (manual_case, dump_store, early_result)；early_result 非 None = 提前结束。
    """
    if switches.no_prof or not switches.manual_data_dirs:
        return None, dump_store, None
    try:
        manual_case = resolve_manual_case(
            context,
            "aclnn",
            switches,
            before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
            require_goldens=False,
        )
    except Exception as exc:
        logging.exception("Manual data loading failure")
        return None, dump_store, prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
    if manual_case is not None or not switches.dump_config.dump_input:
        return manual_case, dump_store, None
    # load 方向目录全空的首跑：正常生成并落盘（后续运行即可加载）
    try:
        dump_store = manual_data_dump_store(context, switches)
    except Exception as exc:
        logging.exception("Manual data dump store failure")
        return None, dump_store, prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    return manual_case, dump_store, None


def _generate_input(context, process_ctx, manual_case, dump_store, dev_id, is_multi_device, device_ids):
    """输入生成（多卡跳过：逐 rank 线程自行生成）+ dump 快照/落盘。

    返回 (prepared_inputs, prepared_scalars, early_result)。
    """
    process_ctx.notify_status("OnGenInput")
    if is_multi_device and device_ids:
        context.my_rank = dev_id
        context.device_ids = tuple(device_ids)
        return None, None, None
    # noinspection PyBroadException
    try:
        input_generator = InputGenerator(context)
        if manual_case is not None:
            input_generator.gen(
                stored_inputs=manual_case.inputs,
                stored_scalars=manual_case.scalars,
            )
        else:
            input_generator.gen()
    except Exception as exc:
        logging.exception("Input data generation failure:")
        if manual_case is not None:
            return None, None, prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
        return None, None, prof_end(context, "INPUT_GEN_FAILURE")

    prepared_inputs = None
    prepared_scalars = None
    if dump_store is not None and manual_case is None:
        # 生成态数据将写入 store：先快照（设备执行可能原地改动 backing storage）
        try:
            prepared_inputs = snapshot_manual_values(context.np_storages, "input")
            prepared_scalars = snapshot_manual_values(context.flat_scalars or (), "scalar")
        except Exception as exc:
            logging.exception("Manual data snapshot failure")
            return None, None, prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")

    if dump_store is None:
        process_ctx.notify_status("OnDumpInputDataIfRequired")
        _dump_input(context)
    return prepared_inputs, prepared_scalars, None


def _device_profiling(
    context, switches, process_ctx, dev_id, device_grant_events, device_granted_indices, is_multi_device, device_ids
):
    """设备执行（设备锁内）：多卡线程组 / 单卡（npusim / 真机 + npu 预处理）。"""
    process_ctx.notify_status("OnAcquireLock")
    use_device = switches.mode.has_device()
    if is_multi_device and device_ids:
        with MultiDeviceLock(
            process_ctx,
            device_ids,
            use_device=use_device,
            grant_events=device_grant_events,
            granted_indices=device_granted_indices,
        ):
            process_ctx.notify_status("OnProfilingPrint")
            _profiling_print(context, dev_id)
            process_ctx.notify_status("OnProfiling")
            try:
                context.prof_result = profile_multi_device(context, list(device_ids), dev_id)
            except Exception:
                logging.exception("Multi-device profiling failed:")
                context.prof_result = ApiProfilingResult.fail("MULTI_DEVICE_FAILED")
        return None
    with DeviceLock(
        process_ctx,
        dev_id,
        use_device=use_device,
        grant_event=device_grant_events.get(dev_id),
        granted_idx=device_granted_indices.get(dev_id),
    ):
        preprocess_error = _run_aclnn_npu_preprocess(context, switches, process_ctx, dev_id)
        if preprocess_error:
            return prof_end(context, preprocess_error)
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(context, dev_id)
        process_ctx.notify_status("OnProfiling")
        if get_global_storage().backend == "npusim":
            from ..common.simulator import run_aclnn_sim

            context.prof_result = run_aclnn_sim(context, dev_id)
        else:
            context.prof_result = execute_on_device(context, dev_id)
    return None


def _execute_failure_row(context, process_ctx):
    """执行失败行：plog 详情或失败原因透传精度列（对齐 kernel），FAIL 落行。"""
    compare_result = ApiComparisonResult(None)
    execute_error = getattr(context.prof_result, "execute_error", None)
    if execute_error:
        one_line_error = " | ".join(line for line in execute_error.splitlines() if line.strip())
        compare_result.set(f"ACLNN_EXECUTE_FAILED: {one_line_error}", "FAIL")
    else:
        # 失败原因透传进精度列（对齐 kernel：失败串 ∉ PLACEHOLDER_PASS → FAIL），
        # 避免 "N/A" 吞掉 MULTI_DEVICE_FAILED/THREAD_EXECUTION_FAILED 等可追溯信息。
        # api_prof 非 str（失败但 profiler 行仍解析出）时用短原因，防止整表 CSV 塞爆。
        api_prof = context.prof_result.api_prof
        reason = api_prof if isinstance(api_prof, str) else "ACLNN_EXECUTE_FAILED"
        compare_result.set(reason, "FAIL")
    return _return_row(context, process_ctx, compare_result)


def _multi_device_row_or_none(context, switches, process_ctx, is_multi_device, device_ids):
    """多卡收尾行：golden 关闭 → EXECUTED 汇总；线程上下文在 → 多卡 golden 比对。"""
    if not (is_multi_device and device_ids):
        return None
    if switches.golden_mode == "Disable":
        thread_contexts = getattr(context, "_multi_device_thread_contexts", {})
        all_passed = all(did in thread_contexts and not thread_contexts[did].prof_result.failed() for did in device_ids)
        release_retained_multi_device_resources(context, device_ids)
        compare_result = ApiComparisonResult(None)
        # golden Disable：未做比对，执行正常呈现 SKIP 而非 PASS（对齐单卡路径）
        compare_result.set(
            "EXECUTED" if all_passed else "MULTI_DEVICE_EXECUTION_FAILURE",
            "SKIP" if all_passed else "FAIL",
        )
        return _return_row(context, process_ctx, compare_result)
    if hasattr(context, "_multi_device_thread_contexts"):
        compare_result = compare_multi_device(context, device_ids)
        return _return_row(context, process_ctx, compare_result)
    return None


def _reload_replay_golden_if_needed(context, process_ctx, manual_case, golden_loaded):
    """动态 shape 的 replay golden 二次装载。

    ACLNN C 接口执行期才返回输出 shape；静态装载（声明 shape）与执行返回
    不一致时，按实际 shape 重载 replay golden。返回值非 None = 提前结束。
    """
    if not (
        manual_case is not None
        and manual_case.has_goldens
        and not (
            golden_loaded and _shapes_match(context.flat_output_view_shapes, context.prof_result.output_view_shapes)
        )
    ):
        return None
    try:
        process_ctx.notify_status("OnLoadManualGolden")
        context.golden_tensors = manual_case.load_goldens(
            shapes=context.prof_result.output_view_shapes,
            dtypes=context.flat_output_dtypes,
        )
        _dump_golden(context)
    except Exception as exc:
        logging.exception("Manual golden loading failure")
        return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
    return None


def _manual_data_dump(
    context,
    switches,
    process_ctx,
    dump_store,
    manual_case,
    prepared_inputs,
    prepared_scalars,
    is_multi_device,
    device_ids,
):
    """in+golden 落盘——golden 生成后、设备执行前（对齐 kernel 时序）。

    full-flow 首跑（空目录回落）落当次实际产物（判据要求 Promote 时即升精度
    标杆），文件名 dtype 自描述、回放按其装载；golden 占位串跳过 golden 位；
    XPU golden 由三方产出整体跳过。no-prof（dump 方向）在此收尾不上板；
    多卡（输入由 rank 线程生成、golden 延后到执行段）无快照可落，亦不走
    dump-only 收尾。返回值非 None = 提前结束。
    """
    no_prof_end = switches.no_prof and not (is_multi_device and device_ids)
    golden_is_xpu = switches.golden_mode == "XPU" and manual_case is None
    if not (dump_store is not None and manual_case is None and not golden_is_xpu and prepared_inputs is not None):
        return _dump_only_end(context, process_ctx) if no_prof_end else None
    goldens = context.golden_tensors or ()
    write_goldens = bool(switches.dump_config.dump_golden) and not any(isinstance(g, str) for g in goldens)
    try:
        process_ctx.notify_status("OnWriteManualData")
        case_dir = dump_store.write_case(
            context,
            "aclnn",
            prepared_inputs,
            goldens if write_goldens else (),
            scalars=prepared_scalars,
            file_format=switches.dump_config.file_format,
            write_goldens=write_goldens,
        )
        logging.info(f"[{context.testcase_name}] manual data dumped: {case_dir}")
    except Exception as exc:
        logging.exception("Manual data dump failure")
        return prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    if no_prof_end:
        return _dump_only_end(context, process_ctx)
    return None


def _collect_third_parties(context, switches, process_ctx, need_3party, golden_is_xpu):
    """XPU 三方采集（cross_check 判据 / XPU golden 模式）。"""
    third_parties = None
    xpu_results = None
    from ..common.xpu import collect_third_party, xpu_mode_of

    xpu_mode = xpu_mode_of(switches, need_3party or golden_is_xpu)
    if xpu_mode:
        process_ctx.notify_status("OnXpuProfiling")
        _, third_parties, xpu_results = collect_third_party(
            op_name=context.api_name,
            inputs=_aclnn_xpu_inputs(context),
            input_names=_aclnn_xpu_input_names(context),
            op_type=None,
            attributes=context.xpu_attrs,
            testcase_name=context.testcase_name,
            switches=switches,
            need_data=need_3party or golden_is_xpu,
            param_order=_aclnn_param_order(context),
            input_dtypes=_aclnn_xpu_input_dtypes(context),
        )
        if need_3party and third_parties is None:
            logging.warning(
                "[%s] cross_check configured but no third_party output "
                "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                context.testcase_name,
            )
    return third_parties, xpu_results


def _compare_and_finalize(context, switches, process_ctx, standards, third_parties, xpu_results, golden_is_xpu):
    """XPU golden 装配 → 比对 → dump_on_fail → 结果行（确定性状态 + dl=3 批摘要）。"""
    if golden_is_xpu:
        # Third-party output is the golden; a missing one fails the case.
        if third_parties is None:
            logging.error(
                "[%s] golden-mode XPU: no third_party output (no third_party declaration / XPU down); case FAIL",
                context.testcase_name,
            )
            compare_result = ApiComparisonResult(None)
            compare_result.set("GOLDEN_FAILURE", "FAIL")
            return _return_row(context, process_ctx, compare_result)
        context.golden_tensors = list(third_parties)
    context.xpu_metrics = format_xpu_metrics(xpu_results) if xpu_results else {}
    process_ctx.notify_status("OnComparison")
    compare_result = Comparator(context, standards, third_parties).compare()
    if compare_result.passed != "PASS" and switches.dump_config.dump_on_fail:
        _dump_on_fail(context)
    process_ctx.notify_status("OnReturning")
    return_structure = AclnnReturnStructure()
    return_structure.construct(context, compare_result)
    return_structure.deterministic_status = getattr(context.prof_result, "deterministic_status", None)
    if resolve_deterministic_level(switches, context) == 3:
        # dl=3 batch 一致性：比对后切 output 0 逐样本 MD5 → batch_summary
        # （随 ReturnStructure 回传，主进程 cleanup 按 bcid 分组比对）
        return_structure.batch_summary = assemble_batch_summary(
            context.prof_result.output_bytes, context, context.batch_consistency_id
        )
    _profiling_end_print(context, return_structure)
    return return_structure


def _generate_golden_with_promote(context, switches, process_ctx, manual_case, is_multi_device, device_ids):
    """容差判据解析 + golden 先行生成（Promote 升精度注入 override，多卡延后）。

    The standards anchor is the CSV-declared output dtype, so resolution
    costs nothing upfront. Golden generation failure leaves a placeholder
    and does not block execution. Multi-device cases keep golden after
    execution (see the multi-device branch below).
    """
    standards, need_3party = _resolve_case_standards(context, switches)
    golden_loaded = False
    if needs_promote_golden(standards) and switches.golden_mode != "XPU":
        # cross_check 三方 / mixed·mix_tolerance 单标杆均升精度；XPU golden
        # mode is mutually exclusive with Promote.
        context.golden_mode_override = "Promote"
    try:
        if not (is_multi_device and device_ids):
            golden_loaded = _generate_golden_ahead(context, process_ctx, manual_case, switches)
    finally:
        if hasattr(context, "golden_mode_override"):
            del context.golden_mode_override
    return standards, need_3party, golden_loaded


def run(
    context: TestcaseAclnn,
    device_grant_events: dict,
    device_granted_indices: dict,
    dev_id: int,
    is_multi_device: bool = False,
    device_ids: list = None,
):
    """
    Op Api Testcase Profiling Entrance (module-level, pickled as Task.func)
    """
    return _run_profile(context, device_grant_events, device_granted_indices, dev_id, is_multi_device, device_ids)


def _run_profile(
    context: TestcaseAclnn,
    device_grant_events: dict,
    device_granted_indices: dict,
    dev_id: int,
    is_multi_device: bool,
    device_ids,
):
    """单卡/多卡测试编排：阶段函数各带显式失败出口（early_result 约定，对齐 kernel worker）。"""
    switches, process_ctx = _setup_profiling_env(context)
    # --seed 用例级播种先于一切数据生成（输入/golden/workspace），与派发解耦；
    # 多卡路径的 per-rank 重播种在其后生效，不受影响
    reseed_per_testcase(context, switches)

    ####################
    # Check whether there is need to do further test
    ####################
    early_result = _check_preconditions(context, switches, is_multi_device)
    if early_result is not None:
        return early_result

    # host 内存闸门：入口位（校验早退之后的首个重负载前）——manual 落盘加载、
    # 输入生成、golden、设备执行全在闸门之后
    gate_host_memory(switches.no_memory_check, process_ctx)
    logging.debug(f"Expecting {context.tensor_bytes} bytes memory usage")

    dump_store, early_result = _open_dump_only_store(context, switches)
    if early_result is not None:
        return early_result

    manual_case, dump_store, early_result = _resolve_manual_stores(context, switches, process_ctx, dump_store)
    if early_result is not None:
        return early_result

    prepared_inputs, prepared_scalars, early_result = _generate_input(
        context, process_ctx, manual_case, dump_store, dev_id, is_multi_device, device_ids
    )
    if early_result is not None:
        return early_result

    # ═══ Tolerance standards + golden, ahead of device execution ═══
    standards, need_3party, golden_loaded = _generate_golden_with_promote(
        context, switches, process_ctx, manual_case, is_multi_device, device_ids
    )

    # ═══ in+golden 落盘（golden 后、设备前）；no-prof 在此收尾 ═══
    early_result = _manual_data_dump(
        context,
        switches,
        process_ctx,
        dump_store,
        manual_case,
        prepared_inputs,
        prepared_scalars,
        is_multi_device,
        device_ids,
    )
    if early_result is not None:
        return early_result

    # ═══ Device execution under the device lock ═══
    early_result = _device_profiling(
        context, switches, process_ctx, dev_id, device_grant_events, device_granted_indices, is_multi_device, device_ids
    )
    if early_result is not None:
        return early_result

    if context.prof_result.failed():
        return _execute_failure_row(context, process_ctx)

    early_result = _multi_device_row_or_none(context, switches, process_ctx, is_multi_device, device_ids)
    if early_result is not None:
        return early_result

    # ═══ dump out（NPU 执行后）═══
    process_ctx.notify_status("OnDumpOutputDataIfRequired")
    _dump_output(context)

    # ═══ Replay golden reload for dynamic output shapes ═══
    early_result = _reload_replay_golden_if_needed(context, process_ctx, manual_case, golden_loaded)
    if early_result is not None:
        return early_result

    # ═══ XPU 三方采集（cross_check 判据 / XPU golden）═══
    golden_is_xpu = switches.golden_mode == "XPU" and manual_case is None
    third_parties = None
    xpu_results = None
    if not context.prof_result.failed():
        third_parties, xpu_results = _collect_third_parties(context, switches, process_ctx, need_3party, golden_is_xpu)

    return _compare_and_finalize(context, switches, process_ctx, standards, third_parties, xpu_results, golden_is_xpu)
