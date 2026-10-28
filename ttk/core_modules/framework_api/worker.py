#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
E2E worker —— 子进程执行编排。

run 为 worker 顶层入口（经 Task.func 跨进程序列化，须保持模块级可导入）；
_run_profile / _run_profile_multi_device 编排 eager/graph 模式与 golden 生成/比对
（比对实现见 compare.py，eager 执行见 execute.py，图模式经 adapter.run_graph）。
"""

import contextlib
import copy
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ttk.core_modules.common.dump import (
    dump_manual_dataset,
    load_replay_goldens,
    manual_data_dump_store,
    resolve_manual_case,
    snapshot_manual_values,
)
from ttk.core_modules.common.testcase.batch import resolve_deterministic_level
from ttk.core_modules.common.testcase.input_data import reseed_per_testcase
from ttk.switches import get_global_storage
from ttk.utilities import render_table
from ttk.utilities.dtypes import resolve_numpy_dtypes

from ..common.backend import get_backend
from ..common.comparison.batch_consistency import assemble_batch_summary
from ..common.comparison.resolve import needs_promote_golden, resolve_tolerance
from ..common.invoke import golden_mode
from ..common.logging import build_single_log_dir, default_logging_config
from ..common.plugin import SpecResolver
from ..common.process import DeviceLock, MultiDeviceLock, gate_host_memory, process_context
from ..common.result import format_xpu_metrics
from .adapter import adapter_for_testcase
from .compare import (
    dump_goldens,
    dump_inputs,
    dump_outputs,
    e2e_xpu_input_dtypes,
    e2e_xpu_input_names,
    e2e_xpu_inputs,
    evaluate_eager_precision,
    evaluate_graph_precision,
)
from .execute import _capture_peak_memory, _execute_eager
from .golden import generate_golden_data, has_dynamic_golden_source
from .input_generate import generate_inputs, set_runtime_tensors
from .result import FrameworkApiReturnStructure
from .utils import (
    resolve_npu_preprocess_hook,
)

# MC2 系共享权重算子（x1/x2 跨 rank 同值，idx 1 权重按 rank 无关种子重生成）
_WEIGHT_SHARED_OPS = (
    "MatmulReduceScatter",
    "AllGatherMatmul",
    "MatmulAllReduce",
    "MatmulAlltoAll",
    "AlltoAllMatmul",
    "MatmulReduceScatterV2",
    "QuantMatmulAllReduce",
    "mm_reduce_scatter",
    "all_gather_mm",
    "mm_all_reduce",
    "mm_all_to_all",
    "all_to_all_mm",
    "npu_moe_distribute_combine",
)
_WEIGHT_SHARED_EXCLUDE_OPS = (
    "GroupedMatMul",
    "AlltoAllvGrouped",
    "BatchMatMul",
    "bmm_reduce_scatter",
    "bmm_reducescatter",
    "all_to_all_all_gather",
    "alltoall_allgather",
)


def _should_run_eager(switches) -> bool:
    """eager（单算子直调）模式执行开关，仅由 ``--eager`` 命令行控制。

    aclgraph 等图模式如需关闭 eager，显式传 ``--eager=false``，不做内置豁免。
    结果语义：eager 关闭时 eager_* 列留空，精度主判据经 ref_nps 回退链
    （result_nps or graph_*）与 construct 首写锚定规则迁至图模式。
    """
    return bool(getattr(switches, "eager_enabled", True))


def _mirror_cst_failure_on_dedup(testcase, switches, return_struct, error_text):
    """失败路径镜像：eager 阶段异常时若双开去重成立，graph_cst 列写入同源错误。

    镜像块（``_execute_phases``）位于 eager 执行之后，异常传播先于镜像赋值 →
    失败用例 GRAPH CST 列留空，偏离「两列恒同值」。此处按
    同一判据（``eager_graph_alias()=='cst'`` + 双开）补镜像：错误文本与
    eager 列同一字符串（调用方传同一份 ``_attach_plog_errors`` 产物——镜像内
    二次附加会因 plog 逐次累积而异文），perf 与输出无值可比对（保持空）。
    判据解析自身失败不得掩盖原异常路径。
    """
    try:
        if not (_should_run_eager(switches) and switches.cst_switches.enabled):
            return
        alias_fn = getattr(adapter_for_testcase(testcase), "eager_graph_alias", None)
        if callable(alias_fn) and alias_fn() == "cst":
            return_struct.graph_cst_precision = error_text
    except Exception:  # noqa: BLE001  # 镜像属尽力而为：任何失败不得掩盖原异常
        logging.debug("cst failure mirror skipped: %s", testcase.testcase_name, exc_info=True)


def _print_get_shape(arr):
    return arr.shape if hasattr(arr, "shape") else arr


def _print_get_dtype(arr):
    return arr.dtype if hasattr(arr, "dtype") else arr


def _profiling_print(testcase, backend, dev_id, switches):
    separator = "-" * 100
    attr_list = []
    for k, v in testcase.attributes.items() if testcase.attributes else []:
        attr_list.append(f"{k}: {v}")
    attrs = "\n".join(attr_list) if attr_list else "N/A"
    logging.info(
        f"\n{separator}\n"
        f"API Name: {testcase.api_name}\n"
        f"Golden API: {testcase.golden_api}\n"
        f"Backend: {backend.device_type()}\n"
        f"////////////// Tensors //////////////\n"
        f"Input View Shapes: {testcase.tensor_view_shapes}\n"
        f"Input Dtypes: {testcase.tensor_dtypes}\n"
        f"Input Storage Shapes: {testcase.tensor_storage_shapes}\n"
        f"Input Formats: {testcase.tensor_formats}\n"
        f"Input View Offsets: {testcase.tensor_view_offsets}\n"
        f"Input View Strides: {testcase.tensor_view_strides}\n"
        f"//////////// Output Info //////////\n"
        f"Output Tensor Indexes: {testcase.output_tensor_indexes}\n"
        f"//////////// Attributes //////////\n"
        f"{attrs}\n"
        f"//////////// Test Config //////////\n"
        f"Input Data Range: {testcase.input_data_ranges}\n"
        f"Precision Tolerance: {testcase.precision_tolerances}\n"
        f"Absolute Precision: {testcase.absolute_precision}\n"
        f"Mode: {switches.mode.name}\n"
        f"PID: {os.getpid()}\n"
        f"Device: {dev_id}\n"
        f"{separator}"
    )


_TABLE_CELL_WIDTH = 80


def _wrap_table_cell(text, width=_TABLE_CELL_WIDTH):
    """长单元格按固定宽度折行（render_table 边框按最长行定宽、不折行）。"""
    text = str(text)
    if len(text) <= width:
        return text
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def _profiling_end_print(testcase, return_struct, golden_nps=None, switches=None):
    """结果后屏显：kernel 同款 render_table（MODE × Device/CPU/Comparison）。

    内核明细表（msprof kernel_details）在前，结果总表收尾——与 kernel 通路
    以结果表结束的风格一致。"""
    golden_nps = golden_nps or []

    def _format_kernel_table(kernels_json, label=""):
        if not kernels_json:
            return ""
        try:
            kernels = json.loads(kernels_json)
        except (json.JSONDecodeError, TypeError):
            return f"\n{label} Kernel Details: {kernels_json}\n"
        if not kernels:
            return ""
        rows = [
            (f"{label} Kernels",),
            ("Name", "avg", "max", "min", "# of Calls"),
        ]
        for k in kernels:
            name = k.get("name", "")
            if len(name) > 40:
                name = name[:38] + ".."
            rows.append(
                (
                    name,
                    f"{k.get('avg', 0):.3f}",
                    f"{k.get('max', 0):.3f}",
                    f"{k.get('min', 0):.3f}",
                    str(k.get("calls", 0)),
                )
            )
        return "\n" + render_table(rows)

    lines = [("MODE", "Device/us", "CPU/us", "Comparison")]
    if return_struct.eager_precision is not None:
        lines.append(
            (
                "EAGER",
                str(return_struct.eager_device_perf_us),
                str(return_struct.eager_cpu_perf_us),
                _wrap_table_cell(return_struct.eager_precision),
            )
        )
    if return_struct.graph_cst_precision is not None:
        lines.append(
            (
                "GRAPH CST",
                str(return_struct.graph_cst_device_perf_us),
                str(return_struct.graph_cst_cpu_perf_us),
                _wrap_table_cell(return_struct.graph_cst_precision),
            )
        )
    if return_struct.graph_dyn_precision is not None:
        lines.append(
            (
                "GRAPH DYN",
                str(return_struct.graph_dyn_device_perf_us),
                str(return_struct.graph_dyn_cpu_perf_us),
                _wrap_table_cell(return_struct.graph_dyn_precision),
            )
        )
    if return_struct.graph_aclgraph_precision is not None:
        lines.append(
            (
                "GRAPH ACLGRAPH",
                str(return_struct.graph_aclgraph_device_perf_us),
                str(return_struct.graph_aclgraph_cpu_perf_us),
                _wrap_table_cell(return_struct.graph_aclgraph_precision),
            )
        )

    if len(lines) == 1 and return_struct.precision_status is None:
        return

    lines.append(("STATUS", "", "", str(return_struct.precision_status)))

    if return_struct.deterministic_status is not None:
        lines.append((f"DETERMINISTIC: {return_struct.deterministic_status}",))

    golden_shapes = tuple(_print_get_shape(g) if isinstance(g, np.ndarray) else g for g in golden_nps)
    golden_dtypes = tuple(str(_print_get_dtype(g)) if isinstance(g, np.ndarray) else g for g in golden_nps)
    lines.append((_wrap_table_cell(f"Golden Shapes: {golden_shapes}"),))
    lines.append((_wrap_table_cell(f"Golden Dtypes: {golden_dtypes}"),))

    msg_parts = []
    for details, label in (
        (return_struct.eager_kernel_details, "Eager"),
        (return_struct.graph_cst_kernel_details, "Graph Cst"),
        (return_struct.graph_dyn_kernel_details, "Graph Dyn"),
        (return_struct.graph_aclgraph_kernel_details, "Graph Aclgraph"),
    ):
        table = _format_kernel_table(details, label)
        if table:
            msg_parts.append(table)
    msg_parts.append("\n" + render_table(lines))
    logging.info("\n".join(msg_parts))


def run(testcase, device_grant_events, device_granted_indices, dev_id, is_multi_device=False, device_ids=None):
    """
    Framework API worker entry — executed in subprocess (module-level, pickled as Task.func).

    Args:
        testcase: TestcaseE2e
        device_grant_events: dict of device_id → Manager().Event()
        device_granted_indices: dict of device_id → Manager().Value('i', -1)
        dev_id: device ID (passed as kwarg by ProcessGroup)
        is_multi_device: whether this is a multi-device task
        device_ids: list of device IDs for multi-device tasks

    Returns:
        FrameworkApiReturnStructure
    """
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.change_name(testcase.testcase_name)

    _log_dir = build_single_log_dir(switches.test_mode, testcase.testcase_name, switches.root_path)
    default_logging_config(
        file_handler=switches.logging_to_file, testcase_name=testcase.testcase_name, log_dir=_log_dir
    )

    return_struct = FrameworkApiReturnStructure()

    if not testcase.is_valid:
        return_struct.eager_precision = testcase.fail_reason
        return_struct.precision_status = "FAIL"
        _profiling_end_print(testcase, return_struct)
        return return_struct

    # --seed 用例级播种先于一切数据生成（输入/golden），与派发解耦；多卡路径
    # 的 per-rank 重播种在其后生效，不受影响
    reseed_per_testcase(testcase, switches)

    gate_host_memory(switches.no_memory_check, process_ctx)
    try:
        backend = _get_or_create_backend(switches)
        deterministic_level = resolve_deterministic_level(switches, testcase)
        _ensure_deterministic_level(process_ctx, backend, testcase, deterministic_level)
        if is_multi_device and device_ids:
            testcase.device_ids = tuple(device_ids)
            _run_profile_multi_device(
                testcase, backend, device_grant_events, device_granted_indices, device_ids, switches, return_struct
            )
        else:
            _run_profile(
                testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct
            )
    except Exception as e:
        logging.error(f"[{testcase.testcase_name}] Error: {e}", exc_info=True)
        # 两列同串：eager 列与镜像列共用同一份 _attach_plog_errors 产物（单次
        # plog 读取——镜像内二次附加会因 plog 逐次累积而异文；裸 str(e) 的多行
        # 文本亦违反「CSV 单元格不能带换行」约束）
        err_text = _attach_plog_errors(str(e))
        return_struct.eager_precision = err_text
        return_struct.precision_status = "FAIL"
        _mirror_cst_failure_on_dedup(testcase, switches, return_struct, err_text)
        _profiling_end_print(testcase, return_struct, switches=switches)

    return return_struct


def _get_or_create_backend(switches):
    """Get or create backend, cached per subprocess via process context."""
    process_ctx = process_context()
    cached = process_ctx.storage.get("framework_api_backend")
    if cached is not None:
        return cached
    framework = getattr(switches, "framework", "torch")
    backend = get_backend(switches.force_cpu, framework=framework)
    process_ctx.storage["framework_api_backend"] = backend
    return backend


def _ensure_deterministic_level(process_ctx, backend, testcase, deterministic_level):
    """Set the NPU level for this testcase without leaking a previous level."""
    previous_level = process_ctx.storage.get("_deterministic_level")
    if previous_level == deterministic_level:
        return
    if backend.is_npu():
        try:
            backend.set_deterministic_level(deterministic_level)
            logging.info(
                "NPU deterministic level=%s for %s",
                deterministic_level,
                testcase.testcase_name,
            )
        except Exception as e:
            process_ctx.storage.pop("_deterministic_level", None)
            raise RuntimeError(
                f"{testcase.testcase_name}: failed to set NPU deterministic level={deterministic_level}; "
                "execution stopped. Check the sourced CANN environment, torch_npu/driver versions "
                f"and device support. Original error: {e}"
            ) from e
    process_ctx.storage["_deterministic_level"] = deterministic_level


def _collect_sim_report(testcase, switches):
    """Move the camodel ``instr.bin`` (written to the worker cwd) into the case
    sim_output dir, and when ``--sim-report`` generate the trace report.

    The E2E npusim backend runs torch_npu directly in the forkserver worker
    (no cannsim record), so camodel writes ``instr.bin`` to the worker cwd
    (``switches.root_path``). Collect it per case before the next case
    overwrites it; report generation is best-effort and never affects
    precision results.
    """
    if getattr(switches, "backend", None) != "npusim":
        return
    from ..common.simulator.case_writer import case_dir

    src = Path(switches.root_path) / "instr.bin"
    if not src.is_file() or src.stat().st_size == 0:
        logging.warning("[%s] no instr.bin in worker cwd; skip sim report", testcase.testcase_name)
        return
    case_path = case_dir(switches, testcase.testcase_name)
    case_path.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(case_path / "instr.bin"))
    except OSError as e:
        # Without this warning the move failure is silent: instr.bin stays in the
        # worker cwd, gets overwritten by the next case, and the user is left
        # without a sim report and without any hint.
        logging.warning("[%s] failed to move instr.bin into %s: %s", testcase.testcase_name, case_path, e)
        return
    if getattr(switches, "sim_report", False):
        from ..common.simulator.report import maybe_generate_sim_report

        maybe_generate_sim_report(switches, case_path, case_path)


def _check_multi_device_preconditions(testcase, return_struct):
    """多卡前置校验：torch 独占（hccl 生态）+ param plan + 动态槽不支持。

    tf 用例显式 FAIL-fast，而非 worker 内 torch import 崩溃。
    返回 (plan, early)。
    """
    framework = adapter_for_testcase(testcase).framework
    if framework != "torch":
        logging.error(
            f"[{testcase.testcase_name}] multi-device (hccl) supports torch only, "
            f"got framework={framework} ({testcase.api_name})"
        )
        return_struct.eager_precision = "MULTI_DEVICE_UNSUPPORTED_FOR_FRAMEWORK"
        return_struct.precision_status = "FAIL"
        return None, True

    plan = testcase.get_param_plan()
    if plan is None:
        return_struct.eager_precision = "PARAM_PLAN_FAILURE"
        return_struct.precision_status = "FAIL"
        logging.error(f"[{testcase.testcase_name}] Cannot resolve param plan for {testcase.api_name}")
        return None, True

    if tuple(getattr(testcase, "dyn_input_slot_indexes", ()) or ()):
        return_struct.eager_precision = "MULTI_DEVICE_UNSUPPORTED_FOR_DYN_SHAPE"
        return_struct.precision_status = "FAIL"
        logging.error(f"[{testcase.testcase_name}] -1 dynamic slots are not supported in multi-device mode")
        return None, True
    return plan, False


def _generate_multi_device_inputs(testcase, switches, backend, plan, process_ctx, return_struct):
    """父进程侧输入生成：raw_inputs（生成阶段产物/失败快速通道）+ per-rank 声明态模板。

    比对在 rank worker 进程内完成，raw_inputs 不再作为父进程 golden 回退
    基准，仅承载输入生成阶段的失败语义。返回 (rank_template, raw_inputs, early)。
    """
    process_ctx.notify_status("OnGenInput")
    # per-rank 生成走声明态模板深拷贝：generate_inputs 会覆写槽表
    # （np_storages/tensors），共享 testcase 被逐 rank 覆写后终态=末 rank，
    # 与 raw_inputs（golden 回退输入）失配。
    rank_template = copy.deepcopy(testcase)
    try:
        raw_inputs = generate_inputs(testcase, switches, backend, plan)
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Input generation failure")
        return_struct.eager_precision = "INPUT_GEN_FAILURE"
        return_struct.precision_status = "FAIL"
        return None, None, True
    return rank_template, raw_inputs, False


def _stage_rank_inputs(testcase, switches, backend, plan, rank_template, ndev, tmp_dir):
    """逐 rank 生成输入并汇成 inputs.npz（rank 特异种子 + 跨 rank 共享位对齐）。

    共享位语义：
    - MC2 matmul 族：x2（权重，idx 1）按 rank 无关种子重生成——x1 各 rank
      不同（每卡自己的输入分片），权重全 rank 同值（对齐 ACLNN 多卡行为）；
    - MoE combine：expand_x/expert_ids（idx 0/1）全 rank 共享，保证
      ep_send_counts 对 HCCL alltoallv 一致，并从 expert_ids 派生一致的
      ep_send_counts/expand_idx/expert_scales。

    返回 input_path。
    """
    input_path = os.path.join(tmp_dir, "inputs.npz")
    save_arrays = {}
    # Generate per-rank inputs with rank-specific seed (matching ACLNN multi-device behavior)
    base_seed = switches.random_seed or 0

    # Determine if this op needs shared weight (x1/x2 same across ranks)
    api_name_str = testcase.api_name or ""
    needs_shared_weight = not any(kw in api_name_str for kw in _WEIGHT_SHARED_EXCLUDE_OPS) and any(
        kw in api_name_str for kw in _WEIGHT_SHARED_OPS
    )
    # MoE combine: expand_x (idx 0) and expert_ids (idx 1) must be shared across ranks
    # so that ep_send_counts are consistent for HCCL alltoallv.
    moe_combine_shared_all = "npu_moe_distribute_combine" in api_name_str

    # rank 循环不变量：声明态字段在逐 rank 生成期间不变（per-rank 拷贝同值）
    flat_dtypes = resolve_numpy_dtypes(testcase.flat_tensor_dtypes)
    flat_ranges = testcase.flat_input_data_ranges or ()

    for rank_idx in range(ndev):
        per_rank_seed = base_seed + rank_idx * 1000
        np.random.seed(per_rank_seed)
        rank_inputs = generate_inputs(copy.deepcopy(rank_template), switches, backend, plan)
        # For MC2 matmul ops, regenerate x2 (weight) with rank-independent seed so
        # all ranks share the same weight (matching ACLNN multi-device behavior).
        # x1 stays per-rank different (each rank has its own input chunk).
        if moe_combine_shared_all and len(rank_inputs) > 0:
            # Share all inputs across ranks for moe combine (idx 0 and 1)
            for t_idx in range(min(2, len(rank_inputs))):
                if rank_inputs[t_idx] is not None:
                    ss = testcase.flat_storage_shape(t_idx)
                    dtype = flat_dtypes[t_idx] if t_idx < len(flat_dtypes) else None
                    data_range = flat_ranges[t_idx] if t_idx < len(flat_ranges) else (None, None)
                    low = data_range[0] if data_range and data_range[0] is not None else -1.0
                    high = data_range[1] if data_range and data_range[1] is not None else 1.0
                    rng = np.random.RandomState(base_seed + t_idx)
                    if "int" in str(dtype):
                        np_arr = rng.randint(int(low), int(high) + 1, ss).astype(dtype, copy=False)
                    else:
                        np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                    rank_inputs[t_idx] = np_arr.reshape(ss)
        elif needs_shared_weight and len(rank_inputs) > 1 and rank_inputs[1] is not None:
            t_idx = 1
            if t_idx < len(rank_inputs) and rank_inputs[t_idx] is not None:
                ss = testcase.flat_storage_shape(t_idx)
                dtype = flat_dtypes[t_idx] if t_idx < len(flat_dtypes) else None
                data_range = flat_ranges[t_idx] if t_idx < len(flat_ranges) else (None, None)
                low = data_range[0] if data_range and data_range[0] is not None else -1.0
                high = data_range[1] if data_range and data_range[1] is not None else 1.0
                rng = np.random.RandomState(base_seed + t_idx)
                np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                rank_inputs[t_idx] = np_arr.reshape(ss)
        # MoE combine: generate consistent ep_send_counts, expand_idx, expert_scales
        # from expert_ids (which is shared across ranks).
        if moe_combine_shared_all and len(rank_inputs) > 1:
            eid = rank_inputs[1]
            if eid is not None:
                bs2 = eid.shape[0]
                k2 = eid.shape[1] if eid.ndim > 1 else 1
                ep_ws2 = int(testcase.attributes.get("ep_world_size", ndev))
                moe_exp2 = int(testcase.attributes.get("moe_expert_num", ep_ws2))
                local_exp2 = moe_exp2 // ep_ws2 if ep_ws2 > 0 else moe_exp2
                send_counts2 = [0] * ep_ws2
                for ii in range(bs2):
                    for jj in range(k2):
                        e_id = int(eid[ii][jj]) if eid.ndim > 1 else int(eid[ii])
                        dest2 = e_id // local_exp2 if local_exp2 > 0 else 0
                        if dest2 >= ep_ws2:
                            dest2 = ep_ws2 - 1
                        send_counts2[dest2] += 1
                cumsum2 = []
                run2 = 0
                for ii in range(ep_ws2):
                    run2 += send_counts2[ii]
                    cumsum2.append(run2)
                if len(rank_inputs) > 3 and rank_inputs[3] is not None:
                    rank_inputs[3] = np.array(cumsum2, dtype=np.int32)
                if len(rank_inputs) > 2 and rank_inputs[2] is not None:
                    rank_inputs[2] = np.arange(bs2 * k2, dtype=np.int32)
                if len(rank_inputs) > 4 and rank_inputs[4] is not None:
                    rank_inputs[4] = np.ones((bs2, k2), dtype=np.float32)
        for i, r in enumerate(rank_inputs):
            if r is not None:
                # For shared weight (idx >= 1 when needs_shared_weight),
                # only save rank 0's copy; other ranks will fall back to rank 0.
                if needs_shared_weight and i >= 1 and rank_idx != 0:
                    continue
                # fp8/hifloat8 dtypes can't round-trip through npz (e5m2 descr '<f1'
                # fails on load). Save as uint8 view; worker restores via tensor_dtypes.
                arr_to_save = r
                if hasattr(r.dtype, "itemsize") and r.dtype.itemsize == 1 and r.dtype.kind in ("V", "f"):
                    # Could be fp8_e4m3fn (|V1), fp8_e5m2 (<f1), fp8_e8m0, hifloat8
                    dtype_str = str(r.dtype)
                    if any(k in dtype_str for k in ("float8", "hifloat8")):
                        arr_to_save = r.view(np.uint8)
                save_arrays[f"inp_{rank_idx}_{i}"] = arr_to_save
    np.savez(input_path, **save_arrays)
    return input_path


def _run_rank_subprocess(testcase, switches, plan, input_path, tmp_dir, device_ids, return_struct):
    """拉起 rank 子进程组（torch/multi_device.py 脚本直启）并守错。

    父进程以 proc_timeout 看护总时长（覆盖 rank ready / HCCL init / 执行段，
    与 Task 级 proc_timeout 语义对齐），超时杀 launcher 并落 MULTI_DEVICE_TIMEOUT
    ——launcher 被 kill 后其 rank 孙进程由自身 HCCL init timeout 兜底退出。
    失败保留 tmp_dir 现场（rc≠0 路径）；超时/无结果文件路径清理 tmp_dir。
    返回 (result_path, early)。
    """
    import json
    import subprocess

    script_path = os.path.join(os.path.dirname(__file__), "torch", "multi_device.py")
    result_path = os.path.join(tmp_dir, "result.npz")
    error_path = os.path.join(tmp_dir, "error.txt")
    proc_timeout = int(getattr(switches, "proc_timeout", 0) or 3600)

    plan_info = {
        "api_name": testcase.api_name,
        "overload_index": plan.overload_index,
        "output_tensor_indexes": testcase.output_tensor_indexes,
        "attributes": testcase.attributes or {},
        "golden_disabled": str(getattr(testcase, "golden_api", "") or "").lower() == "disable",
        "tensor_dtypes": list(testcase.tensor_dtypes) if testcase.tensor_dtypes else [],
        "remark": testcase.remark or "",
        "tensor_view_shapes": list(testcase.tensor_view_shapes) if testcase.tensor_view_shapes else [],
        "testcase_name": getattr(testcase, "testcase_name", ""),
        "deterministic_level": resolve_deterministic_level(switches, testcase),
        "proc_timeout": proc_timeout,
    }

    plan_path = os.path.join(tmp_dir, "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan_info, f)

    env = os.environ.copy()
    env["TTK_E2E_INPUT"] = input_path
    env["TTK_E2E_PLAN"] = plan_path
    env["TTK_E2E_RESULT"] = result_path
    env["TTK_E2E_ERROR"] = error_path
    env["TTK_E2E_DEVICES"] = ",".join(str(d) for d in device_ids)
    env["TTK_E2E_NDEV"] = str(len(device_ids))
    # Pass graph mode to worker: "dynamic"/"static"/"" (multi-device HCCL graph path,
    # mirrors single-device execute_graph driven by -d/-c switches).
    graph_mode = ""
    if switches.dyn_switches.enabled:
        graph_mode = "dynamic"
    elif switches.cst_switches.enabled:
        graph_mode = "static"
    env["TTK_E2E_GRAPH_MODE"] = graph_mode
    # Pass plugin path to worker so it can load per-operator golden specs.
    if switches.plugin_path:
        env["TTK_E2E_PLUGIN"] = ",".join(str(p_) for p_ in switches.plugin_path)

    cmd = [sys.executable, script_path]
    logging.info(f"E2E multi-device: running {cmd} with devices={device_ids} graph_mode={graph_mode or 'none'}")

    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, errors="replace", check=False, timeout=proc_timeout
        )
        if proc.returncode != 0:
            err_msg = proc.stderr[-2000:] if proc.stderr else "unknown"
            if os.path.exists(error_path):
                with open(error_path) as f:
                    err_msg = f.read()[:3000]
            # Fallback: 某些 torch/torch_npu 版本 worker 进程退出阶段
            # 有 "double free or corruption" 析构崩溃（rc=-6），但结果
            # npz 已正常写出。优先采用实际结果，避免误判为 FAIL。
            fallback_used = False
            if os.path.exists(result_path):
                try:
                    _loaded = np.load(result_path, allow_pickle=True)
                    if "precision_0" in _loaded.files and "pass_0" in _loaded.files:
                        logging.warning(
                            f"E2E worker rc={proc.returncode} but result.npz written; "
                            f"using fallback result (worker exit crash ignored)."
                        )
                        fallback_used = True
                except Exception:  # noqa: S110
                    pass
            if not fallback_used:
                logging.error(f"E2E multi-device worker failed (rc={proc.returncode}): {err_msg}")
                logging.error(f"E2E stderr: {proc.stderr[-2000:] if proc.stderr else 'none'}")
                logging.error(f"E2E tmp_dir preserved: {tmp_dir}")
                return_struct.eager_precision = f"MULTI_DEVICE_FAILED: {err_msg}"
                return_struct.precision_status = "FAIL"
                return None, True
    except subprocess.TimeoutExpired:
        import shutil

        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)
        return_struct.eager_precision = "MULTI_DEVICE_TIMEOUT"
        return_struct.precision_status = "FAIL"
        return None, True
    if not os.path.exists(result_path):
        import shutil

        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)
        return_struct.eager_precision = "NO_OUTPUT"
        return_struct.precision_status = "FAIL"
        return None, True
    return result_path, False


def _read_worker_results(testcase, switches, tmp_dir, result_path, return_struct):
    """读回 rank worker 结果并收尾（比对在 rank worker 进程内完成）。

    rank worker 恒定写 precision_0/pass_0（torch.isclose，bf16 兼容），
    可选附带 graph_precision_0/pass_0/mode_0（HCCL 图模式 dynamic/static）。
    tmp_dir 读后清理。
    """
    import shutil

    loaded = np.load(result_path, allow_pickle=True)

    prec = str(loaded["precision_0"]) if "precision_0" in loaded.files else None
    passed = str(loaded["pass_0"]) if "pass_0" in loaded.files else None
    with contextlib.suppress(Exception):
        shutil.rmtree(tmp_dir)
    if prec is None or passed is None:
        # 防御分支：writer 各路径恒写 precision_0/pass_0，缺失即 worker 异常
        logging.error(f"[{testcase.testcase_name}] E2E multi-device result missing precision fields: {loaded.files}")
        return_struct.eager_precision = "NO_OUTPUT"
        return_struct.precision_status = "FAIL"
        _profiling_end_print(testcase, return_struct, switches=switches)
        return
    return_struct.construct(prec, passed, None)
    # Multi-device HCCL graph result (dynamic/static) produced by worker
    if "graph_precision_0" in loaded.files:
        gprec = str(loaded["graph_precision_0"])
        gpass = str(loaded["graph_pass_0"])
        gmode = str(loaded["graph_mode_0"]) if "graph_mode_0" in loaded.files else None
        if gmode == "static":
            return_struct.construct(gprec, gpass, None, mode="static")
        else:
            return_struct.construct(gprec, gpass, None, mode="dynamic")
    _profiling_end_print(testcase, return_struct, [], switches)


def _run_profile_multi_device(
    testcase, backend, device_grant_events, device_granted_indices, device_ids, switches, return_struct
):
    """多卡（HCCL 通信类算子）编排：per-rank 输入 → rank 子进程组执行 → 结果读回。

    rank 侧经 torch/multi_device.py 脚本直启（subprocess，非 pickled target）；
    比对在 rank worker 进程内完成（torch.isclose，bf16 兼容），父进程直采
    结果行。步骤函数返回 early=True 表示用例已收尾。
    """
    plan, early = _check_multi_device_preconditions(testcase, return_struct)
    if early:
        return

    process_ctx = process_context()
    rank_template, _raw_inputs, early = _generate_multi_device_inputs(
        testcase, switches, backend, plan, process_ctx, return_struct
    )
    if early:
        return

    process_ctx.notify_status("OnAcquireLock")
    use_device = backend.has_device()
    with MultiDeviceLock(
        process_ctx,
        device_ids,
        use_device=use_device,
        grant_events=device_grant_events,
        granted_indices=device_granted_indices,
    ):
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, device_ids[0], switches)

        process_ctx.notify_status("OnProfiling")

        tmp_dir = tempfile.mkdtemp(prefix="ttk_e2e_md_")
        input_path = _stage_rank_inputs(testcase, switches, backend, plan, rank_template, len(device_ids), tmp_dir)
        result_path, early = _run_rank_subprocess(
            testcase, switches, plan, input_path, tmp_dir, device_ids, return_struct
        )
        if early:
            return
        _read_worker_results(testcase, switches, tmp_dir, result_path, return_struct)


def _needs_golden_promote(testcase, switches):
    """判据是否要求 golden 升精度 —— cross_check（三方）与 mixed/mix_tolerance（单标杆）则是。

    cross_check 的竞品与 E2E golden 同为 torch aten：golden 若停留在被测 dtype，
    两者实现同源、竞品误差恒为 0，safe_div 的 err 地板接管，判据退化成「与
    torch 逐位一致」——一次末位舍入即被放大成数倍比值。mix_tolerance 为单标杆
    比对，标杆须为更高精度的实现。

    判定锚点 = CSV 声明输出 dtype（resolve_numpy_dtypes(flat_output_dtypes)）：
    golden 生成先于设备执行，此时设备输出尚不存在，锚点不得依赖执行结果；
    声明与实际 dtype 不一致的低频场景由输出校验阶段兜底。

    golden_mode=XPU 与 Promote 互斥（golden 由三方产出，不做 CPU 抬精度）；
    golden_mode=NPU 同样互斥（golden 在 NPU 原生 dtype 上执行，无 CPU 抬精度语义）。
    """
    if switches.golden_mode in ("XPU", "NPU"):
        return False
    output_dtypes = resolve_numpy_dtypes(getattr(testcase, "flat_output_dtypes", ()) or ())
    if not output_dtypes:
        return False
    try:
        standards = resolve_tolerance(
            SpecResolver(testcase.api_name, "e2e", switches.plugin_path).resolve_tolerance(),
            testcase.flat_precision_tolerances,
            testcase.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
        )
        return needs_promote_golden(standards)
    except (KeyError, TypeError, ValueError):
        # 只兜 tolerance spec 配置类错误(字段缺失/类型不对/取值非法)。
        # 用 warning 而非 debug:此处静默跳过 Promote 正是本函数要修的症状
        # (cross_check/mix_tolerance 误判 FAIL 却无任何提示),必须让用户看得见。
        logging.warning(
            "[%s] tolerance spec resolve failed, golden Promote skipped; "
            "cross_check/mix_tolerance may misjudge without high-precision golden",
            testcase.testcase_name,
            exc_info=True,
        )
        return False


def _generate_golden(testcase, raw_inputs, switches, backend, promote, dev_id=0, return_struct=None):
    """生成 golden；promote=True 时临时挂 golden_mode_override=Promote，生成后回写原值。

    TestcaseE2e 的 golden_mode_override 为 __slots__ 槽位且 __init__ 预置
    None，不可 del，用记录回写还原。调试 dump 落盘统一由 DataDumper 门控
    （dump_golden 开关 + no-prof 单点豁免），本层不做分支。dev_id/return_struct
    供 golden-mode NPU 的执行设备与性能列回写（CPU/XPU 路径忽略）。
    """
    if not promote:
        return generate_golden_data(testcase, raw_inputs, switches, backend, dev_id=dev_id, return_struct=return_struct)
    prev = getattr(testcase, "golden_mode_override", None)
    testcase.golden_mode_override = "Promote"
    try:
        return generate_golden_data(testcase, raw_inputs, switches, backend, dev_id=dev_id, return_struct=return_struct)
    finally:
        testcase.golden_mode_override = prev


def _check_preconditions(testcase, backend, switches, return_struct):
    """前置校验：param plan 解析 + -1 动态槽双门（NPU 前处理依赖 / dump 方向不支持）。

    返回 (plan, dyn_indexes, early)；early=True 表示用例已收尾。
    """
    plan = testcase.get_param_plan()
    if plan is None:
        return_struct.eager_precision = "PARAM_PLAN_FAILURE"
        return_struct.precision_status = "FAIL"
        logging.error(f"[{testcase.testcase_name}] Cannot resolve param plan for {testcase.api_name}")
        return None, (), True

    dyn_indexes = tuple(getattr(testcase, "dyn_input_slot_indexes", ()) or ())
    # Prepare/dump（--no-prof）only generates host inputs and records None for
    # dynamic slots.  It must work on CPU; the hook is invoked later during NPU
    # replay.  Direct and replay execution still require an NPU backend.
    if (
        dyn_indexes
        and not switches.no_prof
        and (not backend.is_npu() or resolve_npu_preprocess_hook(testcase, switches) is None)
    ):
        logging.error(
            f"[{testcase.testcase_name}] tensor_view_shapes declares -1 slot(s) {dyn_indexes}, "
            "but no npu_preprocess hook is registered"
        )
        return_struct.eager_precision = "DYN_SHAPE_REQUIRES_NPU_PREPROCESS"
        return_struct.precision_status = "FAIL"
        return None, dyn_indexes, True

    # -1 动态槽以 None 空标记落盘（input_N_none.bin），物化钩子留待 NPU
    # 重放执行；full-flow 首跑落盘在执行物化后写实际张量。
    return plan, dyn_indexes, False


def _resolve_manual_stores(testcase, switches, process_ctx, return_struct):
    """manual 数据双方向开仓：--no-prof 写方向 store / load 方向用例解析。

    load 方向目录全空的首跑回落为生成+落盘（后续运行即可加载）。
    返回 (manual_case, dump_store, early)。
    """
    manual_case = None
    dump_store = None
    try:
        if switches.no_prof:
            # dump 方向：--no-prof 落盘（bridge 已保证写方向激活时设置 manual_data_dirs）
            dump_store = manual_data_dump_store(testcase, switches)
    except Exception as exc:
        logging.exception(f"[{testcase.testcase_name}] Manual data dump store failure")
        return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
        _profiling_end_print(testcase, return_struct, switches=switches)
        return None, None, True
    if not switches.no_prof and switches.manual_data_dirs:
        try:
            manual_case = resolve_manual_case(
                testcase,
                "e2e",
                switches,
                notify_before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
                require_goldens=False,
            )
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data loading failure")
            return_struct.eager_precision = f"MANUAL_DATA_READ_FAILURE: {exc}"
            return_struct.precision_status = "FAIL"
            return None, dump_store, True
        if manual_case is None and switches.dump_config.dump_input:
            # load 方向目录全空的首跑：正常生成并落盘（后续运行即可加载）
            try:
                dump_store = manual_data_dump_store(testcase, switches)
            except Exception as exc:
                logging.exception(f"[{testcase.testcase_name}] Manual data dump store failure")
                return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
                _profiling_end_print(testcase, return_struct, switches=switches)
                return None, None, True
    return manual_case, dump_store, False


def _require_dynamic_golden_source(testcase, switches, manual_case, dyn_indexes, return_struct):
    """动态槽用例的 golden 来源门：CPU 路径 -1 槽保持 None，须有插件/落盘/禁用兜底。

    input-only prepare（--no-prof 且不落 golden）无 golden 消费方，豁免本门；
    full prepare 与 replay 保持校验。
    """
    needs_dynamic_golden = not switches.no_prof or switches.dump_config.dump_golden
    if dyn_indexes and needs_dynamic_golden and not has_dynamic_golden_source(testcase, switches, manual_case):
        logging.error(
            f"[{testcase.testcase_name}] dynamic slot(s) {dyn_indexes} remain None on the CPU path; "
            "provide an E2E Golden plugin, replay saved Goldens, or disable Golden comparison"
        )
        return_struct.eager_precision = "DYN_SHAPE_REQUIRES_GOLDEN_SOURCE"
        return_struct.precision_status = "FAIL"
        return True
    return False


def _generate_input(testcase, backend, switches, process_ctx, plan, manual_case, dump_store, dev_id, return_struct):
    """输入生成（含落盘快照与 dump）。返回 (adapter, raw_inputs, prepared_inputs, early)。"""
    # adapter 须先于输入生成创建：TF 的 L1 张量构造工厂随 tf/ 子包导入注册
    # （framework_hooks），generate_inputs → backend.inputs_from_numpy 依赖它；
    # 此前依赖 get_param_plan → keeper → adapter 的隐式时序，而 testcase 经
    # pickle 携带父进程已解析的 _api_info_cache 时该链路被短路，工厂未注册
    # 即静默退 backend 基础转换（丢失量化 dtype / const 位 / mutable 位处理）。
    # torch 侧为幂等单例，无副作用。
    adapter = adapter_for_testcase(testcase)

    process_ctx.notify_status("OnGenInput")
    # 输入生成不触设备：TF 侧框架张量在 CPU 作用域物化（golden P2 插件契约
    # 即 CPU nested tensors），设备张量由执行侧在 open 后从 raw_inputs
    # 重建（见 _device_profiling）。torch 侧张量本就 CPU-born，无此分野。
    try:
        if manual_case is not None:
            raw_inputs = generate_inputs(
                testcase, switches, backend, plan, stored_inputs=manual_case.inputs, cpu_placement=True
            )
        else:
            raw_inputs = generate_inputs(testcase, switches, backend, plan, cpu_placement=True)
    except Exception as exc:
        logging.exception(f"[{testcase.testcase_name}] Input generation failure")
        prefix = "MANUAL_DATA_READ_FAILURE" if manual_case is not None else "INPUT_GEN_FAILURE"
        return_struct.eager_precision = f"{prefix}: {exc}" if manual_case is not None else prefix
        return_struct.precision_status = "FAIL"
        return None, None, None, True

    prepared_inputs = None
    if dump_store is not None and manual_case is None:
        # 生成态数据将写入 store：先快照（设备执行可能原地改动 backing storage）
        try:
            prepared_inputs = snapshot_manual_values(
                testcase.np_storages if testcase.np_storages is not None else raw_inputs,
                "input",
            )
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data snapshot failure")
            return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return None, None, None, True

    if dump_store is None:
        process_ctx.notify_status("OnDumpInput")
        dump_inputs(testcase, raw_inputs, switches)
    return adapter, raw_inputs, prepared_inputs, False


def _dump_only_end(testcase, return_struct, golden_nps, switches):
    """dump-only（--no-prof）收尾行（MANUAL_DATA_PREPARED）。

    纯收尾：数据落盘在 _manual_data_dump（对齐 kernel/aclnn 分层）。
    """
    return_struct.construct("MANUAL_DATA_PREPARED", "PASS", None)
    _profiling_end_print(testcase, return_struct, golden_nps, switches)


def _resolve_api(testcase, adapter, backend, switches):
    """主 API 解析与图模式开关裁剪（backend 不支持时降级关）。"""
    resolved, is_tensor_method = adapter.load_api(testcase.api_name)
    is_inplace = adapter.is_inplace(resolved, is_tensor_method)
    graph_enabled = (
        switches.cst_switches.enabled or switches.dyn_switches.enabled or getattr(switches, "aclgraph_enabled", False)
    )
    if graph_enabled and not backend.supports_graph_mode():
        if not _should_run_eager(switches):
            # --eager=false 且图模式被后端降级（如 CPU 不支持）：无可执行路径。
            # 提前显式失败（run 的 except 收敛为 FAIL 行），不白算 golden、
            # 不留 status 为空的静默结果行。
            raise ValueError(
                f"--eager=false but backend {backend.device_type()} does not support graph mode; "
                "no executable leg remains"
            )
        logging.warning(f"Graph mode not supported by backend {backend.device_type()}, skipping graph execution")
        graph_enabled = False
    return SimpleNamespace(
        resolved=resolved, is_tensor_method=is_tensor_method, is_inplace=is_inplace, graph_enabled=graph_enabled
    )


def _prepare_golden(
    testcase,
    backend,
    switches,
    process_ctx,
    manual_case,
    raw_inputs,
    dev_id=0,
    return_struct=None,
    device_grant_events=None,
    device_granted_indices=None,
):
    """golden 生成 / replay 装载——先于 no-prof 收尾与设备执行。

    统一编排（四通路一致）：有已存真值就装载（replay 优先，任何 mode）；
    无已存真值按模式裁决——XPU 占位（三方设备后回填）、NPU 设备生成
    （自起设备锁、DUT 前）、Enable 现算。静态装载失败 → FAIL 行。

    返回 SimpleNamespace(promote, nps, early)：early=True 表示用例已收尾
    （装载失败 FAIL 行），_run_profile 应直接返回。
    """
    promote = _needs_golden_promote(testcase, switches)
    if switches.no_prof and not switches.dump_config.dump_golden:
        return SimpleNamespace(promote=promote, nps=None, early=False)
    if manual_case is not None and manual_case.has_goldens:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            nps = load_replay_goldens(
                manual_case,
                shapes=testcase.flat_output_view_shapes,
                dtypes=testcase.flat_output_dtypes,
            )
            dump_goldens(testcase, nps, switches)
            return SimpleNamespace(promote=promote, nps=nps, early=False)
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Static replay golden load failed")
            return_struct.construct(f"MANUAL_DATA_READ_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return SimpleNamespace(promote=promote, nps=None, early=True)
    mode = golden_mode(testcase, switches)
    if mode == "XPU":
        # XPU 真值由三方在设备后回填（_collect_xpu），本侧占位即可——无需延迟
        return SimpleNamespace(promote=promote, nps=["GOLDEN_FAILURE"], early=False)
    if mode == "NPU":
        # 自起设备锁、DUT 前生成（与 kernel gen_output(dev_id)/aclnn gen(dev_id)
        # 同款单步编排；锁段与 DUT 串行错开）
        with DeviceLock(
            process_ctx,
            dev_id,
            use_device=backend.has_device(),
            grant_event=(device_grant_events or {}).get(dev_id),
            granted_idx=(device_granted_indices or {}).get(dev_id),
        ):
            nps = _generate_golden(
                testcase,
                raw_inputs,
                switches,
                backend,
                promote,
                dev_id=dev_id,
                return_struct=return_struct,
            )
        return SimpleNamespace(promote=promote, nps=nps, early=False)
    nps = _generate_golden(
        testcase,
        raw_inputs,
        switches,
        backend,
        promote,
        dev_id=dev_id,
        return_struct=return_struct,
    )
    return SimpleNamespace(promote=promote, nps=nps, early=False)


def _attach_plog_errors(err_msg):
    """异常消息落 CSV 列前拼接 plog 报错并折叠为单行。

    多行异常按行折成 " | "（CSV 单元格不能带换行）。plog 按本进程 pid
    取文件（worker 与 CANN 同进程），提取前 sleep 0.5s 等落盘，只取
    [ERROR] 行。plog 无 [ERROR] 行或读取失败时，返回折叠后的原消息。
    """
    if not err_msg:
        return err_msg
    one_line_err = " | ".join(line.strip() for line in err_msg.splitlines() if line.strip())
    try:
        import time

        from ttk.utilities.plog import extract_plog_errors

        time.sleep(0.5)
        error_lines = [line.strip() for line in extract_plog_errors(max_lines=10) if "[ERROR]" in line and line.strip()]
    except Exception:
        return one_line_err
    if not error_lines:
        return one_line_err
    return f"{one_line_err} | plog: {' | '.join(error_lines)}"


def _run_graph_phase(mode_label, graph_fn, *args, **kwargs):
    """三种图模式（cst/dyn/aclgraph）的统一执行入口。

    返回 4 元组 (nps, perf, det_status, err_msg)。执行器抛出的异常在这里
    转成该模式的失败结果，不再向上抛——err_msg 携带异常与 plog（见
    _attach_plog_errors），最终进结果 CSV 的 graph_<mode>_precision 列，
    其余图模式照常执行。
    """
    try:
        nps, perf, det_status, err = graph_fn(*args, **kwargs)
        return nps, perf, det_status, _attach_plog_errors(err)
    except Exception as exc:
        logging.error(f"Graph {mode_label} execution failed: {exc}", exc_info=True)
        return [], None, None, _attach_plog_errors(f"Graph {mode_label} execution failed: {exc}")


def _device_profiling(
    testcase,
    backend,
    switches,
    process_ctx,
    dev_id,
    device_grant_events,
    device_granted_indices,
    plan,
    adapter,
    api,
    raw_inputs,
    deterministic_level,
    return_struct,
):
    """设备执行（锁内 eager / graph×3 模式）与确定性状态汇总。"""
    # 进程设备在此就绪（首用例触发 npu_device.open/as_default，进程级一次）；
    # 输入生成与 golden 已在 CPU 作用域完成。TF 侧设备张量在 open 后从
    # raw_inputs 重建为 NPU-born（替换输入段的 CPU 版；torch 侧 CPU 张量经
    # to_device 逐模式上板，无需重建）。torch 的 set_device 幂等。
    backend.set_device(dev_id)
    if adapter.framework == "tf":
        set_runtime_tensors(testcase, raw_inputs, backend)
    process_ctx.notify_status("OnAcquireLock")
    use_device = backend.has_device()
    result_nps = None
    perf = None
    eager_det_status = None
    graph_cst_nps = None
    graph_dyn_nps = None
    graph_cst_perf = None
    graph_dyn_perf = None
    graph_aclgraph_nps = None
    graph_aclgraph_perf = None
    graph_cst_det = None
    graph_dyn_det = None
    graph_aclgraph_det = None
    graph_aclgraph_err = None
    graph_cst_err = None
    graph_dyn_err = None
    with DeviceLock(
        process_ctx,
        dev_id,
        use_device=use_device,
        grant_event=device_grant_events.get(dev_id),
        granted_idx=device_granted_indices.get(dev_id),
    ):
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, dev_id, switches)

        # 模式等价去重（adapter 钩子驱动、无框架分支）：onnx 的 eager ≡ 静态图
        # （om 恒为静态图）——双开（--eager + -c）时单次执行，(nps, perf, det) 镜像刷
        # EAGER 与 GRAPH CST 两列；镜像必须在三元组齐备的编排点做（perf/det 是
        # _execute_eager 侧资产，adapter 内 memo 拿不到）。dyn/aclgraph 不去重——
        # 由 adapter.run_graph 快速失败（空 nps 会落 GRAPH_EXEC_FAILURE，不能静默跳空列）。
        # getattr 防御：eager_graph_alias 是可选协议（基类默认 None），
        # duck-typed 测试替身可不携带。
        eager_alias_fn = getattr(adapter, "eager_graph_alias", None)
        dedup_cst = (
            callable(eager_alias_fn)
            and eager_alias_fn() == "cst"
            and _should_run_eager(switches)
            and switches.cst_switches.enabled
        )
        if _should_run_eager(switches):
            process_ctx.notify_status("OnEagerProfiling")
            result_nps, perf, eager_det_status = _capture_peak_memory(
                backend,
                dev_id,
                return_struct,
                _execute_eager,
                testcase,
                backend,
                dev_id,
                switches,
                plan,
                api.resolved,
                api.is_tensor_method,
                api.is_inplace,
                raw_inputs,
                deterministic_level,
            )
            if dedup_cst:
                # 同一份执行结果镜像两列：比对入口对两列独立 evaluate，nps/perf/det
                # 同源 → 两列恒同值
                graph_cst_nps, graph_cst_perf, graph_cst_det = result_nps, perf, eager_det_status
        if api.graph_enabled:
            graph_fn = adapter.run_graph
            if getattr(switches, "aclgraph_enabled", False):
                process_ctx.notify_status("OnGraphAclgraph")
                graph_aclgraph_nps, graph_aclgraph_perf, graph_aclgraph_det, graph_aclgraph_err = _run_graph_phase(
                    "aclgraph",
                    graph_fn,
                    testcase,
                    backend,
                    dev_id,
                    switches,
                    plan,
                    api.resolved,
                    api.is_tensor_method,
                    api.is_inplace,
                    raw_inputs,
                    dynamic=False,
                    is_aclgraph=True,
                    deterministic_level=deterministic_level,
                )
            if switches.cst_switches.enabled and not dedup_cst:
                # 去重时跳过二次执行（eager 通道已镜像）；notify 亦跳过——本用例
                # 无独立 cst 执行阶段
                process_ctx.notify_status("OnGraphCst")
                graph_cst_nps, graph_cst_perf, graph_cst_det, graph_cst_err = _run_graph_phase(
                    "static",
                    graph_fn,
                    testcase,
                    backend,
                    dev_id,
                    switches,
                    plan,
                    api.resolved,
                    api.is_tensor_method,
                    api.is_inplace,
                    raw_inputs,
                    dynamic=False,
                    deterministic_level=deterministic_level,
                )
            if switches.dyn_switches.enabled:
                process_ctx.notify_status("OnGraphDyn")
                graph_dyn_nps, graph_dyn_perf, graph_dyn_det, graph_dyn_err = _run_graph_phase(
                    "dynamic",
                    graph_fn,
                    testcase,
                    backend,
                    dev_id,
                    switches,
                    plan,
                    api.resolved,
                    api.is_tensor_method,
                    api.is_inplace,
                    raw_inputs,
                    dynamic=True,
                    deterministic_level=deterministic_level,
                )
    gc.collect()
    _collect_sim_report(testcase, switches)

    det_statuses = [eager_det_status, graph_cst_det, graph_dyn_det, graph_aclgraph_det]
    det_statuses = [s for s in det_statuses if s is not None]
    if det_statuses:
        return_struct.deterministic_status = "FAIL" if "FAIL" in det_statuses else "PASS"

    return SimpleNamespace(
        result_nps=result_nps,
        perf=perf,
        graph_cst_nps=graph_cst_nps,
        graph_cst_perf=graph_cst_perf,
        graph_dyn_nps=graph_dyn_nps,
        graph_dyn_perf=graph_dyn_perf,
        graph_aclgraph_nps=graph_aclgraph_nps,
        graph_aclgraph_perf=graph_aclgraph_perf,
        graph_aclgraph_err=graph_aclgraph_err,
        graph_cst_err=graph_cst_err,
        graph_dyn_err=graph_dyn_err,
    )


def _manual_data_dump(
    testcase, switches, process_ctx, dump_store, manual_case, prepared_inputs, golden_nps, return_struct
):
    """in+golden 落盘（manual-data store）；no-prof（dump 方向）在此收尾。

    full-flow：golden_nps 就绪（非 None）即落——golden 已在 _prepare_golden
    内一次解析完成（回放装载 / XPU 占位 / NPU 自锁段生成 / Enable 现算），
    不存在 deferred 二次落盘。落当次实际产物（判据要求 Promote 时即升精度
    标杆），文件名 dtype 自描述、回放按其装载；golden 缺失（dump_golden
    关闭 / 占位串）跳过 golden 位。返回 early。
    """
    if not switches.no_prof and golden_nps is None:
        return False
    if dump_store is not None and manual_case is None and (switches.golden_mode != "XPU" or switches.no_prof):
        try:
            process_ctx.notify_status("OnWriteManualData")
            case_dir = dump_manual_dataset(dump_store, testcase, "e2e", prepared_inputs, golden_nps or [], switches)
            logging.info(f"[{testcase.testcase_name}] manual data dumped: {case_dir}")
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data dump failure")
            return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return True
    elif not switches.no_prof and dump_store is not None and manual_case is None and switches.golden_mode == "XPU":
        # 对齐 kernel/aclnn/geir full-flow + XPU：三方 golden 无可落盘，warning 跳过。
        logging.warning(
            "[%s] golden-mode XPU: skip manual-data output (golden produced by third party)",
            testcase.testcase_name,
        )
    if switches.no_prof:
        _dump_only_end(testcase, return_struct, golden_nps or [], switches)
        return True
    return False


def _dispatch_xpu(testcase, switches, process_ctx, manual_case, raw_inputs):
    """e2e 三方 dispatch 前置（NPU 执行期重叠）。

    判据预解析对齐 _needs_promote 先例（flat_output_dtypes 声明 dtype——e2e
    是唯一在执行后解析判据的通路，重叠 dispatch 需要执行前知道 need_3party）。
    预解析失败/无声明 dtype/非 cross_check/golden-XPU/manual 命中/降级 → None
    （_collect_xpu 走现状同步路径）；命中 → background 句柄。
    """
    from ttk.core_modules.common.xpu import (
        XpuDispatch,
        collect_third_party,
        has_manual_xpu_source,
        overlap_deadline_s,
        snapshot_xpu_inputs,
        xpu_join_timeout,
        xpu_overlap_enabled,
    )

    golden_is_xpu = switches.golden_mode == "XPU"
    if golden_is_xpu:
        if has_manual_xpu_source(testcase, manual_case):
            return None  # manual 源——_collect_xpu 同步短路（本地文件必然在，无需设备前收割）
        # golden-XPU：三方即 golden，设备前同步收割（对齐 kernel/aclnn/geir——
        # FAIL 不空跑设备），经 ready 句柄回传、collect 幂等。
        process_ctx.notify_status("OnXpuProfiling")

        def _collect_golden():
            priority, third_parties, xpu_results = collect_third_party(
                op_name=testcase.api_name,
                inputs=e2e_xpu_inputs(testcase, raw_inputs),
                input_names=e2e_xpu_input_names(testcase),
                op_type=None,
                attributes=testcase.attributes or {},
                testcase_name=testcase.testcase_name,
                switches=switches,
                need_data=True,
                input_dtypes=e2e_xpu_input_dtypes(testcase),
            )
            return third_parties, xpu_results, (priority or "xpu")

        return XpuDispatch.ready(_collect_golden())
    if has_manual_xpu_source(testcase, manual_case):
        return None  # manual 源——_collect_xpu 同步短路（声明探测，不读文件）
    if not xpu_overlap_enabled():
        return None
    try:
        output_dtypes = tuple(str(d) for d in (getattr(testcase, "flat_output_dtypes", ()) or ()))
        if not output_dtypes:
            return None
        standards = resolve_tolerance(
            SpecResolver(testcase.api_name, "e2e", switches.plugin_path).resolve_tolerance(),
            testcase.flat_precision_tolerances,
            testcase.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
        )
        if not any(s.token == "cross_check" for s in standards):  # noqa: S105  # token 为精度标准名，非凭据
            return None
    except Exception:  # noqa: BLE001  预解析失败（dtype 档缺失/配置错）回退同步
        return None
    inputs = snapshot_xpu_inputs(e2e_xpu_inputs(testcase, raw_inputs))
    budget = overlap_deadline_s(switches)
    process_ctx.notify_status("OnXpuDispatch")

    def _collect():
        priority, third_parties, xpu_results = collect_third_party(
            op_name=testcase.api_name,
            inputs=inputs,
            input_names=e2e_xpu_input_names(testcase),
            op_type=None,
            attributes=testcase.attributes or {},
            testcase_name=testcase.testcase_name,
            switches=switches,
            need_data=True,
            input_dtypes=e2e_xpu_input_dtypes(testcase),
            deadline_s=budget,
        )
        return third_parties, xpu_results, (priority or "xpu")

    return XpuDispatch.background(_collect, join_timeout=xpu_join_timeout(budget))


def _discard_unconsumed(xpu_dispatch):
    """句柄未被消费时弃置——外门未开（golden 全占位串，比对必败）的后台请求止损。"""
    if xpu_dispatch is not None and xpu_dispatch.pending:
        xpu_dispatch.discard()


def _xpu_promote_golden(testcase, xpu, switches, process_ctx, return_struct):
    """golden-XPU 装配（设备前收割，对齐 kernel/aclnn/geir）：三方即 golden，
    缺失整例 FAIL 提前返回。返回 True = 用例已收尾（结果已写 return_struct）。"""
    from ..common.golden import xpu_golden_perf

    third_parties, xpu_results, xpu_provider = xpu.collect()
    from ..common.xpu import dump_xpu_slot_files

    dump_xpu_slot_files(testcase, switches, third_parties, xpu_provider)
    if third_parties is None:
        logging.error(
            "[%s] golden-mode XPU: no third_party output (XPU down / no third_party declaration); case FAIL",
            testcase.testcase_name,
        )
        return_struct.construct("GOLDEN_FAILURE", "FAIL", None)
        _profiling_end_print(testcase, return_struct, switches=switches)
        return True
    return_struct.set_golden_perf(xpu_golden_perf(xpu_results), source="xpu")
    dump_goldens(testcase, list(third_parties), switches)
    return False


def _cross_check_declared(testcase, switches, ref_nps, golden_nps, golden_is_xpu) -> bool:
    """cross_check 判据解析。dtype 源 = 设备输出 → 真 golden 兜底，golden-XPU
    恒解析（dtype 缺省走空档）；占位 golden（SUPPRESSED/UNSUPPORTED/
    GOLDEN_FAILURE）不阻断判定。"""
    golden_real = bool(golden_nps) and not any(isinstance(g, str) for g in golden_nps)
    if not (golden_is_xpu or golden_real or ref_nps):
        return False
    tolerance = SpecResolver(testcase.api_name, "e2e", switches.plugin_path).resolve_tolerance()
    # 按被测输出的 dtype 解析容差(golden 可能已被 Promote 抬精度,拿它解析会走偏);
    # 拿不到 device 结果时退回 golden。
    dtype_src = ref_nps if ref_nps else (golden_nps or ())
    output_dtypes = tuple(str(g.dtype) if g is not None and hasattr(g, "dtype") else None for g in dtype_src)
    standards = resolve_tolerance(
        tolerance,
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    return any(s.token == "cross_check" for s in standards)  # noqa: S105  # token 为精度标准名，非凭据


def _collect_xpu(
    testcase,
    switches,
    process_ctx,
    manual_case,
    raw_inputs,
    ref_nps,
    golden_nps,
    perf,
    return_struct,
    xpu_dispatch=None,
):
    """三方采集与 XPU golden 回填。返回 (third_parties, xpu_results, golden_nps, early)。

    判据解析（cross_check 判定）依赖输出 dtype：设备输出优先、真 golden 兜底，
    占位 golden 不阻断。采集段由 cross_check / golden-XPU / --xpu-perf /
    --dump xpu 任一触发；golden 为占位串（SUPPRESSED/UNSUPPORTED/
    GOLDEN_FAILURE）时比对按占位语义走，采集照常。

    xpu_dispatch：_dispatch_xpu 预派发的重叠句柄——存在则 collect（NPU
    执行已在进行/完成），跳过本函数的 dispatch 段（判据/golden 升格逻辑不变）。
    """
    third_parties = None
    xpu_results = None
    golden_is_xpu = switches.golden_mode == "XPU"
    need_3party = _cross_check_declared(testcase, switches, ref_nps, golden_nps, golden_is_xpu)
    # 采集段门槛：判据需要 DATA（cross_check/golden-XPU）或显式采集开关
    # （--xpu-perf/--dump xpu）——与 golden 状态解耦
    from ttk.core_modules.common.xpu import (
        MANUAL_XPU_PROVIDER,
        collect_third_party,
        dump_manual_xpu_archive,
        dump_xpu_slot_files,
        load_manual_third_parties,
        manual_xpu_results,
        xpu_mode,
    )

    collection_mode = xpu_mode(switches, need_3party or golden_is_xpu)
    if golden_is_xpu or collection_mode:
        if xpu_dispatch is not None:
            # 预派发重叠句柄：NPU 执行期已在采集，collect 收割（本函数 dispatch 段跳过）
            if xpu_dispatch.pending:
                process_ctx.notify_status("OnXpuCollect")
            third_parties, xpu_results, xpu_provider = xpu_dispatch.collect()
            dump_xpu_slot_files(testcase, switches, third_parties, xpu_provider)
        # manual 三方输出（CSV 显式路径 / manual-data case 目录槽位）：DATA 需求
        # 时本地装载短路，跳过远端 dispatch；PERF-only 不走 manual（性能指标
        # 只能来自在线执行）。device_outputs 供 e2e 无声明输出槽时取 shape/dtype。
        elif need_3party or golden_is_xpu or getattr(getattr(switches, "dump_config", None), "dump_xpu", False):
            manual = load_manual_third_parties(testcase, manual_case, device_outputs=ref_nps)
            if manual is not None:
                process_ctx.notify_status("OnLoadManualXpu")
                dump_manual_xpu_archive(testcase, manual)
                dump_xpu_slot_files(testcase, switches, manual, MANUAL_XPU_PROVIDER)
                third_parties = manual
                xpu_results = manual_xpu_results()
        if collection_mode and xpu_results is None:
            process_ctx.notify_status("OnXpuProfiling")
            priority, third_parties, xpu_results = collect_third_party(
                op_name=testcase.api_name,
                inputs=e2e_xpu_inputs(testcase, raw_inputs),
                input_names=e2e_xpu_input_names(testcase),
                op_type=None,
                attributes=testcase.attributes or {},
                testcase_name=testcase.testcase_name,
                switches=switches,
                need_data=need_3party or golden_is_xpu,
                input_dtypes=e2e_xpu_input_dtypes(testcase),
            )
            dump_xpu_slot_files(testcase, switches, third_parties, priority or "xpu")
        # cross_check 三方缺失告警：同步/重叠/manual 三来源统一在此判
        if need_3party and third_parties is None:
            logging.warning(
                "[%s] cross_check configured but no third_party output "
                "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                testcase.testcase_name,
            )
    _discard_unconsumed(xpu_dispatch)
    if golden_is_xpu:
        # 三方输出升格 golden 位：三方输出即 golden；失败语义 = 整例 FAIL
        # （golden 位真值源缺失，在 compare 前显式落败——比等占位串走
        # validate FAIL 更早、信息更明确；两路殊途同归）。
        if third_parties is None:
            logging.error(
                "[%s] golden-mode XPU: no third_party output (XPU down / no third_party declaration); case FAIL",
                testcase.testcase_name,
            )
            return_struct.construct("GOLDEN_FAILURE", "FAIL", perf)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return None, None, None, True
        golden_nps = list(third_parties)
        # golden 性能回填：三方执行即 golden 真值，device_us 进 golden_* 列
        from ..common.golden import xpu_golden_perf

        return_struct.set_golden_perf(xpu_golden_perf(xpu_results), source="xpu")
        dump_goldens(testcase, golden_nps, switches)
    return_struct.xpu_metrics = format_xpu_metrics(xpu_results) if xpu_results else {}
    return third_parties, xpu_results, golden_nps, False


def _compare_and_finalize(
    testcase, switches, process_ctx, raw_inputs, outputs, golden_nps, third_parties, api, return_struct, backend
):
    """比对收尾：eager / graph 各形态 dump + 精度评估 + 终态屏显。"""
    if outputs.result_nps is None:
        # eager 模式被显式关闭（--eager=false）时不占 NO_OUTPUT 语义——
        # 主判据由图模式承载（ref_nps 回退链）；仅当 eager 本应执行却无产出时报错。
        if _should_run_eager(switches):
            return_struct.eager_precision = "NO_OUTPUT"
        if not api.graph_enabled:
            _profiling_end_print(testcase, return_struct, golden_nps, switches)
            return
    else:
        process_ctx.notify_status("OnDumpOutput")
        dump_outputs(testcase, outputs.result_nps, switches, segment="eager", backend=backend)

        # pre_compare 随 evaluate_eager_precision 内的统一比对入口生效（不再
        # 编排层原地改写——变换不跨 eager→graph 泄漏，golden 每比对独立复制）
        process_ctx.notify_status("OnEagerComparison")
        evaluate_eager_precision(
            testcase, raw_inputs, outputs.result_nps, golden_nps, switches, outputs.perf, return_struct, third_parties
        )

    if api.graph_enabled:
        process_ctx.notify_status("OnGraphComparison")
        if getattr(switches, "aclgraph_enabled", False) and outputs.graph_aclgraph_nps:
            dump_outputs(testcase, outputs.graph_aclgraph_nps, switches, segment="graph_aclgraph", backend=backend)
        if switches.cst_switches.enabled and outputs.graph_cst_nps:
            dump_outputs(testcase, outputs.graph_cst_nps, switches, segment="graph_cst", backend=backend)
        if switches.dyn_switches.enabled and outputs.graph_dyn_nps:
            dump_outputs(testcase, outputs.graph_dyn_nps, switches, segment="graph_dyn", backend=backend)
        if getattr(switches, "aclgraph_enabled", False):
            evaluate_graph_precision(
                testcase,
                raw_inputs,
                outputs.graph_aclgraph_nps,
                golden_nps,
                switches,
                return_struct,
                "aclgraph",
                outputs.graph_aclgraph_perf,
                third_parties,
                err_msg=outputs.graph_aclgraph_err,
            )
        if switches.cst_switches.enabled:
            evaluate_graph_precision(
                testcase,
                raw_inputs,
                outputs.graph_cst_nps,
                golden_nps,
                switches,
                return_struct,
                "static",
                outputs.graph_cst_perf,
                third_parties,
                err_msg=outputs.graph_cst_err,
            )
        if switches.dyn_switches.enabled:
            evaluate_graph_precision(
                testcase,
                raw_inputs,
                outputs.graph_dyn_nps,
                golden_nps,
                switches,
                return_struct,
                "dynamic",
                outputs.graph_dyn_perf,
                third_parties,
                err_msg=outputs.graph_dyn_err,
            )

    _profiling_end_print(testcase, return_struct, golden_nps, switches)


def _run_profile(testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct):
    """单卡用例编排：输入 → golden 前置 → 设备执行 → 三方 → 比对（对齐 kernel/
    aclnn/geir 的阶段序）。

    步骤函数返回 early=True 表示该用例已收尾（结果已写入 return_struct）。
    """
    process_ctx = process_context()
    deterministic_level = resolve_deterministic_level(switches, testcase)
    return_struct.batch_consistency_id = getattr(testcase, "batch_consistency_id", None)

    plan, dyn_indexes, early = _check_preconditions(testcase, backend, switches, return_struct)
    if early:
        return

    manual_case, dump_store, early = _resolve_manual_stores(testcase, switches, process_ctx, return_struct)
    if early:
        return

    if _require_dynamic_golden_source(testcase, switches, manual_case, dyn_indexes, return_struct):
        return

    adapter, raw_inputs, prepared_inputs, early = _generate_input(
        testcase, backend, switches, process_ctx, plan, manual_case, dump_store, dev_id, return_struct
    )
    if early:
        return

    # golden 先于落盘与设备执行（对齐 kernel：tolerance→golden→dump）
    golden = _prepare_golden(
        testcase,
        backend,
        switches,
        process_ctx,
        manual_case,
        raw_inputs,
        dev_id,
        return_struct,
        device_grant_events,
        device_granted_indices,
    )
    if golden.early:
        return

    if _manual_data_dump(
        testcase, switches, process_ctx, dump_store, manual_case, prepared_inputs, golden.nps, return_struct
    ):
        return

    api = _resolve_api(testcase, adapter, backend, switches)

    # XPU 三方 dispatch 前置（NPU 执行期重叠；预解析不命中返回 None 走同步）
    xpu_dispatch = _dispatch_xpu(testcase, switches, process_ctx, manual_case, raw_inputs)
    # golden-XPU：三方即 golden，设备前收割（FAIL 不空跑设备——对齐 kernel/
    # aclnn/geir）。manual 源（dispatch None）由执行后 _collect_xpu 装载。
    if (
        switches.golden_mode == "XPU"
        and xpu_dispatch is not None
        and _xpu_promote_golden(testcase, xpu_dispatch, switches, process_ctx, return_struct)
    ):
        return

    outputs = _device_profiling(
        testcase,
        backend,
        switches,
        process_ctx,
        dev_id,
        device_grant_events,
        device_granted_indices,
        plan,
        adapter,
        api,
        raw_inputs,
        deterministic_level,
        return_struct,
    )

    # 被测侧输出(device 结果)。容差按它的 dtype 解析,不能按 golden 的——
    # Promote 会把 golden 抬到 fp64,拿 golden dtype 去 resolve_tolerance 要么查不到该
    # dtype 的档、要么把 cross_check 判丢,三方就静默不采集了。
    ref_nps = outputs.result_nps or outputs.graph_cst_nps or outputs.graph_dyn_nps or outputs.graph_aclgraph_nps

    golden_nps = golden.nps
    third_parties, _, golden_nps, early = _collect_xpu(
        testcase,
        switches,
        process_ctx,
        manual_case,
        raw_inputs,
        ref_nps,
        golden_nps,
        outputs.perf,
        return_struct,
        xpu_dispatch=xpu_dispatch,
    )
    if early:
        return

    _compare_and_finalize(
        testcase, switches, process_ctx, raw_inputs, outputs, golden_nps, third_parties, api, return_struct, backend
    )
    if deterministic_level == 3:
        # dl=3 batch 一致性：比对后切 output 0 逐样本 MD5 → batch_summary
        # （随 ReturnStructure 回传，主进程 cleanup 按 bcid 分组比对）
        return_struct.batch_summary = assemble_batch_summary(
            ref_nps, testcase, getattr(testcase, "batch_consistency_id", None)
        )
