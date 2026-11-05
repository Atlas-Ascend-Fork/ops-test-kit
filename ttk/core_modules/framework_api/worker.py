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
（比对实现见 compare.py，eager 执行见 adapter.run_eager，图模式经 adapter.run_graph）。
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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ttk.core_modules.common.dump import (
    dump_manual_dataset,
    load_manual_goldens,
    manual_data_dump_case,
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
    evaluate_eager_precision,
    evaluate_graph_precision,
)
from .golden import generate_golden_data, has_dynamic_golden_source
from .input_generate import generate_inputs, set_runtime_tensors
from .result import FrameworkApiReturnStructure
from .utils import (
    WARMUP_COUNT,
    RunCtx,
    align_4bit_golden,
    resolve_npu_preprocess_hook,
)
from .xpu import e2e_xpu_input_dtypes, e2e_xpu_input_names, e2e_xpu_inputs


def _stage_rank_inputs(testcase, switches, backend, plan, rank_template, ndev, tmp_dir):
    """逐 rank 生成输入并汇成 inputs.npz（rank 特异种子 + 跨 rank 共享位对齐）。

    实现位于 torch/rank_inputs.py（多卡为 torch 专属通路），惰性
    import 避免 TF 通路加载 torch 子包。
    """
    from .torch.rank_inputs import stage_rank_inputs

    return stage_rank_inputs(testcase, switches, backend, plan, rank_template, ndev, tmp_dir)


def _should_run_eager(switches) -> bool:
    """eager（单算子直调）执行开关，由 --eager 控制。关闭时 eager_* 列留空，
    主判据由图模式腿承载。"""
    return bool(getattr(switches, "eager_enabled", True))


def _mirror_cst_failure_on_dedup(testcase, switches, return_struct, error_text):
    """失败路径镜像：eager 异常时若 eager≡静态图去重成立，graph_cst 列写同源
    错误文本（perf/输出留空）。判据解析失败不掩盖原异常。"""
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

    # --seed 用例级播种先于一切数据生成；多卡 per-rank 重播种在其后生效
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
        # eager 列与镜像列共用同一份 _attach_plog_errors 产物（两列恒同值）
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
        # 不告警则移动失败静默：instr.bin 被下一用例覆盖，用户无任何线索。
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
    # per-rank 生成走声明态模板深拷贝（generate_inputs 会覆写槽表），
    # 防共享 testcase 被逐 rank 覆写。
    rank_template = copy.deepcopy(testcase)
    try:
        raw_inputs = generate_inputs(testcase, switches, backend, plan)
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Input generation failure")
        return_struct.eager_precision = "INPUT_GEN_FAILURE"
        return_struct.precision_status = "FAIL"
        return None, None, True
    return rank_template, raw_inputs, False


def _run_rank_subprocess(testcase, switches, plan, input_path, tmp_dir, device_ids, return_struct):
    """拉起 rank 子进程组（torch/multi_device.py 脚本直启）并守错。

    父进程以 proc_timeout 看护总时长（覆盖 rank ready / HCCL init / 执行段，
    与 Task 级 proc_timeout 语义对齐），超时杀 launcher 并落 MULTI_DEVICE_TIMEOUT
    ——launcher 被 kill 后其 rank 孙进程由自身 HCCL init timeout 兜底退出。
    失败保留 tmp_dir 现场（rc≠0 路径）；超时/无结果文件路径清理 tmp_dir。
    返回 (result_path, early)。
    """
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
            # 部分 torch 版本退出阶段有析构崩溃（rc=-6）但结果已写出：
            # 优先采用实际结果，避免误判 FAIL。
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
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)
        return_struct.eager_precision = "MULTI_DEVICE_TIMEOUT"
        return_struct.precision_status = "FAIL"
        return None, True
    if not os.path.exists(result_path):
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


@dataclass
class _ProfileContext:
    """单卡用例编排的步骤上下文：阶段产物挂在字段上，步骤函数签名为
    (ctx) -> bool（True = 用例已收尾），替代此前的长位置参数链。"""

    testcase: object
    backend: object
    switches: object
    dev_id: int
    device_grant_events: dict
    device_granted_indices: dict
    return_struct: FrameworkApiReturnStructure
    deterministic_level: int
    # 阶段产物
    process_ctx: object = None
    plan: object = None
    dyn_indexes: tuple = ()
    manual_case: object = None
    dump_case: object = None
    adapter: object = None
    raw_inputs: object = None
    prepared_inputs: object = None
    golden_outputs: object = None
    api_binding: object = None
    npu_results: object = None
    xpu_dispatch: object = None
    xpu_outputs: object = None
    xpu_results: object = None


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
    """cross_check/mix_tolerance 判据要求 golden 升精度（标杆须高于被测 dtype）。

    锚点 = CSV 声明输出 dtype；golden_mode=XPU/NPU 与升精度互斥。"""
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
    """生成 golden；promote=True 时临时挂 golden_mode_override=Promote，生成后
    回写原值。dev_id/return_struct 仅 golden-mode NPU 使用。"""
    if not promote:
        return generate_golden_data(testcase, raw_inputs, switches, backend, dev_id=dev_id, return_struct=return_struct)
    prev = getattr(testcase, "golden_mode_override", None)
    testcase.golden_mode_override = "Promote"
    try:
        return generate_golden_data(testcase, raw_inputs, switches, backend, dev_id=dev_id, return_struct=return_struct)
    finally:
        testcase.golden_mode_override = prev


def _dump_only_end(testcase, return_struct, golden_nps, switches):
    """dump-only（--no-prof）收尾行（MANUAL_DATA_PREPARED）。

    纯收尾：数据落盘在 _manual_data_dump（对齐 kernel/aclnn 分层）。
    """
    return_struct.construct("MANUAL_DATA_PREPARED", "PASS", None)
    _profiling_end_print(testcase, return_struct, golden_nps, switches)


def _attach_plog_errors(err_msg):
    """异常文本折叠为单行并拼接 plog [ERROR] 行（CSV 单元格不能带换行）。"""
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
    """图模式统一执行入口：返回 (nps, perf, det_status, err_msg)，异常转失败
    结果不向上抛，其余图模式照常执行。"""
    try:
        nps, perf, det_status, err = graph_fn(*args, **kwargs)
        return nps, perf, det_status, _attach_plog_errors(err)
    except Exception as exc:
        logging.error(f"Graph {mode_label} execution failed: {exc}", exc_info=True)
        return [], None, None, _attach_plog_errors(f"Graph {mode_label} execution failed: {exc}")


def _capture_peak_memory(backend, dev_id, return_struct, fn, *args, **kwargs):
    """NPU: reset peak stats, run *fn*, record max_memory_allocated (MB) into npu_memory.

    Same caliber as the remote XPU executor (remote/server/executor.py:
    reset_peak -> invoke -> max_memory_allocated / 1e6): peak covers input
    allocations and execution-time temporaries inside *fn*. CANN-internal
    workspace does not go through the torch allocator and is not counted
    (accelerator third-party library workspace is likewise excluded). Non-NPU
    backends pass through, npu_memory stays None.

    Memory-stat APIs are best-effort: any failure (like the remote executor's
    peak_memory_mb=NA) leaves npu_memory untouched instead of failing the case,
    and never masks fn's own result/exception.
    """
    if not backend.is_npu():
        return fn(*args, **kwargs)
    try:
        backend.reset_peak_memory_stats(dev_id)
    except Exception:
        logging.exception(f"[E2E] reset_peak_memory_stats failed on device {dev_id}; npu_memory not captured")
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    finally:
        try:
            peak = backend.max_memory_allocated(dev_id)
        except Exception:
            logging.exception(f"[E2E] max_memory_allocated failed on device {dev_id}; npu_memory not captured")
            peak = None
        if peak is not None:
            return_struct.npu_memory = peak / 1e6


def _discard_unconsumed(xpu_dispatch):
    """后台三方句柄未被消费时弃置（止损未用请求）。"""
    if xpu_dispatch is not None and xpu_dispatch.pending:
        xpu_dispatch.discard()


def _cross_check_declared(testcase, switches, ref_nps, golden_nps, golden_is_xpu) -> bool:
    """cross_check 判据是否声明的解析：dtype 源 = 设备输出，缺省退回 golden。"""
    golden_real = bool(golden_nps) and not any(isinstance(g, str) for g in golden_nps)
    if not (golden_is_xpu or golden_real or ref_nps):
        return False
    tolerance = SpecResolver(testcase.api_name, "e2e", switches.plugin_path).resolve_tolerance()
    # 按被测输出 dtype 解析容差（golden 升精度后会走偏）；无设备结果退回 golden。
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


def _run_profile(testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct):
    """单卡用例编排：输入 → golden 前置 → 设备执行 → 三方 → 比对（对齐 kernel/
    aclnn/geir 的阶段序）。

    步骤函数签名为 (ctx) -> bool：返回 True 表示该用例已收尾（结果已写入
    return_struct）；阶段产物挂在 _ProfileContext 上，不再沿位置参数链膨胀。
    """
    ctx = _ProfileContext(
        testcase,
        backend,
        switches,
        dev_id,
        device_grant_events,
        device_granted_indices,
        return_struct,
        resolve_deterministic_level(switches, testcase),
    )
    ctx.process_ctx = process_context()
    return_struct.batch_consistency_id = getattr(testcase, "batch_consistency_id", None)

    # 前六步失败短路：任一早期收尾即返回
    for step in (
        _check_preconditions,
        _resolve_manual_stores,
        _check_dynamic_golden_source,
        _generate_input,
        _prepare_golden,
        _manual_data_dump,
    ):
        if step(ctx):
            return

    _resolve_api(ctx)

    # XPU 三方预派发（不命中返回 None，走执行后同步采集）
    _dispatch_xpu(ctx)
    # golden-XPU：三方即 golden，设备前收割；manual 源由执行后 _collect_xpu 装载。
    if switches.golden_mode == "XPU" and ctx.xpu_dispatch is not None and _xpu_promote_golden(ctx, ctx.xpu_dispatch):
        return

    # no-prof 各阶段内部早退：_resolve_api 不 import 被测 API、_device_profiling
    # 跳过设备执行、_compare_and_finalize 走 dump-only 收尾
    _device_profiling(ctx)

    if _collect_xpu(ctx):
        return

    _compare_and_finalize(ctx)
    if ctx.deterministic_level == 3:
        # dl=3 batch 一致性：比对后切 output 0 逐样本 MD5，随返回结构回传
        return_struct.batch_summary = assemble_batch_summary(
            ctx.npu_results.device_outputs, testcase, getattr(testcase, "batch_consistency_id", None)
        )


def _check_preconditions(ctx) -> bool:
    """前置校验：param plan 解析 + -1 动态槽门（执行需 NPU 前处理 hook；无则 FAIL）。"""
    ctx.plan = ctx.testcase.get_param_plan()
    if ctx.plan is None:
        ctx.return_struct.eager_precision = "PARAM_PLAN_FAILURE"
        ctx.return_struct.precision_status = "FAIL"
        logging.error(f"[{ctx.testcase.testcase_name}] Cannot resolve param plan for {ctx.testcase.api_name}")
        return True

    ctx.dyn_indexes = tuple(getattr(ctx.testcase, "dyn_input_slot_indexes", ()) or ())
    # Prepare/dump（--no-prof）only generates host inputs and records None for
    # dynamic slots.  It must work on CPU; the hook is invoked later during NPU
    # replay.  Direct and replay execution still require an NPU backend.
    if (
        ctx.dyn_indexes
        and not ctx.switches.no_prof
        and (not ctx.backend.is_npu() or resolve_npu_preprocess_hook(ctx.testcase, ctx.switches) is None)
    ):
        logging.error(
            f"[{ctx.testcase.testcase_name}] tensor_view_shapes declares -1 slot(s) {ctx.dyn_indexes}, "
            "but no npu_preprocess hook is registered"
        )
        ctx.return_struct.eager_precision = "DYN_SHAPE_REQUIRES_NPU_PREPROCESS"
        ctx.return_struct.precision_status = "FAIL"
        return True

    # -1 动态槽以 None 空标记落盘，物化钩子留待 NPU 重放执行。
    return False


def _resolve_manual_stores(ctx) -> bool:
    """manual 数据开仓：--no-prof 建写方向 store；load 方向解析用例（目录空时
    首跑回落生成+落盘）。产物入 ctx（manual_case/dump_case）。"""
    ctx.manual_case = None
    ctx.dump_case = None
    try:
        if ctx.switches.no_prof:
            # dump 方向：--no-prof 落盘（bridge 已保证写方向激活时设置 manual_data_dirs）
            ctx.dump_case = manual_data_dump_case(ctx.testcase, ctx.switches)
    except Exception as exc:
        logging.exception(f"[{ctx.testcase.testcase_name}] Manual data dump store failure")
        ctx.return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
        _profiling_end_print(ctx.testcase, ctx.return_struct, switches=ctx.switches)
        return True
    if not ctx.switches.no_prof and ctx.switches.manual_data_dirs:
        try:
            ctx.manual_case = resolve_manual_case(
                ctx.testcase,
                "e2e",
                ctx.switches,
                notify_before_load=lambda: ctx.process_ctx.notify_status("OnLoadManualData"),
                require_goldens=False,
            )
        except Exception as exc:
            logging.exception(f"[{ctx.testcase.testcase_name}] Manual data loading failure")
            ctx.return_struct.eager_precision = f"MANUAL_DATA_READ_FAILURE: {exc}"
            ctx.return_struct.precision_status = "FAIL"
            return True
        if ctx.manual_case is None and ctx.switches.dump_config.dump_input:
            # load 方向目录全空的首跑：正常生成并落盘（后续运行即可加载）
            try:
                ctx.dump_case = manual_data_dump_case(ctx.testcase, ctx.switches)
            except Exception as exc:
                logging.exception(f"[{ctx.testcase.testcase_name}] Manual data dump store failure")
                ctx.return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
                _profiling_end_print(ctx.testcase, ctx.return_struct, switches=ctx.switches)
                return True
    return False


def _check_dynamic_golden_source(ctx) -> bool:
    """动态槽用例的 golden 来源门：CPU 路径 -1 槽保持 None，须有插件/落盘/禁用
    兜底；input-only prepare（--no-prof 且不落 golden）豁免。"""
    needs_dynamic_golden = not ctx.switches.no_prof or ctx.switches.dump_config.dump_golden
    if (
        ctx.dyn_indexes
        and needs_dynamic_golden
        and not has_dynamic_golden_source(ctx.testcase, ctx.switches, ctx.manual_case)
    ):
        logging.error(
            f"[{ctx.testcase.testcase_name}] dynamic slot(s) {ctx.dyn_indexes} remain None on the CPU path; "
            "provide an E2E Golden plugin, replay saved Goldens, or disable Golden comparison"
        )
        ctx.return_struct.eager_precision = "DYN_SHAPE_REQUIRES_GOLDEN_SOURCE"
        ctx.return_struct.precision_status = "FAIL"
        return True
    return False


def _generate_input(ctx) -> bool:
    """输入生成（含 manual 快照与 dump）；产物入 ctx（adapter/raw_inputs/prepared_inputs）。"""
    # adapter 先于输入生成创建：触发 tf/ 子包导入注册 L1 张量构造工厂
    # （framework_hooks，generate_inputs 依赖）；缺了会静默丢失量化 dtype/
    # const/mutable 位处理。torch 侧幂等，无副作用。
    ctx.adapter = adapter_for_testcase(ctx.testcase)

    ctx.process_ctx.notify_status("OnGenInput")
    # 输入生成不触设备：TF 张量在 CPU 作用域物化，设备张量由执行侧在 open
    # 后按 raw_inputs 重建（torch 侧张量本就 CPU-born）。
    try:
        if ctx.manual_case is not None:
            ctx.raw_inputs = generate_inputs(
                ctx.testcase,
                ctx.switches,
                ctx.backend,
                ctx.plan,
                stored_inputs=ctx.manual_case.inputs,
                cpu_placement=True,
            )
        else:
            ctx.raw_inputs = generate_inputs(ctx.testcase, ctx.switches, ctx.backend, ctx.plan, cpu_placement=True)
    except Exception as exc:
        logging.exception(f"[{ctx.testcase.testcase_name}] Input generation failure")
        prefix = "MANUAL_DATA_READ_FAILURE" if ctx.manual_case is not None else "INPUT_GEN_FAILURE"
        ctx.return_struct.eager_precision = f"{prefix}: {exc}" if ctx.manual_case is not None else prefix
        ctx.return_struct.precision_status = "FAIL"
        return True

    ctx.prepared_inputs = None
    if ctx.dump_case is not None and ctx.manual_case is None:
        # 生成态数据将写入 store：先快照（设备执行可能原地改动 backing storage）
        try:
            ctx.prepared_inputs = snapshot_manual_values(
                ctx.testcase.np_storages if ctx.testcase.np_storages is not None else ctx.raw_inputs,
                "input",
            )
        except Exception as exc:
            logging.exception(f"[{ctx.testcase.testcase_name}] Manual data snapshot failure")
            ctx.return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(ctx.testcase, ctx.return_struct, switches=ctx.switches)
            return True

    if ctx.dump_case is None:
        ctx.process_ctx.notify_status("OnDumpInput")
        dump_inputs(ctx.testcase, ctx.raw_inputs, ctx.switches)
    return False


def _prepare_golden(ctx) -> bool:
    """golden 装载/生成（先于落盘与设备执行）：有已存真值装载（replay 优先），
    否则按模式裁决——XPU 占位、NPU 自起设备锁生成、Enable 现算。失败返回 True。"""
    promote = _needs_golden_promote(ctx.testcase, ctx.switches)
    if ctx.switches.no_prof and not ctx.switches.dump_config.dump_golden:
        return False
    if ctx.manual_case is not None and ctx.manual_case.has_goldens:
        try:
            ctx.process_ctx.notify_status("OnLoadManualGolden")
            nps = load_manual_goldens(
                ctx.manual_case,
                shapes=ctx.testcase.flat_output_view_shapes,
                dtypes=ctx.testcase.flat_output_dtypes,
            )
            dump_goldens(ctx.testcase, nps, ctx.switches)
            ctx.golden_outputs = nps
            return False
        except Exception as exc:
            logging.exception(f"[{ctx.testcase.testcase_name}] Static replay golden load failed")
            ctx.return_struct.construct(f"MANUAL_DATA_READ_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(ctx.testcase, ctx.return_struct, switches=ctx.switches)
            return True
    mode = golden_mode(ctx.testcase, ctx.switches)
    if mode == "XPU":
        # XPU 真值由三方在设备后回填（_collect_xpu），本侧占位即可——无需延迟
        ctx.golden_outputs = ["GOLDEN_FAILURE"]
        return False
    if mode == "NPU":
        # 自起设备锁、DUT 前生成（与 kernel gen_output 同款单步编排
        # ；锁段与 DUT 串行错开）
        with DeviceLock(
            ctx.process_ctx,
            ctx.dev_id,
            use_device=ctx.backend.has_device(),
            grant_event=(ctx.device_grant_events or {}).get(ctx.dev_id),
            granted_idx=(ctx.device_granted_indices or {}).get(ctx.dev_id),
        ):
            # golden 也上板执行时与被测段共用同一 CubeMathType 精度模式（避免
            # 沿用上一用例的进程级状态造成 golden/结果模式错位）
            adapter_for_testcase(ctx.testcase).setup_device_state(ctx.backend, ctx.testcase, ctx.process_ctx)
            nps = _generate_golden(
                ctx.testcase,
                ctx.raw_inputs,
                ctx.switches,
                ctx.backend,
                promote,
                dev_id=ctx.dev_id,
                return_struct=ctx.return_struct,
            )
        ctx.golden_outputs = nps
        return False
    nps = _generate_golden(
        ctx.testcase,
        ctx.raw_inputs,
        ctx.switches,
        ctx.backend,
        promote,
        dev_id=ctx.dev_id,
        return_struct=ctx.return_struct,
    )
    ctx.golden_outputs = nps
    return False


def _manual_data_dump(ctx) -> bool:
    """in+golden 落盘（manual-data store）。

    full-flow：golden_nps 就绪（非 None）即落——golden 已在 _prepare_golden
    内一次解析完成（回放装载 / XPU 占位 / NPU 自锁段生成 / Enable 现算），
    不存在 deferred 二次落盘。落当次实际产物（判据要求 Promote 时即升精度
    标杆），文件名 dtype 自描述、回放按其装载；golden 缺失（dump_golden
    关闭 / 占位串）跳过 golden 位。返回 early。

    no-prof（dump 方向）不在此收尾：不上板只跳过本地 device，收尾由
    _compare_and_finalize 的 dump-only 分支统一承担；远端三方采集照常。
    """
    if not ctx.switches.no_prof and ctx.golden_outputs is None:
        return False
    if ctx.dump_case is not None and ctx.manual_case is None:
        # golden-XPU 占位串过滤后仅落 input，与 kernel/aclnn/geir 同口径
        try:
            ctx.process_ctx.notify_status("OnWriteManualData")
            case_dir = dump_manual_dataset(
                ctx.dump_case, ctx.testcase, "e2e", ctx.prepared_inputs, ctx.golden_outputs or [], ctx.switches
            )
            logging.info(f"[{ctx.testcase.testcase_name}] manual data dumped: {case_dir}")
        except Exception as exc:
            logging.exception(f"[{ctx.testcase.testcase_name}] Manual data dump failure")
            ctx.return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(ctx.testcase, ctx.return_struct, switches=ctx.switches)
            return True
    return False


def _resolve_api(ctx):
    """主 API 解析与图模式开关裁剪（backend 不支持时降级关）。"""
    if ctx.switches.no_prof:
        # 跳过 device 执行，不 resolve 被测 API（制备环境可能未装扩展包），
        # api_binding 留空，后续阶段早退不消费
        return
    resolved, is_tensor_method = ctx.adapter.load_api(ctx.testcase.api_name)
    is_inplace = ctx.adapter.is_inplace(resolved, is_tensor_method)
    graph_enabled = (
        ctx.switches.cst_switches.enabled
        or ctx.switches.dyn_switches.enabled
        or getattr(ctx.switches, "aclgraph_enabled", False)
    )
    if graph_enabled and not ctx.backend.supports_graph_mode():
        if not _should_run_eager(ctx.switches):
            # --eager=false 且图模式被后端降级（如 CPU 不支持）：无可执行路径。
            # 提前显式失败（run 的 except 收敛为 FAIL 行），不白算 golden、
            # 不留 status 为空的静默结果行。
            raise ValueError(
                f"--eager=false but backend {ctx.backend.device_type()} does not support graph mode; "
                "no executable leg remains"
            )
        logging.warning(
            f"Graph mode not supported by ctx.backend {ctx.backend.device_type()}, skipping graph execution"
        )
        graph_enabled = False
    ctx.api_binding = SimpleNamespace(
        resolved=resolved, is_tensor_method=is_tensor_method, is_inplace=is_inplace, graph_enabled=graph_enabled
    )


def _dispatch_xpu(ctx):
    """三方预派发：manual 声明命中即不派远端（probe 短路，装载见
    _collect_xpu）；golden-XPU 同步收割；其余可重叠场景后台派发；无效场景
    由 _collect_xpu 同步补采。"""
    from ttk.core_modules.common.xpu import (
        XPU_DATA,
        XpuDispatch,
        has_manual_xpu_source,
        overlap_deadline_s,
        snapshot_xpu_inputs,
        xpu_join_timeout,
        xpu_mode,
        xpu_overlap_enabled,
    )

    golden_is_xpu = ctx.switches.golden_mode == "XPU"
    need_xpu_data = _cross_check_token(ctx)
    mode = xpu_mode(ctx.switches, need_xpu_data)
    if mode & XPU_DATA and has_manual_xpu_source(ctx.testcase, ctx.manual_case):
        return  # manual 声明命中：不派远端；装载/升格由 _collect_xpu 设备后完成
    if not mode:
        return
    if golden_is_xpu or not xpu_overlap_enabled():
        # golden-XPU：三方即 golden，设备前同步收割（FAIL 不空跑设备）
        ctx.process_ctx.notify_status("OnXpuProfiling")
        ctx.xpu_dispatch = XpuDispatch.ready(
            _collect_xpu_remote(ctx, need_xpu_data, e2e_xpu_inputs(ctx.testcase, ctx.raw_inputs))
        )
        return
    inputs = snapshot_xpu_inputs(e2e_xpu_inputs(ctx.testcase, ctx.raw_inputs))
    budget = overlap_deadline_s(ctx.switches)
    ctx.process_ctx.notify_status("OnXpuDispatch")
    ctx.xpu_dispatch = XpuDispatch.background(
        _collect_xpu_remote, ctx, need_xpu_data, inputs, budget, join_timeout=xpu_join_timeout(budget)
    )


def _cross_check_token(ctx) -> bool:
    """cross_check 判据探测（声明 dtype 可解析；异常/缺档回退 False）。"""
    output_dtypes = tuple(str(d) for d in (getattr(ctx.testcase, "flat_output_dtypes", ()) or ()))
    if not output_dtypes:
        return False
    try:
        standards = resolve_tolerance(
            SpecResolver(ctx.testcase.api_name, "e2e", ctx.switches.plugin_path).resolve_tolerance(),
            ctx.testcase.flat_precision_tolerances,
            ctx.testcase.flat_absolute_precision,
            output_dtypes,
            ctx.switches.compare_method,
        )
    except Exception:  # noqa: BLE001  预解析失败（dtype 档缺失/配置错）回退 False
        return False
    return any(s.token == "cross_check" for s in standards)  # noqa: S105  # token 为精度标准名，非凭据


def _collect_xpu_remote(ctx, need_xpu_data, inputs, deadline_s=None):
    """e2e 远端采集（同步/后台共用）：返回 (third_parties, xpu_results, provider)。"""
    from ttk.core_modules.common.xpu import collect_third_party

    priority, third_parties, xpu_results = collect_third_party(
        op_name=ctx.testcase.api_name,
        inputs=inputs,
        input_names=e2e_xpu_input_names(ctx.testcase),
        op_type=None,
        attributes=ctx.testcase.attributes or {},
        testcase_name=ctx.testcase.testcase_name,
        switches=ctx.switches,
        need_xpu_data=need_xpu_data,
        input_dtypes=e2e_xpu_input_dtypes(ctx.testcase),
        deadline_s=deadline_s,
    )
    return third_parties, xpu_results, (priority or "xpu")


def _xpu_promote_golden(ctx, xpu) -> bool:
    """golden-XPU 设备前收割：三方输出升格为 golden；三方缺失则 FAIL 收尾。"""
    from ..common.golden import xpu_golden_perf

    third_parties, xpu_results, xpu_provider = xpu.collect()
    from ..common.xpu import dump_xpu_slot_files

    dump_xpu_slot_files(ctx.testcase, ctx.switches, third_parties, xpu_provider, dump_case=ctx.dump_case)
    if third_parties is None:
        logging.error(
            "[%s] golden-mode XPU: no third_party output (XPU down / no third_party declaration); case FAIL",
            ctx.testcase.testcase_name,
        )
        ctx.return_struct.construct("GOLDEN_FAILURE", "FAIL", None)
        _profiling_end_print(ctx.testcase, ctx.return_struct, switches=ctx.switches)
        return True
    ctx.golden_outputs = list(third_parties)
    ctx.return_struct.set_golden_perf(xpu_golden_perf(xpu_results), source="xpu")
    dump_goldens(ctx.testcase, ctx.golden_outputs, ctx.switches)
    return False


def _device_profiling(ctx):
    """设备执行（锁内 eager / graph×3 模式）与确定性状态汇总。"""
    if ctx.switches.no_prof:
        # 跳过 device 执行；空 npu_results 供 _collect_xpu 判据与
        # dump-only 收尾使用
        ctx.npu_results = SimpleNamespace(
            result_nps=None,
            perf=None,
            graph_cst_nps=None,
            graph_dyn_nps=None,
            graph_aclgraph_nps=None,
            device_outputs=(),
        )
        return
    # 进程设备在此就绪（首用例触发 open，进程级一次）；输入/golden 已在 CPU
    # 侧完成。TF 设备张量在 open 后按 raw_inputs 重建为 NPU-born；torch 侧
    # CPU 张量 to_device 逐模式上板。
    ctx.backend.set_device(ctx.dev_id)
    if ctx.adapter.framework == "tf":
        set_runtime_tensors(ctx.testcase, ctx.raw_inputs, ctx.backend)
    ctx.process_ctx.notify_status("OnAcquireLock")
    use_device = ctx.backend.has_device()
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
        ctx.process_ctx,
        ctx.dev_id,
        use_device=use_device,
        grant_event=ctx.device_grant_events.get(ctx.dev_id),
        granted_idx=ctx.device_granted_indices.get(ctx.dev_id),
    ):
        ctx.process_ctx.notify_status("OnProfilingPrint")
        # CubeMathType 精度模式先于一切执行段配置（进程级 torch_npu 状态，
        # eager + 三种图形态共享；非 torch 框架不导入）
        ctx.adapter.setup_device_state(ctx.backend, ctx.testcase, ctx.process_ctx)
        _profiling_print(ctx.testcase, ctx.backend, ctx.dev_id, ctx.switches)

        # 模式等价（adapter 钩子驱动）：onnx 的 eager ≡ 静态图（om 恒为静态
        # 图）——eager 请求改由 static 图执行承载、EAGER 列同源镜像，不走
        # run_eager；dyn/aclgraph 不去重：run_graph 快速失败落 GRAPH_EXEC_FAILURE。
        # getattr 防御 duck-typed 测试替身。
        eager_alias_fn = getattr(ctx.adapter, "eager_graph_alias", None)
        alias_cst = callable(eager_alias_fn) and eager_alias_fn() == "cst"
        # eager / graph 共用同一份 RunCtx（deterministic_level 已按用例解析；
        # run_count/warmup_count 由 switches 推出）
        run_ctx = RunCtx(
            backend=ctx.backend,
            dev_id=ctx.dev_id,
            switches=ctx.switches,
            resolved=ctx.api_binding.resolved,
            is_tensor_method=ctx.api_binding.is_tensor_method,
            deterministic_level=ctx.deterministic_level,
            run_count=ctx.switches.run_time,
            warmup_count=(
                WARMUP_COUNT if (ctx.switches.warmup and bool(getattr(ctx.switches, "task_profiling", True))) else 0
            ),
        )
        if alias_cst:
            # eager≡cst 的框架（onnx）：无真 eager，EAGER 列改由 static 图执行
            # 同源镜像（与旧 eager→cst 镜像方向相反），不再触碰 run_eager
            ctx.cst_alias = True
            if _should_run_eager(ctx.switches) or ctx.switches.cst_switches.enabled:
                ctx.process_ctx.notify_status("OnGraphCst")
                graph_cst_nps, graph_cst_perf, graph_cst_det, graph_cst_err = _run_graph_phase(
                    "static",
                    ctx.adapter.run_graph,
                    run_ctx,
                    testcase=ctx.testcase,
                    plan=ctx.plan,
                    is_inplace=ctx.api_binding.is_inplace,
                    raw_inputs=ctx.raw_inputs,
                    dynamic=False,
                )
                if _should_run_eager(ctx.switches):
                    result_nps, perf, eager_det_status = graph_cst_nps, graph_cst_perf, graph_cst_det
        elif _should_run_eager(ctx.switches):
            ctx.process_ctx.notify_status("OnEagerProfiling")
            result_nps, perf, eager_det_status = _capture_peak_memory(
                ctx.backend,
                ctx.dev_id,
                ctx.return_struct,
                ctx.adapter.run_eager,
                run_ctx,
                testcase=ctx.testcase,
                plan=ctx.plan,
                is_inplace=ctx.api_binding.is_inplace,
                raw_inputs=ctx.raw_inputs,
            )
        if ctx.api_binding.graph_enabled:
            graph_fn = ctx.adapter.run_graph
            if getattr(ctx.switches, "aclgraph_enabled", False):
                ctx.process_ctx.notify_status("OnGraphAclgraph")
                graph_aclgraph_nps, graph_aclgraph_perf, graph_aclgraph_det, graph_aclgraph_err = _run_graph_phase(
                    "aclgraph",
                    graph_fn,
                    run_ctx,
                    testcase=ctx.testcase,
                    plan=ctx.plan,
                    is_inplace=ctx.api_binding.is_inplace,
                    raw_inputs=ctx.raw_inputs,
                    dynamic=False,
                    is_aclgraph=True,
                )
            if ctx.switches.cst_switches.enabled and not alias_cst:
                # alias 框架已在上面统一执行 static（eager 列同源镜像）
                ctx.process_ctx.notify_status("OnGraphCst")
                graph_cst_nps, graph_cst_perf, graph_cst_det, graph_cst_err = _run_graph_phase(
                    "static",
                    graph_fn,
                    run_ctx,
                    testcase=ctx.testcase,
                    plan=ctx.plan,
                    is_inplace=ctx.api_binding.is_inplace,
                    raw_inputs=ctx.raw_inputs,
                    dynamic=False,
                )
            if ctx.switches.dyn_switches.enabled:
                ctx.process_ctx.notify_status("OnGraphDyn")
                graph_dyn_nps, graph_dyn_perf, graph_dyn_det, graph_dyn_err = _run_graph_phase(
                    "dynamic",
                    graph_fn,
                    run_ctx,
                    testcase=ctx.testcase,
                    plan=ctx.plan,
                    is_inplace=ctx.api_binding.is_inplace,
                    raw_inputs=ctx.raw_inputs,
                    dynamic=True,
                )
    gc.collect()
    _collect_sim_report(ctx.testcase, ctx.switches)

    det_statuses = [eager_det_status, graph_cst_det, graph_dyn_det, graph_aclgraph_det]
    det_statuses = [s for s in det_statuses if s is not None]
    if det_statuses:
        ctx.return_struct.deterministic_status = "FAIL" if "FAIL" in det_statuses else "PASS"

    ctx.npu_results = SimpleNamespace(
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
        # 四个执行腿回退取第一个有值的输出（判据 dtype 源与 dl=3 切片源）
        device_outputs=result_nps or graph_cst_nps or graph_dyn_nps or graph_aclgraph_nps,
    )


def _collect_xpu(ctx) -> bool:
    """三方采集收口：manual 装载、预派发收割、同步补采、落 xpu 槽位、
    判据告警与指标回填。golden-XPU 升格统一走 _xpu_promote_golden——
    远端源由 runner 设备前收割，manual 源装载后包 ready 句柄同入口。"""
    third_parties = None
    xpu_results = None
    golden_is_xpu = ctx.switches.golden_mode == "XPU"
    # 容差按被测腿输出的 dtype 解析（golden 升精度后会走偏/丢 cross_check）
    need_3party = _cross_check_declared(
        ctx.testcase, ctx.switches, ctx.npu_results.device_outputs, ctx.golden_outputs, golden_is_xpu
    )
    # 采集门槛：判据需 DATA（cross_check/golden-XPU）或 --xpu-perf/--dump xpu
    from ttk.core_modules.common.xpu import (
        MANUAL_XPU_PROVIDER,
        XPU_DATA,
        XpuDispatch,
        collect_third_party,
        dump_xpu_slot_files,
        load_third_parties,
        manual_xpu_results,
        xpu_mode,
    )

    collection_mode = xpu_mode(ctx.switches, need_3party)
    if golden_is_xpu or collection_mode:
        if ctx.xpu_dispatch is not None:
            # 预派发句柄（golden 同步 / 后台重叠）在此收割
            if ctx.xpu_dispatch.pending:
                ctx.process_ctx.notify_status("OnXpuCollect")
            third_parties, xpu_results, xpu_provider = ctx.xpu_dispatch.collect()
            dump_xpu_slot_files(ctx.testcase, ctx.switches, third_parties, xpu_provider, dump_case=ctx.dump_case)
        # manual 三方输出：需 DATA 时本地装载短路远端 dispatch，PERF-only 不走
        # manual。装载留在设备后而非 _dispatch_xpu：e2e 用例 CSV 常不声明输出
        # shape/dtype，裸路径的 manual_xpu_binaries 文件无自描述，须靠
        # device_outputs 推导解析；aclnn/kernel 有声明输出槽，故可前置装载。
        elif collection_mode & XPU_DATA:
            manual = load_third_parties(ctx.testcase, ctx.manual_case, device_outputs=ctx.npu_results.device_outputs)
            if manual is not None:
                ctx.process_ctx.notify_status("OnLoadManualXpu")
                xpu_results = manual_xpu_results()
                if golden_is_xpu:
                    # golden 升格同入口：promote 内落槽 + 回填 golden
                    if _xpu_promote_golden(ctx, XpuDispatch.ready((manual, xpu_results, MANUAL_XPU_PROVIDER))):
                        return True
                else:
                    dump_xpu_slot_files(
                        ctx.testcase, ctx.switches, manual, MANUAL_XPU_PROVIDER, dump_case=ctx.dump_case
                    )
                third_parties = manual
        if collection_mode and xpu_results is None:
            ctx.process_ctx.notify_status("OnXpuProfiling")
            priority, third_parties, xpu_results = collect_third_party(
                op_name=ctx.testcase.api_name,
                inputs=e2e_xpu_inputs(ctx.testcase, ctx.raw_inputs),
                input_names=e2e_xpu_input_names(ctx.testcase),
                op_type=None,
                attributes=ctx.testcase.attributes or {},
                testcase_name=ctx.testcase.testcase_name,
                switches=ctx.switches,
                need_xpu_data=need_3party,
                input_dtypes=e2e_xpu_input_dtypes(ctx.testcase),
            )
            dump_xpu_slot_files(ctx.testcase, ctx.switches, third_parties, priority or "xpu", dump_case=ctx.dump_case)
        # 三方缺失告警（预派发/manual/同步三来源统一判定）
        if need_3party and third_parties is None:
            logging.warning(
                "[%s] cross_check configured but no third_party output "
                "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                ctx.testcase.testcase_name,
            )
    _discard_unconsumed(ctx.xpu_dispatch)
    ctx.return_struct.xpu_metrics = format_xpu_metrics(xpu_results) if xpu_results else {}
    ctx.xpu_outputs = third_parties
    ctx.xpu_results = xpu_results
    return False


def _compare_and_finalize(ctx):
    """比对收尾：eager / graph 各形态 dump + 精度评估 + 终态屏显。"""
    if ctx.switches.no_prof:
        # 跳过 device 执行：无设备输出可比对，dump-only 收尾
        return _dump_only_end(ctx.testcase, ctx.return_struct, ctx.golden_outputs or [], ctx.switches)
    # golden 为 4bit dtype 时各输出腿先对齐封装（eager + 三种图形态）
    if ctx.golden_outputs:
        for output_nps in (
            ctx.npu_results.result_nps,
            ctx.npu_results.graph_cst_nps,
            ctx.npu_results.graph_dyn_nps,
            ctx.npu_results.graph_aclgraph_nps,
        ):
            if output_nps:
                align_4bit_golden(output_nps, ctx.golden_outputs)
    if ctx.npu_results.result_nps is None:
        # --eager=false 时不占 NO_OUTPUT 语义（主判据由图模式承载）；
        # 仅 eager 本应执行却无产出时报错。
        if _should_run_eager(ctx.switches):
            ctx.return_struct.eager_precision = "NO_OUTPUT"
        if not ctx.api_binding.graph_enabled:
            _profiling_end_print(ctx.testcase, ctx.return_struct, ctx.golden_outputs, ctx.switches)
            return None
    else:
        ctx.process_ctx.notify_status("OnDumpOutput")
        dump_outputs(ctx.testcase, ctx.npu_results.result_nps, ctx.switches, segment="eager")

        # pre_compare 随统一比对入口生效（变换不跨 eager→graph 泄漏）
        ctx.process_ctx.notify_status("OnEagerComparison")
        evaluate_eager_precision(
            ctx.testcase,
            ctx.raw_inputs,
            ctx.npu_results.result_nps,
            ctx.golden_outputs,
            ctx.switches,
            ctx.npu_results.perf,
            ctx.return_struct,
            ctx.xpu_outputs,
        )

    if getattr(ctx, "cst_alias", False) and ctx.npu_results.graph_cst_err:
        # eager≡cst 的框架（onnx）：static 执行失败时 EAGER 列落同源错误文本
        ctx.return_struct.eager_precision = ctx.npu_results.graph_cst_err

    if ctx.api_binding.graph_enabled:
        ctx.process_ctx.notify_status("OnGraphComparison")
        if getattr(ctx.switches, "aclgraph_enabled", False) and ctx.npu_results.graph_aclgraph_nps:
            dump_outputs(
                ctx.testcase,
                ctx.npu_results.graph_aclgraph_nps,
                ctx.switches,
                segment="graph_aclgraph",
            )
        if ctx.switches.cst_switches.enabled and ctx.npu_results.graph_cst_nps:
            dump_outputs(ctx.testcase, ctx.npu_results.graph_cst_nps, ctx.switches, segment="graph_cst")
        if ctx.switches.dyn_switches.enabled and ctx.npu_results.graph_dyn_nps:
            dump_outputs(ctx.testcase, ctx.npu_results.graph_dyn_nps, ctx.switches, segment="graph_dyn")
        if getattr(ctx.switches, "aclgraph_enabled", False):
            evaluate_graph_precision(
                ctx.testcase,
                ctx.raw_inputs,
                ctx.npu_results.graph_aclgraph_nps,
                ctx.golden_outputs,
                ctx.switches,
                ctx.return_struct,
                "aclgraph",
                ctx.npu_results.graph_aclgraph_perf,
                ctx.xpu_outputs,
                err_msg=ctx.npu_results.graph_aclgraph_err,
            )
        if ctx.switches.cst_switches.enabled:
            evaluate_graph_precision(
                ctx.testcase,
                ctx.raw_inputs,
                ctx.npu_results.graph_cst_nps,
                ctx.golden_outputs,
                ctx.switches,
                ctx.return_struct,
                "static",
                ctx.npu_results.graph_cst_perf,
                ctx.xpu_outputs,
                err_msg=ctx.npu_results.graph_cst_err,
            )
        if ctx.switches.dyn_switches.enabled:
            evaluate_graph_precision(
                ctx.testcase,
                ctx.raw_inputs,
                ctx.npu_results.graph_dyn_nps,
                ctx.golden_outputs,
                ctx.switches,
                ctx.return_struct,
                "dynamic",
                ctx.npu_results.graph_dyn_perf,
                ctx.xpu_outputs,
                err_msg=ctx.npu_results.graph_dyn_err,
            )

    _profiling_end_print(ctx.testcase, ctx.return_struct, ctx.golden_outputs, ctx.switches)
    return None
