#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Simulator profiling entry points for the kernel mode.

``run_kernel_sim(context)`` replaces the three per-mode ``execute_on_device`` calls
in ``kernel/worker.py`` when ``sw.backend == "npusim"``. It serializes each
enabled mode's ``RTSProfilingParam``, launches one NPUSim ``record`` whose
wrapper runs all modes inside the simulation, then rebuilds
``KernelExecuteResult`` objects from the wrapper-written output files.
"""

import json
import logging
import pickle
from pathlib import Path

from ttk.switches import get_global_storage

from . import case_writer
from . import report as sim_report
from .npusim_runner import run_record

# Kernel mode directory name -> profiling mode name used by execute_on_device /
# construct_profiling_param.
PROFILING_MODE_BY_DIR = {"dyn": "dynamic", "cst": "const", "bin": "binary"}


def _wrapper_error_reason(case_path: Path) -> str:
    """Prefer a wrapper-written traceback over a bare SIM_RESULT_MISSING.

    When the wrapper subprocess crashes before writing its result file it only
    leaves ``wrapper_error.json`` (see the wrapper templates). Attach that
    content so the user sees the real failure instead of an uninformative
    MISSING. Content is truncated to keep result cells / logs readable.
    """
    err_file = case_path / "wrapper_error.json"
    if err_file.is_file():
        content = err_file.read_text(encoding="utf-8").strip()
        if content:
            return f"SIM_RESULT_MISSING (wrapper_error): {content[-500:]}"
    return "SIM_RESULT_MISSING"


def _load_mode_result(case_path: Path, mode: str):
    from ttk.core_modules.kernel.result import KernelExecuteResult

    mdir = case_path / mode
    result_json = mdir / "result.json"
    if not result_json.is_file():
        return KernelExecuteResult.fail(_wrapper_error_reason(case_path))
    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # truncated/corrupt result.json
        return KernelExecuteResult.fail(f"SIM_RESULT_CORRUPT: {exc}")
    if not payload.get("ok"):
        return KernelExecuteResult.fail(payload.get("error") or "SIM_EXECUTION_FAILED")
    output_bytes = []
    idx = 0
    while True:
        bin_file = mdir / f"output_{idx}.bin"
        if not bin_file.is_file():
            break
        output_bytes.append(bin_file.read_bytes())
        idx += 1
    if not output_bytes:
        return KernelExecuteResult.fail("SIM_NO_OUTPUT")
    # output_bytes must be a list: npu/op/comparison.py assigns in place
    # (outputs[idx] = ...), matching the real-device _copy_output_from_hbm.
    return KernelExecuteResult(cycle=payload.get("cycle"), output_bytes=output_bytes, oob=payload.get("oob", "UNKNOWN"))


def _construct_param(context, mode: str, output_placeholder: bool):
    from ttk.core_modules.kernel.execute import construct_profiling_param
    from ttk.core_modules.kernel.result import RTSProfilingParam

    return RTSProfilingParam(*construct_profiling_param(context, mode, output_placeholder))


def _validate_param(param):
    """Mirror the validity checks in ``execute_on_device`` before serializing."""
    from ttk.core_modules.kernel.result import KernelExecuteResult

    if not param.switch:
        return KernelExecuteResult.fail("SUPPRESSED")
    if param.compile_result != "SUCC":
        return KernelExecuteResult.fail(param.compile_result)
    if not param.is_valid:
        return KernelExecuteResult.fail(param.fail_reason)
    if param.block_dim <= 0:
        return KernelExecuteResult.fail("INVALID_TILING")
    return None


def run_kernel_sim(context):
    """Execute all enabled kernel modes (dyn/cst/bin) via one NPUSim record.

    Returns a 3-tuple of KernelExecuteResult aligned to execute_on_device's
    dyn/cst/bin results.
    """
    from ttk.core_modules.kernel.info_keeper import OpInfoKeeper
    from ttk.core_modules.kernel.result import KernelExecuteResult

    sw = get_global_storage()
    # clear_case_dir() itself creates the directory, so case_dir() (no mkdir)
    # is enough here — avoid a redundant ensure_case_dir().
    case_path = case_writer.case_dir(sw, context.testcase_name)
    case_writer.clear_case_dir(sw, context.testcase_name)
    case_writer.dump_switches(sw, case_path)

    output_placeholder = OpInfoKeeper().op_output_defined(context.op_name)
    enabled = case_writer.enabled_kernel_modes(sw)
    results = {}
    serialized_any = False
    for mode in case_writer.KERNEL_MODES:
        if mode not in enabled:
            results[mode] = KernelExecuteResult.fail("SUPPRESSED")
            continue
        param = _construct_param(context, PROFILING_MODE_BY_DIR[mode], output_placeholder)
        invalid = _validate_param(param)
        if invalid is not None:
            results[mode] = invalid
            continue
        param.clear_atomic_output_workspace()
        case_writer.dump_kernel_param(param, case_path, mode)
        serialized_any = True

    if serialized_any:
        from .wrapper import write_kernel_wrapper

        wrapper = write_kernel_wrapper(sw, case_path)
        try:
            export_root = run_record(sw, wrapper, case_path, extra_argv=("-u", str(case_path)))
        except Exception as exc:  # noqa: BLE001 - surface as a structured fail
            # A record-level failure (cannsim missing, timeout, no archive)
            # must not crash the whole run; mark every enabled mode failed.
            logging.error("npusim record failed for %s: %s", context.testcase_name, exc)
            for mode in enabled:
                results[mode] = KernelExecuteResult.fail(f"SIM_RECORD_FAILED: {exc}")
            return results["dyn"], results["cst"], results["bin"]
        if sw.sim_report:
            sim_report.maybe_generate_sim_report(sw, case_path, export_root)
        for mode in enabled:
            results[mode] = _load_mode_result(case_path, mode)
    else:
        logging.warning("no kernel mode was serialized for %s; nothing to simulate", context.testcase_name)
    return results["dyn"], results["cst"], results["bin"]


def _load_aclnn_result(case_path: Path):
    from ttk.core_modules.aclnn.result import ApiProfilingResult

    final_json = case_path / "final_result.json"
    if not final_json.is_file():
        return ApiProfilingResult.fail(_wrapper_error_reason(case_path))
    try:
        payload = json.loads(final_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # truncated/corrupt final_result.json
        return ApiProfilingResult.fail(f"SIM_RESULT_CORRUPT: {exc}")
    if not payload.get("ok"):
        return ApiProfilingResult.fail(payload.get("error") or "SIM_EXECUTION_FAILED")
    output_bytes = []
    idx = 0
    while True:
        bin_file = case_path / f"output_{idx}.bin"
        if not bin_file.is_file():
            break
        output_bytes.append(bin_file.read_bytes())
        idx += 1
    view_shapes = ()
    view_pkl = case_path / "output_view_shapes.pkl"
    if view_pkl.is_file():
        view_shapes = pickle.loads(view_pkl.read_bytes())  # noqa: S301 反序列化自产 sim 用例数据
    if not output_bytes:
        return ApiProfilingResult.fail("SIM_NO_OUTPUT")
    return ApiProfilingResult(
        True,
        api_prof=payload.get("api_prof", "SIM"),
        op_prof=payload.get("op_prof", "SIM"),
        # list, matching the real-device copy_output_from_hbm().
        output_bytes=output_bytes,
        output_view_shapes=tuple(view_shapes),
        oob=payload.get("oob", "UNKNOWN"),
        deterministic_status=payload.get("deterministic_status"),
    )


def run_aclnn_sim(context, dev_id):
    """Execute one aclnn API via NPUSim record and rebuild an ApiProfilingResult.

    The whole ``context`` (TestcaseAclnn) is pickled and replayed inside the
    wrapper, which reuses ``AclOpExecutor`` / ``Phase1ParamBuilder`` — the same
    execution code path as a real device.
    """
    from ttk.core_modules.aclnn.result import ApiProfilingResult

    sw = get_global_storage()
    if not context.is_valid:
        return ApiProfilingResult.fail(context.fail_reason)

    # clear_case_dir() itself creates the directory, so case_dir() (no mkdir)
    # is enough here — avoid a redundant ensure_case_dir().
    case_path = case_writer.case_dir(sw, context.testcase_name)
    case_writer.clear_case_dir(sw, context.testcase_name)
    case_writer.dump_switches(sw, case_path)
    with open(case_path / "context.pkl", "wb") as f:
        pickle.dump(context, f, protocol=2)

    from .wrapper import write_aclnn_wrapper

    wrapper = write_aclnn_wrapper(sw, case_path)
    try:
        export_root = run_record(sw, wrapper, case_path, extra_argv=("-u", str(case_path)))
    except Exception as exc:  # noqa: BLE001 - surface as a structured fail
        logging.error("npusim record failed for %s: %s", context.testcase_name, exc)
        return ApiProfilingResult.fail(f"SIM_RECORD_FAILED: {exc}")
    if sw.sim_report:
        sim_report.maybe_generate_sim_report(sw, case_path, export_root)
    return _load_aclnn_result(case_path)
