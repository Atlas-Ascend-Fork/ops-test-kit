#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/python
# -*- coding: utf-8 -*-
# Standard Packages
import json
import logging
import os
import pathlib
import time

import numpy

from ttk.switches import OPTestSwitch, get_global_storage

from ....utilities import process_kernel_string
from ...common.process import process_context
from ..operator.op_interface import CaseNotSupportedError, OperatorInterface

# Third-Party Packages
from ..runtime.hccl_helper import init_hccl, is_hccl_op
from ..testcase import TestcaseKernel
from .result import DynamicCompilationResult, DynamicOpTilingResult, KernelJsonInfo, compilation_result


def _inject_hccl_group(testcase):
    if not is_hccl_op(testcase.op_name):
        return
    if testcase.attributes and "group" in testcase.attributes:
        try:
            _, group_name, _ = init_hccl()
            if group_name:
                old_group = testcase.attributes["group"]
                testcase.attributes["group"] = group_name
                logging.info(f"hccl: replaced group '{old_group}' -> '{group_name}' for op {testcase.op_name}")
        except Exception as e:
            logging.warning(f"hccl: failed to init HCCL for tiling: {e}")


from .binary_match import binary_kernel_match, parse_matched_bin_info  # noqa: E402
from .common import CceManualCompile, normalize_mode  # noqa: E402


def dynamic_compilation(testcase: TestcaseKernel, mode_name: str = "Dyn"):
    # Check mode name
    switches = get_global_storage()
    mode_name = normalize_mode(mode_name)
    process_context().notify_status(f"Init{mode_name}Compilation")
    ################################
    # Initialize local variables
    ################################
    kernel_name = process_kernel_string(f"{mode_name.lower()}_op_{testcase.testcase_name}")
    switch: OPTestSwitch = getattr(switches, f"{mode_name.lower()}_switches")
    kernel_meta = switches.kernel_meta
    os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
    result: DynamicCompilationResult = compilation_result(mode_name, valid_mode=["Dyn", "Bin"])
    result.kernel_name = kernel_name
    result.tiling_result = DynamicOpTilingResult()
    ################################
    # Compilation
    ################################
    if not switch.enabled:
        result.all_set(f"{mode_name.upper()}_OFF")
        result.tiling_result.all_set(result.compile_result)
        return result
    if not testcase.is_valid:
        result.all_set(testcase.fail_reason)
        result.tiling_result.all_set(result.compile_result)
        return result
    process_context().notify_status(f"{mode_name}EstablishInterface")
    interface = OperatorInterface().with_core_type(testcase.core_type)
    _inject_hccl_group(testcase)
    compilation_id = dynamic_characteristic_code = str(testcase.get_compilation_hash(mode_name == "Bin"))
    if not switch.realtime:
        load_compile_info_cache(dynamic_characteristic_code, result)
        if "FAILURE" in result.compile_result:
            result.tiling_result.all_set(result.compile_result)
            return result
        process_context().notify_status(f"On{mode_name}OpTiling")
        dyn_do_tiling(interface, testcase, result)
        if "FAILURE" in str(result.tiling_key):
            return result
        compilation_id = f"{compilation_id}_{result.tiling_key}"
    # Try to gain compilation access
    if process_context().acquire_semaphore(compilation_id):
        # Win compilation access
        if switch.realtime:
            process_context().notify_status(f"On{mode_name}Compilation")
            if switch.realtime == OPTestSwitch.REUSE_BINARY_RELEASE_KERNEL:
                try:
                    matched_bin_json = binary_kernel_match(interface, testcase, get_global_storage().compile_options)
                    if matched_bin_json:
                        parse_matched_bin_info(interface, testcase, matched_bin_json, result)
                    else:
                        logging.error("Binary kernel match failed.")
                        result.all_set("BINARY_MATCH_FAILURE")
                except CaseNotSupportedError:
                    logging.exception(f"Compilation of {mode_name} operator {kernel_name} unsupported")
                    result.all_set(f"{mode_name.upper()}_UNSUPPORTED")
            else:
                dyn_compile(interface, testcase, result, mode=mode_name)
            # Write extra info into json
            process_context().notify_status(f"{mode_name}WriteJson")
            result.write_json(kernel_meta, override_kernel_name=compilation_id)
        else:
            process_context().notify_status(f"On{mode_name}ManualCompilation")
            dyn_manual_compile(result)
        logging.debug(
            "{} kernel compilation id {} complete".format("Dynamic" if mode_name == "Dyn" else "Binary", compilation_id)
        )
        process_context().set_semaphore(compilation_id, result.standard_get())
    else:
        # Failed to gain access, wait
        logging.debug(
            "{} kernel compilation reuse id {}".format("Dynamic" if mode_name == "Dyn" else "Binary", compilation_id)
        )
        process_context().notify_status(f"{mode_name}WaitCompilation")
        _lock_res = process_context().get_semaphore(compilation_id)
        while _lock_res is None:
            time.sleep(2)
            _lock_res = process_context().get_semaphore(compilation_id)
        if isinstance(_lock_res, BaseException):
            result.all_set(str(_lock_res.args))
        else:
            result.standard_set(*_lock_res)
    if switch.realtime:
        # Op Tiling
        process_context().notify_status(f"On{mode_name}OpTiling")
        if result.compile_result != "SUCC":
            result.tiling_result.all_set(result.compile_result)
        else:
            dyn_do_tiling(interface, testcase, result)
    return result


def load_compile_info_cache(dynamic_characteristic_code: str, result: DynamicCompilationResult):
    kernel_meta = get_global_storage().kernel_meta
    compile_info_path = pathlib.Path(kernel_meta, f"{dynamic_characteristic_code}.ttk")
    if not compile_info_path.is_file():
        logging.error(
            f"Compile cache file {compile_info_path} does not exist."
            f"Please run realtime compilation once before using non-realtime mode!"
        )
        result.all_set("MANUAL_COMPILE_FAILURE")
    else:
        with open(compile_info_path, encoding="UTF-8") as json_file:
            json_data = json.loads(json_file.read())
        kernel_dir = json_data.get("kernel_dir") or kernel_meta
        kernel_json_info = KernelJsonInfo.from_file(pathlib.Path(kernel_dir, json_data["kernel_name"] + ".json"))
        result.standard_set(
            json_data.get("compile_info"),
            json_data.get("tiling_op_type"),
            "UNKNOWN",
            "MANUAL_COMPILE",
            json_data.get("func_params"),
            kernel_json_info,
            json_data["kernel_name"],
            json_data.get("kernel_dir"),
        )


def dyn_manual_compile(result: DynamicCompilationResult) -> None:
    switches = get_global_storage()
    kernel_meta = switches.kernel_meta
    kernel_dir = result.kernel_dir or kernel_meta
    # check fat bin directory exists
    kernel_path = pathlib.Path(kernel_dir, result.kernel_name)
    sub_kernel_path = os.path.join(kernel_path, f"{result.kernel_name}_{result.tiling_key}")
    if (
        any(os.path.exists(f"{sub_kernel_path}.{x}") for x in ("cce", "o"))
        or any(os.path.exists(f"{sub_kernel_path}_mix_aic.{x}") for x in ("cce", "o"))
        or any(os.path.exists(f"{sub_kernel_path}_mix_aiv.{x}") for x in ("cce", "o"))
    ):
        # reset kernel_json_info to sub-kernel
        result.kernel_json_info = KernelJsonInfo.from_file(f"{sub_kernel_path}.json")
        cce_compile_result = CceManualCompile.compile(
            sub_kernel_path,
            result.kernel_json_info.kernel_name,
            switches.dev_plat,
            result.kernel_json_info.core_type,
            result.kernel_json_info.task_ration,
        )
        result.compile_result = cce_compile_result
        result.kernel_name = f"{result.kernel_name}_{result.tiling_key}"
        result.kernel_dir = kernel_path
    else:
        cce_compile_result = CceManualCompile.compile(
            str(kernel_path),
            result.kernel_json_info.kernel_name,
            switches.dev_plat,
            result.kernel_json_info.core_type,
            result.kernel_json_info.task_ration,
        )
        result.compile_result = cce_compile_result


def dyn_compile(
    interface: OperatorInterface, testcase: TestcaseKernel, result: DynamicCompilationResult, mode: str
) -> None:
    """
    Wrapper function for op_interface dynamic operator compilation sequence
    """
    # noinspection PyBroadException
    kernel_meta = get_global_storage().kernel_meta
    kernel_name = result.kernel_name
    try:
        ipt, opt = interface.prepare_operator_parameters(testcase, mode=mode)
        compile_ret_struct = interface.compile_dynamic_shape(ipt + opt, testcase, kernel_name, mode)
        if compile_ret_struct:
            tiling_op_type, compile_info, compile_time, func_params = compile_ret_struct
            try:
                kernel_json_info = KernelJsonInfo.from_file(pathlib.Path(kernel_meta, kernel_name + ".json"))
            except FileNotFoundError:
                logging.error(f"{mode} operator {kernel_name} build artifacts not found")
                result.all_set(f"{mode.upper()}_OPERATOR_BUILD_LOST")
            else:
                logging.debug(
                    f"Compilation of {mode} kernel {kernel_name} success, received compile info:\n{json.dumps(compile_info)}"
                )
                result.standard_set(
                    compile_info, tiling_op_type, "SUCC", compile_time, func_params, kernel_json_info, kernel_name
                )
        else:
            logging.warning(f"{mode} operator {testcase.op_name} not found")
            result.all_set(f"{mode.upper()}_OPERATOR_NOT_FOUND")
    except TimeoutError:
        logging.exception(f"Compilation of {mode} operator {kernel_name} timeout")
        result.all_set(f"{mode.upper()}_COMPILE_TIMEOUT")
    except CaseNotSupportedError:
        logging.exception(f"Compilation of {mode} operator {kernel_name} unsupported")
        result.all_set(f"{mode.upper()}_UNSUPPORTED")
    except Exception:
        logging.exception(f"Compilation of {mode} operator {kernel_name} failed")
        result.all_set(f"{mode.upper()}_COMPILE_FAILURE")


def dyn_do_tiling(interface: OperatorInterface, testcase: TestcaseKernel, result: DynamicCompilationResult) -> None:
    """
    Wrapper function for op_interface dynamic operator op_tiling sequence
    """
    # noinspection PyBroadException
    try:
        tiling_result = interface.call_const_op_tiling(result, testcase)
    except BaseException:
        logging.exception("OPTILING_FAILURE")
        result.tiling_result.all_set("OPTILING_FAILURE")
    else:
        dyn_block_dim = int(tiling_result["block_dim"]) if tiling_result["block_dim"] > 0 else 0
        tiling_key = int(numpy.uint64(tiling_result["tiling_key"]))
        workspaces = tuple(tiling_result["workspaces"])
        local_memory_size = int(numpy.int32(tiling_result.get("local_memory_size", -1)))
        result.tiling_result.standard_set(
            dyn_block_dim,
            tiling_key,
            tiling_result["tiling_data"],
            workspaces,
            tiling_result["tiling_time"],
            local_memory_size,
        )
