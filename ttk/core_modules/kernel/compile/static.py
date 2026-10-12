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

from ttk.switches import get_global_storage

from ....utilities import process_kernel_string
from ...common.process import process_context
from ..operator.op_interface import CaseNotSupportedError, OperatorInterface, OperatorNotFoundError
from ..testcase import TestcaseKernel

# Third-Party Packages
from .common import CceManualCompile, normalize_mode
from .result import KernelJsonInfo, StaticCompilationResult, compilation_result


def static_compilation(testcase: TestcaseKernel, mode_name: str = "Cst"):
    # Check mode name
    mode_name = normalize_mode(mode_name)
    process_context().notify_status(f"Init{mode_name}Compilation")
    ################################
    # Indirect static compiling parameters
    ################################
    kernel_name = process_kernel_string(f"{mode_name.lower()}_op_{testcase.testcase_name}")
    switch = getattr(get_global_storage(), f"{mode_name.lower()}_switches")
    kernel_meta = get_global_storage().kernel_meta
    os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
    result: StaticCompilationResult = compilation_result(mode_name, valid_mode=["Cst"])
    result.kernel_name = kernel_name
    ################################
    # Static Shape Compilation
    ################################
    if switch.enabled:
        # Const shape related with dynamic shape
        if mode_name == "Cst" and not testcase.is_valid:
            result.all_set(testcase.fail_reason)
            return result
        process_context().notify_status(f"{mode_name}EstablishInterface")
        interface = OperatorInterface().with_core_type(testcase.core_type)
        if switch.realtime:
            process_context().notify_status(f"On{mode_name}Compilation")
            cst_compile(interface, testcase, result, mode_name)
            # Write extra info into json
            process_context().notify_status(f"{mode_name}WriteJson")
            result.write_json(kernel_meta)
        else:
            process_context().notify_status(f"On{mode_name}ManualCompilation")
            # Find stored compile info
            cst_manual_compile(result)
        logging.debug(f"Static kernel compilation id {kernel_name} complete")
    else:
        result.all_set(f"{mode_name.upper()}_OFF")
    return result


def cst_manual_compile(result: StaticCompilationResult) -> None:
    kernel_meta = get_global_storage().kernel_meta
    compile_info_path = pathlib.Path(kernel_meta, f"{result.kernel_name}.ttk")
    if not compile_info_path.is_file():
        logging.error(
            f"Compile cache file {compile_info_path} does not exist. "
            f"Please run realtime compilation once before using non-realtime mode!"
        )
        result.all_set("MANUAL_COMPILE_FAILURE")
        return
    with open(compile_info_path, encoding="UTF-8") as json_file:
        json_data = json.loads(json_file.read())
    kernel_dir = json_data.get("kernel_dir", kernel_meta) or kernel_meta
    kernel_name = json_data.get("kernel_name", result.kernel_name) or result.kernel_name
    kernel_json_info = KernelJsonInfo.from_file(pathlib.Path(kernel_dir, f"{kernel_name}.json"))
    # Call ccec
    kernel_path = pathlib.Path(kernel_dir, kernel_name)
    cce_compile_result = CceManualCompile.compile(
        str(kernel_path),
        kernel_json_info.kernel_name,
        get_global_storage().dev_plat,
        kernel_json_info.core_type,
        kernel_json_info.task_ration,
    )
    # set compilation result
    result.standard_set(
        cce_compile_result, "MANUAL_COMPILE", json_data.get("func_params", ()), kernel_json_info, result.kernel_name
    )


def cst_compile(
    interface: OperatorInterface, testcase: TestcaseKernel, result: StaticCompilationResult, mode: str
) -> None:
    """
    Wrapper function for op_interface const operator compilation sequence
    """
    kernel_meta = get_global_storage().kernel_meta
    kernel_name = result.kernel_name
    try:
        op_name = testcase.op_name
        ipt, opt = interface.prepare_operator_parameters_const(testcase)
        compile_result = interface.compile_dynamic_shape(ipt + opt, testcase, kernel_name, mode)
        if compile_result:
            tiling_op_type, compile_info, compile_time, func_params = compile_result
            try:
                kernel_json_info = KernelJsonInfo.from_file(pathlib.Path(kernel_meta, kernel_name + ".json"))
            except FileNotFoundError:
                logging.error(f"{mode} operator {kernel_name} build artifacts not found")
                result.all_set(f"{mode.upper()}_OPERATOR_BUILD_LOST")
            else:
                logging.debug(f"Compilation of {mode} kernel {kernel_name} success")
                result.standard_set("SUCC", compile_time, func_params, kernel_json_info, kernel_name)
        else:
            logging.warning(f"{mode} operator {op_name} not found")
            result.all_set(f"{mode.upper()}_OPERATOR_NOT_FOUND")
    except TimeoutError:
        logging.exception(f"Compilation of {mode} operator {kernel_name} timeout")
        result.all_set(f"{mode.upper()}_COMPILE_TIMEOUT")
    except CaseNotSupportedError:
        logging.exception(f"Compilation of {mode} operator {kernel_name} unsupported")
        result.all_set(f"{mode.upper()}_UNSUPPORTED")
    except OperatorNotFoundError:
        logging.warning(f"{mode} operator {op_name} not found")
        result.all_set(f"{mode.upper()}_OPERATOR_NOT_FOUND")
    except Exception:
        logging.exception(f"Compilation of {mode} operator {kernel_name} failed")
        result.all_set(f"{mode.upper()}_COMPILE_FAILURE")
