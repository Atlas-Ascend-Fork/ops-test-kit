#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn single-device execution: the two-phase aclnn call protocol.

``AclOpExecutor`` owns the rts context/stream lifecycle and drives the
GetWorkspaceSize -> main sequence with per-round memory hygiene and, when
deterministic checking is on, per-round output readback + MD5. Phase-1
parameters (aclTensor*/aclScalar*/C scalars) are assembled by
``Phase1ParamBuilder`` from the testcase's cached param plan.
"""

__all__ = ["AclOpExecutor", "Phase1ParamBuilder", "execute_on_device"]


# Standard Packages
import contextlib
import csv
import ctypes
import logging
import math
import os
import pathlib
import shutil
import time
from typing import Optional  # noqa: F401  # execute_on_device signature annotation

import numpy

try:
    from contextlib import nullcontext
except ImportError:
    print("Python version too low, from contextlib import nullcontext failed")

    @contextlib.contextmanager
    def NULLCXT():
        """NULL CONTEXT"""

    nullcontext = NULLCXT

# Third-Party Packages
from ttk.switches import get_global_storage

from ...utilities import (
    extract_plog_errors,
    get,
    get_dtype_width,
    nest_by_distribution,
    render_frameless_table,
)
from ...utilities.dtypes import is_4bit_dtype
from ..common.process import process_context
from ..common.profiler import MsProfiler, TtkMsProfType
from ..common.testcase.batch import resolve_deterministic_level
from .result import ApiProfilingResult
from .runtime import AclInterface
from .testcase import TestcaseAclnn


def get_aclnn_device(dev_id: int) -> AclInterface:
    device: AclInterface = process_context().storage.get("device", None)
    switches = get_global_storage()
    if not device:
        device = AclInterface(switches.short_soc_version, switches.mode.is_model())
        process_context().storage["device"] = device
    if device.device_id is None:
        device.set_device(dev_id)
        device.set_float_overflow_mode(switches.overflow_mode)
    return device


class Phase1ParamBuilder:
    """
    Builder for aclnn Phase-1 parameters.
    """

    CBaseTypeMap = {
        # bool / int8_t / int / int32_t / uint32_t / float
        # int64_t / uint64_t / double
        "bool": ctypes.c_bool,
        "int8_t": ctypes.c_int8,
        "int": ctypes.c_int,
        "int32_t": ctypes.c_int32,
        "uint32_t": ctypes.c_uint32,
        "float": ctypes.c_float,
        "int64_t": ctypes.c_int64,
        "uint64_t": ctypes.c_uint64,
        "double": ctypes.c_double,
    }

    def __init__(self, context: TestcaseAclnn, device: AclInterface):
        self._ctx = context
        self._dvc = device
        self._flatten_acl_tensor: tuple = ()

    def build(self):
        acl_tensors = self._create_acl_tensor()
        acl_scalars = self._create_acl_scalar()
        case_params = []
        plan = self._ctx.get_param_plan()
        for kind, param_name, acl_type, default in plan.param_layout:
            if kind == "tensor":
                case_params.append(acl_tensors.pop(0))
            elif kind == "scalar":
                case_params.append(acl_scalars.pop(0))
            else:
                # consider remaining as attribute.
                val = self._ctx.attributes.get(param_name, default)
                if "Array" in acl_type:
                    # aclBoolArray/aclIntArray/aclFloatArray
                    typ = acl_type[3 : acl_type.index("Array")]
                    if val is None:
                        case_params.append(None)
                    else:
                        case_params.append(self._dvc.create_acl_array(val, typ))
                elif acl_type == "aclDataType":
                    # int32
                    case_params.append(ctypes.c_int32(val))
                elif acl_type == "char*":
                    val_bytes = val.encode("UTF-8")
                    case_params.append(ctypes.c_char_p(val_bytes))
                else:
                    # Base c types:
                    # bool / int8_t / int / int32_t / uint32_t / float
                    # int64_t / uint64_t / double
                    c_type = self.CBaseTypeMap[acl_type]
                    case_params.append(c_type(val))
        return case_params

    def copy_output_from_hbm(self) -> list:
        output_byte_arrays = []
        for nested_idx in self._ctx.output_tensor_indexes:
            flat_idx = self._ctx.flat_index_of(nested_idx)
            acl_tensor: ctypes.c_void_p = self._flatten_acl_tensor[flat_idx]
            tensor = self._ctx.flat_tensors[flat_idx]
            if acl_tensor is None:
                output_byte_arrays.append(None)
            else:
                npu_ptr = self._dvc.get_device_mem_addr(acl_tensor)
                if isinstance(tensor, numpy.ndarray):
                    # 回退到 numpy base storage 以取整块字节数。隐含假设：flat
                    # 张量的 numpy 元素至多一层 view（np_storages 本体或其直接
                    # view——生成链路 _convert_np_to_numpy_view / npu_preprocess
                    # 物化副本不会产生嵌套 view），故 numpy→numpy 只回退一层即
                    # break；torch base（__array_interface__）转 numpy 后链即终止。
                    # 若未来出现多层 view，此处 byte_size 会按中间层计算（少读
                    # HBM），需改为沿 base 走到底。
                    np_storage = tensor
                    while np_storage.base is not None:
                        if isinstance(np_storage.base, numpy.ndarray):
                            np_storage = np_storage.base
                            break
                        if hasattr(np_storage.base, "__array_interface__"):
                            np_storage = numpy.asarray(np_storage.base)
                            break
                        np_storage = np_storage.base
                    # en_dtypes numpy storage is 1 byte/elem (unpacked view), but the
                    # aclTensor device storage for 4-bit dtypes is packed 2 values/byte.
                    csv_dtype = get(self._ctx.flat_tensor_dtypes, flat_idx)
                    if csv_dtype and is_4bit_dtype(str(csv_dtype)):
                        byte_size = int(math.ceil(np_storage.size / 2))
                    else:
                        byte_size = int(math.ceil(np_storage.size * get_dtype_width(np_storage.dtype)))
                else:
                    byte_size = tensor.storage().nbytes()
                output_byte_arrays.append(self._dvc.get_data_from_hbm(npu_ptr, byte_size))
        return output_byte_arrays

    def collect_output_view_shapes(self) -> list:
        output_view_shapes = []
        for nested_idx in self._ctx.output_tensor_indexes:
            flat_idx = self._ctx.flat_index_of(nested_idx)
            acl_tensor: ctypes.c_void_p = self._flatten_acl_tensor[flat_idx]
            if acl_tensor is None:
                output_view_shapes.append(None)
            else:
                output_view_shapes.append(self._dvc.get_view_shape(acl_tensor))
        return output_view_shapes

    def cleanup(self):
        self._dvc.free_all_memory()

    def _create_acl_tensor(self):
        ptr_lst = []
        # Sub-byte dtypes (fp4 etc.): the CSV view_shape counts unpacked elements
        # while np_storage holds packed uint8. aclCreateTensor needs the unpacked
        # view_shape plus the sub-byte dtype.
        csv_view_shapes = self._ctx.flat_tensor_view_shapes
        csv_dtypes = self._ctx.flat_tensor_dtypes
        from ttk.utilities.dtypes import ACL_DTYPE_TO_ID

        dynamic_slots = set(self._ctx.dyn_input_slot_indexes)
        dynamic_tensors = getattr(self._ctx, "_aclnn_dynamic_tensors", {}) or {}
        for idx, original_tt in enumerate(self._ctx.flat_tensors):
            # 动态 -1 槽：用 npu_preprocess 物化的主机副本建 aclTensor，
            # storage shape 以物化张量实际 shape 为准（upstream !281）
            tt = dynamic_tensors.get(idx, original_tt)
            if tt is None:
                ptr_lst.append(None)
            else:
                fmt = get(self._ctx.flat_tensor_formats, idx)
                ss = tuple(tt.shape) if idx in dynamic_slots else self._ctx.flat_storage_shape(idx)
                view_override = None
                dtype_override = None
                csv_dt = get(csv_dtypes, idx)
                csv_dt_name = str(csv_dt) if csv_dt else ""
                if "float4" in csv_dt_name or "fp4" in csv_dt_name:
                    csv_vs = get(csv_view_shapes, idx)
                    if csv_vs:
                        view_override = tuple(csv_vs)
                    dt_key = csv_dt_name.split(".")[-1] if "." in csv_dt_name else csv_dt_name
                    dt_norm = "float4_e2m1" if "e2m1" in dt_key else ("float4_e1m2" if "e1m2" in dt_key else dt_key)
                    if dt_norm in ACL_DTYPE_TO_ID:
                        dtype_override = ACL_DTYPE_TO_ID[dt_norm]
                ptr_lst.append(
                    self._dvc.create_acl_tensor(
                        tt, fmt, ss, view_shape_override=view_override, acl_dtype_override=dtype_override
                    )
                )
        self._flatten_acl_tensor = tuple(ptr_lst)
        dist = self._ctx.tensor_list_dist
        if not dist:
            return ptr_lst

        ptr_lst_lst = []
        as_list = nest_by_distribution(ptr_lst, dist)
        for maybe_lst in as_list:
            if not isinstance(maybe_lst, (list, tuple)):
                ptr_lst_lst.append(maybe_lst)
            else:
                ptr_lst_lst.append(self._dvc.create_acl_tensor_list(maybe_lst))
        return ptr_lst_lst

    def _create_acl_scalar(self):
        ptr_lst = []
        for s in self._ctx.flat_scalars:
            if s is None:
                ptr_lst.append(None)
            else:
                ptr_lst.append(self._dvc.create_acl_scalar(s))
        if not self._ctx.scalar_list_dist:
            return ptr_lst

        ptr_lst_lst = []
        as_list = nest_by_distribution(ptr_lst, self._ctx.scalar_list_dist)
        for maybe_lst in as_list:
            if not isinstance(maybe_lst, (list, tuple)):
                ptr_lst_lst.append(maybe_lst)
            else:
                ptr_lst_lst.append(self._dvc.create_acl_scalar_list(maybe_lst))
        return ptr_lst_lst


class AclOpExecutor:
    def __init__(self, context: TestcaseAclnn, device: AclInterface):
        self._switches = get_global_storage()
        self._deterministic_level = resolve_deterministic_level(self._switches, context)
        self._phase1_param_builder = Phase1ParamBuilder(context, device)
        self._run_time = self._switches.run_time
        self._ctx = context
        self._dvc = device
        self._prof_type = TtkMsProfType.API if self._switches.task_profiling else TtkMsProfType.NONE
        self._prof_result_path = os.path.join(self._switches.root_path, "msprof", "op_api", self._ctx.testcase_name)

    @contextlib.contextmanager
    def rts_context(self):
        self._dvc.create_context()
        if self._deterministic_level >= 1:
            self._dvc.set_deterministic_level(self._deterministic_level)
        self._apply_simt_stack_config()
        try:
            yield
        finally:
            self._dvc.destroy_context()
            if self._dvc.is_model():
                self._dvc.reset()

    def _apply_simt_stack_config(self):
        """--simt-stack-dcu/dvg：rtLimit 栈配置（--simt-* 在 aclnn 的消费点）。

        kernel/execute.py 同款语义：model（npusim）形态传 device_id（set_device
        前语义在 rts_interface 内部处理），真机在 context 建立后设置。未配置
        （None/None）时不触达 rtLimit——避免每用例无谓调用。"""
        simt_cfg = getattr(self._switches, "simt_cfg", None)
        if simt_cfg is None:
            return
        if simt_cfg.dcu_stack is None and simt_cfg.dvg_stack is None:
            return
        device_id = self._dvc.device_id if self._dvc.is_model() else None
        self._dvc.set_simt_stack_size(simt_cfg.dcu_stack, simt_cfg.dvg_stack, device_id)

    @contextlib.contextmanager
    def rts_stream(self):
        if self._dvc.is_model():
            yield None
        else:
            stream = self._dvc.create_stream()
            try:
                yield stream
            finally:
                self._dvc.destroy_stream_force(stream)

    def do(self, stream: ctypes.c_void_p = None, skip_context_creation: bool = False):
        if skip_context_creation and stream is not None:
            if self._deterministic_level >= 1:
                self._dvc.set_deterministic_level(self._deterministic_level)
            output_byte_arrays, output_view_shapes, success, _, npu_memory, execute_error = self._acl_sequence(
                stream, skip_profiler=True
            )
            return ApiProfilingResult(
                success,
                "UNKNOWN",
                "UNKNOWN",
                output_byte_arrays,
                output_view_shapes,
                npu_memory=npu_memory,
                execute_error=execute_error,
            )
        with self.rts_context():
            self._dvc.warmup(self._switches)
            with self.rts_stream() as stm:
                output_byte_arrays, output_view_shapes, success, det_status, npu_memory, execute_error = (
                    self._acl_sequence(stm)
                )
            # Cycle Analysis
            if self._dvc.is_model():
                api_prof = "UNKNOWN"
                op_prof = "TOTAL_CYCLE_TODO"
            else:
                api_prof, op_prof = self._process_total_cycles()
            return ApiProfilingResult(
                success,
                api_prof,
                op_prof,
                output_byte_arrays,
                output_view_shapes,
                deterministic_status=det_status,
                npu_memory=npu_memory,
                execute_error=execute_error,
            )

    @staticmethod
    def _extract_csv_cell(filename, extract_cols, cmp=None) -> list:
        results = []
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if cmp is None or cmp(row):
                    data_dict = {}
                    for _idx, c in enumerate(extract_cols):
                        try:
                            data_dict.update({c: float(row[c])})
                        except ValueError:
                            data_dict.update({c: row[c]})
                    results.append(data_dict)
        return results

    def _acl_sequence(self, stream: Optional[ctypes.c_void_p] = None, skip_profiler: bool = False):
        output_byte_arrays = ["NO_OUTPUT"] * len(self._ctx.output_tensor_indexes)
        output_view_shapes = ["NO_OUTPUT"] * len(self._ctx.output_tensor_indexes)
        status = "NOK"
        deterministic = self._deterministic_level > 0
        md5_list = []
        det_status = None
        npu_memory = None
        execute_error = None
        try:
            prof_start_at = 1 if self._run_time > 1 else 0
            profiler_ctx = (
                MsProfiler(
                    self._dvc.device_id,
                    result_path=self._prof_result_path,
                    prof_type=self._prof_type,
                    start_step=prof_start_at,
                    is_model=self._dvc.is_model(),
                )
                if not skip_profiler
                else nullcontext()
            )
            with profiler_ctx as profiler:
                for repeat_idx in range(self._run_time):
                    if not skip_profiler and profiler:
                        profiler.step()
                    self._dvc.clear_l0(self._switches)
                    self._dvc.clear_ub(self._switches)
                    self._dvc.clear_l1(self._switches)
                    logging.debug(
                        f"[AclOpExecutor dev={self._dvc.device_id}] building phase1 params, "
                        f"group={self._ctx.attributes.get('group', 'N/A')}"
                    )
                    phase1_params = self._phase1_param_builder.build()
                    logging.debug(f"[AclOpExecutor dev={self._dvc.device_id}] calling acl_get_workspace")
                    try:
                        # call phase 1 interface
                        workspace_size, c_executor = self._dvc.acl_get_workspace(self._ctx.api_name, phase1_params)
                        npu_memory = workspace_size
                        # call phase 2 interface
                        status = self._dvc.acl_execute(self._ctx.api_name, workspace_size, c_executor, stream)
                    except Exception as e:
                        time.sleep(0.5)
                        plog_errors = extract_plog_errors(max_lines=10)
                        if plog_errors:
                            execute_error = os.linesep.join(plog_errors)
                            logging.error(
                                f"aclnn interface {self._ctx.api_name} execute failed: \n"
                                f"***************************************************************************\n"
                                f"{os.linesep.join(plog_errors)}\n"
                                f"***************************************************************************"
                            )
                        else:
                            error_detail = str(e)
                            execute_error = error_detail
                            logging.exception(f"aclnn interface {self._ctx.api_name} execute failed:\n{error_detail}")
                        status = "ACLNN_EXECUTE_FAILED"

                    if status == "OK":
                        if deterministic > 0:
                            out_bytes = self._phase1_param_builder.copy_output_from_hbm()
                            import hashlib

                            md5_list.append(
                                hashlib.md5(  # noqa: S324  # 非安全用途：确定性校验和
                                    b"".join(
                                        bytes(b) if isinstance(b, (bytearray, memoryview)) else b for b in out_bytes
                                    )
                                ).hexdigest()
                            )
                            if repeat_idx == self._run_time - 1:
                                output_byte_arrays = out_bytes
                                output_view_shapes = self._phase1_param_builder.collect_output_view_shapes()
                        elif repeat_idx == self._run_time - 1:
                            # copy output (tensor storage data) from device
                            output_byte_arrays = self._phase1_param_builder.copy_output_from_hbm()
                            output_view_shapes = self._phase1_param_builder.collect_output_view_shapes()
                    elif execute_error is None:
                        # phase-2 failure signaled via status string, no exception raised here
                        time.sleep(0.5)
                        plog_errors = extract_plog_errors(max_lines=10)
                        execute_error = os.linesep.join(plog_errors) if plog_errors else status
                    self._dvc.free_all_memory()
                    if status != "OK":
                        break
        finally:
            self._dvc.free_all_memory()

        if deterministic > 0 and len(md5_list) > 1:
            if len(set(md5_list)) != 1:
                logging.error(f"[{self._ctx.testcase_name}] MD5 mismatch across {len(md5_list)} runs: {md5_list}")
                status = "DETERMINISTIC_MD5_MISMATCH"
                det_status = "FAIL"
            else:
                logging.info(f"[{self._ctx.testcase_name}] MD5 consistent across {len(md5_list)} runs: {md5_list[0]}")
                det_status = "PASS"
        elif deterministic > 0 and len(md5_list) == 1:
            det_status = "PASS"

        return output_byte_arrays, output_view_shapes, status == "OK", det_status, npu_memory, execute_error

    def _process_total_cycles(self):
        """
        analysis profiling data & print to stdout.
        """
        api_prof = "UNKNOWN"
        op_prof = "UNKNOWN"
        prof_result_path = pathlib.Path(self._prof_result_path)
        if not prof_result_path.is_dir():
            return api_prof, op_prof
        csv_files = list(prof_result_path.glob("**/*.csv"))
        if not csv_files:
            return api_prof, op_prof
        for item in pathlib.Path(os.path.dirname(csv_files[0])).iterdir():
            if item.is_file():
                with contextlib.suppress(shutil.SameFileError):
                    shutil.copy(item, prof_result_path.joinpath(item.name))
        for item in prof_result_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.name.startswith("api_statistic_"):
                api_prof = self._extract_csv_cell(
                    item,
                    ("API Name", "Time(us)", "Count", "Avg(us)", "Min(us)", "Max(us)"),
                    cmp=lambda row: row["Level"] == "acl",
                )
            elif item.name.startswith("op_statistic_"):
                op_prof = self._extract_csv_cell(
                    item,
                    ("OP Type", "Total Time(us)", "Count", "Avg Time(us)", "Min Time(us)", "Max Time(us)", "Core Type"),
                )
                for p in op_prof:
                    p["OP Type"] = p["OP Type"] + ("_AiCpu" if "cpu" in p["Core Type"].lower() else "_AiCore")
                    del p["Core Type"]
        # print
        lines = [["Name", "Total/us", "Avg/us", "Min/us", "Max/us", "# of Calls"]]
        if not isinstance(api_prof, str):
            lines.extend(
                [
                    [
                        item["API Name"],
                        "{:.2f}".format(item["Time(us)"]),
                        "{:.2f}".format(item["Avg(us)"]),
                        "{:.2f}".format(item["Min(us)"]),
                        "{:.2f}".format(item["Max(us)"]),
                        int(item["Count"]),
                    ]
                    for item in api_prof
                ]
            )
        if not isinstance(op_prof, str):
            lines.extend(
                [
                    [
                        item["OP Type"],
                        "{:.2f}".format(item["Total Time(us)"]),
                        "{:.2f}".format(item["Avg Time(us)"]),
                        "{:.2f}".format(item["Min Time(us)"]),
                        "{:.2f}".format(item["Max Time(us)"]),
                        int(item["Count"]),
                    ]
                    for item in op_prof
                ]
            )
        logging.info(render_frameless_table(lines))
        return api_prof, op_prof


def execute_on_device(context: TestcaseAclnn, dev_id: int) -> ApiProfilingResult:
    """
    Profiling wrapper
    """
    logging.debug("Entering op api profiling sequence")
    switches = get_global_storage()
    if not context.is_valid:
        return ApiProfilingResult.fail(context.fail_reason)
    # noinspection PyBroadException
    try:
        device = get_aclnn_device(dev_id)
        return AclOpExecutor(context, device).do()
    except Exception as e:
        raise RuntimeError(f"Profiling Sequence of mode {switches.mode} failed") from e
    finally:
        os.chdir(switches.root_path)
