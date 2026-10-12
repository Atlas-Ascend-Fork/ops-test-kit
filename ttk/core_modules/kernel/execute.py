#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Kernel 设备执行：三模式分派 + RTS 执行序列。

- execute_on_device：dyn/cst/bin 三模式分派 + 前置检查（编译结果/合法性/block_dim）
- KernelExecutor：kernel 装载 + 逐轮 launch（HBM 分配/恢复、helper kernel 清理、
  dl=1 逐轮 MD5 确定性校验）+ 输出取回与 OOB 检测
- execute_kernel：设备类型薄分派
执行参数/结果数据类见 result.py（RTSProfilingParam / KernelExecuteResult）。
"""

__all__ = ["KernelExecutor", "construct_profiling_param", "execute_kernel", "execute_on_device"]

# Standard Packages
import contextlib
import ctypes
import logging
import math
import os
import pathlib
import re
import subprocess
import threading
from typing import Dict, List, Optional, Union

import numpy

from ttk.switches import get_global_storage

from ...utilities import deep_flatten, get_dtype_width
from ..common.process import process_context
from ..common.profiler import MsProfiler, MsProfOpDfx, TtkMsProfType
from ..common.profiler.ms_profiler import read_task_durations
from .info_keeper import OpInfoKeeper
from .result import KernelExecuteResult, RTSProfilingParam
from .runtime import LaunchKernelArgs, RTSInterface, RTSInterfaceBase, rts_info
from .runtime.adx_interface import AdxInterface
from .runtime.hccl_helper import init_hccl, is_hccl_op
from .testcase import TestcaseKernel

# Third-party Packages


def construct_profiling_param(context: TestcaseKernel, mode: str, output_placeholder: bool = True) -> tuple:
    flat_input_arrays = tuple(deep_flatten(context.input_arrays or ()))
    if mode == "dynamic":
        return (
            context.dyn_compile_result,  # 0
            flat_input_arrays,  # 1
            context.output_arrays,  # 2
            context.dyn_workspace_arrays,  # 3
            context.dyn_tiling_data_bytes,  # 4
            output_placeholder,  # 5
            context.dyn_clear_atomic,  # 6
            get_global_storage().dyn_switches.prof,  # 7
            context.is_valid,  # 8
            context.fail_reason,  # 9
            context.tensor_list_distribution,  # 10
            context.testcase_name,  # 11
            mode,
        )  # 12
    if mode == "const":
        return (
            context.cst_compile_result,  # 0
            flat_input_arrays,  # 1
            context.output_arrays,  # 2
            context.cst_workspace_arrays,  # 3
            None,  # 4
            output_placeholder,  # 5
            context.cst_clear_atomic,  # 6
            get_global_storage().cst_switches.prof,  # 7
            context.is_valid,  # 8
            context.fail_reason,  # 9
            context.tensor_list_distribution,  # 10
            context.testcase_name,  # 11
            mode,
        )  # 12
    if mode == "binary":
        return (
            context.bin_compile_result,  # 0
            flat_input_arrays,  # 1
            context.output_arrays,  # 2
            context.bin_workspace_arrays,  # 3
            context.bin_tiling_data_bytes,  # 4
            output_placeholder,  # 5
            context.bin_clear_atomic,  # 6
            get_global_storage().bin_switches.prof,  # 7
            context.is_valid,  # 8
            context.fail_reason,  # 9
            context.tensor_list_distribution,  # 10
            context.testcase_name,  # 11
            mode,
        )  # 12
    raise RuntimeError(f"Unknown profiling mode {mode}")


def _get_rts_interface(device_id: int) -> RTSInterfaceBase:
    switches = get_global_storage()
    device: RTSInterface = process_context().storage.get("device", None)
    if not device or device.device_id is None:
        device = RTSInterface(switches.mode.is_model(), short_soc_version=switches.short_soc_version)
        if device.is_model():
            device.set_simt_stack_size(switches.simt_cfg.dcu_stack, switches.simt_cfg.dvg_stack, device_id)
        device.set_device(device_id)
        device.set_float_overflow_mode(switches.overflow_mode)
        if not device.is_model():
            device.set_simt_stack_size(switches.simt_cfg.dcu_stack, switches.simt_cfg.dvg_stack)
        process_context().storage["device"] = device
    return device


def execute_on_device(context: TestcaseKernel, mode: str) -> KernelExecuteResult:
    """
    Three-mode dispatch (dyn/const/binary) with pre-checks.
    """
    logging.debug(f"Entering profiling sequence with {mode} mode")
    switches = get_global_storage()
    op_name = context.op_name
    output_placeholder: bool = OpInfoKeeper().op_output_defined(op_name)
    param = RTSProfilingParam(*construct_profiling_param(context, mode, output_placeholder), op_name=op_name)
    if param.switch:
        if param.compile_result != "SUCC":
            result = KernelExecuteResult.fail(param.compile_result)
        elif not param.is_valid:
            result = KernelExecuteResult.fail(param.fail_reason)
        elif param.block_dim <= 0:
            result = KernelExecuteResult.fail("INVALID_TILING")
        else:
            param.clear_atomic_output_workspace()
            # noinspection PyBroadException
            try:
                device = _get_rts_interface(context.device_id)
                result = execute_kernel(device, param)
            except Exception as e:
                raise RuntimeError(f"Profiling Sequence of mode {mode} failed") from e
            finally:
                os.chdir(switches.root_path)
    else:
        result = KernelExecuteResult.fail("SUPPRESSED")
    return result


def _process_total_cycles(results: list):
    # kick-off UNKNOWN
    kicked_prof_task = [ele for ele in results if ele != "UNKNOWN"]
    # Check if profiling result is valid
    profiling_data_valid = kicked_prof_task and all(isinstance(ele, (int, float)) for ele in kicked_prof_task)
    logging.debug(f"HWTS Task Profiling result: {results}")
    if profiling_data_valid:
        kicked_prof_task.sort()
        return numpy.median(kicked_prof_task)
    task_data = tuple(map(str, results))
    return ",".join(task_data)


def execute_kernel(device: RTSInterfaceBase, profiling_param: RTSProfilingParam):
    if isinstance(device, RTSInterface):
        executor = KernelExecutor(device, profiling_param)
        return executor.execute()
    raise RuntimeError(f"Unrecognized device type: {type(device)}")


def _pack_launch_op_args(dvc_mem, param: RTSProfilingParam) -> list:
    """launch 参数打包：input [+output if placeholder] + workspace 顺序扁平化，
    TensorList 槽位转 DynamicTensorInfo。rank0（KernelExecutor）与其余 rank
    （MultiRankLauncher）共用同一打包序。"""
    mem_group = [dvc_mem[KernelExecutor.INPUT_MEM], dvc_mem[KernelExecutor.WS_MEM]]
    arr_group = [param.input_arrays, param.workspace_arrays]

    if param.output_placeholder:  # likely
        mem_group.insert(1, dvc_mem[KernelExecutor.OUTPUT_MEM])
        arr_group.insert(1, param.output_arrays)

    op_args = [item for sublist in mem_group for item in sublist]
    nd_arrays = [item for sublist in arr_group for item in sublist]

    for idx, arg in enumerate(op_args):
        if isinstance(arg, (list, tuple)):
            op_args[idx] = RTSInterface.prepare_tensor_list_info(arg, nd_arrays[idx])

    return op_args


class KernelExecutor:
    """RTS execution sequence: context → register → per-round launch loop → retrieve."""

    INPUT_MEM, OUTPUT_MEM, WS_MEM, DFX_MEM = 0, 1, 2, 3

    def __init__(self, device: RTSInterface, param: RTSProfilingParam):
        self._switches = get_global_storage()
        self._device = device
        self._param = param
        self._registered_binary = None
        self._registered_func = None
        self._run_time = self._switches.run_time
        self._rts_task_fail_cb_set: bool = False
        self._task_fail_cb_invoked: ctypes.c_uint64 = ctypes.c_uint64(0)
        self._prof_type = TtkMsProfType.OP if self._switches.task_profiling else TtkMsProfType.NONE
        self._prof_result_path = os.path.join(
            self._switches.root_path, "msprof", "op", self._param.testcase_name, self._param.run_mode
        )
        # IN / OUT / WS / DFX
        self._dvc_mem: List[List[Optional[Union[int, list, ctypes.c_void_p]]]] = [[], [], [], []]
        # id(np.ndarray) -> device memory
        self._nd_array_maps: Dict[int, ctypes.c_void_p] = {}
        assert self._param.kernel_json_info, "kernel_json_info is None."

    @contextlib.contextmanager
    def rts_context(self):
        reserved_mem_addr = None
        self._device.create_context()
        try:
            if self._switches.reserve_hbm:
                reserved_mem_addr = self._device.malloc(
                    self._switches.reserve_hbm * 1024 * 1024,
                    rts_info.RTS_MEMORY_TYPE.RT_MEMORY_HBM,
                    "RT_MEMORY_POLICY_HUGE_PAGE_ONLY",
                )
                logging.debug(f"HBM {self._switches.reserve_hbm}MB is reserved.")
            yield
        finally:
            if reserved_mem_addr is not None:
                self._device.free(reserved_mem_addr)
            self._device.destroy_context()
            if self._device.is_model():
                self._device.reset()

    @contextlib.contextmanager
    def register_kernel(self):
        err_ret = self._register_kernel()
        if err_ret is not None:
            yield err_ret
        else:
            try:
                yield
            finally:
                self._device.unregister_device_binary_kernel(self._registered_binary)

    @contextlib.contextmanager
    def rts_stream(self):
        if self._device.is_model():
            yield None
        else:
            stream = self._device.create_stream()
            try:
                yield stream
            finally:
                self._device.destroy_stream_force(stream)

    def execute(self):
        with self.rts_context():
            # warmup
            self._device.warmup(self._switches)
            # Kernel Registration
            with self.register_kernel() as register_result:
                if register_result is not None:
                    return register_result
                with self.rts_stream() as stm:
                    # Launch
                    actual_result_byte_arrays, profiling_data = self._rts_kernel_sequence_v2(stm)
            # Cycle Analysis
            if self._device.is_model():
                total_cycle = self._process_model_cycles()
            else:
                total_cycle = _process_total_cycles(profiling_data["TASK"])
            return KernelExecuteResult(
                total_cycle, actual_result_byte_arrays, profiling_data["OOB"], profiling_data.get("DET")
            )

    def _register_kernel(self) -> Optional[KernelExecuteResult]:
        magic = self._param.kernel_json_info.magic
        if isinstance(magic, str):
            magic = self._device.int_magic(magic)
        kernel_full_path = str(pathlib.Path(self._param.kernel_dir, f"{self._param.kernel_name}.o"))
        if not self._kernel_symbol_exists():
            logging.error(
                f"Kernel register failed. "
                f"Expected symbol {self._param.kernel_main_func_name} does not exist "
                f"in {kernel_full_path}. "
                f"This is usually caused by wrong tiling key."
            )
            return KernelExecuteResult.fail("RTS_BINARY_FAILURE")
        try:
            if self._param.kernel_json_info.is_fat_bin:
                self._registered_binary = self._device.register_all_kernel(kernel_full_path, magic)
            else:
                self._registered_binary = self._device.register_device_binary_kernel(kernel_full_path, magic)
                self._registered_func = self._device.register_function(
                    self._registered_binary, self._param.kernel_main_func_name
                )
            return None
        except Exception:
            if self._registered_binary is None:
                logging.exception(
                    f"RTS Register Binary failed, kernel object {kernel_full_path} does not exist or is invalid."
                )
                return KernelExecuteResult.fail("RTS_BINARY_FAILURE")
            logging.error(
                f"RTS Register Function failed, "
                f"Expect symbol {self._param.kernel_main_func_name} does not exist "
                f"in {kernel_full_path}. "
                f"this is usually caused by wrong tiling key"
            )
            return KernelExecuteResult.fail("RTS_FUNCTION_FAILURE")

    def _rts_kernel_sequence_v2(self, stream: Optional[ctypes.c_void_p] = None):
        # Profiling Preparation
        actual_result_byte_arrays = ["NO_OUTPUT"]
        profiling_data = {"TASK": [], "OOB": "UNKNOWN"}
        status = "OK"
        # Profiling Main Sequence
        try:
            op_dfx = MsProfOpDfx(
                self._param.kernel_name,
                self._param.block_dim,
                self._param.kernel_json_info.core_type,
                self._param.kernel_json_info.task_ration,
            )
            with MsProfiler(
                self._device.device_id,
                result_path=self._prof_result_path,
                prof_type=self._prof_type,
                start_step=0,
                is_model=(self._device.is_model()),
            ) as profiler:
                md5_list = []
                self._deterministic_md5_enabled = (
                    getattr(self._switches, "deterministic_level", 0) == 1 and not self._device.is_model()
                )
                for repeat_idx in range(self._run_time):
                    profiler.step()
                    self._device.clear_l0(self._switches)
                    self._device.clear_ub(self._switches)
                    self._device.clear_l1(self._switches)
                    # Prepare Memory on HBM
                    self._alloc_device_memory(repeat_idx)
                    # Launch Kernel
                    self._kernel_assert_setup()
                    # noinspection PyBroadException
                    try:
                        self._launch_kernel(stream)
                        profiler.report_op_dfx(op_dfx)
                    except Exception:
                        logging.exception("Kernel launch failed with an unexpected RTS error:")
                        status = "LAUNCH_FAILED"
                    else:
                        try:
                            rts_timeout = 0 if self._device.is_model() else self._switches.run_timeout
                            self._device.synchronize_with_stream(stream, timeout=rts_timeout)
                        except RuntimeError as e:
                            status = self._device.handle_stream_sync_exception(e.args[0])
                        if repeat_idx == 0:
                            self._print_dump_workspace()
                    # Collect Data
                    profiling_data["TASK"].append(status)
                    if self._deterministic_md5_enabled and status == "OK":
                        # dl=1：逐轮取回→decode→MD5（确定性校验；pre_compare
                        # 规范化在 compare 侧经 invoke_pre_compare 逐轮处理）
                        self._collect_round_md5(md5_list)
                    if repeat_idx == self._run_time - 1 and status == "OK":
                        actual_result_byte_arrays = self._copy_output_from_hbm()
                        profiling_data["OOB"] = self._check_output_out_of_bounds()
                    # Free memory
                    if not self._switches.reuse_hbm:
                        self._free_device_memory()
                    if status != "OK":
                        break
        finally:
            self._free_device_memory()
        if self._deterministic_md5_enabled:
            if len(md5_list) > 1 and len(set(md5_list)) != 1:
                logging.error(f"MD5 mismatch across {len(md5_list)} runs: {md5_list}")
                profiling_data["DET"] = "FAIL"
            elif md5_list:
                profiling_data["DET"] = "PASS"
        if self._switches.task_profiling:
            task_duration = self._process_msprof_cycles()
            if isinstance(task_duration, str):
                # check TASK execute status.
                if all([x == "OK"] for x in profiling_data["TASK"]):
                    # no msprof python tool
                    profiling_data["TASK"] = ["UNKNOWN"]
            else:
                profiling_data["TASK"] = [task_duration]
        return actual_result_byte_arrays, profiling_data

    def _alloc_device_memory(self, repeat_idx: int):
        if repeat_idx == 0 or not self._switches.reuse_hbm:
            self._alloc_hbm()
        else:
            self._reuse_hbm()

    def _get_inplace_array_ipt_ids(self) -> list:
        ipt_ids = [id(i) for i in self._param.flatten_input_arrays if isinstance(i, numpy.ndarray)]
        inplace_ids = []
        for o in self._param.flatten_output_arrays:
            if isinstance(o, numpy.ndarray) and id(o) in ipt_ids:
                inplace_ids.append(id(o))
        return inplace_ids

    def _alloc_hbm(self):
        inplace_ids = self._get_inplace_array_ipt_ids()
        self._alloc_input_hbm(inplace_ids)
        self._alloc_output_hbm(inplace_ids)
        self._alloc_workspace_hbm()
        self._alloc_dfx_hbm()

    def _alloc_input_hbm(self, inplace_ids: list):
        for ia in self._param.input_arrays:
            if ia is None:
                mem = ctypes.c_void_p(None)
            elif isinstance(ia, int):
                # some outer users wanna to set a B64 argument.
                # for example: AscendC
                mem = ia
            elif isinstance(ia, (list, tuple)):
                # tensor list
                mem = []
                for t in ia:
                    m = self._device.copy_nparray_to_hbm(t, fill_oob_flag=id(t) in inplace_ids)
                    mem.append(m)
                    self._nd_array_maps[id(t)] = m
            else:
                # tensor
                mem = self._device.copy_nparray_to_hbm(ia, fill_oob_flag=id(ia) in inplace_ids)
                self._nd_array_maps[id(ia)] = mem
            self._dvc_mem[self.INPUT_MEM].append(mem)

    def _alloc_output_hbm(self, inplace_ids: list):
        for oa in self._param.output_arrays:
            if oa is None:
                mem = ctypes.c_void_p(None)
            elif isinstance(oa, (list, tuple)):
                # tensor list
                refs = []
                for t in oa:
                    if isinstance(t, numpy.ndarray) and id(t) in inplace_ids:
                        refs.append(id(t))
                if refs:
                    # inplace dynamic tensor-list
                    if len(refs) != len(oa):
                        raise RuntimeError("Dynamic output with partial inplace inputs.")
                    # find the inplace-ed input tensor list with address.
                    mem = None
                    for ipt_idx, ipt in enumerate(self._param.input_arrays):
                        if isinstance(ipt, (list, tuple)):
                            ipts = [id(x) for x in ipt]
                            if sorted(ipts) == sorted(refs):
                                mem = self._dvc_mem[self.INPUT_MEM][ipt_idx]
                                break
                    if mem is None:
                        raise RuntimeError("Dynamic output inplace-ed dynamic input is not found.")
                else:
                    # non-inplace dynamic tensor-list
                    mem = []
                    for t in oa:
                        m = self._device.copy_nparray_to_hbm(t, fill_oob_flag=True)
                        mem.append(m)
                        self._nd_array_maps[id(t)] = m
            elif id(oa) in inplace_ids:
                # inplace tensor
                mem = self._nd_array_maps[id(oa)]
            else:
                # tensor
                mem = self._device.copy_nparray_to_hbm(oa, fill_oob_flag=True)
                self._nd_array_maps[id(oa)] = mem
            self._dvc_mem[self.OUTPUT_MEM].append(mem)

    def _alloc_workspace_hbm(self):
        for ws in self._param.workspace_arrays:
            if ws is None:
                mem = ctypes.c_void_p(None)
            else:
                mem = self._device.copy_nparray_to_hbm(ws, fill_oob_flag=True)
                self._nd_array_maps[id(ws)] = mem
            self._dvc_mem[self.WS_MEM].append(mem)

    def _alloc_dfx_hbm(self):
        for da in self._param.dfx_arrays:
            if da is None:
                mem = ctypes.c_void_p(None)
            else:
                mem = self._device.copy_nparray_to_hbm(da)
                self._nd_array_maps[id(da)] = mem
            self._dvc_mem[self.DFX_MEM].append(mem)

    def _reuse_hbm(self):
        self._reuse_input_hbm()
        # do not touch outputs to match scenario of other tools when reusing hbm.
        self._reuse_common_hbm(self._param.workspace_arrays)
        self._reuse_common_hbm(self._param.dfx_arrays)

    def _reuse_input_hbm(self):
        # only recover the inplace inputs
        inplace_ids = self._get_inplace_array_ipt_ids()
        for o in self._param.flatten_output_arrays:
            if not isinstance(o, numpy.ndarray):
                continue
            if id(o) in inplace_ids:
                mem = self._nd_array_maps[id(o)]
            else:
                continue
            self._device.copy_nparray_to_hbm_ptr(o, mem)

    def _reuse_common_hbm(self, arrays: tuple):
        for a in arrays:
            if a is None:
                continue
            mem = self._nd_array_maps[id(a)]
            self._device.copy_nparray_to_hbm_ptr(a, mem)

    def _free_device_memory(self):
        for _, dev_address in self._nd_array_maps.items():
            if not isinstance(dev_address, ctypes.c_void_p):
                continue
            if not dev_address.value:  # nullptr: None or 0
                continue
            self._device.free(dev_address)
        self._nd_array_maps.clear()
        for x_mem in self._dvc_mem:
            x_mem.clear()

    def _collect_round_md5(self, md5_list: list):
        """dl=1 逐轮确定性取回：输出 bytes 拼接 → MD5。"""
        import hashlib

        round_bytes = self._copy_output_from_hbm()
        digest = hashlib.md5(  # noqa: S324 非安全用途：确定性校验和
            b"".join(bytes(arr) if isinstance(arr, (bytes, bytearray)) else arr.tobytes() for arr in round_bytes)
        ).hexdigest()
        md5_list.append(digest)

    def _copy_output_from_hbm(self) -> list:
        byte_results = []
        for oa in self._param.flatten_output_arrays:
            if oa is None:  # nullptr
                byte_results.append(None)
                continue
            hbm_addr = self._nd_array_maps[id(oa)]
            byte_size = int(math.ceil(oa.size * get_dtype_width(oa.dtype)))
            data = self._device.get_data_from_hbm(hbm_addr, byte_size)
            byte_results.append(data)
        return byte_results

    def _check_output_out_of_bounds(self) -> str:
        oob_results = []
        total_arrays = self._param.flatten_output_arrays + self._param.workspace_arrays
        output_count = len(self._param.flatten_output_arrays)
        for idx, a in enumerate(total_arrays):
            if a is None:  # nullptr
                oob_results.append("OK")
                continue
            hbm_addr = self._nd_array_maps[id(a)]
            array_size = int(math.ceil(a.size * get_dtype_width(a.dtype)))
            oob_offset, oob_size = self._device.get_oob_check_offset_bytes(array_size)
            hbm_addr = ctypes.c_void_p(hbm_addr.value + oob_offset)
            oob_bytes = self._device.get_data_from_hbm(hbm_addr, oob_size)
            oob_sentinels = numpy.frombuffer(oob_bytes, dtype="uint8")
            if numpy.all(oob_sentinels == self._device.OOB_SENTINEL):
                oob_results.append("OK")
            else:
                oob_results.append("FAIL")
                log_idx = idx if idx < output_count else idx - output_count
                log_info = "output" if idx < output_count else "workspace"
                logging.error(
                    f"The {log_idx}th (count from 0) {log_info} "
                    f"memory out of bound !! "
                    f"All data in the list below should all be "
                    f"{self._device.OOB_SENTINEL} (uint8), "
                    f"but got:\n{oob_sentinels}"
                )
        return ",".join(oob_results)

    def _pack_launch_op_args(self):
        return _pack_launch_op_args(self._dvc_mem, self._param)

    def _launch_kernel(self, stream: Optional[ctypes.c_void_p] = None):
        if is_hccl_op(getattr(self._param, "op_name", "") or ""):
            MultiRankLauncher(self).launch(stream)
            return
        self._launch_single_rank(stream)

    def _launch_single_rank(self, stream: Optional[ctypes.c_void_p] = None):
        handle = self._registered_binary if self._param.kernel_json_info.is_fat_bin else self._registered_func
        self._launch_with(self._device, stream, handle, self._pack_launch_op_args(), self._dvc_mem[self.DFX_MEM])

    def _launch_with(self, device, stream, handle, op_args, dfx_args):
        mix_kernel = self._param.kernel_json_info.is_mix_kernel
        if self._param.kernel_json_info.is_fat_bin:
            launch_args = LaunchKernelArgs(
                func_or_binary_hdl=handle,
                op_args=op_args,
                dfx_args=dfx_args,
                block_dim=self._param.block_dim,
                tiling_key=self._param.tiling_key,
                tiling_data=self._param.tiling_data,
                mix_kernel=mix_kernel,
                schedule_mode=self._param.kernel_json_info.schedule_mode,
                simt_share_memory_size=self._param.simt_share_memory_size,
            )
            device.launch_kernel_with_handle(launch_args, stream)
        else:
            launch_args = LaunchKernelArgs(
                func_or_binary_hdl=handle,
                op_args=op_args,
                dfx_args=dfx_args,
                block_dim=self._param.block_dim,
                mix_kernel=mix_kernel,
                tiling_data=self._param.tiling_data,
                schedule_mode=self._param.kernel_json_info.schedule_mode,
                simt_share_memory_size=self._param.simt_share_memory_size,
            )
            device.launch_kernel_with_flag(launch_args, stream)

    def _process_model_cycles(self):
        return "UNKNOWN"

    def _process_msprof_cycles(self):
        # CSV 读取在 common/profiler/ms_profiler（谁采集谁解析）；
        # 首轮剔除（aclprof 异步启动的首 launch 计时不准）与 median 聚合
        # 是通路编排语义，留在通路侧
        total_cycle = "UNKNOWN"
        op_prof = read_task_durations(self._prof_result_path)
        if op_prof:
            results = op_prof[0:1]
            if len(op_prof) > 1:
                results = op_prof[1:]
            logging.debug(f"MsProf Task Profiling result: {results}")
            total_cycle = numpy.round(numpy.median(results), 3)

        return total_cycle

    def _kernel_symbol_exists(self):
        kernel_full_path = str(pathlib.Path(self._param.kernel_dir, f"{self._param.kernel_name}.o"))
        cmds = ["nm", kernel_full_path]
        try:
            out = subprocess.check_output(cmds).decode("utf-8")
        except Exception as exc:
            logging.debug(f"nm command failed for {kernel_full_path}: {exc}")
        else:
            if out:
                splits = re.split(r"[ \n]", out)
                exist = self._param.kernel_main_func_name in splits
                if not exist and self._param.kernel_json_info.is_mix_kernel:
                    exist = (
                        f"{self._param.kernel_main_func_name}_mix_aic" in splits
                        or f"{self._param.kernel_main_func_name}_mix_aiv" in splits
                    )
                return exist
        return True

    def _print_dump_workspace(self):
        if not self._dvc_mem[self.WS_MEM]:
            return
        first_workspace_addr = self._dvc_mem[self.WS_MEM][0]

        if self._param.kernel_json_info.printf_enabled() or (
            self._param.kernel_json_info.assert_enabled() and self._task_fail_cb_invoked.value == 1
        ):
            try:
                debug_buf_size = self._param.kernel_json_info.debug_buf_size
                c_buffer = self._device.get_data_from_hbm(first_workspace_addr, debug_buf_size)
                AdxInterface.print_dump_info(
                    ctypes.c_void_p(ctypes.addressof(c_buffer)), debug_buf_size, self._switches.short_soc_version
                )
            except RuntimeError:
                return

    def _kernel_assert_setup(self):
        self._task_fail_cb_invoked = ctypes.c_uint64(0)
        if not self._param.kernel_json_info.assert_enabled():
            return
        if not self._dvc_mem[self.WS_MEM]:
            return
        if not self._rts_task_fail_cb_set:
            self._device.set_task_fail_callback()
            self._rts_task_fail_cb_set = True
        # rtSetExceptionExtInfo
        with contextlib.suppress(RuntimeError):
            self._device.set_exception_extend_info(ctypes.c_void_p(ctypes.addressof(self._task_fail_cb_invoked)))


class MultiRankLauncher:
    """HCCL 融合算子多卡 RTS 原生 launch（kernel 通路闭环，不经 aclnn API）。

    rank0 复用 KernelExecutor 既有资源（device/context/binary 注册/HBM 账本）；
    其余 rank 每轮起线程自建链路：clone_for_thread → rtSetDevice → rtCtxCreate
    → binary 注册 → 同源数据本卡 HBM 拷贝，barrier 齐发后各自 rtKernelLaunch +
    rtStreamSynchronize。任一 rank 失败即 abort barrier——集合通信里晚到的失败
    会让已发射的 rank 卡死在硬件集合点，宁可全体快速失败。
    """

    BARRIER_TIMEOUT = 120

    def __init__(self, executor: KernelExecutor):
        self._executor = executor
        self._param = executor._param
        self._switches = executor._switches

    def launch(self, stream: Optional[ctypes.c_void_p] = None) -> None:
        hccl_device_ids, _group_name, rank_size = init_hccl()
        rank0_dev = self._executor._device.device_id
        extra_devices = [d for d in hccl_device_ids if d != rank0_dev] if rank_size > 1 else []
        if not extra_devices:
            self._executor._launch_single_rank(stream)
            return
        logging.info(f"hccl: multi-rank RTS launch, rank0={rank0_dev}, extra ranks={extra_devices}")
        barrier = threading.Barrier(len(extra_devices) + 1)
        errors: Dict[int, BaseException] = {}
        threads = [
            threading.Thread(target=self._run_extra_rank, args=(dev, barrier, errors), name=f"hccl-rank-{dev}")
            for dev in extra_devices
        ]
        for t in threads:
            t.start()
        try:
            barrier.wait(timeout=self.BARRIER_TIMEOUT)
        except threading.BrokenBarrierError as barrier_error:
            stuck = self._join_ranks(threads, timeout=self.BARRIER_TIMEOUT)
            raise RuntimeError(
                f"HCCL multi-rank launch aborted before launch: {self._format_errors(errors)}{stuck}"
            ) from barrier_error
        rank0_error = None
        try:
            self._executor._launch_single_rank(stream)
        except BaseException as exc:
            rank0_error = exc
        stuck = self._join_ranks(threads, timeout=self.BARRIER_TIMEOUT + max(self._switches.run_timeout, 0))
        if rank0_error is not None:
            raise RuntimeError(
                f"HCCL multi-rank launch failed (rank0={rank0_error!r}, extra={self._format_errors(errors)}{stuck})"
            ) from rank0_error
        if errors or stuck:
            raise RuntimeError(f"HCCL multi-rank launch failed: {self._format_errors(errors)}{stuck}")

    def _run_extra_rank(self, dev_id: int, barrier: threading.Barrier, errors: dict) -> None:
        device = None
        cleanups = []
        try:
            device = self._executor._device.clone_for_thread(dev_id)
            device.set_device(dev_id)
            context = device.create_context()
            # clone.skip_teardown 令 destroy_context 直接返回，context 释放须直调 rtCtxDestroy
            cleanups.append(lambda: device.api_call("rtCtxDestroy", None, context))
            stream = device.create_stream()
            cleanups.append(lambda: device.destroy_stream_force(stream))
            binary_hdl, func_hdl = self._register_rank_kernel(device)
            cleanups.append(lambda: device.unregister_device_binary_kernel(binary_hdl))
            dvc_mem = self._alloc_rank_memory(device)
            cleanups.append(lambda: self._free_rank_memory(device, dvc_mem))
            barrier.wait(timeout=self.BARRIER_TIMEOUT)
            handle = binary_hdl if self._param.kernel_json_info.is_fat_bin else func_hdl
            op_args = _pack_launch_op_args(dvc_mem, self._param)
            self._executor._launch_with(device, stream, handle, op_args, dvc_mem[KernelExecutor.DFX_MEM])
            rts_timeout = 0 if device.is_model() else self._switches.run_timeout
            device.synchronize_with_stream(stream, timeout=rts_timeout)
            logging.info(f"hccl: rank{dev_id} RTS launch + sync done")
        except threading.BrokenBarrierError:
            pass
        except BaseException as exc:
            with contextlib.suppress(Exception):
                barrier.abort()
            errors[dev_id] = exc
            logging.exception(f"hccl: rank{dev_id} failed")
        finally:
            for cleanup in reversed(cleanups):
                with contextlib.suppress(Exception):
                    cleanup()

    def _register_rank_kernel(self, device) -> tuple:
        magic = self._param.kernel_json_info.magic
        if isinstance(magic, str):
            magic = device.int_magic(magic)
        kernel_full_path = str(pathlib.Path(self._param.kernel_dir, f"{self._param.kernel_name}.o"))
        if self._param.kernel_json_info.is_fat_bin:
            return device.register_all_kernel(kernel_full_path, magic), None
        binary_hdl = device.register_device_binary_kernel(kernel_full_path, magic)
        func_hdl = device.register_function(binary_hdl, self._param.kernel_main_func_name)
        return binary_hdl, func_hdl

    def _alloc_rank_memory(self, device) -> list:
        def _copy(value, fill_oob: bool):
            if value is None:
                return ctypes.c_void_p(None)
            if isinstance(value, int):
                return value
            if isinstance(value, (list, tuple)):
                return [device.copy_nparray_to_hbm(t, fill_oob_flag=fill_oob) for t in value]
            return device.copy_nparray_to_hbm(value, fill_oob_flag=fill_oob)

        # 与 KernelExecutor._alloc_hbm 同序；inplace 别名是 rank0 语义，附加 rank 全量新配
        return [
            [_copy(ia, False) for ia in self._param.input_arrays],
            [_copy(oa, True) for oa in self._param.output_arrays],
            [_copy(ws, True) for ws in self._param.workspace_arrays],
            [_copy(da, False) for da in self._param.dfx_arrays],
        ]

    @staticmethod
    def _free_rank_memory(device, dvc_mem: list) -> None:
        for group in dvc_mem:
            for entry in group:
                for mem in entry if isinstance(entry, list) else [entry]:
                    if isinstance(mem, ctypes.c_void_p) and mem.value:
                        device.free(mem)

    @staticmethod
    def _join_ranks(threads, timeout: int) -> str:
        for t in threads:
            t.join(timeout=timeout)
        stuck = [t.name for t in threads if t.is_alive()]
        return f", stuck threads={stuck}" if stuck else ""

    @staticmethod
    def _format_errors(errors: dict) -> str:
        if not errors:
            return "{}"
        return "{" + ", ".join(f"rank{dev}: {exc!r}" for dev, exc in errors.items()) + "}"
