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
RTS Interface
"""

# Standard Packages
import atexit
import ctypes
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy

from ....utilities import align, get_dtype_width, get_loaded_so_path, is_4bit_dtype, pack_4bits, read_file
from ....utilities.proc import msdebug_runtime_injection_enabled

# Third-Party Packages
from . import rts_info, rts_structures
from .rts_interface_base import RTSInterfaceBase

if TYPE_CHECKING:
    import ttk.utilities


# noinspection PyPep8Naming,PyUnusedLocal
@ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
def MsprofReporterCallbackPlaceholder(_moduleId, _type, _data, _len=None):
    """
    This is just a placeholder for MsprofReporterCallback
    :return:
    """
    return 0


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def rts_task_fail_callback(c_exception_info_p: ctypes.c_void_p):
    if not c_exception_info_p:
        return
    c_exception_info = ctypes.cast(c_exception_info_p, ctypes.POINTER(rts_structures.RtExceptionInfo))[0]
    c_size_info = c_exception_info.exception_args.size_info
    if not c_size_info.magic_match():
        c_exception_info = ctypes.cast(c_exception_info_p, ctypes.POINTER(rts_structures.RtExceptionInfoV2))[0]
        c_size_info = c_exception_info.exception_args.size_info
    if not c_size_info.magic_match():
        return
    c_info_addr_p = ctypes.cast(c_size_info.info_addr, ctypes.POINTER(ctypes.c_uint64))
    if not c_info_addr_p:
        return
    c_cb_invoked_flag_p = ctypes.cast(c_info_addr_p[2], ctypes.POINTER(ctypes.c_uint64))
    c_cb_invoked_flag_p[0] = ctypes.c_uint64(1)
    return


class RTSInterface(RTSInterfaceBase):
    """
    RTS Function Wrappers
    """

    OOB_SENTINEL = 2
    OOB_BYTES = 256
    DEVICE_ALIGN_SIZE = 32
    # atomic overflow will raise AIC Error. Moving-out needs block align, will not check oob.
    ATOMIC_ADD_BLACKLIST_SOC = ("Ascend910", "Ascend310P", "Ascend610", "Ascend310")

    def __init__(
        self,
        camodel: Optional[str] = None,
        rts_custom_path: str = "",
        short_soc_version: str = None,
        skip_teardown: bool = False,
    ):
        super().__init__()
        self.context = None
        self.camodel = camodel
        # Skip destroy_context()/reset() entirely. Used by the NPUSim wrapper:
        # camodel teardown (rtCtxDestroy/rtDeviceReset) can busy-spin forever on
        # some cases, and the wrapper always os._exit(0)s anyway (process exit
        # reclaims camodel resources), so teardown calls are unnecessary there.
        self.skip_teardown = skip_teardown
        # Data storage
        self.kernel_binary_storage = {}
        self.context_storage = []
        self.memory_manager = {}
        self.size_info_storage = []  # class member to hold template memory allocated.
        # module_id used for rtMalloc interface
        self.module_id = 58  # 58 - TBE
        self.short_soc_version = short_soc_version
        self.prof_switch_version = -1
        self.device_id = None
        self.rt_args_version = ""  # rts changes addrOffset from uint16 to uint32.
        if camodel:
            self.rtsdll = ctypes.CDLL(f"{rts_custom_path}libruntime_camodel.so")
            self.print_so_path()
        else:
            runtime_path = f"{rts_custom_path}libruntime.so"
            self._origin_rtsdll = ctypes.CDLL(runtime_path, mode=ctypes.RTLD_GLOBAL)
            if msdebug_runtime_injection_enabled():
                self.rtsdll = ctypes.CDLL(None)
                logging.debug("Using global RTS symbols for msopprof injection")
            else:
                self.rtsdll = self._origin_rtsdll
                self.print_so_path()
        atexit.register(self.reset)

    def __del__(self):
        self.reset()

    def clone_for_thread(self, device_id: int) -> "RTSInterface":
        """线程级克隆：共享 rtsdll，重置全部可变账本并抑制 teardown。

        多卡执行线程（kernel hccl 多 rank / aclnn 多卡路径）与主线程共享同一
        libruntime 句柄（进程内 CDLL 单例语义），但 per-thread 可变状态（设备内存/上下文/
        模板/二进制账本）必须私有——共享容器的跨线程读写只能靠 GIL 侥幸
        （与 AclInterface._SO_CACHE 并发加载事故同源）。skip_teardown 置位：
        线程侧 context/device 生命周期由调用方显式管理（kernel 侧
        MultiRankLauncher 逐轮 rtCtxDestroy；aclnn 侧 HcclResourceScope 统一
        清理），线程 __del__/reset 提前 rtDeviceReset 会 double-free 主线程句柄。
        """
        clone = RTSInterface.__new__(RTSInterface)
        clone.__dict__ = self.__dict__.copy()
        clone.skip_teardown = True
        clone.context = None
        clone.kernel_binary_storage = {}
        clone.context_storage = []
        clone.memory_manager = {}
        clone.size_info_storage = []
        clone.device_id = device_id
        return clone

    def is_model(self):
        return self.camodel is not None

    def print_so_path(self):
        """
        Print a debug message for libruntime.so path
        """
        logging.debug(f"Using libruntime.so from {get_loaded_so_path(self.rtsdll)}")

    def api_call(self, rt_api_name: str, extra_log: Optional[str], *api_args):
        if extra_log is None:
            extra_log = ""
        try:
            api = getattr(self.rtsdll, rt_api_name)
        except AttributeError:
            raise RuntimeError(f"RTS api [{rt_api_name}] is not defined.") from None
        start_time = time.time()
        api.restype = ctypes.c_uint64
        rt_error = api(*api_args)
        self.parse_error(rt_error, rt_api_name, f"{extra_log} costs {round(time.time() - start_time, 3)} seconds")

    def set_device(self, device_id: int, skip_rt_intf: bool = False) -> None:
        """
        Set device_id for current thread
        """
        if self.device_id is not None:
            self.reset(skip_rt_intf)
        if not skip_rt_intf:
            self.api_call("rtSetDevice", f"on device {device_id}", ctypes.c_int32(device_id))
        self.device_id = device_id
        self._detect_rt_args_version()

    def create_context(self, context_mode: str = "RT_CTX_NORMAL_MODE") -> ctypes.c_void_p:
        """
        Create a new context and bind it with current thread

        :param context_mode: Check runtime.rts_info for available context mode
        :return: context pointer
        """
        c_context = ctypes.c_void_p()
        c_context_p = ctypes.c_void_p(ctypes.addressof(c_context))
        self.api_call(
            "rtCtxCreate",
            None,
            c_context_p,
            ctypes.c_uint32(rts_info.rt_context_mode[context_mode]),
            ctypes.c_int32(self.device_id),
        )
        self.context = c_context
        self.context_storage.append(c_context)
        return c_context

    def destroy_context(self, c_context: ctypes.c_void_p = None) -> None:
        """
        Destroy used context
        :param c_context: context pointer
        :return:
        """
        if self.skip_teardown:
            return
        context = self.context if c_context is None else c_context
        if context not in self.context_storage:
            raise ValueError("Input context does not exist in current interface's context storage")
        self.api_call("rtCtxDestroy", None, context)
        self.context_storage.remove(context)
        if context == self.context:
            self.context = None

    def set_context(self, c_context):
        """
        Bind a context to current thread
        :param c_context: context pointer
        :return:
        """
        if c_context not in self.context_storage:
            raise ValueError("Input context does not exist in current interface's context storage")
        self.api_call("rtCtxSetCurrent", None, c_context)
        self.context = c_context

    def create_stream(self, priority=0) -> ctypes.c_void_p:
        """
        Create a new stream on current thread
        :param priority: Default priority at 0
        :return: c_stream: a void* representing the stream
        """
        c_stream = ctypes.c_void_p()
        c_stream_p = ctypes.c_void_p(ctypes.addressof(c_stream))
        self.api_call("rtStreamCreate", None, c_stream_p, priority)
        return c_stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        """
        Destroy stream
        :param stream: void* of the stream you want to destroy
        :return: None
        """
        self.api_call("rtStreamDestroy", None, stream)

    def destroy_stream_force(self, stream: ctypes.c_void_p) -> None:
        """
        Destroy stream
        :param stream: void* of the stream you want to destroy
        :return: None
        """
        if stream is None:
            logging.warning("RTS interface [rtStreamDestroyForce] does not support default stream in context.")
        self.api_call("rtStreamDestroyForce", None, stream)

    def register_device_binary_kernel(self, kernel_path: str, magic: int) -> ctypes.c_void_p:
        """
        Register device kernel on current thread
        :param kernel_path: path to the device kernel binary
        :param magic: kernel magic specified in compiled json file
        :return: rts_binary_handle: a void* representing the kernel
        """
        return self._register_kernel(kernel_path, magic, api_name="rtDevBinaryRegister")

    def unregister_device_binary_kernel(self, rts_binary_handle: ctypes.c_void_p):
        """
        Unregister device kernel
        :param rts_binary_handle: pointer to device binary
        :return:
        """
        if rts_binary_handle and rts_binary_handle.value != 0:
            self.api_call("rtDevBinaryUnRegister", None, rts_binary_handle)
            del self.kernel_binary_storage[rts_binary_handle.value]

    def register_function(
        self, rts_binary_handle: ctypes.c_void_p, kernel_name: str, func_mode: int = 0
    ) -> ctypes.c_void_p:
        """
        Register function in device kernel on current thread
        :param rts_binary_handle: pointer to device binary
        :param kernel_name: function name
        :param func_mode: function mode
        :return:
        """
        kernel_name_bytes = kernel_name.encode("UTF-8")
        c_kernel_name_p = ctypes.c_char_p(kernel_name_bytes)
        c_func_mode = ctypes.c_uint32(func_mode)
        self.api_call(
            "rtFunctionRegister",
            None,
            rts_binary_handle,
            c_kernel_name_p,
            c_kernel_name_p,
            c_kernel_name_p,
            c_func_mode,
        )
        return ctypes.cast(c_kernel_name_p, ctypes.c_void_p)

    def register_all_kernel(self, kernel_path: str, magic: int) -> ctypes.c_void_p:
        """register all kernels in the kernel. only for dynamic kernel."""
        return self._register_kernel(kernel_path, magic, api_name="rtRegisterAllKernel")

    def copy_bin_file_to_hbm(self, bin_path: str) -> ctypes.c_void_p:
        """
        Copy data of binary file into device hbm
        :param bin_path: path to the binary
        :return:
        """
        data = read_file(bin_path, 32212254720)  # 30GB
        return self.copy_bin_to_hbm(data)

    def copy_bin_to_hbm(self, _bin: bytes) -> ctypes.c_void_p:
        """
        Copy raw binaries into device hbm
        """
        if not isinstance(_bin, bytes):
            raise TypeError(f"Copy binary to hbm supports bytes only, received {str(type(_bin))}")
        aligned_size = self.get_npu_aligned_size(len(_bin))
        c_memory_p = self.malloc(
            aligned_size,
            rts_info.RTS_MEMORY_TYPE.RT_MEMORY_HBM,
            "RT_MEMORY_POLICY_HUGE_PAGE_ONLY" if aligned_size > 2048 else "RT_MEMORY_POLICY_NONE",
        )
        missing_size = aligned_size - len(_bin)
        if missing_size > 0:
            self._fill_align_bytes(missing_size, ctypes.c_void_p(c_memory_p.value + len(_bin)))
        self.memcpy(c_memory_p, aligned_size, _bin, len(_bin), "RT_MEMCPY_HOST_TO_DEVICE")
        return c_memory_p

    def copy_nparray_to_hbm(self, _nparray: numpy.ndarray, fill_oob_flag: bool = False) -> ctypes.c_void_p:
        """
        Copy numpy array into device hbm
        """
        _nparray = self._flatten_numpy_array(_nparray)
        return self._copy_bytes_to_hbm(_nparray.__array_interface__["data"][0], _nparray.nbytes, fill_oob_flag)

    def copy_torch_tensor_to_hbm(self, torch_tensor, fill_oob_flag: bool = False) -> ctypes.c_void_p:
        """
        Copy torch tensor storage data into device hbm
        """
        from torch import Tensor

        if not isinstance(torch_tensor, Tensor):
            raise TypeError(f"Copy torch tensor to hbm supports Tensor only, but received {str(type(torch_tensor))}")
        return self._copy_bytes_to_hbm(
            torch_tensor.storage().data_ptr(), torch_tensor.storage().nbytes(), fill_oob_flag
        )

    def copy_bin_to_hbm_ptr(self, _bin: bytes, hbm_ptr: ctypes.c_void_p):
        """
        Copy raw binaries into device hbm with memory allocated.
        """
        if not isinstance(_bin, bytes):
            raise TypeError(f"Copy binary to hbm supports bytes only, received {str(type(_bin))}")
        self.memcpy(hbm_ptr, len(_bin), _bin, len(_bin), "RT_MEMCPY_HOST_TO_DEVICE")

    def copy_nparray_to_hbm_ptr(self, _nparray: numpy.ndarray, hbm_ptr: ctypes.c_void_p):
        """
        Copy numpy array into device hbm with memory allocated.
        """
        _nparray = self._flatten_numpy_array(_nparray)
        self.memcpy(
            hbm_ptr,
            _nparray.nbytes,
            ctypes.c_void_p(_nparray.__array_interface__["data"][0]),
            _nparray.nbytes,
            "RT_MEMCPY_HOST_TO_DEVICE",
        )

    def get_data_from_hbm(self, c_memory_p: Union[ctypes.c_void_p, int], data_size: int):
        """
        :param c_memory_p: a void* which points to the hbm address you want to access
        :param data_size: data size in bytes
        :return: bytes
        """
        if not isinstance(c_memory_p, ctypes.c_void_p):
            c_memory_p = ctypes.c_void_p(c_memory_p)
        # noinspection PyTypeChecker
        c_buffer_type: Any = ctypes.c_char * data_size
        c_buffer: Any = c_buffer_type()
        self.memcpy(c_buffer, data_size, c_memory_p, data_size, "RT_MEMCPY_DEVICE_TO_HOST")
        return c_buffer

    def memcpy(
        self,
        c_memory_p: ctypes.c_void_p,
        memory_size: int,
        data: Union[bytes, ctypes.c_void_p],
        data_size: int,
        memcpy_kind: str = "RT_MEMCPY_HOST_TO_HOST",
    ) -> None:
        """
        RTS memcpy interface
        :param c_memory_p:
        :param memory_size:
        :param data:
        :param data_size:
        :param memcpy_kind:
        :return:
        """
        if memory_size <= 0 or data_size <= 0:
            return
        if isinstance(data, bytes) and len(data) < data_size:
            logging.warning("rtMemcpy() called with insufficient data, filled with zero!")
            data = numpy.zeros((data_size,), dtype="uint8").tobytes()
        if isinstance(data, bytes):
            c_data_p = ctypes.c_char_p(data)
        elif isinstance(data, ctypes.c_void_p):
            c_data_p = data
        else:
            raise TypeError("Runtime function memcpy supports bytes or c_voidp only!")
        c_data_size = ctypes.c_uint64(data_size)
        c_memory_size = ctypes.c_uint64(memory_size)
        self.api_call(
            "rtMemcpy", None, c_memory_p, c_memory_size, c_data_p, c_data_size, rts_info.rt_memcpy_kind[memcpy_kind]
        )

    def malloc(
        self,
        memory_size: int,
        memory_type: rts_info.RTS_MEMORY_TYPE = rts_info.RTS_MEMORY_TYPE.RT_MEMORY_DEFAULT,
        memory_policy: str = "RT_MEMORY_POLICY_NONE",
    ) -> ctypes.c_void_p:
        """
        RTS malloc interface
        :param memory_size:
        :param memory_type:
        :param memory_policy:
        :return:
        """
        if memory_size <= 0:
            logging.warning("rtMalloc() called with negative or zero size, aligned to 1!")
            memory_size = 1
        real_mem_size = memory_size + 512
        c_memory_p = ctypes.c_void_p()
        c_memory_size = ctypes.c_uint64(real_mem_size)
        self.api_call(
            "rtMalloc",
            f"trying to allocate {real_mem_size} bytes,",
            ctypes.c_void_p(ctypes.addressof(c_memory_p)),
            c_memory_size,
            memory_type.value | rts_info.rt_memory_policy[memory_policy],
            self.module_id,
        )
        ret_c_memory_p = ctypes.c_void_p(512 * ((c_memory_p.value + 512 - 1) // 512))
        self.memory_manager[ret_c_memory_p.value] = c_memory_p.value
        logging.debug(f"rtMalloc : {hex(ret_c_memory_p.value)}")
        return ret_c_memory_p

    def free(self, c_memory_p: ctypes.c_void_p):
        """
        RTS memfree interface
        :param c_memory_p:
        :return:
        """
        if c_memory_p.value not in self.memory_manager:
            raise RuntimeError(f"Invalid HBM ptr: {c_memory_p}")
        ori_c_memory_p = ctypes.c_void_p(self.memory_manager[c_memory_p.value])
        self.api_call("rtFree", None, ori_c_memory_p)
        del self.memory_manager[c_memory_p.value]

    def launch_kernel(
        self,
        stubfunc: ctypes.c_void_p,
        blockdim: int,
        args: tuple,
        s_args: int = 0,
        sm_desc: Optional[Union[int, ctypes.c_uint64]] = None,
        stream: Optional[ctypes.c_void_p] = None,
        mix_kernel: bool = False,
        simt_share_memory_size: int = 0,
    ) -> None:
        """
        Launch registered kernel function on device
        """
        launch_args = rts_structures.LaunchKernelArgs(
            func_or_binary_hdl=stubfunc,
            block_dim=blockdim,
            op_args=args,
            simt_share_memory_size=simt_share_memory_size,
            mix_kernel=mix_kernel,
            sm_desc=sm_desc,
        )
        self.launch_kernel_with_flag(launch_args, stream)

    def launch_kernel_with_handle(
        self, launch_args: rts_structures.LaunchKernelArgs, stream: Optional[ctypes.c_void_p] = None
    ):
        self._launch_kernel_v2(launch_args, stream)

    def launch_kernel_with_flag(
        self, launch_args: rts_structures.LaunchKernelArgs, stream: Optional[ctypes.c_void_p] = None
    ):
        self._launch_kernel_v2(launch_args, stream)

    def synchronize_with_stream(self, stream: Optional[ctypes.c_void_p], timeout: int = 0) -> None:
        """
        Synchronize with device, get device task status
        """
        if timeout > 0:
            c_timeout = ctypes.c_int32(timeout)
            self.api_call("rtStreamSynchronizeWithTimeout", None, stream, c_timeout)
        else:
            self.api_call("rtStreamSynchronize", None, stream)

    def reset(self, skip_rt_intf: bool = False):
        """
        Reset device
        """
        if self.skip_teardown:
            return
        if self.device_id is None:
            return
        for binary_kernel_pointer in tuple(self.kernel_binary_storage.keys()):
            self.unregister_device_binary_kernel(ctypes.c_void_p(binary_kernel_pointer))
        for c_memory_p_value in tuple(self.memory_manager.keys()):
            c_memory_p = ctypes.c_void_p(c_memory_p_value)
            self.free(c_memory_p)
        self.size_info_storage = []
        if not skip_rt_intf:
            self.api_call("rtDeviceReset", None, ctypes.c_int32(self.device_id))
        self.device_id = None

    def start_profiler(self, device_id: int):
        """
        Start profiling
        :param device_id:
        :return:
        """
        # noinspection PyBroadException
        try:
            self.api_call("rtSetMsprofReporterCallback", None, MsprofReporterCallbackPlaceholder)
        except Exception:
            logging.warning("Set MsprofReporterCallback failed")
        c_device_ids_type: Any = ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM
        c_device_ids = c_device_ids_type(device_id)
        c_prof_config = ctypes.c_uint64(0b11111)
        # noinspection PyBroadException
        if self.prof_switch_version < 0:
            try:
                self.set_prof_switch_v2(
                    rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_START, c_device_ids, c_prof_config
                )
                self.prof_switch_version = 2
            except Exception:
                try:
                    self.set_prof_switch(
                        rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_START, c_device_ids, c_prof_config
                    )
                    self.prof_switch_version = 1
                except Exception:
                    self.api_call(
                        "rtProfilerStart",
                        None,
                        c_prof_config,
                        ctypes.c_int32(1),
                        ctypes.c_void_p(ctypes.addressof(c_device_ids)),
                    )
                    self.prof_switch_version = 0
        elif self.prof_switch_version == 0:
            self.api_call(
                "rtProfilerStart",
                None,
                c_prof_config,
                ctypes.c_int32(1),
                ctypes.c_void_p(ctypes.addressof(c_device_ids)),
            )
        elif self.prof_switch_version == 1:
            self.set_prof_switch(
                rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_START, c_device_ids, c_prof_config
            )
        else:
            self.set_prof_switch_v2(
                rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_START, c_device_ids, c_prof_config
            )

    def set_prof_switch(self, prof_switch_command_type, c_device_ids, c_prof_config):
        command_type = prof_switch_command_type.value
        rt_prof_command_handle = rts_structures.RtProfCommandHandle(
            prof_switch=c_prof_config, cmd_type=command_type, c_dev_ids=c_device_ids
        )
        self.api_call(
            "rtProfSetProSwitch",
            None,
            ctypes.pointer(rt_prof_command_handle),
            ctypes.sizeof(rts_structures.RtProfCommandHandle),
        )

    def set_prof_switch_v2(self, prof_switch_command_type, c_device_ids, c_prof_config):
        command_type = prof_switch_command_type.value
        rt_prof_command_handle = rts_structures.RtProfCommandHandleV2(
            prof_switch=c_prof_config, cmd_type=command_type, c_dev_ids=c_device_ids
        )
        self.api_call(
            "rtProfSetProSwitch",
            None,
            ctypes.pointer(rt_prof_command_handle),
            ctypes.sizeof(rts_structures.RtProfCommandHandleV2),
        )

    def stop_profiler(self, device_id: int):
        """
        Start profiling
        :param device_id:
        :return:
        """
        c_device_ids_type: Any = ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM
        c_device_ids = c_device_ids_type(device_id)
        c_prof_config = ctypes.c_uint64(0b11111)
        # noinspection PyBroadException
        if self.prof_switch_version < 0:
            raise RuntimeError("start_profiler should be called before stop_profiler")
        if self.prof_switch_version == 0:
            self.api_call(
                "rtProfilerStop",
                None,
                c_prof_config,
                ctypes.c_int32(1),
                ctypes.c_void_p(ctypes.addressof(c_device_ids)),
            )
        elif self.prof_switch_version == 1:
            self.set_prof_switch(
                rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_STOP, c_device_ids, c_prof_config
            )
        else:
            self.set_prof_switch_v2(
                rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_STOP, c_device_ids, c_prof_config
            )

    def set_task_fail_callback(self):
        self.api_call("rtSetTaskFailCallback", None, rts_task_fail_callback)

    def set_exception_extend_info(self, c_cb_invoked_flag_p: ctypes.c_void_p):
        c_info_addr_array_p = (ctypes.c_uint64 * 3)(0, 1, c_cb_invoked_flag_p.value)
        c_info_addr_p = ctypes.cast(c_info_addr_array_p, ctypes.c_void_p)
        c_size_info = rts_structures.RtArgsSizeInfo(info_addr=c_info_addr_p)
        self.size_info_storage.append(c_info_addr_array_p)
        self.api_call("rtSetExceptionExtInfo", None, ctypes.pointer(c_size_info))

    def set_float_overflow_mode(self, mode: Union[rts_info.RtFloatOverflowMode, int]):
        """
        rtError_t rtSetDeviceSatMode(rtFloatOverflowMode_t floatOverflowMode)
        """
        if isinstance(mode, rts_info.RtFloatOverflowMode):
            mode = mode.value
        self.api_call("rtSetDeviceSatMode", None, ctypes.c_int(mode))

    def set_simt_stack_size(
        self, dcu_stack: Optional[int] = None, dvg_stack: Optional[int] = None, device_id: Optional[int] = None
    ):
        if dcu_stack is not None:
            self._set_simt_stack_size(rts_info.RtLimitType.SIMT_WARP_STACK_SIZE.value, dcu_stack, device_id)
        if dvg_stack is not None:
            self._set_simt_stack_size(rts_info.RtLimitType.SIMT_DVG_WARP_STACK_SIZE.value, dvg_stack, device_id)

    def get_oob_check_offset_bytes(self, array_size) -> tuple:
        aligned_byte_size = self.get_npu_aligned_size(array_size, fill_oob_flag=True)
        if self.short_soc_version in self.ATOMIC_ADD_BLACKLIST_SOC:
            oob_bytes = self.OOB_BYTES
        else:
            oob_bytes = self.DEVICE_ALIGN_SIZE + self.OOB_BYTES
        return aligned_byte_size - oob_bytes, oob_bytes

    def get_npu_aligned_size(self, array_size, fill_oob_flag: bool = False):
        align_to = self.DEVICE_ALIGN_SIZE
        return align(array_size, align_to) + align_to + (0 if not fill_oob_flag else self.OOB_BYTES)

    def warmup(self, switches: "ttk.switches.SWITCHES"):
        if switches.warmup:
            self._launch_helper_kernel(os.path.join(switches.kernel_meta, "warmup.o"), args=(0,))

    def clear_l1(self, switches: "ttk.switches.SWITCHES"):
        if switches.force_clear_l1 is None or switches.mode.is_model():
            return
        clean_val = switches.force_clear_l1
        dtype_bytes = clean_val.dtype.itemsize
        input_np_array = numpy.array([clean_val.item(0)] * (128 * 1024 // dtype_bytes), dtype=clean_val.dtype)
        self._launch_helper_kernel(os.path.join(switches.kernel_meta, "clear_l1.o"), args=(input_np_array,))

    def clear_l0(self, switches: "ttk.switches.SWITCHES"):
        if switches.force_clear_l0 is None or switches.mode.is_model():
            return
        clean_val = switches.force_clear_l0
        verify_arg = None
        from ttk.core_modules.common.device import PLATFORM_BEFORE_DAVID, get_l0_clear_tile_params

        if self.short_soc_version not in PLATFORM_BEFORE_DAVID:
            m_dim, k_dim, n_dim, _ = get_l0_clear_tile_params(switches.dev_plat)
            input_np_array = numpy.array(
                [numpy.float16(clean_val.item(0))] * (m_dim * k_dim + k_dim * n_dim), dtype=numpy.float16
            )
            # David-generation chips use an Ascend C tensor_api kernel
            # (two GM args: input and fixpipe output)
            output_np_array = numpy.zeros(m_dim * n_dim, dtype=numpy.float32)
            args = (input_np_array, output_np_array)
            # fp16 x fp16 -> fp32 mmad of all-clean_val matrices: every element == K * v^2 exactly
            v = numpy.float32(numpy.float16(clean_val.item(0)))
            verify_arg = (1, v * v * numpy.float32(k_dim))
        else:
            input_np_array = numpy.array([numpy.float16(clean_val.item(0))] * (128 * 1024 // 2), dtype=numpy.float16)
            args = (input_np_array,)
        logging.info(f"[clear_l0] launching helper kernel, clean_val={clean_val}, device={self.device_id}")
        self._launch_helper_kernel(os.path.join(switches.kernel_meta, "clear_l0.o"), args=args, verify_arg=verify_arg)
        logging.info("[clear_l0] helper kernel launched successfully")

    def clear_ub(self, switches: "ttk.switches.SWITCHES"):
        if switches.force_clear_ub is None or switches.mode.is_model():
            return
        clean_val = switches.force_clear_ub
        dtype_bytes = clean_val.dtype.itemsize
        input_np_array = numpy.array([clean_val.item(0)] * (32 // dtype_bytes), dtype=clean_val.dtype)
        self._launch_helper_kernel(os.path.join(switches.kernel_meta, "clear_ub.o"), args=(input_np_array,))

    def _launch_helper_kernel(
        self,
        kernel: str,
        args: Optional[Union[type(None), numpy.ndarray]] = None,
        verify_arg: Optional[tuple] = None,
    ):
        stream = self.create_stream()
        kernel_name = Path(kernel).stem
        kernel_json = os.path.join(os.path.dirname(kernel), f"{kernel_name}.json")
        with open(kernel_json) as f:
            json_data = json.load(f)
        magic = self.int_magic(json_data["magic"])
        kernel_main_func = json_data["kernelName"]
        registered_binary = self.register_device_binary_kernel(kernel, magic)
        stubfunc_p = self.register_function(registered_binary, kernel_main_func)
        dev_mem_addrs = []
        if args:
            for arg in args:
                if isinstance(arg, numpy.ndarray):
                    dev_mem_addrs.append(self.copy_nparray_to_hbm(arg))
                else:
                    dev_mem_addrs.append(arg)
        try:
            self.launch_kernel(
                stubfunc_p, json_data["blockDim"], tuple(dev_mem_addrs), len(dev_mem_addrs), None, stream
            )
        except Exception as e:
            raise RuntimeError(f"launch kernel {kernel_name} failed.") from e
        else:
            try:
                self.synchronize_with_stream(stream)
                if verify_arg is not None:
                    idx, expected = verify_arg
                    npu_ptr, np_array = dev_mem_addrs[idx], args[idx]
                    byte_size = int(math.ceil(np_array.size * get_dtype_width(np_array.dtype)))
                    got = numpy.frombuffer(self.get_data_from_hbm(npu_ptr, byte_size), dtype=np_array.dtype)
                    mismatch = int((got != expected).sum())
                    if mismatch == 0:
                        logging.info(
                            f"[{kernel_name}] device result verified: all {got.size} elems == {expected}, "
                            "full GM->L1->L0A/L0B->Mmad->L0C->fixpipe chain executed"
                        )
                    else:
                        logging.warning(
                            f"[{kernel_name}] device result VERIFICATION FAILED: {mismatch}/{got.size} "
                            f"elems != {expected}"
                        )
            except Exception as e:
                logging.exception(f"synchronize_with_stream [{kernel_name}] failed. {e}")
        finally:
            self.unregister_device_binary_kernel(registered_binary)
            for addr in dev_mem_addrs:
                if isinstance(addr, ctypes.c_void_p):
                    self.free(addr)
            self.destroy_stream(stream)

    def _fill_align_bytes(self, align_size, c_align_mem_start_p: ctypes.c_void_p, fill_oob_flag: bool = False):
        # fill with random bytes
        if self.short_soc_version in self.ATOMIC_ADD_BLACKLIST_SOC:
            extra_bin = numpy.zeros(shape=(align_size,), dtype=numpy.uint8)
        else:
            extra_bin = numpy.random.randint(0, 255, size=(align_size,), dtype=numpy.uint8)
            random_size = align_size - (self.DEVICE_ALIGN_SIZE if fill_oob_flag else 0)
            # fill extra memory with memory out-of-bound flag. latest soc needs to check oob exactly
            extra_bin[random_size:] = self.OOB_SENTINEL
        self.memcpy(
            c_align_mem_start_p,
            align_size,
            ctypes.c_void_p(extra_bin.__array_interface__["data"][0]),
            align_size,
            "RT_MEMCPY_HOST_TO_DEVICE",
        )

    def _copy_bytes_to_hbm(self, data_ptr, nbytes: int, fill_oob_flag) -> ctypes.c_void_p:
        aligned_size = self.get_npu_aligned_size(nbytes, fill_oob_flag)
        c_memory_p = self.malloc(
            aligned_size,
            rts_info.RTS_MEMORY_TYPE.RT_MEMORY_HBM,
            "RT_MEMORY_POLICY_HUGE_PAGE_ONLY" if aligned_size > 2048 else "RT_MEMORY_POLICY_NONE",
        )
        missing_size = aligned_size - nbytes - (0 if not fill_oob_flag else self.OOB_BYTES)
        if missing_size > 0:
            self._fill_align_bytes(missing_size, ctypes.c_void_p(c_memory_p.value + nbytes), fill_oob_flag)
        if fill_oob_flag:
            # fill extra memory with memory out-of-bound flag.
            extra_bin = numpy.array([self.OOB_SENTINEL] * self.OOB_BYTES, dtype=numpy.uint8)
            self.memcpy(
                ctypes.c_void_p(c_memory_p.value + aligned_size - self.OOB_BYTES),
                self.OOB_BYTES,
                ctypes.c_void_p(extra_bin.__array_interface__["data"][0]),
                self.OOB_BYTES,
                "RT_MEMCPY_HOST_TO_DEVICE",
            )
        self.memcpy(c_memory_p, nbytes, ctypes.c_void_p(data_ptr), nbytes, "RT_MEMCPY_HOST_TO_DEVICE")
        return c_memory_p

    def _register_kernel(self, kernel_path: str, magic: int, api_name: str):
        # Read kernel
        kernel = read_file(kernel_path)
        kernel_size = len(kernel)
        logging.debug(f"Read {kernel_size} bytes from kernel")
        # Check kernel size
        if kernel_size <= 0:
            raise OSError(f"RTS Interface received kernel of invalid size {kernel_size} with path {kernel_path}")
        c_kernel_p = ctypes.c_char_p(kernel)
        # Construct device binary structure
        rts_device_binary = rts_structures.RtDevBinary(
            data=c_kernel_p,
            length=ctypes.c_uint64(kernel_size),
            version=ctypes.c_uint32(0),
            magic=ctypes.c_uint32(magic),
        )
        # Prepare result structure
        rts_binary_handle = ctypes.c_void_p()
        self.api_call(
            api_name,
            None,
            ctypes.c_void_p(ctypes.addressof(rts_device_binary)),
            ctypes.c_void_p(ctypes.addressof(rts_binary_handle)),
        )
        self.kernel_binary_storage[rts_binary_handle.value] = kernel
        return rts_binary_handle

    def _get_overflow_addr(self) -> ctypes.c_void_p:
        # each rts.context has overflow_addr_ with 512 bytes.
        c_overflow_addr = ctypes.c_void_p()
        c_overflow_addr_p = ctypes.c_void_p(ctypes.addressof(c_overflow_addr))
        self.api_call("rtCtxGetOverflowAddr", None, c_overflow_addr_p)
        return c_overflow_addr

    def _get_c2c_ctrl_addr(self, mix_kernel: bool):
        """
        rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len)
        """
        if mix_kernel and self.short_soc_version in ("Ascend910B", "Ascend910_93"):
            ret_addr = (ctypes.c_uint64 * 8)()
            ret_len = (ctypes.c_uint32 * 8)()
            self.api_call("rtGetC2cCtrlAddr", None, ret_addr, ret_len)
            ffts_addr = ctypes.c_void_p(ret_addr[0])
            return 0 if self.is_model() and ffts_addr.value is None else ffts_addr
        return None

    def _launch_kernel_v2(self, launch_args: rts_structures.LaunchKernelArgs, stream: Optional[ctypes.c_void_p] = None):
        launch_args.insert_ffts_addr(self._get_c2c_ctrl_addr(launch_args.mix_kernel))
        rt_args = launch_args.construct_rt_args(self.rt_args_version)
        c_block_dim = ctypes.c_uint32(launch_args.block_dim)
        cfg_info = self._build_rt_task_cfg_info(launch_args.simt_share_memory_size, launch_args.schedule_mode)
        c_cfg_info_p = None if cfg_info is None else ctypes.c_void_p(ctypes.addressof(cfg_info))
        c_sm_desc_p = None if launch_args.sm_desc is None else ctypes.c_void_p(ctypes.addressof(launch_args.sm_desc))
        if launch_args.tiling_key is None:
            c_flags = ctypes.c_uint32(0)
            self.api_call(
                "rtKernelLaunchWithFlagV2",
                None,
                launch_args.func_or_binary_hdl,
                c_block_dim,
                ctypes.c_void_p(ctypes.addressof(rt_args)),
                c_sm_desc_p,
                stream,
                c_flags,
                c_cfg_info_p,
            )
        else:
            c_tiling_key = ctypes.c_uint64(launch_args.tiling_key)
            self.api_call(
                "rtKernelLaunchWithHandleV2",
                None,
                launch_args.func_or_binary_hdl,
                c_tiling_key,
                c_block_dim,
                ctypes.c_void_p(ctypes.addressof(rt_args)),
                c_sm_desc_p,
                stream,
                c_cfg_info_p,
            )

    def _detect_rt_args_version(self):
        if not hasattr(self.rtsdll, "rtCreateLaunchArgs"):
            return
        c_args_size = ctypes.c_uint32(1)
        c_host_info_total_size = ctypes.c_uint32(2)
        c_host_info_num = ctypes.c_uint32(1)
        c_args_data = ctypes.c_void_p(ctypes.addressof(c_args_size))
        c_launch_args_p = ctypes.c_void_p()
        c_launch_args_pp = ctypes.c_void_p(ctypes.addressof(c_launch_args_p))
        self.api_call(
            "rtCreateLaunchArgs",
            None,
            c_args_size,
            c_host_info_total_size,
            c_host_info_num,
            c_args_data,
            c_launch_args_pp,
        )
        px = ctypes.cast(c_launch_args_p, ctypes.POINTER(rts_structures.RtLaunchArgs))
        self.rt_args_version = "" if px[0].args_info.no_need_h2d_copy == 1 else "V2"
        self.api_call("rtDestroyLaunchArgs", None, c_launch_args_p)

    def _is_0903_branch(self):
        return not hasattr(self.rtsdll, "rtMemcpyAsyncPtrV2")

    def _build_rt_task_cfg_info(self, simt_share_memory_size: int, schedule_mode: int):
        if simt_share_memory_size <= 0 and schedule_mode == 0:
            return None
        if self._is_0903_branch():
            return rts_structures.RtTaskCfgInfoBranch0903(simt_share_memory_size, schedule_mode)
        return rts_structures.RtTaskCfgInfo(simt_share_memory_size, schedule_mode)

    def _set_simt_stack_size(self, typ: int, stack_size: int, device_id: Optional[int] = None):
        if stack_size < 0:
            return
        if self.short_soc_version not in ("Ascend950",):
            return
        if not hasattr(self.rtsdll, "rtDeviceSetLimit"):
            logging.warning("Current RTS do not support rtDeviceSetLimit.")
            return
        device_id = self.device_id if device_id is None else device_id
        self.api_call("rtDeviceSetLimit", None, ctypes.c_int(device_id), ctypes.c_int(typ), ctypes.c_uint32(stack_size))

    @staticmethod
    def prepare_tensor_list_info(addrs: Union[list, tuple], arrays: List[numpy.ndarray]):
        if len(addrs) != len(arrays):
            raise RuntimeError(
                f"Length of device addresses and numpy arrays not match: "
                f"{len(addrs)} vs {len(arrays)}."
                f"It should never happen. Maybe a BUG !!!"
            )
        return rts_structures.DynamicTensorInfo(addresses=addrs, arrays=arrays)

    @staticmethod
    def handle_stream_sync_exception(exception: str) -> str:
        if "AICORE_TRAP_EXCEPTION" in exception:
            logging.error("Reached AICORE Trap Exception")
            return "TRAP"
        if "AICORE_EXCEPTION" in exception:
            logging.error("AIC_ERROR encountered")
            return "AIC_ERROR"
        if "VECTOR_CORE_EXCEPTION" in exception:
            logging.error("VEC_ERROR encountered")
            return "VEC_ERROR"
        if "AICORE_TIMEOUT" in exception or "RT_STREAM_SYNC_TIMEOUT" in exception:
            logging.error("AIC Task TIMEOUT")
            return "TIMEOUT"
        if "HEARTBEAT" in exception:
            logging.critical("Detected critical device heartbeat lost exception, process will halt.")
            if shutil.which("msnpureport") is not None:
                os.system(f"mkdir -p errors/{os.getpid()} && cd errors/{os.getpid()} && msnpureport && cd -")  # noqa: S605
            while True:
                time.sleep(10)
                logging.critical("This testcase is killing device!!!! AND DEVICE WAS ALREADY DEAD")
        else:
            logging.exception("Unexpected RTS error during finish stage")
            return "UNKNOWN_RTS_ERROR"

    @staticmethod
    def _flatten_numpy_array(_nparray: numpy.ndarray):
        if not isinstance(_nparray, numpy.ndarray):
            raise TypeError(f"Copy numpy array to hbm supports ndarray only, but received {str(type(_nparray))}")
        if is_4bit_dtype(_nparray.dtype):
            return pack_4bits(_nparray)
        return _nparray.flatten()

    @staticmethod
    def _parse_error_code(error_type: int, error_code: int) -> str:
        if error_code >= len(rts_info.rt_raw_error_code_dict[error_type]):
            return hex(0x07000000 + error_type + error_code)
        return rts_info.rt_raw_error_code_dict[error_type][error_code]

    @staticmethod
    def parse_acl_error_code(error_code: int) -> str:
        if error_code in rts_info.rt_acl_error_code_dict:
            return rts_info.rt_acl_error_code_dict[error_code]
        return ""

    def parse_error(self, rt_error: ctypes.c_uint64, rt_api_name: str, extra_info: str) -> None:
        """
        Parse error code returned by rts interface
        :param rt_error:
        :param rt_api_name:
        :param extra_info:
        :return:
        """
        # Convert rt_error to int if received ctypes object
        if isinstance(rt_error, ctypes.c_uint64):
            rt_error = rt_error.value
        elif isinstance(rt_error, int):
            pass
        else:
            raise TypeError(f"Invalid rt_error type {str(type(rt_error))} for {str(rt_error)}")
        # RT Success
        if rt_error == 0x0:
            logging.debug(f"Runtime API Call {rt_api_name}() Success, {extra_info}")
            return

        rt_error_magic = rt_error & 0xFF000000
        if rt_error_magic != 0x07000000 and not self.is_model():
            acl_result = self.parse_acl_error_code(rt_error)
            if acl_result:
                raise RuntimeError(f"Runtime API Call {rt_api_name}() Failed: {acl_result}, {extra_info}")
            raise RuntimeError(
                f"Received invalid runtime error code for Runtime API Call {rt_api_name}(): "
                f"{hex(rt_error)}, {extra_info}"
            )
        rt_error_type = rt_error & 0x00FF0000
        if rt_error_type not in rts_info.rt_error_type_dict and not self.is_model():
            raise RuntimeError(
                f"Received invalid runtime error type for Runtime API Call {rt_api_name}(): "
                f"{hex(rt_error)}, {extra_info}"
            )
        rt_error_code = rt_error & 0x0000FFFF
        raise RuntimeError(
            f"Runtime API Call {rt_api_name}() Failed: "
            f"{self._parse_error_code(rt_error_type, rt_error_code)}, {extra_info}"
        )

    @staticmethod
    def int_magic(magic: str) -> int:
        if magic in rts_info.rt_binary_magic_dict:
            return rts_info.rt_binary_magic_dict[magic]
        raise RuntimeError(f"Unknown kernel magic: {magic}")

    @staticmethod
    def core_type_to_magic(core_type: str) -> int:
        if core_type == "AiCore":
            return rts_info.rt_binary_magic_dict["RT_DEV_BINARY_MAGIC_ELF"]
        if core_type == "VectorCore":
            return rts_info.rt_binary_magic_dict["RT_DEV_BINARY_MAGIC_ELF_AIVEC"]
        if core_type == "CubeCore":
            return rts_info.rt_binary_magic_dict["RT_DEV_BINARY_MAGIC_ELF_AICUBE"]
        if core_type == "AiCpu":
            return rts_info.rt_binary_magic_dict["RT_DEV_BINARY_MAGIC_ELF_AICPU"]
        raise RuntimeError(f"Unknown core type: {core_type}")
