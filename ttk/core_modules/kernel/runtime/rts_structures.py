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
Structures used by RTS
"""

# Standard Packages
import ctypes
import struct
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Any, List, Optional, Tuple, Union

import numpy

# Third-Party Packages
from . import rts_info


# For ctypes interpretation and construction of rtDevBinary_t from pointer
class RtDevBinary(ctypes.Structure):
    """
    Device Binary structure
    """

    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_char_p),
        ("length", ctypes.c_uint64),
    ]


# For profiler
class RtCommandHandleParams(ctypes.Structure):
    """
    Profiling switch structure
    """

    _fields_ = [
        ("pathLen", ctypes.c_uint32),
        ("storageLimit", ctypes.c_uint32),
        ("profDataLen", ctypes.c_uint32),
        ("path", ctypes.c_char * (rts_info.RT_PROF_PATH_LEN_MAX + 1)),
        ("profData", ctypes.c_char * (rts_info.RT_PROF_PARAM_LEN_MAX + 1)),
    ]


class RtProfCommandHandle(ctypes.Structure):
    """
    Profiling switch structure
    """

    _fields_ = [
        ("prof_switch", ctypes.c_uint64),
        ("prof_switch_hi", ctypes.c_uint64),
        ("dev_nums", ctypes.c_uint32),
        ("dev_id_list", ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM),
        ("model_id", ctypes.c_uint32),
        ("cmd_type", ctypes.c_uint32),
        ("cmd_handle_params", RtCommandHandleParams),
    ]

    def __init__(self, prof_switch: ctypes.c_uint64, cmd_type: int, c_dev_ids: Any, dev_num: int = 1):
        super().__init__()
        self.prof_switch = prof_switch
        self.prof_switch_hi = 0
        self.dev_nums = dev_num
        self.dev_id_list = c_dev_ids
        self.model_id = 3
        self.cmd_type = cmd_type
        self.cmd_handle_params = RtCommandHandleParams()


class RtProfCommandHandleV2(ctypes.Structure):
    """
    Profiling switch structure
    """

    _fields_ = [
        ("prof_switch", ctypes.c_uint64),
        ("prof_switch_hi", ctypes.c_uint64),
        ("dev_nums", ctypes.c_uint32),
        ("dev_id_list", ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM),
        ("model_id", ctypes.c_uint32),
        ("cmd_type", ctypes.c_uint32),
        ("cache_flag", ctypes.c_uint32),
        ("cmd_handle_params", RtCommandHandleParams),
    ]

    def __init__(self, prof_switch: ctypes.c_uint64, cmd_type: int, c_dev_ids: Any, dev_num: int = 1):
        super().__init__()
        self.prof_switch = prof_switch
        self.prof_switch_hi = 0
        self.dev_nums = dev_num
        self.dev_id_list = c_dev_ids
        self.model_id = 3
        self.cmd_type = cmd_type
        self.cache_flag = 0
        self.cmd_handle_params = RtCommandHandleParams()


class RtHostInputInfo(ctypes.Structure):
    """rtHostInputInfo_t"""

    _fields_ = [("addr_offset", ctypes.c_uint16), ("data_offset", ctypes.c_uint16)]

    def __init__(self, addr_offset: int, data_offset: int):
        super().__init__()
        self.addr_offset = addr_offset
        self.data_offset = data_offset


class RtHostInputInfoV2(ctypes.Structure):
    """rtHostInputInfo_t"""

    _fields_ = [("addr_offset", ctypes.c_uint32), ("data_offset", ctypes.c_uint32)]

    def __init__(self, addr_offset: int, data_offset: int):
        super().__init__()
        self.addr_offset = addr_offset
        self.data_offset = data_offset


RT_HOST_INPUT_INFO_VER = {"": RtHostInputInfo, "V2": RtHostInputInfoV2}


class RtArgsEx(ctypes.Structure):
    """rtArgsEx_t"""

    _fields_ = [
        ("args", ctypes.c_void_p),
        ("host_input", ctypes.c_void_p),
        ("args_size", ctypes.c_uint32),
        ("tiling_addr_offset", ctypes.c_uint16),
        ("tiling_data_offset", ctypes.c_uint16),
        ("host_input_num", ctypes.c_uint16),
        ("has_tiling", ctypes.c_uint8),
        ("no_need_h2d_copy", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 4),
    ]

    def __init__(self):
        super().__init__()
        self.args = None
        self.host_input = None
        self.tiling_addr_offset = 0
        self.tiling_data_offset = 0
        self.has_tiling = 0
        self.args_size = 0
        self.host_input_num = 0
        self.no_need_h2d_copy = 0
        # to hold memory allocated in `add_host_inputs`
        self._host_ipt_info_array = None
        self._c_buf = None

    def alloc_buf(self, args_size: int):
        self.args_size = args_size
        c_buf_ptr = None
        if self.args_size > 0:
            self._c_buf = (ctypes.c_ubyte * args_size)()
            c_buf_ptr = ctypes.addressof(self._c_buf)
            self.args = ctypes.c_void_p(c_buf_ptr)
        return c_buf_ptr

    def add_tiling(self, addr_offset: int, data_offset: int):
        self.tiling_addr_offset = addr_offset
        self.tiling_data_offset = data_offset
        self.has_tiling = int(data_offset > 0)

    def add_host_inputs(self, host_input_infos: list):
        self.host_input_num = len(host_input_infos)
        if self.host_input_num > 0:
            HostIptInfoArray = RtHostInputInfo * self.host_input_num
            self._host_ipt_info_array = HostIptInfoArray(*host_input_infos)
            self.host_input = ctypes.c_void_p(ctypes.addressof(self._host_ipt_info_array))


class RtArgsExV2(ctypes.Structure):
    """rtArgsEx_t"""

    _fields_ = [
        ("args", ctypes.c_void_p),
        ("host_input", ctypes.c_void_p),
        ("args_size", ctypes.c_uint32),
        ("tiling_addr_offset", ctypes.c_uint32),
        ("tiling_data_offset", ctypes.c_uint32),
        ("host_input_num", ctypes.c_uint16),
        ("has_tiling", ctypes.c_uint8),
        ("no_need_h2d_copy", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 4),
    ]

    def __init__(self):
        super().__init__()
        self.args = None
        self.host_input = None
        self.tiling_addr_offset = 0
        self.tiling_data_offset = 0
        self.has_tiling = 0
        self.args_size = 0
        self.host_input_num = 0
        self.no_need_h2d_copy = 0
        # to hold memory allocated in `add_host_inputs`
        self._host_ipt_info_array = None
        self._c_buf = None

    # All the methods below is duplicate .
    # YES !!! i tried to inherit from `RtArgsEx`, but failed: args will be nullptr.

    def alloc_buf(self, args_size: int):
        self.args_size = args_size
        c_buf_ptr = None
        if self.args_size > 0:
            self._c_buf = (ctypes.c_ubyte * args_size)()
            c_buf_ptr = ctypes.addressof(self._c_buf)
            self.args = ctypes.c_void_p(c_buf_ptr)
        return c_buf_ptr

    def add_tiling(self, addr_offset: int, data_offset: int):
        self.tiling_addr_offset = addr_offset
        self.tiling_data_offset = data_offset
        self.has_tiling = int(data_offset > 0)

    def add_host_inputs(self, host_input_infos: list):
        self.host_input_num = len(host_input_infos)
        if self.host_input_num > 0:
            HostIptInfoArray = RtHostInputInfoV2 * self.host_input_num
            self._host_ipt_info_array = HostIptInfoArray(*host_input_infos)
            self.host_input = ctypes.c_void_p(ctypes.addressof(self._host_ipt_info_array))


RT_ARGS_VER = {"": RtArgsEx, "V2": RtArgsExV2}


class RtTaskCfgInfoBranch0903(ctypes.Structure):
    """rtTaskCfgInfo_t"""

    _fields_ = [
        ("qos", ctypes.c_uint8),
        ("part_id", ctypes.c_uint8),
        ("schedule_mode", ctypes.c_uint8),  # 0:normal;1:batch;2:sync
        ("res", ctypes.c_uint8),
        ("block_dim_offset", ctypes.c_uint32),
        ("dynamic_share_mem_size", ctypes.c_uint32),
        ("dump_flag", ctypes.c_uint8),
    ]

    def __init__(self, dynamic_share_mem_size: int, schedule_mode: int = 0):
        super().__init__()
        self.schedule_mode = schedule_mode
        self.qos = 0
        self.part_id = 0
        self.res = 0
        self.block_dim_offset = 0
        self.dynamic_share_mem_size = dynamic_share_mem_size
        self.dump_flag = 0


class RtTaskCfgInfo(ctypes.Structure):
    """rtTaskCfgInfo_t"""

    _fields_ = [
        ("qos", ctypes.c_uint8),
        ("part_id", ctypes.c_uint8),
        ("schedule_mode", ctypes.c_uint8),  # 0:normal;1:batch;2:sync
        ("d2d_cross_flag", ctypes.c_bool),
        ("block_dim_offset", ctypes.c_uint32),
        ("dump_flag", ctypes.c_uint8),
        ("rev", ctypes.c_uint8 * 3),
        ("dynamic_share_mem_size", ctypes.c_uint32),
    ]

    def __init__(self, dynamic_share_mem_size: int, schedule_mode: int = 0):
        super().__init__()
        self.schedule_mode = schedule_mode
        self.qos = 0
        self.part_id = 0
        self.d2d_cross_flag = False
        self.block_dim_offset = 0
        self.dynamic_share_mem_size = dynamic_share_mem_size
        self.dump_flag = 0
        self.rev = (ctypes.c_uint8 * 3)()


class RtLaunchAttributeId(Enum):
    BLOCK_DIM = 0
    DYNAMIC_SHARE_MEM_SIZE = 1
    GROUP = 2
    QOS = 3
    PART_ID = 4
    SCHEDULE_MODE = 5
    BLOCK_DIM_OFFSET = 6
    DUMP_FLAG = 7

    def lower_name(self):
        return self.name.lower()


class RtLaunchAttributeGroup(ctypes.Structure):
    _fields_ = [("group_dim", ctypes.c_uint32), ("group_block_dim", ctypes.c_uint32)]


class RtLaunchAttributeValue(ctypes.Union):
    __slots__ = [
        "block_dim",
        "dynamic_share_mem_size",
        "group",
        "qos",
        "part_id",
        "schedule_mode",
        "block_dim_offset",
        "dump_flag",
    ]
    _fields_ = [
        ("block_dim", ctypes.c_uint32),
        ("dynamic_share_mem_size", ctypes.c_uint32),
        ("group", RtLaunchAttributeGroup),
        ("qos", ctypes.c_uint8),
        ("part_id", ctypes.c_uint8),
        ("schedule_mode", ctypes.c_uint8),
        ("block_dim_offset", ctypes.c_uint32),
        ("dump_flag", ctypes.c_uint8),
    ]


class RtLaunchAttribute(ctypes.Structure):
    """rtLaunchConfig_t"""

    _fields_ = [("id", ctypes.c_int), ("value", RtLaunchAttributeValue)]

    def __init__(self, attr_id: RtLaunchAttributeId, attr_val, **kargs):
        super().__init__()
        self.id = attr_id.value
        union_name = attr_id.lower_name()
        if attr_id == RtLaunchAttributeId.GROUP:
            self.value.group.group_dim = kargs["group_dim"]
            self.value.group.group_block_dim = kargs["group_block_dim"]
        else:
            setattr(self.value, union_name, attr_val)


class RtLaunchConfig(ctypes.Structure):
    """rtLaunchConfig_t"""

    _fields_ = [("attrs", ctypes.c_void_p), ("attr_num", ctypes.c_uint32)]


class RtLaunchArgs(ctypes.Structure):
    """rtLaunchArgs_t"""

    _fields_ = [
        ("args_info", RtArgsEx),
        ("args_addr_offset", ctypes.c_uint16),
        ("args_data_offset", ctypes.c_uint16),
        ("host_info_max_num", ctypes.c_uint16),
        ("args_host_input_offset", ctypes.c_uint16),
    ]


class RtLaunchArgsV2(ctypes.Structure):
    """rtLaunchArgs_t"""

    _fields_ = [
        ("args_info", RtArgsExV2),
        ("args_addr_offset", ctypes.c_uint16),
        ("args_data_offset", ctypes.c_uint16),
        ("host_info_max_num", ctypes.c_uint16),
        ("args_host_input_offset", ctypes.c_uint16),
    ]


class RtArgsSizeInfo(ctypes.Structure):
    MAGIC = 0xABCDEF09
    """rtArgsSizeInfo"""
    _fields_ = [("info_addr", ctypes.c_void_p), ("atomic_index", ctypes.c_uint32)]

    def __init__(self, info_addr: ctypes.c_void_p):
        super().__init__()
        atomic_index = self.MAGIC
        c_info_addr_array_p = ctypes.cast(info_addr.value, ctypes.POINTER(ctypes.c_uint64))
        c_info_addr_array_p[0] = ctypes.c_uint64(atomic_index)
        self.atomic_index = atomic_index
        self.info_addr = info_addr

    def magic_match(self):
        return self.atomic_index == self.MAGIC


class RtExceptionArgsInfo(ctypes.Structure):
    """rtExceptionArgsInfo"""

    _fields_ = [("arg_size", ctypes.c_uint32), ("arg_addr", ctypes.c_void_p), ("size_info", RtArgsSizeInfo)]


class RtDoorBellInfo(ctypes.Structure):
    """rtDoorBellInfo"""

    _fields_ = [("reserve", ctypes.c_uint8 * 6)]


class RtDoorBellExDetailInfo(ctypes.Structure):
    """rtDoorBellExDetailInfo"""

    _fields_ = [("doorbell_num", ctypes.c_uint8), ("info", RtDoorBellInfo * 4)]


class RtDirectWqeExDetailInfo(ctypes.Structure):
    """rtDirectWqeExDetailInfo"""

    _fields_ = [("reserve", ctypes.c_uint8 * 4)]


class RtFftsPlusExDetailInfo(ctypes.Structure):
    """rtFftsPlusExDetailInfo"""

    _fields_ = [("reserve", ctypes.c_uint16 * 2)]


class UnionRtExpandInfoDetail(ctypes.Union):
    _fields_ = [("ffts_plus_info", RtFftsPlusExDetailInfo)]


class UnionRtExpandInfoDetailV2(ctypes.Union):
    _fields_ = [
        ("ffts_plus_info", RtFftsPlusExDetailInfo),
        ("direct_wqe_info", RtDirectWqeExDetailInfo),
        ("door_bell_info", RtDoorBellExDetailInfo),
    ]


class RtExceptionExpandInfo(ctypes.Structure):
    """rtExceptionExpandInfo_t"""

    _fields_ = [("type", ctypes.c_uint32), ("u", UnionRtExpandInfoDetail)]


class RtExceptionExpandInfoV2(ctypes.Structure):
    """rtExceptionExpandInfo_t"""

    _fields_ = [("type", ctypes.c_uint32), ("u", UnionRtExpandInfoDetailV2)]


class RtExceptionInfo(ctypes.Structure):
    """rtExceptionInfo_t"""

    _fields_ = [
        ("reserve", ctypes.c_uint32 * 5),
        ("expand_info", RtExceptionExpandInfo),
        ("exception_args", RtExceptionArgsInfo),
    ]


class RtExceptionInfoV2(ctypes.Structure):
    """rtExceptionInfo_t"""

    _fields_ = [
        ("reserve", ctypes.c_uint32 * 5),
        ("expand_info", RtExceptionExpandInfoV2),
        ("exception_args", RtExceptionArgsInfo),
    ]


class TensorShapeInfo:
    def __init__(self, shape: Union[list, tuple]):
        self._dims = len(shape)
        self._shape = shape

    def bytes(self):
        return 8 * (1 + len(self._shape))

    def pack_uint64(self) -> list:
        dims_cnt = struct.pack("=LL", self._dims, 1)
        packed = list(struct.unpack("=Q", dims_cnt))
        packed.extend(self._shape)
        return packed


class DynamicTensorInfo:
    def __init__(self, addresses: Union[list, tuple], arrays: List[numpy.ndarray]):
        self._tensor_shapes: List[TensorShapeInfo] = [TensorShapeInfo(shape=a.shape) for a in arrays]
        self._device_address_offset = 8 + self._all_shape_bytes()
        self._tensor_addresses = addresses

    def bytes(self):
        return self._device_address_offset + 8 * len(self._tensor_addresses)

    def pack_uint64(self) -> list:
        p = [self._device_address_offset]
        for x in self._tensor_shapes:
            p.extend(x.pack_uint64())
        p.extend([addr.value if isinstance(addr, ctypes.c_void_p) else addr for addr in self._tensor_addresses])
        return p

    def _all_shape_bytes(self):
        return reduce(lambda x, y: x + y, [s.bytes() for s in self._tensor_shapes], 0)


@dataclass
class LaunchKernelArgs:
    func_or_binary_hdl: ctypes.c_void_p = None
    op_args: Union[Tuple, List] = field(default_factory=list)
    dfx_args: Union[Tuple, List] = field(default_factory=list)
    block_dim: int = 0
    tiling_key: Optional[int] = None
    # set this if tiling data has not alloc device memory yet.
    tiling_data: Optional[bytes] = None
    mix_kernel: bool = False
    schedule_mode: int = 0
    simt_share_memory_size: int = 0
    sm_desc: Optional[Union[int, ctypes.c_uint64]] = None
    # private params to hold temp/local memory
    _total_args: list = field(default_factory=list, init=False)
    _host_ipt_infos: list = field(default_factory=list, init=False)
    _host_ipt_data: list = field(default_factory=list, init=False)
    _first_host_ipt_data_offset: int = field(default=0, init=False)

    def __post_init__(self):
        self.op_args = [x for x in self.op_args if x is not None]
        self.dfx_args = [x for x in self.dfx_args if x is not None]

    def insert_ffts_addr(self, ffts_addr: Optional[Union[int, ctypes.c_void_p]]):
        if ffts_addr is None:
            return
        self.op_args.insert(0, ffts_addr)

    def construct_rt_args(self, rt_args_version: str = "V2"):
        C_POINTER_BYTES = ctypes.sizeof(ctypes.c_void_p)

        RtHostInputInfoCls = RT_HOST_INPUT_INFO_VER[rt_args_version]
        RtArgsCls = RT_ARGS_VER[rt_args_version]
        rt_args = RtArgsCls()

        host_data_offset = C_POINTER_BYTES * (len(self.op_args) + len(self.dfx_args))

        has_tiling = self.tiling_data is not None and len(self.tiling_data) > 0

        # args sequence: ffts / input / output / workspace / tiling / dfx
        if has_tiling:
            tiling_ptr_offset = len(self.op_args) * C_POINTER_BYTES  # bytes
            tiling_data_size = 32 * ((len(self.tiling_data) + 31) // 32)  # 32 bytes align
            self.op_args.append(0)  # placeholder for tiling.
            rt_args.add_tiling(addr_offset=tiling_ptr_offset, data_offset=host_data_offset + C_POINTER_BYTES)
            host_data_offset += C_POINTER_BYTES + tiling_data_size

        self._total_args.extend(self.op_args)
        self._total_args.extend(self.dfx_args)

        # handle TensorList
        for idx, arg in enumerate(self._total_args):
            if not isinstance(arg, DynamicTensorInfo):
                continue
            hi = RtHostInputInfoCls(addr_offset=idx * C_POINTER_BYTES, data_offset=host_data_offset)
            self._host_ipt_infos.append(hi)
            # update
            if self._first_host_ipt_data_offset == 0:
                self._first_host_ipt_data_offset = host_data_offset
            host_data_offset += arg.bytes()
            self._host_ipt_data.extend(arg.pack_uint64())
            self._total_args[idx] = 0  # placeholder for HostInput
        rt_args.add_host_inputs(self._host_ipt_infos)

        # build whole buffer
        self._build_rt_args_buf(rt_args, host_data_offset)

        return rt_args

    def _build_rt_args_buf(self, rt_args: Union[RtArgsEx, RtArgsExV2], args_size: int):
        c_buf_ptr = rt_args.alloc_buf(args_size)
        # # args
        if self._total_args:
            ArgArray = ctypes.c_uint64 * len(self._total_args)
            arg_array = ArgArray(
                *[
                    arg if not isinstance(arg, ctypes.c_void_p) else (0 if arg.value is None else arg.value)
                    for arg in self._total_args
                ]
            )
            ctypes.memmove(c_buf_ptr, ctypes.byref(arg_array), ctypes.sizeof(arg_array))
        # # tiling
        if rt_args.tiling_data_offset > 0:
            ctypes.memmove(c_buf_ptr + rt_args.tiling_data_offset, self.tiling_data, len(self.tiling_data))
        # # host inputs
        if self._host_ipt_data:
            HostIptArray = ctypes.c_uint64 * len(self._host_ipt_data)
            host_ipt_array = HostIptArray(*self._host_ipt_data)
            ctypes.memmove(
                c_buf_ptr + self._first_host_ipt_data_offset,
                ctypes.byref(host_ipt_array),
                ctypes.sizeof(host_ipt_array),
            )
