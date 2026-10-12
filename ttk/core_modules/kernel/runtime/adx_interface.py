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
ADump Interface, refer to 'dump_printf.cpp'
"""

# Standard Packages
import ctypes
import logging
import re
from functools import partial, reduce
from typing import Type, Union

import numpy

from ....utilities import (
    ACL_ID_TO_DTYPE,
    Singleton,
    get_dtype_width,
    resolve_numpy_dtypes,
)
from ...common.device import PLATFORM_BEFORE_DAVID

# Third-Party Packages
from .adx_structure import (
    AdxBlockInfo,
    AdxDumpInfoHead,
    AdxDumpMessageHead,
    AdxDumpMeta,
    AdxDumpShapeMessageHead,
    AdxDumpTimestampHead,
    AdxSimtDumpMeta,
    AscendDumpType,
)


def __format_decimal(fmt_str: str, fmt_start: int, fmt_stop: int, c_param_p: ctypes.c_void_p, c_type):
    num = ctypes.cast(c_param_p, ctypes.POINTER(c_type))[0]
    return fmt_str[:fmt_start] + str(num) + fmt_str[fmt_stop:]


def __format_hex(fmt_str: str, fmt_start: int, fmt_stop: int, c_param_p: ctypes.c_void_p, c_type):
    num = ctypes.cast(c_param_p, ctypes.POINTER(c_type))[0]
    return fmt_str[:fmt_start] + f"0x{num:x}" + fmt_str[fmt_stop:]


def __format_str(fmt_str: str, fmt_start: int, fmt_stop: int, c_param_p: ctypes.c_void_p):
    str_offset = ctypes.cast(c_param_p, ctypes.POINTER(ctypes.c_uint64))[0]
    c_str_p = ctypes.cast(ctypes.c_void_p(c_param_p.value + str_offset), ctypes.c_char_p)
    return fmt_str[:fmt_start] + c_str_p.value.decode() + fmt_str[fmt_stop:]


ADX_FORMATTERS = {
    "%d": partial(__format_decimal, c_type=ctypes.c_int64),
    "%ld": partial(__format_decimal, c_type=ctypes.c_int64),
    "%lld": partial(__format_decimal, c_type=ctypes.c_int64),
    "%i": partial(__format_decimal, c_type=ctypes.c_int64),
    "%li": partial(__format_decimal, c_type=ctypes.c_int64),
    "%lli": partial(__format_decimal, c_type=ctypes.c_int64),
    "%u": partial(__format_decimal, c_type=ctypes.c_uint64),
    "%lu": partial(__format_decimal, c_type=ctypes.c_uint64),
    "%llu": partial(__format_decimal, c_type=ctypes.c_uint64),
    "%f": partial(__format_decimal, c_type=ctypes.c_float),
    "%F": partial(__format_decimal, c_type=ctypes.c_float),
    "%x": partial(__format_hex, c_type=ctypes.c_int64),
    "%lx": partial(__format_hex, c_type=ctypes.c_int64),
    "%llx": partial(__format_hex, c_type=ctypes.c_int64),
    "%X": partial(__format_hex, c_type=ctypes.c_int64),
    "%lX": partial(__format_hex, c_type=ctypes.c_int64),
    "%llX": partial(__format_hex, c_type=ctypes.c_int64),
    "%p": partial(__format_hex, c_type=ctypes.c_uint64),
    "%s": __format_str,
}


ADX_TENSOR_POSITIONS = {
    0: "GM",
    1: "UB",
    2: "L1",
    3: "L0A",
    4: "L0B",
    5: "L0C",
    6: "BIAS",
    7: "FIXBUF",
}


class AdxInterface(metaclass=Singleton):
    """
    Class Interface for ADump
    """

    ADX_PRINT_ARG_LEN = 8
    ADX_PRINT_MAGIC = 0x5AA5BCCD
    ADX_PRINT_NO_ENOUGH_SPACE_FLAG = 7
    ADX_MAX_STR_LEN = 1024 * 1024
    ADX_ASSERT_LEN = 1024
    ADX_SIMT_PRINT_LEN = 2 * 1024
    ADX_SIMT_MAX_THREAD_NUM = 2048

    def __init__(self):
        pass

    @classmethod
    def print_dump_info(cls, c_ws_data_p: ctypes.c_void_p, debug_buf_size: int, short_soc_version: str):
        mem_bound = ctypes.c_void_p(c_ws_data_p.value + debug_buf_size)
        block_data_len = cls._get_block_data_len(c_ws_data_p, debug_buf_size)
        simd_dump_cores = cls._get_max_dump_core_num(short_soc_version)
        for block_idx in range(simd_dump_cores):
            offset = block_data_len * block_idx
            c_ws_ptr = ctypes.c_void_p(c_ws_data_p.value + offset)
            cls._print_block_info(c_ws_ptr, block_data_len, mem_bound, block_idx, AdxDumpMeta, short_soc_version)

        if not cls._has_simt_dump_info(debug_buf_size, short_soc_version):
            return

        simt_addr_offset = simd_dump_cores * block_data_len
        c_simt_data_p = ctypes.c_void_p(c_ws_data_p.value + simt_addr_offset)
        simt_block_info = ctypes.cast(c_simt_data_p, ctypes.POINTER(AdxBlockInfo))[0]
        simt_block_data_len = simt_block_info.len
        if simt_block_data_len != cls.ADX_SIMT_PRINT_LEN:
            logging.warning(f"Simt block info length [{simt_block_data_len}] is illegal")
            return
        simt_dump_len = debug_buf_size - simt_addr_offset
        simt_dump_cores = simt_dump_len // (simt_block_data_len * cls.ADX_SIMT_MAX_THREAD_NUM)
        for block_idx in range(simt_dump_cores):
            for thread_idx in range(cls.ADX_SIMT_MAX_THREAD_NUM):
                offset = (block_idx * cls.ADX_SIMT_MAX_THREAD_NUM + thread_idx) * simt_block_data_len
                c_ws_ptr = ctypes.c_void_p(c_simt_data_p.value + offset)
                cls._print_block_info(
                    c_ws_ptr, simt_block_data_len, mem_bound, block_idx, AdxSimtDumpMeta, short_soc_version
                )

    @staticmethod
    def _get_max_dump_core_num(short_soc_version: str) -> int:
        if short_soc_version in PLATFORM_BEFORE_DAVID:
            return 75
        return 108

    @staticmethod
    def _get_aic_idx_offset(short_soc_version: str) -> int:
        if short_soc_version in PLATFORM_BEFORE_DAVID:
            return 50
        return 72

    @staticmethod
    def _has_simt_dump_info(debug_buf_size: int, short_soc_version: str) -> bool:
        max_dump_core = AdxInterface._get_max_dump_core_num(short_soc_version)
        return (
            short_soc_version not in PLATFORM_BEFORE_DAVID
            and debug_buf_size > max_dump_core * AdxInterface.ADX_MAX_STR_LEN
        )

    @staticmethod
    def _get_block_data_len(c_ws_data_p: ctypes.c_void_p, debug_buf_size: int):
        block_info = ctypes.cast(c_ws_data_p, ctypes.POINTER(AdxBlockInfo))[0]
        block_data_len = block_info.len
        if (block_data_len not in (AdxInterface.ADX_MAX_STR_LEN, AdxInterface.ADX_ASSERT_LEN)) or block_data_len == 0:
            length = debug_buf_size // ctypes.sizeof(ctypes.c_uint32)
            if length < AdxBlockInfo.size():
                raise RuntimeError(f"debug_buf_size [{debug_buf_size}] is illegal.")
            Uint32Array = ctypes.c_uint32 * length
            uint32_array = ctypes.cast(c_ws_data_p, ctypes.POINTER(Uint32Array)).contents
            for i in range(length):
                if i < 4:  # 4 refers to AdxBlockInfo structure
                    continue
                if uint32_array[i] == AdxInterface.ADX_PRINT_MAGIC:
                    block_data_len = uint32_array[i - 4]
                    break

        if (block_data_len not in (AdxInterface.ADX_MAX_STR_LEN, AdxInterface.ADX_ASSERT_LEN)) or block_data_len == 0:
            raise RuntimeError(f"BlockDataLen [{block_data_len} bytes] is illegal.")

        return block_data_len

    @staticmethod
    def _print_block_info(
        c_ws_data_p: ctypes.c_void_p,
        data_len: int,
        mem_bound: ctypes.c_void_p,
        block_idx: int,
        dump_meta_cls: Union[Type[AdxSimtDumpMeta], Type[AdxDumpMeta]],
        short_soc_version: str,
    ):
        if c_ws_data_p.value + data_len >= mem_bound.value:
            return
        block_info = ctypes.cast(c_ws_data_p, ctypes.POINTER(AdxBlockInfo))[0]
        if block_info.magic != AdxInterface.ADX_PRINT_MAGIC:
            logging.debug(f"Block info [{block_idx}] is illegal, magic is {block_info.magic}.")
            return
        block_info_hdr_len = AdxBlockInfo.size()
        dump_meta_hdr_len = dump_meta_cls.size()
        dump_info_hdr_len = AdxDumpInfoHead.size()
        max_data_len = data_len - block_info_hdr_len - dump_meta_hdr_len
        if max_data_len < block_info.remain_len:
            logging.debug(
                f"Block info remain_len [{block_info.remain_len}] is illegal, must small than {max_data_len}."
            )
            return

        total_log = [""]
        dump_meta = ctypes.cast(block_info.data_ptr(), ctypes.POINTER(dump_meta_cls))[0]
        if dump_meta_cls is AdxSimtDumpMeta:
            core_id = f"AIV-{block_info.core}"
            total_log.append(
                f"Simt DumpHead: {core_id}, ThreadId={dump_meta.thread_id}, "
                f"TotalBlockNum={block_info.block_num}, BlockRemainLen={block_info.remain_len}, "
                f"BlockInitialSpace={block_info.len}, RSV={block_info.rsv}, "
                f"Magic=0x{block_info.magic:x}"
            )
        else:
            core_type = "MIX" if dump_meta.mix_flag != 0 else "AIC" if dump_meta.core_type == 1 else "AIV"
            aic_idx_offset = AdxInterface._get_aic_idx_offset(short_soc_version)
            core_id = (
                f"AIC-{block_info.core - aic_idx_offset}" if dump_meta.core_type == 1 else f"AIV-{block_info.core}"
            )
            total_log.append(
                f"DumpHead: {core_id}, CoreType={core_type}, BlockDim={dump_meta.block_dim}, "
                f"TotalBlockNum={block_info.block_num}, BlockRemainLen={block_info.remain_len}, "
                f"BlockInitialSpace={block_info.len}, RSV={block_info.rsv}, "
                f"Magic=0x{block_info.magic:x}"
            )
        if block_info.rsv == AdxInterface.ADX_PRINT_NO_ENOUGH_SPACE_FLAG:
            logging.warning("Remain block space is not enough, printing information may be incomplete!")
        user_data_len = max_data_len - block_info.remain_len
        offset = 0
        tensor_shape: list = []
        while offset + dump_info_hdr_len <= user_data_len:
            dump_info_head = ctypes.cast(dump_meta.data_ptr_offset(offset), ctypes.POINTER(AdxDumpInfoHead))[0]
            offset += dump_info_hdr_len + dump_info_head.info_len
            if offset > user_data_len:
                logging.debug(f"Dump data info len {dump_info_head.info_len} illegal.")
                break
            AdxInterface._print_user_data(dump_info_head, tensor_shape, total_log)
        if len(total_log) > 2:
            logging.info("\n".join(total_log))

    @staticmethod
    def _print_user_data(dump_info: AdxDumpInfoHead, tensor_shape: list, total_log: list):
        dump_type = dump_info.type
        if dump_type in (AscendDumpType.SCALAR.value, AscendDumpType.ASSERT.value, AscendDumpType.SIMT.value):
            total_log.extend(AdxInterface._print_user_data_scalar(dump_info))
        elif dump_type == AscendDumpType.TENSOR.value:
            total_log.extend(AdxInterface._print_user_data_tensor(dump_info, tensor_shape))
        elif dump_type == AscendDumpType.SHAPE.value:
            total_log.extend(AdxInterface._get_user_data_shape(dump_info, tensor_shape))
        elif dump_type == AscendDumpType.TIMESTAMP.value:
            total_log.extend(AdxInterface._print_user_data_timestamp(dump_info))
        else:
            logging.warning(f"Dump type {dump_type} is illegal.")

    @staticmethod
    def _print_user_data_scalar(dump_info: AdxDumpInfoHead):
        if dump_info.info_len < AdxInterface.ADX_PRINT_ARG_LEN:
            return None
        c_param_begin_p = dump_info.data_ptr()
        fmt_str_offset = ctypes.cast(c_param_begin_p, ctypes.POINTER(ctypes.c_size_t))[0]
        c_fmt_str_p = ctypes.cast(ctypes.c_void_p(c_param_begin_p.value + fmt_str_offset), ctypes.c_char_p)
        fmt_str = c_fmt_str_p.value.decode()
        matches = AdxInterface._match_fmt_pattern(fmt_str)
        if matches:
            # replace in reverse order
            param_num = len(matches)
            for rank, m in enumerate(sorted(matches, reverse=True, key=lambda x: x[1])):
                idx = param_num - rank
                c_param_p = ctypes.c_void_p(c_param_begin_p.value + idx * AdxInterface.ADX_PRINT_ARG_LEN)
                fmt_str = AdxInterface._replace_fmt_pattern(fmt_str, m, c_param_p)
        return [fmt_str]

    @staticmethod
    def _match_fmt_pattern(fmt_str: str):
        matches = []
        pattern = "|".join(list(ADX_FORMATTERS.keys()))
        for match in re.finditer(pattern, fmt_str):
            matches.append((match.group(), match.start(), match.end()))
        return matches

    @staticmethod
    def _replace_fmt_pattern(fmt_str: str, match: tuple, c_param_p: ctypes.c_void_p):
        formatter = ADX_FORMATTERS[match[0]]
        return formatter(fmt_str=fmt_str, fmt_start=match[1], fmt_stop=match[2], c_param_p=c_param_p)

    @staticmethod
    def _print_user_data_tensor(dump_info: AdxDumpInfoHead, tensor_shape: list):
        dump_msg_hdr_len = AdxDumpMessageHead.size()
        if dump_info.info_len < dump_msg_hdr_len:
            return []
        logs = []
        dump_msg_hdr = ctypes.cast(dump_info.data_ptr(), ctypes.POINTER(AdxDumpMessageHead))[0]
        position = (
            ADX_TENSOR_POSITIONS[dump_msg_hdr.position]
            if dump_msg_hdr.position in ADX_TENSOR_POSITIONS
            else str(dump_msg_hdr.position)
        )
        dtype = ACL_ID_TO_DTYPE.get(dump_msg_hdr.dtype)
        logs.append(
            f"DumpTensor: Desc={dump_msg_hdr.desc}, Address=0x{dump_msg_hdr.addr:x}, "
            f"Dtype={dtype or str(dump_msg_hdr.dtype)}, Position={position}"
        )
        data_len = dump_info.info_len - dump_msg_hdr_len
        if data_len % AdxInterface.ADX_PRINT_ARG_LEN != 0:
            logging.debug(f"Data len {data_len} is illegal, must a multiple of {AdxInterface.ADX_PRINT_ARG_LEN}.")
        if not dtype:
            logging.debug(f"Dump tensor does not support dtype [{dtype}]")
        else:
            c_dump_data_p = dump_msg_hdr.data_ptr()
            data_count = data_len // get_dtype_width(dtype)
            c_buffer_type = ctypes.c_char * (data_count * get_dtype_width(dtype))
            c_buffer = c_buffer_type.from_address(c_dump_data_p.value)
            np_array: numpy.ndarray = numpy.frombuffer(
                c_buffer, dtype=resolve_numpy_dtypes([dtype])[0], count=data_count
            )
            if tensor_shape:
                shape_prod = reduce(lambda x, y: x * y, tensor_shape)
                if shape_prod == np_array.size:
                    np_array = np_array.reshape(tensor_shape)
                tensor_shape.clear()
            logs.append(f"\n{np_array}")
        return logs

    @staticmethod
    def _get_user_data_shape(dump_info: AdxDumpInfoHead, tensor_shape: list):
        dump_shape_hdr_len = ctypes.sizeof(AdxDumpShapeMessageHead)
        if dump_info.info_len < dump_shape_hdr_len:
            return []
        dump_shape_hdr = ctypes.cast(dump_info.data_ptr(), ctypes.POINTER(AdxDumpShapeMessageHead))[0]
        dims = dump_shape_hdr.dim
        if dims > 8:
            logging.debug(f"Dump tensor's shape specified is invalid: {dims}")
        else:
            for i in range(dims):
                tensor_shape.append(dump_shape_hdr.shape[i])
        return []

    @staticmethod
    def _print_user_data_timestamp(dump_info: AdxDumpInfoHead):
        dump_ts_hdr_len = ctypes.sizeof(AdxDumpTimestampHead)
        if dump_info.info_len < dump_ts_hdr_len:
            return []
        dump_ts_hdr = ctypes.cast(dump_info.data_ptr(), ctypes.POINTER(AdxDumpTimestampHead))[0]
        return [
            f"DescId={dump_ts_hdr.desc_id}, "
            f"Rsv={dump_ts_hdr.rsv}, "
            f"Timestamp={dump_ts_hdr.sys_cycle}, "
            f"PcPtr={dump_ts_hdr.current_pc}"
        ]
