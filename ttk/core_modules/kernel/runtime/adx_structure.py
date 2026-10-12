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
Structures of ADump
"""

# Standard Packages
import ctypes
from enum import Enum


class AscendDumpType(Enum):
    DEFAULT = 0
    SCALAR = 1
    TENSOR = 2
    SHAPE = 3
    ASSERT = 4
    META = 5
    TIMESTAMP = 6
    SIMT = 7


class AdxHeaderBase(ctypes.Structure):
    @classmethod
    def size(cls):
        return ctypes.sizeof(cls)

    def data_ptr(self) -> ctypes.c_void_p:
        c_self_p = ctypes.addressof(self)
        return ctypes.c_void_p(c_self_p + self.size())


class AdxBlockInfo(AdxHeaderBase):
    """
    AdxBlockInfo
    """

    _fields_ = [
        ("len", ctypes.c_uint32),
        ("core", ctypes.c_uint32),
        ("block_num", ctypes.c_uint32),
        ("remain_len", ctypes.c_uint32),
        ("magic", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32),
        ("dump_addr", ctypes.c_uint64),
    ]


class AdxDumpMeta(AdxHeaderBase):
    """
    AdxDumpMeta
    """

    _fields_ = [
        ("type_id", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
        ("block_dim", ctypes.c_uint16),
        ("core_type", ctypes.c_uint8),
        ("mix_flag", ctypes.c_uint8),
        ("rsv", ctypes.c_uint32),
    ]

    def data_ptr_offset(self, offset: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(self.data_ptr().value + offset)


class AdxSimtDumpMeta(AdxHeaderBase):
    """
    AdxSimtDumpMeta
    """

    _fields_ = [
        ("type_id", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
        ("thread_id", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32),
    ]

    def data_ptr_offset(self, offset: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(self.data_ptr().value + offset)


class AdxDumpInfoHead(AdxHeaderBase):
    """
    AdxDumpInfoHead
    """

    _pack_ = 1
    _fields_ = [("type", ctypes.c_uint32), ("info_len", ctypes.c_uint32)]


class AdxDumpMessageHead(AdxHeaderBase):
    """
    AdxDumpMessageHead
    """

    _fields_ = [
        ("addr", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("desc", ctypes.c_uint32),
        ("buffer_id", ctypes.c_uint32),
        ("position", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32),
    ]


class AdxDumpShapeMessageHead(ctypes.Structure):
    """
    AdxDumpShapeMessageHead
    """

    _fields_ = [("dim", ctypes.c_uint32), ("shape", ctypes.c_uint32 * 8), ("rsv", ctypes.c_uint32)]


class AdxDumpTimestampHead(ctypes.Structure):
    _fields_ = [
        ("desc_id", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32),
        ("sys_cycle", ctypes.c_uint64),
        ("current_pc", ctypes.c_uint64),
    ]
