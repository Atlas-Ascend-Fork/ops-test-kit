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
Structures used by DRV
"""

# Standard Packages
import ctypes

MAX_CHIP_NAME = 32


class dsmi_chip_info_stru(ctypes.Structure):
    _fields_ = [
        ("chip_type", ctypes.c_char * MAX_CHIP_NAME),
        ("chip_name", ctypes.c_char * MAX_CHIP_NAME),
        ("chip_ver", ctypes.c_char * MAX_CHIP_NAME),
    ]

    def get_complete_platform(self) -> str:
        res = self.chip_type + self.chip_name
        return res.decode("UTF-8")

    def get_ver(self) -> str:
        return self.chip_ver.decode("UTF-8")

    def __str__(self):
        return f"chip_type: {self.chip_type}, chip_name: {self.chip_name}, chip_version: {self.chip_ver}"

    def __repr__(self):
        return f"chip_type: {self.chip_type}, chip_name: {self.chip_name}, chip_version: {self.chip_ver}"


class dsmi_aicpu_info_stru(ctypes.Structure):
    _fields_ = [
        ("maxFreq", ctypes.c_uint),
        ("curFreq", ctypes.c_uint),
        ("aicpuNum", ctypes.c_uint),
        ("utilRate", ctypes.c_uint * 16),
    ]

    def get_max_frequency(self):
        return int(self.maxFreq)

    def get_cur_frequency(self):
        return int(self.curFreq)

    def get_aicpu_count(self):
        return int(self.aicpuNum)

    def get_util_rate(self, idx: int = 0):
        return int(self.utilRate[idx])

    def get_avg_util_rate(self) -> float:
        return sum(self.utilRate) / 16


class dsmi_memory_info_stru(ctypes.Structure):
    _fields_ = [("memory_size", ctypes.c_ulonglong), ("curFreq", ctypes.c_uint), ("util", ctypes.c_uint)]

    def get_memory_size(self):
        return int(self.memory_size)

    def get_cur_frequency(self):
        return int(self.curFreq)

    def get_util_rate(self):
        return int(self.util)


class dsmi_hbm_info_stru(ctypes.Structure):
    _fields_ = [
        ("memory_size", ctypes.c_ulonglong),
        ("curFreq", ctypes.c_uint),
        ("memory_usage", ctypes.c_ulonglong),
        ("temp", ctypes.c_int),
        ("bandwidth_util_rate", ctypes.c_uint),
    ]

    def get_memory_size(self):
        return int(self.memory_size)

    def get_cur_frequency(self):
        return int(self.curFreq)

    def get_memory_used_size(self):
        return int(self.memory_usage)

    def get_memory_temp(self):
        return int(self.temp)

    def get_bandwidth_util_rate(self):
        return int(self.bandwidth_util_rate)


class dsmi_ecc_info_stru(ctypes.Structure):
    _fields_ = [
        ("enable_flag", ctypes.c_int),
        ("single_bit_error_count", ctypes.c_uint),
        ("double_bit_error_count", ctypes.c_uint),
    ]

    def get_enabled(self):
        if int(self.enable_flag) == 0:
            return False
        if int(self.enable_flag) == 1:
            return True
        return None

    def get_single_bit_error_count(self):
        return int(self.single_bit_error_count)

    def get_double_bit_error_count(self):
        return int(self.double_bit_error_count)
