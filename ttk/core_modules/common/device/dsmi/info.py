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
DSMI Info
"""

# Standard Packages
from enum import Enum


class DSMI_ERROR_CODE(Enum):
    DSMI_ERROR_NONE = 0
    DSMI_ERROR_NO_DEVICE = 1
    DSMI_ERROR_INVALID_DEVICE = 2
    DSMI_ERROR_INVALID_HANDLE = 3
    DSMI_ERROR_INNER_ERR = 7
    DSMI_ERROR_PARA_ERROR = 8
    DSMI_ERROR_NOT_EXIST = 11
    DSMI_ERROR_BUSY = 13
    DSMI_ERROR_WAIT_TIMEOUT = 16
    DSMI_ERROR_IOCRL_FAIL = 17
    DSMI_ERROR_SEND_MESG = 27
    DSMI_ERROR_OPER_NOT_PERMITTED = 46
    DSMI_ERROR_TRY_AGAIN = 51
    DSMI_ERROR_MEMORY_OPT_FAIL = 58
    DSMI_ERROR_PARTITION_NOT_RIGHT = 86
    DSMI_ERROR_RESOURCE_OCCUPIED = 87
    DSMI_ERROR_NOT_SUPPORT = 0xFFFE


class DSMI_HEALTH_STATE(Enum):
    OK = 0
    WARNING = 1
    IMPORTANT = 2
    CRITICAL = 3
    OFF = 0xFFFFFFFF


class DSMI_FREQ_DEVICE_TYPE(Enum):
    MEMORY = 1
    CTRLCPU = 2
    HBM = 6
    AICORE_CURRENT = 7
    AICORE_RATED = 9
    VECCORE_CURRENT = 12


DsmiFreqSupportedType: dict = {
    "Ascend310": (1, 2, 6, 7, 9),
    "Ascend310B": (1, 2, 7, 9),
    "AS31XM1": (1, 2, 7, 9),
    "Ascend310P": (1, 2, 7, 9, 12),
    "Ascend610": (1, 2, 7, 9, 12),
    "Ascend910": (1, 2, 6, 7, 9),
    "Ascend910B": (2, 6, 7, 9),
    "Ascend910_93": (2, 6, 7, 9),
    "BS9SX1A": (1, 2, 7, 9, 12),
    "Ascend950": (2, 6, 7, 9),
}


class DSMI_UTIL_DEVICE_TYPE(Enum):
    MEMORY = 1
    AICORE = 2
    AICPU = 3
    CTRLCPU = 4
    BUS = 5
    HBM = 6
    DDR = 8
    HBM_BANDWIDTH = 10
    VECCORE = 12


DsmiUtilSupportedType: dict = {
    "Ascend310": (1, 2, 3, 4, 5, 6, 8),
    "Ascend310B": (1, 2, 3, 4, 5, 12),
    "AS31XM1": (1, 2, 3, 4, 5, 12),
    "Ascend310P": (1, 2, 3, 4, 5, 12),
    "Ascend610": (1, 2, 3, 4, 5, 12),
    "Ascend910": (1, 2, 3, 4, 5, 6, 10),
    "Ascend910B": (2, 3, 4, 6, 10, 12),
    "Ascend910_93": (2, 3, 4, 6, 10, 12),
    "BS9SX1A": (1, 2, 3, 4, 5, 12),
    "Ascend950": (2, 3, 4, 6, 10, 12),
}


class DSMI_ECC_DEVICE_TYPE(Enum):
    DDR = 0
    SRAM = 1
    HBM = 2
    HBM_RECORDED_SINGLE_ADDR = 3
    HBM_RECORDED_MULTI_ADDR = 4
    NONE = 0xFF
