#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""DSMI（NPU 管面查询，libdrvdsmi_host.so ctypes）。

文件拆分（原 3 文件平铺 → 语义命名）：
- interface.py：DSMIInterface（Singleton，libdsmi so 加载 + 查询方法）
- structures.py：ctypes 结构（dsmi_chip_info_stru / aicpu / memory / hbm / ecc）
- info.py：枚举常量（ERROR_CODE / HEALTH_STATE / FREQ·UTIL·ECC_DEVICE_TYPE）
"""

__all__ = ["DSMIInterface", "DSMI_HEALTH_STATE"]

from .info import DSMI_HEALTH_STATE
from .interface import DSMIInterface
