#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/device：NPU 硬件世界的只读地图——设备枚举（DSMI ctypes）、芯片身份
（SoC/平台）、路径解析（OPP/算子实现）、健康状态。

边界：只查询不控制（set_device/内存分配在通路 runtime 层）；DSMI 管**运行态**
（libdrvdsmi_host.so，实例化时才 CDLL），platform 管**静态映射**（纯文件路径/型号
→SoC 参数，零 .so）。

打屏约束：主进程只有 logging 的一个 stdout handler 与
print_progress 两个写点，均在单线程 run_loop 内顺序调用。
"""

__all__ = [
    "DSMIInterface",
    "DSMI_HEALTH_STATE",
    "PLATFORM_BEFORE_DAVID",
    "get_opp_paths",
    "get_op_impl_paths",
    "get_impl_base_paths",
    "get_op_info_paths",
    "get_ascend_lib64_path",
    "get_npu_hw_info",
    "validate_core_limit",
    "get_l0_clear_tile_params",
    "get_ascend_scene_info",
]

from .dsmi import DSMI_HEALTH_STATE, DSMIInterface
from .platform import (
    PLATFORM_BEFORE_DAVID,
    get_ascend_lib64_path,
    get_ascend_scene_info,
    get_impl_base_paths,
    get_l0_clear_tile_params,
    get_npu_hw_info,
    get_op_impl_paths,
    get_op_info_paths,
    get_opp_paths,
    validate_core_limit,
)
