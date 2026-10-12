#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Declaration info from acl interface.
"""

__all__ = ["ACL_ERROR_DESC_DICT", "TtkMsProfType", "AclProfType", "AiCoreProfMetrics", "MsProfOpDfx"]


import ctypes
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

ACL_ERROR_DESC_DICT = {
    0: "ACL_SUCCESS",
    100000: "ACL_ERROR_INVALID_PARAM",
    100001: "ACL_ERROR_UNINITIALIZE",
    100002: "ACL_ERROR_REPEAT_INITIALIZE",
    100003: "ACL_ERROR_INVALID_FILE",
    100004: "ACL_ERROR_WRITE_FILE",
    100005: "ACL_ERROR_INVALID_FILE_SIZE",
    100006: "ACL_ERROR_PARSE_FILE",
    100007: "ACL_ERROR_FILE_MISSING_ATTR",
    100008: "ACL_ERROR_FILE_ATTR_INVALID",
    100009: "ACL_ERROR_INVALID_DUMP_CONFIG",
    100010: "ACL_ERROR_INVALID_PROFILING_CONFIG",
    100011: "ACL_ERROR_INVALID_MODEL_ID",
    100012: "ACL_ERROR_DESERIALIZE_MODEL",
    100013: "ACL_ERROR_PARSE_MODEL",
    100014: "ACL_ERROR_READ_MODEL_FAILURE",
    100015: "ACL_ERROR_MODEL_SIZE_INVALID",
    100016: "ACL_ERROR_MODEL_MISSING_ATTR",
    100017: "ACL_ERROR_MODEL_INPUT_NOT_MATCH",
    100018: "ACL_ERROR_MODEL_OUTPUT_NOT_MATCH",
    100019: "ACL_ERROR_MODEL_NOT_DYNAMIC",
    100020: "ACL_ERROR_OP_TYPE_NOT_MATCH",
    100021: "ACL_ERROR_OP_INPUT_NOT_MATCH",
    100022: "ACL_ERROR_OP_OUTPUT_NOT_MATCH",
    100023: "ACL_ERROR_OP_ATTR_NOT_MATCH",
    100024: "ACL_ERROR_OP_NOT_FOUND",
    100025: "ACL_ERROR_OP_LOAD_FAILED",
    100026: "ACL_ERROR_UNSUPPORTED_DATA_TYPE",
    100027: "ACL_ERROR_FORMAT_NOT_MATCH",
    100028: "ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED",
    100029: "ACL_ERROR_KERNEL_NOT_FOUND",
    100030: "ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED",
    100031: "ACL_ERROR_KERNEL_ALREADY_REGISTERED",
    100032: "ACL_ERROR_INVALID_QUEUE_ID",
    100033: "ACL_ERROR_REPEAT_SUBSCRIBE",
    100034: "ACL_ERROR_STREAM_NOT_SUBSCRIBE",
    100035: "ACL_ERROR_THREAD_NOT_SUBSCRIBE",
    100036: "ACL_ERROR_WAIT_CALLBACK_TIMEOUT",
    100037: "ACL_ERROR_REPEAT_FINALIZE",
    100038: "ACL_ERROR_NOT_STATIC_AIPP",
    100039: "ACL_ERROR_COMPILING_STUB_MODE",
    100040: "ACL_ERROR_GROUP_NOT_SET",
    100041: "ACL_ERROR_GROUP_NOT_CREATE",
    100042: "ACL_ERROR_PROF_ALREADY_RUN",
    100043: "ACL_ERROR_PROF_NOT_RUN",
    100044: "ACL_ERROR_DUMP_ALREADY_RUN",
    100045: "ACL_ERROR_DUMP_NOT_RUN",
    148046: "ACL_ERROR_PROF_REPEAT_SUBSCRIBE",
    148047: "ACL_ERROR_PROF_API_CONFLICT",
    148048: "ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG",
    148049: "ACL_ERROR_INVALID_OPP_PATH",
    148050: "ACL_ERROR_OP_UNSUPPORTED_DYNAMIC",
    148051: "ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED",
    148052: "ACL_ERROR_UNSUPPORTED_JPEG",
    200000: "ACL_ERROR_BAD_ALLOC",
    200001: "ACL_ERROR_API_NOT_SUPPORT",
    200002: "ACL_ERROR_INVALID_DEVICE",
    200003: "ACL_ERROR_MEMORY_ADDRESS_UNALIGNED",
    200004: "ACL_ERROR_RESOURCE_NOT_MATCH",
    200005: "ACL_ERROR_INVALID_RESOURCE_HANDLE",
    200006: "ACL_ERROR_FEATURE_UNSUPPORTED",
    200007: "ACL_ERROR_PROF_MODULES_UNSUPPORTED",
    300000: "ACL_ERROR_STORAGE_OVER_LIMIT",
    500000: "ACL_ERROR_INTERNAL_ERROR",
    500001: "ACL_ERROR_FAILURE",
    500002: "ACL_ERROR_GE_FAILURE",
    500003: "ACL_ERROR_RT_FAILURE",
    500004: "ACL_ERROR_DRV_FAILURE",
    500005: "ACL_ERROR_PROFILING_FAILURE",
}


class AclProfType(Enum):
    # 表示采集AscendCL接口的性能数据，
    # 包括Host与Device之间、Device间的同步异步内存复制时延等
    ACL_API = 0x0001
    # 采集算子下发耗时、算子执行耗时数据
    # 以及算子基本信息数据，提供更全面的性能分析数据
    TASK_TIME = 0x0002
    # 表示采集AI Core性能指标数据，
    # 逻辑或时必须包括该宏，aicoreMetrics入参处配置的性能指标采集项才有效
    AICORE_METRICS = 0x0004
    # 表示采集AI CPU任务的开始、结束数据
    AICPU = 0x0008
    # 表示采集L2 Cache数据
    L2CACHE = 0x0010
    # 控制HCCL数据采集开关
    HCCL_TRACE = 0x0020
    # 采集迭代轨迹数据开关，即训练任务及AI软件栈的软件信息，
    # 实现对训练任务的性能分析，重点关注前后向计算和梯度聚合更新等相关数据
    TRAINING_TRACE = 0x0040
    # 获取用户和上层框架程序输出的性能数据。
    MSPROFTX = 0x0080
    # 控制runtime api性能数据采集开关
    RUNTIME_API = 0x0100
    # 采集算子下发耗时、算子执行耗时数据。
    # 与ACL_PROF_TASK_TIME相比，
    # 由于不采集算子基本信息数据，
    # 采集时性能开销较小，可更精准统计相关耗时数据
    TASK_TIME_L0 = 0x0800
    # 控制CANN算子的内存占用情况采集开关。仅采集GE组件算子
    TASK_MEMORY = 0x1000
    # 控制采集算子的属性信息开关，当前仅支持采集aclnnApi
    OP_ATTR = 0x4000


class AiCoreProfMetrics(Enum):
    ARITHMETIC_UTILIZATION = 0
    PIPE_UTILIZATION = 1
    MEMORY_BANDWIDTH = 2
    L0B_AND_WIDTH = 3
    RESOURCE_CONFLICT_RATIO = 4
    MEMORY_UB = 5
    L2_CACHE = 6
    NONE = 0xFF


class TtkMsProfType(Enum):
    NONE = 0
    OP = AclProfType.TASK_TIME.value | AclProfType.AICORE_METRICS.value | AclProfType.L2CACHE.value
    API = AclProfType.TASK_TIME.value | AclProfType.ACL_API.value | AclProfType.AICORE_METRICS.value


class MsprofTaskType(Enum):
    MSPROF_TASK_TYPE_AI_CORE = 0
    MSPROF_TASK_TYPE_AIV = 2
    MSPROF_TASK_TYPE_MIX_AIC = 4
    MSPROF_TASK_TYPE_MIX_AIV = 5


class MsprofNodeBasicInfo(ctypes.Structure):
    _fields_ = [
        ("op_name", ctypes.c_uint64),
        ("task_type", ctypes.c_uint32),
        ("op_type", ctypes.c_uint64),
        ("block_dim", ctypes.c_uint32),
        ("op_flag", ctypes.c_uint32),
    ]

    def __init__(self, op_name: int, task_type: int, op_type: int, block_dim: int):
        self.op_name = op_name
        self.task_type = task_type
        self.op_type = op_type
        self.block_dim = block_dim
        self.op_flag = 0


class MsprofCompactInfoUnionData(ctypes.Union):
    _fields_ = [
        ("info", ctypes.c_uint8 * 40),
        ("node_basic_info", MsprofNodeBasicInfo),
    ]


class MsprofCompactInfo(ctypes.Structure):
    """MsprofReportCompactInfo buffer data"""

    _fields_ = [
        ("magic", ctypes.c_uint16),
        ("level", ctypes.c_uint16),
        ("type", ctypes.c_uint32),
        ("thread_id", ctypes.c_uint32),
        ("data_len", ctypes.c_uint32),
        ("time_stamp", ctypes.c_uint64),
        ("data", MsprofCompactInfoUnionData),
    ]

    def __init__(self, time_stamp: int, op_name: int, task_type: int, block_dim: int):
        super().__init__()
        self.magic = 0x5A5A
        self.level = 10000
        self.type = 0
        self.thread_id = threading.get_ident()
        self.data_len = ctypes.sizeof(MsprofNodeBasicInfo)
        basic_info = MsprofNodeBasicInfo(op_name, task_type, op_name, block_dim)
        self.data.node_basic_info = basic_info


@dataclass
class MsProfOpDfx:
    kernel_name: str
    block_dim: int
    core_type: str
    task_ration: Optional[tuple] = None
