# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.common.profiler.types: ACL error map, prof enums & compact info struct."""

import ctypes

from ttk.core_modules.common.profiler.types import (
    ACL_ERROR_DESC_DICT,
    AclProfType,
    AiCoreProfMetrics,
    MsprofCompactInfo,
    MsProfOpDfx,
    MsprofTaskType,
    TtkMsProfType,
)


def test_acl_error_desc_known_codes():
    assert ACL_ERROR_DESC_DICT[0] == "ACL_SUCCESS"
    assert ACL_ERROR_DESC_DICT[100000] == "ACL_ERROR_INVALID_PARAM"
    assert ACL_ERROR_DESC_DICT[500000] == "ACL_ERROR_INTERNAL_ERROR"


def test_acl_prof_type_bit_flags():
    assert AclProfType.ACL_API.value == 0x0001
    assert AclProfType.TASK_TIME.value == 0x0002
    assert AclProfType.AICORE_METRICS.value == 0x0004


def test_aicore_prof_metrics():
    assert AiCoreProfMetrics.ARITHMETIC_UTILIZATION.value == 0
    assert AiCoreProfMetrics.NONE.value == 0xFF


def test_ttk_msprof_type_op_combines_flags():
    expected = AclProfType.TASK_TIME.value | AclProfType.AICORE_METRICS.value | AclProfType.L2CACHE.value
    assert TtkMsProfType.OP.value == expected
    assert TtkMsProfType.NONE.value == 0


def test_ttk_msprof_type_api_combines_flags():
    expected = AclProfType.TASK_TIME.value | AclProfType.ACL_API.value | AclProfType.AICORE_METRICS.value
    assert TtkMsProfType.API.value == expected


def test_msprof_task_type_values():
    assert MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value == 0
    assert MsprofTaskType.MSPROF_TASK_TYPE_MIX_AIV.value == 5


def test_msprof_compact_info_construction():
    info = MsprofCompactInfo(time_stamp=100, op_name=200, task_type=0, block_dim=4)
    assert info.magic == 0x5A5A
    assert info.level == 10000
    assert info.data_len == ctypes.sizeof(info.data.node_basic_info)
    assert info.data.node_basic_info.op_name == 200
    assert info.data.node_basic_info.block_dim == 4


def test_msprof_op_dfx_dataclass():
    dfx = MsProfOpDfx(kernel_name="k", block_dim=2, core_type="AIC")
    assert dfx.kernel_name == "k"
    assert dfx.block_dim == 2
    assert dfx.task_ration is None


def test_msprof_compact_info_union_data_size():
    info = MsprofCompactInfo(time_stamp=0, op_name=0, task_type=0, block_dim=0)
    assert len(info.data.info) == 40
