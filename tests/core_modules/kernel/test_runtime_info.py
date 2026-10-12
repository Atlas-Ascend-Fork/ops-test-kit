# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.kernel.runtime.rts_info & rts_structures: enums, error maps & ctypes structs."""

import ctypes

import pytest

from ttk.core_modules.kernel.runtime.rts_info import (
    RTS_MEMORY_TYPE,
    MsprofCommandHandleType,
    RtFloatOverflowMode,
    RtLimitType,
    rt_acl_error_code_dict,
    rt_binary_magic_dict,
    rt_context_mode,
    rt_error_type_dict,
    rt_memcpy_kind,
    rt_memory_policy,
)
from ttk.core_modules.kernel.runtime.rts_structures import (
    LaunchKernelArgs,
    RtArgsEx,
    RtArgsSizeInfo,
    RtDevBinary,
    RtLaunchAttribute,
    RtLaunchAttributeId,
)


def test_binary_magic_dict_values():
    assert rt_binary_magic_dict["RT_DEV_BINARY_MAGIC_PLAIN"] == 0xABCEED50
    assert rt_binary_magic_dict["RT_DEV_BINARY_MAGIC_ELF"] == 0x43554245


def test_memory_type_enum():
    assert RTS_MEMORY_TYPE.RT_MEMORY_HBM.value == 2
    assert RTS_MEMORY_TYPE.RT_MEMORY_L1.value == 0x10000


def test_command_handle_type_enum():
    assert MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_INIT.value == 0
    assert MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_STOP.value == 2


def test_limit_type_enum():
    assert RtLimitType.SIMT_WARP_STACK_SIZE.value == 1
    assert RtLimitType.SIMT_DVG_WARP_STACK_SIZE.value == 2


def test_float_overflow_mode_enum():
    assert RtFloatOverflowMode.RT_OVERFLOW_MODE_SATURATION.value == 0


def test_memory_policy_dict():
    assert rt_memory_policy["RT_MEMORY_POLICY_NONE"] == 0x0
    assert rt_memory_policy["RT_MEMORY_POLICY_HUGE_PAGE_ONLY"] == 0x800


def test_memcpy_kind_dict():
    assert rt_memcpy_kind["RT_MEMCPY_HOST_TO_DEVICE"] == 1
    assert rt_memcpy_kind["RT_MEMCPY_DEVICE_TO_HOST"] == 2


def test_context_mode_dict():
    assert rt_context_mode["RT_CTX_NORMAL_MODE"] == 0
    assert rt_context_mode["RT_CTX_GEN_MODE"] == 1


def test_error_type_dict():
    assert rt_error_type_dict[0x00010000] == "DEVICE"
    assert rt_error_type_dict[0x00080000] == "KERNEL"


def test_acl_error_code_dict():
    assert rt_acl_error_code_dict[0] == "ACL_RT_SUCCESS"
    assert rt_acl_error_code_dict[107000] == "ACL_ERROR_RT_PARAM_INVALID"
    assert rt_acl_error_code_dict[507015] == "ACL_ERROR_RT_AICORE_EXCEPTION"


@pytest.mark.parametrize("attr", [RtLaunchAttributeId.BLOCK_DIM, RtLaunchAttributeId.DUMP_FLAG])
def test_launch_attribute_id_lower_name(attr):
    assert attr.lower_name() == attr.name.lower()


def test_rt_dev_binary_structure_fields():
    binary = RtDevBinary()
    binary.magic = 0xABCEED50
    binary.version = 1
    binary.length = 100
    assert binary.magic == 0xABCEED50
    assert binary.length == 100


def test_rt_args_ex_alloc_buf():
    args = RtArgsEx()
    ptr = args.alloc_buf(32)
    assert ptr is not None
    assert args.args_size == 32


def test_rt_args_size_info_magic():
    buf = (ctypes.c_uint64 * 4)()
    info = RtArgsSizeInfo(ctypes.c_void_p(ctypes.addressof(buf)))
    assert info.magic_match() is True


def test_launch_attribute_block_dim():
    attr = RtLaunchAttribute(RtLaunchAttributeId.BLOCK_DIM, 8)
    assert attr.id == RtLaunchAttributeId.BLOCK_DIM.value
    assert attr.value.block_dim == 8


def test_launch_kernel_args_post_init_filters_none():
    args = LaunchKernelArgs(op_args=[1, None, 2], dfx_args=[None, 3])
    assert args.op_args == [1, 2]
    assert args.dfx_args == [3]
