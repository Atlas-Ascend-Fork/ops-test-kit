#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MC2 HCCL helper for Kernel path.

Provides HCCL communicator initialization and group-name injection so that
HCCL communication-fusion operator tiling (GetRankSize) and multi-rank RTS
launch work in the Kernel path.
"""

import ctypes
import logging
from typing import List, Optional, Tuple

_hccl_dll = None
_acl_dll = None
_hccl_comm_handles: Optional[List[int]] = None
_hccl_group_name: Optional[str] = None
_hccl_rank_size: int = 0
_hccl_device_ids: Optional[List[int]] = None


def _ensure_loaded():
    global _hccl_dll, _acl_dll
    if _hccl_dll is None:
        _hccl_dll = ctypes.CDLL("libhcomm.so")
        _hccl_dll.HcclCommInitAll.restype = ctypes.c_uint32
        _hccl_dll.HcclGetCommName.restype = ctypes.c_int
        _hccl_dll.HcclGetRankSize.restype = ctypes.c_int
        _hccl_dll.HcclCommDestroy.restype = ctypes.c_int
    if _acl_dll is None:
        _acl_dll = ctypes.CDLL("libascendcl.so")


def init_hccl(device_ids: List[int] = None) -> Tuple[List[int], str, int]:
    """Initialize HCCL communicators for the given device IDs.

    Returns (comm_handles, group_name, rank_size).
    """
    global _hccl_comm_handles, _hccl_group_name, _hccl_rank_size, _hccl_device_ids
    if _hccl_comm_handles is not None:
        return _hccl_comm_handles, _hccl_group_name, _hccl_rank_size
    if device_ids is None:
        device_ids = [0, 1]
    _ensure_loaded()
    _acl_dll.aclrtSetDevice(ctypes.c_int32(device_ids[0]))
    ndev = len(device_ids)
    c_devices = (ctypes.c_int32 * ndev)(*device_ids)
    hccl_comms = (ctypes.c_void_p * ndev)()
    ret = _hccl_dll.HcclCommInitAll(ctypes.c_uint32(ndev), c_devices, hccl_comms)
    if ret != 0:
        raise RuntimeError(f"HcclCommInitAll failed with ret={ret} for devices {device_ids}")
    handles = []
    for i in range(ndev):
        v = hccl_comms[i]
        if hasattr(v, "value"):
            v = v.value
        handles.append(v)
    name_buf = ctypes.create_string_buffer(256)
    ret = _hccl_dll.HcclGetCommName(ctypes.c_void_p(handles[0]), name_buf)
    if ret != 0:
        raise RuntimeError(f"HcclGetCommName failed with ret={ret}")
    group_name = name_buf.value.decode("utf-8")
    rank_size = ctypes.c_uint32(0)
    _hccl_dll.HcclGetRankSize(ctypes.c_void_p(handles[0]), ctypes.pointer(rank_size))
    _hccl_comm_handles = handles
    _hccl_group_name = group_name
    _hccl_rank_size = rank_size.value
    _hccl_device_ids = device_ids
    logging.info(f"hccl: initialized HCCL devices={device_ids}, group='{group_name}', rank_size={_hccl_rank_size}")
    return handles, group_name, _hccl_rank_size


def get_hccl_group_name() -> Optional[str]:
    if _hccl_group_name is None:
        init_hccl()
    return _hccl_group_name


def get_hccl_rank_size() -> int:
    if _hccl_rank_size == 0:
        init_hccl()
    return _hccl_rank_size


_HCCL_OPS = frozenset(
    {
        "all_gather_matmul",
        "matmul_all_reduce",
        "matmul_reduce_scatter",
        "allto_all_matmul",
        "matmul_allto_all",
        "matmul_reduce_scatter_v2",
        "all_gather_matmul_v2",
        "grouped_mat_mul_all_reduce",
        "grouped_mat_mul_allto_allv",
        "inplace_matmul_all_reduce_add_rms_norm",
        "matmul_all_reduce_add_rms_norm",
        "batch_mat_mul_reduce_scatter_allto_all",
        "allto_all_all_gather_batch_mat_mul",
        "allto_allv_grouped_mat_mul",
        "allto_allv_quant_grouped_mat_mul",
        "quant_all_reduce",
        "quant_reduce_scatter",
        "quant_grouped_mat_mul_allto_allv",
        "distribute_barrier",
        "distribute_barrier_extend",
        "moe_distribute_dispatch",
        "moe_distribute_combine",
        "moe_distribute_dispatch_v2",
        "moe_distribute_dispatch_v3",
        "moe_distribute_combine_v2",
        "moe_distribute_combine_v3",
        "moe_distribute_combine_add_rms_norm",
        "moe_update_expert",
        "mega_moe",
        "moe_ep_dispatch",
        "moe_ep_dispatch_epilogue",
        "moe_ep_combine",
    }
)


def is_hccl_op(op_name: str) -> bool:
    """Check if an operator is an HCCL communication-fusion operator."""
    return op_name in _HCCL_OPS
