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
多卡（HCCL）rank 输入 staging：逐 rank 生成输入并汇成 inputs.npz。

MC2 共享权重与 MoE combine 一致性的算子特定语义亦在本模块实现。
编排层（worker.py）惰性导入本模块，仅多卡路径加载。
"""

import copy
import os

import numpy as np

from ttk.utilities.dtypes import resolve_numpy_dtypes

from ..input_generate import generate_inputs

# MC2 系共享权重算子（x1/x2 跨 rank 同值，idx 1 权重按 rank 无关种子重生成）
_WEIGHT_SHARED_OPS = (
    "MatmulReduceScatter",
    "AllGatherMatmul",
    "MatmulAllReduce",
    "MatmulAlltoAll",
    "AlltoAllMatmul",
    "MatmulReduceScatterV2",
    "QuantMatmulAllReduce",
    "mm_reduce_scatter",
    "all_gather_mm",
    "mm_all_reduce",
    "mm_all_to_all",
    "all_to_all_mm",
    "npu_moe_distribute_combine",
)
_WEIGHT_SHARED_EXCLUDE_OPS = (
    "GroupedMatMul",
    "AlltoAllvGrouped",
    "BatchMatMul",
    "bmm_reduce_scatter",
    "bmm_reducescatter",
    "all_to_all_all_gather",
    "alltoall_allgather",
)


def stage_rank_inputs(testcase, switches, backend, plan, rank_template, ndev, tmp_dir):
    """逐 rank 生成输入并汇成 inputs.npz（rank 特异种子 + 跨 rank 共享位对齐）。

    共享位语义：
    - MC2 matmul 族：x2（权重，idx 1）按 rank 无关种子重生成——x1 各 rank
      不同（每卡自己的输入分片），权重全 rank 同值（对齐 ACLNN 多卡行为）；
    - MoE combine：expand_x/expert_ids（idx 0/1）全 rank 共享，保证
      ep_send_counts 对 HCCL alltoallv 一致，并从 expert_ids 派生一致的
      ep_send_counts/expand_idx/expert_scales。

    返回 input_path。
    """
    input_path = os.path.join(tmp_dir, "inputs.npz")
    save_arrays = {}
    # Generate per-rank inputs with rank-specific seed (matching ACLNN multi-device behavior)
    base_seed = switches.random_seed or 0

    # Determine if this op needs shared weight (x1/x2 same across ranks)
    api_name_str = testcase.api_name or ""
    needs_shared_weight = not any(kw in api_name_str for kw in _WEIGHT_SHARED_EXCLUDE_OPS) and any(
        kw in api_name_str for kw in _WEIGHT_SHARED_OPS
    )
    # MoE combine: expand_x (idx 0) and expert_ids (idx 1) must be shared across ranks
    # so that ep_send_counts are consistent for HCCL alltoallv.
    moe_combine_shared_all = "npu_moe_distribute_combine" in api_name_str

    # rank 循环不变量：声明态字段在逐 rank 生成期间不变（per-rank 拷贝同值）
    flat_dtypes = resolve_numpy_dtypes(testcase.flat_tensor_dtypes)
    flat_ranges = testcase.flat_input_data_ranges or ()

    for rank_idx in range(ndev):
        per_rank_seed = base_seed + rank_idx * 1000
        np.random.seed(per_rank_seed)
        rank_inputs = generate_inputs(copy.deepcopy(rank_template), switches, backend, plan)
        # For MC2 matmul ops, regenerate x2 (weight) with rank-independent seed so
        # all ranks share the same weight (matching ACLNN multi-device behavior).
        # x1 stays per-rank different (each rank has its own input chunk).
        if moe_combine_shared_all and len(rank_inputs) > 0:
            # Share all inputs across ranks for moe combine (idx 0 and 1)
            for t_idx in range(min(2, len(rank_inputs))):
                if rank_inputs[t_idx] is not None:
                    ss = testcase.flat_storage_shape(t_idx)
                    dtype = flat_dtypes[t_idx] if t_idx < len(flat_dtypes) else None
                    data_range = flat_ranges[t_idx] if t_idx < len(flat_ranges) else (None, None)
                    low = data_range[0] if data_range and data_range[0] is not None else -1.0
                    high = data_range[1] if data_range and data_range[1] is not None else 1.0
                    rng = np.random.RandomState(base_seed + t_idx)
                    if "int" in str(dtype):
                        np_arr = rng.randint(int(low), int(high) + 1, ss).astype(dtype, copy=False)
                    else:
                        np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                    rank_inputs[t_idx] = np_arr.reshape(ss)
        elif needs_shared_weight and len(rank_inputs) > 1 and rank_inputs[1] is not None:
            t_idx = 1
            if t_idx < len(rank_inputs) and rank_inputs[t_idx] is not None:
                ss = testcase.flat_storage_shape(t_idx)
                dtype = flat_dtypes[t_idx] if t_idx < len(flat_dtypes) else None
                data_range = flat_ranges[t_idx] if t_idx < len(flat_ranges) else (None, None)
                low = data_range[0] if data_range and data_range[0] is not None else -1.0
                high = data_range[1] if data_range and data_range[1] is not None else 1.0
                rng = np.random.RandomState(base_seed + t_idx)
                np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                rank_inputs[t_idx] = np_arr.reshape(ss)
        # MoE combine: generate consistent ep_send_counts, expand_idx, expert_scales
        # from expert_ids (which is shared across ranks).
        if moe_combine_shared_all and len(rank_inputs) > 1:
            eid = rank_inputs[1]
            if eid is not None:
                bs2 = eid.shape[0]
                k2 = eid.shape[1] if eid.ndim > 1 else 1
                ep_ws2 = int(testcase.attributes.get("ep_world_size", ndev))
                moe_exp2 = int(testcase.attributes.get("moe_expert_num", ep_ws2))
                local_exp2 = moe_exp2 // ep_ws2 if ep_ws2 > 0 else moe_exp2
                send_counts2 = [0] * ep_ws2
                for ii in range(bs2):
                    for jj in range(k2):
                        e_id = int(eid[ii][jj]) if eid.ndim > 1 else int(eid[ii])
                        dest2 = e_id // local_exp2 if local_exp2 > 0 else 0
                        if dest2 >= ep_ws2:
                            dest2 = ep_ws2 - 1
                        send_counts2[dest2] += 1
                cumsum2 = []
                run2 = 0
                for ii in range(ep_ws2):
                    run2 += send_counts2[ii]
                    cumsum2.append(run2)
                if len(rank_inputs) > 3 and rank_inputs[3] is not None:
                    rank_inputs[3] = np.array(cumsum2, dtype=np.int32)
                if len(rank_inputs) > 2 and rank_inputs[2] is not None:
                    rank_inputs[2] = np.arange(bs2 * k2, dtype=np.int32)
                if len(rank_inputs) > 4 and rank_inputs[4] is not None:
                    rank_inputs[4] = np.ones((bs2, k2), dtype=np.float32)
        for i, r in enumerate(rank_inputs):
            if r is not None:
                # For shared weight (idx >= 1 when needs_shared_weight),
                # only save rank 0's copy; other ranks will fall back to rank 0.
                if needs_shared_weight and i >= 1 and rank_idx != 0:
                    continue
                # fp8/hifloat8 dtypes can't round-trip through npz (e5m2 descr '<f1'
                # fails on load). Save as uint8 view; worker restores via tensor_dtypes.
                arr_to_save = r
                if hasattr(r.dtype, "itemsize") and r.dtype.itemsize == 1 and r.dtype.kind in ("V", "f"):
                    # Could be fp8_e4m3fn (|V1), fp8_e5m2 (<f1), fp8_e8m0, hifloat8
                    dtype_str = str(r.dtype)
                    if any(k in dtype_str for k in ("float8", "hifloat8")):
                        arr_to_save = r.view(np.uint8)
                save_arrays[f"inp_{rank_idx}_{i}"] = arr_to_save
    np.savez(input_path, **save_arrays)
    return input_path
