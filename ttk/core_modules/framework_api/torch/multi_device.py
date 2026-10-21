#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
import datetime
import importlib
import json
import logging
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

_moe_dispatch_module = None
_moe_combine_module = None
_moe_context_manager_cls = None
_mega_moe_module = None


_TORCH_NPU_MOE_TENSOR_PARAMS = {
    "torch_npu.npu_moe_distribute_dispatch": ("x", "expert_ids"),
    "torch_npu.npu_moe_distribute_dispatch_v2": ("x", "expert_ids"),
    "torch_npu.npu_moe_distribute_dispatch_setup": ("x", "expert_ids"),
    "torch_npu.npu_moe_distribute_dispatch_teardown": ("x", "y", "expert_ids", "comm_cmd_info"),
    "torch_npu.npu_moe_distribute_combine": ("expand_x", "expert_ids", "expand_idx", "ep_send_counts", "expert_scales"),
    "torch_npu.npu_moe_distribute_combine_v2": (
        "expand_x",
        "expert_ids",
        "assist_info_for_combine",
        "ep_send_counts",
        "expert_scales",
    ),
    "torch_npu.npu_moe_distribute_combine_add_rms_norm": (
        "expand_x",
        "expert_ids",
        "expand_idx",
        "ep_send_counts",
        "expert_scales",
        "residual_x",
        "gamma",
    ),
    "torch_npu.npu_moe_distribute_combine_setup": ("expand_x", "expert_ids", "assist_info_for_combine"),
    "torch_npu.npu_moe_distribute_combine_teardown": (
        "expand_x",
        "quant_expand_x",
        "expert_ids",
        "expand_idx",
        "expert_scales",
        "comm_cmd_info",
    ),
}


def _uses_private_moe_backend(api_name):
    """Return whether the API needs TTK's private CANN MoE extension path."""
    return api_name.startswith(("cann_ops_transformer.", "torch.ops.cann_ops_transformer."))


def _call_torch_npu_moe(api_name, dev_tensors, hcomm, world_size, attrs):
    """Call public torch_npu MoE APIs without the private context tensor."""
    resolved = _resolve_api(api_name)
    kw = dict(attrs)
    kw.pop("_tensor_dtypes", None)
    tensor_count = len(_TORCH_NPU_MOE_TENSOR_PARAMS[api_name])
    tensors = list(dev_tensors[:tensor_count])
    ep_world_size = int(kw.pop("ep_world_size", world_size))
    # In multi-device E2E, _rank is the actual per-rank value set by the
    # worker. CSV ep_rank_id is only present to satisfy overload validation;
    # the actual per-rank value must take precedence.
    if "_rank" in kw:
        ep_rank_id = int(kw.pop("_rank"))
        kw.pop("ep_rank_id", None)
    else:
        ep_rank_id = int(kw.pop("ep_rank_id", 0))
    moe_expert_num = int(kw.pop("moe_expert_num"))
    kw.pop("group_ep", None)
    kw["group_tp"] = kw.get("group_tp", "")
    return resolved(*tensors, hcomm, ep_world_size, ep_rank_id, moe_expert_num, **kw)


def _get_moe_ccl_buffer_size(
    world_size, num_tokens, hidden, num_experts, topk, shared_experts=0, shared_expert_ranks=0, comm_alg=""
):
    """Mirror MoeDistributeBuffer.get_low_latency_ccl_buffer_size without JIT imports."""

    def align(value, base):
        return (value + base - 1) // base * base

    expert_world_size = world_size - shared_expert_ranks
    if expert_world_size <= 0:
        raise ValueError(
            f"shared_expert_ranks must be less than world_size (got {shared_expert_ranks} and {world_size})"
        )
    local_experts = num_experts // expert_world_size
    token_bytes = align(hidden * 2, 32) + 44
    if comm_alg == "fullmesh_v2":
        dispatch_token_bytes = align(token_bytes, 480) // 480 * 512
    else:
        dispatch_token_bytes = align(token_bytes, 512)
    combine_token_bytes = align(hidden * 2, 512)
    minimum_bytes = (
        2
        * (
            num_tokens * dispatch_token_bytes * world_size * local_experts
            + num_tokens * combine_token_bytes * (topk + shared_experts)
        )
        + 1024 * 1024
    )
    return align(align(minimum_bytes, 1024 * 1024) // (1024 * 1024), 2) // 2


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _GoldenContext:
    """Bundles the golden-relevant parameters to keep function signatures lean."""

    __slots__ = ("cpu_inputs_per_rank", "attrs", "rank", "ws", "dist_avail")

    def __init__(self, cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
        self.cpu_inputs_per_rank = cpu_inputs_per_rank
        self.attrs = attrs
        self.rank = rank
        self.ws = ws
        self.dist_avail = dist_avail


def _try_plugin_golden(api_name, ctx):
    """Attempt to load golden from plugin spec. Returns list or None."""
    plugin_path = os.environ.get("TTK_E2E_PLUGIN", "")
    if not plugin_path:
        return None
    try:
        from ttk.core_modules.common.plugin import SpecResolver

        plugin_paths = tuple(p_ for p_ in plugin_path.split(",") if p_)
        fn = SpecResolver(api_name, "e2e", plugin_paths).resolve_golden()
        if fn is None:
            return None
        result = fn(ctx.cpu_inputs_per_rank, ctx.attrs, ctx.rank, ctx.ws, ctx.dist_avail)
        if result is None:
            return None
        return result if isinstance(result, list) else [result]
    except Exception:
        import traceback

        traceback.print_exc()
        return None


def _compute_golden(api_name, cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
    """Plugin-first golden dispatch. Returns the per-rank golden list or None.

    多卡 golden 依赖跨 rank 语义：无插件（或插件返回 None）时返回 None，由
    worker 落 NO_GOLDEN_SOURCE FAIL——不在进程内伪造占位值，父进程 CPU 链
    也无法复算（migrated 算子的 golden 均在 ops-transformer per-operator
    tests/assets 下，经 --plugin 激活）。
    """
    # Plugin-first: if the operator has a golden spec,
    # delegate to it. No plugin → explicit golden-source failure.
    ctx = _GoldenContext(cpu_inputs_per_rank, attrs, rank, ws, dist_avail)
    plugin_result = _try_plugin_golden(api_name, ctx)
    if plugin_result is not None:
        return plugin_result
    logging.warning(
        "[e2e-md rank=%s] no golden plugin result for %s; provide the operator's golden "
        "plugin via --plugin, or set golden_api=disable to skip comparison",
        rank,
        api_name,
    )
    return None


# ACLNN-only params that don't exist in torch_npu API
_ACLNN_ONLY_PARAMS = (
    "transposeX1",
    "transposeX2",
    "group",
    "is_trans_b",
    "network_name",
    "graph_type",
    "seed",
    "isBias",
    "weight_same",
    "hcom",
    "world_size",
    "expPerCard",
    "commTurn",
    "streamMode",
    "gather_index",
    "mm_out_flag",
    "trans_gmm_weight",
    "trans_mm_weight",
    "reduceOp",
    "ep_world_size",
    "is_trans",
    "is_bias",
    "groupEp",
    "groupTp",
    "_tensor_dtypes",
    "alltoAllAxesOptional",
    "all2all_axes_optional",
    "isTrans",
    "is_transX",
    "is_transX2",
    "is_transY",
    "exp_token_nums",
    "ep_ws",
)


def _strip_aclnn_only_params(kw):
    for k in _ACLNN_ONLY_PARAMS:
        kw.pop(k, None)


def _coerce_string_bools(kw):
    # Convert string bool values to actual bool
    for k in ["gather_output"]:
        if k in kw and isinstance(kw[k], str):
            kw[k] = kw[k].lower() in ("true", "1", "yes")


def _call_elastic_buffer_moe(api_name, dev_tensors, hcomm, world_size, kw):
    from cann_ops_transformer.ops.elastic_buffer import ElasticBuffer

    num_experts = int(kw.get("num_experts", 4))
    num_max_tokens = int(kw.get("num_max_tokens_per_rank", dev_tensors[0].shape[0]))
    topk = int(dev_tensors[1].shape[1])
    hidden = int(dev_tensors[0].shape[1])
    topk_ids = (
        torch.arange(dev_tensors[1].numel(), device=dev_tensors[1].device, dtype=torch.int32).reshape(
            dev_tensors[1].shape
        )
        % num_experts
    )
    elastic = ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=num_max_tokens,
        hidden=hidden,
        num_topk=topk,
    )
    try:
        recv_x, _, _, handle = elastic.dispatch(
            dev_tensors[0],
            topk_idx=topk_ids,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens,
            expert_alignment=1,
            do_cpu_sync=False,
        )
        combined_x, _ = elastic.combine(recv_x, handle)
        return combined_x
    finally:
        elastic.destroy()


def _call_mega_moe(api_name, dev_tensors, hcomm, world_size, kw):
    context = kw.pop("_moe_context")
    ccl_buffer_size = int(kw.pop("_moe_ccl_buffer_size"))
    topo_type = int(kw.pop("_moe_topo_type"))
    rank_num_per_server = int(kw.pop("_moe_rank_num_per_server"))
    topk_ids = torch.arange(dev_tensors[1].numel(), device=dev_tensors[1].device, dtype=torch.int32).reshape(
        dev_tensors[1].shape
    ) % int(kw.get("moe_expert_num"))
    args = (
        context,
        dev_tensors[0],
        topk_ids,
        dev_tensors[2],
        [dev_tensors[3]],
        [dev_tensors[4]],
        int(kw.get("moe_expert_num")),
        int(kw.get("ep_world_size", world_size)),
        ccl_buffer_size,
        [dev_tensors[6]],
        [dev_tensors[7]],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        int(
            kw.get(
                "max_recv_token_num",
                kw.get("num_max_tokens_per_rank", dev_tensors[0].shape[0]) * int(kw.get("ep_world_size", world_size)),
            )
        ),
        int(kw.get("dispatch_quant_mode", 4)),
        int(kw.get("combine_quant_mode", 0)),
        str(kw.get("comm_alg", "")),
        int(kw.get("num_max_tokens_per_rank", dev_tensors[0].shape[0])),
        "swiglu",
        [],
        int(kw.get("dispatch_quant_out_dtype", 23)),
        None,
        None,
        topo_type,
        rank_num_per_server,
        0,
    )
    try:
        return _mega_moe_module.npu_mega_moe(*args)
    except TypeError:
        # The older extension omits shared-expert tensors and the final
        # topkWeightsType argument from its Python binding.
        return _mega_moe_module.npu_mega_moe(
            context,
            dev_tensors[0],
            topk_ids,
            dev_tensors[2],
            [dev_tensors[3]],
            [dev_tensors[4]],
            int(kw.get("moe_expert_num")),
            int(kw.get("ep_world_size", world_size)),
            ccl_buffer_size,
            [dev_tensors[6]],
            [dev_tensors[7]],
            None,
            None,
            None,
            int(
                kw.get(
                    "max_recv_token_num",
                    kw.get("num_max_tokens_per_rank", dev_tensors[0].shape[0])
                    * int(kw.get("ep_world_size", world_size)),
                )
            ),
            int(kw.get("dispatch_quant_mode", 4)),
            int(kw.get("combine_quant_mode", 0)),
            str(kw.get("comm_alg", "")),
            int(kw.get("num_max_tokens_per_rank", dev_tensors[0].shape[0])),
            "swiglu",
            None,
            int(kw.get("dispatch_quant_out_dtype", 23)),
            topo_type,
            rank_num_per_server,
        )


def _call_private_moe_dispatch(api_name, dev_tensors, hcomm, world_size, kw):
    context = kw.pop("_moe_context")
    ccl_buffer_size = int(kw.pop("_moe_ccl_buffer_size"))
    ep_rank_id = int(kw.pop("_moe_rank"))
    ep_world_size = int(kw.pop("ep_world_size", world_size))
    moe_expert_num = int(kw.pop("moe_expert_num"))
    return _moe_dispatch_module.npu_moe_distribute_dispatch(
        context,
        dev_tensors[1],
        dev_tensors[2],
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        dev_tensors[3],
        dev_tensors[4],
        dev_tensors[5],
        dev_tensors[6],
        dev_tensors[7],
        int(kw.get("tp_world_size", 0)),
        int(kw.get("tp_rank_id", 0)),
        int(kw.get("expert_shard_type", 0)),
        int(kw.get("shared_expert_num", 1)),
        int(kw.get("shared_expert_rank_num", 0)),
        int(kw.get("quant_mode", 0)),
        int(kw.get("global_bs", 0)),
        int(kw.get("expert_token_nums_type", 1)),
        str(kw.get("comm_alg", "")),
        int(kw.get("zero_expert_num", 0)),
        int(kw.get("copy_expert_num", 0)),
        int(kw.get("const_expert_num", 0)),
        kw.get("y_dtype"),
        kw.get("x_dtype"),
        kw.get("scales_dtype"),
    )


def _call_private_moe_combine(api_name, dev_tensors, hcomm, world_size, kw):
    context = kw.pop("_moe_context")
    ccl_buffer_size = int(kw.pop("_moe_ccl_buffer_size"))
    ep_rank_id = int(kw.pop("_moe_rank"))
    ep_world_size = int(kw.pop("ep_world_size", world_size))
    moe_expert_num = int(kw.pop("moe_expert_num"))
    expert_ids = dev_tensors[2]
    expert_scales = dev_tensors[5]
    dispatch_x = dev_tensors[1][: expert_ids.shape[0]].contiguous()
    dispatch = _moe_dispatch_module.npu_moe_distribute_dispatch(
        context,
        dispatch_x,
        expert_ids,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        None,
        None,
        expert_scales,
        None,
        None,
        0,
        0,
        int(kw.get("expert_shard_type", 0)),
        0,
        0,
        0,
        int(kw.get("global_bs", 0)),
        int(kw.pop("expert_token_nums_type", 0)),
        str(kw.get("comm_alg", "")),
        0,
        0,
        0,
        None,
        None,
        None,
    )
    expand_x, _, assist_info, _, ep_send_counts, tp_send_counts, expand_scales = dispatch
    return _moe_combine_module.npu_moe_distribute_combine(
        context,
        expand_x,
        expert_ids,
        assist_info,
        ep_send_counts,
        expert_scales,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        tp_send_counts,
        dev_tensors[7],
        expand_scales,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        int(kw.get("tp_world_size", 0)),
        int(kw.get("tp_rank_id", 0)),
        int(kw.get("expert_shard_type", 0)),
        int(kw.get("shared_expert_num", 1)),
        int(kw.get("shared_expert_rank_num", 0)),
        int(kw.get("global_bs", 0)),
        int(kw.get("comm_quant_mode", 0)),
        str(kw.get("comm_alg", "")),
        int(kw.get("zero_expert_num", 0)),
        int(kw.get("copy_expert_num", 0)),
        int(kw.get("const_expert_num", 0)),
    )


def _call_quant_gmm_alltoallv(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    send_counts = kw.pop("sendCounts", kw.pop("send_counts", []))
    recv_counts = kw.pop("recvCounts", kw.pop("recv_counts", []))
    ep_ws = int(kw.pop("ep_world_size", kw.pop("epWorldSize", world_size)))
    gmm_y_dtype = int(kw.pop("gmm_y_dtype", kw.pop("gmmYType", 1)))
    gmm_x_quant_mode = kw.pop("gmm_x_quant_mode", None)
    gmm_weight_quant_mode = kw.pop("gmm_weight_quant_mode", None)
    # ACLNN exposes comm_mode, but the torch_npu quantized E2E schema does
    # not.  Keep the attribute in ACLNN cases without forwarding it here.
    kw.pop("comm_mode", None)
    kw.pop("epWorldSize", None)
    args = (
        dev_tensors[0],
        dev_tensors[1],
        dev_tensors[2],
        dev_tensors[3],
        hcomm,
        ep_ws,
        send_counts,
        recv_counts,
        gmm_y_dtype,
    )
    if "npu_alltoallv_quant_gmm" in api_name:
        return resolved(*args, gmm_x_quant_mode=gmm_x_quant_mode, gmm_weight_quant_mode=gmm_weight_quant_mode)
    return resolved(*args, gmm_x_quant_mode=gmm_x_quant_mode, gmm_weight_quant_mode=gmm_weight_quant_mode)


def _call_all_gather_base_mm(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)


def _call_all_gather_quant_mm(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    kw.pop("comm_mode", None)
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)


def _call_mm_reduce_scatter_base(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)


def _call_mm_all_reduce_add_rms_norm(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    kw.pop("world_size", None)
    reduce_op = kw.pop("reduce_op", "sum")
    epsilon = kw.pop("epsilon", 1e-6)
    return resolved(
        dev_tensors[0],
        dev_tensors[1],
        dev_tensors[2],
        dev_tensors[3],
        hcomm,
        reduce_op=reduce_op,
        epsilon=epsilon,
        **kw,
    )


def _call_mm_all_reduce_base(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    bias = dev_tensors[2] if len(dev_tensors) > 2 and dev_tensors[2] is not None else None
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, bias=bias)


def _call_matmul_all_to_all(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    # Match mc2_test: explicitly pass bias=None and all2all_axes=None
    kw.setdefault("bias", None)
    kw.setdefault("all2all_axes", None)
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)


def _call_all_to_all_matmul(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    kw.setdefault("bias", None)
    kw.setdefault("all2all_axes", None)
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)


def _call_all_to_all_quant_matmul(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)


def _call_gmm_alltoallv(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    send_counts = kw.pop("sendCounts", kw.pop("send_counts", []))
    recv_counts = kw.pop("recvCounts", kw.pop("recv_counts", []))
    ep_ws = int(kw.pop("ep_world_size", kw.pop("epWorldSize", world_size)))
    kw.pop("epWorldSize", None)
    if "transGmmWeight" in kw:
        kw["trans_gmm_weight"] = kw.pop("transGmmWeight")
    if "transMmWeight" in kw:
        kw["trans_mm_weight"] = kw.pop("transMmWeight")
    # dev_tensors: [gmm_x, gmm_weight, (mm_x, mm_weight) if mm_out]
    gmm_x = dev_tensors[0]
    gmm_weight = dev_tensors[1]
    mm_x = dev_tensors[2] if len(dev_tensors) > 2 else None
    mm_weight = dev_tensors[3] if len(dev_tensors) > 3 else None
    return resolved(
        gmm_x=gmm_x,
        gmm_weight=gmm_weight,
        ep_world_size=ep_ws,
        hcom=hcomm,
        send_counts=send_counts,
        recv_counts=recv_counts,
        send_counts_tensor=None,
        recv_counts_tensor=None,
        mm_x=mm_x,
        mm_weight=mm_weight,
        **kw,
    )


def _call_alltoallv_gmm(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    send_counts = kw.pop("sendCounts", kw.pop("send_counts", []))
    recv_counts = kw.pop("recvCounts", kw.pop("recv_counts", []))
    ep_ws = int(kw.pop("ep_world_size", kw.pop("epWorldSize", world_size)))
    kw.pop("epWorldSize", None)
    if "transGmmWeight" in kw:
        kw["trans_gmm_weight"] = kw.pop("transGmmWeight")
    if "transMmWeight" in kw:
        kw["trans_mm_weight"] = kw.pop("transMmWeight")
    if "permuteOutFlag" in kw:
        kw["permute_out_flag"] = kw.pop("permuteOutFlag")
    gmm_x = dev_tensors[0]
    gmm_weight = dev_tensors[1]
    mm_x = dev_tensors[2] if len(dev_tensors) > 2 else None
    mm_weight = dev_tensors[3] if len(dev_tensors) > 3 else None
    return resolved(
        gmm_x=gmm_x,
        gmm_weight=gmm_weight,
        ep_world_size=ep_ws,
        hcom=hcomm,
        send_counts=send_counts,
        recv_counts=recv_counts,
        send_counts_tensor=None,
        recv_counts_tensor=None,
        mm_x=mm_x,
        mm_weight=mm_weight,
        **kw,
    )


def _call_bmm_reducescatter_alltoall(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    ep_ws = int(kw.pop("group_ep_worldsize", kw.pop("epWorldSize", 0)))
    tp_ws = int(kw.pop("group_tp_worldsize", kw.pop("tpWorldSize", 0)))
    ep_hc = kw.pop("ep_hcomm", hcomm)
    tp_hc = kw.pop("tp_hcomm", hcomm)
    shard = int(kw.pop("shard_type", kw.pop("yShardType", 0)))
    is_bias = bool(kw.pop("isBias", False))
    bias = dev_tensors[2] if (is_bias and len(dev_tensors) > 2) else None
    # Drop all remaining attrs not in API schema
    kw.clear()
    return resolved(dev_tensors[0], dev_tensors[1], ep_hc, ep_ws, tp_hc, tp_ws, bias=bias, shard_type=shard)


def _call_alltoall_allgather_bmm(resolved, api_name, dev_tensors, hcomm, world_size, kw):
    ep_ws = int(kw.pop("group_ep_worldsize", kw.pop("epWorldSize", 0)))
    tp_ws = int(kw.pop("group_tp_worldsize", kw.pop("tpWorldSize", 0)))
    ep_hc = kw.pop("ep_hcomm", hcomm)
    tp_hc = kw.pop("tp_hcomm", hcomm)
    act = kw.pop("act_type", kw.pop("actType", "none"))
    if isinstance(act, int):
        act_map = {0: "none", 1: "gelu", 2: "silu", 3: "relu", 4: "fastgelu"}
        act = act_map.get(act, "none")
    shard = int(kw.pop("shard_type", kw.pop("xShardType", 0)))
    need_ag = bool(kw.pop("need_allgather_out", kw.pop("needAllgatherOut", False)))
    need_act = bool(kw.pop("need_activation_feature", kw.pop("needActivationFeature", False)))
    # Drop attrs not in API schema
    for k in [
        "isBias",
        "isTrans",
        "epWorldSize",
        "tpWorldSize",
        "xShardType",
        "actType",
        "needAllgatherOut",
        "needActivationFeature",
    ]:
        kw.pop(k, None)
    bias = dev_tensors[2] if len(dev_tensors) > 2 else None
    return resolved(
        dev_tensors[0],
        dev_tensors[1],
        ep_hc,
        ep_ws,
        tp_hc,
        tp_ws,
        bias=bias,
        shard_type=shard,
        act_type=act,
        need_allgather_out=need_ag,
        need_activation_feature=need_act,
    )


# Substring pattern → call adapter. Patterns are mutually exclusive; order
# mirrors the historical if-elif chain. Ops without a matching pattern fall
# through to the generic resolved(*dev_tensors, hcom=..., world_size=...) call.
_MC2_API_ADAPTERS = (
    ("npu_all_gather_base_mm", _call_all_gather_base_mm),
    ("npu_all_gather_quant_mm", _call_all_gather_quant_mm),
    ("npu_mm_reduce_scatter_base", _call_mm_reduce_scatter_base),
    ("npu_quant_mm_reduce_scatter", _call_mm_reduce_scatter_base),
    ("npu_mm_all_reduce_add_rms_norm", _call_mm_all_reduce_add_rms_norm),
    ("npu_mm_all_reduce_base", _call_mm_all_reduce_base),
    ("npu_matmul_all_to_all", _call_matmul_all_to_all),
    ("npu_all_to_all_matmul", _call_all_to_all_matmul),
    ("npu_all_to_all_quant_matmul", _call_all_to_all_quant_matmul),
    ("npu_gmm_alltoallv", _call_gmm_alltoallv),
    ("npu_alltoallv_gmm", _call_alltoallv_gmm),
    ("bmm_reducescatter_alltoall", _call_bmm_reducescatter_alltoall),
    ("alltoall_allgather_bmm", _call_alltoall_allgather_bmm),
)


def _call_api(api_name, dev_tensors, hcomm, world_size, attrs):
    """Call the E2E API via per-operator adapters.

    Private-MoE / torch_npu MoE conventions first, then generic resolve;
    quant-GMM alltoallv ops take a dedicated pre-adapter branch, other APIs
    match against the ``_MC2_API_ADAPTERS`` substring table; APIs matching
    no pattern fall through to ``resolved(*dev_tensors, hcom=..., world_size=...)``.
    """
    kw = dict(attrs)

    if _uses_private_moe_backend(api_name) and "elastic_buffer_moe_ep_chain" in api_name:
        return _call_elastic_buffer_moe(api_name, dev_tensors, hcomm, world_size, kw)

    if _uses_private_moe_backend(api_name) and "mega_moe" in api_name:
        return _call_mega_moe(api_name, dev_tensors, hcomm, world_size, kw)

    if _uses_private_moe_backend(api_name) and "npu_moe_distribute_dispatch" in api_name:
        return _call_private_moe_dispatch(api_name, dev_tensors, hcomm, world_size, kw)

    if _uses_private_moe_backend(api_name) and "npu_moe_distribute_combine" in api_name:
        return _call_private_moe_combine(api_name, dev_tensors, hcomm, world_size, kw)

    if api_name in _TORCH_NPU_MOE_TENSOR_PARAMS:
        return _call_torch_npu_moe(api_name, dev_tensors, hcomm, world_size, attrs)
    if api_name == "torch_npu._npu_distribute_barrier":
        resolved = _resolve_api(api_name)
        return resolved(dev_tensors[0], hcomm, world_size)
    if api_name == "torch_npu.npu_moe_update_expert":
        resolved = _resolve_api(api_name)
        kw = dict(attrs)
        kw.pop("_tensor_dtypes", None)
        if kw.get("local_rank_id", -1) < 0:
            kw["local_rank_id"] = int(kw.pop("_rank", 0))
        return resolved(*dev_tensors, **kw)

    resolved = _resolve_api(api_name)

    if "npu_quant_gmm_alltoallv" in api_name or "npu_alltoallv_quant_gmm" in api_name:
        return _call_quant_gmm_alltoallv(resolved, api_name, dev_tensors, hcomm, world_size, kw)

    _strip_aclnn_only_params(kw)
    _coerce_string_bools(kw)

    for pattern, adapter in _MC2_API_ADAPTERS:
        if pattern in api_name:
            return adapter(resolved, api_name, dev_tensors, hcomm, world_size, kw)

    kw["hcom"] = hcomm
    kw["world_size"] = world_size
    return resolved(*dev_tensors, **kw)


class _MC2GraphModel(torch.nn.Module):
    """nn.Module wrapper for HCCL communication-fusion ops, mirroring mc2_test's
    MatmulAlltoAllGraphModel pattern. forward delegates to _call_api so all
    HCCL ops (matmul_all_to_all, all_gather_mm, gmm_alltoallv, ...) are supported."""

    def __init__(self, api_name, hcomm, world_size, attrs):
        super().__init__()
        self._api_name = api_name
        self._hcomm = hcomm
        self._world_size = world_size
        self._attrs = attrs

    def forward(self, *dev_tensors):
        return _call_api(self._api_name, list(dev_tensors), self._hcomm, self._world_size, self._attrs)


class _AlltoAllvGmmGraphModel(torch.nn.Module):
    """Dynamo-friendly GraphModel for npu_alltoallv_gmm / npu_gmm_alltoallv.

    dynamo cannot trace _MC2GraphModel because:
    (1) forward(*dev_tensors) uses *args (variable positional)
    (2) _call_api resolves the op via a big if-elif and calls with kwargs
        like resolved(gmm_x=..., gmm_weight=...) which dynamo mis-traces.
    (3) dynamo converts self._send_counts (list[int]) to immutable_list,
        but the C++ op requires List[int]. Passing as forward arg makes
        dynamo treat it as a constant (not traced).
    This model mirrors mc2_test's AlltoAllvGroupedMatmulGraphModel: explicit
    positional forward signature, direct torch_npu call inside.
    """

    def __init__(self, api_name):
        super().__init__()
        self._api_name = api_name

    def forward(
        self,
        gmm_x,
        gmm_weight,
        hcom,
        ep_ws,
        send_counts,
        recv_counts,
        trans_gmm,
        trans_mm,
        permute,
        mm_x=None,
        mm_weight=None,
    ):
        is_v2 = "gmm_alltoallv" in self._api_name
        fn = torch_npu.npu_gmm_alltoallv if is_v2 else torch_npu.npu_alltoallv_gmm
        send_counts = [int(x) for x in send_counts]
        recv_counts = [int(x) for x in recv_counts]
        if is_v2:
            return fn(
                gmm_x,
                gmm_weight,
                hcom,
                ep_ws,
                send_counts,
                recv_counts,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                mm_x=mm_x,
                mm_weight=mm_weight,
                trans_gmm_weight=trans_gmm,
                trans_mm_weight=trans_mm,
            )
        return fn(
            gmm_x,
            gmm_weight,
            hcom,
            ep_ws,
            send_counts,
            recv_counts,
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=mm_x,
            mm_weight=mm_weight,
            trans_gmm_weight=trans_gmm,
            trans_mm_weight=trans_mm,
            permute_out_flag=permute,
        )


def _run_hccl_graph(api_name, dev_tensors, hcomm, world_size, attrs, graph_mode):
    """Run HCCL op in GE graph mode via torch.compile + torchair backend.

    Mirrors mc2_test's define_model(graph_type=2): torch.compile(model,
    backend=npu_backend, dynamic=True/False). Returns NPU output tensor
    (same shape/dtype as eager) or raises.

    For npu_alltoallv_gmm / npu_gmm_alltoallv, uses _AlltoAllvGmmGraphModel
    (dynamo-friendly explicit forward signature) instead of the generic
    _MC2GraphModel, because dynamo cannot trace *args + kwarg call inside
    _call_api.
    """
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    is_a2av_gmm = "npu_alltoallv_gmm" in api_name or "npu_gmm_alltoallv" in api_name
    if is_a2av_gmm:
        model = _AlltoAllvGmmGraphModel(api_name)
        invoke = _build_a2av_gmm_invoke(dev_tensors, hcomm, world_size, attrs)
    else:
        model = _MC2GraphModel(api_name, hcomm, world_size, attrs)
        invoke = dev_tensors
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    dynamic = graph_mode == "dynamic"
    compiled = torch.compile(model, backend=npu_backend, dynamic=dynamic)
    result = compiled(*invoke)
    torch.npu.synchronize()
    return result


def _build_a2av_gmm_invoke(dev_tensors, hcomm, world_size, attrs):
    """Build positional invoke args for _AlltoAllvGmmGraphModel.forward.
    forward(gmm_x, gmm_weight, hcom, ep_ws, send_counts, recv_counts,
            trans_gmm, trans_mm, permute, mm_x=None, mm_weight=None).
    Non-tensor args are passed as forward params so dynamo treats them as
    constants (avoids immutable_list conversion of self attributes)."""
    gmm_x = dev_tensors[0]
    gmm_weight = dev_tensors[1]
    mm_x = dev_tensors[2] if len(dev_tensors) > 2 else None
    mm_weight = dev_tensors[3] if len(dev_tensors) > 3 else None
    ep_ws = int(attrs.get("ep_world_size", attrs.get("epWorldSize", world_size)))
    send_counts = list(attrs.get("send_counts", attrs.get("sendCounts", [])))
    recv_counts = list(attrs.get("recv_counts", attrs.get("recvCounts", [])))
    trans_gmm = bool(attrs.get("trans_gmm_weight", attrs.get("transGmmWeight", False)))
    trans_mm = bool(attrs.get("trans_mm_weight", attrs.get("transMmWeight", False)))
    permute = bool(attrs.get("permute_out_flag", attrs.get("permuteOutFlag", False)))
    return [gmm_x, gmm_weight, hcomm, ep_ws, send_counts, recv_counts, trans_gmm, trans_mm, permute, mm_x, mm_weight]


def _is_dual_group_op(api_name):
    return "bmm_reducescatter_alltoall" in api_name or "alltoall_allgather_bmm" in api_name


def _is_gmm_op(api_name):
    return "npu_gmm_alltoallv" in api_name or "npu_alltoallv_gmm" in api_name


def _needs_private_moe(api_name):
    return _uses_private_moe_backend(api_name) and ("npu_moe_distribute_" in api_name or "mega_moe" in api_name)


def _load_plan(plan_path):
    with open(plan_path) as _pf:
        plan = json.load(_pf)
    return plan, int(plan.get("proc_timeout", 3600)), int(plan.get("deterministic_level", 0) or 0)


def _init_distributed(rank, world_size, port, timeout, deterministic_level, device_id):
    torch.npu.set_device(device_id)
    if deterministic_level:
        torch_npu.npu.set_deterministic_level(deterministic_level)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=datetime.timedelta(seconds=timeout),
    )
    return str(dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank))


def _parse_plan(plan):
    """Extract worker-relevant fields and fold tensor_dtypes into attrs."""
    api_name = plan["api_name"]
    attrs = plan.get("attributes", {})
    golden_disabled = bool(plan.get("golden_disabled", False))
    remark = plan.get("remark", "")
    # Pass tensor_dtypes to attrs so golden functions can simulate NPU dtype precision
    tensor_dtypes = plan.get("tensor_dtypes", [])
    if tensor_dtypes:
        attrs = dict(attrs)
        attrs["_tensor_dtypes"] = tensor_dtypes
    return api_name, attrs, golden_disabled, remark, tensor_dtypes


def _create_moe_group(api_name, rank, world_size):
    """Create the EP-only communication domain for private MoE ops.

    WORLD is retained for TTK barriers/result gathering and must not be reused.
    """
    if not _needs_private_moe(api_name):
        return None
    moe_group = dist.new_group(ranks=list(range(world_size)), backend="hccl")
    return str(moe_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=True))


def _setup_dual_ep_tp_groups(api_name, rank, world_size, attrs):
    """Create EP/TP sub-groups for dual-domain ops and expose their hccl names."""
    if not _is_dual_group_op(api_name):
        return attrs
    dist.barrier()
    ep_ws = int(attrs.get("epWorldSize", 0))
    tp_ws = int(attrs.get("tpWorldSize", 0))
    if not (ep_ws and tp_ws):
        return attrs
    # EP/TP group layout matching mc2_test reference:
    # EP groups: consecutive ranks [i*ep_ws:(i+1)*ep_ws] for i in range(tp_ws)
    # TP groups: strided ranks [i, i+ep_ws, i+2*ep_ws, ...] for i in range(ep_ws)
    ep_groups = []
    for i in range(tp_ws):
        ep_ranks = [x + ep_ws * i for x in range(ep_ws)]
        ep_groups.append(dist.new_group(backend="hccl", ranks=ep_ranks))
    tp_groups = []
    for i in range(ep_ws):
        tp_ranks = [x * ep_ws + i for x in range(tp_ws)]
        tp_groups.append(dist.new_group(backend="hccl", ranks=tp_ranks))
    ep_rank = rank % ep_ws
    tp_rank = rank // ep_ws
    attrs = dict(attrs)
    attrs["ep_hcomm"] = str(ep_groups[tp_rank]._get_backend(torch.device("npu")).get_hccl_comm_name(rank))
    attrs["tp_hcomm"] = str(tp_groups[ep_rank]._get_backend(torch.device("npu")).get_hccl_comm_name(rank))
    return attrs


def _load_rank_inputs(input_path, rank):
    """Load this rank's inputs from npz (keyed by inp_{rank}_{idx})."""
    data = np.load(input_path, allow_pickle=False)
    my_prefix = f"inp_{rank}_"
    my_input_keys = sorted([k for k in data.files if k.startswith(my_prefix)], key=lambda x: int(x.split("_")[2]))
    # Determine which indices we need. For shared weight ops, missing idx>=1
    # tensors are filled from rank 0 (shared weight only saved once).
    my_indices = [int(k.split("_")[2]) for k in my_input_keys]
    if my_indices:
        rank0_keys = sorted([k for k in data.files if k.startswith("inp_0_")], key=lambda x: int(x.split("_")[2]))
        rank0_by_idx = {int(k.split("_")[2]): k for k in rank0_keys}
        max_idx = max(*my_indices, max(rank0_by_idx.keys()) if rank0_by_idx else 0)
        for idx in range(max_idx + 1):
            if idx not in my_indices and idx in rank0_by_idx:
                my_input_keys.append(rank0_by_idx[idx])
        my_input_keys.sort(key=lambda x: int(x.split("_")[2]))
    return [data[k] for k in my_input_keys]


def _flat_declared_dtype(tensor_dtypes, index):
    flat = []
    for item in tensor_dtypes:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat[index] if index < len(flat) else None


def _np_to_torch(np_arr, tensor_dtypes, index):
    from ml_dtypes import bfloat16 as np_bf16

    if np_arr.dtype == np_bf16 or (
        hasattr(np_arr.dtype, "itemsize") and np_arr.dtype.itemsize == 2 and np_arr.dtype.kind == "V"
    ):
        t = torch.from_numpy(np_arr.view(np.uint16)).view(torch.bfloat16)
        return t.reshape(np_arr.shape)
    declared_dtype = _flat_declared_dtype(tensor_dtypes, index)
    if declared_dtype in ("float8_e5m2", "float8_e4m3fn"):
        return torch.from_numpy(np_arr.view(np.uint8)).view(getattr(torch, declared_dtype))
    if declared_dtype in ("float8_e8m0", "float8_e8m0fnu"):
        return torch.from_numpy(np_arr.view(np.uint8)).view(torch.float8_e8m0fnu)
    return torch.from_numpy(np_arr)


def _materialize_inputs(my_inputs_np, tensor_dtypes, device_id):
    """CPU copy (float32 for golden) and NPU copy (original dtype for API call)."""

    def is_none_input(value):
        return value.dtype == object and value.shape == () and value.item() is None

    cpu_inputs_for_rank = [
        None if is_none_input(x) else _np_to_torch(x.copy(), tensor_dtypes, i).float()
        for i, x in enumerate(my_inputs_np)
    ]
    dev_tensors = [
        None if is_none_input(x) else _np_to_torch(x.copy(), tensor_dtypes, i).npu(device_id)
        for i, x in enumerate(my_inputs_np)
    ]
    return cpu_inputs_for_rank, dev_tensors


def _setup_moe_attrs(api_name, rank, world_size, moe_hcomm, attrs):
    """Create the private MoE context and stash its handle in attrs."""
    if not _needs_private_moe(api_name):
        return attrs
    custom_ccl_buffer_size = int(os.environ.get("HCCL_BUFFSIZE", "0")) * 2 * 1024 * 1024
    try:
        moe_context_manager = _moe_context_manager_cls(
            moe_hcomm,
            world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            customCclBufferSize=custom_ccl_buffer_size,
        )
    except TypeError:
        # Older installed comm_context extensions only accept a string
        # backend and do not expose customCclBufferSize.
        moe_context_manager = _moe_context_manager_cls(moe_hcomm, world_size, backend="kfc")
    attrs = dict(attrs)
    attrs["_moe_context"] = moe_context_manager.create_context()
    attrs["_moe_ccl_buffer_size"] = moe_context_manager.ccl_buffer_size
    attrs["_moe_topo_type"] = getattr(moe_context_manager, "topo_type", 0)
    attrs["_moe_rank_num_per_server"] = getattr(moe_context_manager, "rank_num_per_server", 0)
    attrs["_moe_rank"] = rank
    return attrs


def _apply_transposes(attrs, dev_tensors):
    if attrs.get("transposeX2", False) or attrs.get("is_trans_b", False):
        dev_tensors[1] = dev_tensors[1].t().contiguous()
    if attrs.get("transposeX1", False):
        dev_tensors[0] = dev_tensors[0].t().contiguous()


def _patch_gmm_attrs(api_name, attrs, dev_tensors, remark, rank, world_size):
    """Generate sendCounts/recvCounts matching ACLNN __patch_gmm_rank_attributes
    (seed comes from remark (e.g. "seed=1"), ep_ws from attrs)."""
    if not _is_gmm_op(api_name):
        return attrs
    seed_val = 1
    for part in (remark or "").split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "seed":
            with contextlib.suppress(ValueError):
                seed_val = int(kv[1].strip())
    ep_ws = int(attrs.get("epWorldSize", world_size))
    epc = int(dev_tensors[1].shape[0]) if len(dev_tensors) > 1 else 1
    M_per_rank = int(dev_tensors[0].shape[0]) if len(dev_tensors) > 0 else 0
    A_array = [M_per_rank] * ep_ws
    from ttk.core_modules.common.xpu import generate_gmm_alltoallv_matrix as _generate_gmm_alltoallv_matrix

    exp_token_nums = _generate_gmm_alltoallv_matrix(A_array, epc, seed=seed_val)
    attrs = dict(attrs)
    if "npu_gmm_alltoallv" in api_name:
        recv_counts = list(exp_token_nums[rank])
        send_counts = []
        for i in range(ep_ws):
            send_counts.extend(exp_token_nums[i][rank * epc : (rank + 1) * epc])
    else:
        send_counts = list(exp_token_nums[rank])
        recv_counts = []
        for i in range(ep_ws):
            recv_counts.extend(exp_token_nums[i][rank * epc : (rank + 1) * epc])
    attrs["sendCounts"] = send_counts
    attrs["recvCounts"] = recv_counts
    attrs["exp_token_nums"] = exp_token_nums
    attrs["expPerCard"] = epc
    attrs["ep_ws"] = ep_ws
    attrs["seed"] = seed_val
    return attrs


def _save_rank_status(rank, world_size, result_path, per_rank_status, passed):
    """Gather per-rank status strings and have rank 0 write the combined result."""
    all_prec_strings = [None] * world_size
    dist.all_gather_object(all_prec_strings, per_rank_status)
    if rank == 0:
        np.savez(
            result_path,
            precision_0=np.array(",".join(all_prec_strings)),
            pass_0=np.array(passed),
        )
    dist.barrier()
    dist.destroy_process_group()


def _save_executed_pass(rank, world_size, result_path):
    """Gather EXECUTED status from all ranks and have rank 0 write a PASS result."""
    _save_rank_status(rank, world_size, result_path, f"rank{rank}:PASS(EXECUTED)", "PASS")


def _cross_rank_mask(api_name, num_tensors):
    """Which of this rank's inputs golden needs from other ranks.

    Default: only x (idx 0), weight is shared (own copy suffices).
    """
    needs_cross_rank = [True] + [False] * (num_tensors - 1)
    if "npu_moe_distribute_dispatch" in api_name:
        # MoE dispatch golden needs all ranks' x AND expert_ids (both per-rank)
        needs_cross_rank = [True] * min(num_tensors, 2)
    if "npu_moe_distribute_combine" in api_name:
        # MoE combine golden needs all ranks' expand_x AND expert_ids (both per-rank)
        needs_cross_rank = [True] * num_tensors
    if _is_gmm_op(api_name):
        # GMM golden needs all ranks' gmm_x AND gmm_weight
        needs_cross_rank = [True] * num_tensors
    # MatmulAlltoAll / AlltoAllMatmul: x2 is NOT shared (per-rank different)
    if "npu_matmul_all_to_all" in api_name or "npu_all_to_all_matmul" in api_name:
        needs_cross_rank = [True] * min(num_tensors, 2)
    return needs_cross_rank


def _share_inputs_via_files(rank, world_size, result_path, cpu_inputs_for_rank, cpu_inputs_per_rank):
    """Share per-rank inputs via files.

    BMM_RS_A2A: weight is per-rank different (NOT shared) and too large for
    dist.all_gather (8 ranks * 2.68GB = 21GB > NPU memory). Golden needs all
    ranks' x AND weight, so share via per-rank files instead.
    """
    t0 = time.time()
    rank_file = os.path.join(os.path.dirname(result_path), f"ttk_e2e_md_rank_{rank}.pt")
    torch.save(cpu_inputs_for_rank, rank_file)
    logging.info("[BMM_IO rank=%s] saved t=%.1fs", rank, time.time() - t0)
    dist.barrier()
    logging.info("[BMM_IO rank=%s] barrier1 t=%.1fs", rank, time.time() - t0)
    for r in range(world_size):
        if r == rank:
            continue
        peer_file = os.path.join(os.path.dirname(result_path), f"ttk_e2e_md_rank_{r}.pt")
        cpu_inputs_per_rank[r] = torch.load(peer_file, weights_only=True)
        logging.info("[BMM_IO rank=%s] loaded r=%s t=%.1fs", rank, r, time.time() - t0)
    dist.barrier()
    logging.info("[BMM_IO rank=%s] barrier2 t=%.1fs", rank, time.time() - t0)
    with contextlib.suppress(Exception):
        os.remove(rank_file)


def _gather_cross_rank_inputs(api_name, rank, world_size, result_path, cpu_inputs_for_rank, device_id):
    """Gather all ranks' CPU inputs to every rank via dist (data transport only, NOT golden).

    Use torch_npu dist all_gather to move CPU inputs across cards.
    OPTIMIZATION: Only gather tensors that golden actually needs from other ranks.
    For MC2 matmul ops, weight (idx 1+) is per-rank/shared, only x (idx 0) needs cross-rank.
    For BMM_RS_A2A / A2A_AG_BMM: golden needs all ranks' x but only own weight.
    This avoids OOM when weight is large (e.g. 16x8192x2560 ~0.5GB * 8 ranks = 4GB).
    """
    cpu_inputs_per_rank = [None] * world_size
    cpu_inputs_per_rank[rank] = cpu_inputs_for_rank
    needs_cross_rank = _cross_rank_mask(api_name, len(cpu_inputs_for_rank))
    if "bmm_reducescatter_alltoall" in api_name:
        _share_inputs_via_files(rank, world_size, result_path, cpu_inputs_for_rank, cpu_inputs_per_rank)
        return cpu_inputs_per_rank
    for tensor_idx in range(len(cpu_inputs_for_rank)):
        if not needs_cross_rank[tensor_idx]:
            continue
        t_cpu = cpu_inputs_for_rank[tensor_idx]
        # Move to NPU for all_gather transport, then back to CPU for golden
        t_npu = t_cpu.npu(device_id)
        gather_list = [torch.empty_like(t_npu) for _ in range(world_size)]
        dist.all_gather(gather_list, t_npu)
        for r in range(world_size):
            if r == rank:
                continue
            if cpu_inputs_per_rank[r] is None:
                cpu_inputs_per_rank[r] = [None] * len(cpu_inputs_for_rank)
            cpu_inputs_per_rank[r][tensor_idx] = gather_list[r].cpu().float()
        del t_npu, gather_list

    # Fill in other ranks' non-gathered tensors with own copy (shared weight assumption)
    for r in range(world_size):
        if r == rank:
            continue
        if cpu_inputs_per_rank[r] is None:
            cpu_inputs_per_rank[r] = list(cpu_inputs_for_rank)
        else:
            for idx in range(len(cpu_inputs_for_rank)):
                if cpu_inputs_per_rank[r][idx] is None:
                    cpu_inputs_per_rank[r][idx] = cpu_inputs_for_rank[idx]
    return cpu_inputs_per_rank


def _extract_golden_outputs(result, golden):
    """Handle multi-output golden (e.g. GMM returns (main, mm) tuple, A2A_AG_BMM returns dict)."""
    npu_mm_out = None
    if isinstance(result, (tuple, list)) and len(result) > 1:
        npu_mm_out = result[1]
    if isinstance(golden, dict):
        # A2A_AG_BMM: golden is {'main':..., 'allgather':..., 'bmm':...}
        # NPU result is (main, allgather, bmm) tuple
        golden = golden.get("main")
        if isinstance(result, (tuple, list)) and len(result) > 1:
            npu_mm_out = result[1]  # allgather
            # bmm is result[2] if present
    if isinstance(golden, (tuple, list)):
        golden_mm = golden[1] if len(golden) > 1 else None
        golden = golden[0]
    else:
        golden_mm = None
    return golden, golden_mm, npu_mm_out


def _cast_golden(golden, golden_mm, npu_out, npu_mm_out):
    """Cast golden to NPU output dtype (matching ACLNN: golden.to(output_dtype))."""
    golden_t = None
    if golden is not None and isinstance(golden, torch.Tensor):
        npu_dtype = npu_out.dtype
        golden_t = golden.to(npu_dtype)

    golden_mm_t = None
    if golden_mm is not None and isinstance(golden_mm, torch.Tensor) and npu_mm_out is not None:
        golden_mm_t = golden_mm.to(npu_mm_out.dtype)
    return golden_t, golden_mm_t


def _measure_precision(npu_out, golden_t, npu_mm_out, golden_mm_t, rtol, atol, ptol):
    """Each rank compares its own NPU output vs its own CPU golden using torch.isclose."""
    npu_flat = npu_out.cpu().contiguous().view(-1)
    gold_flat = golden_t.cpu().contiguous().view(-1)
    close = torch.isclose(npu_flat, gold_flat, rtol=rtol, atol=atol, equal_nan=True)
    precision = float(close.sum().item()) / float(close.numel())
    is_pass = (1.0 - precision) <= ptol

    # Also compare mm_out if present
    if golden_mm_t is not None and npu_mm_out is not None:
        npu_mm_flat = npu_mm_out.cpu().contiguous().view(-1)
        gold_mm_flat = golden_mm_t.cpu().contiguous().view(-1)
        close_mm = torch.isclose(npu_mm_flat, gold_mm_flat, rtol=rtol, atol=atol, equal_nan=True)
        precision_mm = float(close_mm.sum().item()) / float(close_mm.numel())
        is_pass_mm = (1.0 - precision_mm) <= ptol
        # Combined precision: average of both outputs
        precision = (precision + precision_mm) / 2
        is_pass = is_pass and is_pass_mm
    return precision, is_pass


def _report_and_save(rank, world_size, result_path, golden_t, precision, is_pass):
    """Gather precision from all ranks; rank 0 collects and writes the combined result."""
    if golden_t is not None:
        rank_status = "PASS" if is_pass else "FAIL"
        rank_precision_str = f"rank{rank}:{rank_status}({precision * 100}%)"
    else:
        # 插件返回了列表但本 rank 位缺失/非 tensor——golden 不可用，同 NO_GOLDEN 语义
        rank_precision_str = f"rank{rank}:FAIL(NO_GOLDEN)"
    all_prec_strings = [None] * world_size
    dist.all_gather_object(all_prec_strings, rank_precision_str)

    if rank == 0:
        # Combine per-rank precision (matching ACLNN format: rank0:PASS(99.97%),rank1:PASS(...),...)
        combined_prec = ",".join(all_prec_strings)
        has_fail = any("FAIL" in s for s in all_prec_strings)
        np.savez(
            result_path,
            precision_0=np.array(combined_prec),
            pass_0=np.array("PASS" if not has_fail else "FAIL"),
        )


def _run_graph_phase(
    rank, world_size, result_path, api_name, dev_tensors, hcomm, attrs, graph_mode, golden_t, rtol, atol, ptol
):
    """Multi-device mc2 graph mode (mirrors mc2_test define_model).

    Runs the same HCCL op through torch.compile + torchair NPU backend
    (dynamic=True when graph_mode=="dynamic"), then compares against the
    same CPU golden used for eager. Result is surfaced as GRAPH DYN/CST
    in the framework_api return structure.
    """
    if graph_mode not in ("dynamic", "static") or golden_t is None:
        return
    dist.barrier()
    graph_precision = None
    graph_is_pass = False
    graph_err = None
    try:
        graph_result = _run_hccl_graph(api_name, dev_tensors, hcomm, world_size, attrs, graph_mode)
        graph_out = graph_result[0] if isinstance(graph_result, (tuple, list)) else graph_result
        g_flat = graph_out.cpu().contiguous().view(-1)
        close_g = torch.isclose(g_flat, golden_t.cpu().contiguous().view(-1), rtol=rtol, atol=atol, equal_nan=True)
        graph_precision = float(close_g.sum().item()) / float(close_g.numel())
        graph_is_pass = (1.0 - graph_precision) <= ptol
    except Exception as ge:
        graph_err = str(ge)[:500]

    g_rank_status = "PASS" if graph_is_pass else "FAIL"
    if graph_err:
        g_rank_str = f"rank{rank}:ERROR({graph_err[:80]})"
    else:
        g_rank_str = f"rank{rank}:{g_rank_status}({graph_precision * 100}%)"
    all_g_strings = [None] * world_size
    dist.all_gather_object(all_g_strings, g_rank_str)

    if rank == 0:
        combined_g = ",".join(all_g_strings)
        g_has_fail = any("FAIL" in s or "ERROR" in s for s in all_g_strings)
        # Reload save_arrays (rank 0 wrote it above) and append graph fields
        loaded_so_far = np.load(result_path, allow_pickle=False)
        save_arrays = {k: loaded_so_far[k] for k in loaded_so_far.files}
        save_arrays["graph_precision_0"] = np.array(combined_g)
        save_arrays["graph_pass_0"] = np.array("PASS" if not g_has_fail else "FAIL")
        save_arrays["graph_mode_0"] = np.array(graph_mode)
        np.savez(result_path, **save_arrays)


def worker(rank, world_size, port, input_path, plan_path, result_path, error_path, graph_mode="", device_ids=None):
    plan, timeout, deterministic_level = _load_plan(plan_path)
    # Rank is logical; the selected physical device may have a different ID.
    device_id = rank if device_ids is None else int(device_ids[rank])
    try:
        hcomm = _init_distributed(rank, world_size, port, timeout, deterministic_level, device_id)
        api_name, attrs, golden_disabled, remark, tensor_dtypes = _parse_plan(plan)
        moe_hcomm = _create_moe_group(api_name, rank, world_size)

        is_dual = _is_dual_group_op(api_name)
        attrs = _setup_dual_ep_tp_groups(api_name, rank, world_size, attrs)

        my_inputs_np = _load_rank_inputs(input_path, rank)
        cpu_inputs_for_rank, dev_tensors = _materialize_inputs(my_inputs_np, tensor_dtypes, device_id)

        attrs = _setup_moe_attrs(api_name, rank, world_size, moe_hcomm, attrs)

        _apply_transposes(attrs, dev_tensors)

        is_gmm = _is_gmm_op(api_name)
        attrs = _patch_gmm_attrs(api_name, attrs, dev_tensors, remark, rank, world_size)

        # Call NPU API
        if api_name.startswith("torch_npu.npu_moe_distribute_") or api_name == "torch_npu.npu_moe_update_expert":
            attrs = dict(attrs)
            attrs["_rank"] = rank
        result = _call_api(api_name, dev_tensors, hcomm, world_size, attrs)
        torch.npu.synchronize(device_id)

        if result is None:
            _save_executed_pass(rank, world_size, result_path)
            return

        npu_out = result[0] if isinstance(result, (tuple, list)) else result

        if golden_disabled:
            _save_executed_pass(rank, world_size, result_path)
            return

        cpu_inputs_per_rank = _gather_cross_rank_inputs(
            api_name, rank, world_size, result_path, cpu_inputs_for_rank, device_id
        )

        # Pure CPU golden (each golden function references ACLNN hccl_dispatcher.py logic)
        goldens = _compute_golden(
            api_name, cpu_inputs_per_rank, attrs, rank=rank, ws=world_size, dist_avail=(is_gmm or is_dual)
        )
        # 插件按 rank 取数可能部分失败：先聚合各 rank 的 golden 可用性、全员一致
        # 决策，避免 rank 间分支分歧导致后续 collective 错位
        has_golden = [None] * world_size
        dist.all_gather_object(has_golden, goldens is not None)
        if not all(has_golden):
            _save_rank_status(rank, world_size, result_path, f"rank{rank}:FAIL(NO_GOLDEN_SOURCE)", "FAIL")
            return
        if is_gmm or is_dual:
            golden = goldens[0] if goldens else None
        else:
            golden = goldens[rank] if rank < len(goldens) else None

        golden, golden_mm, npu_mm_out = _extract_golden_outputs(result, golden)
        golden_t, golden_mm_t = _cast_golden(golden, golden_mm, npu_out, npu_mm_out)

        npu_dtype_str = str(npu_out.dtype).split(".")[-1]
        precision = is_pass = None
        rtol = atol = ptol = None
        if golden_t is not None:
            rtol = 0.001 if npu_dtype_str in ("float16", "bfloat16") else 0.0001
            atol = 1e-8
            ptol = 0.001 if npu_dtype_str in ("float16", "bfloat16") else 0.0001
            precision, is_pass = _measure_precision(npu_out, golden_t, npu_mm_out, golden_mm_t, rtol, atol, ptol)

        _report_and_save(rank, world_size, result_path, golden_t, precision, is_pass)

        _run_graph_phase(
            rank, world_size, result_path, api_name, dev_tensors, hcomm, attrs, graph_mode, golden_t, rtol, atol, ptol
        )

        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        import traceback

        if rank == 0:
            with open(error_path, "w") as f:
                f.write(traceback.format_exc())
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _resolve_api(api_name):
    if "bmm_reducescatter_alltoall" in api_name:
        from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall

        return npu_bmm_reducescatter_alltoall
    if "alltoall_allgather_bmm" in api_name:
        from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm

        return npu_alltoall_allgather_bmm
    if "npu_mm_all_reduce_add_rms_norm" in api_name:
        from mindspeed.ops.npu_mm_all_reduce_add_rms_norm import npu_mm_all_reduce_add_rms_norm

        return npu_mm_all_reduce_add_rms_norm
    parts = api_name.split(".")
    obj = importlib.import_module(parts[0])
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj


if __name__ == "__main__":
    input_path = os.environ["TTK_E2E_INPUT"]
    plan_path = os.environ["TTK_E2E_PLAN"]
    result_path = os.environ["TTK_E2E_RESULT"]
    error_path = os.environ["TTK_E2E_ERROR"]
    device_ids = [int(d) for d in os.environ["TTK_E2E_DEVICES"].split(",")]
    ndev = int(os.environ["TTK_E2E_NDEV"])
    graph_mode = os.environ.get("TTK_E2E_GRAPH_MODE", "")

    rank_value = os.environ.get("TTK_E2E_RANK")
    if rank_value is not None:
        with open(plan_path) as plan_file:
            rank_plan = json.load(plan_file)
        worker_api_name = rank_plan.get("api_name", "")
        start_wait_timeout = int(rank_plan.get("proc_timeout", 0) or 3600)
        if _uses_private_moe_backend(worker_api_name) and (
            "npu_moe_distribute_" in worker_api_name or "mega_moe" in worker_api_name
        ):
            import importlib.util

            extension_root = os.path.expanduser(
                f"~/.cache/torch_extensions/py{sys.version_info.major}{sys.version_info.minor}_cpu"
            )

            def load_extension(name):
                path = os.path.join(extension_root, name, f"{name}.so")
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            comm_context_module = load_extension("comm_context")
            _moe_context_manager_cls = comm_context_module.CommContextManager
            _moe_dispatch_module = load_extension("npu_moe_distribute_dispatch")
            if "combine" in worker_api_name:
                _moe_combine_module = load_extension("npu_moe_distribute_combine")
            if "mega_moe" in worker_api_name:
                _mega_moe_module = load_extension("npu_mega_moe")
        ready_path = f"{result_path}.rank{rank_value}.ready"
        with open(ready_path, "w") as ready_file:
            ready_file.write("ready")
        start_path = f"{result_path}.start"
        import time

        # launcher 全 rank ready 后才写 start 文件；等待须有期限，否则 launcher
        # 异常退出时本 rank 永远空转（孤儿进程占卡）
        start_deadline = time.monotonic() + start_wait_timeout
        while not os.path.exists(start_path):
            if time.monotonic() >= start_deadline:
                raise TimeoutError(f"rank {rank_value}: timed out waiting for launcher start signal")
            time.sleep(0.05)
        worker(
            int(rank_value),
            ndev,
            int(os.environ["TTK_E2E_PORT"]),
            input_path,
            plan_path,
            result_path,
            error_path,
            graph_mode,
            device_ids,
        )
        sys.exit(1 if os.path.exists(error_path) else 0)

    with open(plan_path) as plan_file:
        parent_plan = json.load(plan_file)
    parent_api_name = parent_plan.get("api_name", "")
    if _uses_private_moe_backend(parent_api_name) and (
        "npu_moe_distribute_" in parent_api_name or "mega_moe" in parent_api_name
    ):
        parent_attrs = parent_plan.get("attributes", {})
        input_shapes = parent_plan.get("tensor_view_shapes", [])
        x_shape = input_shapes[1]
        expert_ids_shape = input_shapes[2]
        num_tokens = int(x_shape[0])
        # Combine CSV carries the maximum dispatched buffer as x. Its paired
        # dispatch input size is encoded by expert_ids.
        if "combine" in parent_api_name:
            num_tokens = int(expert_ids_shape[0])
        hidden = int(x_shape[1])
        topk = int(expert_ids_shape[1])
        if "mega_moe" in parent_api_name:
            # Mirror get_mega_moe_ccl_buffer_size for the E2E worker without
            # importing the JIT-loading Python wrapper in the parent process.
            def align(value, base):
                return (value + base - 1) // base * base

            local_experts = int(parent_attrs["moe_expert_num"]) // ndev
            compare_count = align(num_tokens * topk * 4, 256) // 4
            mask_slot_size = align(compare_count // 8, 32) + 32
            mask_recv_size = align(local_experts * ndev * mask_slot_size, 512)
            token_bytes = align(align(hidden, 256) + (hidden + 31) // 32, 32)
            total = (
                60 * 1024
                + mask_recv_size
                + align(num_tokens * token_bytes, 512)
                + align(num_tokens * hidden * topk * 2, 512)
            )
            buffer_size = align(align(total, 1024 * 1024) // (1024 * 1024), 2) // 2
        else:
            buffer_size = _get_moe_ccl_buffer_size(
                ndev,
                num_tokens,
                hidden,
                int(parent_attrs["moe_expert_num"]),
                topk,
                int(parent_attrs.get("shared_expert_num", 0)),
                int(parent_attrs.get("shared_expert_rank_num", 0)),
                str(parent_attrs.get("comm_alg", "")),
            )
        os.environ["HCCL_WHITELIST_DISABLE"] = "1"
        os.environ["HCCL_BUFFSIZE"] = str(buffer_size)

    port = find_free_port()
    processes = []
    try:
        for rank in range(ndev):
            child_env = os.environ.copy()
            child_env["TTK_E2E_RANK"] = str(rank)
            child_env["TTK_E2E_PORT"] = str(port)
            processes.append(subprocess.Popen([sys.executable, __file__], env=child_env))
            ready_path = f"{result_path}.rank{rank}.ready"
            deadline = time.monotonic() + 120
            while not os.path.exists(ready_path):
                if processes[-1].poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"Rank {rank} failed during E2E worker initialization")
                time.sleep(0.05)

        with open(f"{result_path}.start", "w") as start_file:
            start_file.write("start")

        for p in processes:
            return_code = p.wait()
            if return_code != 0:
                if not os.path.exists(error_path):
                    with open(error_path, "w") as f:
                        f.write(f"Worker process exited with code {return_code}")
                break
    finally:
        # 失败路径（ready 超时/异常/提前 break）收割全部 rank 子进程，防孤儿占卡
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            if p.poll() is None:
                with contextlib.suppress(Exception):
                    p.wait(timeout=30)
                if p.poll() is None:
                    p.kill()

    if not os.path.exists(result_path) and not os.path.exists(error_path):
        with open(error_path, "w") as f:
            f.write("No result file and no error file produced")

    if os.path.exists(error_path):
        sys.exit(1)
