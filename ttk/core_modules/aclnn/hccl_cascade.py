#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""真·小算子级联 golden：在 NPU 上跑 matmul + 真实 HCCL 通信原语。

参考 mc2_test/op_class/aclnnMatmulAlltoAll.py:96-110 的 get_hccl_mm 实现：
  mm_output = torch.matmul(input, weight)
  perOut = mm_output.view(temp_shape).permute(1, 0, 2).contiguous()
  dist.all_to_all_single(alltoall_out, perOut)

本模块采用多进程 + torch.distributed(hccl backend) 方案，与 mc2_test 一致。
fork 子进程后通过 dist.init_process_group 建立 HCCL 通信域，
通过 dist.all_to_all_single 调真实 HCCL 通信，避免单进程多线程直调 libhccl 的 runtime 错误。

参考实现：ttk/core_modules/framework_api/torch/multi_device.py worker()
"""

import os
import tempfile
from typing import Dict, List

import numpy as np


# Default timeout (seconds) for HCCL communication and subprocess joins.
def _configure_hccl_env(timeout=3600):
    """Configure HCCL port range. Timeout is passed directly to workers."""
    try:
        from ....config.loader import get_cascade_config

        _cfg = get_cascade_config()
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", _cfg.hccl_port_range)
    except Exception:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")


def _get_timeout(thread_contexts, device_ids):
    """Extract proc_timeout from thread_contexts attributes; default 3600."""
    first_ctx = thread_contexts.get(device_ids[0]) if device_ids else None
    if first_ctx and hasattr(first_ctx, "attributes"):
        return int(first_ctx.attributes.get("proc_timeout", 3600) or 3600)
    return 3600


def _save_inputs_per_rank(thread_contexts, device_ids, path):
    """把每个 rank 的 flat_tensors 保存到 npz 文件。

    bf16/fp16 不能直接 .numpy()，先转 fp32 保存（子进程加载后再转回原 dtype）。
    fp8/e8m0/hif8 也不能直接 .numpy()，view as uint8 保存（子进程恢复原 dtype）。
    保存原 dtype 字符串到独立字段，便于子进程恢复。
    """
    import torch

    arrays = {}
    dtypes = {}
    for did in device_ids:
        tc = thread_contexts[did]
        flat = tc.flat_tensors
        for i, t in enumerate(flat):
            if t is None:
                continue
            if isinstance(t, torch.Tensor):
                dtype_str = str(t.dtype).replace("torch.", "")
                if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    arrays[f"did{did}_t{i}"] = t.float().cpu().numpy()
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
                elif "float8" in dtype_str or "hifloat8" in dtype_str:
                    # fp8/e8m0/hif8: view as uint8 numpy (1 byte per element)
                    arr = t.view(torch.uint8).cpu().numpy()
                    arrays[f"did{did}_t{i}"] = arr
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
                else:
                    # int8/int32 等直接保存
                    arrays[f"did{did}_t{i}"] = t.cpu().numpy()
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
            else:
                np_dtype_str = str(getattr(t, "dtype", ""))
                if any(d in np_dtype_str for d in ("float8", "hifloat8", "e8m0")):
                    arr = np.frombuffer(np.ascontiguousarray(t).tobytes(), dtype=np.uint8).reshape(t.shape)
                    arrays[f"did{did}_t{i}"] = arr
                    dtypes[f"did{did}_t{i}_dtype"] = np_dtype_str
                else:
                    arrays[f"did{did}_t{i}"] = np.asarray(t)
                    dtypes[f"did{did}_t{i}_dtype"] = "numpy"
    # 把 dtype 字典也保存进去
    arrays.update(dtypes)
    np.savez(path, **arrays)


def _load_inputs_for_rank(input_path, did):
    """子进程从 npz 加载本 rank 的输入，并恢复原 dtype。

    fp8/e8m0/hif8 从 uint8 numpy 恢复为对应 torch dtype（view）。
    """
    import torch

    data = np.load(input_path, allow_pickle=False)
    tensors = {}
    for key in data.files:
        if key.startswith(f"did{did}_t") and not key.endswith("_dtype"):
            idx = int(key.split("_t")[1])
            arr = data[key]
            dtype_key = f"{key}_dtype"
            dtype_str = str(data[dtype_key]) if dtype_key in data.files else "float32"
            if arr.dtype.kind == "V":
                itemsize = arr.dtype.itemsize
                total_bytes = arr.size * itemsize
                arr = np.frombuffer(arr.tobytes(), dtype=np.uint8, count=total_bytes)
                arr = arr.reshape(arr.shape if arr.ndim > 0 else (1,))
            t = torch.from_numpy(arr.copy())
            if dtype_str == "torch.bfloat16":
                t = t.to(torch.bfloat16)
            elif dtype_str == "torch.float16":
                t = t.to(torch.float16)
            elif dtype_str == "torch.float32":
                t = t.to(torch.float32)
            elif "float8_e4m3" in dtype_str:
                t = t.view(torch.uint8).to(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else t
            elif "float8_e5m2" in dtype_str:
                t = t.view(torch.uint8).to(torch.float8_e5m2) if hasattr(torch, "float8_e5m2") else t
            elif "float8_e8m0" in dtype_str:
                t = t.view(torch.uint8).to(torch.float8_e8m0) if hasattr(torch, "float8_e8m0") else t.view(torch.uint8)
            elif "hifloat8" in dtype_str:
                # hif8: keep as uint8 view (mc2_test passes uint8 + dtype enum to npu_quant_matmul)
                t = t.view(torch.uint8)
            tensors[idx] = t
    return tensors


def _worker_matmul_alltoall(
    rank, world_size, port, input_path, result_path, transpose_x1, transpose_x2, mm_m, chunk_n, error_path, timeout=3600
):
    """子进程：单 rank 跑 matmul + 真HCCL all_to_all_single，结果写回文件。

    对齐 mc2_test aclnnMatmulAlltoAll.get_hccl_mm 行 96-110：
      mm_output = torch.matmul(self.input, self.weight)
      perOut = mm_output.view(temp_shape).permute(1, 0, 2).contiguous()
      dist.all_to_all_single(self.alltoall_out, perOut)
      output_hccl = alltoall_out.reshape(output_shape)
    """
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = tensors[2].npu() if 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0 else None

        # 转置处理（mc2_test: self.input = x1.t() if transposeX1 else x1）
        input_mat = x1.t().contiguous() if transpose_x1 else x1
        weight_mat = x2.t().contiguous() if transpose_x2 else x2

        # matmul（int8/fp8 weight需转float，plain matmul不支持这些dtype）
        w_dtype_str = str(weight_mat.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            mm_out = torch.matmul(input_mat.float(), weight_mat.float())
        else:
            mm_out = torch.matmul(input_mat, weight_mat)
        if bias is not None:
            mm_out = mm_out + bias

        # mc2_test: perOut = mm_output.view(temp_shape).permute(1, 0, 2).contiguous()
        # temp_shape = [mm_m, ws, chunk_n]
        per_out = mm_out.view(mm_m, world_size, chunk_n).permute(1, 0, 2).contiguous()
        # per_out 形状 [ws, M, chunk_n]

        # 真实 HCCL all_to_all
        alltoall_out = torch.empty_like(per_out)
        dist.all_to_all_single(alltoall_out, per_out)
        torch.npu.synchronize()

        # mc2_test: output_hccl = alltoall_out.reshape(output_shape)
        # output_shape = [ws*M, chunk_n]
        out_cpu = alltoall_out.reshape(mm_m * world_size, chunk_n).contiguous().cpu()
        _append_result(result_path, rank, out_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        # 子进程异常写到 error_path，便于父进程诊断
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def _append_result(result_path, rank, arr):
    """子进程把自己的结果写到独立文件 result_path.did{rank}.npz。

    每 rank 一个独立文件，避免多进程同时写同一文件的并发问题。
    父进程在所有子进程 join 后逐个加载合并。
    """
    key = f"cascade_did{rank}"
    rank_file = f"{result_path}.did{rank}.npz"
    np.savez(rank_file, **{key: arr})


def _load_cascade_outputs(result_path, device_ids):
    """父进程加载子进程写回的级联结果（每 rank 一个独立文件）。"""
    import torch

    outs = {}
    for did in device_ids:
        rank_file = f"{result_path}.did{did}.npz"
        if os.path.exists(rank_file):
            data = np.load(rank_file, allow_pickle=False)
            key = f"cascade_did{did}"
            if key in data.files:
                outs[did] = torch.from_numpy(data[key].copy())
    return outs


def _find_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# 多 case 顺序跑时，前一个 case 的 HCCL 子进程退出后 socket 可能处于 TIME_WAIT，
# 同一端口会被 HCCL 内部 socket bind 冲突。维护一个递增 base_port 避免复用。
# 端口范围从 cascade 配置读取 (default.yaml → ttk.conf.yaml → --config)。
_next_port_base = [None]  # lazy init from config


def _next_port():
    """获取一个递增端口，避免跨 case 端口复用导致 HCCL bind 冲突。"""
    import threading

    _configure_hccl_env()  # port range only; timeout set by run_xxx_cascade
    with threading.Lock():  # noqa: PLW2101 语义为"锁对象作上下文"，保持既有行为
        # Lazy-init port base/step/max from config on first call
        if _next_port_base[0] is None:
            try:
                from ....config.loader import get_cascade_config

                _cfg = get_cascade_config()
            except Exception:
                _cfg = None
            _next_port_base[0] = _cfg.port_base if _cfg else 30000
            _next_port_step = _cfg.port_step if _cfg else 13
            _next_port_max = _cfg.port_max if _cfg else 60000
            _next_port_base.extend([_next_port_step, _next_port_max])
        port = _next_port_base[0]
        _next_port_base[0] += _next_port_base[1]  # step
        if _next_port_base[0] > _next_port_base[2]:  # max
            _next_port_base[0] -= _next_port_base[2] - _next_port_base[1]  # wrap to base
        return port


def run_matmul_alltoall_cascade(
    thread_contexts: Dict[int, "object"], device_ids: List[int], transpose_x1: bool = False, transpose_x2: bool = False
) -> Dict[int, "object"]:
    """aclnnMatmulAlltoAll 真级联 golden。

    通过 fork 多进程 + torch.distributed(hccl) 实现，与 mc2_test get_hccl_mm 等价。

    参数：
      thread_contexts: did -> TestcaseAclnn（含 flat_tensors）
      device_ids: 参与通信的 device id 列表
      transpose_x1/transpose_x2: 是否转置 x1/x2

    返回：{did: torch.Tensor}（cpu tensor，作为 cross_check 的 third_party）
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    # 推断 mm_m / chunk_n（所有 rank 一致，weight_shared）
    first_ctx = thread_contexts[device_ids[0]]
    x1 = first_ctx.flat_tensors[0]
    x2 = first_ctx.flat_tensors[1]
    mm_m = x1.shape[1] if transpose_x1 else x1.shape[0]
    mm_n = x2.shape[0] if transpose_x2 else x2.shape[1]
    chunk_n = mm_n // n
    if chunk_n * n != mm_n:
        raise ValueError(f"MatmulAlltoAll: mm_n={mm_n} not divisible by world_size={n}")

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        # 父进程的 HCCL comm / NPU context 已在 profiling.py 的 golden 前清理释放
        # 用 spawn 启动子进程，子进程重新 init torch_npu + dist HCCL
        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_matmul_alltoall,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    transpose_x1,
                    transpose_x2,
                    mm_m,
                    chunk_n,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        # 检查错误
        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        outs = _load_cascade_outputs(result_path, device_ids)
        if len(outs) != n:
            raise RuntimeError(f"cascade result incomplete: got {len(outs)}/{n} ranks\n{error_msg}")
        return outs


# ============================================================
# aclnnAlltoAllMatmul: all_to_all(x1) -> matmul(a2a_out, x2)
# ============================================================
# 与 MatmulAlltoAll 差异：
#   MatmulAlltoAll: matmul(x1, x2) -> all_to_all -> output
#   AlltoAllMatmul: all_to_all(x1) -> matmul(a2a_out, x2) -> output
# AlltoAllMatmul 有两个输出：
#   - output:        matmul 结果，shape [M_chunk, N]
#   - alltoall_output: all_to_all 中间结果，shape [M_chunk, ws*K]（可选）
# 参考 mc2_test/op_class/aclnnAlltoAllMatmul.py:119-136 get_hccl_mm


def _worker_alltoall_matmul(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    transpose_x1,
    transpose_x2,
    mm_m_chunk,
    k_dim,
    n_dim,
    is_alltoall_output,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 真HCCL all_to_all + matmul。

    对齐 mc2_test aclnnAlltoAllMatmul.get_hccl_mm 行 119-136：
      input_re = self.input.reshape(temp_shape)            # temp_shape=[ws, M_chunk, K]
      dist.all_to_all_single(alltoall_out, input_re)
      alltoall_out = alltoall_out.permute(1, 0, 2).reshape(alltoall_shape)
                                                  # alltoall_shape=[M_chunk, ws*K]
      hccl_mm_out = torch.matmul(alltoall_out, weight) [+ bias]
    """
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = tensors[2].npu() if 2 in tensors and tensors[2].numel() > 0 else None

        # 转置处理（mc2_test: self.input = x1.t() if transposeX1 else x1）
        input_mat = x1.t().contiguous() if transpose_x1 else x1
        weight_mat = x2.t().contiguous() if transpose_x2 else x2

        # mc2_test: input_re = self.input.reshape(temp_shape)
        # temp_shape = [ws, M_chunk, K]
        input_re = input_mat.reshape(world_size, mm_m_chunk, k_dim).contiguous()

        # 真实 HCCL all_to_all
        alltoall_out = torch.empty(world_size, mm_m_chunk, k_dim, dtype=input_re.dtype, device=f"npu:{rank}")
        dist.all_to_all_single(alltoall_out, input_re)
        torch.npu.synchronize()

        # mc2_test: alltoall_out = alltoall_out.permute(1, 0, 2).reshape(alltoall_shape)
        # alltoall_shape = [M_chunk, ws*K]
        a2a_out = alltoall_out.permute(1, 0, 2).reshape(mm_m_chunk, world_size * k_dim).contiguous()

        # mc2_test: matmul(alltoall_out, weight) [+ bias]
        # 注意 mc2_test 的 bias 处理：bias_dtype==fp32 时 matmul 先 to(fp32) 再加 bias
        # fp8/int8 weight: cast to float for matmul (plain matmul doesn't support these dtypes)
        w_dtype_str = str(weight_mat.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            mm_out = torch.matmul(a2a_out.float(), weight_mat.float())
        else:
            mm_out = torch.matmul(a2a_out, weight_mat)
        if bias is not None:
            mm_out = mm_out + bias

        # 写回 main output（matmul 结果）
        mm_cpu = mm_out.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, mm_cpu.numpy())

        # 若需要 alltoall_output，写到独立文件
        if is_alltoall_output:
            a2a_cpu = a2a_out.reshape(-1).contiguous().cpu()
            _append_a2a_result(result_path, rank, a2a_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def _append_a2a_result(result_path, rank, arr):
    """子进程把 alltoall_output 写到独立文件（与 main output 分离）。"""
    key = f"cascade_a2a_did{rank}"
    rank_file = f"{result_path}.a2a_did{rank}.npz"
    np.savez(rank_file, **{key: arr})


def _load_cascade_a2a_outputs(result_path, device_ids):
    """父进程加载子进程写回的 alltoall_output（可选输出）。"""
    import torch

    outs = {}
    for did in device_ids:
        rank_file = f"{result_path}.a2a_did{did}.npz"
        if os.path.exists(rank_file):
            data = np.load(rank_file, allow_pickle=False)
            key = f"cascade_a2a_did{did}"
            if key in data.files:
                outs[did] = torch.from_numpy(data[key].copy())
    return outs


def run_alltoall_matmul_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    is_alltoall_output: bool = False,
) -> Dict[int, "object"]:
    """aclnnAlltoAllMatmul 真级联 golden。

    通过 spawn 多进程 + torch.distributed(hccl) 实现，与 mc2_test get_hccl_mm 等价。
    流程：all_to_all(x1) -> matmul(a2a_out, x2) -> output

    参数：
      thread_contexts: did -> TestcaseAclnn（含 flat_tensors）
      device_ids: 参与通信的 device id 列表
      transpose_x1/transpose_x2: 是否转置 x1/x2
      is_alltoall_output: 是否输出 all_to_all 中间结果（对应 mc2_test is_alltoall_output）

    返回：{did: {'main': torch.Tensor, 'alltoall': torch.Tensor|None}}（cpu tensor）
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    # 推断 mm_m_chunk / k_dim / n_dim（所有 rank 一致）
    first_ctx = thread_contexts[device_ids[0]]
    x1 = first_ctx.flat_tensors[0]
    x2 = first_ctx.flat_tensors[1]
    # input = x1 (or x1.t if transpose_x1)
    input_mat_shape = (x1.shape[1], x1.shape[0]) if transpose_x1 else (x1.shape[0], x1.shape[1])
    weight_mat_shape = (x2.shape[1], x2.shape[0]) if transpose_x2 else (x2.shape[0], x2.shape[1])
    mm_m_chunk = input_mat_shape[0] // n  # M_chunk = M_total / ws
    k_dim = input_mat_shape[1]
    n_dim = weight_mat_shape[1]
    if mm_m_chunk * n != input_mat_shape[0]:
        raise ValueError(f"AlltoAllMatmul: M_total={input_mat_shape[0]} not divisible by world_size={n}")

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_a2amm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        # spawn 子进程（同 run_matmul_alltoall_cascade 模式）
        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_alltoall_matmul,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    transpose_x1,
                    transpose_x2,
                    mm_m_chunk,
                    k_dim,
                    n_dim,
                    is_alltoall_output,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        a2a_outs = _load_cascade_a2a_outputs(result_path, device_ids) if is_alltoall_output else {}
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 组装返回：{did: {'main': tensor, 'alltoall': tensor|None}}
        # 把 main 的 flatten tensor reshape 为 [M_chunk, N]
        result = {}
        for did in device_ids:
            main_t = main_outs[did].reshape(mm_m_chunk, n_dim)
            a2a_t = a2a_outs.get(did).reshape(mm_m_chunk, n * k_dim) if did in a2a_outs else None
            result[did] = {"main": main_t, "alltoall": a2a_t}
        return result


# ============================================================
# aclnnAllGatherMatmul: all_gather(x1) -> matmul(gathered, x2)
# ============================================================
# 参考 mc2_test/op_class/aclnnAlltoAllMatmul.py:80-87 get_hccl_mm:
#   dist._all_gather_base(tensor_allgather, x1)
#   output_golden = torch.matmul(tensor_allgather, x2)
#   gather_output_golden = tensor_allgather
# 注意：mc2_test 的 x2 已在 get_input_weight 内做了 is_trans_b 转置（x2 = x2.t()）
# 所以这里 weight_mat 直接用 x2（若 is_trans_b，调用方应传转置后的 x2）
# 但 ttk 的 thread_contexts.flat_tensors[1] 是 CSV 里的原始 x2 shape，
# 我们在 worker 里按 is_trans_b 处理。


def _worker_allgather_matmul(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    is_trans_b,
    m_dim,
    k_dim,
    n_dim,
    is_gather_output,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 真HCCL all_gather + matmul。

    对齐 mc2_test aclnnAllGatherMatmul.get_hccl_mm 行 80-87：
      dist._all_gather_base(tensor_allgather, x1)
      output_golden = torch.matmul(tensor_allgather, x2)
      gather_output_golden = tensor_allgather

    注意：CSV 里 x2 shape 已是 (K, N)，直接用，不转置。
    """
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = tensors[2].npu() if 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0 else None

        # CSV x2 shape 已是 (K, N)，直接用
        weight_mat = x2

        # 真实 HCCL all_gather（_all_gather_base = all_gather_into_tensor）
        gathered = torch.empty(world_size * m_dim, k_dim, dtype=x1.dtype, device=f"npu:{rank}")
        dist._all_gather_base(gathered, x1)
        torch.npu.synchronize()

        # mc2_test: output = matmul(gathered, x2)
        w_dtype_str = str(weight_mat.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            mm_out = torch.matmul(gathered.float(), weight_mat.float())
        else:
            mm_out = torch.matmul(gathered, weight_mat)
        if bias is not None and bias.numel() > 0:
            mm_out = mm_out + bias

        # main output 写回
        mm_cpu = mm_out.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, mm_cpu.numpy())

        # gather_output 写到独立文件
        if is_gather_output:
            gathered_cpu = gathered.reshape(-1).contiguous().cpu()
            _append_a2a_result(result_path, rank, gathered_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_allgather_matmul_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    is_trans_b: bool = False,
    is_gather_output: bool = False,
) -> Dict[int, "object"]:
    """aclnnAllGatherMatmul 真级联 golden。

    通过 spawn 多进程 + torch.distributed(hccl) 实现，与 mc2_test get_hccl_mm 等价。
    流程：all_gather(x1) -> matmul(gathered, x2) -> output

    参数：
      thread_contexts: did -> TestcaseAclnn（含 flat_tensors）
      device_ids: 参与通信的 device id 列表
      is_trans_b: 是否转置 x2（CSV 里 x2 已是 (K,N)，此处保留参数但暂不转置）
      is_gather_output: 是否输出 all_gather 中间结果

    返回：{did: {'main': tensor, 'gather': tensor|None}}（cpu tensor）
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    # 推断 M / K / N（所有 rank 一致）
    first_ctx = thread_contexts[device_ids[0]]
    x1 = first_ctx.flat_tensors[0]
    x2 = first_ctx.flat_tensors[1]
    m_dim = x1.shape[0]
    k_dim = x1.shape[1]
    # CSV x2 shape 已是 (K, N)，N = x2.shape[1]
    n_dim = x2.shape[1]

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_agmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_allgather_matmul,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    is_trans_b,
                    m_dim,
                    k_dim,
                    n_dim,
                    is_gather_output,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        gather_outs = _load_cascade_a2a_outputs(result_path, device_ids) if is_gather_output else {}
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        result = {}
        for did in device_ids:
            main_t = main_outs[did].reshape(n * m_dim, n_dim)
            gather_t = gather_outs.get(did).reshape(n * m_dim, k_dim) if did in gather_outs else None
            result[did] = {"main": main_t, "gather": gather_t}
        return result


# ============================================================
# aclnnMatmulReduceScatter: matmul(x1, x2) -> reduce_scatter(output)
# ============================================================
# 参考 mc2_test/op_class/aclnnMatmulReduceScatter.py:60-64 get_hccl_mm:
#   output = torch.matmul(self.x1, self.x2)
#   dist._reduce_scatter_base(self.tensor_scatter, output, op=ReduceOp.SUM)
#   output_golden = self.tensor_scatter
# 注意：mc2_test 的 x2 已在 get_input_weight 内做了 is_trans_b 转置


def _worker_matmul_reducescatter(
    rank, world_size, port, input_path, result_path, is_trans_b, m_dim, k_dim, n_dim, error_path, timeout=3600
):
    """子进程：单 rank 跑 matmul + 真HCCL reduce_scatter。

    对齐 mc2_test aclnnMatmulReduceScatter.get_hccl_mm 行 60-64：
      output = torch.matmul(x1, x2)
      dist._reduce_scatter_base(tensor_scatter, output, op=SUM)

    注意：CSV 里 x2 shape 已是 (K, N)，直接用，不转置。
    mc2_test 的 is_trans_b 转置发生在 get_input_weight 的 non-graph scene，
    但 aclnn/hccl_mm 场景下 self.x2 保持 (K, N) 形状，matmul(x1, x2) 直接算。
    """
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from torch.distributed import ReduceOp

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()

        # CSV x2 shape 已是 (K, N)，直接用
        weight_mat = x2

        # mc2_test: output = torch.matmul(x1, x2)
        w_dtype_str = str(weight_mat.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            mm_out = torch.matmul(x1.float(), weight_mat.float())
        else:
            mm_out = torch.matmul(x1, weight_mat)

        # 真实 HCCL reduce_scatter（_reduce_scatter_base = reduce_scatter_into_tensor）
        # 输出 shape [M/ws, N]
        m_chunk = m_dim // world_size
        scatter = torch.empty(m_chunk, n_dim, dtype=mm_out.dtype, device=f"npu:{rank}")
        dist._reduce_scatter_base(scatter, mm_out, op=ReduceOp.SUM)
        torch.npu.synchronize()

        # main output 写回
        sc_cpu = scatter.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, sc_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_matmul_reducescatter_cascade(
    thread_contexts: Dict[int, "object"], device_ids: List[int], is_trans_b: bool = False
) -> Dict[int, "object"]:
    """aclnnMatmulReduceScatter 真级联 golden。

    通过 spawn 多进程 + torch.distributed(hccl) 实现，与 mc2_test get_hccl_mm 等价。
    流程：matmul(x1, x2) -> reduce_scatter(SUM) -> output

    参数：
      thread_contexts: did -> TestcaseAclnn（含 flat_tensors）
      device_ids: 参与通信的 device id 列表
      is_trans_b: 是否转置 x2（CSV 里 x2 已是 (K,N)，此处保留参数但暂不转置）

    返回：{did: torch.Tensor}（cpu tensor，作为 cross_check 的 third_party）
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    first_ctx = thread_contexts[device_ids[0]]
    x1 = first_ctx.flat_tensors[0]
    x2 = first_ctx.flat_tensors[1]
    m_dim = x1.shape[0]
    k_dim = x1.shape[1]
    # CSV x2 shape 已是 (K, N)，N = x2.shape[1]
    n_dim = x2.shape[1]
    if m_dim % n != 0:
        raise ValueError(f"MatmulReduceScatter: M={m_dim} not divisible by world_size={n}")

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_rsmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_matmul_reducescatter,
                args=(rank, n, port, input_path, result_path, is_trans_b, m_dim, k_dim, n_dim, error_path, timeout),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        outs = _load_cascade_outputs(result_path, device_ids)
        if len(outs) != n:
            raise RuntimeError(f"cascade result incomplete: got {len(outs)}/{n} ranks\n{error_msg}")

        # 把 flatten tensor reshape 为 [M/ws, N]
        result = {}
        m_chunk = m_dim // n
        for did in device_ids:
            result[did] = outs[did].reshape(m_chunk, n_dim)
        return result


# ============================================================
# GMM 算子共享 helper：permute / unpermute / group_list
# ============================================================
# 参考 mc2_test aclnnAlltoAllvGroupedMatMul.permute_with_npu 和
#        aclnnGroupedMatMulAlltoAllv.unpermute_npu
# 这两个操作都是按 expTokenNums 矩阵重排 tokens，逻辑与 CPU golden
# (hccl_dispatcher.py 退场前的 __permute_a2a_gmm / __unpermute_mc2) 等价，但在 NPU 上做 index_select


def _permute_a2a_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    """AlltoAllvGroupedMatMul 的 permute：a2a 后 tokens 重排为按 expert 分组。

    对齐 mc2_test aclnnAlltoAllvGroupedMatMul.permute_with_npu 行 231-253。
    输入 tokens: [sum(recv_counts), K]（a2a 输出，已在 NPU 上）
    返回 permuted: [sum(per_expert_sizes), K]（按 expert 分组，给 npu_gmm 用）
    """
    import torch

    device = tokens.device
    indices = torch.zeros(exp_per_card, ep_ws).long().to(device)
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = expTokenNums[i][j + exp_per_card * rank_idx]
    trans = indices.permute(1, 0)
    flaten = trans.reshape(-1)
    sum_list = torch.cumsum(flaten, dim=0)
    tmp = []
    for i in range(len(sum_list)):
        if i == 0:
            tmp.append(range(int(sum_list[i])))
        else:
            tmp.append(range(int(sum_list[i - 1]), int(sum_list[i])))
    out = []
    expert_sizes = []
    for e in range(exp_per_card):
        exp_token = []
        for r in range(ep_ws):
            exp_token += list(tmp[e + r * exp_per_card])
        combined = torch.tensor(exp_token).long().to(device)
        out.append(tokens.index_select(0, combined))
        expert_sizes.append(len(exp_token))
    return torch.cat(out, dim=0), expert_sizes


def _unpermute_gmm_a2a(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    """GroupedMatMulAlltoAllv 的 unpermute：gmm 后 tokens 重排为按目的 rank 分组。

    对齐 mc2_test aclnnGroupedMatMulAlltoAllv.unpermute 行 168-194。
    输入 tokens: [sum(per_expert), N]（gmm 输出，已在 NPU 上）
    返回 unpermuted: [sum(send_counts), N]（按目的 rank 分组，给 a2a 用）
    """
    import numpy as np
    import torch

    device = tokens.device
    empty_arr = np.zeros((ep_ws, exp_per_card), dtype=np.int64)
    for i in range(ep_ws):
        tmp = expTokenNums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card]
        empty_arr[i:] = tmp
    tmp1 = empty_arr.T
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2 = np.cumsum(sum_list1, axis=0)
    offsets = [0] + sum_list2[:-1].tolist()
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for i in range(exp_per_card):
        tmp = []
        for j in range(ep_ws):
            if j == 0:
                tmp.append(list(map(lambda x: x + offsets[i], list(range(int(sum_list[i][j]))))))  # noqa: C417 保持
            else:
                tmp.append(
                    list(map(lambda x: x + offsets[i], list(range(int(sum_list[i][j - 1]), int(sum_list[i][j])))))  # noqa: C417 保持
                )
        indices_list.append(tmp)
    selected = []
    for i in range(ep_ws):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
            selected.append(tokens.index_select(dim=0, index=indices))
    return torch.cat(selected, dim=0).to(tokens.dtype)


def _gmm_group_list_cumsum(expTokenNums, rank_idx, exp_per_card, ep_ws):
    """计算 npu_gmm 需要的 group_list（cumsum 后的累加值）。

    对齐 mc2_test get_group_list + get_tmp_tensor 行 240:
      group_list = sum_over_ranks(expTokenNums[i][rank*exp_per_card+j])
      group_list_npu = cumsum(group_list)
    """
    from itertools import accumulate

    group_list = []
    for j in range(exp_per_card):
        total = sum(expTokenNums[i][rank_idx * exp_per_card + j] for i in range(ep_ws))
        group_list.append(total)
    return list(accumulate(group_list))


def _save_gmm_meta(input_path, expTokenNums, ep_ws, exp_per_card):
    """把 expTokenNums 矩阵保存到 npz（所有 rank 共享）。"""
    arr = np.array(expTokenNums, dtype=np.int64)
    np.savez(
        input_path + ".meta.npz",
        expTokenNums=arr,
        ep_ws=np.array([ep_ws], dtype=np.int64),
        exp_per_card=np.array([exp_per_card], dtype=np.int64),
    )


def _load_gmm_meta(meta_path):
    """子进程加载 expTokenNums 等元数据。"""
    data = np.load(meta_path, allow_pickle=False)
    expTokenNums = data["expTokenNums"].tolist()
    ep_ws = int(data["ep_ws"][0])
    exp_per_card = int(data["exp_per_card"][0])
    return expTokenNums, ep_ws, exp_per_card


# ============================================================
# aclnnAlltoAllvGroupedMatMul: all_to_allv(gmm_x) -> permute -> gmm
# ============================================================
# 参考 mc2_test/op_class/aclnnAlltoAllvGroupedMatMul.py:262-279 get_hccl_mm:
#   dist.all_to_all_single(out, gmm_x, output_split_sizes, input_split_sizes)
#   permuted_out = permute_with_npu(alltoallv_out)
#   output_gmm = gmm.npu_gmm(permuted_out, gmm_weight, group_list, group_type=0)
#   mm_out = torch.mm(mm_x, mm_weight)  # 可选


def _worker_alltoallv_gmm(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    trans_gmm_weight,
    trans_mm_weight,
    permute_out_flag,
    mm_out_flag,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 真HCCL all_to_allv + permute + npu_gmm。"""
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from mindspeed.ops import gmm

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        # 加载元数据
        expTokenNums, ep_ws, exp_per_card = _load_gmm_meta(input_path + ".meta.npz")

        # 加载本 rank 输入
        tensors = _load_inputs_for_rank(input_path, rank)
        gmm_x = tensors[0].npu()
        gmm_weight = tensors[1].npu()
        mm_x = tensors[4].npu() if 4 in tensors and tensors[4] is not None and tensors[4].numel() > 0 else None
        mm_weight = tensors[5].npu() if 5 in tensors and tensors[5] is not None and tensors[5].numel() > 0 else None

        # 转置 gmm_weight（mc2_test: trans_gmm_weight_flag 时 gmm_weight = transpose(gmm_weight, 1, 2)）
        if trans_gmm_weight:
            gmm_weight = torch.transpose(gmm_weight, 1, 2).contiguous()
        if trans_mm_weight and mm_out_flag and mm_weight is not None:
            mm_weight = torch.transpose(mm_weight, 0, 1).contiguous()

        # 计算 input_splits / output_splits（mc2_test gen_input_splits / gen_output_splits）
        my_row = expTokenNums[rank]
        input_splits = [int(sum(my_row[t * exp_per_card : (t + 1) * exp_per_card])) for t in range(ep_ws)]
        output_splits = [
            int(sum(expTokenNums[i][rank * exp_per_card : (rank + 1) * exp_per_card])) for i in range(ep_ws)
        ]
        K = gmm_x.shape[1]

        # 真实 HCCL all_to_allv（变长 split）
        a2a_out = torch.empty(sum(output_splits), K, dtype=gmm_x.dtype, device=f"npu:{rank}")
        dist.all_to_all_single(a2a_out, gmm_x, output_split_sizes=output_splits, input_split_sizes=input_splits)
        torch.npu.synchronize()

        # permute（按 expert 分组）
        permuted, expert_sizes = _permute_a2a_gmm(a2a_out, exp_per_card, ep_ws, rank, expTokenNums)
        permuted = permuted.npu()

        # npu_gmm
        group_list = _gmm_group_list_cumsum(expTokenNums, rank, exp_per_card, ep_ws)
        group_list_tensor = torch.tensor(group_list, dtype=torch.int64).npu()
        gmm_out = gmm.npu_gmm(permuted, gmm_weight, bias=None, group_list=group_list_tensor, group_type=0)

        # main output 写回
        gmm_cpu = gmm_out.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, gmm_cpu.numpy())

        # permute_out 写到独立文件
        if permute_out_flag:
            permute_cpu = permuted.reshape(-1).contiguous().cpu()
            _append_a2a_result(result_path, rank, permute_cpu.numpy())

        # mm_out
        if mm_out_flag and mm_x is not None:
            mm_out = torch.mm(mm_x, mm_weight)
            mm_cpu = mm_out.reshape(-1).contiguous().cpu()
            _append_mm_result(result_path, rank, mm_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def _append_mm_result(result_path, rank, arr):
    """子进程把 mm_out 写到独立文件。"""
    key = f"cascade_mm_did{rank}"
    rank_file = f"{result_path}.mm_did{rank}.npz"
    np.savez(rank_file, **{key: arr})


def _load_cascade_mm_outputs(result_path, device_ids):
    """父进程加载子进程写回的 mm_out。"""
    import torch

    outs = {}
    for did in device_ids:
        rank_file = f"{result_path}.mm_did{did}.npz"
        if os.path.exists(rank_file):
            data = np.load(rank_file, allow_pickle=False)
            key = f"cascade_mm_did{did}"
            if key in data.files:
                outs[did] = torch.from_numpy(data[key].copy())
    return outs


def run_alltoallv_gmm_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    expTokenNums,
    ep_ws: int,
    exp_per_card: int,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    permute_out_flag: bool = False,
    mm_out_flag: bool = False,
) -> Dict[int, "object"]:
    """aclnnAlltoAllvGroupedMatMul 真级联 golden。

    流程：all_to_allv(gmm_x) -> permute -> npu_gmm -> output
    对齐 mc2_test aclnnAlltoAllvGroupedMatMul.get_hccl_mm 行 262-279。

    返回：{did: {'main': tensor, 'permute': tensor|None, 'mm': tensor|None}}
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_a2agmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)
        _save_gmm_meta(input_path, expTokenNums, ep_ws, exp_per_card)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_alltoallv_gmm,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    trans_gmm_weight,
                    trans_mm_weight,
                    permute_out_flag,
                    mm_out_flag,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        permute_outs = _load_cascade_a2a_outputs(result_path, device_ids) if permute_out_flag else {}
        mm_outs = _load_cascade_mm_outputs(result_path, device_ids) if mm_out_flag else {}
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 推断 main output shape [sum(recv), N]
        first_ctx = thread_contexts[device_ids[0]]
        gmm_weight = first_ctx.flat_tensors[1]
        n_dim = gmm_weight.shape[1] if trans_gmm_weight else gmm_weight.shape[2]

        result = {}
        for did in device_ids:
            rank_idx = list(device_ids).index(did)
            recv_total = sum(
                sum(expTokenNums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card]) for i in range(ep_ws)
            )
            main_t = main_outs[did].reshape(recv_total, n_dim)
            permute_t = None
            if permute_out_flag and did in permute_outs:
                K = first_ctx.flat_tensors[0].shape[1]
                permute_t = permute_outs[did].reshape(recv_total, K)
            mm_t = mm_outs.get(did) if did in mm_outs else None
            result[did] = {"main": main_t, "permute": permute_t, "mm": mm_t}
        return result


# ============================================================
# aclnnGroupedMatMulAlltoAllv: npu_gmm -> unpermute -> all_to_allv
# ============================================================
# 参考 mc2_test/op_class/aclnnGroupedMatMulAlltoAllv.py:258-271 get_hccl_mm:
#   gmm_out = gmm.npu_gmm(gmm_x, gmm_weight, group_list, group_type=0)
#   unpermuted_out = unpermute_npu(gmm_out)
#   dist.all_to_all_single(output, unpermuted_out, output_split_sizes, input_split_sizes)
#   mm_out = torch.mm(mm_x, mm_weight)  # 可选


def _worker_gmm_alltoallv(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    trans_gmm_weight,
    trans_mm_weight,
    mm_out_flag,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 npu_gmm + unpermute + 真HCCL all_to_allv。"""
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from mindspeed.ops import gmm

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        expTokenNums, ep_ws, exp_per_card = _load_gmm_meta(input_path + ".meta.npz")

        tensors = _load_inputs_for_rank(input_path, rank)
        gmm_x = tensors[0].npu()
        gmm_weight = tensors[1].npu()
        mm_x = tensors[4].npu() if 4 in tensors and tensors[4] is not None and tensors[4].numel() > 0 else None
        mm_weight = tensors[5].npu() if 5 in tensors and tensors[5] is not None and tensors[5].numel() > 0 else None

        if trans_gmm_weight:
            gmm_weight = torch.transpose(gmm_weight, 1, 2).contiguous()
        if trans_mm_weight and mm_out_flag and mm_weight is not None:
            mm_weight = torch.transpose(mm_weight, 0, 1).contiguous()

        # npu_gmm（group_list 是 cumsum 形式）
        group_list = _gmm_group_list_cumsum(expTokenNums, rank, exp_per_card, ep_ws)
        group_list_tensor = torch.tensor(group_list, dtype=torch.int64).npu()
        gmm_out = gmm.npu_gmm(gmm_x, gmm_weight, bias=None, group_list=group_list_tensor, group_type=0)

        # unpermute（按目的 rank 分组）
        unpermuted = _unpermute_gmm_a2a(gmm_out, exp_per_card, ep_ws, rank, expTokenNums)
        unpermuted = unpermuted.npu()

        # 计算 input_splits / output_splits（mc2_test gen_input_splits / gen_output_splits）
        # input_splits: 本 rank 各 expert 分组发往各 rank 的 sum
        my_row = expTokenNums[rank]
        input_splits = [int(sum(my_row[t * exp_per_card : (t + 1) * exp_per_card])) for t in range(ep_ws)]
        # output_splits: 各 rank 发给本 rank 的 sum
        output_splits = [
            int(sum(expTokenNums[i][rank * exp_per_card : (rank + 1) * exp_per_card])) for i in range(ep_ws)
        ]
        N = gmm_out.shape[1] if gmm_out.dim() > 1 else 1

        # 真实 HCCL all_to_allv
        a2a_out = torch.empty(sum(output_splits), N, dtype=gmm_x.dtype, device=f"npu:{rank}")
        dist.all_to_all_single(a2a_out, unpermuted, output_split_sizes=output_splits, input_split_sizes=input_splits)
        torch.npu.synchronize()

        # main output 写回
        out_cpu = a2a_out.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, out_cpu.numpy())

        # mm_out
        if mm_out_flag and mm_x is not None:
            mm_out = torch.mm(mm_x, mm_weight)
            mm_cpu = mm_out.reshape(-1).contiguous().cpu()
            _append_mm_result(result_path, rank, mm_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_gmm_alltoallv_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    expTokenNums,
    ep_ws: int,
    exp_per_card: int,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    mm_out_flag: bool = False,
) -> Dict[int, "object"]:
    """aclnnGroupedMatMulAlltoAllv 真级联 golden。

    流程：npu_gmm -> unpermute -> all_to_allv -> output
    对齐 mc2_test aclnnGroupedMatMulAlltoAllv.get_hccl_mm 行 258-271。

    返回：{did: {'main': tensor, 'mm': tensor|None}}
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_gmma2a_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)
        _save_gmm_meta(input_path, expTokenNums, ep_ws, exp_per_card)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_gmm_alltoallv,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    trans_gmm_weight,
                    trans_mm_weight,
                    mm_out_flag,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        mm_outs = _load_cascade_mm_outputs(result_path, device_ids) if mm_out_flag else {}
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 推断 N（gmm 输出第二维）
        first_ctx = thread_contexts[device_ids[0]]
        gmm_weight = first_ctx.flat_tensors[1]
        n_dim = gmm_weight.shape[1] if trans_gmm_weight else gmm_weight.shape[2]

        result = {}
        for did in device_ids:
            rank_idx = list(device_ids).index(did)
            recv_total = sum(
                sum(expTokenNums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card]) for i in range(ep_ws)
            )
            main_t = main_outs[did].reshape(recv_total, n_dim)
            mm_t = mm_outs.get(did) if did in mm_outs else None
            result[did] = {"main": main_t, "mm": mm_t}
        return result


# ============================================================
# aclnnBatchMatMulReduceScatterAlltoAll: bmm -> reduce_scatter(TP) -> all_to_all(EP)
# ============================================================
# 双通信域算子：EP 组做 all_to_all，TP 组做 reduce_scatter
# 参考 mc2_test/op_class/aclnnBatchMatMulReduceScatterAlltoAll.py:118-146 get_hccl_mm:
#   bmm_out = torch.bmm(input, weight)
#   bmm_out = bmm_out.reshape(reshape_1).permute(...).reshape(reshape_2).contiguous()
#   dist._reduce_scatter_base(rs_out, bmm_out, op=SUM, group=group_tp)
#   [可选 bias add]
#   rs_out = rs_out.reshape(reshape_3).permute(1,0,2,3).contiguous()
#   dist.all_to_all_single(a2a_out, rs_out, group=group_ep)
#   output = a2a_out.reshape(reshape_4)
#
# EP/TP 划分（对齐 mc2_test setup_ep_tp）:
#   EP 组 i: [x * tp_size + i for x in range(ep_size)]   (tp_size 个 EP 组)
#   TP 组 i: [x + tp_size * i for x in range(tp_size)]   (ep_size 个 TP 组)


def _worker_bmm_rs_a2a(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    ep_ws,
    tp_ws,
    shard_type,
    is_trans,
    is_bias,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 bmm + 真HCCL reduce_scatter(TP) + all_to_all(EP)。"""
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from torch.distributed import ReduceOp

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        # 建 EP/TP 子通信域（对齐 mc2_test setup_ep_tp）
        # EP 组 i: [x * tp_ws + i for x in range(ep_ws)]
        # TP 组 i: [x + tp_ws * i for x in range(tp_ws)]
        ep_group = None
        tp_group = None
        for i in range(tp_ws):
            ep_ranks = [x * tp_ws + i for x in range(ep_ws)]
            g = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = g
        for i in range(ep_ws):
            tp_ranks = [x + tp_ws * i for x in range(tp_ws)]
            g = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group = g

        # 加载输入
        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = None
        if is_bias and 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0:
            bias = tensors[2].npu()

        # mc2_test: is_trans 时 weight 已在 get_input_weight 里 permute(0,2,1)
        # ttk CSV 的 x2 shape 已是 (B, K, N)，is_trans 时需转置为 (B, N, K) 再 bmm
        # 但 mc2_test get_input_weight 行 78-79: is_trans 时 weight = weight.permute(0,2,1)
        # 即 weight 从 (B, K, N) 变 (B, N, K)，然后 bmm(input, weight) = (B, M, K) * (B, K, N)... 不对
        # 实际 mc2_test tmp_weight_shape = [w0, w2, w1] if is_trans else weight_shape
        # 即 is_trans 时先生成 (B, N, K) 再 permute 回 (B, K, N) 供 bmm
        # 所以 ttk 这里 x2 已是 (B, K, N)，直接 bmm 即可，无需额外转置
        weight_mat = x2

        # bmm（dtype-native，对齐 mc2_test get_hccl_mm 行 119）
        bmm_out = torch.bmm(x1, weight_mat)

        # 推断形状参数（对齐 mc2_test get_tmp_tensor 行 84-110）
        # input_shape = x1.shape = (E/ep, C*ep, ...) 实际 CSV x1=(B, M, K)
        # mc2_test: E = input_shape[0] * ep_size; M = input_shape[2] * tp_size
        #           C = input_shape[1] / ep_size; H = weight_shape[2]
        x1.shape[0]
        E_div_ep = x1.shape[0]
        x_dim1 = x1.shape[1]
        H = weight_mat.shape[2]
        if shard_type == 0:
            C = x_dim1 // ep_ws
        else:
            C_div_tp = x_dim1 // ep_ws // tp_ws

        # reshape_1 / reshape_2 / reshape_3 / reshape_4（对齐 mc2_test 行 90-110）
        if shard_type == 0:
            reshape_1 = [E_div_ep, ep_ws * C, tp_ws, H // tp_ws]
            reshape_2 = [tp_ws * E_div_ep, ep_ws * C, H // tp_ws]
            reshape_3 = [E_div_ep, ep_ws, C, H // tp_ws]
            reshape_4 = [E_div_ep * ep_ws, C, H // tp_ws]
            tensor_scatter_shape = [E_div_ep, ep_ws * C, H // tp_ws]
            alltoall_shape = [ep_ws, E_div_ep, C, H // tp_ws]
        else:
            reshape_1 = [E_div_ep, ep_ws, tp_ws, C_div_tp, H]
            reshape_2 = [tp_ws * E_div_ep, ep_ws * C_div_tp, H]
            reshape_3 = [E_div_ep, ep_ws, C_div_tp, H]
            reshape_4 = [E_div_ep * ep_ws, C_div_tp, H]
            tensor_scatter_shape = [E_div_ep, ep_ws * C_div_tp, H]
            alltoall_shape = [ep_ws, E_div_ep, C_div_tp, H]

        # reshape + permute（对齐 mc2_test 行 121-130）
        bmm_out = bmm_out.reshape(reshape_1)
        bmm_out = bmm_out.permute(2, 0, 1, 3) if shard_type == 0 else bmm_out.permute(2, 0, 1, 3, 4)
        bmm_out = bmm_out.reshape(reshape_2).contiguous()

        # 真HCCL reduce_scatter（TP 组）
        rs_out = torch.zeros(tensor_scatter_shape, dtype=x1.dtype, device=f"npu:{rank}")
        dist._reduce_scatter_base(rs_out, bmm_out, op=ReduceOp.SUM, group=tp_group)
        torch.npu.synchronize()

        # bias add（可选）
        if is_bias and bias is not None:
            if bias.dim() == 2:
                bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
            rs_out = rs_out + bias

        # reshape + permute（对齐 mc2_test 行 139-141）
        rs_out = rs_out.reshape(reshape_3)
        rs_out = rs_out.permute(1, 0, 2, 3).contiguous()

        # 真HCCL all_to_all（EP 组）
        a2a_out = torch.zeros(alltoall_shape, dtype=x1.dtype, device=f"npu:{rank}")
        dist.all_to_all_single(a2a_out, rs_out, group=ep_group)
        torch.npu.synchronize()

        # reshape 输出
        output = a2a_out.reshape(reshape_4)

        # 写回
        out_cpu = output.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, out_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_bmm_rs_a2a_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    ep_ws: int,
    tp_ws: int,
    shard_type: int,
    is_trans: bool = False,
    is_bias: bool = False,
) -> Dict[int, "object"]:
    """aclnnBatchMatMulReduceScatterAlltoAll 真级联 golden。

    流程：bmm -> reduce_scatter(TP) -> all_to_all(EP) -> output
    对齐 mc2_test aclnnBatchMatMulReduceScatterAlltoAll.get_hccl_mm 行 118-146。

    返回：{did: {'main': tensor}}
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}
    if n != ep_ws * tp_ws:
        raise ValueError(f"BMM_RS_A2A: world_size={n} != ep_ws*tp_ws={ep_ws * tp_ws}")

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_bmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_bmm_rs_a2a,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    ep_ws,
                    tp_ws,
                    shard_type,
                    is_trans,
                    is_bias,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 推断输出 shape（reshape_4）
        first_ctx = thread_contexts[device_ids[0]]
        out_idx = first_ctx.output_tensor_indexes[0]
        out_shape = first_ctx.tensor_view_shapes[out_idx]

        result = {}
        for did in device_ids:
            main_t = main_outs[did].reshape(out_shape)
            result[did] = {"main": main_t}
        return result


# ============================================================
# aclnnAlltoAllAllGatherBatchMatMul: all_to_all(EP) -> all_gather(TP) -> bmm
# ============================================================
# 双通信域算子：EP 组做 all_to_all，TP 组做 all_gather
# 参考 mc2_test/op_class/aclnnAlltoAllAllGatherBatchMatMul.py:179-206 get_hccl_mm:
#   dist.all_to_all_single(a2a_out, input, group=group_ep)
#   a2a_out = a2a_out.reshape(reshape_1).permute(1,0,2,3).contiguous()
#   dist._all_gather_base(ag_out, a2a_out, group=group_tp)
#   ag_out = ag_out.reshape(reshape_2)
#   [shard_type 0: permute(1,2,3,0,4); 1: permute(1,2,0,3,4)]
#   ag_out = ag_out.reshape(reshape_3)
#   bmm_out = torch.bmm(ag_out, weight)
#   [可选 bias add + activation]
#   return output, gather_output, bmm_out


def _activate_npu(x, act_type):
    """对齐 mc2_test activate，在 NPU 上计算。"""
    import torch
    import torch_npu  # noqa: F401

    if act_type == 0:
        return x
    if act_type == 1:
        return torch.nn.functional.gelu(x)
    if act_type == 2:
        return torch.nn.functional.silu(x)
    if act_type == 3:
        return torch.relu(x)
    if act_type == 4:
        return x / (1.0 + torch.exp(-1.702 * x))
    return x


def _worker_a2a_ag_bmm(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    ep_ws,
    tp_ws,
    shard_type,
    is_trans,
    is_bias,
    act_type,
    need_ag_out,
    need_act_feat,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 真HCCL all_to_all(EP) + all_gather(TP) + bmm。"""
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        # 建 EP/TP 子通信域
        ep_group = None
        tp_group = None
        for i in range(tp_ws):
            ep_ranks = [x * tp_ws + i for x in range(ep_ws)]
            g = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = g
        for i in range(ep_ws):
            tp_ranks = [x + tp_ws * i for x in range(tp_ws)]
            g = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group = g

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        weight = tensors[1].npu()
        bias = None
        if is_bias and 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0:
            bias = tensors[2].npu()

        # 推断形状（对齐 mc2_test get_tmp_tensor 行 122-144）
        # input_shape = x1.shape = (E, C, H/tp) for shard 0; (E, C/tp, H) for shard 1
        E = x1.shape[0]
        E_div_ep = E // ep_ws
        if shard_type == 0:
            C = x1.shape[1]
            H_div_tp = x1.shape[2]
            H = H_div_tp * tp_ws
            reshape_1 = [ep_ws, E_div_ep, C, H_div_tp]
            tensor_ag_shape = [tp_ws * E_div_ep, ep_ws, C, H_div_tp]
            reshape_2 = [tp_ws, E_div_ep, ep_ws, C, H_div_tp]
            reshape_3 = [E_div_ep, ep_ws * C, H]
        else:
            C_div_tp = x1.shape[1]
            H = x1.shape[2]
            C = C_div_tp * tp_ws
            reshape_1 = [ep_ws, E_div_ep, C_div_tp, H]
            tensor_ag_shape = [tp_ws * E_div_ep, ep_ws, C_div_tp, H]
            reshape_2 = [tp_ws, E_div_ep, ep_ws, C_div_tp, H]
            reshape_3 = [E_div_ep, ep_ws * C, H]

        # 真HCCL all_to_all（EP 组）
        a2a_out = torch.zeros_like(x1)
        dist.all_to_all_single(a2a_out, x1, group=ep_group)
        torch.npu.synchronize()

        # reshape + permute（对齐 mc2_test 行 184）
        a2a_out = a2a_out.reshape(reshape_1).permute(1, 0, 2, 3).contiguous()

        # 真HCCL all_gather（TP 组）
        ag_out = torch.zeros(tensor_ag_shape, dtype=x1.dtype, device=f"npu:{rank}")
        dist._all_gather_base(ag_out, a2a_out, group=tp_group)
        torch.npu.synchronize()

        # reshape + permute（对齐 mc2_test 行 188-194）
        ag_out = ag_out.reshape(reshape_2)
        if shard_type == 0:
            ag_out = ag_out.permute(1, 2, 3, 0, 4).contiguous()
        else:
            ag_out = ag_out.permute(1, 2, 0, 3, 4).contiguous()
        gather_output = ag_out.reshape(reshape_3)

        # bmm（对齐 mc2_test 行 198）
        bmm_out = torch.bmm(gather_output, weight)

        # bias add（对齐 mc2_test 行 200-204）
        if is_bias and bias is not None:
            if bias.dim() == 2:
                bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
            bmm_out = bmm_out + bias

        # activation（对齐 mc2_test 行 205）
        act_out = _activate_npu(bmm_out, act_type)

        # 写回 main output（activation 后）
        main_cpu = act_out.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, main_cpu.numpy())

        # allgather output
        if need_ag_out:
            ag_cpu = gather_output.reshape(-1).contiguous().cpu()
            _append_a2a_result(result_path, rank, ag_cpu.numpy())

        # bmm output（activation 前）
        if need_act_feat:
            bmm_cpu = bmm_out.reshape(-1).contiguous().cpu()
            _append_mm_result(result_path, rank, bmm_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_a2a_ag_bmm_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    ep_ws: int,
    tp_ws: int,
    shard_type: int,
    is_trans: bool = False,
    is_bias: bool = False,
    act_type: int = 0,
    need_ag_out: bool = True,
    need_act_feat: bool = False,
) -> Dict[int, "object"]:
    """aclnnAlltoAllAllGatherBatchMatMul 真级联 golden。

    流程：all_to_all(EP) -> all_gather(TP) -> bmm -> [bias + act] -> output
    对齐 mc2_test aclnnAlltoAllAllGatherBatchMatMul.get_hccl_mm 行 179-206。

    返回：{did: {'main': tensor, 'allgather': tensor|None, 'bmm': tensor|None}}
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}
    if n != ep_ws * tp_ws:
        raise ValueError(f"A2A_AG_BMM: world_size={n} != ep_ws*tp_ws={ep_ws * tp_ws}")

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_a2aagbmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_a2a_ag_bmm,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    ep_ws,
                    tp_ws,
                    shard_type,
                    is_trans,
                    is_bias,
                    act_type,
                    need_ag_out,
                    need_act_feat,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        ag_outs = _load_cascade_a2a_outputs(result_path, device_ids) if need_ag_out else {}
        bmm_outs = _load_cascade_mm_outputs(result_path, device_ids) if need_act_feat else {}
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 推断输出 shape（按 output_tensor_indexes 顺序：main/allgather/bmm）
        first_ctx = thread_contexts[device_ids[0]]
        out_idxs = first_ctx.output_tensor_indexes
        out_shapes = [first_ctx.tensor_view_shapes[oi] for oi in out_idxs]

        result = {}
        for did in device_ids:
            main_t = main_outs[did].reshape(out_shapes[0])
            ag_t = (
                ag_outs[did].reshape(out_shapes[1]) if need_ag_out and did in ag_outs and len(out_shapes) > 1 else None
            )
            bmm_t = (
                bmm_outs[did].reshape(out_shapes[2])
                if need_act_feat and did in bmm_outs and len(out_shapes) > 2
                else None
            )
            result[did] = {"main": main_t, "allgather": ag_t, "bmm": bmm_t}
        return result


# ============================================================
# aclnnMatmulAllReduce: matmul -> all_reduce(SUM)
# ============================================================
# 全局通信域（无 EP/TP 分组）
# 参考 mc2_test/op_class/aclnnMatmulAllReduce.py:66-72 get_hccl_mm:
#   output = torch.matmul(x1, x2)
#   if is_bias: output += bias
#   dist.all_reduce(output, op=SUM)
#   return output


def _worker_matmul_allreduce(
    rank, world_size, port, input_path, result_path, transpose_x1, transpose_x2, is_bias, error_path, timeout=3600
):
    """子进程：单 rank 跑 matmul + 真HCCL all_reduce(SUM)。"""
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from torch.distributed import ReduceOp

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = (
            tensors[2].npu() if is_bias and 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0 else None
        )

        # 转置处理
        input_mat = x1.t().contiguous() if transpose_x1 else x1
        weight_mat = x2.t().contiguous() if transpose_x2 else x2

        # matmul（int8/fp8 weight需转float，plain matmul不支持这些dtype）
        w_dtype_str = str(weight_mat.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            output = torch.matmul(input_mat.float(), weight_mat.float())
        else:
            output = torch.matmul(input_mat, weight_mat)
        if bias is not None:
            output = output + bias

        # 真HCCL all_reduce（SUM，全局通信域）
        dist.all_reduce(output, op=ReduceOp.SUM)
        torch.npu.synchronize()

        # 写回
        out_cpu = output.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, out_cpu.numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_matmul_allreduce_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    is_bias: bool = False,
) -> Dict[int, "object"]:
    """aclnnMatmulAllReduce 真级联 golden。

    流程：matmul -> [bias] -> all_reduce(SUM) -> output
    对齐 mc2_test aclnnMatmulAllReduce.get_hccl_mm 行 66-72。

    返回：{did: {'main': tensor}}
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_mm_ar_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_matmul_allreduce,
                args=(rank, n, port, input_path, result_path, transpose_x1, transpose_x2, is_bias, error_path, timeout),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 推断输出 shape
        first_ctx = thread_contexts[device_ids[0]]
        out_idx = first_ctx.output_tensor_indexes[0]
        out_shape = first_ctx.tensor_view_shapes[out_idx]

        result = {}
        for did in device_ids:
            main_t = main_outs[did].reshape(out_shape)
            result[did] = {"main": main_t}
        return result


# ============================================================
# aclnnAllGatherMatmulV2 (量化路径): all_gather(x1) -> [all_gather(x1scale)] -> npu_quant_matmul
# ============================================================
# 参考 mc2_test/op_class/aclnnAllGatherMatmulV2.py:173-199 get_hccl_mm:
#   dist._all_gather_base(tensor_allgather, x1)
#   if is_mxFp or per_block_flag:
#       dist._all_gather_base(tensor_allgather_x1scale, x1scale)
#       x1scale = tensor_allgather_x1scale
#   output = torch_npu.npu_quant_matmul(tensor_allgather, x2, pertoken_scale=x1scale, scale=x2scale, ...)
#   gather_output = tensor_allgather
#
# flat_tensors 布局（V2）: [x1, x2, bias, x1scale, x2scale, ...]
#   slot 0: x1 (fp8/hif8)
#   slot 1: x2 (fp8/hif8)
#   slot 2: bias (fp32, 可选)
#   slot 3: x1scale (fp32 per_tensor / fp8_e8m0 mxfp&per_block)
#   slot 4: x2scale (fp32 per_tensor / fp8_e8m0 mxfp&per_block)


# dtype enum for npu_quant_matmul (mc2_test DTYPE_ENUM func.py:95)
_QUANT_DTYPE_ENUM = {
    "fp8_e8m0": 293,
    "float8_e8m0": 293,
    "hif8": 290,
    "hifloat8": 290,
}


def _worker_allgather_quant_matmul_v2(
    rank,
    world_size,
    port,
    input_path,
    result_path,
    is_trans_b,
    is_bias,
    is_mxfp,
    per_block_flag,
    x1_dtype_str,
    x2_dtype_str,
    x1scale_dtype_str,
    x2scale_dtype_str,
    out_dtype_str,
    group_size,
    error_path,
    timeout=3600,
):
    """子进程：单 rank 跑 真HCCL all_gather(x1) + [all_gather(x1scale)] + npu_quant_matmul。"""
    import datetime
    import traceback

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    try:
        _configure_hccl_env(timeout)

        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=timeout),
        )

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0]  # fp8/hif8 torch tensor
        x2 = tensors[1]
        bias = (
            tensors[2].npu() if is_bias and 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0 else None
        )
        x1scale = tensors[3] if 3 in tensors and tensors[3] is not None else None
        x2scale = tensors[4] if 4 in tensors and tensors[4] is not None else None

        # 真HCCL all_gather(x1)（全局通信域，_all_gather_base = all_gather_into_tensor）
        # x1 是 fp8/hif8 tensor，HCCL 支持按 byte 通信
        x1_npu = x1.npu()
        gathered_x1_flat = torch.empty(world_size * x1.numel(), dtype=x1.dtype, device=f"npu:{rank}")
        dist._all_gather_base(gathered_x1_flat, x1_npu)
        torch.npu.synchronize()
        gathered_x1 = gathered_x1_flat.view(world_size, *x1.shape).reshape(world_size * x1.shape[0], *x1.shape[1:])

        # x2 / x2scale 直接用本 rank 的（weight 共享语义，各 rank 相同）
        x2_npu = x2.npu()
        x2s_npu = x2scale.npu() if x2scale is not None else None

        # x1scale: per_tensor 用本 rank 的；mxfp/per_block 需 all_gather
        if is_mxfp or per_block_flag:
            if x1scale is not None:
                x1s_npu_src = x1scale.npu()
                gathered_x1s_flat = torch.empty(world_size * x1scale.numel(), dtype=x1scale.dtype, device=f"npu:{rank}")
                dist._all_gather_base(gathered_x1s_flat, x1s_npu_src)
                torch.npu.synchronize()
                x1s_npu = gathered_x1s_flat.view(world_size, *x1scale.shape).reshape(
                    world_size * x1scale.shape[0], *x1scale.shape[1:]
                )
            else:
                x1s_npu = None
        else:
            x1s_npu = x1scale.npu() if x1scale is not None else None

        # e8m0 scale 需要 view 为 float8_e8m0（mc2_test common.py:332）
        if x1s_npu is not None and x1s_npu.dtype == torch.uint8 and hasattr(torch, "float8_e8m0"):
            x1s_npu = x1s_npu.view(torch.float8_e8m0)
        if x2s_npu is not None and x2s_npu.dtype == torch.uint8 and hasattr(torch, "float8_e8m0"):
            x2s_npu = x2s_npu.view(torch.float8_e8m0)

        # 输出 dtype：别名归一后转 torch，未识别回落 bfloat16
        from ...utilities.dtypes import DTYPE_MAP, str_to_torch_dtype

        out_dtype = str_to_torch_dtype(DTYPE_MAP.get(out_dtype_str, out_dtype_str)) or torch.bfloat16

        # 构建 npu_quant_matmul kwargs（对齐 mc2_test get_hccl_mm 行 185-199）
        npu_kwargs = dict(  # noqa: C408 保持
            scale=x2s_npu,
            pertoken_scale=x1s_npu,
            bias=bias,
            output_dtype=out_dtype,
            offset=None,
            y_scale=None,
        )
        # dtype enum 仅 e8m0/hif8 需要显式传
        x1_enum = _QUANT_DTYPE_ENUM.get(x1_dtype_str)
        x2_enum = _QUANT_DTYPE_ENUM.get(x2_dtype_str)
        x1s_enum = _QUANT_DTYPE_ENUM.get(x1scale_dtype_str)
        x2s_enum = _QUANT_DTYPE_ENUM.get(x2scale_dtype_str)
        if x1_enum is not None:
            npu_kwargs["x1_dtype"] = x1_enum
        if x2_enum is not None:
            npu_kwargs["x2_dtype"] = x2_enum
        if x1s_enum is not None:
            npu_kwargs["pertoken_scale_dtype"] = x1s_enum
        if x2s_enum is not None:
            npu_kwargs["scale_dtype"] = x2s_enum
        if is_mxfp:
            npu_kwargs["group_sizes"] = None

        # npu_quant_matmul（NPU 真算子）
        output = torch_npu.npu_quant_matmul(gathered_x1, x2_npu, **npu_kwargs)
        torch.npu.synchronize()

        # main output 写回（转 fp32 便于 npz 保存）
        out_cpu = output.reshape(-1).contiguous().cpu()
        _append_result(result_path, rank, out_cpu.float().numpy())

        # gather output（可选，对齐 mc2_test gather_output_golden）
        gather_cpu = gathered_x1.reshape(-1).contiguous().cpu()
        _append_a2a_result(result_path, rank, gather_cpu.view(torch.uint8).numpy())

        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        raise


def run_allgather_quant_matmul_v2_cascade(
    thread_contexts: Dict[int, "object"],
    device_ids: List[int],
    is_trans_b: bool = False,
    is_bias: bool = False,
    is_mxfp: bool = False,
    per_block_flag: bool = False,
    is_gather_output: bool = False,
) -> Dict[int, "object"]:
    """aclnnAllGatherMatmulV2 量化路径真级联 golden。

    流程：all_gather(x1) -> [all_gather(x1scale) if mxfp/per_block] -> npu_quant_matmul -> output
    对齐 mc2_test aclnnAllGatherMatmulV2.get_hccl_mm 行 173-199。

    返回：{did: {'main': tensor, 'gather': tensor|None}}
    """
    import torch.multiprocessing as mp

    timeout = _get_timeout(thread_contexts, device_ids)
    _configure_hccl_env(timeout)
    n = len(device_ids)
    if n < 2:
        return {}

    # 从 first_ctx 解析 dtype 信息
    first_ctx = thread_contexts[device_ids[0]]
    flat_dtypes = list(first_ctx.flat_tensor_dtypes or [])
    x1_dtype_str = flat_dtypes[0] if len(flat_dtypes) > 0 else "float8_e4m3fn"
    x2_dtype_str = flat_dtypes[1] if len(flat_dtypes) > 1 else "float8_e4m3fn"
    x1scale_dtype_str = flat_dtypes[3] if len(flat_dtypes) > 3 else "float32"
    x2scale_dtype_str = flat_dtypes[4] if len(flat_dtypes) > 4 else "float32"
    out_dtypes = first_ctx.flat_output_dtypes if first_ctx.flat_output_dtypes else []
    out_dtype_str = out_dtypes[0] if len(out_dtypes) > 0 else "bfloat16"
    attrs = first_ctx.attributes or {}
    group_size = attrs.get("groupSize", 0)

    port = _next_port()
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_agqmv2_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)

        ctx = mp.get_context("forkserver")
        procs = []
        for rank in range(n):
            p = ctx.Process(
                target=_worker_allgather_quant_matmul_v2,
                args=(
                    rank,
                    n,
                    port,
                    input_path,
                    result_path,
                    is_trans_b,
                    is_bias,
                    is_mxfp,
                    per_block_flag,
                    x1_dtype_str,
                    x2_dtype_str,
                    x1scale_dtype_str,
                    x2scale_dtype_str,
                    out_dtype_str,
                    group_size,
                    error_path,
                    timeout,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=_get_timeout(thread_contexts, device_ids))

        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"cascade worker rank exited with code {p.exitcode}\n{error_msg}")

        main_outs = _load_cascade_outputs(result_path, device_ids)
        gather_outs = _load_cascade_a2a_outputs(result_path, device_ids) if is_gather_output else {}
        if len(main_outs) != n:
            raise RuntimeError(f"cascade main result incomplete: got {len(main_outs)}/{n} ranks\n{error_msg}")

        # 推断输出 shape
        out_idxs = first_ctx.output_tensor_indexes
        out_shapes = [first_ctx.tensor_view_shapes[oi] for oi in out_idxs]

        result = {}
        for did in device_ids:
            main_t = main_outs[did].reshape(out_shapes[0])
            gather_t = None
            if is_gather_output and did in gather_outs and len(out_shapes) > 1:
                gather_t = gather_outs[did].reshape(out_shapes[1])
            result[did] = {"main": main_t, "gather": gather_t}
        return result
