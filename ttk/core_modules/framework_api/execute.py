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
E2E eager 执行 —— 设备张量构建 + 单算子执行 + 采集。

框架差异经 FrameworkAdapter 钩子（wrap_eager / eager_call / eager_*）；
图模式见 torch/graph_runner 与 tf/graph_runner。
"""

import logging

from ttk.core_modules.common.testcase.batch import resolve_deterministic_level

from ..common.invoke import Invoke
from ..common.profiler import ProfilerConfig, get_profiler
from .utils import (
    compute_output_md5,
    finalize_det_status,
    prepare_device_args,
    resolve_npu_preprocess_hook,
    unpack_4bit_outputs,
)

WARMUP_COUNT = 1


def _capture_peak_memory(backend, dev_id, return_struct, fn, *args, **kwargs):
    """NPU: reset peak stats, run *fn*, record max_memory_allocated (MB) into npu_memory.

    Same caliber as the GPU executor (remote/server/executor.py: reset_peak ->
    invoke -> max_memory_allocated / 1e6): peak covers input allocations and
    execution-time temporaries inside *fn*. CANN-internal workspace does not go
    through the torch allocator and is not counted (GPU cuBLAS/cuDNN workspace
    is likewise excluded). Non-NPU backends pass through, npu_memory stays None.

    Memory-stat APIs are best-effort: any failure (like the GPU executor's
    peak_memory_mb=NA) leaves npu_memory untouched instead of failing the case,
    and never masks fn's own result/exception.
    """
    if not backend.is_npu():
        return fn(*args, **kwargs)
    try:
        backend.reset_peak_memory_stats(dev_id)
    except Exception:
        logging.exception(f"[E2E] reset_peak_memory_stats failed on device {dev_id}; npu_memory not captured")
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    finally:
        try:
            peak = backend.max_memory_allocated(dev_id)
        except Exception:
            logging.exception(f"[E2E] max_memory_allocated failed on device {dev_id}; npu_memory not captured")
            peak = None
        if peak is not None:
            return_struct.npu_memory = peak / 1e6


def _collect_run_nps(
    backend, r, args, inplace_input_indexes, api_name=None, kwargs=None, adapter=None, backward_mode=False
):
    """确定性 MD5 取回（重构自 inline——backward 元组 (r, grads) 两侧展开）。"""
    run_nps = []
    if r is not None:
        run_nps.extend(_result_to_numpy_flat(backend, r, backward=backward_mode))
    elif api_name is not None:
        # 无返回值有状态算子(TF raw Resource*): 逐轮回读当轮变量(克隆)的更新值
        if adapter is None:
            from .adapter import adapter_for_api_name

            adapter = adapter_for_api_name(api_name)
        run_nps.extend(adapter.mutable_readback(api_name, args, kwargs))
    if inplace_input_indexes:
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                run_nps.append(backend.to_numpy(args[idx], safe=True))
    return run_nps


def _result_to_numpy_flat(backend, result, backward=False):
    """结果展开（backward=True 时拆 backward 包装 (forward, grads) 两侧拼接为输出列表）。

    元组启发式必须带 backward 门控：包装仅在 backward_mode 产生
    （r = (r, grads) if grads is not None），原生返回 (Tensor, list[Tensor])
    形态的 API 不带门控会被误拆——普通 result_to_numpy 对 list 项走
    np.array 堆叠，与本函数的逐项展开语义不同。"""
    if backward and (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], list)
        and (result[1] and hasattr(result[1][0], "numpy") or result[1] == [])
    ):
        forward, grads = result
        nps = backend.result_to_numpy(forward) if forward is not None else []
        for g in grads:
            nps.extend(
                backend.result_to_numpy(g)
                if not isinstance(g, list)
                else [x for g2 in g for x in backend.result_to_numpy(g2)]
            )
        return nps
    return backend.result_to_numpy(result)


def _eager_mutable_keys(adapter, api_name, args, kwargs, is_inplace):
    """克隆轮换键（int = args 下标，str = kwargs 参数名）：mutable_slots 模型派生；
    torch 解析降级（命名为原地而模型无被写位）回退槽 0 惯例。"""
    positions, names = adapter.mutable_slots(api_name)
    keys = {i for i in positions if i < len(args) and args[i] is not None}
    keys |= {name for name in names if kwargs.get(name) is not None}
    if not keys and is_inplace and args and args[0] is not None:
        keys = {0}
    return keys


def _apply_e2e_compile_mode(backend, adapter, switches):
    """eager 模式的算子编译：-b 默认二进制（jitCompile=disable 复用预编译 host 代码），
    -b=false 恢复在线编译。进程级全局、幂等；仅 NPU torch 后端生效（TF/xpu 不涉及）。"""
    if not backend.is_npu() or getattr(adapter, "framework", None) != "torch":
        return
    binary = getattr(switches, "e2e_binary", False)
    logging.info("[E2E] eager aclnn compile mode: %s", "binary reuse (-b)" if binary else "online (-b=false)")
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=not binary)


def _execute_eager(
    testcase,
    backend,
    dev_id,
    switches,
    plan,
    resolved,
    is_tensor_method,
    is_inplace,
    raw_inputs,
    deterministic_level=None,
):
    """Build device tensors, run API in eager mode with profiling, return (result_nps, perf, det_status)."""
    from .adapter import adapter_for_testcase

    adapter = adapter_for_testcase(testcase)
    backend.set_device(dev_id)
    _apply_e2e_compile_mode(backend, adapter, switches)
    resolved = adapter.wrap_eager(resolved, testcase.api_name, backend)
    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs, adapter=adapter)

    if backend.is_npu():
        result = Invoke.npu_preprocess(
            testcase,
            resolve_npu_preprocess_hook(testcase, switches),
            switches,
            device_args=(args, kwargs),
            device_scope=lambda: backend.device_scope(dev_id),
        )
        testcase.absorb_preprocess(result, plan, args, kwargs)

    profiling_enabled = bool(getattr(switches, "task_profiling", True))
    if deterministic_level is None:
        deterministic_level = resolve_deterministic_level(switches, testcase)
    deterministic = deterministic_level > 0
    run_count = switches.run_time
    warmup_count = WARMUP_COUNT if (switches.warmup and profiling_enabled) else 0
    profiler = get_profiler(
        testcase.api_name,
        backend,
        ProfilerConfig(
            testcase_name=testcase.testcase_name,
            root_path=switches.root_path,
            dev_id=dev_id,
            enabled=profiling_enabled,
            warmup_count=warmup_count,
        ),
    )

    inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
    # 键约定: int = args 下标; str = kwargs 参数名(TF raw_ops 全部按名传参, args 为空)
    inplace_idxs = _eager_mutable_keys(adapter, testcase.api_name, args, kwargs, is_inplace)
    inplace_idxs.update(i for i in inplace_input_indexes if i < len(args) and args[i] is not None)

    total = warmup_count + run_count
    # 逐轮可变槽载体预置（backend 封装：torch 逐轮克隆 / tf 同变量 assign 复位）
    stage_mutable = backend.stage_mutable_rounds(args, kwargs, inplace_idxs, total)

    result = None
    md5_list = []
    # 多通路共享门面（geir/execute 同款引用）：UB/L1/L0 是设备级资源，helper
    # kernel 编译机制物理归属 kernel 通路，故门面留在 kernel/ 而非 common
    from ..kernel.device_state import clear_device_state

    backward_mode = getattr(testcase, "backward_mode", False)
    if backward_mode:
        # backward：浮点输入自动 requires_grad=True（整数索引类天然无梯度，
        # autograd.grad allow_unused 兜底）。梯度是第二组返回值（unused 位过滤
        # 丢弃，golden 侧同构），追加在 outputs 尾部——比对/判据/ dump/结果打包零改动。
        adapter.eager_prepare_backward(backend, args)

    with profiler:
        for i in range(total):
            clear_device_state(switches)  # 同进程轮前 RTS 清（裸 aclrt 与 torch_npu 共存）
            is_warmup = i < warmup_count
            stage_mutable(i)
            with backend.device_scope(dev_id):
                r = adapter.eager_call(resolved, args, kwargs, is_tensor_method)
                if backward_mode:
                    # forward（采集窗口【外】）→ 只采反向（resume 动态窗口；
                    # torch_npu resume 支持为真机验证点，不支持时 degrade 窗口全收）
                    grads = adapter.eager_backward(backend, r, args)
                    r = (r, grads) if grads is not None else r
                if is_warmup:
                    profiler.step()
                    continue
                result = r
                if deterministic:
                    backend.synchronize(dev_id)
                    run_nps = _collect_run_nps(
                        backend,
                        r,
                        args,
                        inplace_input_indexes,
                        api_name=testcase.api_name,
                        kwargs=kwargs,
                        adapter=adapter,
                        backward_mode=backward_mode,
                    )
                    md5_list.append(compute_output_md5(run_nps))
        backend.synchronize(dev_id)

    perf = profiler.result(run_count)

    # 输出 = 返回值（有）否则被写参数位回读（无返回值的有状态算子：多轮克隆下
    # args 里是末轮变量，从初值起只施加一次更新；有返回值时不回读——snake_case
    # 封装的 lazy-read 已含更新结果，重复回读会双倍输出）
    if result is not None:
        result_nps = _result_to_numpy_flat(backend, result, backward=backward_mode)
    else:
        result_nps = adapter.mutable_readback(testcase.api_name, args, kwargs) or None

    if inplace_input_indexes:
        if result_nps is None:
            result_nps = []
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                inplace_np = backend.to_numpy(args[idx], safe=True)
                result_nps.append(inplace_np)

    det_status = finalize_det_status(md5_list, testcase.testcase_name)

    result_nps = unpack_4bit_outputs(testcase, result_nps)
    return result_nps, perf, det_status


def call_api(api_name, matched_oidx, resolved, args, kwargs):
    """Call resolved API with error logging.

    Safe wrapper that logs TypeError details before re-raising.
    Used by worker.py and golden.py.
    """
    try:
        return resolved(*args, **kwargs)
    except TypeError as e:
        logging.warning(
            f"API call {api_name}(oidx={matched_oidx}) failed with TypeError: {e}. "
            f"args={len(args)}, kwargs={list(kwargs.keys())}"
        )
        raise
