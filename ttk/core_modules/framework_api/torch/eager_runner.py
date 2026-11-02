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
torch eager 模式执行器 —— 逐轮策略 + Tensor 方法分发 + backward（requires_grad / autograd.grad）。
"""


def run_torch_eager_rounds(
    adapter,
    ctx,
    *,
    api_name,
    args,
    kwargs,
    inplace_input_indexes,
    mutable_loader,
    profiler,
    backward_mode,
):
    """torch eager 逐轮策略：单采集窗内 warmup + 正确性轮。

    每轮先经 mutable_loader 装载可变槽载体（引用替换，克隆在窗口外预置），
    原地算子只改克隆、不碰源张量；确定性轮同时在窗口内取 MD5。返回
    (result, result_nps, md5_list)，result_nps 为末轮输出的 numpy。
    """
    from ...kernel.device_state import clear_device_state
    from ..utils import compute_output_md5, output_nps, round_nps

    # ctx.resolved 由 worker 共享给 graph，包装产物仅在 eager 内使用、不回写 ctx
    resolved = adapter.wrap_eager(ctx.resolved, api_name, ctx.backend)
    result = None
    md5_list = []
    deterministic = ctx.deterministic_level > 0
    total = ctx.warmup_count + ctx.run_count
    with profiler:
        for i in range(total):
            clear_device_state(ctx.switches)  # 同进程轮前 RTS 清（裸 aclrt 与 torch_npu 共存）
            is_warmup = i < ctx.warmup_count
            mutable_loader(i)
            with ctx.backend.device_scope(ctx.dev_id):
                r = adapter.eager_call(resolved, args, kwargs, ctx.is_tensor_method)
                if backward_mode:
                    # forward（采集窗口【外】）→ 只采反向（resume 动态窗口；
                    # torch_npu resume 支持为真机验证点，不支持时 degrade 窗口全收）
                    grads = adapter.eager_backward(ctx.backend, r, args)
                    r = (r, grads) if grads is not None else r
                if is_warmup:
                    profiler.step()
                    continue
                result = r
                if deterministic:
                    ctx.backend.synchronize(ctx.dev_id)
                    run_nps = round_nps(
                        ctx.backend,
                        r,
                        args,
                        inplace_input_indexes,
                        api_name=api_name,
                        kwargs=kwargs,
                        adapter=adapter,
                        backward_mode=backward_mode,
                    )
                    md5_list.append(compute_output_md5(run_nps))
        ctx.backend.synchronize(ctx.dev_id)
    result_nps = output_nps(
        ctx.backend,
        result,
        args,
        kwargs,
        inplace_input_indexes,
        api_name=api_name,
        adapter=adapter,
        backward_mode=backward_mode,
    )
    return result, result_nps, md5_list


def call(resolved, args, kwargs, is_tensor_method):
    """单次调用：Tensor 方法经 args[0] 分发（resolved 为方法名字符串）。"""
    if is_tensor_method:
        return getattr(args[0], resolved)(*args[1:], **kwargs) if args[0] is not None else None
    return resolved(*args, **kwargs)


def enable_requires_grad(backend, args):  # noqa: ARG001
    """backward：浮点输入置 requires_grad（克隆后轮结构零改动——clone 保留标志）。"""
    for arg in args:
        if hasattr(arg, "requires_grad_") and hasattr(arg, "is_floating_point") and arg.is_floating_point():
            arg.requires_grad_(True)


def run_backward(backend, output, args):  # noqa: ARG001
    """backward：autograd.grad（函数式）——梯度是返回值，unused 位直接过滤丢弃。
    设计支点：多轮零状态（无 .grad 累积）+ 结果采集复用正向体系。"""
    import torch
    from torch import autograd

    grad_inputs = [a for a in args if hasattr(a, "requires_grad") and a.requires_grad]
    if not grad_inputs or output is None or not getattr(output, "requires_grad", False):
        return None
    if isinstance(output, (tuple, list)):
        outputs = [o for o in output if getattr(o, "requires_grad", False)]
    else:
        outputs = [output]
    if not outputs:
        return None
    # 非标量输出需显式 grad_outputs（ones_like——全 1 种子，与数值内容无关）
    grad_outputs = [torch.ones_like(o) for o in outputs]
    grads = autograd.grad(outputs, grad_inputs, grad_outputs=grad_outputs, allow_unused=True)
    return [g for g in grads if g is not None] if grads else None
