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
torch eager 模式执行器 —— Tensor 方法分发 + backward（requires_grad / autograd.grad）。
"""


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
    """backward：autograd.grad（函数式）——梯度是返回值，unused 位直接过滤丢弃
    （golden 侧 _run_backward_on_cpu 同构过滤，两侧对称、不按输入位对齐）。
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
