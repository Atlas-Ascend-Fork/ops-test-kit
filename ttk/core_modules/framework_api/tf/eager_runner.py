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
tf eager 模式执行器 —— 调用包装 + mutable 位回读。

wrap_eager_callable 自 common/backend/{tf_backend, npu_tf_backend}.wrap_eager_callable
迁入（断链的终态修复：backends 不再依赖通路层 tf_stateful）：
  - NPU：npu_device 只对 tf.function 内的 op 触发 GE 编译（真 NPU kernel），
    裸 eager 调用会静默回落 CPU；tf.function(autograph=False) 保持单算子语义。
  - CPU：仅 tf.raw_ops.Resource* 的 DT_RESOURCE 位需把 tf.Variable 换 .handle。
"""

import inspect


def wrap_eager_callable(resolved, api_name, backend):
    """按 backend 设备形态包装 eager 调用（TfAdapter.wrap_eager 委托）。"""
    if backend.is_npu():
        return _wrap_eager_npu(resolved, api_name)
    return _wrap_eager_cpu(resolved, api_name)


def _wrap_eager_cpu(resolved, api_name):
    """CPU eager: tf.raw_ops.Resource* 需要把 DT_RESOURCE 位的 tf.Variable
    换成 .handle(Variable 在 op 边界被自动解引用为值, resource 类型校验
    必挂); 无输出的有状态 raw op 在 eager 下本就返回 None, 无需处理。
    非 Resource* API 原样返回, 不改变既有调用方式。
    """
    from .param_extractor import (
        convert_variables_to_handles,
        resource_param_names,
    )

    resource_names = resource_param_names(api_name) if api_name else frozenset()
    if not resource_names:
        return resolved

    def wrapper(*args, **kwargs):
        args = list(args)
        convert_variables_to_handles(api_name, args, kwargs)
        return resolved(*args, **kwargs)

    return wrapper


def _wrap_eager_npu(resolved, api_name):
    """NPU eager：tf.function(autograph=False) 包装，使算子经 GE 编译在 NPU 执行
    （npu_device 只对 tf.function 内的 op 触发 NPU 分发，裸 eager 落 CPU）。

    - raw_ops 按参数名绑 kwargs（tf.raw_ops 只收关键字）；
    - Resource* 的 Variable 在 trace 内换 .handle（op 边界解引用会挂类型校验）；
    - 无输出有状态 op trace 下返回 Operation，tf.function 拒绝 → 置 None。"""
    import tensorflow as tf

    from .param_extractor import resource_param_names

    try:
        sig = inspect.signature(resolved)
        param_names = [
            name
            for name, p in sig.parameters.items()
            if name != "name" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
    except (ValueError, TypeError):
        param_names = []

    resource_names = resource_param_names(api_name) if api_name else frozenset()

    if param_names:
        names = param_names

        def wrapper(*args, **kwargs):
            call_kwargs = {}
            for i, name in enumerate(names):
                if i < len(args) and args[i] is not None:
                    call_kwargs[name] = args[i]
            call_kwargs.update(kwargs)
            if resource_names:
                for name in resource_names:
                    v = call_kwargs.get(name)
                    if isinstance(v, tf.Variable):
                        call_kwargs[name] = v.handle
            r = resolved(**call_kwargs)
            return None if isinstance(r, tf.Operation) else r

    else:
        wrapper = resolved

    return tf.function(autograph=False)(wrapper)


def call(resolved, args, kwargs, is_tensor_method):  # noqa: ARG001
    """单次调用（tf 无 Tensor 方法分发，is_tensor_method 恒 False）。"""
    return resolved(*args, **kwargs)


def readback_mutable(api_name, args, kwargs):
    """被写参数位（tf.Variable）的当前值回读——结果回填与逐轮 MD5 共用（TfAdapter 钩子）。"""
    from .param_extractor import collect_mutable_variables, read_mutable_values

    return read_mutable_values(collect_mutable_variables(api_name, args, kwargs))
