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
tf tensor 绑定 —— L1 张量构造（np → tf.Tensor/tf.Variable）+ L2 绑定特化。

L1（np_to_tf_inputs）：
  - TF 张量不可变，非连续 view 无法 as_strided，改按视图生成连续张量；
  - attributes 来源的 0-D/标量位以 tf.constant 物化——dtype 按声明钉死（含量化
    dtype），是 testcase.tensors 上的通用表示；注意 tf.constant 在 tf.function
    下 trace 成 Placeholder、并不折叠为 Const 节点，真正的图模式 Const 折叠由
    L2 以 Python 值（const_input_values）替换实现，tf.constant 形态仅在值
    未被记录时（如 TensorList 位）作兜底；
  - mutable 参数（ParamInfo.is_mutable，OpDef 填充）包成 tf.Variable（见 param_extractor.py）。
  经 common/backend/framework_hooks 注册表注入 TfBackend.inputs_from_numpy——
  common 不反向 import 通路（依赖方向约束）。

L2（bind_tensors）：mutable-ref 变量复位 + const 位以 Python 值传入。
"""

import numpy as np

from ...common.backend.framework_hooks import register_input_factory


def np_to_tf_inputs(testcase, raw_inputs):
    """Convert numpy arrays to tf.Tensor.

    TF tensors are immutable, so non-contiguous views cannot use as_strided.
    Instead, we generate contiguous tensors from the raw numpy views.
    Attribute-sourced slots are materialized as tf.constant with the declared
    dtype pinned (incl. quantized dtypes) — a backend-agnostic representation.
    tf.constant does NOT fold into Const nodes under tf.function (it traces as
    a Placeholder); graph-mode const folding happens later in bind_tensors,
    which swaps these slots to plain Python values (const_input_values). The
    tf.constant form remains the fallback when no Python value was recorded
    (e.g. TensorList positions).
    Mutable-ref params (OpDef is_ref) are wrapped as tf.Variable — see
    param_extractor.py.
    """
    import tensorflow as tf

    from ttk.utilities import get
    from ttk.utilities.dtypes import normalize_to_tf_dtype, tf_quantized_dtype

    from .param_extractor import mutable_flat_indexes

    const_indexes = getattr(testcase, "const_input_indexes", None) or set()
    # mutable 下标(ParamInfo.is_mutable, 张量序位次)与张量参数列表对位;
    # TensorList 参数会造成 flat 下标偏移, 而 mutable-ref 算子没有 TensorList
    # 参数, 此时直接不启用 (dist 编码: 0=普通张量, >0=TensorList 子张量数)
    dist = testcase.tensor_list_dist or ()
    mutable_idx = set(mutable_flat_indexes(testcase.api_name)) if not any(d > 0 for d in dist) else set()
    flat_dtypes = getattr(testcase, "flat_tensor_dtypes", None) or ()
    result = []
    for idx, arr in enumerate(raw_inputs):
        if arr is None:
            result.append(None)
            continue
        # 0-D 本就连续；ascontiguousarray 会把标量提升为 (1,)，破坏 TF 0-D 参数校验
        contiguous = np.ascontiguousarray(arr) if arr.ndim else arr
        contiguous = normalize_to_tf_dtype(contiguous)
        # TF 量化 dtype 的 numpy 存储是普通整型(uint8/int8/...)，仅凭存储 dtype
        # 会推断成 tf.uint8 等；须按声明 dtype 显式传 dtype=tf.quint8 恢复语义
        tf_dtype = tf_quantized_dtype(get(flat_dtypes, idx))
        if idx in const_indexes:
            # dtype 钉死的通用物化；图模式 Const 折叠由 L2 bind_tensors 以
            # Python 值替换实现（tf.constant 在 tf.function 下是 Placeholder）
            result.append(tf.constant(contiguous, dtype=tf_dtype))
        elif idx in mutable_idx:
            # tf.Variable 随默认设备放置; 勿强制 CPU 放置(见 param_extractor 模块 docstring)
            result.append(tf.Variable(contiguous, dtype=tf_dtype))
        else:
            result.append(tf.convert_to_tensor(contiguous, dtype=tf_dtype))
    return result


def bind_tensors(testcase, backend, dev_tensors, raw_inputs):
    """L2 绑定特化（TfAdapter.bind_tensors 委托）。

    1. mutable-ref 变量复位：eager/graph/ golden 三个模式共用同一批 tf.Variable
       （testcase.tensors 只创建一次），每个模式入口按 raw_inputs（原始 numpy，
       不被任何执行阶段改写）复位到初值。
    2. const 位以记录的 Python 值传入：tf.function 只把 Python 值（0-D 标量 /
       N-D list）折叠为 Const 节点，EagerTensor（含 tf.constant）与 ndarray 入参
       一律 trace 成 Placeholder，GE infershape 对 value-depend 输入读不到值。
    """
    from .param_extractor import reset_ref_variables

    reset_ref_variables(dev_tensors, raw_inputs)
    const_values = getattr(testcase, "const_input_values", None) or {}
    if const_values and getattr(backend, "tf_device_type", None):
        dev_tensors = [const_values.get(i, t) for i, t in enumerate(dev_tensors)]
    return dev_tensors


# 注册 L1 张量构造：TfBackend.inputs_from_numpy 经 framework_hooks 取用。
# tf/ 包导入（get_adapter("tf")）即完成注册；worker 在输入生成前创建 adapter。
register_input_factory("tf", np_to_tf_inputs)
