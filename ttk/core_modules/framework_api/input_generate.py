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
Input data generation for framework_api tests.

Generates numpy arrays from testcase metadata, applies custom plugin overrides,
and converts to framework tensors (torch).
"""

from ttk.core_modules.common.testcase.input_data import (
    assign_tensor_value,
    broadcast_tensorlist_values,
    generate_np_storages,
)
from ttk.core_modules.common.testcase.tensor import make_storage_view
from ttk.utilities import get
from ttk.utilities.container import nest_by_distribution
from ttk.utilities.dtypes import DTYPE_MAP

from ..common.plugin import get_plugin_function
from .utils import build_plugin_call


def _cpu_placement_scope(backend, enabled):
    """张量物化作用域：TF 后端且 enabled 时强制 CPU 放置，其余返回空上下文。

    TF 侧 npu_device open 后默认设备为 NPU、tf.device 的设备参数失效，
    CPU 物化须经 context.device 原语（cpu_tf_backend.cpu_device_context）。
    torch 的 from_numpy 产物即 CPU 张量，不进作用域（也不触达 tensorflow）。"""
    if enabled and hasattr(backend, "tf_device_type"):
        from ..common.backend.cpu_tf_backend import cpu_device_context

        return cpu_device_context()
    from contextlib import nullcontext

    return nullcontext()


def generate_inputs(testcase, switches, backend, plan, stored_inputs=None, cpu_placement=False):
    """输入生成入口：replay（stored_inputs）装载，或 默认生成 → attributes 覆盖 →
    input 插件改写。插件按 ParamPlan 装配调用、无返回值（x[:] = value 原地改）。
    cpu_placement=True 时框架张量在 CPU 作用域物化（设备张量执行侧重建）。"""
    if stored_inputs is not None:
        testcase.np_storages = list(stored_inputs)
        raw_inputs = build_views_from_storages(testcase)
    else:
        raw_inputs = default_generate_inputs(testcase, switches)
        apply_attribute_tensors(testcase, raw_inputs)

    plugin_path = switches.plugin_path
    input_func = get_plugin_function(testcase.api_name, "input", "e2e", plugin_path)
    plugin_inputs = None
    if input_func is not None:
        with _cpu_placement_scope(backend, cpu_placement):
            plugin_inputs = backend.inputs_from_numpy(testcase, raw_inputs)
        use_numpy = backend.needs_numpy_fallback(testcase)
        dist = testcase.tensor_list_dist
        nested_for_plugin = nest_by_distribution(plugin_inputs, dist) if dist else plugin_inputs
        args, kwargs = build_plugin_call(
            input_func,
            testcase,
            switches,
            plan,
            nested_for_plugin,
            backend=backend.device_type(),
            use_numpy=use_numpy,
            input_ranges=testcase.input_data_ranges,
        )
        input_func(*args, **kwargs)
    # 统一重建 runtime tensors：数据/shape 恒来自 raw_inputs（插件对象不进入）；
    # 插件命中的仅回放其置的 requires_grad（部分算子按该标志门控输入）。
    set_runtime_tensors(
        testcase,
        raw_inputs,
        backend,
        plugin_tensors=plugin_inputs,
        cpu_placement=cpu_placement,
    )
    return raw_inputs


def set_runtime_tensors(testcase, raw_inputs, backend, plugin_tensors=None, cpu_placement=False):
    """Rebuild framework tensors and TensorList nesting from backing storages.

    plugin_tensors: 可选，input 插件处理过的 flat tensor 列表（与 raw_inputs 逐索引
    对齐）。重建后按索引把插件置的 requires_grad 重放到新 tensor 上——重建产物
    （数据/shape）始终来自 raw_inputs，插件对象不进 testcase.tensors，净化语义统一。
    cpu_placement: TF 侧在 CPU 作用域物化（输入生成/golden 阶段形态）；False 时
    随默认设备（open 后 = NPU，执行侧的设备张量重建用此形态）。
    """
    with _cpu_placement_scope(backend, cpu_placement):
        flat_tensors = backend.inputs_from_numpy(testcase, raw_inputs)
    if plugin_tensors is not None:
        for i, pt in enumerate(plugin_tensors):
            if i < len(flat_tensors) and pt is not None and getattr(pt, "requires_grad", False):
                flat_tensors[i].requires_grad_(True)
    dist = testcase.tensor_list_dist
    if dist:
        testcase.tensors = nest_by_distribution(flat_tensors, dist)
    else:
        testcase.tensors = flat_tensors


def apply_attribute_tensors(testcase, raw_inputs):
    """把 attributes 按参数名声明的值写进输入张量，并记 const 位。

    普通张量：标量或可广播数组；TensorList 两态：标量广播全体 / 逐张量列表。
    const_input_indexes/const_input_values 供后端建常量张量（tf.constant/torch.tensor）。"""
    info = testcase.get_api_info()
    if not info or not testcase.attributes:
        return
    dist = testcase.tensor_list_dist
    nested_np = nest_by_distribution(raw_inputs, dist) if dist else list(raw_inputs)
    tensor_params = info.tensors
    for idx, param in enumerate(tensor_params):
        if idx >= len(nested_np):
            break
        if nested_np[idx] is None:
            continue
        if param.name not in testcase.attributes:
            continue
        val = testcase.attributes[param.name]
        if param.is_tensor_like and val in (None, "None", ""):
            raise ValueError(
                f"[{testcase.testcase_name}] Invalid testcase: param '{param.name}' "
                f"has a tensor in view_shapes but attributes specifies "
                f"{repr(val)}.  Tensor params should be provided via "
                f"view_shapes or with a concrete value in attributes, not None."
            )
        testcase.const_input_indexes.add(idx)
        if param.is_tensor_list:
            sub_tensors = nested_np[idx]
            per_tensor_val = broadcast_tensorlist_values(val, len(sub_tensors), param.name)
            for j in range(len(sub_tensors)):
                assign_tensor_value(sub_tensors[j], per_tensor_val[j], f"{param.name}[{j}]")
        else:
            assign_tensor_value(nested_np[idx], val, param.name)
            # const 位记录归一化 Python 值（attributes 值已按声明 dtype 落盘），
            # 供调用边界直接以 Python 值传入：tf.function 只把 Python 标量/list
            # 折叠为 Const 节点，EagerTensor（含 tf.constant）与 ndarray 入参一律
            # trace 成 Placeholder，GE infershape 对 value-depend 输入读不到值。
            # 有 TensorList 时不记录：顶层参数下标与平铺下标错位，避免替换错位
            if not any(d > 0 for d in dist):
                if nested_np[idx].ndim == 0:
                    testcase.const_input_values[idx] = nested_np[idx].item()
                else:
                    testcase.const_input_values[idx] = nested_np[idx].tolist()


def default_generate_inputs(testcase, switches):
    """Default input generation -- flat iteration over all tensors.

    Iterates flat_tensor_view_shapes so nested TensorList structures
    are handled correctly (each sub-tensor generated independently).
    Returns flat list of numpy arrays (possibly non-contiguous views).
    """
    storages, _ = generate_np_storages(testcase, switches, zero_init_apis=("torch.ones", "tf.ones"))
    testcase.np_storages = storages
    return build_views_from_storages(testcase)


def build_views_from_storages(testcase):
    """Build (possibly non-contiguous) numpy views from contiguous storages."""
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = testcase.flat_tensor_dtypes or ()
    inputs = []
    for idx, storage in enumerate(testcase.np_storages):
        if storage is None:
            inputs.append(None)
            continue
        dtype = get(flat_dtypes, idx)
        if DTYPE_MAP.get(dtype, dtype) == "complex32":
            # complex32 存储 = fp16(声明shape+[2])：numpy 无 complex32 做不了
            # dtype 重解释，按声明形状切视图会取到半数据；视图即存储，复数语义
            # 投影由框架张量层完成（torch.complex32 / TF fp16 lanes）
            inputs.append(storage)
            continue
        view_shape = flat_shapes[idx]
        v_stride = testcase.flat_view_stride(idx)
        v_offset = testcase.flat_view_offset(idx)

        if v_stride is not None or v_offset != 0:
            inputs.append(make_storage_view(storage, view_shape, v_stride, v_offset))
        else:
            inputs.append(storage)
    return inputs
