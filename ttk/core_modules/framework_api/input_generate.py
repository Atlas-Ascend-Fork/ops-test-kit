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

from ttk.core_modules.common.testcase.batch import batch_relation_kwargs
from ttk.core_modules.common.testcase.input_data import (
    assign_tensor_value,
    broadcast_tensorlist_values,
    generate_np_storages,
)
from ttk.core_modules.common.testcase.tensor import make_storage_view
from ttk.utilities.container import nest_by_distribution

from ..common.invoke import accepts_var_keyword, sig_params
from ..common.plugin import get_plugin_function


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
    """Generate numpy input arrays. Default generation first, then optional custom plugin.

    Custom input plugin interface (signature matches API definition, same as golden plugin):
        def my_input_torch_gather(input, dim, index, sparse_grad=False, out=None):
            input[:] = np.zeros_like(input)
            index[:] = np.random.randint(0, input.shape[1], size=index.shape, dtype=index.dtype)

    Plugin has no return value, modifies tensor arrays in-place via x[:] = value.
    Called with same ParamPlan arg order as profiling execution and golden generation.

    cpu_placement=True：框架张量物化在 CPU 作用域（输入生成/golden 阶段的
    形态；设备张量由执行侧另行重建）。
    """
    if stored_inputs is not None:
        testcase.np_storages = list(stored_inputs)
        raw_inputs = build_views_from_storages(testcase)
        set_runtime_tensors(testcase, raw_inputs, backend, cpu_placement=cpu_placement)
        return raw_inputs

    raw_inputs = default_generate_inputs(testcase, switches)
    override_tensors_from_attributes(testcase, raw_inputs)

    plugin_path = switches.plugin_path
    input_func = get_plugin_function(testcase.api_name, "input", "e2e", plugin_path)
    if input_func is not None:
        with _cpu_placement_scope(backend, cpu_placement):
            plugin_inputs = backend.inputs_from_numpy(testcase, raw_inputs)
        use_numpy = backend.needs_numpy_fallback(testcase)
        dist = testcase.tensor_list_dist
        nested_for_plugin = nest_by_distribution(plugin_inputs, dist) if dist else plugin_inputs
        args, kwargs, extra_attrs = plan.build_args(nested_for_plugin)
        extra = {
            "backend": backend.device_type(),
            "tensor_formats": testcase.tensor_formats,
            "tensor_dtypes": testcase.tensor_dtypes,
            "use_numpy": use_numpy,
            "short_soc_version": switches.short_soc_version,
            "testcase_name": testcase.testcase_name,
            "input_ranges": testcase.input_data_ranges,
        }
        extra.update(extra_attrs)

        extra.update(batch_relation_kwargs(testcase))

        params = sig_params(input_func)
        if accepts_var_keyword(params):
            kwargs.update(extra)
        else:
            kwargs.update({k: v for k, v in extra.items() if k in params})
        input_func(*args, **kwargs)
        # 重建 testcase.tensors，并把插件在输入上置的 requires_grad 重放到重建后的
        # tensor 上（经 set_runtime_tensors 的 plugin_tensors 参数）。这样
        # testcase.tensors 始终是重建产物（净化语义统一），插件对象不进 runtime tensors。
        #
        # 背景：op-plugin 一类算子会按输入 requires_grad 门控（如
        # RotaryMulBackwardKernelNpuOpApi.cpp 用 r1.requires_grad() 短路 xOptional
        # 输入，requires_grad=False 时只算 dx、dcos/dsin 全 0）。ttk 此前无条件从
        # raw_inputs 重建 testcase.tensors，会丢弃插件置的 requires_grad，导致这类
        # 算子 e2e 无法比对 dcos/dsin。插件按约定以 x[:] = value 原地改数据
        # （from_numpy 共享内存），数据与重建路径一致；此处仅重放 requires_grad。
        set_runtime_tensors(testcase, raw_inputs, backend, plugin_tensors=plugin_inputs, cpu_placement=cpu_placement)
    else:
        set_runtime_tensors(testcase, raw_inputs, backend, cpu_placement=cpu_placement)
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


def override_tensors_from_attributes(testcase, raw_inputs):
    """Override tensor values from testcase attributes dict by parameter name.

    For regular tensor: val is scalar or array broadcastable to target shape.
    For TensorList, two modes:
      1. Scalar broadcast: val is a single number or [scalar] -> all sub-tensors get same value
      2. Per-tensor: val is a list, len must match sub-tensor count -> applied one-by-one

    Records tensor indexes sourced from attributes in testcase.const_input_indexes
    so backends can create const tensors (tf.constant / torch.tensor) for them.
    """
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
    inputs = []
    for idx, storage in enumerate(testcase.np_storages):
        if storage is None:
            inputs.append(None)
            continue
        view_shape = flat_shapes[idx]
        v_stride = testcase.flat_view_stride(idx)
        v_offset = testcase.flat_view_offset(idx)

        if v_stride is not None or v_offset != 0:
            inputs.append(to_non_contiguous_view(storage, view_shape, v_stride, v_offset))
        else:
            inputs.append(storage)
    return inputs


def to_non_contiguous_view(storage, view_shape, view_stride, view_offset):
    """Create non-contiguous view from contiguous storage using numpy as_strided."""
    return make_storage_view(storage, view_shape, view_stride, view_offset)
