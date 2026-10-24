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
Input generation method for Universal testcases
"""

# Standard Packages
import logging

import numpy

from ttk.switches import get_global_storage

from ...utilities import (
    RandomData,
    deep_flatten,
    get,
    load_array,
    nest_by_distribution,
    promote_scalar_shapes,
    resolve_numpy_dtypes,
)
from ...utilities import format as format_transformation

# Third-party Packages
from ..common.plugin import get_plugin_function
from ..common.testcase.batch import batch_relation_kwargs, has_complete_batch_relation
from ..common.testcase.batch_replace import apply_slot_batch_replacement
from ..common.testcase.input_data import assign_tensor_value, broadcast_tensorlist_values
from .info_keeper import OpInfoKeeper
from .params import adapt_pickup_by_names
from .testcase import TestcaseKernel


def _use_manual_input(context: TestcaseKernel):
    flat_binaries = context.flat_manual_input_binaries
    flat_shapes = promote_scalar_shapes(context.flat_input_shapes)
    flat_dtypes = resolve_numpy_dtypes(context.flat_input_dtypes)

    input_arrays = []
    for idx, fp in enumerate(flat_binaries):
        shape = get(flat_shapes, idx)
        if shape is None or fp is None:
            input_arrays.append(None)
            continue
        logging.info(f"Reading manual input file: {fp} ...")
        input_arrays.append(load_array(fp, get(flat_dtypes, idx), shape))

    # flat → nested (与 input_shapes 同构)
    nested_arrays = nest_by_distribution(input_arrays, context.input_distribution)
    context.input_arrays = tuple(nested_arrays)
    context.actual_input_data_ranges = ((None, None),)


def _need_transform(ori_format, format_, ori_shape, shape):
    if ori_format == format_:
        return False
    if (ori_shape is None) != (shape is None):
        raise ValueError(
            f"ori_shape and shape must be both None or both not None, got ori_shape={ori_shape}, shape={shape}"
        )
    if ori_shape is None:
        return False
    return not (tuple(ori_shape) == tuple(shape) and (ori_format == "ND" or format_ == "ND"))


def _transform_single_to_ori(context, arr, nested_ori, group_idx, sub_idx):
    """Transform array from current format to ori_format, update nested_ori inplace."""
    dist = context.input_distribution or ()
    flat_idx = sum(max(d, 1) for d in dist[:group_idx])
    if sub_idx is not None:
        flat_idx += sub_idx

    ori_format = get(context.flat_input_ori_formats, flat_idx)
    format_ = get(context.flat_input_formats, flat_idx)
    flat_shapes = promote_scalar_shapes(context.flat_input_shapes)
    flat_ori_shapes = promote_scalar_shapes(context.flat_input_ori_shapes)
    shape_ = get(flat_shapes, flat_idx)
    ori_shape = get(flat_ori_shapes, flat_idx)

    if not _need_transform(ori_format, format_, ori_shape, shape_):
        if sub_idx is not None:
            nested_ori[group_idx][sub_idx][:] = arr
        else:
            nested_ori[group_idx][:] = arr
        return

    if not format_transformation.is_transformable(format_, ori_format):
        raise RuntimeError(f"Can not transform from [{format_}] to [{ori_format}].")

    transformed = format_transformation.transform(
        arr, format_, ori_format, ori_shape, groups=context.attributes.get("groups")
    )
    if sub_idx is not None:
        nested_ori[group_idx][sub_idx][:] = transformed
    else:
        nested_ori[group_idx][:] = transformed


def _override_inputs_from_attributes(context: TestcaseKernel):
    if not context.attributes:
        return
    op_info = OpInfoKeeper().info_of(context.op_name)
    if not op_info:
        return
    input_names = [inp["name"] for inp in op_info["inputs"]]
    attr_inputs = adapt_pickup_by_names(context.spec_tensors, tuple(input_names))
    dist = context.input_distribution or ()

    nested_arrays = list(context.input_arrays)
    nested_ori_arrays = list(context.original_input_arrays) if context.original_input_arrays else None

    assert len(nested_arrays) == len(input_names), (
        f"input_arrays({len(nested_arrays)}) != input_names({len(input_names)})"
    )

    for group_idx, name in enumerate(input_names):
        if name not in attr_inputs:
            continue
        val = attr_inputs[name]
        item = nested_arrays[group_idx]
        if item is None:
            continue

        d = dist[group_idx] if group_idx < len(dist) else 0
        if d > 0:
            # TensorList
            per_tensor_val = broadcast_tensorlist_values(val, len(item), name)
            for j in range(len(item)):
                assign_tensor_value(item[j], per_tensor_val[j], f"{name}[{j}]")
                if nested_ori_arrays is not None and nested_ori_arrays[group_idx] is not None:
                    _transform_single_to_ori(context, item[j], nested_ori_arrays, group_idx, j)
        else:
            # Single tensor
            assign_tensor_value(item, val, name)
            if nested_ori_arrays is not None:
                _transform_single_to_ori(context, item, nested_ori_arrays, group_idx, None)

    context.input_arrays = tuple(nested_arrays)
    if nested_ori_arrays is not None:
        context.original_input_arrays = tuple(nested_ori_arrays)


def _transform_to_original_format(context: TestcaseKernel):
    if context.manual_golden_binaries:
        return

    flat_shapes = promote_scalar_shapes(context.flat_input_shapes)
    flat_ori_shapes = promote_scalar_shapes(context.flat_input_ori_shapes)
    flat_arrays = deep_flatten(context.input_arrays)

    ori_arrays = list(flat_arrays)
    for idx, input_array in enumerate(flat_arrays):
        if input_array is None:
            continue
        ori_format = get(context.flat_input_ori_formats, idx)
        format_ = get(context.flat_input_formats, idx)
        ori_shape = get(flat_ori_shapes, idx)
        shape_ = get(flat_shapes, idx)
        if not _need_transform(ori_format, format_, ori_shape, shape_):
            continue
        if not format_transformation.is_transformable(format_, ori_format):
            raise RuntimeError(f"Can not transform from [{format_}] to [{ori_format}].")
        transformed = format_transformation.transform(
            input_array, format_, ori_format, ori_shape, groups=context.attributes.get("groups")
        )
        if list(transformed.shape) != list(ori_shape):
            raise RuntimeError(
                f"Try to transform from [{format_}] to original format [{ori_format}] failed: "
                f"Transformed shape: {transformed.shape}, but expected shape: {ori_shape}. "
                f"From shape is: {shape_}."
            )
        ori_arrays[idx] = transformed

    # flat → nested
    context.original_input_arrays = tuple(nest_by_distribution(ori_arrays, context.input_distribution))


def _realtime_random_input(context: TestcaseKernel):
    """Realtime Input Data Generation (Default)"""
    switches = get_global_storage()
    flat_shapes = promote_scalar_shapes(context.flat_input_shapes)
    flat_ori_shapes = promote_scalar_shapes(context.flat_input_ori_shapes)
    flat_dtypes = resolve_numpy_dtypes(context.flat_input_dtypes)
    base_seed = getattr(switches, "random_seed", None)
    has_batch_relation = has_complete_batch_relation(context)

    input_arrays = []
    ori_input_arrays = []
    actual_input_data_ranges = []
    # 免上传：登记 {内容指纹: 生成配方}，供三方采集判断哪些叶子可以不传（关着开关时为 None）。

    for idx, shape in enumerate(flat_shapes):
        dtype = get(flat_dtypes, idx)
        ori_format = get(context.flat_input_ori_formats, idx)
        format_ = get(context.flat_input_formats, idx)
        ori_shape = get(flat_ori_shapes, idx)
        data_range = get(context.flat_input_data_ranges, idx)

        if shape is None:
            input_arrays.append(None)
            ori_input_arrays.append(None)
            actual_input_data_ranges.append((None, None))
            continue

        if base_seed and has_batch_relation:
            # batch 一致性：跨用例同 shape 同值（对齐 aclnn/e2e 播种语义）
            numpy.random.seed(base_seed + idx)
        if _need_transform(ori_format, format_, ori_shape, shape):
            if not format_transformation.is_transformable(ori_format, format_):
                raise RuntimeError(
                    f"Can not transform from [{ori_format}] to [{format_}]. "
                    f"Please check `input_ori_formats` and `input_formats`."
                )
            rd = RandomData(dtype, ori_shape, data_range)
            ori_arr = rd.generate(switches.input_distribution)
            # batch 片段替换在 ori（逻辑布局）态做，transform 后两份一致；
            # batch_axis 语义 = ori_shape（用户声明的逻辑 shape）上的轴
            apply_slot_batch_replacement(
                context,
                idx,
                ori_arr,
                dtype=dtype,
                data_range=data_range,
                distribution=switches.input_distribution,
            )
            transformed = format_transformation.transform(
                ori_arr, ori_format, format_, shape, groups=context.attributes.get("groups")
            )
            if list(transformed.shape) != list(shape):
                raise RuntimeError(
                    f"Try to transform from original format [{ori_format}] to [{format_}] failed: "
                    f"Transformed shape: {transformed.shape}, but expected shape: {shape}. "
                    f"From original shape is: {ori_shape}."
                )
            input_arrays.append(transformed)
            ori_input_arrays.append(ori_arr)
        else:
            rd = RandomData(dtype, shape, data_range)
            arr = rd.generate(switches.input_distribution)
            apply_slot_batch_replacement(
                context,
                idx,
                arr,
                dtype=dtype,
                data_range=data_range,
                distribution=switches.input_distribution,
            )
            input_arrays.append(arr)
            ori_input_arrays.append(arr)
        actual_input_data_ranges.append(tuple(rd.data_range))

    context.input_arrays = tuple(nest_by_distribution(input_arrays, context.input_distribution))
    context.original_input_arrays = tuple(nest_by_distribution(ori_input_arrays, context.input_distribution))
    context.actual_input_data_ranges = tuple(nest_by_distribution(actual_input_data_ranges, context.input_distribution))


def gen_input(context: TestcaseKernel, stored_inputs=None):
    switches = get_global_storage()

    if stored_inputs is not None:
        flat_inputs = list(stored_inputs)
        context.input_arrays = tuple(nest_by_distribution(flat_inputs, context.input_distribution))
        context.actual_input_data_ranges = tuple(
            nest_by_distribution([(None, None)] * len(flat_inputs), context.input_distribution)
        )
        context.invalidate_flat_cache("input_arrays")
        _transform_to_original_format(context)
    elif context.manual_input_binaries:
        # Manual inputs
        _use_manual_input(context)
        # Try to convert to original format if golden needs original input arrays
        _transform_to_original_format(context)
    else:
        # Automatic random input
        _realtime_random_input(context)
        _override_inputs_from_attributes(context)

        input_func = get_plugin_function(context.op_name, "input", "kernel", switches.plugin_path)
        if input_func:
            input_arrays = list(context.input_arrays)
            kwargs = _collect_dynamic_kwargs(context)
            input_arrays = input_func(*input_arrays, **kwargs)
            # flatten (flat or nested) → validate → nest
            flat = deep_flatten(input_arrays)
            if len(flat) != len(context.flat_input_shapes):
                raise RuntimeError(
                    f"Input plugin returned {len(flat)} arrays, expected {len(context.flat_input_shapes)}"
                )
            context.input_arrays = tuple(nest_by_distribution(flat, context.input_distribution))
            _transform_to_original_format(context)


def _collect_dynamic_kwargs(context: TestcaseKernel):
    switches = get_global_storage()
    kwargs = context.attributes.copy()
    # delete internal attributes
    keys = list(kwargs.keys())
    for k in keys:
        if str(k)[0] in ("!", "#", "@"):
            del kwargs[k]
    # delete const inputs in attributes
    op_info = OpInfoKeeper().info_of(context.op_name)
    if op_info:
        for ipt in op_info["inputs"]:
            kwargs.pop(ipt["name"], None)
    # add some additional information.
    kwargs.update(
        {
            "full_soc_version": switches.dev_plat,
            "short_soc_version": switches.short_soc_version,
            "testcase_name": context.testcase_name,
            "input_ori_shapes": context.input_ori_shapes,
            "output_ori_shapes": context.output_ori_shapes,
            "input_formats": context.input_formats,
            "output_formats": context.output_formats,
            "input_ori_formats": context.input_ori_formats,
            "output_ori_formats": context.output_ori_formats,
            "input_dtypes": context.input_dtypes,
            "output_dtypes": context.output_dtypes,
            "input_ranges": context.input_data_ranges,
        }
    )
    # batch 一致性字段组（all-or-nothing）+ 框架已替换标记（对齐 aclnn/e2e 注入）
    kwargs.update(batch_relation_kwargs(context))
    kwargs["batch_applied"] = has_complete_batch_relation(context)
    return kwargs
