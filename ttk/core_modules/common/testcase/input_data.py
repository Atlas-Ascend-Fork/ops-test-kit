#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/testcase/input_data.py：输入数据生成期共享件（跨通路）。

- generate_np_storages：连续 numpy 存储生成（随机 + pure_output 常量初始化 +
  batch 一致性种子 + 片段替换），aclnn/e2e 共用；kernel 不参与（format 变换
  体系不同构）
- reseed_per_testcase：--seed 的用例级播种（worker 每用例入口调用），使输入/
  golden/workspace 生成与派发到的 worker、执行顺序无关
- assign_tensor_value / broadcast_tensorlist_values：attributes 指定输入值的
  原地覆盖，kernel/e2e 共用

消费者契约：testcase 需暴露 ApiTestcaseBase 的 flat_tensor_view_shapes /
flat_storage_shape / flat_tensor_dtypes / pure_output_indexes /
flat_input_data_ranges / api_name 槽。
"""

__all__ = [
    "generate_np_storages",
    "reseed_per_testcase",
    "assign_tensor_value",
    "broadcast_tensorlist_values",
]

# Standard Packages
import hashlib
import logging
import random

import numpy

from ....utilities import RandomData, constant_array, get, resolve_numpy_dtypes, shape_stride
from ....utilities.file_utils import load_array
from .batch import has_complete_batch_relation
from .batch_replace import apply_slot_batch_replacement


def _is_view_storage_isomorphic(testcase, idx, view_shape) -> bool:
    """batch_axis 是视图逻辑轴，替换写在连续存储上——仅 view 与 storage
    同构（同形 / 连续 stride / 零偏移）的槽可按存储轴替换。

    fp8 storage shape、非连续布局槽 warning 跳过（尽力而为，不阻断，
    对齐 slice_batch_hashes 语义）。
    """
    if view_shape is None:
        return False
    storage_shape = testcase.flat_storage_shape(idx)
    if storage_shape is not None and tuple(storage_shape) != tuple(view_shape):
        logging.warning(
            "[%s] batch replacement skip slot %d: storage shape %r != view shape %r",
            getattr(testcase, "testcase_name", None),
            idx,
            storage_shape,
            view_shape,
        )
        return False
    stride = testcase.flat_view_stride(idx)
    if stride is not None and tuple(stride) != tuple(shape_stride(view_shape)):
        logging.warning(
            "[%s] batch replacement skip slot %d: non-contiguous view stride %r",
            getattr(testcase, "testcase_name", None),
            idx,
            stride,
        )
        return False
    offset = testcase.flat_view_offset(idx)
    if offset not in (None, 0):
        logging.warning(
            "[%s] batch replacement skip slot %d: non-zero view offset %r",
            getattr(testcase, "testcase_name", None),
            idx,
            offset,
        )
        return False
    return True


def generate_np_storages(testcase, switches, zero_init_apis=(), overrides=None):
    """生成连续 numpy 存储数组（不写 testcase，落槽由调用方决定）。

    输入张量按 RandomData 随机生成；batch 一致性场景（batch 关系字段完整）按
    base_seed+idx 播种，使同 shape 张量跨用例同值，再按 batch_seed/slice_info
    逐样本区段重生成（仅 view/storage 同构槽）；pure_output 张量以常量
    初始化——zero_init_apis 中的 API 初始化为 0，其余为 1。

    overrides: {flat_idx: 文件路径} —— 声明槽位改由文件装载（load_array，
    .pt/.npy/裸 bin），dtype/shape 按 CSV 声明推导；未声明槽位保持上述常规
    语义。声明命中动态/未知 shape 槽时报错（无 shape 无从装载）。

    Returns: (storages, actual_data_ranges)，e2e 忽略 ranges。
    """
    dtypes = resolve_numpy_dtypes(testcase.flat_tensor_dtypes)
    flat_shapes = testcase.flat_tensor_view_shapes
    ranges = testcase.flat_input_data_ranges or ()
    base_seed = getattr(switches, "random_seed", None)
    has_batch_relation = has_complete_batch_relation(testcase)
    pure_output_indexes = set(testcase.pure_output_indexes)
    storages = []
    actual_data_ranges = []
    for idx, view_shape in enumerate(flat_shapes):
        fp = overrides.get(idx) if overrides else None
        if fp is not None:
            if view_shape is None or -1 in view_shape:
                raise RuntimeError(f"manual tensor file declared for dynamic/unknown-shape slot {idx}: {fp}")
            s_shape = testcase.flat_storage_shape(idx) or view_shape
            storages.append(load_array(fp, get(dtypes, idx), s_shape))
            actual_data_ranges.append((None, None))
            continue
        if view_shape is None or -1 in view_shape:
            # -1 = npu_preprocess 运行时物化的动态槽，此阶段不生成（upstream !281）
            storages.append(None)
            actual_data_ranges.append((None, None))
            continue
        dtype = get(dtypes, idx)
        # Storage shape may be None even if view_shape is not (e.g. fp8 tensors
        # without explicit storage shape). Fall back to view_shape.
        s_shape = testcase.flat_storage_shape(idx)
        if s_shape is None:
            s_shape = view_shape
        data_range = ranges[idx] if idx < len(ranges) else (None, None)

        if idx not in pure_output_indexes:
            if base_seed and has_batch_relation:
                # batch consistency compare different case support same shape tensor has same value
                numpy.random.seed(base_seed + idx)
            rd = RandomData(dtype, s_shape, data_range)
            storages.append(rd.generate(switches.input_distribution))
            if has_batch_relation and _is_view_storage_isomorphic(testcase, idx, view_shape):
                # batch 片段替换：逐样本区段按 sample_seed 重生成（视图转换前，
                # 存储与视图同构时存储轴 == 视图逻辑轴）
                apply_slot_batch_replacement(
                    testcase,
                    idx,
                    storages[-1],
                    dtype=dtype,
                    data_range=data_range,
                    distribution=switches.input_distribution,
                )
            actual_data_ranges.append(tuple(rd.data_range))
        else:
            init_val = 0 if testcase.api_name in zero_init_apis else 1
            storages.append(constant_array(dtype, s_shape, init_value=init_val))
            actual_data_ranges.append(data_range)
    return storages, actual_data_ranges


def reseed_per_testcase(testcase, switches) -> bool:
    """--seed 的用例级播种：worker 每用例入口调用。

    进程级播种（worker 启动时 seed 一次）+ 运行期动态派发（哪条用例落到哪个
    worker 由编译/执行耗时决定）叠加，导致同 --seed 两次运行同一条用例拿到的
    输入不同——A/B 精度对比失真。此处以 (--seed, testcase_name) 派生确定性种子
    重置本 worker 的全局 RNG（numpy legacy 流 + python random；原生整型输入经
    legacy 流派生 default_rng 种子，同样被覆盖），使该用例后续的输入/golden/
    workspace 生成成为 (seed, 用例身份) 的纯函数，与 worker 落点、执行顺序、
    同 worker 先前跑过的用例全部无关。

    batch relation 完整的用例不受影响：生成期本就按 base_seed+idx 逐槽重播种
    （跨用例同 shape 同值契约优先）。--seed 未设置时不播种，保持随机语义。
    返回是否执行了播种。
    """
    base_seed = getattr(switches, "random_seed", None)
    if not base_seed:
        return False
    name = getattr(testcase, "testcase_name", None) or ""
    digest = hashlib.blake2b(f"ttk-case-seed:v1:{base_seed}:{name}".encode(), digest_size=4).digest()
    case_seed = int.from_bytes(digest, "big")
    numpy.random.seed(case_seed)
    random.seed(case_seed)
    return True


def assign_tensor_value(arr, val, label):
    """Inplace: arr[:] = numpy.array(val, dtype=arr.dtype)。

    兼容 0-D（item 赋值）与元素数相等但 shape 不同（reshape 后赋值）；
    广播失败时抛带 label 的错误。
    """
    if arr is None or arr.size == 0:
        return
    spec = numpy.array(val, dtype=arr.dtype)
    try:
        if arr.ndim == 0:
            arr[...] = spec.item()
        elif spec.shape == arr.shape:
            arr[:] = spec
        elif spec.size == arr.size:
            arr[:] = spec.reshape(arr.shape)
        else:
            arr[:] = spec
    except ValueError as e:
        raise ValueError(f"Specify tensor [{label}] from `attributes` fail: {e}") from e


def broadcast_tensorlist_values(val, num_sub, label):
    """TensorList 属性值 → 逐子张量值列表：标量/[标量] 广播到全部，list 对位。

    长度不匹配抛 ValueError。
    """
    if not isinstance(val, (list, tuple)):
        return [val] * num_sub
    if len(val) == 1 and not isinstance(val[0], (list, tuple)):
        return [val[0]] * num_sub
    if len(val) != num_sub:
        raise ValueError(
            f"Specify TensorList [{label}] from `attributes` length mismatch: got {len(val)}, expected {num_sub}"
        )
    return list(val)
