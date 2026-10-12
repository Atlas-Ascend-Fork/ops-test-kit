#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tensor 槽模型（AoS）：单张量描述符 + 槽表物化/读写原语。

本模块提供：
- ``Tensor``：单张量的声明信息 + 运行时数据（storage=原始底层，data=对外提供的
  执行形态——API 系非连续 view 共享内存 / Kernel format 转换有损重排 / 无派生
  时同一对象）；
- ``TensorSlot``：``Tensor | tuple[Tensor, ...] | None``——TensorList 是槽的
  容器形态（物化恒为 tuple，运行时镜像/写入兼容 list），optional 空位由
  None 占位；
- 物化原语：SoA flat 列 + dist → 槽表（``materialize_slots``）、运行时镜像
  （``slot_data_mirror``）与值结构 skeleton（``values_skeleton``）。

CSV 列与 field_parser 不动（外部契约不变）；SoA 声明列仍为声明权威源，
本模块提供 AoS 视图与运行期 write-through 基底（np_storages 平行表统一为
``Tensor.storage``）。
"""

__all__ = [
    "Tensor",
    "TensorSlot",
    "materialize_slots",
    "values_skeleton",
    "slot_data_mirror",
    "slot_storage_mirror",
    "write_slot_data",
    "write_slot_storage",
    "values_match_slots",
    "flat_slots",
]

from dataclasses import dataclass
from math import prod
from typing import Any, List, Optional, Sequence, Tuple, Union

from ....utilities.container import deep_flatten
from ....utilities.dtypes import DTYPE_MAP, get_dtype_width
from ....utilities.shape import shape_stride
from ....utilities.views import numpy_as_strided_safe

TensorSlot = Union["Tensor", Tuple["Tensor", ...], None]


@dataclass
class Tensor:
    """单张量的声明信息 + 数据。

    数据填充三规则（输入生成）：
      value 非空        → 直接使用 value（显式常量）
      value 空且 shape=() → RandomData(dtype, [1], data_range).squeeze()
      value 空且 shape≠() → RandomData(dtype, shape, data_range) + view 装配

    dtype 构造时自动归一化（fp16→float16、float→float32，未知值原样保留）；
    ori_shape/ori_format 缺省回退 shape/format（Kernel 生态）；
    TensorList 由外层 tuple[Tensor, ...] 表达，本类不处理嵌套；
    可选参数空位由外层 None 占位，不是 Tensor(dtype=None)。
    """

    shape: Tuple[int, ...] = ()
    dtype: Optional[str] = None
    format: Optional[str] = "ND"

    ori_shape: Optional[Tuple[int, ...]] = None
    ori_format: Optional[str] = None

    storage_shape: Optional[Tuple[int, ...]] = None
    view_offset: Optional[int] = None
    view_stride: Optional[Tuple[int, ...]] = None

    value: Optional[Any] = None
    data_range: Optional[Tuple] = None

    data: Optional[Any] = None
    storage: Optional[Any] = None
    data_kind: Optional[str] = None

    name: Optional[str] = None

    def __post_init__(self):
        if self.dtype is not None:
            self.dtype = DTYPE_MAP.get(self.dtype, self.dtype)
        if self.ori_shape is None:
            self.ori_shape = self.shape
        if self.ori_format is None:
            self.ori_format = self.format

    @property
    def is_scalar(self) -> bool:
        return self.shape == ()

    @property
    def is_contiguous(self) -> bool:
        if self.view_stride is None:
            return True
        if self.view_offset not in (None, 0):
            return False
        return tuple(self.view_stride) == shape_stride(self.shape)

    @property
    def is_const(self) -> bool:
        return self.value is not None

    @property
    def nbytes(self) -> int:
        if self.dtype is None:
            return 0
        width = get_dtype_width(self.dtype)
        return width if self.is_scalar else int(prod(self.shape) * width)


def make_storage_view(storage, view_shape, view_strides, view_offset):
    """按槽模型的 view 声明（shape/stride/offset，stride 为元素步长）在 storage 上切视图。

    要求 storage C 连续（ravel 视图保证 write-through）；stride 缺省按连续布局推导；
    越界视图（offset + Σ(dim-1)*stride 超出 storage）显式报错——as_strided 自身
    不做边界检查，静默越界即内存破坏。0-d storage 且视图同为 0-d 时原样返回；
    第三方 dtype（ml_dtypes float8 等）的 as_strided 兼容兜底由
    utilities.views.numpy_as_strided_safe 承担。
    """
    if storage.ndim == 0 and view_shape == ():
        return storage
    if view_offset is not None and view_offset < 0:
        raise ValueError(f"view_offset must be non-negative, got {view_offset}")
    if not storage.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "make_storage_view requires C-contiguous storage: ravel would copy and silently break write-through"
        )
    view_strides = tuple(view_strides) if view_strides else shape_stride(view_shape)
    _check_view_bounds(storage, view_shape, view_strides, view_offset or 0)
    byte_strides = tuple(s * storage.itemsize for s in view_strides)
    base = storage.ravel()[view_offset:] if view_offset and view_offset > 0 else storage.ravel()
    return numpy_as_strided_safe(base, shape=view_shape, strides=byte_strides)


def _check_view_bounds(storage, view_shape, view_strides, offset):
    """视图访问的元素区间 [min_idx, max_idx] 必须落在 storage 内（空视图免检）。"""
    if any(d == 0 for d in view_shape):
        return
    min_idx = max_idx = offset
    for dim, stride in zip(view_shape, view_strides):
        if dim < 0:
            raise ValueError(f"view dim must be non-negative, got {view_shape}")
        extent = stride * (dim - 1)
        if extent >= 0:
            max_idx += extent
        else:
            min_idx += extent
    if min_idx < 0 or max_idx >= storage.size:
        raise ValueError(
            f"view shape={view_shape} strides={view_strides} offset={offset} "
            f"accesses [{min_idx}, {max_idx}] beyond storage of {storage.size} elements"
        )


def _at(column: Optional[Sequence], idx: int):
    if column is None:
        return None
    return column[idx] if idx < len(column) else None


def _make_tensor(idx: int, columns: dict) -> "Tensor":
    shape = _at(columns.get("view_shapes"), idx)
    if shape is None:
        return None
    return Tensor(
        shape=tuple(shape),
        dtype=_at(columns.get("dtypes"), idx),
        format=_at(columns.get("formats"), idx),
        ori_shape=_at(columns.get("ori_shapes"), idx),
        ori_format=_at(columns.get("ori_formats"), idx),
        storage_shape=_at(columns.get("storage_shapes"), idx),
        view_offset=_at(columns.get("view_offsets"), idx),
        view_stride=_at(columns.get("view_strides"), idx),
        data_range=_at(columns.get("data_ranges"), idx),
    )


def materialize_slots(columns: dict, distribution: Sequence[int]) -> Tuple[TensorSlot, ...]:
    """SoA flat 列 + dist → 槽表。

    columns 的键：view_shapes（必填，其余可选），值均为按 flat 张量序的序列；
    view_shapes 为 None 的位置物化为 None 槽（optional 空位）。
    """
    slots: List[TensorSlot] = []
    idx = 0
    for count in distribution or ():
        if count > 0:
            slots.append(tuple(_make_tensor(idx + j, columns) for j in range(count)))
            idx += count
        else:
            slots.append(_make_tensor(idx, columns))
            idx += 1
    return tuple(slots)


def values_skeleton(values) -> Tuple[TensorSlot, ...]:
    """运行时嵌套值 → data-only 槽表（无声明路径的兜底骨架）。

    结构按值的嵌套推断（与 deep_flatten 同语义）：list/tuple 容器展开为
    TensorList 槽，原子值（ndarray/torch.Tensor/None）为单槽叶子。
    """
    slots: List[TensorSlot] = []
    for value in values:
        if isinstance(value, (tuple, list)):
            slots.append(tuple(Tensor(data=ele) if ele is not None else None for ele in value))
        else:
            slots.append(Tensor(data=value) if value is not None else None)
    return tuple(slots)


def _slot_field_mirror(slots: Sequence[TensorSlot], field: str) -> tuple:
    return tuple(
        [None if t is None else getattr(t, field) for t in slot]
        if isinstance(slot, (tuple, list))
        else (getattr(slot, field) if slot is not None else None)
        for slot in slots
    )


def slot_data_mirror(slots: Sequence[TensorSlot]) -> tuple:
    """槽表 → 嵌套运行时镜像（list[data | list[data] | None]，对齐 nest_by_distribution 形态）。"""
    return _slot_field_mirror(slots, "data")


def slot_storage_mirror(slots: Sequence[TensorSlot]) -> tuple:
    """槽表 → 嵌套 storage 镜像（Kernel 生态 ori 侧视图）。"""
    return _slot_field_mirror(slots, "storage")


def write_slot_data(slots: Sequence[TensorSlot], values) -> None:
    """嵌套运行时值 → slot.data（结构须与槽表一致，由 values_match_slots 保证；
    TensorList 内的 None 空槽跳过）。"""
    for slot, value in zip(slots, values):
        if isinstance(slot, (tuple, list)):
            for t, v in zip(slot, value):
                if t is not None:
                    t.data = v
        elif slot is not None:
            slot.data = value


def write_slot_storage(slots: Sequence[TensorSlot], values) -> None:
    """嵌套运行时值 → slot.storage（Kernel 生态 ori 侧写入，结构须与槽表一致；
    TensorList 内的 None 空槽跳过）。"""
    for slot, value in zip(slots, values):
        if isinstance(slot, (tuple, list)):
            for t, v in zip(slot, value):
                if t is not None:
                    t.storage = v
        elif slot is not None:
            slot.storage = value


def values_match_slots(slots: Sequence[TensorSlot], values) -> bool:
    """嵌套值结构是否与槽表逐位一致（None 槽对 None 值，TensorList 槽对等长序列）。"""
    if len(slots) != len(values):
        return False
    for slot, value in zip(slots, values):
        if isinstance(slot, (tuple, list)):
            if not isinstance(value, (tuple, list)) or len(value) != len(slot):
                return False
        elif slot is None:
            if value is not None:
                return False
        elif isinstance(value, (tuple, list)):
            return False
    return True


def flat_slots(slots: Sequence[TensorSlot]) -> tuple:
    """槽表深展平（Tensor 是叶子，无结构/dist 歧义）。"""
    return deep_flatten(slots)
