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
Shared base class for aclnn (op_api) and framework_api (e2e) testcase structures.
"""

__all__ = ["ApiTestcaseBase"]

# Standard Packages
import logging
import os

from ....utilities import get, shape_stride
from ....utilities.container import (
    deep_flatten,
    flatten_by_distribution,
    flatten_groups,
    infer_list_distribution_from_nesting,
)
from .base import TestcaseBase
from .tensor import (
    Tensor,
    flat_slots,
    materialize_slots,
    slot_data_mirror,
    values_match_slots,
    values_skeleton,
    write_slot_data,
)

# npu_preprocess 契约失败前缀——与 Invoke.NPU_PREPROCESS_FAILURE 同值。
# common 内 testcase 与 invoke 不互 import（内部依赖表），此处独立定义、
# 测试钉死同值（tests/.../test_dyn_shape_slots.py::test_failure_prefix_matches_invoke_constant）。
_NPU_PREPROCESS_FAILURE_PREFIX = "NPU_PREPROCESS_FAILURE:"


class ApiTestcaseBase(TestcaseBase):
    """Base class with shared tensor property accessors, distribution inference,
    field normalization, and per-index accessors.

    Inherited by:
      - TestcaseAclnn (aclnn)
      - TestcaseE2e (framework_api / e2e)

    Not used by:
      - TestcaseKernel (kernel-level, no tensor fields)

    运行时数据基底是 AoS 槽表（tensor_slots）：
    ``tensors``（嵌套镜像）与 ``np_storages``（flat storage 表）是槽表的
    write-through 兼容视图——读取派生、赋值写入槽表；运行期逐槽变更走
    ``set_flat_tensor``/``set_flat_storage``（替代旧 _flat_tensors 影子表）。

    声明面（六列 tensor_* 字段）同为双模：物化前（parse/validate 期）是 raw
    存储透传；物化后（声明槽表就位）嵌套视图与 flat 族从槽表派生——运行期
    声明写点（golden promote 换 dtype、worker int64 转换）经属性赋值自动
    同步槽表（携带 data/storage 重建），flat 族的 dist 切割歧义随之消失。
    """

    __slots__ = (
        # === 公开声明面（CSV 声明 + 运行期读写契约）=== #
        "api_name",
        "output_tensor_indexes",
        "inplace_input_indexes",
        "prof_result",
        # const 位（attributes 标量源记录：input_generate 写 / 调用边界读）
        "const_input_indexes",
        "const_input_values",
        # backward 协议标志（api_name 物化时剥离 .backward 后缀置位）
        "backward_mode",
        # manual XPU 三方输出本地装载（--dump xpu 回放契约；aclnn/e2e 共享，
        # 装载/校验语义见 common/xpu.py load_manual_third_parties）
        "manual_xpu_binaries",
        # === 私有：六列声明原始存储（property 双模——物化前 raw 透传）=== #
        "_tensor_view_shapes",
        "_tensor_dtypes",
        "_tensor_formats",
        "_tensor_storage_shapes",
        "_tensor_view_offsets",
        "_tensor_view_strides",
        # === 私有：运行时槽表（AoS 基底 + 物化/写穿标志）=== #
        "_tensor_slots",
        "_slots_from_declaration",
        "_tensors_written",
        "_np_storages_written",
        # === 私有：dist / flat 派生缓存（invalidate_flat_cache 失效域）=== #
        "_tensor_list_dist",
        "_output_dist",
        "_flat_tensor_view_shapes",
        "_flat_tensors",
        "_flat_scalars",
        "_flat_tensor_dtypes",
        "_flat_tensor_formats",
        "_flat_tensor_storage_shapes",
        "_flat_tensor_view_offsets",
        "_flat_tensor_view_strides",
        "_flat_input_data_ranges",
        "_flat_precision_tolerances",
        "_flat_absolute_precision",
        "_flat_manual_xpu_binaries",
        "_pure_output_indexes",
        "_is_torch_dtype_support",
        "_is_tf_dtype_support",
        # === 私有：签名解析缓存（终稿挂载，随 pickle 进 worker）=== #
        "_param_plan_cache",
    )

    _scalar_tensor_fields = (
        "tensor_dtypes",
        "tensor_formats",
        "tensor_view_offsets",
    )
    _shape_tensor_fields = (
        "tensor_view_strides",
        "tensor_storage_shapes",
    )

    def __init__(self):
        super().__init__()
        # ── 公开声明面 ──
        self.api_name = None
        self.output_tensor_indexes = ()
        self.inplace_input_indexes = ()
        self.prof_result = None
        self.const_input_indexes = set()
        self.const_input_values = {}
        # backward 协议：api_name = <forward_api>[.backward]——解析物化时剥离
        # 后缀，forward 名进 api_name 生态（keeper/param_plan/签名对账），backward
        # 标志独立。执行链按 backward_mode 走 autograd 分支（只采反向）。
        self.backward_mode: bool = False
        # manual XPU 三方输出本地装载（未声明 None；声明后 worker 侧短路远端采集）
        self.manual_xpu_binaries = None
        # ── 六列声明原始存储 ──
        self._tensor_view_shapes = None
        self._tensor_dtypes = None
        self._tensor_formats = ()
        self._tensor_storage_shapes = ()
        self._tensor_view_offsets = ()
        self._tensor_view_strides = ()
        # ── 运行时槽表 ──
        self._tensor_slots = None
        self._slots_from_declaration = False
        self._tensors_written = False
        self._np_storages_written = False
        # ── dist / flat 派生缓存 ──
        self._tensor_list_dist = None
        self._output_dist = None
        self._flat_tensor_view_shapes = None
        self._flat_tensors = None
        self._flat_scalars = None
        self._flat_tensor_dtypes = None
        self._flat_tensor_formats = None
        self._flat_tensor_storage_shapes = None
        self._flat_tensor_view_offsets = None
        self._flat_tensor_view_strides = None
        self._flat_input_data_ranges = None
        self._flat_precision_tolerances = None
        self._flat_absolute_precision = None
        self._flat_manual_xpu_binaries = None
        self._pure_output_indexes = None
        self._is_torch_dtype_support = None
        self._is_tf_dtype_support = None
        # ── 签名解析缓存 ──
        self._param_plan_cache = None

    @property
    def op_name(self):
        return self.api_name

    def _strip_backward_suffix(self) -> None:
        """api_name 物化时剥离 '.backward' 裸后缀（不带括号）。

        forward 名进本字段与 keeper/param_plan/签名对账生态；golden_api 默认 =
        api_name 原样（含后缀——CPU 参照自动 forward+grad）；后缀语法演化预留：
        .backward(...) 带参形式现不解析，报"不支持的后缀"。"""
        if not self.api_name:
            return
        has_suffix = self.api_name.endswith(".backward") or ".backward(" in self.api_name
        if not has_suffix:
            return
        stripped = self.api_name.split(".backward", 1)[0]
        if stripped.endswith(".") or not stripped:
            self.is_valid = False
            self.fail_reason = f"unsupported backward suffix: {self.api_name!r}"
            return
        if "(" in self.api_name:
            self.is_valid = False
            self.fail_reason = f"unsupported backward suffix (parameterized): {self.api_name!r}"
            return
        self.api_name = stripped
        self.backward_mode = True

    # ========== Declaration dual-mode properties（物化前 raw 透传，物化后槽表派生） ==========

    @property
    def tensor_view_shapes(self):
        return self._declaration_view("_tensor_view_shapes", "shape")

    @tensor_view_shapes.setter
    def tensor_view_shapes(self, value):
        self._tensor_view_shapes = value
        self._after_declaration_write()

    @property
    def tensor_dtypes(self):
        return self._declaration_view("_tensor_dtypes", "dtype")

    @tensor_dtypes.setter
    def tensor_dtypes(self, value):
        self._tensor_dtypes = value
        self._after_declaration_write()

    @property
    def tensor_formats(self):
        return self._declaration_view("_tensor_formats", "format")

    @tensor_formats.setter
    def tensor_formats(self, value):
        self._tensor_formats = value
        self._after_declaration_write()

    @property
    def tensor_storage_shapes(self):
        return self._declaration_view("_tensor_storage_shapes", "storage_shape")

    @tensor_storage_shapes.setter
    def tensor_storage_shapes(self, value):
        self._tensor_storage_shapes = value
        self._after_declaration_write()

    @property
    def tensor_view_offsets(self):
        return self._declaration_view("_tensor_view_offsets", "view_offset")

    @tensor_view_offsets.setter
    def tensor_view_offsets(self, value):
        self._tensor_view_offsets = value
        self._after_declaration_write()

    @property
    def tensor_view_strides(self):
        return self._declaration_view("_tensor_view_strides", "view_stride")

    @tensor_view_strides.setter
    def tensor_view_strides(self, value):
        self._tensor_view_strides = value
        self._after_declaration_write()

    def _declaration_view(self, private, attr):
        """嵌套声明视图：声明槽表物化且 raw 非空时从槽表派生，否则 raw 透传。"""
        raw = getattr(self, private)
        if not self._slots_from_declaration or not raw:
            return raw
        return tuple(
            tuple(None if t is None else getattr(t, attr) for t in slot)
            if isinstance(slot, (tuple, list))
            else (None if slot is None else getattr(slot, attr))
            for slot in self._tensor_slots
        )

    def _flat_slot_view(self, attr):
        """flat 声明视图：逐槽位派生（None 槽位 None）。"""
        return tuple(None if t is None else getattr(t, attr) for t in flat_slots(self._tensor_slots))

    def _after_declaration_write(self):
        """声明写点同步：清派生缓存；槽表已物化时携带运行时数据重建。"""
        for cache in (
            "_flat_tensor_view_shapes",
            "_flat_tensor_dtypes",
            "_flat_tensor_formats",
            "_flat_tensor_storage_shapes",
            "_flat_tensor_view_offsets",
            "_flat_tensor_view_strides",
            "_flat_input_data_ranges",
            "_flat_tensors",
            "_tensor_list_dist",
            "_output_dist",
            "_pure_output_indexes",
        ):
            setattr(self, cache, None)
        if self._tensor_slots is not None:
            self._rematerialize_preserving_runtime()

    def _rematerialize_preserving_runtime(self):
        """按当前 raw 声明重建槽表，按 flat 位携带 data/storage/data_kind/name。"""
        old_flat = flat_slots(self._tensor_slots) if self._tensor_slots is not None else ()
        new_slots = self._materialize_slots_from_declaration()
        if new_slots is None:
            return
        for new, old in zip(flat_slots(new_slots), old_flat):
            if new is None or old is None:
                continue
            new.data = old.data
            new.storage = old.storage
            new.data_kind = old.data_kind
            new.name = old.name
        self._tensor_slots = new_slots
        self._slots_from_declaration = True

    # ========== Distribution ==========

    @property
    def tensor_list_dist(self):
        """TensorList distribution inferred from raw tensor_view_shapes nesting.

        Cached on first access. Returns () when no tensor_view_shapes.
        """
        if self._tensor_list_dist is None and self._tensor_view_shapes:
            self._tensor_list_dist = infer_list_distribution_from_nesting(self._tensor_view_shapes)
        return self._tensor_list_dist or ()

    def flat_index_of(self, nested_idx: int) -> int:
        """嵌套槽序号 → flat 张量槽序号（TensorList 分布前缀和；无分布时恒等）。

        单一实现：output_* 属性族、输出字节/shape 收集、manual golden 槽映射共用。
        """
        dist = self.tensor_list_dist
        if not dist:
            return nested_idx
        return sum(max(d, 1) for d in dist[:nested_idx])

    @property
    def output_dist(self):
        """Output tensor distribution: projection of tensor_list_dist onto output_tensor_indexes.

        Cached on first access. Returns () when no output tensors. Out-of-range
        indexes are skipped here; validate marks them invalid.
        """
        if self._output_dist is None:
            dist = self.tensor_list_dist
            if dist and self.output_tensor_indexes:
                self._output_dist = tuple(dist[i] for i in self.output_tensor_indexes if 0 <= i < len(dist))
            else:
                self._output_dist = ()
        return self._output_dist

    # ========== Flat properties ==========

    @property
    def flat_tensor_view_shapes(self):
        if self._flat_tensor_view_shapes is not None:
            return self._flat_tensor_view_shapes
        raw = self._tensor_view_shapes
        if not raw:
            return raw
        if self._slots_from_declaration:
            self._flat_tensor_view_shapes = self._flat_slot_view("shape")
        else:
            self._flat_tensor_view_shapes = flatten_groups(raw)
        return self._flat_tensor_view_shapes

    @property
    def flat_manual_xpu_binaries(self):
        """manual_xpu_binaries 扁平化（TensorList 嵌套声明随输出分布展开；str 视为叶子）。

        口径对齐 kernel flat_manual_xpu_binaries（deep_flatten）与
        flat_manual_tensor_binaries；无声明/无分布时原样返回。
        """
        if self._flat_manual_xpu_binaries is not None:
            return self._flat_manual_xpu_binaries
        if not self.manual_xpu_binaries or not self.output_dist:
            self._flat_manual_xpu_binaries = self.manual_xpu_binaries
        else:
            from ttk.utilities import deep_flatten

            self._flat_manual_xpu_binaries = deep_flatten(self.manual_xpu_binaries)
        return self._flat_manual_xpu_binaries

    @property
    def flat_tensor_dtypes(self):
        if self._flat_tensor_dtypes is not None:
            return self._flat_tensor_dtypes
        raw = self._tensor_dtypes
        if not raw:
            return raw
        if self._slots_from_declaration:
            # 声明槽表派生：无叶子歧义（结构即事实），dist 切割仅物化前使用。
            self._flat_tensor_dtypes = self._flat_slot_view("dtype")
        else:
            dist = self.tensor_list_dist
            if dist:
                self._flat_tensor_dtypes = flatten_by_distribution(raw, dist)
            else:
                self._flat_tensor_dtypes = raw
        return self._flat_tensor_dtypes

    @property
    def flat_tensors(self):
        if self._flat_tensors is not None:
            return self._flat_tensors
        tensors = self.tensors
        self._flat_tensors = deep_flatten(tensors) if tensors is not None else None
        return self._flat_tensors

    # ========== AoS slot substrate ==========

    @property
    def tensor_slots(self):
        """AoS 槽表（声明物化 + 运行时数据）。首次访问按 raw 声明列物化。"""
        if self._tensor_slots is None:
            slots = self._materialize_slots_from_declaration()
            self._tensor_slots = slots if slots is not None else ()
            self._slots_from_declaration = slots is not None
        return self._tensor_slots

    def _materialize_slots_from_declaration(self):
        """raw 声明列 → 槽表（无声明返回 None，交由值结构 skeleton 兜底）。

        直读 raw 私有槽（不走公共 flat/嵌套视图）——物化/重建期间公共视图
        可能反过来从旧槽表派生，经 raw 才无环。
        """
        if not self._tensor_view_shapes:
            return None
        dist = infer_list_distribution_from_nesting(self._tensor_view_shapes)
        return materialize_slots(
            {
                "view_shapes": flatten_groups(self._tensor_view_shapes),
                "dtypes": self._flat_decl_by_dist(self._tensor_dtypes, dist),
                "formats": self._flat_decl_by_dist(self._tensor_formats, dist),
                "storage_shapes": self._flat_decl_by_dist(self._tensor_storage_shapes, dist),
                "view_offsets": self._flat_decl_by_dist(self._tensor_view_offsets, dist),
                "view_strides": self._flat_decl_by_dist(self._tensor_view_strides, dist),
                "data_ranges": self._flat_decl_by_dist(self.input_data_ranges, dist),
            },
            dist,
        )

    @staticmethod
    def _flat_decl_by_dist(values, dist):
        if not values:
            return values
        return flatten_by_distribution(values, dist) if dist else values

    def _rebuild_slots_for_values(self, value):
        """结构不一致时重建槽表：声明优先，值结构兜底；按 flat 位保留已有
        storage/data_kind/name（与 _rematerialize_preserving_runtime 同口径）。"""
        old_flat = flat_slots(self._tensor_slots) if self._tensor_slots is not None else ()
        new_slots = self._materialize_slots_from_declaration()
        if new_slots is not None and values_match_slots(new_slots, value):
            self._slots_from_declaration = True
        else:
            new_slots = values_skeleton(value)
            self._slots_from_declaration = False
        write_slot_data(new_slots, value)
        for new, old in zip(flat_slots(new_slots), old_flat):
            if new is None or old is None:
                continue
            if old.storage is not None:
                new.storage = old.storage
            new.data_kind = old.data_kind
            new.name = old.name
        self._tensor_slots = new_slots

    @property
    def tensors(self):
        """嵌套运行时镜像（list[data | list[data] | None]）——槽表派生兼容视图。"""
        if self._tensor_slots is None or not self._tensors_written:
            return None
        return slot_data_mirror(self._tensor_slots)

    @tensors.setter
    def tensors(self, value):
        if value is None:
            self._tensors_written = False
            self._flat_tensors = None
            return
        slots = self._tensor_slots
        if slots is not None and values_match_slots(slots, value):
            write_slot_data(slots, value)
        else:
            self._rebuild_slots_for_values(value)
        self._tensors_written = True
        self._flat_tensors = None

    @property
    def np_storages(self):
        """flat storage 表（逐张量 Tensor.storage）——槽表派生兼容视图。"""
        if self._tensor_slots is None or not self._np_storages_written:
            return None
        return tuple(None if slot is None else slot.storage for slot in flat_slots(self._tensor_slots))

    @np_storages.setter
    def np_storages(self, value):
        if value is None:
            self._np_storages_written = False
            return
        existing = self._tensor_slots
        if existing is not None and len(flat_slots(existing)) == len(value):
            for slot, v in zip(flat_slots(existing), value):
                if slot is not None:
                    slot.storage = v
        else:
            has_runtime_data = (
                existing is not None
                and bool(flat_slots(existing))
                and (self._tensors_written or self._np_storages_written)
            )
            if has_runtime_data:
                raise ValueError(
                    f"np_storages length {len(value)} does not match existing "
                    f"{len(flat_slots(existing))} tensor slots; refusing to discard runtime data"
                )
            slots = self._materialize_slots_from_declaration()
            if slots is not None and len(flat_slots(slots)) == len(value):
                for slot, v in zip(flat_slots(slots), value):
                    if slot is not None:
                        slot.storage = v
                self._slots_from_declaration = True
            else:
                slots = tuple(Tensor(storage=v) if v is not None else None for v in value)
                self._slots_from_declaration = False
            self._tensor_slots = slots
        self._np_storages_written = True

    def set_flat_tensor(self, idx: int, value):
        """运行期逐槽数据写入（替代旧 _flat_tensors 影子表的就地变更）。"""
        slot = self._flat_slot_for_write(idx)
        slot.data = value
        self._tensors_written = True
        self._flat_tensors = None

    def set_flat_storage(self, idx: int, storage):
        """运行期逐槽 storage 写入（替代旧 np_storages 列表就地变更）。"""
        slot = self._flat_slot_for_write(idx)
        slot.storage = storage
        self._np_storages_written = True

    def _flat_slot_for_write(self, idx: int):
        slots = self.tensor_slots
        flat = flat_slots(slots)
        slot = flat[idx] if idx < len(flat) else None
        if slot is None:
            raise IndexError(f"flat tensor index {idx} has no Tensor slot (optional empty or out of range)")
        return slot

    @property
    def flat_scalars(self):
        if self._flat_scalars is not None:
            return self._flat_scalars
        self._flat_scalars = deep_flatten(self.scalars) if self.scalars is not None else None
        return self._flat_scalars

    @property
    def flat_tensor_formats(self):
        if self._flat_tensor_formats is not None:
            return self._flat_tensor_formats
        raw = self._tensor_formats
        if not raw:
            return raw
        if self._slots_from_declaration:
            self._flat_tensor_formats = self._flat_slot_view("format")
        else:
            dist = self.tensor_list_dist
            if dist:
                self._flat_tensor_formats = flatten_by_distribution(raw, dist)
            else:
                self._flat_tensor_formats = raw
        return self._flat_tensor_formats

    @property
    def flat_tensor_storage_shapes(self):
        if self._flat_tensor_storage_shapes is not None:
            return self._flat_tensor_storage_shapes
        raw = self._tensor_storage_shapes
        if not raw:
            return raw
        if self._slots_from_declaration:
            self._flat_tensor_storage_shapes = self._flat_slot_view("storage_shape")
        else:
            dist = self.tensor_list_dist
            if dist:
                self._flat_tensor_storage_shapes = flatten_by_distribution(raw, dist)
            else:
                self._flat_tensor_storage_shapes = raw
        return self._flat_tensor_storage_shapes

    @property
    def flat_tensor_view_offsets(self):
        if self._flat_tensor_view_offsets is not None:
            return self._flat_tensor_view_offsets
        raw = self._tensor_view_offsets
        if not raw:
            return raw
        if self._slots_from_declaration:
            self._flat_tensor_view_offsets = self._flat_slot_view("view_offset")
        else:
            dist = self.tensor_list_dist
            if dist:
                self._flat_tensor_view_offsets = flatten_by_distribution(raw, dist)
            else:
                self._flat_tensor_view_offsets = raw
        return self._flat_tensor_view_offsets

    @property
    def flat_tensor_view_strides(self):
        if self._flat_tensor_view_strides is not None:
            return self._flat_tensor_view_strides
        raw = self._tensor_view_strides
        if not raw:
            return raw
        if self._slots_from_declaration:
            self._flat_tensor_view_strides = self._flat_slot_view("view_stride")
        else:
            dist = self.tensor_list_dist
            if dist:
                self._flat_tensor_view_strides = flatten_by_distribution(raw, dist)
            else:
                self._flat_tensor_view_strides = raw
        return self._flat_tensor_view_strides

    @property
    def flat_input_data_ranges(self):
        """Flat per-tensor ranges.

        Assumes _normalize_compressed_fields has already expanded compressed
        forms.  Simply flattens the normalized nested structure.
        """
        if self._flat_input_data_ranges is not None:
            return self._flat_input_data_ranges
        raw = self.input_data_ranges
        if not raw:
            return raw
        if self._slots_from_declaration:
            self._flat_input_data_ranges = self._flat_slot_view("data_range")
        else:
            dist = self.tensor_list_dist
            if dist:
                self._flat_input_data_ranges = flatten_by_distribution(raw, dist)
            else:
                self._flat_input_data_ranges = raw
        return self._flat_input_data_ranges

    @property
    def flat_precision_tolerances(self):
        """Flat per-output precision tolerances. Assumes normalize already done."""
        if self._flat_precision_tolerances is not None:
            return self._flat_precision_tolerances
        if not self.precision_tolerances:
            return self.precision_tolerances
        odist = self.output_dist
        if odist:
            self._flat_precision_tolerances = flatten_by_distribution(self.precision_tolerances, odist)
        else:
            self._flat_precision_tolerances = self.precision_tolerances
        return self._flat_precision_tolerances

    @property
    def flat_absolute_precision(self):
        """Flat per-output absolute precision. Assumes normalize already done."""
        if self._flat_absolute_precision is not None:
            return self._flat_absolute_precision
        if not isinstance(self.absolute_precision, tuple):
            return self.absolute_precision
        odist = self.output_dist
        if odist:
            self._flat_absolute_precision = flatten_by_distribution(self.absolute_precision, odist)
        else:
            self._flat_absolute_precision = self.absolute_precision
        return self._flat_absolute_precision

    def invalidate_flat_cache(self, *fields):
        """Invalidate flat cache(s) by field name.

        Example: invalidate_flat_cache("tensor_dtypes") sets self._flat_tensor_dtypes = None.
        Uses hasattr guard: nonexistent _flat_{field} is safely skipped.
        """
        for field in fields:
            attr = f"_flat_{field}"
            if hasattr(self, attr):
                setattr(self, attr, None)

    # ========== Per-flat-index accessors ==========

    def is_torch_dtype_support(self) -> bool:
        """Check if all dtypes in testcase are supported by the current torch natively."""
        if self._is_torch_dtype_support is not None:
            return self._is_torch_dtype_support
        from ttk.utilities.dtypes import is_torch_native_dtype

        result = True
        for dtype in self.flat_tensor_dtypes:
            if dtype is not None and not is_torch_native_dtype(dtype):
                result = False
                break
        self._is_torch_dtype_support = result
        return result

    def is_tf_dtype_support(self) -> bool:
        """Check if all dtypes in testcase are supported by TF natively."""
        if self._is_tf_dtype_support is not None:
            return self._is_tf_dtype_support
        from ttk.utilities.dtypes import is_tf_native_dtype

        result = True
        for dtype in self.flat_tensor_dtypes:
            if dtype is not None and not is_tf_native_dtype(dtype):
                result = False
                break
        self._is_tf_dtype_support = result
        return result

    def is_dtype_support(self) -> bool:
        """Framework-aware dtype support check."""
        from ..backend import detect_framework

        framework = detect_framework(self.api_name)
        if framework == "tf":
            return self.is_tf_dtype_support()
        return self.is_torch_dtype_support()

    # ========== Legacy per-flat-index accessors (kept for backward compat) ==========

    def flat_storage_shape(self, idx: int):
        flat_shapes = self.flat_tensor_view_shapes
        view_shape = flat_shapes[idx] if idx < len(flat_shapes) else None
        s_shapes = self.flat_tensor_storage_shapes
        if not s_shapes or idx >= len(s_shapes):
            return view_shape
        val = s_shapes[idx]
        if val is None:
            return view_shape
        return val

    def flat_view_stride(self, idx: int):
        flat_shapes = self.flat_tensor_view_shapes
        view_shape = flat_shapes[idx] if idx < len(flat_shapes) else None
        if view_shape is None:
            return None
        strides = self.flat_tensor_view_strides or ()
        if idx < len(strides):
            s = strides[idx]
            if s is not None and s != ():
                return s
        return shape_stride(view_shape)

    def flat_view_offset(self, idx: int):
        flat_shapes = self.flat_tensor_view_shapes
        view_shape = flat_shapes[idx] if idx < len(flat_shapes) else None
        if view_shape is None:
            return None
        offsets = self.flat_tensor_view_offsets or ()
        if idx < len(offsets):
            val = offsets[idx]
            if val is not None:
                return val
        return 0

    @property
    def dyn_input_slot_indexes(self):
        """Flat tensor slots whose storage is materialized by npu_preprocess（-1 标记，upstream !281）。"""
        return tuple(
            idx for idx, shape in enumerate(self.flat_tensor_view_shapes or ()) if shape is not None and -1 in shape
        )

    def _materialized_dynamic_slots(self):
        """-1 槽的当前物化态：{flat_idx: tensor}（hook 反刷输入位后经槽表可见）。

        物化是"-1 参数位存在可用 Tensor"的事实：dict 返回（显式通道）之外，
        hook 就地反刷输入位（槽表写入，隐式通道）同样成立，本方法即隐式通道
        的核验面。"""
        got = {}
        tensors = self.flat_tensors
        for idx in self.dyn_input_slot_indexes:
            if tensors is not None and idx < len(tensors) and tensors[idx] is not None:
                got[idx] = tensors[idx]
        return got

    def absorb_preprocess(self, result, plan, args=None, kwargs=None):
        """npu_preprocess 结果落地（通路多态，由通路调用点在 invoke 之后直呼）。

        None 且无 -1 声明 → 纯副作用 no-op（基类默认）；其余由通路子类实现：
        None 且有声明 → 隐式通道核验；dict → 显式通道校验+回填（apply 内含
        逐名声明校验）。plan/args/kwargs 为执行现场的装配上下文（E2E 的
        args/kwargs 是执行局部量，与主 API 共享同一对象）。"""
        if result is None and not self.dyn_input_slot_indexes:
            return
        raise NotImplementedError(
            f"{type(self).__name__} does not absorb npu_preprocess results; "
            "pathway subclasses implement dynamic-slot absorption"
        )

    def _require_dynamic_slots_materialized(self):
        """None 返回 + 有 -1 声明时的隐式通道核验：槽位已全部反刷 → 放行；
        否则契约失败（前缀与 Invoke.NPU_PREPROCESS_FAILURE 同值——common 内
        testcase 与 invoke 不互 import，同值由测试钉住）。"""
        got = self._materialized_dynamic_slots()
        if got and len(got) == len(self.dyn_input_slot_indexes):
            return got
        raise RuntimeError(
            f"{_NPU_PREPROCESS_FAILURE_PREFIX} declared -1 dynamic slot(s) not materialized: "
            "hook returned None and no input slot was refreshed "
            "(materialize via dict return or slot-table write)"
        )

    def _check_manual_xpu_binaries(self):
        """aclnn/e2e 的 manual_xpu_binaries 校验：路径存在 + 数量不超 flat 输出槽数。

        与 kernel 的差异：aclnn/e2e 真跑设备（golden 有设备真值源），不要求与
        manual_input/golden 成套——manual_xpu 仅替代三方采集的远端产出。
        输出槽取子类 ``flat_output_view_shapes``（kernel 通路名 ``flat_output_shapes``）。
        """
        if not self.is_valid or not self.manual_xpu_binaries:
            return
        flat_paths = self.flat_manual_xpu_binaries or ()
        for path in flat_paths:
            if path is None:
                continue
            if not isinstance(path, str) or not os.path.isfile(path):
                self.is_valid = False
                self.fail_reason = "MANUAL_XPU_BINARIES_INVALID"
                logging.error("manual_xpu_binaries path invalid: %r", path)
                return
        outputs = getattr(self, "flat_output_view_shapes", None)
        if outputs is None:
            outputs = getattr(self, "flat_output_shapes", None)
        if outputs and len(flat_paths) > len(outputs):
            self.is_valid = False
            self.fail_reason = "MANUAL_XPU_BINARIES_INVALID"
            logging.error("manual_xpu_binaries count %d exceeds flat output slots %d", len(flat_paths), len(outputs))

    def _check_dyn_shape_slots(self):
        """Validate constraints shared by E2E and ACLNN dynamic input slots."""
        if not self.is_valid:
            return

        flat_shapes = self.flat_tensor_view_shapes or ()
        if any(shape is not None and -2 in shape for shape in flat_shapes):
            self.is_valid = False
            self.fail_reason = "DYN_SHAPE_INVALID"
            logging.error(f"[{self.testcase_name}] -2 (unknown rank) is not supported in tensor_view_shapes")
            return

        dyn_indexes = self.dyn_input_slot_indexes
        if not dyn_indexes:
            return

        top_of_flat = []
        for top, shape in enumerate(self.tensor_view_shapes or ()):
            count = len(shape) if shape and isinstance(shape[0], (tuple, list)) else 1
            top_of_flat.extend([top] * count)

        dist = self.tensor_list_dist
        output_indexes = set(self.output_tensor_indexes or ())
        inplace_indexes = set(
            getattr(self, "output_inplace_indexes", None) or getattr(self, "inplace_input_indexes", None) or ()
        )
        flat_strides = self.flat_tensor_view_strides or ()
        flat_offsets = self.flat_tensor_view_offsets or ()
        flat_storages = self.flat_tensor_storage_shapes or ()
        for idx in dyn_indexes:
            top = top_of_flat[idx]
            if dist and dist[top] > 0:
                self.is_valid = False
                self.fail_reason = "DYN_SHAPE_IN_TENSORLIST"
                logging.error(f"[{self.testcase_name}] -1 slot inside TensorList is not supported (slot {top})")
                return
            if top in output_indexes:
                self.is_valid = False
                self.fail_reason = "DYN_SHAPE_ON_OUTPUT_SLOT"
                logging.error(f"[{self.testcase_name}] -1 slot {top} is marked as an output")
                return
            if top in inplace_indexes:
                self.is_valid = False
                self.fail_reason = "DYN_SHAPE_ON_INPLACE_SLOT"
                logging.error(f"[{self.testcase_name}] -1 slot {top} is marked as an inplace input/output")
                return
            if (
                get(flat_strides, idx, out_of_range=())
                or get(flat_offsets, idx, out_of_range=0)
                or get(flat_storages, idx, out_of_range=())
            ):
                self.is_valid = False
                self.fail_reason = "DYN_SHAPE_WITH_VIEW_OVERRIDE"
                logging.error(
                    f"[{self.testcase_name}] -1 flat slot {idx} must not configure "
                    "tensor_view_strides/tensor_view_offsets/tensor_storage_shapes; "
                    "singleton compressed values are broadcast to every slot, "
                    "so use an explicit None placeholder for this dynamic slot"
                )
                return

        from ....utilities.dtypes import is_tf_native_dtype, is_torch_native_dtype
        from ..backend.framework_detector import detect_framework

        dtype_supported = is_tf_native_dtype if detect_framework(self.api_name) == "tf" else is_torch_native_dtype
        dtypes = self.flat_tensor_dtypes or ()
        unsupported = [idx for idx in dyn_indexes if idx >= len(dtypes) or not dtype_supported(dtypes[idx])]
        if unsupported:
            self.is_valid = False
            self.fail_reason = "DYN_SHAPE_DTYPE_UNSUPPORTED"
            logging.error(f"[{self.testcase_name}] -1 slot dtype is not framework-native: {unsupported}")

    # ========== Pure output indexes ==========

    @property
    def pure_output_indexes(self):
        if self._pure_output_indexes is not None:
            return self._pure_output_indexes
        dist = self.tensor_list_dist
        flat_output = set()
        for idx in self.output_tensor_indexes or ():
            if not 0 <= idx < len(dist):
                continue
            flat_idx = self.flat_index_of(idx)
            count = dist[idx] if dist[idx] > 0 else 1
            flat_output.update(range(flat_idx, flat_idx + count))
        self._pure_output_indexes = sorted(flat_output)
        return self._pure_output_indexes

    # ========== Normalize ==========

    def _describe_operator_name(self):
        return self.api_name

    def _describe_gate_fields(self):
        return ("tensor_view_shapes", "tensor_dtypes")

    def _assemble_described_shapes(self, shapes, dtypes) -> None:
        """describe 的 shapes/dtypes → tensor_view_shapes/tensor_dtypes（缺啥补啥）。

        aclnn 的 shapes 按 API 参数列表含输出槽位（输出索引经既有
        auto-fill / CSV output_tensor_indexes 列消费）。
        """
        if shapes is not None:
            self._merge_described("tensor_view_shapes", shapes)
        if dtypes is not None:
            self._merge_described("tensor_dtypes", dtypes)

    def _normalize_compressed_fields(self):
        if not self.is_valid or not self.tensor_view_shapes:
            return
        dist = self.tensor_list_dist
        if any(not 0 <= i < len(dist) for i in self.output_tensor_indexes or ()):
            self.is_valid = False
            self.fail_reason = (
                f"output_tensor_indexes {self.output_tensor_indexes} out of range for {len(dist)} tensor slots"
            )
            return
        if dist:
            for field_name in self._scalar_tensor_fields:
                self._normalize_scalar_field_by_dist(field_name, dist)
            for field_name in (*self._shape_tensor_fields, "input_data_ranges"):
                self._normalize_range_field_by_dist(field_name, dist)
            odist = self.output_dist
            if odist:
                self._normalize_scalar_field_by_dist("absolute_precision", odist)
                self._normalize_range_field_by_dist("precision_tolerances", odist)
