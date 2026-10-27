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
API Testcase Structure
"""

__all__ = ["TestcaseAclnn", "AclnnApiParamResolver"]


# Standard Packages
import copy
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy.random

# Third-party Packages
from ttk.switches import get_global_storage

from ...utilities import (
    get,
    get_dtype_width,
    map_dtypes,
    normalize_dtype,
    shape_numel,
    shape_span,
    shape_stride,
)
from ...utilities.container import (
    deep_flatten,
    flatten_by_distribution,
    infer_list_distribution_from_nesting,
    nest_by_distribution,
)
from ..common.testcase.api_base import ApiTestcaseBase
from ..common.testcase.field_types import FIELD_TYPES
from ..common.testcase.tensor import (
    flat_slots,
    materialize_slots,
    slot_data_mirror,
    values_match_slots,
    values_skeleton,
    write_slot_data,
)
from .api_param import AclnnApiParamResolver
from .info_keeper import OpApiInfo, OpApiInfoKeeper


class TestcaseAclnn(ApiTestcaseBase):
    """
    Structure for Op Api Profiling
    """

    describe_pathway = "aclnn"

    __slots__ = (
        # === aclnn-specific testcase configurations === #
        "output_inplace_indexes",
        # input attributes
        "attributes",
        "attributes1",
        "attributes2",
        "attributes3",
        "attributes4",
        "attributes5",
        "attributes6",
        "attributes7",
        "attributes8",
        "attributes9",
        # specify dump file prefix
        "dump_file_prefix",
        # manual configurations
        "manual_tensor_binaries",
        # used for compare.
        "manual_golden_binaries",
        # === batch consistency === #
        # === multi-device (HCCL) support === #
        "my_rank",
        "hccl_comm",
        "hccl_comms",
        # === Runtime parameters below === #
        # torch.Tensor or sequence of torch.Tensor with strides & offsets.
        # （tensors/scalars 由 AoS 槽表属性提供）
        "golden_tensors",
        "golden_mode_override",
        "xpu_metrics",
        # golden-mode NPU（设备阶段执行 + 采集）
        "golden_perf",
        "golden_source",
        # === 私有 === #
        # scalar（scalar_dtypes/scalar_data_ranges 双模属性；scalars 槽表视图）
        "_scalar_dtypes",
        # other help configuration
        "_scalar_data_ranges",
        "_output_dtypes",
        "_output_view_shapes",
        "_output_view_offsets",
        "_output_view_strides",
        "_output_storage_shapes",
        "_actual_scalar_data_ranges",
        "_pure_attrs",
        "_pure_output_indexes",
        "_scalar_list_dist",
        "_scalar_slots",
        "_scalar_slots_from_declaration",
        "_scalars_written",
        "_multi_device_thread_contexts",
        "_multi_device_hccl_handles",
        # Host copies of dynamic inputs produced by npu_preprocess.  These are
        # consumed only while constructing ACLNN tensors; CPU golden/XPU paths
        # must continue to observe the CSV None marker.
        "_aclnn_dynamic_tensors",
    )

    identity_headers: Dict[str, tuple] = {
        **ApiTestcaseBase.identity_headers,
        "api_name": (FIELD_TYPES.STRING, None),  # Required
        "my_rank": (FIELD_TYPES.INT, None, None),
    }
    tensor_property_headers: Dict[str, tuple] = {
        "tensor_view_shapes": (FIELD_TYPES.SHAPELIKE_DYN_NESTED, None, ()),
        "tensor_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ("ND",)),
        "tensor_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
        "tensor_storage_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
        "tensor_view_offsets": (FIELD_TYPES.INT_CONTAINER_NESTED, None, ()),
        "tensor_view_strides": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
    }
    output_property_headers: Dict[str, tuple] = {
        "output_tensor_indexes": (FIELD_TYPES.INT_CONTAINER, None, None),  # required
        "output_inplace_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
    }
    attr_property_headers: Dict[str, tuple] = {
        "attributes": (FIELD_TYPES.DICT, None, {}),
        **{f"attributes{i}": (FIELD_TYPES.DICT, None, {}) for i in range(1, 10)},
    }
    scalar_property_headers: Dict[str, tuple] = {
        "scalar_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
    }
    special_property_headers: Dict[str, tuple] = {
        **ApiTestcaseBase.special_property_headers,
        "scalar_data_ranges": (FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED, None, ((None, None),)),
    }
    property_headers: Dict[str, tuple] = {
        **tensor_property_headers,
        **output_property_headers,
        **attr_property_headers,
        **scalar_property_headers,
        **special_property_headers,
    }
    option_headers: Dict[str, tuple] = {
        **ApiTestcaseBase.option_headers,
        # Manually controlled property
        "dump_file_prefix": (FIELD_TYPES.STRING, None, None),
        "manual_tensor_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
        "manual_golden_binaries": (FIELD_TYPES.STRING_CONTAINER, None, ()),
        # XPU 三方输出本地装载（--dump xpu 回放契约；FREE_EVAL 支持嵌套 TensorList 声明）
        "manual_xpu_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
    }
    batch_consistency_headers: Dict[str, tuple] = {}
    complete_headers: Dict[str, tuple] = {
        **identity_headers,
        **property_headers,
        **option_headers,
        **batch_consistency_headers,
    }

    def __init__(self):
        super().__init__()
        # output properties
        self.output_inplace_indexes: Optional[tuple] = None
        # attributes
        self.attributes: Optional[dict] = None
        self.attributes1 = None
        self.attributes2 = None
        self.attributes3 = None
        self.attributes4 = None
        self.attributes5 = None
        self.attributes6 = None
        self.attributes7 = None
        self.attributes8 = None
        self.attributes9 = None
        # Manual controlled parameters
        self.dump_file_prefix = None
        self.manual_tensor_binaries: Optional[Tuple[str, ...]] = None
        self.manual_golden_binaries = None
        # multi-device HCCL support
        self.my_rank: Optional[int] = None
        self.hccl_comm = None
        self.hccl_comms = None
        # End of testcase valid configurations
        self.golden_tensors = None
        self.golden_mode_override = None
        self.xpu_metrics = {}
        # golden-mode NPU（设备阶段执行 + 采集）
        self.golden_perf = None
        self.golden_source = None
        # === 私有 === #
        # scalar or scalarList dtype
        self._scalar_dtypes: Optional[tuple] = None
        # others
        self._scalar_data_ranges = None
        self._output_dtypes: Optional[tuple] = None
        self._output_view_shapes: Optional[tuple] = None
        self._output_view_offsets: Optional[tuple] = None
        self._output_view_strides: Optional[tuple] = None
        self._output_storage_shapes: Optional[tuple] = None
        self._actual_scalar_data_ranges = None
        self._pure_attrs: Optional[dict] = None
        self._pure_output_indexes: Optional[tuple] = None
        self._scalar_list_dist: Optional[tuple] = None
        self._scalar_slots = None
        self._scalar_slots_from_declaration = False
        self._scalars_written = False
        self._multi_device_thread_contexts = None
        self._aclnn_dynamic_tensors = {}

    @property
    def tensor_bytes(self):
        """Calculate total bytes of all flat tensors based on storage shape and dtype width."""
        bytes_lst: list = []
        for idx, vs in enumerate(self.flat_tensor_view_shapes):
            if vs is None or -1 in vs:
                continue
            try:
                bytes_lst.append(
                    shape_numel(self.flat_storage_shape(idx)) * get_dtype_width(get(self.flat_tensor_dtypes, idx))
                )
            except Exception:
                bytes_lst.append(0)
        return sum(bytes_lst)

    @property
    def output_dtypes(self) -> tuple:
        """Resolve output dtypes from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes:
          - Single tensor → single dtype string
          - TensorList → tuple of dtype strings
        """
        if self._output_dtypes is None:
            dist = self.tensor_list_dist
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = self.flat_index_of(idx)
                num = dist[idx] if idx < len(dist) and dist[idx] > 0 else 0
                count = max(num, 1)
                dtypes = self.flat_tensor_dtypes[flat_idx : flat_idx + count]
                result.append(dtypes[0] if num == 0 else dtypes)
            self._output_dtypes = tuple(result)
        return self._output_dtypes

    @property
    def flat_output_dtypes(self):
        """Flatten output dtypes to one-per-tensor. Assumes output_dtypes is correctly nested."""
        if not self.output_dtypes:
            return self.output_dtypes
        odist = self.output_dist
        return flatten_by_distribution(self.output_dtypes, odist) if odist else self.output_dtypes

    @property
    def output_view_shapes(self) -> tuple:
        """Resolve output view shapes from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_view_shapes is None:
            dist = self.tensor_list_dist
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = self.flat_index_of(idx)
                num = dist[idx] if idx < len(dist) and dist[idx] > 0 else 0
                count = max(num, 1)
                shapes = self.flat_tensor_view_shapes[flat_idx : flat_idx + count]
                result.append(shapes[0] if num == 0 else shapes)
            self._output_view_shapes = tuple(result)
        return self._output_view_shapes

    @property
    def flat_output_view_shapes(self):
        """Flatten output view shapes to one-per-tensor. Assumes output_view_shapes is correctly nested."""
        if not self.output_view_shapes:
            return self.output_view_shapes
        odist = self.output_dist
        return flatten_by_distribution(self.output_view_shapes, odist) if odist else self.output_view_shapes

    @property
    def output_view_offsets(self) -> tuple:
        """Resolve output view offsets from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_view_offsets is None:
            dist = self.tensor_list_dist
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = self.flat_index_of(idx)
                num = dist[idx] if idx < len(dist) and dist[idx] > 0 else 0
                count = max(num, 1)
                if num == 0:
                    result.append(self.flat_view_offset(flat_idx))
                else:
                    result.append(tuple(self.flat_view_offset(flat_idx + i) for i in range(count)))
            self._output_view_offsets = tuple(result)
        return self._output_view_offsets

    @property
    def flat_output_view_offsets(self):
        """Flatten output view offsets to one-per-tensor. Assumes output_view_offsets is correctly nested."""
        if not self.output_view_offsets:
            return self.output_view_offsets
        odist = self.output_dist
        return flatten_by_distribution(self.output_view_offsets, odist) if odist else self.output_view_offsets

    @property
    def output_view_strides(self) -> tuple:
        """Resolve output view strides from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_view_strides is None:
            dist = self.tensor_list_dist
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = self.flat_index_of(idx)
                num = dist[idx] if idx < len(dist) and dist[idx] > 0 else 0
                count = max(num, 1)
                if num == 0:
                    result.append(self.flat_view_stride(flat_idx))
                else:
                    result.append(tuple(self.flat_view_stride(flat_idx + i) for i in range(count)))
            self._output_view_strides = tuple(result)
        return self._output_view_strides

    @property
    def flat_output_view_strides(self):
        """Flatten output view strides to one-per-tensor. Assumes output_view_strides is correctly nested."""
        if not self.output_view_strides:
            return self.output_view_strides
        odist = self.output_dist
        return flatten_by_distribution(self.output_view_strides, odist) if odist else self.output_view_strides

    @property
    def output_storage_shapes(self) -> tuple:
        """Resolve output storage shapes from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_storage_shapes is None:
            dist = self.tensor_list_dist
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = self.flat_index_of(idx)
                num = dist[idx] if idx < len(dist) and dist[idx] > 0 else 0
                count = max(num, 1)
                if num == 0:
                    result.append(self.flat_storage_shape(flat_idx))
                else:
                    result.append(tuple(self.flat_storage_shape(flat_idx + i) for i in range(count)))
            self._output_storage_shapes = tuple(result)
        return self._output_storage_shapes

    @property
    def flat_output_storage_shapes(self):
        """Flatten output storage shapes to one-per-tensor. Assumes output_storage_shapes is correctly nested."""
        if not self.output_storage_shapes:
            return self.output_storage_shapes
        odist = self.output_dist
        return flatten_by_distribution(self.output_storage_shapes, odist) if odist else self.output_storage_shapes

    @property
    def actual_scalar_data_ranges(self):
        """Return scalar data ranges as tuple of (min, max) pairs, stripping extra fields."""
        data_range = self._actual_scalar_data_ranges or self.scalar_data_ranges
        return data_range if not data_range else tuple([tuple(dr[:2]) for dr in data_range])

    @property
    def flat_manual_tensor_binaries(self):
        """manual_tensor_binaries 扁平化（嵌套声明随 TensorList 分布展开；str 视为叶子）。"""
        if not self.manual_tensor_binaries:
            return ()
        return deep_flatten(self.manual_tensor_binaries)

    @property
    def full_scalar_data_ranges(self):
        """Return full scalar data ranges including extra fields (e.g., distribution type)."""
        return self._actual_scalar_data_ranges or self.scalar_data_ranges

    @property
    def pure_attrs(self) -> dict:
        """Return attributes that are pure API parameters (not tensor/scalar overrides)."""
        if self._pure_attrs is None:
            self._pure_attrs = {}
            op_api_info = OpApiInfoKeeper().info_of(self.api_name)
            if op_api_info is None:
                # 显式报错：此前直接解引用 None.params 抛 AttributeError，
                # 正确性依赖 worker 前置 has_api 检查的隐式时序（本属性还被
                # _profiling_print / xpu_attrs 访问），显式化防未来调用方绕过。
                raise RuntimeError(f"[{self.testcase_name}] API {self.api_name} not registered (pure_attrs)")
            for k, v in self.attributes.items():
                if k not in op_api_info.params:
                    continue
                if k in op_api_info.tensors or k in op_api_info.scalars:
                    continue
                self._pure_attrs.update({k: v})
        return self._pure_attrs

    @property
    def xpu_attrs(self) -> dict:
        """Return attrs for XPU dispatch: pure_attrs + scalars (scalars are not in
        X-Input-Schema, which only transports tensors, so they must go via attrs).

        ScalarList 参数（scalar_list_dist>0）按签名分布聚合为列表整体下发，
        与执行路径 _create_acl_scalar 的聚合方式一致；普通标量仍单值下发。"""
        attrs = dict(self.pure_attrs)
        op_api_info = OpApiInfoKeeper().info_of(self.api_name)
        if op_api_info and self.scalars is not None:
            flat_scalars = deep_flatten(self.scalars)
            dist = self.scalar_list_dist
            groups = nest_by_distribution(flat_scalars, dist) if dist else flat_scalars
            for name, group in zip(op_api_info.scalars, groups):
                if isinstance(group, (tuple, list)):
                    if any(s is not None for s in group):
                        attrs[name] = [s.item() if hasattr(s, "item") else s for s in group]
                elif group is not None:
                    attrs[name] = group.item() if hasattr(group, "item") else group
        return attrs

    @property
    def pure_output_indexes(self):
        """Return flat indexes of output tensors that are NOT inplace.

        Computed from nested output_tensor_indexes and output_inplace_indexes
        expanded through distribution to flat indexes. Out-of-range indexes
        are skipped; validate marks them invalid.
        """
        if self._pure_output_indexes is None:
            dist = self.tensor_list_dist
            flat_output = set()
            for idx in self.output_tensor_indexes:
                if not 0 <= idx < len(dist):
                    continue
                flat_idx = self.flat_index_of(idx)
                count = dist[idx] if dist[idx] > 0 else 1
                flat_output.update(range(flat_idx, flat_idx + count))
            flat_inplace = set()
            for idx in self.output_inplace_indexes or ():
                if not 0 <= idx < len(dist):
                    continue
                flat_idx = self.flat_index_of(idx)
                count = dist[idx] if dist[idx] > 0 else 1
                flat_inplace.update(range(flat_idx, flat_idx + count))
            self._pure_output_indexes = sorted(flat_output - flat_inplace)
        return self._pure_output_indexes

    # ========== npu_preprocess absorption ==========

    def absorb_preprocess(self, result, plan, args=None, kwargs=None):
        """ACLNN 落地：dict → 校验+主机副本写 _aclnn_dynamic_tensors；None 且
        声明 -1 → 隐式通道核验（槽位反刷经 flat_tensors 由 Phase1ParamBuilder
        原路消费，无需补丁）。"""
        from ..common.invoke import apply_aclnn_npu_preprocess_result

        if result is None:
            if not self.dyn_input_slot_indexes:
                return None
            self._require_dynamic_slots_materialized()
            return True
        apply_aclnn_npu_preprocess_result(self, plan, result)
        return True

    @actual_scalar_data_ranges.setter
    def actual_scalar_data_ranges(self, value):
        self._actual_scalar_data_ranges = value

    def validate(self):
        """Run all validation checks on testcase configuration."""
        super().validate()
        self._parse_device_ids_field()
        self._check_api_name()
        self._apply_describe_shapes()
        if not self.is_valid:
            return
        self._normalize_compressed_fields()
        self._check_tensor_parm()
        self._parse_tensor_dtypes()
        self._check_tensor_list_configuration()
        self._check_scalar_list_configuration()
        self._auto_fill_output_inplace_indices()
        self._check_output_configuration()
        self._check_dyn_shape_slots()
        self._parse_scalar_dtypes()
        self._check_params_count()
        self._stc_shape_size_check()
        self._check_manual_xpu_binaries()
        self._generate_batch_consistency_id()

    def _stc_shape_size_check(self):
        """HBM 输入守门（-l/--limit）：输入张量总量超限判无效（对齐 kernel 通路）。

        aclnn 输出 shape 由 aclnnGetWorkspaceSize 运行期决定（CSV 不声明），
        守门按输入侧计；-1 动态槽位（None shape）跳过。"""
        if not self.is_valid:
            return
        switches = get_global_storage()
        total_bytes = 0
        flat_shapes = self.flat_tensor_view_shapes or ()
        flat_dtypes = self.flat_tensor_dtypes or ()
        for idx, shape in enumerate(flat_shapes):
            if shape is None:
                continue
            dtype = get(flat_dtypes, idx)
            total_bytes += shape_numel(shape) * (get_dtype_width(dtype) if dtype else 0)
        if total_bytes > switches.hbm_size_limit * 1024 * 1024 * 1024:
            logging.error(
                f"Testcase {self.testcase_name} is invalid because shape is out of bound: "
                f"limit={switches.hbm_size_limit}GB, inputs: {self.tensor_view_shapes}, "
                f"dtypes: {self.tensor_dtypes} (output shapes are runtime-resolved, inputs counted)"
            )
            self.is_valid = False
            self.fail_reason = "STC_SHAPE_OUT_OF_BOUND"

    def _normalize_compressed_fields(self):
        """Expand compressed tensor/scalar fields to fully specified nested format.

        After normalization each field matches tensor_view_shapes nesting exactly:
          - Per-TensorList: fully expanded, no len-1 inner tuples
          - Top-level: fully expanded, no len-1 broadcast
        e.g. tensor_dtypes ('float32',) with dist (2,0) -> (('float32','float32'),'float32')
        """
        if not self.is_valid:
            return
        sdist = self.scalar_list_dist
        if sdist:
            self._normalize_scalar_field_by_dist("scalar_dtypes", sdist)
            self._normalize_range_field_by_dist("scalar_data_ranges", sdist)
        super()._normalize_compressed_fields()

    # ========== Scalar AoS substrate（标量 = shape=() 的 Tensor） ==========

    @property
    def scalar_dtypes(self):
        return self._scalar_declaration_view("_scalar_dtypes", "dtype")

    @scalar_dtypes.setter
    def scalar_dtypes(self, value):
        self._scalar_dtypes = value
        self._after_scalar_declaration_write()

    @property
    def scalar_data_ranges(self):
        return self._scalar_declaration_view("_scalar_data_ranges", "data_range")

    @scalar_data_ranges.setter
    def scalar_data_ranges(self, value):
        self._scalar_data_ranges = value
        self._after_scalar_declaration_write()

    @property
    def scalar_slots(self):
        """AoS 标量槽表（shape=() 的 Tensor）。首次访问按 raw 声明列物化。"""
        if self._scalar_slots is None:
            slots = self._materialize_scalar_slots()
            self._scalar_slots = slots if slots is not None else ()
            self._scalar_slots_from_declaration = slots is not None
        return self._scalar_slots

    def _materialize_scalar_slots(self):
        """raw 标量声明列 → 标量槽表（dtype None 位物化为 None 槽）。

        dist：keeper 签名优先（scalar_list_dist）；无 keeper 场景回退
        raw 嵌套推断（保证顶层槽位数）。
        """
        if not self._scalar_dtypes:
            return None
        dist = self.scalar_list_dist or infer_list_distribution_from_nesting(self._scalar_dtypes)
        flat_dtypes = self._flat_decl_by_dist(self._scalar_dtypes, dist)
        return materialize_slots(
            {
                "view_shapes": tuple(None if d is None else () for d in flat_dtypes),
                "dtypes": flat_dtypes,
                "data_ranges": self._flat_decl_by_dist(self._scalar_data_ranges, dist),
            },
            dist,
        )

    def _scalar_declaration_view(self, private, attr):
        """嵌套声明视图：标量槽表物化且 raw 非空时从槽表派生，否则 raw 透传。"""
        raw = getattr(self, private)
        if not self._scalar_slots_from_declaration or not raw:
            return raw
        return tuple(
            tuple(None if t is None else getattr(t, attr) for t in slot)
            if isinstance(slot, (tuple, list))
            else (None if slot is None else getattr(slot, attr))
            for slot in self._scalar_slots
        )

    def _after_scalar_declaration_write(self):
        # 不重置 _scalar_list_dist：dtype 值写不改结构，且重物化依赖 keeper dist。
        self._flat_scalars = None
        if self._scalar_slots is not None:
            self._rematerialize_scalar_slots_preserving_runtime()

    def _rematerialize_scalar_slots_preserving_runtime(self):
        old_flat = flat_slots(self._scalar_slots) if self._scalar_slots is not None else ()
        new_slots = self._materialize_scalar_slots()
        if new_slots is None:
            return
        for new, old in zip(flat_slots(new_slots), old_flat):
            if new is None or old is None:
                continue
            new.data = old.data
            new.storage = old.storage
            new.data_kind = old.data_kind
            new.name = old.name
        self._scalar_slots = new_slots
        self._scalar_slots_from_declaration = True

    @property
    def scalars(self):
        """嵌套运行时镜像（list[data | list[data] | None]）——标量槽表派生兼容视图。"""
        if self._scalar_slots is None or not self._scalars_written:
            return None
        return slot_data_mirror(self._scalar_slots)

    @scalars.setter
    def scalars(self, value):
        if value is None:
            self._scalars_written = False
            self._flat_scalars = None
            return
        slots = self._scalar_slots
        if slots is not None and values_match_slots(slots, value):
            write_slot_data(slots, value)
        else:
            self._rebuild_scalar_slots_for_values(value)
        self._scalars_written = True
        self._flat_scalars = None

    def _rebuild_scalar_slots_for_values(self, value):
        """结构不一致时重建标量槽表：声明优先，值结构兜底；按 flat 位保留已有 storage。"""
        old_flat = flat_slots(self._scalar_slots) if self._scalar_slots is not None else ()
        new_slots = self._materialize_scalar_slots()
        if new_slots is not None and values_match_slots(new_slots, value):
            self._scalar_slots_from_declaration = True
        else:
            new_slots = values_skeleton(value)
            self._scalar_slots_from_declaration = False
        write_slot_data(new_slots, value)
        for slot, old in zip(flat_slots(new_slots), old_flat):
            if slot is not None and old is not None and old.storage is not None:
                slot.storage = old.storage
        self._scalar_slots = new_slots

    @property
    def scalar_list_dist(self):
        """ScalarList distribution from API info + scalar_dtypes.

        Uses API info to determine which positions are aclScalarList*.
        For ScalarList positions, takes element count from scalar_dtypes.
        For Scalar positions, marks as 0 (supports compressed format).

        Returns a tuple where each element corresponds to a top-level parameter:
          >0 means ScalarList with that many scalars, 0 means single scalar.
        Cached on first access.
        """
        if self._scalar_list_dist is not None:
            return self._scalar_list_dist
        if not self.api_name or not self.scalar_dtypes:
            return ()
        try:
            op_api_info = OpApiInfoKeeper().info_of(self.api_name)
        except Exception:
            return ()
        if not op_api_info.scalars:
            return ()
        result = []
        for idx, name in enumerate(op_api_info.scalars):
            param_type = op_api_info.params[name]["type"]
            if param_type == "aclScalarList*":
                element = self.scalar_dtypes[idx] if idx < len(self.scalar_dtypes) else ()
                if isinstance(element, (tuple, list)):
                    result.append(len(element))
                elif isinstance(element, str):
                    result.append(1)
                else:
                    result.append(0)
            else:
                result.append(0)
        self._scalar_list_dist = tuple(result)
        return self._scalar_list_dist

    @property
    def flat_scalar_dtypes(self):
        """Flatten normalized scalar_dtypes to per-scalar dtypes.

        标量槽表物化后从槽表派生（无叶子歧义）；物化前 dist 切割（legacy）。
        """
        raw = self._scalar_dtypes
        if not raw:
            return raw
        if self._scalar_slots_from_declaration:
            return tuple(None if t is None else t.dtype for t in flat_slots(self._scalar_slots))
        dist = self.scalar_list_dist
        if dist:
            return flatten_by_distribution(raw, dist)
        return raw

    @property
    def flat_scalar_data_ranges(self):
        """Flatten normalized scalar_data_ranges to per-scalar (min, max) pairs.

        标量槽表物化后从槽表派生；物化前 dist 切割（legacy）。
        """
        raw = self._scalar_data_ranges
        if not raw:
            return raw
        if self._scalar_slots_from_declaration:
            return tuple(None if t is None else t.data_range for t in flat_slots(self._scalar_slots))
        dist = self.scalar_list_dist
        if dist:
            return flatten_by_distribution(raw, dist)
        return raw

    @staticmethod
    def _expand_by_distribution(values, distribution):
        """Expand a short tuple to match flattened tensor count.

        e.g. values=('float32',), distribution=(2,0) -> ('float32','float32')
        """
        if not values or not distribution:
            return values
        result = []
        vi = 0
        for num in distribution:
            if num == 0:
                result.append(get(values, vi))
                vi += 1
            else:
                for _ in range(num):
                    result.append(get(values, vi))
                vi += 1
        return tuple(result)

    def is_torch_dtype_support(self) -> bool:
        """Check if all dtypes in testcase are supported by the current torch natively."""
        from ttk.utilities.dtypes import is_torch_native_dtype

        for dtype in self.flat_tensor_dtypes:
            if dtype is not None and not is_torch_native_dtype(dtype):
                return False
        # flat_scalar_dtypes：嵌套 ScalarList 声明须按元素判定（遍历嵌套结构会把
        # tuple 元素当 dtype 误判为非原生，导致全原生 dtype 用例被错误路由到 numpy 通道）
        return all(dtype is None or is_torch_native_dtype(dtype) for dtype in self.flat_scalar_dtypes)

    def parse_device_ids(self, raw_value):
        if raw_value is None or raw_value == "":
            self.device_ids = None
            self.my_rank = None
            return
        if isinstance(raw_value, str):
            self.device_ids = tuple(int(d.strip()) for d in raw_value.split(",") if d.strip())
        elif isinstance(raw_value, (tuple, list)):
            self.device_ids = tuple(int(d) for d in raw_value)
        else:
            self.device_ids = (int(raw_value),)

    def get_param_plan(self):
        """Resolve and cache parameter assembly plan for this testcase.

        The plan captures the C function signature layout once,
        then custom input, custom golden, and profiling all reuse it
        — same param order everywhere.

        Returns:
            AclnnApiParamResolver or None (if API info unavailable)
        """
        if self._param_plan_cache is not None:
            return self._param_plan_cache
        if not self.api_name:
            return None
        try:
            op_api_info = OpApiInfoKeeper().info_of(self.api_name)
            if op_api_info is None:
                return None
            plan = AclnnApiParamResolver(self.api_name, op_api_info)
            self._param_plan_cache = plan
            return plan
        except Exception:
            return None

    @staticmethod
    def _tensor_param_valid(view_shape, view_stride, view_offset, storage_shape):  # noqa: PLR0911
        """Validate that view_shape/stride/offset fit within storage_shape."""
        if not view_stride and not view_offset and not storage_shape:
            return True
        # per-slot None falls back to the default 0 (flat_view_offset contract)
        view_offset = view_offset or 0
        if view_offset < 0:
            logging.error(f"Negative view_offset is not supported: {view_offset}")
            return False
        if not view_stride:
            view_stride = shape_stride(view_shape)
        if not storage_shape:
            storage_shape = view_shape
        if len(view_shape) != len(view_stride):
            logging.error(f"Rank of view_shape/view_strides should be same: {view_shape} vs {view_stride}")
            return False
        for _, v in enumerate(view_stride):
            if v < 0:
                logging.error(f"Negative view_strides are not supported: {view_stride}.")
                return False
        view_span = shape_span(view_shape, view_stride)
        if view_span == 0:
            # (a tensor with arbitrary 0 dims)'s storage can have any numel
            return True
        storage_numel = shape_numel(storage_shape)
        if view_span + view_offset > storage_numel:
            logging.error(
                f"view_shape ({view_shape}), view_strides ({view_stride}), view_offset ({view_offset}) "
                f"requiring a storage with elements {view_span + view_offset} "
                f"is out of bounds for storage with elements {storage_numel} ({storage_shape})"
            )
            return False
        return True

    @staticmethod
    def _infer_storage_shape(shape, stride, offset):
        """Infer minimum storage shape from view shape, stride and offset."""
        min_storage_size = shape_span(shape, stride)

        def find_original_dims(_stride):
            dims = []
            for i in range(len(_stride) - 1):
                if _stride[i] == 0:
                    dims.append(1)
                elif _stride[i + 1] == 0:
                    dims.append(_stride[i])
                elif _stride[i] % _stride[i + 1] == 0:
                    dims.append(_stride[i] // _stride[i + 1])
                else:
                    dims.append(1)
            dims.append(_stride[-1] or 1)
            return dims

        original_dims = find_original_dims(stride)
        while True:
            total_elements = 1
            for dim in original_dims:
                total_elements *= dim
            if total_elements >= min_storage_size:
                break
            original_dims[0] += 1

        original_dims[-1] += offset
        return tuple(original_dims)

    def _random_non_contiguous(self, view_shape, offset):
        """Generate random non-contiguous stride and storage shape for a given view shape."""
        if len(view_shape) == 1:
            if view_shape[0] >= 1024 * 1024 * 512:  # 2GB for fp32, 1GB for bfp16/fp16
                return (1,), view_shape + offset
            return (2,), self._infer_storage_shape(view_shape, (2,), offset)
        if all(view_shape[idx] == view_shape[0] for idx in range(len(view_shape))):
            stride = list(shape_stride(view_shape))
            ex_idx = sorted(numpy.random.choice(len(stride), size=2, replace=False))
            stride[ex_idx[0]], stride[ex_idx[1]] = stride[ex_idx[1]], stride[ex_idx[0]]
            ss = list(copy.deepcopy(view_shape))
            ss[-1] += offset
            return tuple(stride), tuple(ss)
        vs = list(view_shape)
        ex_idx = sorted(numpy.random.choice(len(vs), size=2, replace=False))
        vs[ex_idx[0]], vs[ex_idx[1]] = vs[ex_idx[1]], vs[ex_idx[0]]
        stride = shape_stride(vs)
        vs[-1] += offset
        return stride, tuple(vs)

    def _check_api_name(self):
        """Validate api_name is specified and exists in OpApiInfo."""
        if not self.api_name:
            self.is_valid = False
            self.fail_reason = "API_NAME_MISSING"
            logging.error("api_name must be specified.")
            return
        if not OpApiInfoKeeper().has_api(self.api_name):
            self.is_valid = False
            self.fail_reason = "OP_API_NOT_FOUND"
            logging.error(f"OpApi [{self.api_name}] is not found.")

    def _check_tensor_parm(self):
        """Validate view/stride/offset/storage consistency for each flat tensor."""
        if not self.is_valid:
            return
        self._check_shape_nesting(self.tensor_view_strides, "tensor_view_strides")
        self._check_shape_nesting(self.tensor_storage_shapes, "tensor_storage_shapes")
        if not self.is_valid:
            return
        flat_strides = self.flat_tensor_view_strides or ()
        flat_offsets = self.flat_tensor_view_offsets or ()
        flat_storages = self.flat_tensor_storage_shapes or ()
        for idx, view_shape in enumerate(self.flat_tensor_view_shapes):
            if view_shape is None:
                continue
            view_stride = get(flat_strides, idx, out_of_range=())
            view_offset = get(flat_offsets, idx, out_of_range=0)
            storage_shape = get(flat_storages, idx, out_of_range=())
            if not self._tensor_param_valid(view_shape, view_stride, view_offset, storage_shape):
                self.is_valid = False
                self.fail_reason = "TENSOR_PARAM_INVALID"
                return

    def _check_shape_nesting(self, field, field_name):
        """Validate that field's top-level count matches tensor_view_shapes when nested.

        view_strides and storage_shapes must either be empty, flat (match flat_count),
        or have the same top-level count as tensor_view_shapes.
        """
        if not self.is_valid or not field or not self.tensor_view_shapes:
            return
        top_count = len(self.tensor_view_shapes)
        if len(field) not in (0, 1, len(self.flat_tensor_view_shapes), top_count):
            self.is_valid = False
            self.fail_reason = "TENSOR_PARAM_INVALID"
            logging.error(
                f"{field_name} length [{len(field)}] is inconsistent with "
                f"tensor_view_shapes top-level count [{top_count}]."
            )

    def _build_non_contiguous_tensor(self):
        """Auto-generate random non-contiguous strides and storage shapes for tensors that lack them."""

        def __fill(_fill_size: int, _fill_ele: Any, _to_fill: Union[list, tuple]):
            if not _to_fill:
                _to_fill = [_fill_ele] * _fill_size
            _num = len(_to_fill)
            if _num < _fill_size:
                _to_fill = list(_to_fill).append([_fill_ele] * (_fill_size - _num))
            return list(_to_fill)

        def _random_offset():
            return numpy.random.randint(0, 32)

        if not self.is_valid:
            return
        flat_shapes = self.flat_tensor_view_shapes
        tensor_num = len(flat_shapes)
        view_strides = __fill(tensor_num, (), self.flat_tensor_view_strides)
        view_offsets = __fill(tensor_num, None, self.flat_tensor_view_offsets)
        storage_shapes = __fill(tensor_num, (), self.flat_tensor_storage_shapes)
        for idx, view_shape in enumerate(flat_shapes):
            # if storage shape is specified or scalar tensor. skip
            if view_shape is None or len(storage_shapes[idx]) > 0 or len(view_shape) == 0:
                # if storage shape is specified or scalar tensor. skip
                if view_offsets[idx] is None:
                    view_offsets[idx] = 0
                continue
            if view_offsets[idx] is None:
                view_offsets[idx] = _random_offset()
            if view_offsets[idx] < 0:
                # invalid !! leave it to validate later.
                continue
            if not view_strides[idx]:
                # neither storage shape or stride is specified. random it.
                view_strides[idx], storage_shapes[idx] = self._random_non_contiguous(view_shape, view_offsets[idx])
            else:
                # if storage shape is not specified. try to infer it.
                if len(view_shape) != len(view_strides[idx]):
                    # invalid !! leave it to validate later.
                    continue
                if any(d < 0 for d in view_strides[idx]):
                    # invalid !! leave it to validate later.
                    continue
                storage_shapes[idx] = self._infer_storage_shape(view_shape, view_strides[idx], view_offsets[idx])
        self.tensor_view_strides = tuple(view_strides)
        self.tensor_view_offsets = tuple(view_offsets)
        self.tensor_storage_shapes = tuple(storage_shapes)

    def _parse_tensor_dtypes(self):
        """Parse and normalize tensor dtype strings to standard form, preserving nesting."""
        if not self.is_valid:
            return
        try:
            if self.tensor_dtypes:
                self.tensor_dtypes = map_dtypes(self.tensor_dtypes, normalize_dtype)
        except Exception:
            self.is_valid = False
            self.fail_reason = "TENSOR_DTYPES_INVALID"
            logging.exception(f"Tensor dtypes parse failed: {self.tensor_dtypes}")

    def _parse_scalar_dtypes(self):
        """Parse and normalize scalar dtype strings to standard form, preserving nesting."""
        if not self.is_valid:
            return
        try:
            if self.scalar_dtypes:
                self.scalar_dtypes = map_dtypes(self.scalar_dtypes, normalize_dtype)
        except Exception:
            self.is_valid = False
            self.fail_reason = "SCALAR_DTYPES_INVALID"
            logging.exception(f"Scalar dtypes parse failed: {self.scalar_dtypes}")

    def _check_param_configuration(self, case_values, api_param_names, kind, is_nested_fn):
        """Generic check for tensor/scalar parameter count and type matching against API definition.

        Args:
            case_values: Nested testcase values (tensor_view_shapes or scalar_dtypes).
            api_param_names: Ordered parameter names from API info.
            kind: 'Tensor' or 'Scalar' for error messages.
            is_nested_fn: Callable to detect if an element represents a List type.
        """
        if not self.is_valid:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        api_count = len(api_param_names)
        case_count = len(case_values) if case_values else 0
        if case_count == 0 and api_count == 0:
            return
        if case_count != api_count:
            self.is_valid = False
            self.fail_reason = f"{kind}_COUNT_MISMATCH"
            logging.error(
                f"L2 interface [{self.api_name}] has {api_count} {kind.lower()} parameters, "
                f"but testcase configured {case_count}."
            )
            return
        single_type = f"acl{kind}*"
        list_type = f"acl{kind}List*"
        for idx, element in enumerate(case_values):
            name = api_param_names[idx]
            typ = op_api_info.params[name]["type"]
            if is_nested_fn(element):
                if typ == list_type:
                    continue
                self.is_valid = False
                self.fail_reason = "PARAM_TYPE_MISMATCH"
                logging.error(f"Parameter [{name}] type is [{typ}]. But got {kind}List.")
                return
            if element is not None and typ != single_type:
                self.is_valid = False
                self.fail_reason = "PARAM_TYPE_MISMATCH"
                logging.error(f"Parameter [{name}] type is [{typ}]. But got {kind}.")
                return

    @staticmethod
    def _is_tensor_list_element(element):
        """Check if element represents a TensorList (tuple of shape tuples)."""
        return isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list))

    @staticmethod
    def _is_scalar_list_element(element):
        """Check if element represents a ScalarList (tuple of dtype strings, not a single string)."""
        return isinstance(element, (tuple, list)) and len(element) > 0 and not isinstance(element[0], str)

    def _check_tensor_list_configuration(self):
        """Validate tensor parameters match API definition in count and type (Tensor/TensorList)."""
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        self._check_param_configuration(
            self.tensor_view_shapes, op_api_info.tensors, "Tensor", self._is_tensor_list_element
        )

    def _check_scalar_list_configuration(self):
        """Validate scalar parameters match API definition in count and type (Scalar/ScalarList).

        Uses API info to determine which positions are aclScalarList*,
        since scalar_dtypes nesting alone cannot distinguish ScalarList from multiple Scalars.
        """
        if not self.is_valid:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        api_scalars = op_api_info.scalars
        case_count = len(self.scalar_dtypes) if self.scalar_dtypes else 0
        api_count = len(api_scalars)
        if case_count == 0 and api_count == 0:
            return
        if case_count != api_count:
            self.is_valid = False
            self.fail_reason = "Scalar_COUNT_MISMATCH"
            logging.error(
                f"L2 interface [{self.api_name}] has {api_count} scalar parameters, "
                f"but testcase configured {case_count}."
            )
            return
        for idx, name in enumerate(api_scalars):
            typ = op_api_info.params[name]["type"]
            element = self.scalar_dtypes[idx]
            is_list_type = typ == "aclScalarList*"
            is_nested = isinstance(element, (tuple, list)) and len(element) > 0
            if is_list_type and not is_nested:
                self.is_valid = False
                self.fail_reason = "PARAM_TYPE_MISMATCH"
                logging.error(
                    f"Parameter [{name}] type is [{typ}]. "
                    f"But scalar_dtypes element is not a tuple/list (expected ScalarList). "
                    f"Got: {element}"
                )
                return
            if not is_list_type and is_nested and isinstance(element[0], (tuple, list)):
                self.is_valid = False
                self.fail_reason = "PARAM_TYPE_MISMATCH"
                logging.error(f"Parameter [{name}] type is [{typ}]. But got ScalarList.")
                return

    def _auto_fill_output_inplace_indices(self):
        """
        automatic fill output_inplace_indexes
        as per parameter's name (ends with `Ref`)
        """
        if not self.is_valid:
            return
        if self.output_inplace_indexes:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        ref_lst = [n for n in op_api_info.tensors if n.endswith("Ref")]
        ref_cnt = len(ref_lst)
        if ref_cnt == 0:
            return
        logging.warning(
            f"L2 interface [{self.api_name}] has some inplace tensors: {ref_lst}. "
            f"But `output_inplace_indexes` is not configured. "
            f"Try to fill it automatically."
        )
        inplace_indices = []
        for idx, element in enumerate(self.tensor_view_shapes):
            param_name = op_api_info.tensors[idx]
            if param_name in ref_lst:
                if element is None:
                    logging.info(f"Inplace parameter [{param_name}] is None (nullptr), skipping.")
                    continue
                inplace_indices.append(idx)
        self.output_inplace_indexes = tuple(inplace_indices)

    # Backward/Grad API 中排除的输入参数名（以 Out/Output 结尾但实际是输入）
    _BACKWARD_OUTPUT_EXCLUDE = frozenset(
        {
            "gradOutput",
            "gradOut",
            "grad_output",
            "attentionOut",
        }
    )

    def _auto_fill_output_tensor_indexes(self):
        """Auto-fill output_tensor_indexes from tensor param naming conventions.

        Rules (in priority order):
          b. *Ref suffix → inplace output
          c. *Out / *Output / "output" → output candidate
             Backward/Grad API exclusions:
               - names in _BACKWARD_OUTPUT_EXCLUDE → skip
               - "output" that is NOT the last tensor → skip
          f. No candidates found → fallback to (-1,) (last tensor)
        """
        if not self.is_valid or self.output_tensor_indexes:
            return

        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        is_backward = "Backward" in self.api_name or "Grad" in self.api_name
        tensor_names = op_api_info.tensors
        output_indices = []

        for idx, name in enumerate(tensor_names):
            if name.endswith("Ref"):
                output_indices.append(idx)
                continue
            is_output_name = name == "output" or name.endswith(("Out", "OutOptional", "Output", "OutputOptional"))
            if not is_output_name:
                continue
            if is_backward:
                if name in self._BACKWARD_OUTPUT_EXCLUDE:
                    continue
                if name == "output" and idx != len(tensor_names) - 1:
                    continue
            output_indices.append(idx)

        if not output_indices:
            output_indices = [-1]

        self.output_tensor_indexes = tuple(output_indices)

    def _check_output_configuration(self):
        """
        1. if output_tensor_indexes is not set, auto-fill from naming conventions.
        2. correct negative indexes, validate range.
        3. output_inplace_indexes should be included in output_indexes
        """
        if not self.is_valid:
            return
        self._auto_fill_output_tensor_indexes()
        # correct negative indexes to positive ones
        self.output_tensor_indexes = tuple(
            [idx + len(self.tensor_view_shapes) if idx < 0 else idx for idx in self.output_tensor_indexes]
        )
        out_of_ranges = [idx for idx in self.output_tensor_indexes if idx < 0 or idx >= len(self.tensor_view_shapes)]
        if out_of_ranges:
            self.is_valid = False
            self.fail_reason = "OUTPUT_INDEX_INVALID"
            logging.error(
                f"Indexes in `output_tensor_indexes` {self.output_tensor_indexes} "
                f"out of range of `tensor_view_shapes`'s indexes: "
                f"[0, {len(self.tensor_view_shapes)})"
            )
            return
        if not self.output_inplace_indexes:
            return
        # correct negative indexes to positive ones
        self.output_inplace_indexes = tuple(
            [idx + len(self.tensor_view_shapes) if idx < 0 else idx for idx in self.output_inplace_indexes]
        )
        out_of_ranges = [idx for idx in self.output_inplace_indexes if idx < 0 or idx >= len(self.tensor_view_shapes)]
        if out_of_ranges:
            self.is_valid = False
            self.fail_reason = "OUTPUT_INPLACE_INDEX_INVALID"
            logging.error(
                f"Indexes in `output_inplace_indexes` {self.output_inplace_indexes} "
                f"out of range of `tensor_view_shapes`'s indexes: "
                f"[0, {len(self.tensor_view_shapes)})"
            )
            return
        for ii in self.output_inplace_indexes:
            if ii in self.output_tensor_indexes:
                continue
            self.is_valid = False
            self.fail_reason = "OUTPUT_INDEX_MISMATCH"
            logging.error(
                f"Indexes in `output_tensor_indexes` and "
                f"`output_inplace_indexes` mismatch: "
                f"{self.output_tensor_indexes} vs {self.output_inplace_indexes}"
            )
            return

    def _check_params_count(self):
        """Validate total parameter count (tensor + scalar + attribute) matches API definition."""
        if not self.is_valid:
            return
        op_api_info = OpApiInfoKeeper().info_of(self.api_name)
        tensor_count = len(self.tensor_view_shapes or ())
        scalar_count = len(self.scalar_dtypes or ())
        attrs = self.pure_attrs
        attr_count = len(attrs.keys())
        case_param_count = tensor_count + scalar_count + attr_count
        if case_param_count != len(op_api_info.params.keys()):
            self.is_valid = False
            self.fail_reason = "PARAM_COUNT_MISMATCH"
            logging.error(
                f"OpApi [{self.api_name}] L2 interface has "
                f"{len(op_api_info.params.keys())} parameters: "
                f"{op_api_info.params.keys()}. "
                f"But testcase configured {case_param_count}: "
                f"Tensor: {tensor_count}, Scalar: {scalar_count}, "
                f"Attribute: {attr_count}"
            )
