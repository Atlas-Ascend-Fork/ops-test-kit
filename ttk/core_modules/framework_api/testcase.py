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
Testcase structure for framework_api tests.
"""

import logging

from ttk.core_modules.common.testcase.api_base import ApiTestcaseBase
from ttk.core_modules.common.testcase.field_types import FIELD_TYPES
from ttk.core_modules.framework_api.api_param import ParamPlan, match_overload
from ttk.utilities import get, shape_stride
from ttk.utilities.container import flatten_by_distribution, flatten_groups

# 通信算子的运行时注入位：值由多卡 worker 在通信域初始化后物化
# （multi_device.py: init_process_group → get_hccl_comm_name / len(device_ids)），
# 非 CSV 声明——签名里它们是必选参数（npu._schemas 如此），校验层须豁免
# "必选非张量参数必须出现在 attributes" 规则，否则 hccl 用例误报
# MISSING_REQUIRED_ATTR。张量计数对账不受影响（str/int 非 tensor-like）。
RUNTIME_INJECTED_PARAMS = {"hcom", "world_size", "ep_hcomm", "tp_hcomm"}


class TestcaseE2e(ApiTestcaseBase):
    """Testcase structure for framework-level API testing (torch/tf).

    Supports nested tensor_view_shapes for TensorList grouping:
      - Flat:  ((2,3),(3,4)) → 2 independent tensors
      - Nested: (((2,3),(3,4)),(5,6)) → TensorList of 2 + single tensor

    Compressed field syntax (after parsing, before validate):
      - ('float32',) broadcasts to all tensors
      - ('float32',('int8','int16')) per-param compression via dist

    Distribution is inferred from tensor_view_shapes nesting:
      dist = (2, 0) means param-0 is TensorList(2), param-1 is single Tensor
    """

    __slots__ = (
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
        "golden_api",
        "golden_data",
        # cross_check 判据下由 profiling 侧临时置 "Promote"（终态回写）；
        # tensors 由 ApiTestcaseBase 以 AoS 槽表属性提供（无独立 slot）
        "golden_mode_override",
        # === 私有 === #
        "_api_info_cache",
        "_multi_device_thread_contexts",
    )

    identity_headers = {
        **ApiTestcaseBase.identity_headers,
        "api_name": (FIELD_TYPES.STRING, None),
    }

    tensor_property_headers = {
        "tensor_view_shapes": (FIELD_TYPES.SHAPELIKE_DYN_NESTED, None),
        "tensor_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None),
    }

    optional_tensor_headers = {
        "tensor_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
        "tensor_storage_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
        "tensor_view_offsets": (FIELD_TYPES.INT_CONTAINER_NESTED, None, ()),
        "tensor_view_strides": (FIELD_TYPES.SHAPE_STRIDE, None, ()),
        "output_tensor_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "inplace_input_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
    }

    attr_headers = {
        "attributes": (FIELD_TYPES.DICT, None, {}),
        **{f"attributes{i}": (FIELD_TYPES.DICT, None, {}) for i in range(1, 10)},
    }

    batch_consistency_headers = {}

    golden_headers = {
        "golden_api": (FIELD_TYPES.STRING, None, ""),
    }

    special_property_headers = {
        **ApiTestcaseBase.special_property_headers,
    }

    option_headers = {
        **ApiTestcaseBase.option_headers,
        # XPU 三方输出本地装载（--dump xpu 回放契约；FREE_EVAL 支持嵌套 TensorList 声明）
        "manual_xpu_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
    }

    complete_headers = {
        **identity_headers,
        **tensor_property_headers,
        **optional_tensor_headers,
        **attr_headers,
        **golden_headers,
        **special_property_headers,
        **batch_consistency_headers,
        **option_headers,
    }

    def __init__(self):
        super().__init__()
        self.attributes = {}
        self.attributes1 = None
        self.attributes2 = None
        self.attributes3 = None
        self.attributes4 = None
        self.attributes5 = None
        self.attributes6 = None
        self.attributes7 = None
        self.attributes8 = None
        self.attributes9 = None
        self.golden_api = None
        self.golden_data = None
        self._api_info_cache = None
        self.device_ids = None
        self._multi_device_thread_contexts = None
        # cross_check 判据下由 profiling 侧临时置 "Promote",令 golden 抬成高精度真值。
        # 必须在 __slots__ 中声明:该类及其基类均无 __dict__,否则赋值抛 AttributeError,
        # 被调用侧 try/except 咽掉后 Promote 静默空转。范式同 TestcaseKernel / TestcaseAclnn。
        self.golden_mode_override = None

    @classmethod
    def get_all_visible_headers(cls):
        return tuple(cls.complete_headers.keys())

    def get_api_info(self):
        """Get cached APIParamInfo for this testcase's api_name."""
        if self._api_info_cache is not None:
            return self._api_info_cache
        if not self.api_name:
            return None
        try:
            from ttk.core_modules.framework_api.info_keeper import (
                FrameworkApiInfoKeeper,
            )

            self._api_info_cache = FrameworkApiInfoKeeper().get(self.api_name)
        except Exception:
            self._api_info_cache = None
        return self._api_info_cache

    def get_param_plan(self):
        """Resolve and cache parameter assembly plan for this testcase."""
        if self._param_plan_cache is not None:
            return self._param_plan_cache
        if not self.api_name:
            return None

        try:
            info = self.get_api_info()
            if info is None:
                logging.warning(f"[{self.testcase_name}] get_api_info returned None for {self.api_name}")
                return None

            out_indices = set(self.output_tensor_indexes or ())
            top_count = len(self.tensor_view_shapes or ())
            if self._is_inplace_tensor_method(self.api_name):
                input_count = top_count
            else:
                input_count = sum(
                    1 for i in range(top_count) if self.tensor_view_shapes[i] is not None and i not in out_indices
                )

            dist = self.tensor_list_dist
            tensor_distribution = [d > 0 for d in dist] if dist else None

            params, oidx = match_overload(
                self.api_name,
                input_count,
                attributes=self.attributes,
                tensor_distribution=tensor_distribution,
                api_info=info,
            )
            if params is None:
                return None

            plan = ParamPlan(
                api_name=self.api_name,
                overload_params=params,
                overload_index=oidx,
                output_tensor_indexes=self.output_tensor_indexes,
                attributes=self.attributes,
            )
            self._param_plan_cache = plan
            return plan
        except Exception:
            # 记划构建属正常编程路径（match 失败走上面的 return None），
            # 此处只剩真实异常——吞掉会让排障只见 PARAM_PLAN_FAILURE
            logging.warning(f"[{self.testcase_name}] get_param_plan raised for {self.api_name}", exc_info=True)
            return None

    # ========== Flat properties ==========

    @property
    def flat_tensor_view_shapes(self):
        """Flatten nested tensor_view_shapes to a flat tuple of shapes.

        (((3,3),(3,2)),(3,5)) -> ((3,3),(3,2),(3,5))
        """
        if not self.tensor_view_shapes:
            return self.tensor_view_shapes
        return flatten_groups(self.tensor_view_shapes)

    @property
    def flat_tensor_dtypes(self):
        """Flatten nested tensor_dtypes to per-tensor dtypes matching flat_tensor_view_shapes."""
        if not self.tensor_dtypes:
            return self.tensor_dtypes
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.tensor_dtypes
        # 条目数相同不代表已经展平: 校验阶段会按 tensor_list_dist 把 dtype 重新嵌套成
        # "每参数一项"(TensorList 参数为元组), 当参数个数恰好等于张量个数时(即每个
        # TensorList 都只含 1 个张量), 长度判断会误判为已展平, 把 ('float32',) 这样的
        # 1 元组原样漏给 numpy.astype(), 触发 "TypeError: Tuple must have size 2"。
        # 故还需确认条目本身不是容器, 否则继续按 dist 展平。
        if len(self.tensor_dtypes) == flat_count and not any(isinstance(d, (tuple, list)) for d in self.tensor_dtypes):
            return self.tensor_dtypes
        dist = self.tensor_list_dist
        if dist and len(self.tensor_dtypes) == len(dist):
            return flatten_by_distribution(self.tensor_dtypes, dist)
        if len(self.tensor_dtypes) == 1:
            val = self.tensor_dtypes[0]
            if isinstance(val, (tuple, list)) and len(val) == 1:
                val = val[0]
            return (val,) * flat_count
        return flatten_groups(self.tensor_dtypes)

    @property
    def flat_tensor_formats(self):
        """Flatten nested tensor_formats to per-tensor format strings."""
        if not self.tensor_formats:
            return self.tensor_formats
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.tensor_formats
        if len(self.tensor_formats) == flat_count:
            return self.tensor_formats
        dist = self.tensor_list_dist
        if dist and len(self.tensor_formats) == len(dist):
            return flatten_by_distribution(self.tensor_formats, dist)
        if len(self.tensor_formats) == 1:
            return (self.tensor_formats[0],) * flat_count
        return flatten_groups(self.tensor_formats)

    @property
    def flat_tensor_storage_shapes(self):
        """Flatten nested tensor_storage_shapes to per-tensor storage shapes."""
        if not self.tensor_storage_shapes:
            return self.tensor_storage_shapes
        flat_count = len(self.flat_tensor_view_shapes)
        if len(self.tensor_storage_shapes) == flat_count:
            return self.tensor_storage_shapes
        return flatten_groups(self.tensor_storage_shapes)

    @property
    def flat_tensor_view_offsets(self):
        """Flatten nested tensor_view_offsets to per-tensor offsets."""
        if not self.tensor_view_offsets:
            return self.tensor_view_offsets
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.tensor_view_offsets
        if len(self.tensor_view_offsets) == flat_count:
            return self.tensor_view_offsets
        dist = self.tensor_list_dist
        if dist and len(self.tensor_view_offsets) == len(dist):
            return flatten_by_distribution(self.tensor_view_offsets, dist)
        if len(self.tensor_view_offsets) == 1:
            return (self.tensor_view_offsets[0],) * flat_count
        return flatten_groups(self.tensor_view_offsets)

    @property
    def flat_tensor_view_strides(self):
        """Flatten nested tensor_view_strides to per-tensor strides."""
        if not self.tensor_view_strides:
            return self.tensor_view_strides
        flat_count = len(self.flat_tensor_view_shapes)
        if len(self.tensor_view_strides) == flat_count:
            return self.tensor_view_strides
        return flatten_groups(self.tensor_view_strides)

    # ========== Per-flat-index accessors ==========

    def flat_storage_shape(self, idx: int):
        """Get storage shape for flat tensor at idx. Falls back to view shape."""
        flat_shapes = self.flat_tensor_view_shapes
        s_shapes = self.flat_tensor_storage_shapes
        if not s_shapes:
            return flat_shapes[idx]
        return get(s_shapes, idx, out_of_range=flat_shapes[idx])

    def flat_view_stride(self, idx: int):
        """Get view stride for flat tensor at idx. Auto-compute from shape if not specified."""
        strides = self.flat_tensor_view_strides or ()
        s = get(strides, idx, out_of_range=())
        if not s:
            s = shape_stride(self.flat_tensor_view_shapes[idx])
        return s

    def flat_view_offset(self, idx: int):
        """Get view offset for flat tensor at idx. Default 0."""
        offsets = self.flat_tensor_view_offsets or ()
        return get(offsets, idx, out_of_range=0)

    # ========== Legacy per-top-level accessors (kept for backward compat) ==========

    def storage_shape(self, idx: int):
        v_shapes = self.tensor_view_shapes
        s_shapes = self.tensor_storage_shapes
        return get(s_shapes, idx, out_of_range=v_shapes[idx])

    def view_stride(self, idx: int):
        strides = self.tensor_view_strides or ()
        s = get(strides, idx, out_of_range=())
        if not s:
            s = shape_stride(self.tensor_view_shapes[idx])
        return s

    def view_offset(self, idx: int):
        offsets = self.tensor_view_offsets or ()
        return get(offsets, idx, out_of_range=0)

    # ========== actual_input_data_ranges ==========

    @property
    def actual_input_data_ranges(self):
        return self.input_data_ranges

    @property
    def pure_output_indexes(self):
        base = super().pure_output_indexes
        if self._is_inplace_tensor_method(self.api_name) and 0 in base:
            return sorted(set(base) - {0})
        return base

    @property
    def flat_output_dtypes(self):
        """Flat 输出 dtype——逐输出张量一项，TensorList 槽展开。

        与 result_to_numpy 的输出顺序对齐（按 output_tensor_indexes 声明序），
        供 unpack_4bit_outputs 等按输出位消费（口径对齐 TestcaseAclnn）。
        """
        flat_dtypes = self.flat_tensor_dtypes
        if not flat_dtypes or not self.output_tensor_indexes:
            return ()
        dist = self.tensor_list_dist
        result = []
        for idx in self.output_tensor_indexes:
            if not 0 <= idx < len(dist):
                continue
            flat_idx = self.flat_index_of(idx)
            count = dist[idx] if dist[idx] > 0 else 1
            result.extend(flat_dtypes[flat_idx : flat_idx + count])
        return tuple(result)

    @property
    def flat_output_view_shapes(self):
        """Flat 输出 view shape——逐输出张量一项，TensorList 槽展开。

        口径对齐 flat_output_dtypes（按 output_tensor_indexes 声明序），
        供 golden 静态 replay 装载（shape 取 CSV 声明）；声明 -1/None 的
        动态槽输出 shape 运行前未知，由调用方延迟到执行后装载。
        """
        flat_shapes = self.flat_tensor_view_shapes
        if not flat_shapes or not self.output_tensor_indexes:
            return ()
        dist = self.tensor_list_dist
        result = []
        for idx in self.output_tensor_indexes:
            if not 0 <= idx < len(dist):
                continue
            flat_idx = self.flat_index_of(idx)
            count = dist[idx] if dist[idx] > 0 else 1
            result.extend(flat_shapes[flat_idx : flat_idx + count])
        return tuple(result)

    # ========== npu_preprocess absorption ==========

    def absorb_preprocess(self, result, plan, args=None, kwargs=None):
        """E2E 落地：dict → 校验+原地回填 args/kwargs；None 且声明 -1 →
        隐式通道核验（槽位→args 补丁待后续批次）。"""
        from ..common.invoke import apply_npu_preprocess_result

        if result is None:
            if not self.dyn_input_slot_indexes:
                return None
            self._require_dynamic_slots_materialized()
            return None
        apply_npu_preprocess_result(self, plan, args, kwargs, result)
        return result

    # ========== Validate ==========

    def validate(self):
        super().validate()
        self._strip_backward_suffix()  # 后缀剥离在签名对账生态消费前
        if not self.api_name:
            self.is_valid = False
            self.fail_reason = "api_name is empty"
            return
        if not self.tensor_view_shapes:
            info = self._try_get_api_info_for_factory()
            if info is None:
                return
        if self.tensor_view_shapes and not self.tensor_dtypes:
            self.is_valid = False
            self.fail_reason = "tensor_dtypes is empty"
            return
        # auto-fill 须先于 normalize：normalize 会按 output_tensor_indexes 计算并
        # 缓存 output_dist，inplace 方法补 (0,) 若发生在其后，odist 类归一化被跳过
        # 且缓存过期无失效钩子（plain 属性不走六列 setter 的同步路径）。
        self._auto_fill_inplace_tensor_method()
        self._normalize_compressed_fields()
        self._parse_device_ids_field()
        self._check_top_level_counts()
        self._check_dyn_shape_slots()
        self._check_tensor_configuration()
        self._check_output_configuration()
        self._check_manual_xpu_binaries()
        self._generate_batch_consistency_id()

    def _try_get_api_info_for_factory(self):
        """Check if API is a factory function (no required input tensors)."""
        info = self.get_api_info()
        if info is None:
            self.is_valid = False
            self.fail_reason = "tensor_view_shapes is empty"
            return None
        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like] if not p.is_optional) for ov in info.overloads
        )
        if required_min == 0:
            return info
        self.is_valid = False
        self.fail_reason = "tensor_view_shapes is empty"
        return None

    @staticmethod
    def _is_inplace_tensor_method(api_name):
        from ttk.core_modules.common.backend import is_inplace_tensor_method

        return is_inplace_tensor_method(api_name)

    def _auto_fill_inplace_tensor_method(self):
        if not self._is_inplace_tensor_method(self.api_name):
            return
        if not self.output_tensor_indexes:
            self.output_tensor_indexes = (0,)

    def _check_tensor_configuration(self):
        """Validate tensor parameters match API definition in count and type."""
        if not self.is_valid:
            return
        info = self.get_api_info()
        if info is None:
            self.is_valid = False
            self.fail_reason = "API_PARSE_FAIL"
            return
        if self._check_input_count_exceeded(info):
            return
        if self._is_factory_api(info):
            return
        if self._check_all_tensors_output(info):
            return
        self._check_overload_match(info)

    def _check_input_count_exceeded(self, info):
        """Fail if non-output tensor count exceeds API's max input tensor parameters.

        Overloads containing a VAR_POSITIONAL (*args) tensor parameter are exempt
        because they accept an unbounded number of tensors.
        """
        out_indices = set(self.output_tensor_indexes or ())
        top_count = len(self.tensor_view_shapes or ())
        # Match get_param_plan logic: filter None tensors (placeholders for optional
        # params not used in this case) in addition to output tensors.
        input_count = sum(
            1 for i in range(top_count) if i not in out_indices and self.tensor_view_shapes[i] is not None
        )
        # If any overload has a VAR_POSITIONAL tensor param, it can accept
        # any number of tensors — skip the count check entirely.
        has_var_pos = any(
            any(getattr(p, "is_var_positional", False) for p in ov.params if p.is_tensor_like and p.name != "out")
            for ov in info.overloads
        )
        if has_var_pos:
            return False
        max_input = max(sum(1 for p in ov.params if p.is_tensor_like and p.name != "out") for ov in info.overloads)
        if input_count > max_input:
            self.is_valid = False
            self.fail_reason = "INPUT_COUNT_EXCEEDED"
            logging.error(
                f"[{self.testcase_name}] API [{self.api_name}] has at most {max_input} input tensor "
                f"parameters (excluding out), but testcase configured {input_count} "
                f"input tensor(s) (excluding {len(out_indices)} output tensor(s)). "
                f"(source: {info.source})"
            )
            return True
        return False

    def _is_factory_api(self, info):
        """Return True if API requires no input tensors (factory function like torch.zeros)."""
        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )
        return required_min == 0

    def _check_all_tensors_output(self, info):
        """Fail if all tensors are marked as output with none left as input."""
        out_indices = set(self.output_tensor_indexes or ())
        top_count = len(self.tensor_view_shapes)
        if self._is_inplace_tensor_method(self.api_name):
            input_count = top_count
        else:
            input_count = sum(1 for i in range(top_count) if i not in out_indices)
        if input_count > 0:
            return False
        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )
        self.is_valid = False
        self.fail_reason = "ALL_TENSORS_MARKED_OUTPUT"
        logging.error(
            f"[{self.testcase_name}] API [{self.api_name}] requires at least {required_min} input tensor "
            f"parameters (excluding out), but all {top_count} tensor(s) are "
            f"marked as output (output_tensor_indexes={sorted(out_indices)}). "
            f"(source: {info.source})"
        )
        return True

    @staticmethod
    def _classify_input_types(input_shapes):
        """Classify each input shape as nested (TensorList) or None placeholder."""
        nested_flags = []
        has_none = []
        for element in input_shapes:
            if element is None:
                nested_flags.append(False)
                has_none.append(True)
            else:
                is_nested = (
                    isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list))
                )
                nested_flags.append(is_nested)
                has_none.append(False)
        return nested_flags, has_none

    def _count_scalar_attrs_for_tensor_params(self, info):
        """Count tensor-like params satisfied by scalar values in attributes."""
        if not self.attributes:
            return 0
        attr_keys = set(self.attributes.keys())
        count = 0
        for ov in info.overloads:
            ov_count = 0
            for p in ov.layout.input_params:
                if p.name in attr_keys and p.name != "self":
                    ov_count += 1
            count = max(count, ov_count)
        return count

    def _align_shapes_to_overload(self, input_params, out_indices):
        """按给定重载的张量参数表对位 top-level shapes。

        A top-level None is an omitted optional Tensor, except for a
        TensorList parameter where it represents an omitted list while
        still occupying that parameter's position.
        """
        input_shapes = []
        skip_flags = []
        tensor_pos = 0
        for i, shape in enumerate(self.tensor_view_shapes):
            if i in out_indices:
                continue
            param = input_params[tensor_pos] if tensor_pos < len(input_params) else None
            if shape is None and (param is None or not param.is_tensor_list):
                continue
            input_shapes.append(shape)
            skip_flags.append(shape is None)
            tensor_pos += 1
        return input_shapes, skip_flags

    def _check_overload_match(self, info):
        """Fail if input tensors don't match any overload in count/type/nesting.

        None 槽的对位（省略可选 Tensor vs TensorList 占位）与重载相关：
        逐重载对位并尝试匹配，任一重载匹配即通过（与 match_overload 的
        匹配语义一致）。scalar 补偿与报错口径沿用首个重载的对位结果。
        """
        out_indices = set(self.output_tensor_indexes or ())
        if self._is_inplace_tensor_method(self.api_name):
            input_shapes = list(self.tensor_view_shapes)
            skip_flags = [shape is None for shape in input_shapes]
            input_count = len(input_shapes)
            nested_flags, has_none = self._classify_input_types(input_shapes)
            matched, _, oidx = info.match_overload(input_count, nested_flags, skip_flags)
            if matched:
                self._check_required_attrs(info, oidx)
                return
        else:
            for ov in info.overloads:
                input_shapes, skip_flags = self._align_shapes_to_overload(ov.layout.input_params, out_indices)
                nested_flags, _has_none = self._classify_input_types(input_shapes)
                matched, _, oidx = info.match_overload(len(input_shapes), nested_flags, skip_flags)
                if matched:
                    self._check_required_attrs(info, oidx)
                    return
            # 全部重载未匹配：报错口径沿用首个重载的对位结果
            input_shapes, skip_flags = self._align_shapes_to_overload(
                info.overloads[0].layout.input_params, out_indices
            )
            input_count = len(input_shapes)
            nested_flags, has_none = self._classify_input_types(input_shapes)

        scalar_attr_count = self._count_scalar_attrs_for_tensor_params(info)
        if scalar_attr_count > 0:
            effective_count = input_count + scalar_attr_count
            matched, _, oidx = info.match_overload(effective_count, None, None)
            if matched:
                self._check_required_attrs(info, oidx)
                return

        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )
        required_min -= scalar_attr_count
        count_matched = info.match_overload(input_count, None, None)
        if count_matched[0]:
            self.is_valid = False
            self.fail_reason = "PARAM_TYPE_MISMATCH"
            logging.error(
                f"API [{self.api_name}] input tensor count matches an overload, "
                f"but type (Tensor/TensorList) does not. "
                f"nested={nested_flags}. "
                f"(source: {info.source})"
            )
        elif input_count > info.tensor_count:
            self.is_valid = False
            self.fail_reason = "TENSOR_COUNT_MISMATCH"
            logging.error(
                f"API [{self.api_name}] has at most {info.tensor_count} tensor parameters "
                f"in any overload, but testcase configured {input_count} input tensors "
                f"(excluding {len(out_indices)} output tensors). "
                f"(source: {info.source})"
            )
        else:
            self.is_valid = False
            self.fail_reason = "TENSOR_COUNT_MISMATCH"
            logging.error(
                f"API [{self.api_name}] requires at least {required_min} input tensor "
                f"parameters (excluding out), but testcase configured {input_count}. "
                f"(source: {info.source})"
            )

    def _check_required_attrs(self, info, oidx):
        """Fail if matched overload has required non-tensor params missing from attributes.

        If the matched overload has missing attrs but another overload also matches
        the tensor count without missing attrs, skip the check (build_args will try all).
        """
        ov = info.overloads[oidx]
        attrs = set(self.attributes.keys()) if self.attributes else set()
        missing = [
            p.name
            for p in ov.params
            if not p.is_tensor_like
            and not p.is_keyword_only
            and p.name != "out"
            and not p.is_optional
            and p.name not in attrs
            and p.name not in RUNTIME_INJECTED_PARAMS
        ]
        if not missing:
            return

        out_indices = set(self.output_tensor_indexes or ())
        top_count = len(self.tensor_view_shapes or ())
        input_count = (
            top_count
            if self._is_inplace_tensor_method(self.api_name)
            else sum(1 for i in range(top_count) if i not in out_indices)
        )

        for alt_oidx, alt_ov in enumerate(info.overloads):
            if alt_oidx == oidx:
                continue
            alt_tensors = [p for p in alt_ov.params if p.is_tensor_like and p.name != "out"]
            alt_req = sum(1 for p in alt_tensors if not p.is_optional)
            alt_total = len(alt_tensors)
            if not (alt_req <= input_count <= alt_total):
                continue
            alt_missing = [
                p.name
                for p in alt_ov.params
                if not p.is_tensor_like
                and not p.is_keyword_only
                and p.name != "out"
                and not p.is_optional
                and p.name not in attrs
                and p.name not in RUNTIME_INJECTED_PARAMS
            ]
            if not alt_missing:
                return

        self.is_valid = False
        self.fail_reason = "MISSING_REQUIRED_ATTR"
        logging.error(
            f"[{self.testcase_name}] API [{self.api_name}] overload[{oidx}] requires "
            f"non-tensor attribute(s) {missing} but attributes only have "
            f"{sorted(attrs)}. (source: {info.source})"
        )

    def _check_output_configuration(self):
        """Validate output_tensor_indexes against API's out parameter definition.

        Checks:
        1. Indexes within top-level range
        2. If API has a required out parameter, testcase must provide output tensors
        3. If API has a Tensor[] out, output count must match out_expected_count
        """
        if not self.is_valid:
            return
        info = self.get_api_info()
        if not self.output_tensor_indexes:
            if info is not None:
                any_out_required = any(ov.layout.is_out_required for ov in info.overloads)
                if any_out_required:
                    self.is_valid = False
                    self.fail_reason = "MISSING_REQUIRED_OUTPUT"
                    logging.error(
                        f"[{self.testcase_name}] API [{self.api_name}] has overloads "
                        f"with required 'out' parameter, but testcase provides no "
                        f"output_tensor_indexes. (source: {info.source})"
                    )
            return
        top_count = len(self.tensor_view_shapes)
        out_of_range = [i for i in self.output_tensor_indexes if i < 0 or i >= top_count]
        if out_of_range:
            self.is_valid = False
            self.fail_reason = "OUTPUT_INDEX_INVALID"
            logging.error(f"output_tensor_indexes {out_of_range} out of range [0, {top_count})")
            return
        if info is None:
            return
        out_count = len(self.output_tensor_indexes)
        matching_overloads = []
        for oidx, ov in enumerate(info.overloads):
            layout = ov.layout
            if layout.out_param is None:
                if out_count == 0:
                    matching_overloads.append(oidx)
            elif layout.is_out_tensor_list:
                if layout.out_expected_count > 0:
                    # Exact validation: we know the expected out tensor count
                    if (
                        layout.is_out_required
                        and out_count != layout.out_expected_count
                        or not layout.is_out_required
                        and out_count not in (0, layout.out_expected_count)
                    ):
                        continue
                # Loose validation: TensorList but unknown count (e.g. TypeError path)
                elif layout.is_out_required and out_count == 0:
                    continue
                matching_overloads.append(oidx)
            else:
                if layout.is_out_required and out_count != 1 or not layout.is_out_required and out_count > 1:
                    continue
                matching_overloads.append(oidx)
        if not matching_overloads:
            any_out_required = any(ov.layout.is_out_required for ov in info.overloads)
            any_tensor_list_out = any(ov.layout.is_out_tensor_list for ov in info.overloads)
            if any_out_required and any_tensor_list_out:
                expected = {ov.layout.out_expected_count for ov in info.overloads if ov.layout.is_out_required}
                self.is_valid = False
                self.fail_reason = "OUTPUT_COUNT_MISMATCH"
                logging.error(
                    f"[{self.testcase_name}] API [{self.api_name}] requires exactly "
                    f"{expected} output tensor(s) for Tensor[] 'out', but testcase "
                    f"provides {out_count}. (source: {info.source})"
                )
            elif any_out_required:
                self.is_valid = False
                self.fail_reason = "OUTPUT_COUNT_MISMATCH"
                logging.error(
                    f"[{self.testcase_name}] API [{self.api_name}] requires exactly "
                    f"1 output tensor, but testcase provides {out_count}. "
                    f"(source: {info.source})"
                )

    def _check_top_level_counts(self):
        """Validate that all fields have matching top-level count after normalization."""
        if not self.is_valid:
            return
        top_count = len(self.tensor_view_shapes)
        if len(self.tensor_dtypes) != top_count:
            self.is_valid = False
            self.fail_reason = "DTYPES_COUNT_MISMATCH"
            logging.error(
                f"[{self.testcase_name}] tensor_dtypes top-level count ({len(self.tensor_dtypes)}) "
                f"!= tensor_view_shapes count ({top_count})"
            )
            return
        if self.tensor_formats and len(self.tensor_formats) != top_count:
            self.is_valid = False
            self.fail_reason = "FORMATS_COUNT_MISMATCH"
            logging.error(
                f"[{self.testcase_name}] tensor_formats top-level count ({len(self.tensor_formats)}) "
                f"!= tensor_view_shapes count ({top_count})"
            )
            return
        if self.tensor_storage_shapes and len(self.tensor_storage_shapes) != top_count:
            self.is_valid = False
            self.fail_reason = "STORAGE_SHAPES_COUNT_MISMATCH"
            logging.error(
                f"[{self.testcase_name}] tensor_storage_shapes top-level count ({len(self.tensor_storage_shapes)}) "
                f"!= tensor_view_shapes count ({top_count})"
            )
