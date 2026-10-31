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
        "golden_perf",
        "golden_source",
        # === 私有 === #
        "_api_info_cache",
        "_multi_device_thread_contexts",
    )

    describe_pathway = "e2e"

    identity_headers = {
        **ApiTestcaseBase.identity_headers,
        "api_name": (FIELD_TYPES.STRING, None),
    }

    tensor_property_headers = {
        # 缺省 () 使参数化用例（无 shapes/dtypes 列，由 spec.describe 组装）
        # 可过 factory 必填字段解析；空值行仍按 validate 判无效（输入张量
        # 全部可选的算子短路除外）。CSV 显式声明列时行为不变。
        "tensor_view_shapes": (FIELD_TYPES.SHAPELIKE_DYN_NESTED, None, ()),
        "tensor_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
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
        self.device_ids = None
        # cross_check 判据下由 profiling 侧临时置 "Promote",令 golden 抬成高精度真值。
        # 必须在 __slots__ 中声明:该类及其基类均无 __dict__,否则赋值抛 AttributeError,
        # 被调用侧 try/except 咽掉后 Promote 静默空转。范式同 TestcaseKernel / TestcaseAclnn。
        self.golden_mode_override = None
        self.golden_perf = None
        self.golden_source = None
        # === 私有 === #
        self._api_info_cache = None
        self._multi_device_thread_contexts = None

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
        except Exception as error:
            logging.error(
                "[%s] API_PARSE_FAIL: cannot inspect %s: %s: %s",
                self.testcase_name,
                self.api_name,
                type(error).__name__,
                error,
            )
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
        val = get(s_shapes, idx, out_of_range=flat_shapes[idx])
        if val is None:
            return flat_shapes[idx]
        return val

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

        None 位回落 CSV 声明 dtype：onnx 域输出槽为 None-shape
        （shape 由 atc 推断，CSV 槽仅声明 dtype），槽表对 None-shape 槽不物化张量
        对象 → 槽视图取 None；而 worker._needs_golden_promote 等判据解析以声明
        输出 dtype 为锚点，声明通道失联会使 golden Promote 静默跳过。声明列经
        `_flat_decl_by_dist`（与槽表物化 dtypes 列同源）按 flat 位对齐回填。
        """
        flat_dtypes = self.flat_tensor_dtypes
        if not flat_dtypes or not self.output_tensor_indexes:
            return ()
        dist = self.tensor_list_dist
        # 常态（无 None 位）零开销：仅当槽视图出现 None 位时构造声明回填列。
        declaration = self._flat_decl_by_dist(self._tensor_dtypes, dist) if any(d is None for d in flat_dtypes) else ()
        result = []
        for idx in self.output_tensor_indexes:
            if not 0 <= idx < len(dist):
                continue
            flat_idx = self.flat_index_of(idx)
            count = dist[idx] if dist[idx] > 0 else 1
            for offset, slot_dtype in enumerate(flat_dtypes[flat_idx : flat_idx + count]):
                position = flat_idx + offset
                result.append(
                    declaration[position] if slot_dtype is None and position < len(declaration) else slot_dtype
                )
        return tuple(result)

    @property
    def flat_output_view_shapes(self):
        """Flat 输出 view shape——逐输出张量一项，TensorList 槽展开。

        口径对齐 flat_output_dtypes（按 output_tensor_indexes 声明序），
        供 golden 静态 replay 装载（shape 取 CSV 声明）；输出槽不可能是
        动态槽（-1 输出被 DYN_SHAPE_ON_OUTPUT_SLOT 校验拒绝），仅 optional
        输出声明 None（对应 `*_none` 槽位）。装载失败仍由调用方延迟到执行
        后按实际返回 shape 重装载。
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
        self._validate_csv_inputs()
        if not self.is_valid:
            return
        self._validate_api_contract()

    def _validate_csv_inputs(self):
        """CSV 输入侧：语法归一、缺失回填、完整性、补位、压缩展开、列数一致性。"""
        self._strip_backward_suffix()  # 后缀剥离在签名对账生态消费前
        self._apply_describe_shapes()
        self._check_declarations()
        if not self.is_valid:
            return
        # auto-fill 须先于 normalize：normalize 会按 output_tensor_indexes 计算并
        # 缓存 output_dist，inplace 方法补 (0,) 若发生在其后，odist 类归一化被跳过
        # 且缓存过期无失效钩子（plain 属性不走六列 setter 的同步路径）。
        self._fill_inplace_output_slot()
        self._expand_compressed_fields()
        self._check_field_counts()
        self._check_dyn_shape_slots()
        self._check_manual_xpu_binaries()

    def _validate_api_contract(self):
        """API 签名侧：重载匹配、必选属性与 out 参数对账。"""
        self._check_api_signature()
        self._check_output_slots()

    def _check_declarations(self):
        """声明基础校验：api_name 非空；无 shapes 时须为输入张量全部可选的算子；
        有 shapes 须有 dtypes。"""
        if not self.api_name:
            self.is_valid = False
            self.fail_reason = "api_name is empty"
            return
        if not self.tensor_view_shapes:
            info = self._check_missing_shapes()
            if info is None:
                return
        if self.tensor_view_shapes and not self.tensor_dtypes:
            self.is_valid = False
            self.fail_reason = "tensor_dtypes is empty"

    def _check_missing_shapes(self):
        """shapes 为空时校验 API 类型：输入张量全部可选的算子（如 torch.zeros）
        放行并返回签名，其余判无效。解析失败报 API_PARSE_FAIL。"""
        info = self.get_api_info()
        if info is None:
            self._report_api_parse_failure()
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

    def _fill_inplace_output_slot(self):
        if not self._is_inplace_tensor_method(self.api_name):
            return
        if not self.output_tensor_indexes:
            self.output_tensor_indexes = (0,)

    def _check_api_signature(self):
        """张量参数与 API 签名对账：框架门禁 → 张量声明对账（全输出拒绝/重载匹配/数量超界）。"""
        if not self.is_valid:
            return
        info = self.get_api_info()
        if info is None:
            self._report_api_parse_failure()
            self.is_valid = False
            self.fail_reason = "API_PARSE_FAIL"
            return
        self._check_framework_gates(info)
        if not self.is_valid:
            return
        self._check_tensor_declaration(info)

    def _check_framework_gates(self, info):
        """框架专属门禁：onnx 域的未知属性与动态 shape 声明在校验期拒绝。"""
        from ..common.backend import detect_framework

        if detect_framework(self.api_name) != "onnx":
            return
        self._check_onnx_unknown_attrs(info)
        self._check_onnx_dynamic_input(info)

    def _report_api_parse_failure(self):
        from .info_keeper import FrameworkApiInfoKeeper

        reason = FrameworkApiInfoKeeper().get_failure_reason(self.api_name) or "API parameter inspection failed"
        logging.error("[%s] API_PARSE_FAIL: cannot parse %s: %s", self.testcase_name, self.api_name, reason)

    def _check_onnx_unknown_attrs(self, info):
        """onnx 域：attributes 列出现 schema 未声明的属性名时显式报错。

        onnx 域 attributes 列即节点属性（kw-only ParamInfo，schema 单源）；
        build_args 会把未声明属性滤入 extra_attrs 丢弃（api_param._build_args），
        model_builder._normalize_attrs 的 unknown 检查只看得到过滤后的 kwargs，
        CSV 拼写错误因此静默假 PASS——违反「CSV 属性拼写错误不静默落
        extra_attrs」。此处校验期对 schema 直查。
        torch/tf 域 extra_attrs 有插件 kwargs 消费方（golden 的
        _call_plugin_with_plan 经 extra 透传），不做同口径限制。

        Returns:
            True if the testcase was invalidated (unknown attributes found).
        """
        if not self.attributes:
            return False
        declared = {p.name for ov in info.overloads for p in ov.params if p.is_keyword_only}
        unknown = sorted(set(self.attributes) - declared)
        if not unknown:
            return False
        self.is_valid = False
        self.fail_reason = "UNKNOWN_ATTRIBUTE"
        logging.error(
            f"[{self.testcase_name}] API [{self.api_name}] has unknown attribute(s) {unknown} "
            f"in testcase attributes; declared attribute(s) are {sorted(declared)}. "
            f"onnx node attributes are schema-declared — misspelled names fail "
            f"instead of being silently ignored. (source: {info.source})"
        )
        return True

    def _check_onnx_dynamic_input(self, info):
        """onnx 域：输入槽含 dim<0（-1 动态标记）时校验期显式拒绝。

        动态 shape 是设计非目标（om 恒为静态图，仅支持静态 shape dim ≥ 0）。
        分层：-2/TensorList 内 -1/输出位与原地位的 -1 已由通用校验层
        （api_base._check_dyn_shape_slots）全域拒绝；普通输入位的 -1 在 torch
        域经 npu_preprocess hook 物化（worker._check_preconditions 门），而
        onnx 域不存在该通道——不在此拦截则落到 worker 门报
        DYN_SHAPE_REQUIRES_NPU_PREPROCESS，措辞指向 onnx 域不存在的机制。
        与 -d 模式的 ONNX_DYNAMIC_GRAPH_UNSUPPORTED 同体系（设计 §9）。

        Returns:
            True if the testcase was invalidated (dynamic dims found).
        """
        dyn_slots = tuple(
            idx
            for idx, shape in enumerate(self.flat_tensor_view_shapes or ())
            if shape is not None and any(dim < 0 for dim in shape)
        )
        if not dyn_slots:
            return False
        self.is_valid = False
        self.fail_reason = "ONNX_DYNAMIC_INPUT_UNSUPPORTED"
        logging.error(
            f"[{self.testcase_name}] API [{self.api_name}] declares dynamic dim(s) (<0) in "
            f"tensor_view_shapes flat slot(s) {dyn_slots}; the onnx domain supports static "
            "shapes only (dim >= 0, om is always a static graph). Remove -1/-2 slots, "
            "or use kernel/aclnn modes for dynamic-shape coverage."
        )
        return True

    @staticmethod
    def _max_input_tensors(info):
        """所有重载中输入张量参数（非 out，含可选）数量的最大值。"""
        return max(sum(1 for p in ov.params if p.is_tensor_like and p.name != "out") for ov in info.overloads)

    @staticmethod
    def _required_input_tensor_count(info):
        """所有重载中必选输入张量参数（非 out）数量的最小值。"""
        return min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )

    @staticmethod
    def _has_no_required_input(info) -> bool:
        """输入张量全部可选：所有重载均无必选输入张量（如 torch.zeros，out 为可选）。"""
        return TestcaseE2e._required_input_tensor_count(info) == 0

    def _check_tensor_declaration(self, info):
        """张量声明对账：全输出拒绝 + 重载匹配；输入张量全部可选的算子
        （如 torch.zeros）没有必选输入可对账，直接豁免。"""
        if self._has_no_required_input(info):
            return
        if self._check_all_tensors_output(info):
            return
        self._check_overload_match(info)

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
        required_min = self._required_input_tensor_count(info)
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
        """逐位标出是否 TensorList 嵌套；None 槽标 False（无形状信息）。"""
        return [
            False
            if element is None
            else isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list))
            for element in input_shapes
        ]

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

    def _try_match_inputs(self, input_shapes, skip_flags, info) -> bool:
        """按该对位结果试匹配：命中则查必选属性并返回 True。"""
        nested_flags = self._classify_input_types(input_shapes)
        matched, _, oidx = info.match_overload(len(input_shapes), nested_flags, skip_flags)
        if matched:
            self._check_required_attrs(info, oidx)
        return matched

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
        """输入张量声明与重载对账：逐重载匹配，任一命中即通过并查必选属性；
        全失配按三档报错（类型/嵌套不对、超数量上限、低于必选下限）。

        四步：① inplace 方法原样对位 ② 普通函数逐重载对位（输出位剔除、
        None 槽按该重载布局判"省略可选 vs TensorList 占位"）③ attributes
        标量顶替张量位后合并数量再试 ④ 失配报错（口径取首个重载对位结果）。
        """
        out_indices = set(self.output_tensor_indexes or ())
        # ① inplace Tensor 方法：shapes 原样即输入位（self 在第 0 槽）
        if self._is_inplace_tensor_method(self.api_name):
            input_shapes = list(self.tensor_view_shapes)
            skip_flags = [shape is None for shape in input_shapes]
            if self._try_match_inputs(input_shapes, skip_flags, info):
                return
            input_count = len(input_shapes)
        else:
            # ② 逐重载对位：输出位剔除、None 槽按该重载的 Tensor 位置对齐
            first_aligned = None
            for ov in info.overloads:
                input_shapes, skip_flags = self._align_shapes_to_overload(ov.layout.input_params, out_indices)
                if first_aligned is None:
                    first_aligned = input_shapes  # 全部失配时按首个重载对位结果报错
                if self._try_match_inputs(input_shapes, skip_flags, info):
                    return
            input_shapes = first_aligned
            input_count = len(input_shapes)
        nested_flags = self._classify_input_types(input_shapes)

        # ③ attributes 标量顶替张量位：合并数量再试一次
        scalar_attr_count = self._count_scalar_attrs_for_tensor_params(info)
        if scalar_attr_count > 0:
            effective_count = input_count + scalar_attr_count
            matched, _, oidx = info.match_overload(effective_count, None, None)
            if matched:
                self._check_required_attrs(info, oidx)
                return

        # ④ 全失配：三档报错
        required_min = self._required_input_tensor_count(info)
        required_min -= scalar_attr_count
        count_matched = info.match_overload(input_count, None, None)
        if count_matched[0]:
            # 数量对得上某重载，但 Tensor/TensorList 嵌套不符
            self.is_valid = False
            self.fail_reason = "PARAM_TYPE_MISMATCH"
            logging.error(
                f"API [{self.api_name}] input tensor count matches an overload, "
                f"but type (Tensor/TensorList) does not. "
                f"nested={nested_flags}. "
                f"(source: {info.source})"
            )
        elif input_count > self._max_input_tensors(info):
            # 任何重载都容不下这个数量（VAR_POSITIONAL 重载在前序循环恒匹配，
            # 到不了这里）：按跨重载最大输入数报超界
            self.is_valid = False
            self.fail_reason = "INPUT_COUNT_EXCEEDED"
            logging.error(
                f"API [{self.api_name}] has at most {self._max_input_tensors(info)} input tensor "
                f"parameters (excluding out), but testcase configured {input_count} "
                f"input tensor(s) (excluding {len(out_indices)} output tensor(s)). "
                f"(source: {info.source})"
            )
        else:
            # 数量低于必选下限（含 scalar 补偿后的折算）
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

    def _check_output_slots(self):
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

    def _check_field_counts(self):
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
