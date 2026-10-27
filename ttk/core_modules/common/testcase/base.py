#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
testcase base class
"""

__all__ = ["TestcaseBase"]


import logging
from abc import ABCMeta
from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Optional, Tuple

from .field_parser import (
    int_container,
    process_bool,
    process_dict,
    process_dynamic_inferable_shapelike,
    process_dynamic_shapelike,
    process_eval,
    process_int,
    process_string,
    scalar_nested,
    shape_stride,
    shapelike_dyn_nested,
    shapelike_float,
    shapelike_float_nested,
    shapelike_float_signed,
    shapelike_float_signed_nested,
    shapelike_stc,
    shapelike_stc_ex,
    shapelike_stc_ex_nested,
    shapelike_stc_nested,
    string_container,
)
from .field_types import FIELD_TYPES

_NO_PAD = object()


type_processing_func: Dict[FIELD_TYPES, Callable] = {
    FIELD_TYPES.STRING: process_string,
    FIELD_TYPES.SHAPELIKE_DYN: process_dynamic_shapelike,
    FIELD_TYPES.SHAPELIKE_DYN_EX: process_dynamic_inferable_shapelike,
    FIELD_TYPES.SHAPELIKE_STC: shapelike_stc,
    FIELD_TYPES.SHAPELIKE_STC_EX: shapelike_stc_ex,
    FIELD_TYPES.SHAPELIKE_FLOAT: shapelike_float,
    FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED: shapelike_float_signed,
    FIELD_TYPES.STRING_CONTAINER: string_container,
    FIELD_TYPES.INT_CONTAINER: int_container,
    FIELD_TYPES.INT: process_int,
    FIELD_TYPES.BOOL: process_bool,
    FIELD_TYPES.DICT: process_dict,
    FIELD_TYPES.FREE_EVAL: process_eval,
    FIELD_TYPES.SHAPE_STRIDE: shape_stride,
    FIELD_TYPES.SHAPELIKE_STC_NESTED: shapelike_stc_nested,
    FIELD_TYPES.SHAPELIKE_STC_EX_NESTED: shapelike_stc_ex_nested,
    FIELD_TYPES.STRING_SCALAR_NESTED: partial(scalar_nested, allowed_type=str),
    FIELD_TYPES.FLOAT_SCALAR_NESTED: partial(scalar_nested, allowed_type=(int, float)),
    FIELD_TYPES.INT_CONTAINER_NESTED: partial(scalar_nested, allowed_type=int),
    FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED: shapelike_float_signed_nested,
    FIELD_TYPES.SHAPELIKE_FLOAT_NESTED: shapelike_float_nested,
    FIELD_TYPES.SHAPELIKE_DYN_NESTED: shapelike_dyn_nested,
}


class TestcaseBase(metaclass=ABCMeta):  # noqa: B024
    __slots__ = (
        # === testcase valid configurations === #
        "testcase_name",
        "input_data_ranges",
        "is_enabled",
        "priority",
        # precision
        "precision_tolerances",
        "absolute_precision",
        # testcase remark
        "remark",
        # soc series (short soc version) to specify to run or disable.
        "soc_series",
        # CSV 可选（设备指定）
        "device_ids",
        # CSV 可选（batch 一致性，Level=3）
        "batch_axis",
        "batch_slice_info",
        "batch_seed",
        "batch_consistency_id",
        # Temp
        "original_line",
        "original_dict",
        # === Runtime parameters below === #
        "device_id",
        "is_valid",
        "fail_reason",
        # private
        "_actual_input_data_ranges",
        "_csv_row_index",
        "xpu_results",
    )

    identity_headers: Dict[str, tuple] = {
        "testcase_name": (FIELD_TYPES.STRING, None),
    }

    # spec.describe 参数化用例的通路锚点：None = 该通路不参与（describe 不触发）。
    # 四通路类覆写："kernel"/"geir"/"aclnn"/"e2e"（注册名与查询通路由此
    # 派生，见 _apply_describe_shapes / _describe_operator_name）。
    describe_pathway: Optional[str] = None
    special_property_headers: Dict[str, tuple] = {
        "input_data_ranges": (FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED, None, ((None, None),)),
        "precision_tolerances": (FIELD_TYPES.SHAPELIKE_FLOAT_NESTED, None, None),
        "absolute_precision": (FIELD_TYPES.FLOAT_SCALAR_NESTED, None, 1e-8),
    }
    option_headers: Dict[str, tuple] = {
        # Manually controlled property
        "is_enabled": (FIELD_TYPES.BOOL, None, True),
        # 设备指定："0,1,2" 多卡（与 CLI 池取交集）/ "all" 池全量 / "3" 指定单卡（pin）/ 空 调度自选
        "device_ids": (FIELD_TYPES.STRING, None, None),
        # batch 一致性字段组
        "batch_axis": (FIELD_TYPES.FREE_EVAL, None, None),
        "batch_slice_info": (FIELD_TYPES.FREE_EVAL, None, None),
        "batch_seed": (FIELD_TYPES.FREE_EVAL, None, None),
        "remark": (FIELD_TYPES.STRING, None, None),
        "soc_series": (FIELD_TYPES.STRING_CONTAINER, None, None),
        "priority": (FIELD_TYPES.INT, None, 0),
    }
    complete_headers: Dict[str, tuple] = {**identity_headers, **special_property_headers, **option_headers}

    def __init__(self):
        super().__init__()
        self.testcase_name: Optional[str] = None
        self.input_data_ranges = None
        # precision
        self.precision_tolerances = None
        self.absolute_precision: float = 1e-8
        # control
        self.is_enabled: bool = True
        self.device_ids: Optional[tuple] = None
        # batch 一致性（validate 时派生 bcid；None = 不参与分组）
        self.batch_axis = None
        self.batch_slice_info = None
        self.batch_seed = None
        self.batch_consistency_id = None
        self.priority: int = 0  # testcase priority.
        self.remark = None
        self.soc_series = None
        # End of testcase valid configurations
        # Temp
        self.original_line: Optional[tuple] = None
        self.original_dict: Optional[dict] = None
        # === Runtime parameters below === #
        self.device_id = None
        self.is_valid: Optional[bool] = True
        self.fail_reason: Optional[str] = None
        # private
        self._actual_input_data_ranges = None
        self._csv_row_index: int = -1
        self.xpu_results = {}

    def __hash__(self):
        return hash(self.testcase_name)

    @property
    def actual_input_data_ranges(self):
        if self._actual_input_data_ranges is None:
            self._actual_input_data_ranges = self.input_data_ranges
        return self._actual_input_data_ranges

    @actual_input_data_ranges.setter
    def actual_input_data_ranges(self, value):
        self._actual_input_data_ranges = value

    @classmethod
    def is_legit_header(cls, header_name: str) -> bool:
        return header_name in cls.complete_headers

    @classmethod
    def has_equivalent_header(cls, header_name: str) -> bool:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return bool(cls.complete_headers[header_name][1])

    @classmethod
    def get_equivalent_headers(cls, header_name: str) -> Optional[tuple]:
        if cls.has_equivalent_header(header_name):
            return cls.complete_headers[header_name][1]
        return ()

    @classmethod
    def get_all_legit_headers(cls) -> Tuple[str]:
        return tuple(cls.complete_headers.keys())

    @classmethod
    def has_default_value(cls, header_name: str) -> bool:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return len(cls.complete_headers[header_name]) >= 3

    @classmethod
    def get_default_value(cls, header_name: str) -> Any:
        if cls.has_default_value(header_name):
            return cls.complete_headers[header_name][2]
        raise ValueError(f"Could not find default value for header {header_name}, required field missing!")

    @classmethod
    def get_header_type(cls, header_name: str) -> FIELD_TYPES:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return cls.complete_headers[header_name][0]

    @classmethod
    def get_header_func(cls, header_name: str) -> Callable:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return type_processing_func[cls.get_header_type(header_name)]

    @classmethod
    def get_all_visible_headers(cls) -> Tuple[str]:
        """All visible headers as csv titles"""
        return cls.get_all_legit_headers()

    @staticmethod
    def supported_rerun_title() -> tuple:
        return ()

    def ready_for_profile(self) -> bool:
        """Ready for profile"""
        return True

    def pick_data(self, titles: Tuple[str]) -> tuple:
        """Pick testcase input data via titles（解析值直出）"""
        data = []
        legit_headers = self.get_all_legit_headers()
        for t in titles:
            original_input = self.original_dict.get(t, "")
            data.append(getattr(self, t) if t in legit_headers else original_input)
        return tuple(data)

    def validate(self):
        """归一化入口。base 层：device_ids 解析（"0,1,2"→(0,1,2)，非法值报错）
        + batch_consistency_id 派生（seed+axis+slice 拼接）；子类 override
        追加 keeper 对账。"""
        self._merge_extended_attributes()
        self._normalize_input_data_range()
        self._parse_device_ids_field()
        self._generate_batch_consistency_id()

    # ========== spec.describe 参数化用例 ==========

    def _apply_describe_shapes(self):
        """CSV 未提供形状/类型列时，未识别列 → params 调 spec.describe。

        契约：describe(params) 只返回 {"shapes": ..., "dtypes": ...} 两个
        key；各通路经 _assemble_described_shapes 把二者映射成本通路的
        shape 字段（kernel/geir→input_shapes/input_dtypes，aclmn/e2e→
        tensor_view_shapes/tensor_dtypes）。aclnn 的 shapes 按 API 参数
        列表含输出槽位（输出索引/属性等其余字段仍走 CSV 列）。
        shapes 与 dtypes 不强绑定：两者任一缺省即触发 describe，缺啥补
        啥——CSV 已显式声明的字段不回填（"不缺的不读"）。describe 未
        声明 → 静默跳过（沿用原"形状为空"语义）；抛异常 / 非 dict 除
        外，缺 shapes 且 CSV 未提供形状 → 用例置无效。仅主进程
        validate 期调用一次。
        """
        pathway = self.describe_pathway
        if not pathway or not self.is_valid:
            return
        name = self._describe_operator_name()
        gate_fields = self._describe_gate_fields()
        if not name or not gate_fields:
            return
        shapes_field, dtypes_field = gate_fields
        if not shapes_field or not dtypes_field:
            return
        if getattr(self, shapes_field, None) and getattr(self, dtypes_field, None):
            return
        from ttk.switches import get_global_storage

        from ..plugin import InvokeDescribeError, SpecResolver
        from .describe import collect_describe_params

        params = collect_describe_params(self)
        if not params:
            return
        resolver = SpecResolver(name, pathway, get_global_storage().plugin_path)
        try:
            declared = resolver.invoke_describe(params)
        except InvokeDescribeError as exc:
            self.is_valid = False
            self.fail_reason = exc.code
            logging.error("[%s] %s", self.testcase_name, exc)
            return
        if declared is None:
            return  # spec 未注册 describe：静默跳过（沿用原"形状为空"语义）
        shapes = declared.get("shapes")
        if shapes is None and not getattr(self, shapes_field, None):
            self.is_valid = False
            self.fail_reason = "DESCRIBE_MISSING_SHAPES"
            logging.error("[%s] describe(%s) must return a 'shapes' key", self.testcase_name, name)
            return
        self._assemble_described_shapes(shapes, declared.get("dtypes"))

    def _describe_operator_name(self) -> Optional[str]:
        """describe 的注册名：kernel/geir 用 op_name，aclmn/e2e 用 api_name。"""
        return None

    def _describe_gate_fields(self) -> Optional[tuple]:
        """本通路的 (形状字段, 类型字段)：两者均非空时不触发 describe；
        任一缺省则触发，且只回填缺省字段（CSV 已声明字段优先）。"""
        return None

    def _assemble_described_shapes(self, shapes, dtypes) -> None:  # noqa: B027 通路覆写钩子（默认 no-op）
        """把 describe 的 shapes/dtypes 映射到本通路 shape 字段（通路覆写）。"""

    def _merge_described(self, field, value) -> None:
        """describe 值回填：CSV 显式声明（非空单元格）优先，不覆盖。"""
        if (self.original_dict or {}).get(field):
            return
        try:
            setattr(self, field, value)
        except AttributeError as exc:
            logging.warning("[%s] describe cannot set %r: %s", self.testcase_name, field, exc)

    def _parse_device_ids_field(self):
        """CSV 字符串解析："0,1,2"→(0,1,2) / "all"→探测全量 / 空→None / 已 tuple 透传。

        非法值（非整数 token / "all" 时设备数未知）置 is_valid=False 并给
        明确 fail_reason——不向 parse_cases 抛异常（一条坏值不炸整仓跑）。
        """
        if self.device_ids is not None and isinstance(self.device_ids, str):
            raw = self.device_ids.strip()
            if raw.lower() == "all":
                self.device_ids = self._expand_all_device_ids()
            elif raw:
                try:
                    self.device_ids = tuple(int(d.strip()) for d in raw.split(",") if d.strip())
                except ValueError:
                    self.is_valid = False
                    self.fail_reason = (
                        f"invalid device_ids {raw!r}: expect 'all' or comma-separated integers (e.g. '0,1,2')"
                    )
                    self.device_ids = None
            else:
                self.device_ids = None
        if self.device_ids is not None and not isinstance(self.device_ids, tuple):
            self.device_ids = tuple(self.device_ids)

    def _expand_all_device_ids(self):
        """'all' → tuple(range(device_count))。count 来自探测/--dev（bridge 已写）。

        --validate-only 早退于 detect_device_count：此时 count 未知（-1），
        明确报错而非静默展开空集。
        """
        from ttk.switches import get_global_storage

        count = getattr(get_global_storage(), "device_count", -1)
        if count is None or count < 1:
            self.is_valid = False
            self.fail_reason = (
                "device_ids='all' requires a known device count, but device detection has "
                "not run (--validate-only returns before detection); pass --dev N or drop --validate-only"
            )
            return None
        return tuple(range(count))

    def is_multi_device(self) -> bool:
        """len(device_ids) > 1。"""
        return self.device_ids is not None and len(self.device_ids) > 1

    def _generate_batch_consistency_id(self):
        """根据 batch_seed/axis/slice_info 生成 batch_consistency_id
        （None 元数据 → bcid=None 不参与分组）。"""
        if self.batch_seed is None or self.batch_axis is None or self.batch_slice_info is None:
            self.batch_consistency_id = None
            return
        slice_key = []
        for axis_pos, slices, seed in zip(self.batch_axis, self.batch_slice_info, self.batch_seed):
            if axis_pos is None or slices is None or seed is None:
                continue
            slice_axes = []
            for axis_idx, slices_idx, seed_idx in zip(axis_pos, slices, seed):
                if axis_idx is None or slices_idx is None or seed_idx is None:
                    slice_axes.append("None")
                    continue
                slice_ids = []
                for sl, seed_value in zip(slices_idx, seed_idx):
                    if sl is None:
                        continue
                    start, stop, step = sl[0], sl[1], sl[2]
                    try:
                        length = len(range(start, stop, step)) if step > 0 and start >= 0 and stop >= 0 else 0
                    except (TypeError, ValueError):
                        length = 0
                    slice_id = f"{seed_value}_{axis_idx}_{start}_{stop}_{step}"
                    if length == 0:
                        logging.warning(
                            f"testcase: {self.testcase_name}, slice_id is: {slice_id}, slice is:{sl} this slice is Invalid"
                        )
                    slice_ids.append(slice_id)
                slice_axes.append(tuple(slice_ids))
            slice_key.append(tuple(slice_axes))
        self.batch_consistency_id = tuple(slice_key) if slice_key else None

    def _normalize_input_data_range(self):
        """if only low is given, fix it to inf. Supports nested structures."""
        if not self.input_data_ranges:
            self.input_data_ranges = ((None, None),)
            return
        self.input_data_ranges = self._normalize_range_recursive(self.input_data_ranges)

    @staticmethod
    def _normalize_range_recursive(ranges):
        result = []
        for r in ranges:
            if r is None:
                result.append(None)
            elif isinstance(r, (tuple, list)) and len(r) > 0 and isinstance(r[0], (tuple, list)):
                result.append(TestcaseBase._normalize_range_recursive(r))
            else:
                normalized = tuple(r)
                if len(normalized) >= 2:
                    result.append(normalized)
                elif len(r) == 0:
                    result.append((None, None))
                else:
                    result.append((normalized[0], normalized[0]))
        return tuple(result)

    @classmethod
    def _resolve_pad_value(cls, field_name, explicit_pad=_NO_PAD):
        """Resolve per-tensor pad value.

        If caller provides explicit_pad, use it directly.
        Otherwise try to derive from complete_headers field default:
          - default is None           → pad is None
          - default is scalar         → pad is that scalar
          - default is (val,) tuple   → pad is val (per-tensor default)
          - no default / empty tuple  → no pad (_NO_PAD)
        """
        if explicit_pad is not _NO_PAD:
            return explicit_pad
        if field_name not in cls.complete_headers:
            return _NO_PAD
        entry = cls.complete_headers[field_name]
        if len(entry) < 3:
            return _NO_PAD
        default = entry[2]
        if default is None:
            return None
        if isinstance(default, tuple) and len(default) == 1:
            return default[0]
        if isinstance(default, (int, float)):
            return default
        return _NO_PAD

    @classmethod
    def _is_scalar_group(cls, v):
        """Scalar group: tuple/list of atomic str/int/float values."""
        return isinstance(v, (tuple, list))

    @classmethod
    def _is_range_group(cls, v):
        """Range group: tuple/list where elements are also tuple/list."""
        return bool(isinstance(v, (tuple, list)) and v and isinstance(v[0], (tuple, list)))

    def _write_back_normalized(self, field_name, result):
        """Write back normalized field and clear its flat cache."""
        setattr(self, field_name, tuple(result))
        cache_attr = f"_flat_{field_name}"
        if hasattr(self, cache_attr):
            setattr(self, cache_attr, None)

    def _normalize_field_by_dist(self, field_name, dist, is_group_fn, pad_value=_NO_PAD):
        """Normalize a field to match distribution nesting exactly.

        Unified method serving both scalar and range fields. The is_group_fn
        callback distinguishes the two semantics:
          - Scalar (_is_scalar_group): any tuple/list is a group.
          - Range  (_is_range_group): tuple/list with tuple/list elements is a group.

        Leaf values are preserved atomically:
          - Scalar: str/int/float values broadcast as-is.
          - Range: (min,max) or (rtol,ptol) pairs are NOT destructed.

        Caller must ensure field is a tuple/list (wrap bare scalar values before
        calling). Empty/None fields should be handled by caller and not reach here.

        Already-nested detection:
          len(field) == len(dist), and at each position:
            num>0: isinstance(val, tuple/list) and len(val) == num
            num>1: is_group_fn(val) == True (prevents false positives where a
                   non-group value happens to match the count).

        Supported scenarios (len(field) relative to dist):
          1. len==1, single value: broadcast to all positions, then expand
             each TensorList position internally. (most common compressed form)
          2. len==1, multi-element group: ambiguous → mark invalid.
          3. 1 < len <= len(dist): pad missing positions with pad_value, then expand.
          4. len > len(dist): ambiguous → mark invalid.

        Rule 1 (TensorList internal, dist[i] > 0): accept only:
          - Fully-specified group (len == num)
          - Single-element group (g,) → broadcast g to num copies
          - Bare value (not group) → broadcast to num copies
          - Otherwise (group length mismatch) → mark invalid (CASE_FIELD_AMBIGUOUS)

        Rule 2 (field-level padding): len(field) < len(dist) → pad with pad_value.
        If pad_value is _NO_PAD and padding is needed → mark invalid.
        """
        if not self.is_valid:
            return
        field = getattr(self, field_name)
        if not field:
            return
        # Already-nested check
        if len(field) == len(dist):
            nested = True
            for val, num in zip(field, dist):
                if num > 0:
                    if not isinstance(val, (tuple, list)) or len(val) != num:
                        nested = False
                        break
                    if num > 1 and not is_group_fn(val):
                        nested = False
                        break
            if nested:
                return
        # Resolve pad value
        pad_value = self._resolve_pad_value(field_name, pad_value)
        # Branch by len(field) vs len(dist)
        if len(field) > len(dist):
            self.is_valid = False
            self.fail_reason = "CASE_FIELD_AMBIGUOUS"
            logging.error(f"Field [{field_name}] of [{self.testcase_name}] is invalid.")
            return
        if len(field) == 1:
            val = field[0]
            # Unwrap single-element group: (('NCHW',),) → 'NCHW'
            if is_group_fn(val) and len(val) == 1:
                val = val[0]
            # Reject multi-element group: (('a','b'),) with dist=(2,0)
            if is_group_fn(val) and len(val) > 1:
                self.is_valid = False
                self.fail_reason = "CASE_FIELD_AMBIGUOUS"
                logging.error(f"Field [{field_name}] of [{self.testcase_name}] is invalid.")
                return
            per_param = [val] * len(dist)
        elif len(field) < len(dist):
            if pad_value is _NO_PAD:
                self.is_valid = False
                self.fail_reason = "CASE_FIELD_AMBIGUOUS"
                logging.error(f"Field [{field_name}] of [{self.testcase_name}] is invalid.")
                return
            per_param = list(field) + [pad_value] * (len(dist) - len(field))
        else:
            per_param = list(field)
        # Per-position expansion
        result = []
        for num in dist:
            val = per_param[len(result)]
            if num == 0:
                result.append(val)
            elif is_group_fn(val):
                if len(val) == num:
                    result.append(tuple(val))
                elif len(val) == 1:
                    result.append(tuple([val[0]] * num))
                else:
                    self.is_valid = False
                    self.fail_reason = "CASE_FIELD_AMBIGUOUS"
                    logging.error(f"Field [{field_name}] of [{self.testcase_name}] is invalid.")
                    return
            else:
                result.append(tuple([val] * num))
        self._write_back_normalized(field_name, result)

    def _normalize_scalar_field_by_dist(self, field_name, dist):
        """Normalize a scalar/string field to match distribution nesting exactly.

        Bare value (1e-5) → wrapped to (1e-5,) → len==1 broadcast.
        Delegates to _normalize_field_by_dist with _is_scalar_group.
        """
        if not self.is_valid:
            return
        field = getattr(self, field_name)
        if not field:
            return
        if not isinstance(field, (tuple, list)):
            # Bare scalar value — broadcast
            result = []
            for num in dist:
                if num == 0:
                    result.append(field)
                else:
                    result.append(tuple([field] * num))
            self._write_back_normalized(field_name, result)
            return
        self._normalize_field_by_dist(field_name, dist, self._is_scalar_group)

    def _normalize_range_field_by_dist(self, field_name, dist, pad_value=_NO_PAD):
        """Normalize a range/pair field to match distribution nesting.

        Range fields have atomic leaf values like (min,max) or (rtol,ptol) that
        must NOT be destructed during expansion. Delegates to
        _normalize_field_by_dist with _is_range_group for range-aware group
        detection. See _normalize_field_by_dist for full documentation.
        """
        if not self.is_valid:
            return
        field = getattr(self, field_name)
        if not field or not isinstance(field, (tuple, list)):
            return
        self._normalize_field_by_dist(field_name, dist, self._is_range_group, pad_value=pad_value)

    def _merge_extended_attributes(self):
        """Merge ``attributes1``~``attributes9`` into ``attributes`` (as attributes0).

        Columns are merged in order attributes(0) -> attributes1 -> ... -> attributes9.
        Later columns override earlier ones on key conflicts (no error).
        """
        if not self.is_valid:
            return
        if not any(getattr(self, f"attributes{i}", None) for i in range(1, 10)):
            return
        merged = dict(getattr(self, "attributes", None) or {})
        for i in range(1, 10):
            ext = getattr(self, f"attributes{i}", None)
            if ext:
                merged.update(ext)
        self.attributes = merged
