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

from ...utilities import get_global_storage
from .field_parser import (
    int_container,
    process_bool,
    process_dict,
    process_dynamic_inferable_shapelike,
    process_dynamic_shapelike,
    process_eval,
    process_float,
    process_int,
    process_string,
    rangelike,
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
    FIELD_TYPES.RANGELIKE: rangelike,
    FIELD_TYPES.SHAPELIKE_STC: shapelike_stc,
    FIELD_TYPES.SHAPELIKE_STC_EX: shapelike_stc_ex,
    FIELD_TYPES.SHAPELIKE_FLOAT: shapelike_float,
    FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED: shapelike_float_signed,
    FIELD_TYPES.STRING_CONTAINER: string_container,
    FIELD_TYPES.INT_CONTAINER: int_container,
    FIELD_TYPES.INT: process_int,
    FIELD_TYPES.FLOAT: process_float,
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
        "network_name",
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
        "network_name": (FIELD_TYPES.STRING, None, None),
    }
    special_property_headers: Dict[str, tuple] = {
        "input_data_ranges": (FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED, None, ((None, None),)),
        "precision_tolerances": (FIELD_TYPES.SHAPELIKE_FLOAT_NESTED, None, None),
        "absolute_precision": (FIELD_TYPES.FLOAT_SCALAR_NESTED, None, 1e-8),
    }
    option_headers: Dict[str, tuple] = {
        # Manually controlled property
        "is_enabled": (FIELD_TYPES.BOOL, None, True),
        "remark": (FIELD_TYPES.STRING, None, None),
        "soc_series": (FIELD_TYPES.STRING_CONTAINER, None, None),
        "priority": (FIELD_TYPES.INT, None, 0),
    }
    complete_headers: Dict[str, tuple] = {**identity_headers, **special_property_headers, **option_headers}

    def __init__(self):
        super().__init__()
        self.testcase_name: Optional[str] = None
        self.network_name: Optional[str] = None
        self.input_data_ranges = None
        # precision
        self.precision_tolerances = None
        self.absolute_precision: float = 1e-8
        # control
        self.is_enabled: bool = True
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
        """Pick testcase input data via titles"""
        data = []
        legit_headers = self.get_all_legit_headers()
        for t in titles:
            original_input = self.original_dict.get(t, "")
            if get_global_storage().preserve_original_csv:
                data.append(original_input)
            else:
                data.append(getattr(self, t) if t in legit_headers else original_input)
        return tuple(data)

    def validate(self):
        self._merge_extended_attributes()
        self._normalize_input_data_range()

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
                item = tuple(r)
                if len(item) >= 2:
                    result.append(item)
                elif len(item) == 0:
                    result.append((None, None))
                else:
                    result.append((item[0], item[0]))
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

    @staticmethod
    def _flatten_by_distribution(values, distribution):
        """Flatten a distribution-aligned field to one-value-per-tensor."""
        result = []
        for vi, num in enumerate(distribution):
            val = values[vi]
            if num > 0:
                if isinstance(val, (tuple, list)):
                    if len(val) == num:
                        result.extend(val)
                    elif len(val) == 1:
                        result.extend([val[0]] * num)
                    else:
                        result.extend(val)
                else:
                    result.extend([val] * num)
            else:
                result.append(val)
        return tuple(result)

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
