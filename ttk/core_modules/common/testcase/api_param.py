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
ResolveApiParamBase：API 参数解析与调用参数装配契约 + 重载匹配/值强转公共件。

match_overload / coerce_value 原居 framework_api/api_param.py（上提为公开面），对 torch / tf 参数解析器产出的 ParamInfo/APIParamInfo 契约
（api_model.py）做重载匹配与属性值强转；E2E 的装配器 ParamPlan 留在
framework_api/api_param.py，aclnn 的 C 签名变体在 aclnn/api_param.py。
"""

import ast
import re
from abc import ABC, abstractmethod

from ttk.utilities.dtypes import str_to_torch_dtype


class ResolveApiParamBase(ABC):
    """API parameter resolution and argument assembly contract.

    A resolver parses a signature once at construction time into a param
    layout, then assembles call arguments from testcase data on demand:
    golden generation, input customization and profiling share one cached
    instance per testcase (``testcase.get_param_plan()``). Resolvers are
    worker-stage objects — testcase validation never builds one.

    ACLNN implements the C-signature variant (all-positional, three
    queues); E2E implements the Python-signature variant (overload
    matching, keyword-only args, out= separation).
    """

    api_name: str

    @abstractmethod
    def resolve_args(self, *args, **kwargs) -> tuple:
        """Assemble call arguments -> (args_list, kwargs_dict, extra_attrs)."""


def _safe_eval_division(s):
    if "/" not in s:
        return None
    parts = s.split("/")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]) / float(parts[1])
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _score_attr_type_compatibility(overload_params, attr_values):
    """Score how well attribute *values* match their expected param types in an overload.

    Used as a tiebreaker when multiple overloads have the same key-overlap score.
    For each (param, attr_value) pair, a lightweight isinstance check determines
    whether the runtime value is compatible with the declared param type, without
    invoking the full coerce_value pipeline.

    Scoring rules (per matched attribute):
        - union types ('a|b'): +1 (member types are not checked cheaply)
        - int: +1 if value is int/bool
        - float/Number/Scalar: +1 if value is int/float/bool
        - bool: +1 if value is bool
        - str: +1 if value is a str
        - tuple/list/torch.Size: +1 if value is tuple/list
        - otherwise (unknown types): +1 (assumed compatible — cannot verify cheaply)

    Args:
        overload_params: list of ParamInfo for one overload signature.
        attr_values: dict of {param_name: raw_value} from CSV attributes.

    Returns:
        int — number of attributes whose values are type-compatible.
    """
    score = 0
    for p in overload_params:
        if p.is_tensor_like or p.name == "out":
            continue
        if p.name not in attr_values:
            continue
        raw = attr_values[p.name]
        ptype = p.type

        if "|" in ptype:
            score += 1
            continue

        if ptype in ("int",):
            if isinstance(raw, (int, bool)):
                score += 1
        elif ptype in ("float", "Number", "Scalar"):
            if isinstance(raw, (int, float, bool)):
                score += 1
        elif ptype == "bool":
            if isinstance(raw, bool):
                score += 1
        elif ptype == "str":
            if isinstance(raw, str):
                score += 1
        elif "tuple" in ptype or "list" in ptype or ptype == "torch.Size":
            if isinstance(raw, (tuple, list)):
                score += 1
        else:
            score += 1
    return score


def match_overload(api_name, input_tensor_count, attributes=None, tensor_distribution=None, api_info=None):
    """Match testcase to a specific overload signature.

    Matching criteria (in order):
        1. Input tensor count (excluding out) must fit required..total range
        2. Tensor distribution must match (TensorList vs Tensor)
        3. Attribute key overlap — prefer overload whose non-tensor param names
           have the most overlap with provided attributes
        4. (Tiebreaker) Attribute value type compatibility — prefer overload
           whose declared param types best match the runtime attribute values

    Args:
        api_info: **Required**. APIParamInfo from FrameworkApiInfoKeeper.get() or equivalent.

    Returns:
        (overload_params, overload_index) or (None, -1)
    """
    info = api_info
    if info is None:
        return None, -1

    attrs = set(attributes.keys()) if attributes else set()
    attr_values = attributes or {}
    candidates = []

    for oidx, ov_info in enumerate(info.overloads):
        overload_params = ov_info.params
        input_tensors = [p for p in overload_params if p.is_tensor_like and p.name != "out"]
        has_var = any(getattr(p, "is_var_positional", False) for p in input_tensors)
        required = sum(1 for p in input_tensors if not p.is_optional and not getattr(p, "is_var_positional", False))
        # Shapes fill tensor params positionally in signature order; attributes
        # for those shape-covered params only override values (const folding),
        # not the param count. Only tensor params beyond the shape-covered
        # prefix may be satisfied by scalar attributes.
        shape_covered = min(input_tensor_count, len(input_tensors))
        scalar_cover = sum(1 for p in input_tensors[shape_covered:] if p.name in attrs and p.name != "self")
        effective_count = input_tensor_count + scalar_cover
        if has_var:
            if effective_count < required:
                continue
        else:
            total = len(input_tensors)
            if effective_count < required or effective_count > total:
                continue

        if tensor_distribution is not None:
            type_ok = True
            non_var_tensors = [p for p in input_tensors if not getattr(p, "is_var_positional", False)]
            for idx in range(min(input_tensor_count, len(non_var_tensors))):
                if tensor_distribution[idx]:
                    param = non_var_tensors[idx]
                    if not param.is_tensor_list:
                        type_ok = False
                        break
                else:
                    param = non_var_tensors[idx]
                    if param.is_tensor_list:
                        type_ok = False
                        break
            if not type_ok:
                continue

        non_tensor_names = {p.name for p in overload_params if not p.is_tensor_like and p.name != "out"}
        key_score = len(attrs & non_tensor_names)
        candidates.append((key_score, oidx, overload_params))

    if not candidates:
        return None, -1

    candidates.sort(key=lambda x: -x[0])
    top_key_score = candidates[0][0]
    tied = [(ks, oidx, op) for ks, oidx, op in candidates if ks == top_key_score]

    if len(tied) > 1:
        value_scored = [(ks, _score_attr_type_compatibility(op, attr_values), oidx, op) for ks, oidx, op in tied]
        value_scored.sort(key=lambda x: -x[1])
        return value_scored[0][3], value_scored[0][2]

    return tied[0][2], tied[0][1]


def _coerce_value(raw, target_type):
    """Coerce a raw value from CSV attributes to the expected Python type.

    Supports union types expressed as 'type1|type2' (e.g., 'int|tuple of ints').
    For union types, tries each member type in order and returns the first
    successful coercion. If all fail, raises ValueError with details.
    Single types are matched in order: bool / int / float·Number·Scalar / str /
    dtype names / typed arrays ('int[n]') / tuple·list·torch.Size / ``torch.*``
    attribute lookup (Device passes through raw); unmatched types raise ValueError.
    """
    if "|" in target_type:
        errors = []
        for member_type in target_type.split("|"):
            try:
                return _coerce_value(raw, member_type)
            except (ValueError, TypeError) as e:
                errors.append(str(e))
        raise ValueError(
            f"Cannot coerce {raw!r} to union type {target_type}: none of the members succeeded: {'; '.join(errors)}"
        )
    if raw is None:
        return None
    if isinstance(raw, str) and raw == "None":
        return None
    if type(raw).__module__ != "builtins":
        return raw
    if target_type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() in ("true", "1")
        return bool(raw)
    if target_type in ("int",):
        if isinstance(raw, int):
            return raw
        # int 目标接受标量序列（与下方 float/Number 分支同款的标量-序列宽容）：
        # 部分算子的"int"参数实为 int-or-int[] 声明，序列值逐位转 int
        if isinstance(raw, (tuple, list)):
            return tuple(int(v) for v in raw)
        if isinstance(raw, str):
            _REDUCTION_STR_TO_INT = {"none": 0, "mean": 1, "sum": 2, "elementwise_mean": 1}
            if raw.lower() in _REDUCTION_STR_TO_INT:
                return _REDUCTION_STR_TO_INT[raw.lower()]
            torch_obj = str_to_torch_dtype(raw)
            if torch_obj is not None:
                return torch_obj
        try:
            return int(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot coerce {raw!r} to int: {e}") from e
    if target_type in ("float", "Number", "Scalar"):
        if isinstance(raw, (int, float, bool)):
            return raw
        if isinstance(raw, (tuple, list)):
            return tuple(float(v) for v in raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except (ValueError, TypeError) as err:
                result = _safe_eval_division(raw)
                if result is not None:
                    return result
                raise ValueError(f"Cannot coerce {raw!r} to {target_type}: not a numeric value") from err
        return float(raw)
    if target_type == "str":
        if (
            isinstance(raw, str)
            and len(raw) >= 2
            and ((raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'"))
        ):
            return raw[1:-1]
        return str(raw)
    if target_type in ("ScalarType", "Dtype", "torch.dtype"):
        obj = str_to_torch_dtype(raw)
        if obj is not None:
            return obj
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}: not a torch/torch_npu dtype.")
    _ARRAY_TYPE_RE = re.compile(r"^(int|float|bool)\[(\d*)\]?\??$")
    m = _ARRAY_TYPE_RE.match(target_type)
    if m:
        elem_type = m.group(1)
        if isinstance(raw, (tuple, list)):
            return tuple(_coerce_value(v, elem_type) for v in raw)
        if isinstance(raw, bool if elem_type == "bool" else (int if elem_type == "int" else float)):
            return (_coerce_value(raw, elem_type),)
        if isinstance(raw, str):
            if not raw or raw in ("[]", "()"):
                return ()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (tuple, list)):
                    return tuple(_coerce_value(v, elem_type) for v in parsed)
                return (_coerce_value(parsed, elem_type),)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Cannot coerce {raw!r} to {target_type}: {e}") from e
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}")
    if "tuple" in target_type or "list" in target_type or target_type == "torch.Size":
        if isinstance(raw, (tuple, list)):
            return raw
        if isinstance(raw, int):
            return (raw,)
        if isinstance(raw, float):
            return (raw,)
        if isinstance(raw, str):
            if not raw or raw in ("[]", "()"):
                return ()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (tuple, list)):
                    return tuple(parsed)
                return (parsed,)
            except (ValueError, SyntaxError):
                try:
                    return (int(raw),)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot coerce {raw!r} to {target_type}: {e}") from e
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}")
    if isinstance(raw, str) and raw.startswith("torch."):
        import torch

        attr_name = raw.split(".", 1)[1] if "." in raw else raw
        obj = getattr(torch, attr_name, None)
        if obj is not None:
            return obj
    if target_type in ("torch.memory_format", "memory_format", "Layout", "torch.layout") and isinstance(raw, str):
        import torch

        obj = getattr(torch, raw, None)
        if obj is not None:
            return obj
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}: not a torch attribute")
    if target_type in ("Device", "torch.device"):
        return raw
    raise ValueError(f"Cannot coerce {raw!r} to {target_type}: unsupported type")
