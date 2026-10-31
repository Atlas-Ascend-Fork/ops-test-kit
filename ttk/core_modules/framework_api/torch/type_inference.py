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
torch 参数类型启发式与归一化。

多源解析链共用的「查表 + 规则」层：docstring/annotations/inspect 等源产出
的原始类型串经此归一（_normalize_args_type）、按参数名/默认值推断类型
（_infer_type_from_name/_infer_type_from_value）、默认值字符串还原
（_coerce_default_value）。仅 torch 域内消费。
"""

import ast
import re
from typing import List

from ..api_model import ParamInfo

_ARGS_TYPE_NORMALIZE = {
    "Tensor": "Tensor",
    "tensor": "Tensor",
    "``Tensor``": "Tensor",
    "Number": "Number",
    "Scalar": "Number",
    "``scalar``": "Number",
    "bool": "bool",
    ":class:`bool`": "bool",
    "SymBool or bool": "bool",
    "SymInt or int": "int",
    "int": "int",
    "``int``": "int",
    "integer": "int",
    "int32": "int",
    "int64": "int",
    "float": "float",
    "str": "str",
    "dtype": "Dtype",
    ":class:`torch.dtype`": "Dtype",
    "torch.dtype": "Dtype",
    "device": "Device",
    ":class:`torch.device`": "Device",
    "torch.device": "Device",
    "layout": "Layout",
    ":class:`torch.layout`": "Layout",
    "torch.layout": "Layout",
    "torch.Size": "tuple of ints",
    ":class:`torch.Size`": "tuple of ints",
    ":class:`torch.Generator`": "Generator",
    "torch.Generator": "Generator",
    "Generator": "Generator",
    ":class:`torch.memory_format`": "MemoryFormat",
    "torch.memory_format": "MemoryFormat",
    "memory_format": "MemoryFormat",
    "BoolTensor": "Tensor",
    "IntTensor or LongTensor": "Tensor",
    "LongTensor": "Tensor",
    "torch.ByteTensor": "Tensor",
    "Tensors...": "tuple of Tensors",
    "Tensor[]": "tuple of Tensors",
    "List[Tensor]": "tuple of Tensors",
    "list of Tensor": "tuple of Tensors",
    "sequence of Tensors": "tuple of Tensors",
    "sequence of float": "list of Numbers",
    "torch.Tensor": "Tensor",
    "torch.BoolTensor": "Tensor",
    "torch.IntTensor": "Tensor",
    "torch.LongTensor": "Tensor",
    "torch.FloatTensor": "Tensor",
    "torch.DoubleTensor": "Tensor",
}


def _normalize_args_type(type_str: str) -> str:
    """Normalize Args section type to our type system.

    Handles: "Tensor", "int or tuple of ints", "``Tensor``", ":class:`bool`", etc.

    Union types ("X or Y") are handled as follows:
    - Tensor-containing members (Tensor, tuple of Tensors) are excluded because
      those overloads are already captured by TypeError multi-overload resolution.
    - Remaining scalar-level members are normalized and joined with '|'.
    - If only one non-Tensor member remains, it is returned directly (no union).
    - If no non-Tensor members survive, returns empty string.

    Examples:
        'int or Tuple[int]'    → 'int|tuple of ints'
        'float or Tuple[float]'→ 'float|tuple of floats'
        'Tensor or Number'     → 'Number'
        'Tensor or float'      → 'float'
        'int or list or tuple of ints' → 'int|tuple of ints'
    """
    type_str = type_str.strip()
    if not type_str:
        return ""
    # Handle Optional[X] — strip the wrapper and normalize the inner type
    opt_match = re.match(r"Optional\[(.+)\]$", type_str)
    if opt_match:
        return _normalize_args_type(opt_match.group(1))
    if type_str in _ARGS_TYPE_NORMALIZE:
        return _ARGS_TYPE_NORMALIZE[type_str]
    # Handle backtick-quoted types: ``int``, ``list of int``
    if type_str.startswith("``") and type_str.endswith("``"):
        inner = type_str[2:-2]
        if inner in _ARGS_TYPE_NORMALIZE:
            return _ARGS_TYPE_NORMALIZE[inner]
        return ""
    # Handle :class:`...` pattern
    cm = re.match(r":class:`([^`]+)`", type_str)
    if cm:
        return _normalize_args_type(cm.group(1))
    # Union types: "int or Tuple[int]", "Tensor or Number", etc.
    # Tensor-containing members are excluded (handled by overload resolution).
    # Scalar-level members are joined with '|' for _coerce_value to try in order.
    if " or " in type_str:
        parts = [p.strip() for p in type_str.split(" or ")]
        _TENSOR_TYPES = frozenset({"Tensor", "tuple of Tensors", "list of Tensors", "List[Tensor]", "Tensor[]"})
        normalized_parts = []
        for part in parts:
            n = _normalize_args_type(part)
            if n and n not in _TENSOR_TYPES:
                normalized_parts.append(n)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for n in normalized_parts:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        if len(unique) >= 2:
            return "|".join(unique)
        if len(unique) == 1:
            return unique[0]
        return ""
    # Handle comma-separated: "int, float, inf, -inf, 'fro', 'nuc'"
    # Treated as union types (same as "X or Y"), each member normalized
    # and joined with '|'. Quoted values like 'fro' are recognized as 'str'.
    # Skipped for types that genuinely contain commas (e.g., "tuple of ints").
    if "," in type_str and not type_str.startswith(("tuple", "list", "Tuple", "List")):
        parts = [p.strip() for p in type_str.split(",")]
        _TENSOR_TYPES = frozenset({"Tensor", "tuple of Tensors", "list of Tensors", "List[Tensor]", "Tensor[]"})
        normalized_parts = []
        for part in parts:
            # Quoted string values like 'fro', 'nuc' → 'str'
            if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):  # noqa: PIE810 保持
                normalized_parts.append("str")
                continue
            n = _normalize_args_type(part)
            if n and n not in _TENSOR_TYPES:
                normalized_parts.append(n)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for n in normalized_parts:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        if len(unique) >= 2:
            return "|".join(unique)
        if len(unique) == 1:
            return unique[0]
        return ""
    # Handle "tuple of ints", "list of Tensors"
    if type_str.startswith(("tuple of ", "list of ")):
        inner = type_str.split(" of ")[1].strip()
        if inner in ("ints", "int"):
            return "tuple of ints"
        if inner in ("floats", "float"):
            return "tuple of floats"
        if inner in ("Tensors", "Tensor"):
            return "tuple of Tensors"
        if inner in ("Scalars", "Scalar", "numbers", "Number"):
            return "list of Numbers"
    # Handle "Tuple[int]", "Tuple[int, int]", "List[float]", etc.
    # Extract element type from generic bracket syntax and map to our type system.
    bracket_match = re.match(r"(Tuple|List)\[([^\]]+)\]", type_str)
    if bracket_match:
        container = bracket_match.group(1)
        inner = bracket_match.group(2).strip()
        parts = [p.strip() for p in inner.split(",")]
        unique = set(parts)
        if unique <= {"int", "int64", "int32"}:
            return "tuple of ints"
        if unique <= {"float", "double"}:
            return "tuple of floats"
        if unique == {"bool"}:
            return "tuple of bools"
        if unique == {"Tensor"}:
            return "tuple of Tensors"
        if len(unique) == 1 and parts[0] in ("float", "int", "bool", "str"):
            return f"tuple of {parts[0]}s"
        return "tuple of ints" if container == "Tuple" else "list of Numbers"
    # If it's a known simple type, return it
    known_simple = {
        "Tensor",
        "tensor",
        "Number",
        "Scalar",
        "bool",
        "int",
        "float",
        "str",
        "int64",
        "int32",
        "object",
        "Object",
        "Generator",
        "function",
        "callable",
    }
    if type_str in known_simple:
        return _ARGS_TYPE_NORMALIZE.get(type_str, type_str)
    # Descriptive text — return empty, let caller fall back to name inference
    return ""


def _infer_type_from_name(name: str) -> str:
    """按参数名推断类型（命名启发式表：input/weight→Tensor、dim→int、dtype→
    torch.dtype、p→float…）。供 docstring/inspect 等无类型标注的源兜底；产物经
    _is_name_inferred_type 识别，后续可被 annotations/别名覆盖。
    """
    tensor_names = {
        "input",
        "output",
        "other",
        "self",
        "tensor",
        "weight",
        "bias",
        "mat1",
        "mat2",
        "x",
        "y",
        "src",
        "dst",
        "query",
        "key",
        "mask",
        "grad",
        "indices",
        "values",
        "h_0",
        "c_0",
        "hidden",
        "cell",
        "source",
        "target",
    }
    tensor_list_names = {
        "tensors",
        "inputs",
        "targets",
        "features",
        "labels",
        "boxes",
    }
    scalar_names = {
        "dim",
        "dims",
        "axis",
        "size",
        "sizes",
        "shape",
        "shapes",
        "stride",
        "strides",
        "padding",
        "dilation",
        "alpha",
        "beta",
        "gamma",
        "epsilon",
        "momentum",
        "p",
        "dropout",
        "training",
        "inplace",
    }
    special_types = {
        "dtype": "Dtype",
        "layout": "Layout",
        "device": "Device",
        "memory_format": "MemoryFormat",
        "requires_grad": "bool",
        "pin_memory": "bool",
        "inplace": "bool",
        "training": "bool",
        "keepdim": "bool",
        "keepdims": "bool",
        "approximate": "str",
        "ceil_mode": "bool",
        "return_indices": "bool",
        "kernel_size": "int",
        "output_size": "int",
        "eps": "float",
        "epsilon": "float",
        "exponential_average_factor": "float",
        "align_corners": "bool",
        "transposed": "bool",
        "groups": "int",
        "numel": "int",
        "n_bins": "int",
        "ratio": "float",
        "bit_width": "int",
        "count": "int",
        "counts": "int",
        "mode": "int",
        "hidden_size": "int",
        "num_layers": "int",
        "batch_first": "bool",
        "bidirectional": "bool",
        "has_biases": "bool",
        "reverse": "bool",
        "bias_defined": "bool",
        "n": "int",
        "c": "int",
        "h": "int",
        "w": "int",
        "hxw": "int",
        "group": "int",
        "full": "bool",
        "reduction": "str",
        "log_input": "bool",
        "observer_on": "bool",
        "fake_quant_on": "bool",
        "averaging_const": "float",
        "quant_min": "int",
        "quant_max": "int",
        "ch_axis": "int",
        "interpolation_mode": "int",
        "padding_mode": "int",
        "benchmark": "bool",
        "deterministic": "bool",
        "allow_tf32": "bool",
        "weight_stride0": "int",
        "dropout_state": "bool",
        "use_input_stats": "bool",
        "cudnn_enabled": "bool",
        "train": "bool",
        "reduce": "str",
        "include_self": "bool",
    }
    name_lower = name.lower().strip("\"'")
    if name_lower in special_types:
        return special_types[name_lower]
    if name_lower in tensor_list_names:
        return "tuple of Tensors"
    if name_lower in tensor_names:
        return "Tensor"
    if name_lower in scalar_names:
        if name_lower in ("p", "alpha", "beta", "gamma", "epsilon", "dropout"):
            return "float"
        return "int"
    # Pattern-based inference for parameter names not in exact lists
    # dim0/dim1/axis0/axis1 etc.
    if name_lower.startswith(("dim", "axis")):
        return "int"
    # chunks/split_size/chunk_size etc.
    if "chunk" in name_lower or "split" in name_lower:
        return "int"
    # threshold/lambd/tol/tolerance
    if any(k in name_lower for k in ("threshold", "lambd", "tol")):
        return "float"
    # pad/padding_mode (but 'padding' already in scalar_names)
    if name_lower.startswith("pad") and name_lower not in tensor_names:
        return "int"
    # scale/scale_value
    if name_lower.startswith("scale"):
        return "float"
    # negative_slope/lower/upper/min_val/max_val
    if name_lower in (
        "negative_slope",
        "lower",
        "upper",
        "min_val",
        "max_val",
        "correction",
        "fill_value",
        "margin",
        "value",
    ):
        return "float"
    return "Tensor"


def _infer_type_from_value(value: str) -> str:
    """Infer parameter type from its default value string.

    Priority: None→Optional[Tensor], True/False→bool, quoted→str,
    numeric (e.g., 1.0, 3)→float/int.
    Module attribute references like 'torch.xxx' are NOT treated as float
    even though they contain '.'.
    """
    value = value.strip()
    if value == "None":
        return "Optional[Tensor]"
    if value in ("True", "False"):
        return "bool"
    if value.startswith(("'", '"')):
        return "str"
    # Try list/tuple literal: [1, 2], (0, 0) → tuple of ints/floats
    if value.startswith(("[", "(")) and value.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                if all(isinstance(x, bool) for x in parsed):
                    return "tuple of bools"
                if all(isinstance(x, int) for x in parsed):
                    return "tuple of ints"
                return "tuple of floats"
        except (ValueError, SyntaxError):
            pass  # noqa: S110 协议探测：失败跳过属预期
    # Try parsing as numeric first — only values that are actually numbers
    # should be classified as int/float. Strings like 'torch.preserve_format'
    # or 'fro' will fail numeric parsing and fall through to return "".
    try:
        float(value)
        if "." in value or "e" in value.lower():
            return "float"
        return "int"
    except (ValueError, OverflowError):
        return ""


def _coerce_default_value(raw: str, type_hint: str):
    """默认值字符串按目标类型还原（str 引号剥壳、bool、int 含 at::Reduction 枚举
    映射、'None'→None）。
    """
    if raw == "None":
        return None
    if type_hint == "str":
        if len(raw) >= 2 and ((raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'")):
            return raw[1:-1]
        return raw
    if type_hint == "bool":
        if isinstance(raw, bool):
            return raw
        return raw.lower() in ("true", "1")
    if type_hint == "int":
        if isinstance(raw, str):
            _REDUCTION_CPP = {
                "at::Reduction::None": 0,
                "at::Reduction::Mean": 1,
                "at::Reduction::Sum": 2,
                "Reduction::None": 0,
                "Reduction::Mean": 1,
                "Reduction::Sum": 2,
            }
            if raw in _REDUCTION_CPP:
                return _REDUCTION_CPP[raw]
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    if type_hint == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    return raw


def _is_name_inferred_type(type_str: str) -> bool:
    """Check if a type was inferred from name rather than from an authoritative source.

    For union types (e.g., 'int|tuple of ints'), returns True only if ALL members
    are inferred types. If any member comes from an authoritative source (e.g.,
    Args section), the whole type is considered non-inferred.
    """
    _INFERRED = frozenset(
        {
            "Tensor",
            "float",
            "int",
            "bool",
            "str",
            "tuple of Tensors",
            "tuple of ints",
            "tuple of floats",
            "list of Numbers",
            "Number",
            "Dtype",
            "Device",
            "Layout",
        }
    )
    return all(t in _INFERRED for t in type_str.split("|"))


def _enrich_types_from_annotations(params: List[ParamInfo], ann_params: List[ParamInfo]) -> List[ParamInfo]:
    """Replace inferred types with real types from __annotations__.

    Only overwrites types that came from _infer_type_from_name.
    Authoritative types from docstring Args section or TypeError are kept.
    Union types (containing '|') from Args section are considered authoritative
    and are not overwritten by annotations.
    Also fills in defaults from annotations/signature when missing.
    """
    if not ann_params:
        return params
    ann_map = {p.name: p for p in ann_params if p.type}
    for p in params:
        ann_p = ann_map.get(p.name)
        if ann_p is None:
            continue
        # Union types from Args section are authoritative — don't overwrite
        if _is_name_inferred_type(p.type) and "|" not in p.type and ann_p.type:
            p.type = ann_p.type
        if p.default is None and ann_p.default is not None:
            p.default = ann_p.default
        if not p.is_optional and ann_p.is_optional:
            p.is_optional = ann_p.is_optional
    return params
