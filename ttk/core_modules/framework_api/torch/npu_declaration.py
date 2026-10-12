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
torch_npu C++ Declaration 签名解析。

obj._op() 触发 RuntimeError 携带 Declaration 串（带类型/返回值），按括号
配平提取参数表——torch_npu 老版本（无 _schemas 暴露）的主解析路径；
新版由 aten._schemas 权威源顶替，本模块为版本兼容保留。
"""

import re
from typing import List, Optional, Tuple

from ..api_model import ParamInfo, resolve_function, split_by_comma
from .type_inference import _coerce_default_value


def _extract_params_from_op_declaration(api_name: str) -> Optional[Tuple[List[ParamInfo], str, int]]:
    """Extract params from torch_npu C++ Declaration by calling obj._op().

    Handles both torch_npu.xxx and torch.npu_xxx API naming conventions.
    Uses obj._op() (not obj()) to trigger RuntimeError containing the
    C++ Declaration string with typed parameter info.

    Returns:
        (params, source, return_count) or None
    """
    # Only handle NPU APIs: torch_npu.xxx or torch.npu_xxx
    if not (api_name.startswith(("torch_npu.", "torch.npu_"))):
        return None
    obj = resolve_function(api_name)
    if obj is None:
        return None
    # Use _op() to trigger Declaration — works for both torch_npu.xxx
    # and torch.npu_xxx (deprecated alias). obj() may hit deprecation
    # warnings instead of producing a Declaration for torch.npu_xxx.
    error_msg = None
    try:
        obj._op()
    except Exception as e:
        error_msg = str(e)
    if not error_msg:
        return None
    decl_match = re.search(r"Declaration:\s*[\w:]+::([\w.]+)\(", error_msg)
    if not decl_match:
        return None
    rest = error_msg[decl_match.end() :]
    depth = 1
    end = len(rest) - 1
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    sig = rest[:end]
    params = _parse_npu_declaration(sig)
    if params:
        return_count = _parse_return_count_from_error(rest, end, error_msg)
        return params, "torch_npu._op()", return_count
    return None


def _parse_return_count_from_error(rest, paren_end, error_msg):
    """Parse the number of return tensors from a C++ Declaration error message.

    Handles:
      -> (Tensor, Tensor, Tensor, Tensor) → 4
      -> Tensor(a!) → 1
      -> Tensor → 1
    """
    after_paren = rest[paren_end + 1 :].strip()
    arrow_match = re.match(r"\s*->\s*(.+)", after_paren)
    if not arrow_match:
        return 0
    ret_str = arrow_match.group(1).strip()
    if ret_str.startswith("("):
        inner = ret_str[1:]
        close = inner.find(")")
        if close >= 0:
            inner = inner[:close]
        parts = [p.strip() for p in split_by_comma(inner)]
        return sum(1 for p in parts if "Tensor" in p)
    if "Tensor" in ret_str:
        return 1
    return 0


def _parse_npu_declaration(sig: str) -> Optional[List[ParamInfo]]:
    """Declaration 参数串 → List[ParamInfo]（`Type name=default`、`Type?` 可选、
    `*` 后 kw-only、`->` 返回段起跳过）。
    """
    if not sig:
        return None
    params = []
    saw_star = False
    parts = split_by_comma(sig)
    for part in parts:
        part_s = part.strip()
        if not part_s or part_s.startswith("->"):
            continue
        if part_s == "*":
            saw_star = True
            continue
        m = re.match(r"(\w+(?:\([^)]*\))?(?:\[\d*\])?)\??\s+(\w+)(?:=(.+))?", part_s)
        if m:
            type_hint = m.group(1)
            type_hint = re.sub(r"\([^)]*\)", "", type_hint)
            name = m.group(2)
            default = m.group(3) or None
            type_hint = _normalize_npu_type(type_hint)
            is_optional = default is not None or "?" in part_s.split()[0]
            if default is not None:
                default = _coerce_default_value(default, type_hint)
            params.append(
                ParamInfo(name=name, type=type_hint, default=default, is_optional=is_optional, is_keyword_only=saw_star)
            )
    return params if params else None


def _normalize_npu_type(t: str) -> str:
    """Declaration 类型词归一：Optional/List 剥壳 + npu 词表（TensorList→tuple of
    Tensors、Scalar→Number、int64[]→tuple of ints、SymInt→int…）。
    """
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_npu_type(inner)
    if t.startswith("List[") and t.endswith("]"):
        inner = t[len("List[") : -1]
        inner_mapped = _normalize_npu_type(inner)
        if inner_mapped == "Tensor":
            return "tuple of Tensors"
        if inner_mapped == "Number":
            return "tuple of Numbers"
        return f"tuple of {inner_mapped}s"
    mapping = {
        "Tensor": "Tensor",
        "TensorList": "tuple of Tensors",
        "Tensor[]": "tuple of Tensors",
        "Scalar": "Number",
        "ScalarList": "list of Numbers",
        "Scalar[]": "list of Numbers",
        "number": "Number",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "str": "str",
        "complex": "Number",
        "int?": "int",
        "float?": "float",
        "bool?": "bool",
        "str?": "str",
        "int[]": "tuple of ints",
        "float[]": "tuple of floats",
        "bool[]": "tuple of bools",
        "str[]": "tuple of strs",
        "int64[]": "tuple of ints",
        "int32[]": "tuple of ints",
        "SymInt": "int",
        "SymInt?": "int",
        "SymInt[]": "tuple of ints",
        "SymBool": "bool",
        "ScalarType": "torch.dtype",
        "ScalarType?": "torch.dtype",
        "Device": "str",
        "Device?": "str",
        "Layout": "int",
        "Layout?": "int",
        "MemoryFormat": "int",
        "MemoryFormat?": "int",
        "Generator": "None",
        "Generator?": "None",
        "AnyEnumType": "int",
        "Storage": "None",
        "Storage?": "None",
    }
    return mapping.get(t, t)
