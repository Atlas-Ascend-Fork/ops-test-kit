#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""docstring/签名文本参数解析源。

别名类型补全与 Format-2 的 inspect 富化会反向触达主模块解析入口，
因此这两个依赖在函数内惰性 import（模块级不反向依赖）。"""

import re
from typing import Dict, List, Optional, Tuple

from ..api_model import ParamInfo, resolve_function, split_by_comma
from .type_inference import (
    _coerce_default_value,
    _enrich_types_from_annotations,
    _infer_type_from_name,
    _infer_type_from_value,
    _is_name_inferred_type,
    _normalize_args_type,
)


def _parse_params_from_signature(sig: str) -> Optional[List[ParamInfo]]:
    """签名参数串 → List[ParamInfo]（docstring 首行 / TypeError 报错共用）。

    支持 "Type name" / "name" / "name=default" 三形态；`*` 起置 kw-only、`/` 跳过；
    未知类型按参数名推断、默认值串按推断类型还原。空/纯空白返回 None。
    """
    if not sig or sig.isspace():
        return None
    sig = sig.strip()
    if sig.startswith("(") and sig.endswith(")"):
        sig = sig[1:-1]
    params = []
    parts = split_by_comma(sig)
    keyword_only = False
    for part in parts:
        part_s = part.strip()
        if not part_s or part_s == "/":
            continue
        if part_s in ("*", "\\*"):
            keyword_only = True
            continue
        default = None
        is_optional = False
        is_var_positional = False
        if "=" in part_s:
            main_part, default = part_s.split("=", 1)
            main_part = main_part.strip()
            default = default.strip()
            is_optional = True
        else:
            main_part = part_s
        if re.search(r"Optional\[|\|\s*None\b", part_s):
            is_optional = True
        # Detect *args (VAR_POSITIONAL) prefix, e.g. "*operands" or "*tensors"
        if main_part.startswith("*") and not main_part.startswith("**"):
            is_var_positional = True
            main_part = main_part[1:].strip()
        tokens = main_part.split()
        if len(tokens) >= 2 and not tokens[0].startswith('"'):
            first = tokens[0]
            if ":" in first:
                name = first.split(":")[0]
                type_hint = main_part.split(":", 1)[1].strip()
                type_hint = _normalize_doc_type(type_hint)
                if not type_hint:
                    type_hint = _infer_type_from_value(default) if default is not None else ""
                    if not type_hint or type_hint == "Optional[Tensor]":
                        type_hint = _infer_type_from_name(name)
            else:
                type_hint = tokens[0]
                name = tokens[-1].rstrip("?)")
                if type_hint in ("tuple", "list") and len(tokens) >= 3:
                    type_hint = f"{tokens[0]} of {tokens[2]}"
        elif len(tokens) == 1:
            tok = tokens[0].strip("\"'")
            if ":" in tok:
                name, type_hint = tok.split(":", 1)
                name = name.strip()
                type_hint = type_hint.strip()
                type_hint = _normalize_doc_type(type_hint)
                if not type_hint:
                    # Priority: default value inference > name inference
                    # _infer_type_from_value returns empty string for unrecognizable
                    # values (e.g., 'torch.xxx') and 'Optional[Tensor]' for None —
                    # in both cases, fall back to name-based inference.
                    type_hint = ""
                    if default is not None:
                        type_hint = _infer_type_from_value(default)
                    if not type_hint or type_hint == "Optional[Tensor]":
                        type_hint = _infer_type_from_name(name)
            else:
                name = tok
                # Priority: default value inference > name inference
                type_hint = ""
                if default is not None:
                    type_hint = _infer_type_from_value(default)
                if not type_hint or type_hint == "Optional[Tensor]":
                    type_hint = _infer_type_from_name(name)
        else:
            continue
        if default is not None and isinstance(default, str):
            value_type = _infer_type_from_value(default)
            if type_hint == "Tensor" and value_type not in ("Tensor", "Optional[Tensor]", "tuple of Tensors"):
                type_hint = value_type
            default = _coerce_default_value(default, type_hint)
        # Strip trailing '?' from type — it indicates optional in some
        # docstring formats (e.g., 'str?', 'int[]?'). Mark as optional.
        if type_hint.endswith("?"):
            type_hint = type_hint[:-1]
            is_optional = True
        params.append(
            ParamInfo(
                name=name,
                type=type_hint,
                default=default,
                is_optional=is_optional,
                is_keyword_only=keyword_only,
                is_var_positional=is_var_positional,
            )
        )
    return params if params else None


def _normalize_union_members(members_str: str, exclude_tensors: bool) -> str:
    """联合成员归一：未知名（_complex 等）剔除、去重保序、'|' 拼接。

    exclude_tensors=True 用于 Union[...]（Tensor 重载成员由 TypeError 解析覆盖）；
    False 用于 '|' 联合——它是参数自身的显式注解，Tensor 成员保留。
    """
    tensor_members = {"Tensor", "tuple of Tensors", "list of Tensors", "List[Tensor]", "Tensor[]"}
    members = []
    for member in split_by_comma(members_str):
        member_s = member.strip()
        # 归一化前按原文排除 Tensor 类成员：Tuple[Tensor, ...] 经 _normalize_args_type
        # 会退化成 tuple of ints，黑名单拦不住
        if exclude_tensors and "Tensor" in member_s:
            continue
        normalized = _normalize_args_type(member_s)
        if not normalized or normalized in members:
            continue
        if exclude_tensors and normalized in tensor_members:
            continue
        members.append(normalized)
    return "|".join(members) if members else ""


def _normalize_doc_type(type_str: str) -> str:
    """Normalize type annotations from docstrings.

    Examples: Tensor, Optional[Tensor], List[Tensor], Union[Number, Tensor], int, float, bool, str
    """
    type_str = type_str.strip()
    if not type_str:
        return ""
    none_stripped = re.sub(r"\s*\|\s*None\b\s*$", "", type_str).strip()
    if none_stripped != type_str:
        return _normalize_doc_type(none_stripped)
    if type_str.startswith("Optional["):
        inner = type_str[len("Optional[") : -1]
        return _normalize_doc_type(inner)
    if type_str.startswith("List["):
        inner = type_str[len("List[") : -1]
        normalized_inner = _normalize_doc_type(inner)
        return f"List[{normalized_inner}]" if normalized_inner else ""
    if type_str.startswith("Union[") and type_str.endswith("]"):
        return _normalize_union_members(type_str[len("Union[") : -1], exclude_tensors=True)
    if "|" in type_str:
        return _normalize_union_members(type_str.replace("|", ","), exclude_tensors=False)
    return type_str


def _parse_docstring_args_section(doc: str) -> Dict[str, Tuple[str, bool]]:
    """Parse param types from docstring Args/Keyword arguments sections.

    Matches lines like:
        input (Tensor): the input tensor.
        other (Tensor or Number): ...
        out (Tensor, optional): ...

    Returns dict of {param_name: (raw_type_string, is_optional)}.
    """
    if not doc:
        return {}
    type_map = {}
    in_args = False
    args_indent = 0
    args_terminators = {
        "Returns:",
        "Example::",
        "Examples::",
        "Raises:",
        "Note:",
        "Notes:",
        "Warning:",
        "Warnings:",
        "See Also:",
        "References::",
        "References:",
        "Todo:",
    }
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Keyword arguments:", "Keyword Args:", "Keyword args:", "Kwargs:"):
            in_args = True
            args_indent = len(line) - len(line.lstrip())
            continue
        if not in_args:
            continue
        if (
            stripped in args_terminators
            or stripped.startswith(tuple(args_terminators))  # noqa: PIE810 保持
            or (not stripped.startswith((" ", "\t", "*", ".")) and stripped.endswith(":") and stripped == line.rstrip())
        ):
            break
        # Sphinx directives like ".. versionchanged::" appear inside Args
        # sections but should not terminate parsing — just skip them.
        if stripped.startswith(".."):
            continue
        if not stripped:
            continue
        line_indent = len(line) - len(line.lstrip())
        # Skip continuation lines: indent strictly deeper than the first
        # parameter definition level (args_indent + 4).  Using relative
        # indentation instead of a hardcoded threshold (>= 12) ensures
        # correctness for deeply-indented docstrings (e.g. torch.meshgrid).
        if line_indent > args_indent + 4:
            continue
        m = re.match(r"\*{0,2}(\w+)\s*\(([^)]+)\).*:", stripped)
        if m:
            name = m.group(1)
            type_str = m.group(2).strip()
            is_opt = "optional" in type_str.lower()
            type_str = re.sub(r",\s*optional\s*$", "", type_str, flags=re.IGNORECASE)
            type_str = re.sub(r",\s*required\s*$", "", type_str, flags=re.IGNORECASE)
            if type_str:
                type_map[name] = (type_str, is_opt)
            continue
        m = re.match(r"\*{0,2}(\w+)\s*:\s*(.+)", stripped)
        if m:
            name = m.group(1)
            desc = m.group(2).strip()
            if name.lower().startswith(("default", "attr", "note", "example", "deprecated", "version", "see", "if")):
                continue
            if desc:
                type_map.setdefault(name, (desc, False))
    return type_map


def _parse_return_count_from_first_line(first_line: str) -> int:
    """Extract the number of return tensors from a docstring first-line ``->`` annotation.

    Examples::

        "func(...) -> Tensor"                          → 1
        "func(...) -> (Tensor, LongTensor)"            → 2
        "func(...) -> (Tensor, Tensor, Tensor)"        → 3
        "func(...) -> None"                             → 0
        "func(...) -> torch.return_types.sort"          → 0  (unresolvable custom type)
        "func(...)"                                     → 0  (no annotation)
    """
    m = re.search(r"\)\s*->\s*(.+)$", first_line)
    if not m:
        return 0
    ret = m.group(1).strip()
    if ret.startswith("(") and ret.endswith(")"):
        parts = [p.strip() for p in ret[1:-1].split(",") if p.strip()]
        return len(parts)
    if ret.lower() in ("none", ""):
        return 0
    # Single named type like "Tensor" → 1; unresolvable types like
    # "torch.return_types.sort" → 0 (we cannot count fields).
    if re.match(r"^[A-Za-z_]\w*$", ret):
        return 1
    return 0


def _resolve_alias_from_docstring(api_name: str) -> Optional[str]:
    """从 docstring 的 "Alias for :func:`xxx`" 引用解析别名目标名。"""
    doc = getattr(resolve_function(api_name), "__doc__", "") or ""
    m = re.search(r"Alias for :func:`([^`]+)`", doc)
    if m:
        return m.group(1)
    return None


def _extract_params_from_docstring(obj, api_name: str) -> Optional[Tuple[List[ParamInfo], str, int]]:
    """Parse params from __doc__, supporting three docstring formats:

    Format 1: first line is a signature like "func(input, other, *, out=None) -> Tensor"
              → parse first line for names/defaults, enrich types from Args section
    Format 2: first line is description, but Args section has typed params like "input (Tensor): ..."
              → extract names/types directly from Args section
    Format 3: first line has inline type annotations like "func(input: Tensor, *, out: Optional[Tensor])"
              → parse types directly from first line, enrich from Args section

    Returns (params, source, return_count) where return_count is the number of return
    tensors parsed from the ``->`` annotation (0 when absent or unresolvable).
    """
    doc = getattr(obj, "__doc__", None)
    if not doc:
        return None

    args_types = _parse_docstring_args_section(doc)
    first_line = doc.strip().split("\n")[0].strip()
    m = re.match(r"[\w.]+\((.+?)\)\s*(?:->.*)?$", first_line)

    if m:
        # Format 1 or 3: first line contains a signature
        sig = m.group(1)
        params = _parse_params_from_signature(sig)
        if params:
            # Priority 1: Args/Keyword Args section types
            if args_types:
                for p in params:
                    if p.name in args_types:
                        raw_type, is_opt = args_types[p.name]
                        normalized = _normalize_args_type(raw_type)
                        if normalized:
                            p.type = normalized
                        elif any(kw in raw_type.lower() for kw in ("list", "tuple")):
                            elem_type = _infer_type_from_name(p.name)
                            p.type = f"tuple of {elem_type}s"
                        if is_opt:
                            p.is_optional = True
            # Priority 3: alias API types (for params with inferred types only)
            alias_name = _resolve_alias_from_docstring(api_name)
            if alias_name:
                from .param_extractor import extract_api_params

                alias_result = extract_api_params(alias_name)
                if alias_result:
                    alias_map = {p.name: p for p in alias_result.params if p.type}
                    for p in params:
                        if p.name in alias_map and _is_name_inferred_type(p.type):
                            p.type = alias_map[p.name].type
            return (
                params,
                "docstring(first-line+Args)" if args_types else "docstring(first-line)",
                _parse_return_count_from_first_line(first_line),
            )

    # Format 2: no signature in first line, but Args section has typed params
    # Only use Args-only if at least some params had (Type) bracket format.
    # If all params were parsed from "name: description" fallback (no brackets),
    # the types are unreliable — let inspect.signature take priority.
    if args_types:
        has_bracket_types = any(
            _normalize_args_type(raw_type) not in (None, "") for _, (raw_type, _) in args_types.items()
        )
        if not has_bracket_types:
            return None
        params = []
        for name, (raw_type, is_opt) in args_types.items():
            normalized = _normalize_args_type(raw_type)
            params.append(
                ParamInfo(
                    name=name,
                    type=normalized or _infer_type_from_name(name),
                    default=None,
                    is_optional=is_opt or normalized in ("Layout", "Device", "Dtype"),
                )
            )
        if params:
            # Format 2 has no signature line, so is_optional and default come
            # only from the Args section's ", optional" marker.  Many params
            # have default values in the actual Python signature (e.g.
            # ``size=None``) that the docstring omits the optional tag for.
            # Fall back to inspect.signature to enrich these fields.
            from .param_extractor import _extract_params_from_inspect

            inspect_result = _extract_params_from_inspect(obj)
            if inspect_result:
                _enrich_types_from_annotations(params, inspect_result[0])
            return params, "docstring(Args-only)", 0

    return None
