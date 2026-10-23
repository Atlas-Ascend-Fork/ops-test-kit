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
torch 参数解析器（torch/ 域）。

Multi-strategy fallback:
  1. TypeError fallback — call with wrong args, parse error message
  2. Manual override config — for APIs that can't be auto-parsed or have incorrect parsing

Returns APIParamInfo with typed parameter list including Tensor/List[Tensor]/Scalar/etc.
数据模型（ParamInfo/OverloadInfo/APIParamInfo/resolve_function）见 framework_api/api_model.py。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from ..api_model import (
    APIParamInfo,
    OverloadTensorLayout,
    ParamInfo,
    resolve_function,
    split_by_comma,
)
from .npu_declaration import _extract_params_from_op_declaration, _normalize_npu_type
from .pyi_loader import _extract_params_from_pyi
from .type_inference import _coerce_default_value, _infer_type_from_name, _infer_type_from_value, _normalize_args_type


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


def _extract_params_from_type_error(obj) -> Optional[Tuple[List[ParamInfo], str, List[List[ParamInfo]]]]:
    """Parse TypeError from calling obj() with wrong args.

    Returns (primary_params, source, all_overloads).
    For multi-overload APIs, all_overloads contains all parsed signatures.
    For single-signature APIs, all_overloads is empty (use primary_params).
    """
    try:
        obj()
    except TypeError as e:
        error_msg = str(e)
        if "expected one of:" in error_msg:
            all_overloads = []
            for line in error_msg.split("\n"):
                line_s = line.strip()
                if line_s.startswith("*"):
                    sig = line_s[1:].strip()
                    if sig.startswith("(") and sig.endswith(")"):
                        sig = sig[1:-1]
                    parsed = _parse_params_from_signature(sig)
                    if parsed:
                        all_overloads.append(parsed)
            if all_overloads:
                return all_overloads[0], f"TypeError(multi-overload, {len(all_overloads)} signatures)", all_overloads
        m = re.search(r"expected\s*\(([^)]+(?:\([^)]*\))*[^)]*)\)", error_msg)
        if m:
            sig = m.group(1)
            params = _parse_params_from_signature(sig)
            if params:
                return params, "TypeError(single-signature)", []
    except Exception as _e:
        logging.debug("param extract skipped: %s", _e)
    return None


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
                alias_result = extract_api_params(alias_name)
                if alias_result:
                    alias_map = {p.name: p for p in alias_result.params if p.type}
                    for p in params:
                        if p.name in alias_map and _is_inferred_type(p.type):
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
            inspect_result = _extract_params_from_inspect(obj)
            if inspect_result:
                _enrich_types_from_annotations(params, inspect_result[0])
            return params, "docstring(Args-only)", 0

    return None


def _extract_params_from_aten_schemas(
    api_name: str,
) -> Optional[Tuple[List[ParamInfo], str, List[List[ParamInfo]], List[int]]]:
    """从 torch 算子注册表解析（链上第一优先级源，FunctionSchema 权威元数据）。

    OpOverloadPacket 带复数 _schemas；api_name 钉死具名重载时 resolve_function 返回
    OpOverload 只有单数 _schema（不取则退 inspect 得 (*args, **kwargs)）。钉死重载时
    只收该重载——防整个 packet 混入 *_out 变体（is_out_required 会让按返回值取输出的
    用例误判 MISSING_REQUIRED_OUTPUT）。返回 (主参数表, source, 全重载, 返回数列表)。
    """
    obj = resolve_function(api_name)
    if obj is None:
        return None
    if hasattr(obj, "_schemas"):
        schemas = obj._schemas
    elif hasattr(obj, "_schema"):
        # api_name 钉死具名重载时 resolve_function 返回 OpOverload, 它只有单数 _schema;
        # 不取这一支会退到 inspect.signature, 而 OpOverload.__call__ 的签名是
        # (*args, **kwargs), 只能解析出一个 "args", 下游按形参名派发时长度对不上。
        schemas = {api_name.rsplit(".", 1)[-1]: obj._schema}
    else:
        return None
    if not schemas:
        return None
    # api_name 若钉死了具体重载(如 torch.ops.aten._foreach_addcmul.Tensor)，只解析该重载。
    # resolve_function 会退回到 OpOverloadPacket(丢掉重载后缀)，若不在此过滤，整个
    # packet 的重载都会被收进来——其中的 *_out 变体带 is_out_required=True，会让本来
    # 靠返回值取输出的用例被误判为 MISSING_REQUIRED_OUTPUT；多重载并存也会导致
    # CASE_FIELD_AMBIGUOUS。
    overload_tail = api_name.rsplit(".", 1)[-1]
    if overload_tail in schemas:
        schemas = {overload_tail: schemas[overload_tail]}
    all_overloads = []
    return_counts = []
    for schema in schemas.values():
        params = _parse_aten_schema_arguments(schema)
        if params:
            all_overloads.append(params)
            return_counts.append(len(schema.returns) if hasattr(schema, "returns") else 0)
    if not all_overloads:
        return None
    primary = max(all_overloads, key=lambda ov: sum(1 for p in ov if p.is_optional))
    _first_schema = next(iter(schemas.values()))
    namespace = getattr(_first_schema, "name", api_name.rsplit(".", 1)[0]).split("::")[0]
    source = (
        f"{namespace}._schemas({len(all_overloads)} overloads)" if len(all_overloads) > 1 else f"{namespace}._schemas"
    )
    return primary, source, all_overloads, return_counts


def _parse_aten_schema_arguments(schema) -> Optional[List[ParamInfo]]:
    """FunctionSchema.arguments → List[ParamInfo]（默认值 / Optional[...] 可选性 / kwarg_only）。

    写别名（alias_info.is_write，如 add_ 的 Tensor(a!) self）逐参数落 is_mutable ——
    参数级可变性与签名同源，解析期即确定。
    """
    params = []
    for arg in schema.arguments:
        name = arg.name
        if not name:
            continue
        type_str = _normalize_aten_schema_type(str(arg.type))
        default = None
        is_optional = False
        if arg.has_default_value():
            default = arg.default_value
            is_optional = True
            if isinstance(default, (list, tuple)):
                default = list(default)
        elif str(arg.type).startswith("Optional["):
            is_optional = True
        is_keyword_only = arg.kwarg_only if hasattr(arg, "kwarg_only") else False
        alias = getattr(arg, "alias_info", None)
        params.append(
            ParamInfo(
                name=name,
                type=type_str,
                default=default,
                is_optional=is_optional,
                is_keyword_only=is_keyword_only,
                is_mutable=bool(alias and alias.is_write),
            )
        )
    return params if params else None


def _normalize_aten_schema_type(t: str) -> str:
    """schema 类型词归一：Optional[...] 剥壳、List[...] → tuple of xxx，余者走 npu 词表。"""
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_aten_schema_type(inner)
    if t.startswith("List[") and t.endswith("]"):
        inner = t[len("List[") : -1]
        inner_mapped = _normalize_aten_schema_type(inner)
        if inner_mapped == "Tensor":
            return "tuple of Tensors"
        return f"tuple of {inner_mapped}s"
    return _normalize_npu_type(t)


_SCALAR_ANNOTATION_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}

_SEQUENCE_INNER_TYPES = {"Tensor": "tuple of Tensors", "int": "tuple of ints", "float": "tuple of floats"}


def _torch_tensor_type():
    """torch.Tensor;torch 不可用时返回一个不会与任何注解相等的哨兵。"""
    try:
        import torch

        return torch.Tensor
    except ImportError:
        return object()


def _torch_dtype_type():
    """torch.dtype;torch 不可用时返回一个不会与任何注解相等的哨兵。"""
    try:
        import torch

        return torch.dtype
    except ImportError:
        return object()


def _sequence_annotation_to_type(annotation) -> str:
    """list/tuple/Sequence[...] 按其元素类型归一成 tuple of xxx。"""
    args = getattr(annotation, "__args__", ())
    if not args:
        return "int"
    return _SEQUENCE_INNER_TYPES.get(_annotation_to_type(args[0]), "int")


def _annotation_to_type(annotation) -> str:
    """Convert a type annotation object to our type string.

    Returns empty string for None or unrecognized annotations.
    """
    if annotation is None:
        return ""
    import collections.abc
    import types as _types
    import typing

    # PEP 604 的 `str | None` 是 types.UnionType,没有 __origin__(torch>=2.6 的
    # .pyi/源码普遍改用这种写法),故 __origin__ 取不到时用 get_origin 兜底。
    origin = getattr(annotation, "__origin__", None) or typing.get_origin(annotation)
    # 不能写成 origin in (..., getattr(_types, 'UnionType', None)):Python < 3.10 无 UnionType,
    # getattr 返回 None,而无 origin 的注解(int/float/str/bool)其 origin 恰为 None,
    # 会整片落进 Union 分支、args 为空后误返回 "Tensor"。
    if origin in (typing.Optional, typing.Union) or (hasattr(_types, "UnionType") and origin is _types.UnionType):
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        return _annotation_to_type(non_none[0]) if non_none else "Tensor"
    # 序列注解识别:list/tuple 的 origin 即 list/tuple(typing.List/Tuple 亦然),
    # 而 typing.Sequence[T] 与 collections.abc.Sequence[T] 的 origin 是 collections.abc.Sequence。
    if origin in (list, tuple, collections.abc.Sequence):
        return _sequence_annotation_to_type(annotation)
    scalar = _SCALAR_ANNOTATION_TYPES.get(annotation)
    if scalar:
        return scalar
    try:
        if isinstance(annotation, type) and issubclass(annotation, int):
            return "int"
    except TypeError:
        pass  # noqa: S110 协议探测：失败跳过属预期
    if annotation is _torch_dtype_type():
        return "Dtype"
    if annotation is _torch_tensor_type():
        return "Tensor"
    return ""


def _extract_params_from_annotations(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Extract params from obj.__annotations__.

    Works for any Python object that has __annotations__ (functions, classes, etc).
    Does NOT require inspect.signature — safe to call on any object.
    Default values are obtained from inspect.signature if available, otherwise None.
    """
    ann = getattr(obj, "__annotations__", None)
    if not ann:
        return None

    param_names = [k for k in ann if k != "return"]
    if not param_names:
        return None

    import inspect as _inspect

    defaults_map = {}
    kw_only_set = set()
    try:
        sig = _inspect.signature(obj)
        for pname, param in sig.parameters.items():
            if param.default != _inspect.Parameter.empty:
                defaults_map[pname] = param.default
            if param.kind == _inspect.Parameter.KEYWORD_ONLY:
                kw_only_set.add(pname)
    except (ValueError, TypeError):
        pass  # noqa: S110 协议探测：失败跳过属预期

    params = []
    for pname in param_names:
        type_hint = _annotation_to_type(ann[pname])
        if not type_hint:
            type_hint = _infer_type_from_name(pname)
        default_raw = defaults_map.get(pname)
        is_optional = pname in defaults_map
        default = _coerce_default_value(str(default_raw), type_hint) if is_optional else None
        params.append(
            ParamInfo(
                name=pname,
                type=type_hint,
                default=default,
                is_optional=is_optional,
                is_keyword_only=pname in kw_only_set,
            )
        )

    return (params, "__annotations__") if params else None


def _extract_params_from_inspect(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Try inspect.signature for Python-callable functions."""
    import inspect as _inspect

    try:
        sig = _inspect.signature(obj)
    except (ValueError, TypeError):
        return None

    ann = getattr(obj, "__annotations__", None) or {}
    params = []
    for pname, param in sig.parameters.items():
        if param.kind == _inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == _inspect.Parameter.VAR_POSITIONAL:
            type_hint = _infer_type_from_name(pname)
            if type_hint in ("Tensor", "list of Tensors", "tuple of Tensors", "Tensor[]", "List[Tensor]"):
                params.append(
                    ParamInfo(name=pname, type="Tensor", default=None, is_optional=False, is_var_positional=True)
                )
            continue

        if pname in ann:
            type_hint = _annotation_to_type(ann[pname])
        elif param.annotation is not None and param.annotation != _inspect.Parameter.empty:
            type_hint = _annotation_to_type(param.annotation)
        else:
            type_hint = _infer_type_from_name(pname)
        if _is_inferred_type(type_hint) and param.default != _inspect.Parameter.empty:
            if isinstance(param.default, bool):
                type_hint = "bool"
            elif isinstance(param.default, int):
                type_hint = "int"
            elif isinstance(param.default, float):
                type_hint = "float"
            elif isinstance(param.default, str):
                type_hint = "str"
        is_optional = param.default != _inspect.Parameter.empty
        default = None if not is_optional else _coerce_default_value(str(param.default), type_hint)
        # In Python, parameters after *args are implicitly keyword-only
        # (e.g. meshgrid(*tensors, indexing=None)).  Mapping this correctly
        # ensures build_args() routes such params to kwargs instead of args.
        is_kw_only = param.kind == _inspect.Parameter.KEYWORD_ONLY
        params.append(
            ParamInfo(name=pname, type=type_hint, default=default, is_optional=is_optional, is_keyword_only=is_kw_only)
        )
    return (params, "inspect.signature") if params else None


def _is_inferred_type(type_str: str) -> bool:
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
        if _is_inferred_type(p.type) and "|" not in p.type and ann_p.type:
            p.type = ann_p.type
        if p.default is None and ann_p.default is not None:
            p.default = ann_p.default
        if not p.is_optional and ann_p.is_optional:
            p.is_optional = ann_p.is_optional
    return params


def _strip_inplace_if_separate_exists(api_name: str, params: List[ParamInfo]) -> List[ParamInfo]:
    """torch.xxx_ 存在独立原地 API 时移除 inplace 参数（原地变体单独测试的设计约定）。"""
    if not api_name.startswith("torch.") or api_name.startswith("torch.nn."):  # noqa: PIE810 保持
        return params
    if api_name.endswith("_"):
        params[:] = [p for p in params if p.name != "inplace"]
        return params
    inplace_name = api_name + "_"
    inplace_obj = resolve_function(inplace_name)
    if inplace_obj is None:
        return params
    params[:] = [p for p in params if p.name != "inplace"]
    return params


# Step 4 kw-only 校验的类型修正探测表（窄表：仅标量类型互转）
_KW_ONLY_ALT_PROBES = {
    "str": [("int", 1)],
    "int": [("str", "mean")],
    "Dtype": [("int", 0)],
    "torch.dtype": [("int", 0)],
    "bool": [("str", "mean")],
}

# Step 3 fallback 可选参数校验的探测表（宽表：含 Tensor 类型名互转）
_OPTIONAL_ALT_PROBES = {
    "str": [("int", 1)],
    "int": [("str", "mean")],
    "Dtype": [("int", 0)],
    "torch.dtype": [("int", 0)],
    "bool": [("str", "mean")],
    "torch.Tensor": "Tensor",
    "torch.BoolTensor": "Tensor",
    "torch.IntTensor": "Tensor",
    "torch.LongTensor": "Tensor",
    "torch.FloatTensor": "Tensor",
    "torch.DoubleTensor": "Tensor",
}


def _build_probes(params: List[ParamInfo]) -> Optional[list]:
    """按参数类型构造探测值列表；遇到未知类型返回 None。"""
    import torch as _torch

    _PROBE_MAP = {
        "int": 0,
        "float": 1.0,
        "Number": 1.0,
        "bool": False,
        "str": "mean",
        "torch.dtype": _torch.float32,
        "Dtype": _torch.float32,
        "torch.memory_format": _torch.contiguous_format,
    }

    dummy = _torch.zeros(2, 3)
    probes = []
    for p in params:
        if p.is_tensor_like:
            probes.append(dummy)
        else:
            val = _PROBE_MAP.get(p.type)
            if val is None:
                return None
            probes.append(val)
    return probes


def _extract_cpp_param_names(obj) -> Optional[set]:
    """从 "missing N required positional" TypeError 提取 C++ 实参名集。"""
    try:
        obj()
    except TypeError as e:
        msg = str(e)
        m = re.search(r"missing \d+ required positional arguments?: (.+)", msg)
        return {n.strip().strip("\"'") for n in m.group(1).split(",")} if m else None
    except Exception:
        return None


def _detect_max_positional(obj, probes, param_count: int) -> int:
    """渐进调用探测最大位置参数边界。

    已知权衡：任意异常（"error"）按"执行失败≠签名拒绝"计为有效边界（与
    _fix_param_type_by_alt_probes 的探测协议一致），设备级错误会污染判定——
    探测仅发生在解析期，调用方须保证探测环境可用。"""

    def _classify(n):
        try:
            obj(*probes[:n])
            return "ok"
        except TypeError as e:
            msg = str(e).lower()
            if "too many" in msg or ("takes" in msg and "positional" in msg):
                return "too_many"
            if "invalid combination" in msg:
                return "too_many"
            if "missing" in msg:
                return "missing"
            return "error"
        except Exception:
            return "error"

    max_pos = 0
    found_valid = False
    for n in range(param_count + 1):
        result = _classify(n)
        if result in ("ok", "error"):
            max_pos = n
            found_valid = True
        elif result == "missing":
            continue
        elif result == "too_many" and found_valid:
            break
    return max_pos


def _first_optional_index(params: List[ParamInfo]) -> int:
    for i, p in enumerate(params):
        if p.is_optional:
            return i
    return len(params)


def _remove_deprecated_loss_params(params: List[ParamInfo], cpp_param_set: set) -> set:
    """剔除 C++ 签名中不存在的废弃 loss 参数（size_average/reduce），返回剔除名集。"""
    _DEPRECATED_LOSS_PARAMS = {"size_average": "bool", "reduce": "bool"}
    to_remove = []
    removed_names = set()
    for i, p in enumerate(params):
        if (
            p.name in _DEPRECATED_LOSS_PARAMS
            and _DEPRECATED_LOSS_PARAMS[p.name] == p.type
            and p.name not in cpp_param_set
        ):
            to_remove.append(i)
            removed_names.add(p.name)
    for idx in reversed(to_remove):
        params.pop(idx)
    return removed_names


def _fix_param_type_by_alt_probes(obj, call_args, p: ParamInfo, alt_probes) -> bool:
    """按 alt 探测表尝试修正参数类型/默认值；返回是否修正成功。

    TypeError 视为该 alt 值不被接受（换下一个）；其他异常视为"接受但执行
    失败"，同样完成类型修正（与原始探测协议一致）。
    """
    fixed = False
    for alt_type, alt_val in alt_probes.get(p.type, []):
        try:
            obj(*call_args, **{p.name: alt_val})
            p.type = alt_type
            p.default = alt_val
            fixed = True
            break
        except TypeError:
            continue
        except Exception:
            p.type = alt_type
            p.default = alt_val
            fixed = True
            break
    return fixed


def _verify_keyword_only_params(obj, params: List[ParamInfo], probes, max_pos: int):
    """验证 kw-only 参数确实被 C++ 接受；类型不符尝试 alt 探测修正，仍失败则剔除。"""
    kw_only_params = [(i, params[i], probes[i]) for i in range(len(params)) if params[i].is_keyword_only]
    if not kw_only_params:
        return

    base = probes[: _first_optional_index(params)]

    to_remove = []
    for idx, p, val in kw_only_params:
        try:
            obj(*base, **{p.name: val})
        except TypeError as _te:
            _te_msg = str(_te).lower()
            if "unexpected keyword" not in _te_msg:
                continue
            if not _fix_param_type_by_alt_probes(obj, base, p, _KW_ONLY_ALT_PROBES):
                to_remove.append(idx)
        except Exception as _e:
            logging.debug("param extract skipped: %s", _e)
    for idx in reversed(to_remove):
        params.pop(idx)


def _verify_optional_params_by_kwargs(obj, params: List[ParamInfo], probes, max_pos: int):
    """kwarg 探测验证可选参数（Step 3 fallback）；剔除后经约定恢复（size_average→reduction）。"""
    first_optional = _first_optional_index(params)
    if first_optional >= len(params):
        return

    kw_params = [(i, params[i], probes[i]) for i in range(first_optional, len(params))]
    to_remove = []
    for idx, p, val in kw_params:
        test_pos = [probes[j] for j in range(max_pos) if j != idx]
        try:
            obj(*test_pos, **{p.name: val})
        except TypeError as _te:
            _te_msg = str(_te).lower()
            if "missing" in _te_msg and "required positional" in _te_msg:
                continue
            if "unexpected keyword" not in _te_msg and "got an unexpected keyword" not in _te_msg:
                continue
            if not _fix_param_type_by_alt_probes(obj, test_pos, p, _OPTIONAL_ALT_PROBES):
                to_remove.append(idx)
        except Exception as _e:
            logging.debug("param extract skipped: %s", _e)
    removed_names = set()
    for idx in reversed(to_remove):
        removed_names.add(params[idx].name)
        params.pop(idx)

    if removed_names:
        _try_recover_removal(obj, probes[:first_optional], params, removed_names)


def _detect_keyword_only_by_probing(obj, params: List[ParamInfo]):
    """Probe a C++ builtin to detect keyword-only boundary and validate param names.

    Strategy:
    1. Extract actual C++ param names from "missing N required positional" TypeError
    2. Detect max positional boundary by progressive calling
    3. Remove params not in C++ signature, recover via convention (size_average→reduction)

    Only called for C++ builtins (no inspect.signature available).
    """
    if not params or any(p.is_keyword_only for p in params):
        return

    probes = _build_probes(params)
    if probes is None:
        return

    # Step 1: Extract actual C++ param names from TypeError
    cpp_param_set = _extract_cpp_param_names(obj)

    # Step 2: Detect max positional boundary
    max_pos = _detect_max_positional(obj, probes, len(params))
    for i in range(max_pos, len(params)):
        params[i].is_keyword_only = True

    # Step 3: Remove deprecated params not in C++ signature (if we have ground truth)
    if cpp_param_set is not None:
        removed_names = _remove_deprecated_loss_params(params, cpp_param_set)
        if removed_names:
            _try_recover_removal(obj, probes[: _first_optional_index(params)], params, removed_names)

    # Step 4: Verify keyword-only params are actually accepted by C++.
    # 无 kw-only 参数时提前返回——原逻辑此时整个探测收尾，fallback 不再执行。
    if not any(p.is_keyword_only for p in params):
        return
    _verify_keyword_only_params(obj, params, probes, max_pos)

    # Step 3 fallback: verify optional params by kwarg probing
    _verify_optional_params_by_kwargs(obj, params, probes, max_pos)


def _try_recover_removal(obj, pos_probes, params, removed_names):
    """probing 剔除参数后的恢复：已废弃名（size_average/reduce）映射现代等价 reduction——
    已存在则改写其类型/默认；不存在则探调验证可接受后按 kw-only 补插。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return

    _DEPRECATED_TO_MODERN = {
        "size_average": ("reduction", "int", 1),
        "reduce": ("reduction", "int", 1),
    }

    for old_name in removed_names:
        if old_name not in _DEPRECATED_TO_MODERN:
            continue
        new_name, new_type, new_default = _DEPRECATED_TO_MODERN[old_name]
        existing = next((p for p in params if p.name == new_name), None)
        if existing is not None:
            existing.type = new_type
            existing.default = new_default
            existing.is_keyword_only = True
            continue
        try:
            obj(*pos_probes, **{new_name: new_default})
        except TypeError:
            continue
        except Exception as _e:
            logging.debug("param extract skipped: %s", _e)
        existing_kw = [p for p in params if p.is_keyword_only]
        insert_idx = len(params)
        if existing_kw:
            insert_idx = params.index(existing_kw[0])
        params.insert(
            insert_idx,
            ParamInfo(name=new_name, type=new_type, default=new_default, is_optional=True, is_keyword_only=True),
        )


def _strip_alias_extra_params(api_name: str, params: List[ParamInfo]) -> List[ParamInfo]:
    """Post-process parsed params for C++ builtins.

    1. Remove 'inplace' if a separate inplace API exists (design: test inplace separately).
    2. Remove params whose type resolved to 'name' (named-dimension artefact).
    3. For C++ builtins: probe to detect keyword-only boundary and remove
       params the C++ function does not accept at all.
    """
    _strip_inplace_if_separate_exists(api_name, params)

    if not api_name.startswith("torch."):
        return params

    params[:] = [p for p in params if p.type != "name"]

    obj = resolve_function(api_name)
    is_cpp = obj is not None and type(obj).__name__ == "builtin_function_or_method"

    if is_cpp:
        _detect_keyword_only_by_probing(obj, params)

    return params


def _extract_params_from_functional_alias(api_name: str) -> Optional[Tuple[List[ParamInfo], str]]:
    """矩阵第⑦级后备：本体失败时按别名候选解析（候选侧跑完整 D/T/A 裁决）。

    候选三路：torch.X → torch.nn.functional.X；npu_X → torch_npu.X；原地 X_ →
    去下划线的 torch.X / F.X。source 带 alias(候选, 内部胜者) 标记。
    """
    if not api_name.startswith("torch."):
        return None
    func_name = api_name.split(".")[-1]

    candidates = [f"torch.nn.functional.{func_name}"]

    if func_name.startswith("npu_"):
        candidates.append(f"torch_npu.{func_name}")

    if func_name.endswith("_") and not func_name.startswith("_"):
        base_name = func_name[:-1]
        candidates.append(f"torch.{base_name}")
        candidates.append(f"torch.nn.functional.{base_name}")

    for candidate in candidates:
        obj = resolve_function(candidate)
        if obj is None:
            continue
        doc_result = _extract_params_from_docstring(obj, candidate)
        te_result = _extract_params_from_type_error(obj)
        ann_result = _extract_params_from_annotations(obj)

        if te_result:
            params, source, overloads = te_result
            if overloads and len(overloads) > 1:
                _enrich_types_from_annotations(overloads[0], ann_result[0] if ann_result else [])
                _strip_alias_extra_params(api_name, overloads[0])
                return overloads[0], f"alias({candidate}, {source})"

        if doc_result and te_result:
            merged = _merge_docstring_and_type_error(doc_result[0], te_result[0])
            if merged:
                if ann_result:
                    _enrich_types_from_annotations(merged, ann_result[0])
                _strip_alias_extra_params(api_name, merged)
                return merged, f"alias({candidate}, merge({doc_result[1]}, {te_result[1]}))"

        if doc_result:
            params, doc_source, _ = doc_result
            if ann_result:
                _enrich_types_from_annotations(params, ann_result[0])
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, {doc_source})"

        if te_result:
            params, te_source, _ = te_result
            if ann_result:
                _enrich_types_from_annotations(params, ann_result[0])
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, {te_source})"

        if ann_result:
            params = ann_result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, __annotations__)"

        result = _extract_params_from_inspect(obj)
        if result:
            params = result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, inspect.signature)"
        result = _extract_params_from_op_declaration(candidate)
        if result:
            params = result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, torch_npu._op())"
    return None


def _merge_docstring_and_type_error(
    doc_params: List[ParamInfo], te_params: List[ParamInfo]
) -> Optional[List[ParamInfo]]:
    """Merge docstring and TypeError results as equal-priority sources.

    Rules:
    - Use the longer param list as base (more info is better).
    - For each param present in both, take the more authoritative type:
      - If one is clearly inferred (generic Tensor/int/float) and the other
        is specific (Number, torch.dtype, tuple of ints, etc.), prefer the specific one.
      - Otherwise keep the base param's type.
    - Preserve defaults from whichever source has them.
    """
    if not doc_params or not te_params:
        return doc_params or te_params

    # Build name→ParamInfo lookup for the secondary source
    base = doc_params if len(doc_params) >= len(te_params) else te_params
    other = te_params if base is doc_params else doc_params
    other_map = {p.name: p for p in other}

    merged = []
    for bp in base:
        op = other_map.get(bp.name)
        if op is None:
            merged.append(bp)
            continue

        name = bp.name
        # Pick type: prefer authoritative over inferred
        if bp.type == op.type:
            ptype = bp.type
        elif _is_inferred_type(bp.type) and not _is_inferred_type(op.type):
            ptype = op.type
        elif not _is_inferred_type(bp.type) and _is_inferred_type(op.type):
            ptype = bp.type
        else:
            # Both are specific but different — prefer TypeError (C++ ground truth)
            ptype = op.type if other is te_params else bp.type

        # Pick default: prefer non-empty
        default = bp.default if bp.default is not None else op.default

        # Pick optional: either source says optional
        is_optional = bp.is_optional or op.is_optional

        # Pick keyword_only: TypeError is authoritative for positional/keyword boundary
        is_kw = op.is_keyword_only if other is te_params else bp.is_keyword_only

        merged.append(ParamInfo(name=name, type=ptype, default=default, is_optional=is_optional, is_keyword_only=is_kw))

    # Add params only in 'other' that aren't in 'base'
    base_names = {p.name for p in base}
    for op in other:
        if op.name not in base_names:
            merged.append(op)

    return merged if merged else None


def _try_upgrade_to_inspect_if_var_pos(obj, params, api_name, ann_result=None):
    """Try replacing *params* with inspect.signature result when VAR_POSITIONAL is needed.

    Docstring and annotation sources cannot capture ``*args`` tensor parameters:
    - ``__annotations__`` omits ``*args`` entries entirely
    - Docstring Args sections list ``*tensors`` as a normal param without
      the ``is_var_positional`` flag

    When the current extraction source has no VAR_POSITIONAL tensor param
    but inspect.signature finds one, the inspect result is strictly more
    accurate and should be used instead.

    Args:
        obj: Resolved callable.
        params: Current extraction result (list of ParamInfo).
        api_name: Full API name (e.g. "torch.meshgrid").
        ann_result: Optional annotation result for type enrichment.

    Returns:
        APIParamInfo built from inspect params if an upgrade was performed,
        otherwise None.
    """
    has_var_pos = any(getattr(p, "is_var_positional", False) for p in params)
    if has_var_pos:
        return None
    inspect_result = _extract_params_from_inspect(obj)
    if not inspect_result:
        return None
    inspect_has_var_pos = any(getattr(p, "is_var_positional", False) for p in inspect_result[0])
    if not inspect_has_var_pos:
        return None
    inspect_params = inspect_result[0]
    inspect_names = {p.name for p in inspect_params}
    if inspect_names <= {"args", "kwargs"}:
        return None
    if ann_result:
        _enrich_types_from_annotations(inspect_result[0], ann_result[0])
    _strip_alias_extra_params(api_name, inspect_result[0])
    return APIParamInfo(api_name=api_name, params=inspect_result[0], source="inspect.signature")


def _try_upgrade_to_inspect_if_kw_mismatch(obj, params, api_name):
    """Upgrade to inspect.signature when keyword-only params are missing.

    Docstring-based extraction cannot detect keyword-only boundaries (the ``*``
    separator is not preserved).  If inspect.signature reports KEYWORD_ONLY
    params but the current params have none, inspect is strictly more accurate.
    """
    has_kw = any(getattr(p, "is_keyword_only", False) for p in params)
    if has_kw:
        return None
    inspect_result = _extract_params_from_inspect(obj)
    if not inspect_result:
        return None
    inspect_has_kw = any(getattr(p, "is_keyword_only", False) for p in inspect_result[0])
    if not inspect_has_kw:
        return None
    inspect_names = {p.name for p in inspect_result[0]}
    if inspect_names <= {"args", "kwargs"}:
        return None
    _strip_alias_extra_params(api_name, inspect_result[0])
    return APIParamInfo(api_name=api_name, params=inspect_result[0], source="inspect.signature")


def extract_api_params(api_name: str) -> Optional[APIParamInfo]:
    """解析入口（异常收敛）：链上任一源抛异常 → warning 日志 + None。"""
    try:
        return _extract_api_params_impl(api_name)
    except Exception as e:
        logging.warning(f"extract_api_params({api_name}) raised {type(e).__name__}: {e}")
        return None


def _extract_api_params_impl(api_name: str) -> Optional[APIParamInfo]:
    """多源解析链编排：纯路由（对象解析归各族函数），矩阵见 _combine_generic_sources。

    A schema 注册表 —— _extract_params_from_aten_schemas：resolved 对象带
                      _schemas（torch.ops.*、torch_npu.* ≥2.9）即 FunctionSchema
                      权威直出，任意 api_name 形态均先试；
    B npu 族        —— _from_npu_sources：Declaration → docstring → inspect
                      族内闭合链。torch_npu.* 三级穷尽即终止（不出族）；
                      torch.npu_*（deprecated 别名）仅 Declaration 一级；
    C 通用          —— 其余全部（含 B 出族的 torch.npu_* 别名）：
                      D/T/A 组合裁决 + 后备链（inspect → 别名 → pyi）。
    """
    # The extension package is imported by FrameworkApiInfoKeeper.get before
    # this is reached; no import here.
    result = _extract_params_from_aten_schemas(api_name)
    if result:
        params, source, overloads, return_counts = result
        return APIParamInfo(
            api_name=api_name, params=params, source=source, overloads=overloads, _return_counts=return_counts
        )

    info = _from_npu_sources(api_name)
    if info is not None:
        return info

    if api_name.startswith("torch_npu."):
        return None  # npu 族内穷尽（Declaration/docstring/inspect 全败），不出族

    return _combine_generic_sources(api_name)


def _from_npu_sources(api_name: str) -> Optional[APIParamInfo]:
    """npu 族内闭合链：Declaration → docstring → inspect（首到先得）。

    域：torch_npu.*（三级全走，穷尽返回 None，编排层终止）+ torch.npu_*
    （deprecated 别名，仅 Declaration 一级——None 后由编排层路由至通用链）。
    Declaration 入口自门控（npu_declaration.py，内部自行 resolve）；本函数
    只做族内编排与包装。
    """
    result = _extract_params_from_op_declaration(api_name)
    if result:
        params, source, return_count = result
        return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=[return_count])

    if not api_name.startswith("torch_npu."):
        return None
    obj = resolve_function(api_name)
    if obj is None:
        return None
    doc_result = _extract_params_from_docstring(obj, api_name)
    if doc_result:
        params, source, doc_rc = doc_result
        return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=[doc_rc] if doc_rc else [])
    result = _extract_params_from_inspect(obj)
    if result:
        params, source = result
        return APIParamInfo(api_name=api_name, params=params, source=source)
    return None


def _combine_generic_sources(api_name: str) -> Optional[APIParamInfo]:
    """通用家族组合裁决矩阵（D=docstring，T=TypeError，A=annotations，三者先并行采集）。

    首个命中即返回：
      ① T 多重载(>1)    → T 直出——重载集唯一来源（A 补类型）
      ② D ∧ T 单签名    → merge(D,T)：参数更长者作基座、另一侧补类型（A 再补）
      ③ D              → D（kw 键不齐/缺 *args 时升级 inspect 换源重解；A 补类型）
      ④ T 单签名        → T（A 补类型）
      ⑤ A              → A（零张量参数时升级 inspect 补 *args）
      ⑥ inspect         → 纯 Python 函数直解（C++ builtin 上为 ValueError）
      ⑦ functional 别名 → 原地/函数式变体的签名挂在别名上（已按别名侧收尾，不再 strip）
      ⑧ pyi stub        → Tensor 方法/nn.functional 的 stub 签名（多重载/单签名）

    对象解析自包含（resolve 失败即 None——编排层不再预解析传递）。
    """
    obj = resolve_function(api_name)
    if obj is None:
        return None
    doc = _extract_params_from_docstring(obj, api_name)
    te = _extract_params_from_type_error(obj)
    ann = _extract_params_from_annotations(obj)

    if te and len(te[2]) > 1:  # ①
        params, source, overloads = te
        return _finalize(api_name, overloads[0], source, ann, overloads=overloads)

    if doc and te:  # ②
        (doc_params, doc_source, doc_rc), (te_params, te_source, _) = doc, te
        merged = _merge_docstring_and_type_error(doc_params, te_params)
        if merged:
            return _finalize(
                api_name, merged, f"merge({doc_source}, {te_source})", ann, return_counts=[doc_rc] if doc_rc else []
            )

    if doc:  # ③
        params, source, doc_rc = doc
        upgraded = _try_upgrade_to_inspect_if_kw_mismatch(obj, params, api_name)
        if upgraded:
            return upgraded
        upgraded = _try_upgrade_to_inspect_if_var_pos(obj, params, api_name, ann)
        if upgraded:
            return upgraded
        return _finalize(api_name, params, source, ann, return_counts=[doc_rc] if doc_rc else [])

    if te:  # ④
        params, source, _overloads = te
        return _finalize(api_name, params, source, ann, overloads=[params])

    if ann:  # ⑤
        params, source = ann
        # __annotations__ 完全省略 *args（VAR_POSITIONAL）参数：零张量参数时
        # 升级 inspect 以捕获 *args 张量位。
        upgraded = _try_upgrade_to_inspect_if_var_pos(obj, params, api_name)
        if upgraded:
            return upgraded
        return _finalize(api_name, params, source)

    result = _extract_params_from_inspect(obj)  # ⑥
    if result:
        params, source = result
        return _finalize(api_name, params, source)

    result = _extract_params_from_functional_alias(api_name)  # ⑦
    if result:
        params, source = result
        return APIParamInfo(api_name=api_name, params=params, source=source)

    result = _extract_params_from_pyi(api_name)  # ⑧
    if result:
        params, source, overloads = result
        if len(overloads) > 1:
            return _finalize(api_name, overloads[0], source, None, overloads=overloads)
        return _finalize(api_name, params, source, None, overloads=[params])

    return None


def _finalize(
    api_name: str,
    params: List[ParamInfo],
    source: str,
    ann: Optional[Tuple[List[ParamInfo], str]] = None,
    return_counts: Optional[List[int]] = None,
    overloads: Optional[List[List[ParamInfo]]] = None,
) -> APIParamInfo:
    """裁决后统一收尾：A 侧补类型 → 别名后处理（probing 剔除）→ 构造 APIParamInfo。

    overloads 非 None 时走多重载构造（params 须为 overloads[0]）；
    否则按 return_counts（可为空）构造单签名。
    """
    if ann:
        _enrich_types_from_annotations(params, ann[0])
    _strip_alias_extra_params(api_name, params)
    if overloads is not None:
        return APIParamInfo(api_name=api_name, params=params, source=source, overloads=overloads)
    return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=return_counts or [])


_MANUAL_OVERRIDES: Dict[str, APIParamInfo] = {}

# hccl 手工表惰性装载标记（get_api_params 首查触发，进程内一次）
_HCCL_TABLE_READY = False


def _load_hccl_manual_table():
    """hccl 手工表惰性装载（幂等）。

    torch_npu 系不登记（npu._schemas 权威解析），表仅存 mindspeed 项。
    直接存全量 APIParamInfo（含 overloads/return_count）——register_api_params
    的平参登记会重建 overloads 且丢 return_count（out 位期望数）。
    """
    global _HCCL_TABLE_READY
    if _HCCL_TABLE_READY:
        return
    _HCCL_TABLE_READY = True
    from .hccl_apis import hccl_apis as _hccl_table

    _MANUAL_OVERRIDES.update(_hccl_table)


def register_api_params(api_name: str, params: List[ParamInfo], source: str = "manual"):
    """平参登记进 _MANUAL_OVERRIDES（overloads 重建且 return_count=0——需保真
    return_count 的全量登记走 _load_hccl_manual_table 直存 APIParamInfo）。
    """
    _MANUAL_OVERRIDES[api_name] = APIParamInfo(api_name=api_name, params=params, source=source)


def _is_tensor_method(api_name: str) -> bool:
    """torch.Tensor.xxx 方法名判定（≥3 段且第二段为 Tensor）。"""
    parts = api_name.split(".")
    return len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor"


def get_api_params(api_name: str) -> Optional[APIParamInfo]:
    """keeper 面解析入口：manual 表首查（惰性装载 hccl 表）→ 六源链 → Tensor 方法
    补 self 位（别名结果已含 self 等价张量的除外）→ 置解析标志（is_tensor_method /
    is_mutable 命名投影，schema 源已在 _parse_aten_schema_arguments 落位）。
    """
    _load_hccl_manual_table()
    if api_name in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[api_name]
    result = extract_api_params(api_name)
    if result is not None and _is_tensor_method(api_name):
        # Functional alias params already include the self-equivalent tensor
        # (e.g. 'input' from torch.nn.functional.relu_), so skip insertion.
        is_alias = "alias(" in (result.source or "")
        # 重入防护：首参已是 self（重复调用/inspect 源自带 self）不再叠加
        if not is_alias and not (result.params and result.params[0].name == "self"):
            self_param = ParamInfo(name="self", type="Tensor")
            if result.overloads:
                for ov in result.overloads:
                    ov.params.insert(0, self_param)
                    ov.layout = OverloadTensorLayout.build(ov.params, ov.return_count)
                best = max(result.overloads, key=lambda oi: sum(1 for p in oi.params if p.is_optional))
                result.params = best.params
            else:
                result.params.insert(0, self_param)
    if result is not None:
        result._resolve_api_flags()
    return result
