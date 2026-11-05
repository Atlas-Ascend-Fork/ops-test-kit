#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""--signature 签名源：py 文件（ast 只读）→ aten func 行 → parse_schema 后端。

无包制备机（--no-prof 落 input/golden/xpu）上 param plan 解析 import 不了被测
包；ast 只读源码取签名，经 parse_signature_line（torch._C.parse_schema +
_parse_aten_schema_arguments）解析，与 torch.ops schema 源同一套类型/默认值/
kwarg_only 语义。两类签名源（均为 Python/torch 通用形态，无自定义注释协议）：

- 注解 stub：模块级 def 的注解合成 func 行，注解齐全才收，其余跳过；
- schema 字面量：文件内的 aten func 行字符串（torch 扩展注册 schema 形态），
  同名时覆盖注解结果。

登记键 = 函数名 / schema 串名（原样，不做全名合成）；查询侧全名未命中时尾名
唯一匹配兜底（歧义拒绝）。vararg/kwarg、类内方法、无注解/无返回注解的函数跳过。
"""

import ast
import logging
import pathlib
from typing import Dict, Optional, Tuple

from ..api_model import APIParamInfo, OverloadInfo
from .extract_schema import parse_signature_line

SUPPORTED_SUFFIXES = (".py", ".pyi")

# 注解 → aten 类型词（parse_schema 文法子集）。未知注解报错——宁缺毋假。
_SCALAR_TYPES = {"int": "int", "float": "float", "bool": "bool", "str": "str", "complex": "Scalar"}
_TENSOR_TYPES = {"Tensor", "tensor"}
_NUMBER_TYPES = {"Number", "number", "Union"}


def parse_signature_source(source: str) -> Dict[str, APIParamInfo]:
    """单值签名源 → {api_name: APIParamInfo}。当前支持 .py/.pyi stub 文件；
    yaml/内联串文法后续接入（同一 func 行后端）。
    """
    path = pathlib.Path(source)
    if path.suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported signature source {source!r}: expected .py/.pyi stub "
            f"(yaml/inline-string sources are planned; got suffix {path.suffix!r})"
        )
    return parse_signature_file(path)


def parse_signature_file(path) -> Dict[str, APIParamInfo]:
    """py 文件 → {api_name: APIParamInfo}。函数逐个容错（失败告警跳过）；
    schema 字符串字面量后扫（同名覆盖注解结果）。文件级失败上抛。
    """
    source_text = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    entries: Dict[str, APIParamInfo] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        try:
            func_line = _synthesize_func_line(node)
            _, params, return_count = parse_signature_line(func_line)
        except Exception as exc:
            logging.warning("[signature] skip %s:%s %s: %s", path, node.lineno, node.name, exc)
            continue
        _register_entry(entries, node.name, params, return_count, path)
    for candidate in _schema_string_candidates(tree):
        try:
            name, params, return_count = parse_signature_line(candidate)
        except Exception:
            logging.debug("[signature] not a schema string literal: %s", candidate[:60])
            continue
        _register_entry(entries, name, params, return_count, path)
    return entries


def _register_entry(entries, api_name, params, return_count, path) -> None:
    if api_name in entries:
        logging.warning("[signature] duplicate registration %r in %s; later one wins", api_name, path)
    entries[api_name] = APIParamInfo(
        api_name=api_name,
        source=f"signature:{path}",
        overloads=[OverloadInfo(params=params, return_count=return_count)],
    )
    logging.info("[signature] registered %r from %s (%d params, %d returns)", api_name, path, len(params), return_count)


def _schema_string_candidates(tree):
    """提取疑似 aten func 行的字符串字面量（相邻字面量拼接已由 ast 合并）。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if "(" in text and "->" in text:
                yield text


def _synthesize_func_line(node) -> str:
    """FunctionDef → aten func 行（注解 → aten 类型词、默认值字面量、* 分组）。"""
    args = node.args
    if args.vararg or args.kwarg:
        raise ValueError("var-args/kw-args are not expressible in signature stubs")
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    kwonly = list(args.kwonlyargs)
    if not positional and not kwonly:
        raise ValueError("signature stub requires at least one parameter")
    parts = []
    pos_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, pos_defaults):
        parts.append(_param_text(arg, default))
    if kwonly:
        parts.append("*")
        for arg, default in zip(kwonly, args.kw_defaults):
            parts.append(_param_text(arg, default))
    returns = _returns_text(node)
    return f"{node.name}({', '.join(parts)}) -> {returns}"


def _param_text(arg, default: Optional[ast.expr]) -> str:
    """单参数 → 'Tensor q' / 'Tensor? ori_kv=None' / 'float softmax_scale=1.0'。"""
    if arg.annotation is None:
        raise ValueError(f"parameter {arg.arg!r} has no annotation; annotate every parameter in the signature file")
    type_str, optional = _annotation_to_aten_type(arg.annotation)
    default_text = None
    if default is not None:
        default_text = _literal_text(default, arg.arg)
        optional = optional or default_text == "None"
    elif optional:
        default_text = "None"
    if optional and type_str.endswith("[]"):
        text = f"{type_str[:-2]}?[] {arg.arg}"
    else:
        text = f"{type_str}{'?' if optional else ''} {arg.arg}"
    return f"{text}={default_text}" if default_text is not None else text


def _literal_text(node, name: str) -> str:
    """默认值 AST → aten 字面量（None/bool/int/float/str）。"""
    try:
        value = ast.literal_eval(node)
    except Exception as exc:
        raise ValueError(f"parameter {name!r} default is not a literal") from exc
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    raise ValueError(f"parameter {name!r} default {value!r} is not a supported aten literal")


def _annotation_to_aten_type(node) -> Tuple[str, bool]:
    """注解 AST → (aten 类型词, optional 标记)。Optional[X] 置 optional；=None
    默认由 _param_text 补 '?'。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _map_type_word(node.value.strip().rsplit(".", 1)[-1]), False
    if isinstance(node, ast.Name):
        return _map_type_word(node.id), False
    if isinstance(node, ast.Attribute):
        return _map_type_word(node.attr), False
    if isinstance(node, ast.Subscript):
        return _subscript_to_aten_type(node)
    raise ValueError(f"unsupported annotation: {ast.dump(node)[:60]}")


def _subscript_to_aten_type(node) -> Tuple[str, bool]:
    """下标注解（Optional/List/Sequence/Union）→ (aten 类型词, optional 标记)。"""
    head = _subscript_head(node)
    inner = node.slice
    elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
    if head == "Optional":
        if len(elts) != 1:
            raise ValueError(f"Optional[...] with {len(elts)} members is not supported")
        type_str, _ = _annotation_to_aten_type(elts[0])
        return type_str, True
    if head in ("List", "Sequence", "list", "tuple"):
        if len(elts) == 2 and isinstance(elts[1], ast.Constant) and elts[1].value is Ellipsis:
            elts = [elts[0]]
        if len(elts) != 1:
            raise ValueError(f"{head}[...] with {len(elts)} members is not supported")
        inner_str, _ = _annotation_to_aten_type(elts[0])
        return f"{inner_str}[]", False
    if head == "Union":
        members = [_annotation_to_aten_type(elt)[0] for elt in elts]
        if members and set(members) <= {"int", "float", "complex"}:
            return "Scalar", False
        raise ValueError(f"Union[{', '.join(members)}] is not expressible as a single aten type")
    raise ValueError(f"unsupported annotation: {ast.dump(node)[:60]}")


def _subscript_head(node) -> str:
    head = node.value
    if isinstance(head, ast.Name):
        return head.id
    if isinstance(head, ast.Attribute):
        return head.attr
    return ""


def _map_type_word(word: str) -> str:
    if word in _TENSOR_TYPES:
        return "Tensor"
    if word in _SCALAR_TYPES:
        return _SCALAR_TYPES[word]
    if word in _NUMBER_TYPES:
        return "Scalar"
    raise ValueError(f"unknown annotation type {word!r} (supported: Tensor/int/float/bool/str/Number)")


def _returns_text(node) -> str:
    """返回注解 → 'Tensor' / '(Tensor, Tensor)' / '()'。variadic 元组不可计数，报错。"""
    annotation = node.returns
    if annotation is None:
        raise ValueError("missing return annotation; annotate the return type in the signature file")
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return "()"
    members = [annotation]
    if isinstance(annotation, ast.Subscript):
        head = _subscript_head(annotation)
        inner = annotation.slice
        if head in ("Tuple", "tuple", "List", "list", "Sequence"):
            elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
            if any(isinstance(elt, ast.Constant) and elt.value is Ellipsis for elt in elts):
                raise ValueError("variadic return tuple has no fixed return count")
            members = list(elts)
    types = [_annotation_to_aten_type(member)[0] for member in members]
    if len(types) == 1:
        return types[0]
    return "(" + ", ".join(types) + ")"
