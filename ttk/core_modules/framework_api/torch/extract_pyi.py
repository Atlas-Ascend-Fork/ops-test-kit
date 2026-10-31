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
torch .pyi stub 签名装载。

Tensor 方法与 nn.functional 的重载签名在 stub 文件里：多行 def 按括号
配平拼回单行（torch>=2.6 折行）、from-import-as 再导出展开、逐条解析为
ParamInfo。进程内缓存（_PYI_CACHE）。torch 为函数级惰性 import。
"""

import logging
import re
from typing import List, Optional

from ..api_model import ParamInfo, split_by_comma
from .type_inference import _coerce_default_value, _infer_type_from_name

_PYI_CACHE = None

_OPTIONAL_MARKER_RE = re.compile(r"Optional\[|\|\s*None\b")


def _has_optional_marker(part: str) -> bool:
    """Optional[...] / '| None' 类型标记视为可选（归一化剥壳前的原始串判定）。"""
    return bool(_OPTIONAL_MARKER_RE.search(part))


def _generic_inner(t: str) -> str:
    """泛型容器的完整内参（首个 '[' 起括号配平，支持嵌套泛型）。"""
    start = t.index("[") + 1
    depth = 0
    for i in range(start, len(t)):
        if t[i] in "([":
            depth += 1
        elif t[i] in ")]":
            if depth == 0:
                return t[start:i]
            depth -= 1
    return t[start:]


def _first_generic_element(t: str) -> str:
    """泛型容器首个顶层元素（顶层逗号切分，如 tuple[tuple[int, int], ...]）。"""
    inner = _generic_inner(t)
    depth = 0
    for i, ch in enumerate(inner):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            return inner[:i].strip()
    return inner.strip()


def _normalize_pyi_type(t: str) -> str:
    """pyi 类型标注归一：Optional 剥壳、Union 取首个非 None、SymInt/_int→int、
    _dtype/_layout/_device 等 torch 专有名映射。
    """
    t = t.strip()
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_pyi_type(inner)
    if t.startswith("Union[") and t.endswith("]"):
        inner = t[len("Union[") : -1]
        parts = [p.strip() for p in inner.split(",")]
        for p in parts:
            if p == "None":
                continue
            return _normalize_pyi_type(p)
        return "Tensor"
    if t == "Tensor":
        return "Tensor"
    if t in ("_int", "SymInt", "int"):
        return "int"
    if t in ("_float", "float"):
        return "float"
    if t in ("_bool", "bool"):
        return "bool"
    if t in ("str",):
        return "str"
    if t in ("Number", "_complex"):
        return "Number"
    if t in ("_dtype", "torch.dtype"):
        return "Dtype"
    if t in ("_layout", "torch.layout"):
        return "Layout"
    if t in ("_device", "DeviceLikeType", "torch.device"):
        return "Device"
    if t in ("MemoryFormat", "torch.memory_format"):
        return "MemoryFormat"
    if t.startswith("Sequence["):
        inner = t[len("Sequence[") : -1]
        inner_type = _normalize_pyi_type(inner)
        if inner_type == "int":
            return "tuple of ints"
        if inner_type == "float":
            return "tuple of floats"
        if inner_type == "Tensor":
            return "tuple of Tensors"
        return f"tuple of {inner_type}s"
    if t.startswith(("tuple[", "Tuple[")):
        inner_type = _normalize_pyi_type(_first_generic_element(t))
        if inner_type == "Tensor":
            return "tuple of Tensors"
        if inner_type.startswith(("tuple of ", "list of ")):
            return inner_type  # 嵌套容器按扁平元素序列处理
        return f"tuple of {inner_type}s"
    if t.startswith(("list[", "List[")):
        inner_type = _normalize_pyi_type(_generic_inner(t))
        if inner_type == "Tensor":
            return "tuple of Tensors"
        if inner_type.startswith(("tuple of ", "list of ")):
            return inner_type
        return f"list of {inner_type}s"
    return t


def _parse_pyi_param_str(param_str: str) -> Optional[List[ParamInfo]]:
    """单条 def 参数串 → List[ParamInfo]（`*` 起置 kw-only、`/` 跳过、
    `name: Type = default` 形态、`*name` 标记 VAR_POSITIONAL、默认值经
    _coerce_default_value 按目标类型还原）。"""
    parts = split_by_comma(param_str)
    params = []
    kw_only = False
    for part in parts:
        part_s = part.strip()
        if not part_s or part_s == "/":
            continue
        if part_s == "*":
            kw_only = True
            continue
        if part_s.startswith("**"):
            # VAR_KEYWORD（**kwargs）不占参数位，stub 侧罕见
            continue
        default_str = None
        is_optional = False
        is_var_positional = False
        main_part = part_s
        if main_part.startswith("*"):
            is_var_positional = True
            main_part = main_part[1:].strip()
        if "=" in main_part:
            main_part, default_str = main_part.split("=", 1)
            main_part = main_part.strip()
            default_str = default_str.strip()
            is_optional = True
        is_optional = is_optional or _has_optional_marker(part_s)
        if ":" in main_part:
            name, type_hint = main_part.split(":", 1)
            name = name.strip()
            type_hint = _normalize_pyi_type(type_hint.strip())
        else:
            name = main_part.strip()
            type_hint = _infer_type_from_name(name)
        if not type_hint:
            type_hint = _infer_type_from_name(name)
        default = _coerce_default_value(default_str, type_hint) if default_str is not None else None
        params.append(
            ParamInfo(
                name=name,
                type=type_hint,
                default=default,
                is_optional=is_optional,
                is_keyword_only=kw_only,
                is_var_positional=is_var_positional,
            )
        )
        if is_var_positional:
            # Python 语义：*args 之后的参数隐式 keyword-only
            kw_only = True
    return params if params else None


def _load_pyi_signatures():
    """装载 torch 自带 _C/_VariableFunctions.pyi（+ nn/functional.pyi）为进程内缓存
    {api_name: [List[ParamInfo], ...]}；torch 惰性 import，缺文件/异常返已收集部分。
    """
    global _PYI_CACHE
    if _PYI_CACHE is not None:
        return _PYI_CACHE
    _PYI_CACHE = {}
    try:
        import os

        import torch

        pyi_path = os.path.join(os.path.dirname(torch.__file__), "_C", "_VariableFunctions.pyi")
        if not os.path.exists(pyi_path):
            return _PYI_CACHE
        with open(pyi_path) as f:
            lines = f.readlines()
        for func_name, param_str in _iter_pyi_defs(lines):
            full_name = f"torch.{func_name}"
            parsed = _parse_pyi_param_str(param_str)
            if parsed:
                if full_name not in _PYI_CACHE:
                    _PYI_CACHE[full_name] = []
                _PYI_CACHE[full_name].append(parsed)
    except Exception as _e:
        logging.debug("param extract skipped: %s", _e)

    _load_nn_functional_pyi(_PYI_CACHE)
    return _PYI_CACHE


def _iter_pyi_defs(lines):
    """从 .pyi 行序列里产出 (函数名, 参数串)。

    torch>=2.6 起长签名会被折成多行:
        def conv1d(
            input: Tensor,
            ...
        ) -> Tensor: ...
    逐行正则只能匹到单行 def(如 `def abs(...)->Tensor:`),多行的整条漏掉,
    torch.conv1d 这类就进不了缓存。这里按括号配平把整条 def 拼回一行再匹。
    """
    buf = None
    depth = 0
    for line in lines:
        stripped = line.strip()
        if buf is None:
            if not stripped.startswith("def "):
                continue
            buf = stripped
            depth = stripped.count("(") - stripped.count(")")
        else:
            buf += " " + stripped
            depth += stripped.count("(") - stripped.count(")")
        if depth > 0:
            continue
        m = re.match(r"def\s+(\w+)\s*\((.*)\)\s*(?:->.*?)?:", buf)
        buf = None
        if m:
            yield m.group(1), m.group(2)


def _iter_pyi_reexports(lines):
    """产出 .pyi 里 re-export 的 (被导出名, 本地名)。

    两种写法都要认:torch<2.6 是单行 from-import-as;torch>=2.6 起改成括号包起来的
    多行块,每行一个 "X as Y"。只认单行形式会把整块漏掉,torch.nn.functional.*
    全部取不到签名。
    """
    single = re.compile(r"from\s+torch\s+import\s+(\w+)\s+as\s+(\w+)")
    block_open = re.compile(r"from\s+torch\s+import\s*\(\s*$")
    block_item = re.compile(r"(\w+)\s+as\s+(\w+)\s*,?\s*$")
    in_block = False
    for line in lines:
        stripped = line.strip()
        if in_block:
            in_block = not stripped.startswith(")")
            m = block_item.match(stripped)
        elif block_open.match(stripped):
            in_block = True
            m = None
        else:
            m = single.match(stripped)
        if m:
            yield m.group(1), m.group(2)


def _load_nn_functional_pyi(cache):
    """nn/functional.pyi 解析入缓存：from-import-as 再导出展开为 torch.nn.functional.*
    全名 + 本地 def 直收。
    """
    try:
        import os

        import torch

        pyi_path = os.path.join(os.path.dirname(torch.__file__), "nn", "functional.pyi")
        if not os.path.exists(pyi_path):
            return
        with open(pyi_path) as f:
            lines = f.readlines()

        for torch_name, local_name in _iter_pyi_reexports(lines):
            full_name = f"torch.nn.functional.{local_name}"
            torch_full = f"torch.{torch_name}"
            if torch_full in cache and full_name not in cache:
                cache[full_name] = cache[torch_full]

        for func_name, param_str in _iter_pyi_defs(lines):
            full_name = f"torch.nn.functional.{func_name}"
            parsed = _parse_pyi_param_str(param_str)
            if parsed:
                if full_name not in cache:
                    cache[full_name] = []
                cache[full_name].append(parsed)
    except Exception as _e:
        logging.debug("param extract skipped: %s", _e)


def _extract_params_from_pyi(api_name: str):
    """矩阵第⑧级后备：仅 torch.* 前缀；从 pyi 缓存按名取签名。
    返回 (主参数表, source, 全重载)。
    """
    if not api_name.startswith("torch."):
        return None
    sigs = _load_pyi_signatures().get(api_name)
    if not sigs:
        return None
    all_overloads = list(sigs)
    primary = all_overloads[0]
    source = f"pyi-stub({len(all_overloads)} overloads)" if len(all_overloads) > 1 else "pyi-stub"
    return primary, source, all_overloads
