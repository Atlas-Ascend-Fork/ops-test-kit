#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Golden/callable dispatch helpers: framework classification + name-based binding.

Note: bind_by_name mirrors ttk/remote/server/execution_container.py:bind_params in
semantics, but server is a ttk-free hermetic module — core cannot depend on it, so
both copies are maintained independently (isolation boundary)."""

import inspect
from typing import Optional, Tuple

import numpy


class UnknownParamError(TypeError):
    """A parameter name in the golden spec is neither a known input/attribute
    nor self/**kwargs. Turns a silent mis-computation from a typo into a loud failure."""


def framework_of(func) -> Optional[str]:
    """Classify a callable by its framework: 'numpy'/'torch'/'tf'/None (custom).

    Reads the object's real ``__module__`` — does NOT guess from signature style.

    Caveats (verified empirically — do NOT simplify):
    - numpy ufunc has no ``__module__`` (access raises AttributeError) → must use getattr fallback.
    - torch top-level functions: ``type(func).__module__ == 'builtins'``, only
      ``func.__module__ == 'torch'`` → check func BEFORE type(func).
    - Do NOT use ``inspect.getmodule``: returns None for ufuncs and torch.ops.aten (OpOverloadPacket).
    """
    for obj in (func, type(func)):
        mod = getattr(obj, "__module__", "") or ""
        if mod.startswith("numpy"):
            return "numpy"
        if mod.startswith("torch"):
            return "torch"
        if mod.startswith(("tensorflow", "tf")):
            return "tf"
    if isinstance(func, numpy.ufunc):
        return "numpy"
    return None


def bind_by_name(func, pool: dict) -> Tuple[list, dict]:
    """Bind parameters of ``func`` from ``pool`` by name.

    - Parameters before ``*`` (POSITIONAL_OR_KEYWORD) → positional args.
    - Parameters after ``*`` (KEYWORD_ONLY) → keyword kwargs.
    - ``*args`` collects unconsumed pool entries (in insertion order) as
      positional args — use this to receive inputs whose names clash with
      Python reserved slots (e.g. an ACLNN param named ``self``).
    - ``**kwargs`` absorbs leftover pool entries (excluding ``self``).
    - A parameter name present in the signature but missing from pool
      (and not self/*args/**kwargs) → ``UnknownParamError``.
    - An optional param missing from pool uses its default: pre-star params
      get the default appended positionally to hold the slot (skipping it
      would shift every later value one param to the left); post-star
      params are skipped (Python applies the default).

    ``self`` is skipped by name; callers should pass a bound method
    (``inst.__call__``) so ``self`` is auto-stripped by Python.
    A pool entry named ``self`` is never injected into ``**kwargs`` to avoid
    ``multiple values for argument 'self'``; it is reachable via ``*args``.
    """
    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_kw = False
    has_var_pos = False
    consumed = set()
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            seen_star = True
            has_var_pos = True
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
            continue
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            seen_star = True
        if name in pool:
            consumed.add(name)
            if seen_star:
                kwargs[name] = pool[name]
            else:
                args.append(pool[name])
        elif p.default is inspect.Parameter.empty:
            # Required param missing from pool: raise (not greedy-bind an
            # unrelated unconsumed entry) so a golden/input name typo fails
            # loudly instead of silently computing against the wrong value.
            raise UnknownParamError(
                f"parameter '{name}' of {getattr(func, '__qualname__', func)} is not a known input or attribute name"
            )
        elif not seen_star:
            # Pre-star optional param missing from pool: append the default to
            # hold the positional slot — skipping it would shift every later
            # value one param to the left (e.g. alpha's value landing on gamma).
            args.append(p.default)
        # Post-star optional param: leave it to Python (skip, use the default)
    leftover = [(k, v) for k, v in pool.items() if k not in consumed]
    if has_var_pos:
        args.extend(v for _, v in leftover)
    if has_var_kw:
        kwargs.update({k: v for k, v in leftover if k != "self"})
    return args, kwargs


def resolve_callable_str(s: str):
    """Resolve a dotted-path string like 'numpy.abs' / 'torch.mm' / 'tf.raw_ops.Add' into a callable.

    Lazy import：只 import 字符串引用的框架（避免解析 'numpy.add' 时 eager-import
    torch/tensorflow —— 某些环境的 tensorflow C 扩展 import 即 segfault）。
    """
    ns = {"numpy": numpy, "np": numpy}
    if s == "torch" or s.startswith("torch."):
        import torch

        ns["torch"] = torch
    elif s in {"tf", "tensorflow"} or s.startswith(("tf.", "tensorflow.")):
        import tensorflow as tf

        ns["tf"] = tf
        ns["tensorflow"] = tf
    elif s.startswith(("npu_device.", "npu_bridge.")):
        # TF Adapter 暴露的 NPU 自定义算子。这些包把 npu_device.compat 重映射到
        # npu_device._api.compat, 从根包逐级 getattr 取不到(_api 下没有 tbe), 所以改成
        # "导入最深的可导入模块, 余下部分再 getattr"。
        import importlib

        parts = s.split(".")
        for k in range(len(parts) - 1, 1, -1):
            try:
                mod = importlib.import_module(".".join(parts[:k]))
            except ImportError:
                continue
            obj = mod
            try:
                for attr in parts[k:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            return obj
        raise ValueError(f"Cannot resolve golden callable {s!r}: no importable module prefix")
    try:
        # 逐级 getattr 解析点分名(与上面 npu_device 分支同一套做法)。
        # 不用 eval: 名字来自用例 CSV, 用表达式求值既无必要也把任意代码执行引进来。
        parts = s.split(".")
        obj = ns[parts[0]]
        for attr in parts[1:]:
            obj = getattr(obj, attr)
        return obj
    except Exception as e:
        raise ValueError(f"Cannot resolve golden callable {s!r}: {e}") from e
