#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""XPU 三方 API 参数装配（bind_params / invoke_third_party）。

三方 API 直调的绑定单份实现：整仓部署后 remote server 同源使用
（server 直接 import 本模块，client/server 同源）。装配分层 ②——spec 方法
注入（invoke.py 私有 _bind_by_name）与 remote client（HTTP/npz 传输）不在此列。
"""

import inspect
import logging
from typing import Optional


class UnknownParamError(ValueError):
    """A parameter name is neither a known input/attribute nor
    self / a framework-reserved name / **kwargs.

    Turns a misspelled parameter name from a silent mis-computation into a
    loud failure.
    """


def _format_device(device):
    return device


def _self_is_instance_slot(func) -> bool:
    """签名中的 ``self`` 形参是否为实例占位（而非 API 真实参数名）。

    仅未绑定的 ``__init__`` 如此——其首参由 ``cls(*args)`` 填充，永不来自
    输入/属性池。普通函数（含 aten 惯例 ``def f(self, ...)`` 形态的三方
    API/插件函数）的 ``self`` 是真实参数名，须按名绑定；绑定方法的签名已
    由 Python 去掉实例参，不在此列。
    """
    return not inspect.ismethod(func) and getattr(func, "__name__", None) == "__init__"


def _self_kwarg_conflicts(func) -> bool:
    """``self=`` 关键字是否会与实例位冲突（self 名输入不可经 **kwargs 送达）。

    绑定方法（实例已占 ``self`` 位，``inst(self=...)`` → multiple values
    for argument 'self'）与未绑定 ``__init__``（``cls(**{'self': ...})``
    同理）都冲突；普通函数无实例位，安全。
    """
    return not inspect.isfunction(func) or _self_is_instance_slot(func)


def bind_params(func, name_to_value: dict, device: Optional[str] = None, warn_leftover: bool = True) -> tuple:
    """Bind a name->value map onto ``func``'s signature BY PARAMETER NAME.

    Returns ``(args, kwargs)``. Each declared param (except an unbound-
    ``__init__`` instance slot / VAR) is resolved by:
      * name in ``name_to_value`` -> take value (positional before ``*``, keyword after);
      * framework-reserved name (``device``) -> inject ``device`` (when not None);
      * has a default -> use the default: pre-star params get the default appended
        positionally so later values stay in their declared slots (skipping the
        slot would shift every later value one param to the left); post-star
        params are simply skipped (Python applies the default);
      * otherwise -> :class:`UnknownParamError` (neither input/attr nor defaulted).
    A ``self`` parameter is a REAL API param name for plain functions (aten
    convention, e.g. ``torch._foreach_addcmul(self, tensor1, ...)``) and binds
    from the pool like any name; only an unbound ``__init__``'s ``self`` (the
    instance slot, filled by ``cls(*args)``) is skipped.
    ``*args`` collects unconsumed entries (in insertion order) as positional
    args; with ``**kwargs`` present, non-``self`` leftovers go to ``**kwargs``
    instead (no double-feeding). A pool entry named ``self`` (an input whose
    schema name is literally ``self``, e.g. aclnnReal's only input) is
    delivered positionally via ``*args``; it goes through ``**kwargs`` only for
    plain functions (no instance slot to collide with), never for bound
    methods / unbound ``__init__``. ``warn_leftover`` controls whether
    unconsumed/undeliverable entries log a warning (default True).
    """
    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_keyword = False
    has_var_positional = False
    consumed = set()
    for name, param in sig.parameters.items():
        # self 双语义：未绑定 __init__ 的首参 self 是实例占位（cls(*args) 填充），
        # 跳过；普通函数签名里的 self 是 API 真实形参名（aten 惯例
        # torch._foreach_addcmul(self, tensor1, ...)），按名绑定——一律跳过会把
        # 第一个输入静默丢掉（aclnnReal 等 self 名输入的远端三方执行即由此回归）。
        if name == "self" and _self_is_instance_slot(func):
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            seen_star = True
            has_var_positional = True
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            seen_star = True
        if name in name_to_value:
            consumed.add(name)
            if seen_star:
                kwargs[name] = name_to_value[name]
            else:
                args.append(name_to_value[name])
        elif name == "device" and device is not None:
            kwargs[name] = _format_device(device)
        elif param.default is not inspect.Parameter.empty:
            if not seen_star:
                # Pre-star optional param missing from the pool: append the default
                # to hold the positional slot — skipping it would shift every later
                # value one param to the left (e.g. alpha's value landing on gamma).
                args.append(param.default)
            # Post-star optional param: leave it to Python (skip, use the default)
        else:
            raise UnknownParamError(
                f"parameter {name!r} of {getattr(func, '__name__', func)} has no value "
                f"(not in inputs/attrs, no default)"
            )
    leftover = {k: v for k, v in name_to_value.items() if k not in consumed}
    if has_var_positional and not has_var_keyword:
        # 仅 *args：全部 leftover（含 self 名输入）按插入序整体位置送达
        args.extend(leftover.values())
        return tuple(args), kwargs
    unset = object()
    self_value = leftover.pop("self", unset)
    if leftover:
        if has_var_keyword:
            kwargs.update(leftover)
        elif warn_leftover:
            logging.warning("bind_params: unconsumed pool entries %s", sorted(leftover))
    if self_value is not unset:
        if has_var_positional:
            # *args + **kwargs 并存：self 只走 *args——bound 方法的 self= 关键字
            # 会与实例位冲突（multiple values for argument 'self'）
            args.append(self_value)
        elif has_var_keyword and not _self_kwarg_conflicts(func):
            kwargs["self"] = self_value
        elif warn_leftover:
            logging.warning(
                "bind_params: pool entry 'self' undeliverable to %s (no *args; **kwargs collides with the instance slot)",
                getattr(func, "__name__", func),
            )
    return tuple(args), kwargs


def invoke_third_party(func, named: dict, attrs: dict, device, param_order: Optional[list] = None):
    """三方 API 装配执行（装配分层 ②：Remote server 端 + spec third_party 类形态共用）。

    merged = attrs ∪ named → param_order 排序（def.cpp 序，Kernel/ACLNN/GEIR
    各自提供）→ bind_params 签名匹配（具名对位，未消耗 UnknownParamError）→
    func/类双形态（类形态经 bind_params 双阶段绑定 + device 注入）→ raw outputs。
    """
    merged = dict(attrs or {})
    merged.update(named)
    if param_order:
        from collections import OrderedDict

        ordered = OrderedDict()
        for name in param_order:
            if name in merged:
                ordered[name] = merged[name]
        for k, v in merged.items():
            if k not in ordered:
                ordered[k] = v
        merged = ordered
    if inspect.isclass(func):
        if func.__init__ is object.__init__:  # 无自定义 __init__: 直接实例化
            inst = func()
        else:
            try:
                init_args, init_kwargs = bind_params(func.__init__, merged, device=device, warn_leftover=False)
            except UnknownParamError:
                init_args, init_kwargs = (), {}
            inst = func(*init_args, **init_kwargs)
        call_args, call_kwargs = bind_params(inst.__call__, merged, device=device)
        return inst(*call_args, **call_kwargs)
    args, kwargs = bind_params(func, merged, device=device)
    return func(*args, **kwargs)
