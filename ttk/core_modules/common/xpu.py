#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/xpu.py：三方标杆执行器 + 参数绑定。

- XpuResolver 注册制（加 provider = 加一行装饰器）
- invoke_third_party + bind_params：三方 API 直调的绑定单份实现
  （整仓部署后 remote server 同源使用）
- xpu_mode_of / extract_third_party / collect_third_party：worker 三方标杆编排层
  与 remote client 的稳定查询面（三方采集仅 remote：ttk.conf.yaml remote.endpoints）

三套装配分层（并存不合并）：
  ① spec 方法注入（invoke.py 私有 _bind_by_name）——spec 作者的 golden/compare
  ② invoke_third_party（本文件 bind_params）——三方 API 直调（server + spec 类形态）
  ③ remote client（HTTP/npz 传输）——spec third_party 声明经远端执行
"""

__all__ = [
    "generate_gmm_alltoallv_matrix",
    "XpuResolver",
    "get_xpu_resolver",
    "register_xpu_resolver",
    "invoke_third_party",
    "UnknownParamError",
    "bind_params",
    "xpu_mode_of",
    "extract_third_party",
]

# Standard Packages
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

# 数据位模式（need_data / need_perf 按位组合）
XPU_DATA = 1
XPU_PERF = 2


class UnknownParamError(ValueError):
    """A parameter name is neither a known input/attribute nor
    self / a framework-reserved name / **kwargs.

    Turns a misspelled parameter name from a silent mis-computation into a
    loud failure.
    """


def xpu_mode_of(switches, need_data: bool) -> int:
    """三方采集编排：need_data/need_perf → DATA|PERF 按位模式。

    --dump xpu（dump_config.dump_xpu）也触发 DATA 采集（三方数据落盘）。"""
    mode = 0
    if need_data:
        mode |= XPU_DATA
    dump_config = getattr(switches, "dump_config", None)
    if dump_config is not None and getattr(dump_config, "dump_xpu", False):
        mode |= XPU_DATA
    if getattr(switches, "xpu_perf", False):
        mode |= XPU_PERF
    return mode


def extract_third_party(xpu_results, priority):
    """从 priority provider 取 outputs（采集 → 比对的数据桥）。

    fail-closed：无 results / 无 priority / 非 PASS / 无 outputs → None
    （cross_check → GOLDEN_FAILURE）。priority 为 str 或 tuple（多 provider
    优先级序），逐个尝试首个 PASS 且有 outputs 的 provider。
    """
    if not xpu_results or priority is None:
        return None
    if isinstance(priority, str):
        priority = (priority,)
    for provider in priority:
        entry = xpu_results.get(provider, {})
        if entry.get("status") == "PASS" and "outputs" in entry:
            return entry["outputs"]
    return None


def generate_gmm_alltoallv_matrix(a_array_val, exp_per_card, seed):
    """GMM alltoallv 的 token 分布矩阵生成（HCCL 通信类算子共享，E2E 多卡 worker 同源消费）。"""
    import numpy

    n = len(a_array_val)
    rng = numpy.random.default_rng(seed)
    total = sum(a_array_val)
    if total % n != 0:
        return [[total // n] * (exp_per_card * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in a_array_val:
        if a % n != 0:
            return [[col_sum // (exp_per_card)] * (exp_per_card * n) for _ in range(n)]
        k = a // n
        k_values.append(max(k, exp_per_card))
    blocks = []
    for k in k_values:
        block = numpy.zeros((exp_per_card, n), dtype=int)
        for col in range(n):
            counts = rng.multinomial(k - exp_per_card, [1.0 / exp_per_card] * exp_per_card)
            block[:, col] = counts + 1
        blocks.append(block)
    tmp = numpy.vstack(blocks)
    return [list(col) for col in zip(*tmp)]


class XpuResolver(ABC):
    """op_name → 三方 API 搜索（仅 Kernel/ACLNN 通路；E2E 用 resolve_api——
    api 本身就是 torch.xxx，无需翻译）。"""

    @abstractmethod
    def resolve(self, op_name: str, op_type: str) -> tuple: ...  # (callable, resolved_name)


_XPU_RESOLVER_REGISTRY: dict = {}


def register_xpu_resolver(provider: str) -> Callable[[type], type]:
    def decorator(cls):
        _XPU_RESOLVER_REGISTRY[provider] = cls()
        return cls

    return decorator


def get_xpu_resolver(provider: str) -> XpuResolver:
    try:
        return _XPU_RESOLVER_REGISTRY[provider]
    except KeyError:
        raise ValueError(f"Unknown XPU provider {provider!r}; registered: {sorted(_XPU_RESOLVER_REGISTRY)}") from None


# ── bind_params：XPU 执行参数装配的单份契约实现 ──
# （server 直接 import 本模块，client/server 同源）


def _format_device(device):
    return device


def bind_params(func, name_to_value: dict, device: Optional[str] = None, warn_leftover: bool = True) -> tuple:
    """Bind a name->value map onto ``func``'s signature BY PARAMETER NAME.

    Returns ``(args, kwargs)``. Each declared param (except self / VAR) is resolved by:
      * name in ``name_to_value`` -> take value (positional before ``*``, keyword after);
      * framework-reserved name (``device``) -> inject ``device`` (when not None);
      * has a default -> use the default: pre-star params get the default appended
        positionally so later values stay in their declared slots (skipping the
        slot would shift every later value one param to the left); post-star
        params are simply skipped (Python applies the default);
      * otherwise -> :class:`UnknownParamError` (neither input/attr nor defaulted).
    ``*args`` collects unconsumed entries (in insertion order) as positional args.
    ``**kwargs`` absorbs leftover entries (excluding ``self``); otherwise
    ``warn_leftover`` controls whether unconsumed entries log a warning (default True).
    A pool entry named ``self`` is never injected into ``**kwargs``; use ``*args``
    to receive it positionally.
    """
    import inspect

    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_keyword = False
    has_var_positional = False
    consumed = set()
    for name, param in sig.parameters.items():
        # self 仅在未绑定函数上是实例参数；绑定方法签名已去实例参，残留 self 是
        # API 真实形参名（aten 惯例 torch._foreach_addcmul(self, tensor1, ...)）——
        # 一律跳过会把第一个输入静默丢掉
        if name == "self" and not inspect.ismethod(func):
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
    leftover = {k: v for k, v in name_to_value.items() if k not in consumed and k != "self"}
    if leftover:
        if has_var_keyword:
            kwargs.update(leftover)
        elif has_var_positional:
            args.extend(leftover.values())
        elif warn_leftover:
            logging.warning("bind_params: unconsumed pool entries %s", sorted(leftover))
    return tuple(args), kwargs


def invoke_third_party(func, named: dict, attrs: dict, device, param_order: Optional[list] = None):
    """三方 API 装配执行（装配分层 ②：Remote server 端 + spec third_party 类形态共用）。

    merged = attrs ∪ named → param_order 排序（def.cpp 序，Kernel/ACLNN/GEIR
    各自提供）→ bind_params 签名匹配（具名对位，未消耗 UnknownParamError）→
    func/类双形态（类形态经 bind_params 双阶段绑定 + device 注入）→ raw outputs。
    """
    import inspect

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
        try:
            init_args, init_kwargs = bind_params(func.__init__, merged, device=device, warn_leftover=False)
        except UnknownParamError:
            init_args, init_kwargs = (), {}
        inst = func(*init_args, **init_kwargs)
        call_args, call_kwargs = bind_params(inst.__call__, merged, device=device)
        return inst(*call_args, **call_kwargs)
    args, kwargs = bind_params(func, merged, device=device)
    return func(*args, **kwargs)


def collect_third_party(**kwargs):
    """三方采集门面——与 ttk.remote.client.collect_third_party 同签名同返回（仅 remote 通路）。

    远端配置/端点不可用由 client 既有语义降级：(None, None, None) → 调用方
    GOLDEN_FAILURE（cross_check）或空指标（xpu_perf，其缺配置在启动期由
    normalize_xpu_options 硬错）。"""
    from ttk.remote.client import collect_third_party as remote_collect

    return remote_collect(**kwargs)
