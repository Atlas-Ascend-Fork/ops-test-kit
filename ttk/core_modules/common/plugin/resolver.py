#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""SpecResolver：8 个 spec 属性的统一 resolve 前端。

各 resolve 方法内部经 get_plugin_function 双系统合并查询（带缓存）：命中
→ 返回；未命中 → golden 调通路注入的 fallback（通路差异唯一注入点），
其余属性返回 None（调用方走默认行为）。

str→callable 的解析（resolve_callable_str）适用于方法属性
（golden/pre_compare/compare 等）；tolerance 等 dict 属性与 third_party 的
dict/class 形态原样透传。

调用方约定：除 SpecResolver 外，任何代码不得直调 get_spec_attr。
"""

__all__ = ["SpecResolver"]

# Standard Packages
from typing import Any, Callable, Optional

from ..invoke import resolve_callable_str
from .loader import get_plugin_function

# 查询名（get_plugin_function 第二参）→ TestSpec 类属性名的桥接在
# spec_adapter._ATTR_MAP（唯一不对称项：input→customize_inputs）；除该项外
# 查询名与 spec 属性名一致，本类按属性语义直接传参。


class SpecResolver:
    """8 个 spec 属性的统一 resolve 前端（各通路 worker 消费）。"""

    def __init__(self, name: str, pathway: str, plugin_path):
        self._name = name
        self._pathway = pathway
        self._plugin_path = plugin_path

    def _query(self, attr_name: str) -> Any:
        """8 个 resolve 方法的公共前半段（双系统合并查询 + 缓存）。"""
        return get_plugin_function(self._name, attr_name, self._pathway, self._plugin_path)

    # ── 方法属性：str 形态按需转 callable ──

    def _resolve_method(self, attr_name: str) -> Optional[Callable]:
        func = self._query(attr_name)
        if func is not None and isinstance(func, str):
            func = resolve_callable_str(func)
        return func

    def resolve_pre_inputs(self) -> Optional[Callable]:
        return self._resolve_method("input")

    def resolve_golden(self, fallback: Optional[Callable] = None) -> Optional[Callable]:
        """查 spec → 命中返回（str→callable / func / class 保留原形态）；
        未命中调 fallback()（通路差异唯一注入点），fallback 也 None 则 None。"""
        func = self._resolve_method("golden")
        if func is not None:
            return func  # func / class 原形态
        return fallback() if fallback else None

    def resolve_npu_preprocess(self) -> Optional[Callable]:
        return self._resolve_method("npu_preprocess")

    def resolve_pre_compare(self) -> Optional[Callable]:
        return self._resolve_method("pre_compare")

    def resolve_compare(self) -> Optional[Callable]:
        return self._resolve_method("compare")

    def resolve_third_party(self) -> Any:
        """str / dict / class / None（third_party 属性——dict/class 原样透传）。"""
        return self._query("third_party")

    # ── 非方法属性：原样透传 ──

    def resolve_tolerance(self) -> Any:
        """dict 原样透传（str→callable 转换不适用）。"""
        return self._query("tolerance")

    def resolve_torch_graph(self) -> Optional[type]:
        """类形态（nn.Module 子类）。"""
        return self._query("torch_graph")
