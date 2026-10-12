#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""get_plugin_function：插件查询最外层入口（双系统按优先级合并）。

优先级：1. TestSpec 新系统（attr_name 经 spec_adapter._ATTR_MAP 桥接到
spec 属性名） 2. CustomPluginManager 旧系统（__golden__/__input__ 约定，
仅 golden/input 两类有此约定）。未命中返回 None。结果按
(name, attr_name, pathway, path) 四元组缓存，clear_plugin_cache 清空。

参数：name=注册 key（kernel 的 op_name / aclnn 与 e2e 的 api_name，本层
不区分语义）；attr_name=spec 属性语义；pathway=测试通路（新系统仅作
缓存键，旧系统作 level 门控）；plugin_path=str/Path/单值或序列，统一经
_norm_plugin_path 规范化为 tuple[str]——缓存键与 Singleton 实例键同形态，
Path/str、list/tuple 的同目录输入不分叉成多份。
"""

__all__ = ["get_plugin_function", "clear_plugin_cache"]

# Standard Packages
from typing import Callable, Dict, Optional, Tuple

from .custom_plugin_manager import CustomPluginManager
from .spec_adapter import load_spec_function

_func_cache: Dict[Tuple[str, str, str, Optional[tuple]], Optional[Callable]] = {}


def _norm_plugin_path(plugin_path) -> Optional[Tuple[str, ...]]:
    """str/Path/list/tuple → tuple[str]；falsy → None。统一缓存与 Singleton 键形态。"""
    if not plugin_path:
        return None
    if isinstance(plugin_path, (list, tuple)):
        return tuple(str(p) for p in plugin_path)
    return (str(plugin_path),)


def get_plugin_function(
    operator_name: str, plugin_type: str = "golden", level_type: str = "kernel", plugin_path=None
) -> Optional[Callable]:
    """按优先级查询已注册插件函数：1. TestSpec 2. custom plugin。未命中 None。"""
    paths = _norm_plugin_path(plugin_path)
    cache_key = (operator_name, plugin_type, level_type, paths)
    if cache_key in _func_cache:
        return _func_cache[cache_key]

    func = load_spec_function(operator_name, plugin_type, paths)

    # 旧系统仅 golden/input 有约定（门控见 CustomPluginManager.supports）
    if func is None and paths and CustomPluginManager.supports(plugin_type):
        custom_manager = CustomPluginManager(paths)
        func = custom_manager.load_if_available(operator_name, plugin_type, level_type) or None

    _func_cache[cache_key] = func
    return func


def clear_plugin_cache():
    """清空 get_plugin_function 外层查询缓存（测试 monkeypatch 底层查询前后用）。

    仅覆盖本模块 _func_cache；spec 类缓存/模块 exec 缓存在按路径键控的
    Singleton 实例（TestSpecManager/CustomPluginManager）内，热加载需另行
    经各自 clear_cache 重建。"""
    _func_cache.clear()
