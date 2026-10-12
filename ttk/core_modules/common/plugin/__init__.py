#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/plugin：插件加载与 TestSpec 体系——两系统同目录。

新系统（spec_ 前缀）：AST 建 __spec__ 索引（不 import）+ 惰性 exec 取类 +
单例缓存 + 首载校验；`*` 通配兜底（任意 op 未命中专属 spec 时落
`__spec__ = {"*": "DefaultSpec"}` 声明的类，类级兜底不 per-attr 合并）。
旧系统（custom_plugin_ 前缀）：__golden__/__input__ 约定加载，服务尚未
迁移到 TestSpec 的外部算子仓库。

公开面 4 名：get_plugin_function / SpecResolver / get_spec_class_meta /
InvalidSpecError（CustomPluginManager/PluginScanner 不进 __all__，仅 loader
内部 import）。
"""

__all__ = [
    "get_plugin_function",
    "SpecResolver",
    "get_spec_class_meta",
    "InvalidSpecError",
]


class InvalidSpecError(Exception):
    """属性类型不符合约定，validate 失败时抛出（fail-fast）"""


def __getattr__(name):
    if name not in ("get_plugin_function", "SpecResolver", "get_spec_class_meta"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "get_plugin_function":
        from .loader import get_plugin_function as attr
    elif name == "SpecResolver":
        from .resolver import SpecResolver as attr
    else:
        from .spec_manager import get_spec_class_meta as attr
    globals()[name] = attr  # 缓存到模块字典，后续访问不再走 __getattr__
    return attr
