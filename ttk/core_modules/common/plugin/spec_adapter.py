#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""属性桥接层：查询名 plugin_type → TestSpec 类属性名。

_ATTR_MAP 收录唯一不对称项（input→customize_inputs，历史命名），其余
查询名与 spec 属性名一致。get_spec_attr 的函数内 import 为有意为之：
调用期解析绑定，测试可 monkeypatch spec_manager.get_spec_attr 截获查询。
"""

from typing import Callable, Optional

_ATTR_MAP = {
    "input": "customize_inputs",  # 唯一不对称项（历史命名）
}


def load_spec_function(operator_name: str, plugin_type: str, plugin_path) -> Optional[Callable]:
    """从 TestSpec 按属性语义加载（经 spec_manager.get_spec_attr）。"""
    from .spec_manager import get_spec_attr

    attr_name = _ATTR_MAP.get(plugin_type, plugin_type)
    return get_spec_attr(operator_name, attr_name, plugin_path)
