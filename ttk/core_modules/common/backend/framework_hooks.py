#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""框架注入点 —— 通路侧向 common/backend 注册框架专属回调（依赖方向约束）。

common 不得 import 通路。TF 的 L1 张量构造（np_to_tf_inputs，依赖通路层
tf/stateful 的 OpDef 元数据）无法落在 common/backend 内，经本注册表注入：
framework_api/tf/tensor_binding.py 在包导入时注册，TfBackend.inputs_from_numpy
取用；未注册（如独立使用 backend 的场景）退 from_numpy 基础转换。

torch 的 L1（np_to_torch_inputs）无通路依赖，直接实现在
common/backend/torch_backend.py，不经本注册表。
"""

_INPUT_FACTORIES = {}


def register_input_factory(framework: str, fn) -> None:
    """注册框架的 L1 张量构造函数 fn(testcase, raw_inputs) → list。"""
    _INPUT_FACTORIES[framework] = fn


def get_input_factory(framework: str):
    """取框架的 L1 构造函数；未注册返回 None（backend 走基础转换回退）。"""
    return _INPUT_FACTORIES.get(framework)
