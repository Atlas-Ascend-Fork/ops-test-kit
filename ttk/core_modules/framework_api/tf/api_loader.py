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
tf 模块加载 —— api_name → (callable, is_tensor_method=False)。

与 common/backend/api_resolver.py 的 tf 半边保持行为一致（common 侧为公开面
保留自身实现，通路内消费方经 adapter 走本模块）。tensorflow 惰性 import：
resolve_callable_str 仅在解析时触达 TF。
"""

from ...common.invoke import resolve_callable_str


def load_api(api_name: str):
    """tf.xxx / tensorflow.xxx → (callable, False)。TF 无 Tensor 方法协议。"""
    return resolve_callable_str(api_name), False
