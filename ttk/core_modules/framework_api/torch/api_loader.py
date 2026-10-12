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
torch 模块加载 —— api_name → (callable | Tensor 方法名, is_tensor_method)。

与 common/backend/api_resolver.py 的 torch 半边保持行为一致（common 侧为公开面
保留自身实现，通路内消费方经 adapter 走本模块）：
  - TorchOpsPackageLoader.ensure_registered：forkserver worker 侧自定义 torch
    扩展包注册（worker 不继承父进程 sys.modules，须自行注册）；
  - torch.Tensor.xxx → 方法名字符串，调用侧经 args[0] 分发；
  - 其余按根模块 __import__ + 属性遍历。
"""

from ...common.backend.torch_ops_package_loader import TorchOpsPackageLoader

_MODULE_ALIAS = {}


def ensure_registered(api_name: str):
    """注册 api_name 所属的自定义 torch 扩展包（幂等）。"""
    TorchOpsPackageLoader.ensure_registered(api_name)


def load_api(api_name: str):
    """api_name → (callable_or_method_name, is_tensor_method)。

    Examples:
        'torch.add'                 -> (torch.add, False)
        'torch_npu.npu_conv2d'      -> (torch_npu.npu_conv2d, False)
        'torch.Tensor.relu_'        -> ('relu_', True)
        'torch.ops.aten.add.Tensor' -> (OpOverload, False)
    """
    parts = api_name.split(".")

    if len(parts) < 2:
        raise ValueError(f"Invalid api_name: {api_name}, expected format: module.func")

    ensure_registered(api_name)

    # torch.Tensor.xxx -> Tensor method
    if len(parts) >= 3 and parts[1] == "Tensor":
        method_name = ".".join(parts[2:])
        return method_name, True

    # Module function: import root module (with alias support), traverse attributes
    root_name = _MODULE_ALIAS.get(parts[0], parts[0])
    try:
        obj = __import__(root_name)
        for p in parts[1:]:
            obj = getattr(obj, p)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Cannot resolve api_name '{api_name}': {e}") from e
    return obj, False
