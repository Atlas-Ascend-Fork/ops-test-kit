#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""kernel/compile：编译子系统——注册制 compiler + dyn/cst/bin 编排。"""

_LAZY = ("static_compilation",)


def __getattr__(name):
    if name == "static_compilation":
        from .static import static_compilation

        return static_compilation
    # patch("...compile.static.X") 触发的子模块属性访问
    if name in (
        "static",
        "dynamic",
        "common",
        "binary_match",
        "compiler",
        "helper_kernels",
        "result",
        "tiling",
        "tbe",
        "asc",
    ):
        import importlib

        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
