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
ttk entry point

懒加载门面：import ttk 不拉起 core_modules，属性按名转发。
属性按名转发。TTK_LOAD_TF=1 的 eager tensorflow 钩子保持原行为。
"""

import importlib

_SUBPACKAGES = ("core_modules", "remote", "utilities")

# 名 → import 语句映射（延迟执行）
_LAZY_IMPORTS = {
    "runtime": "from ttk.core_modules.kernel import runtime",
    "RTSInterface": "from ttk.core_modules.kernel.runtime import RTSInterface",
    "TestcaseAclnn": "from ttk.core_modules.aclnn.testcase import TestcaseAclnn",
    "TestcaseKernel": "from ttk.core_modules.kernel.testcase import TestcaseKernel",
}


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"ttk.{name}")
    stmt = _LAZY_IMPORTS.get(name)
    if stmt is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    namespace = {}
    exec(stmt, namespace)  # noqa: S102 固定映射表内的 import 语句
    return namespace[name]


def __dir__():
    return sorted(list(_SUBPACKAGES) + list(_LAZY_IMPORTS))


# TTK_LOAD_TF=1 时 eager 加载 tensorflow（原行为保留：发生在任何懒属性之前）
try:
    import os

    if os.getenv("TTK_LOAD_TF") == "1":
        print("Loading tensorflow...")
        __import__("tensorflow")
except Exception:
    print("Tensorflow load failed")
