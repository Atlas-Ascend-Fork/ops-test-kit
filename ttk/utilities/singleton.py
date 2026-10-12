#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Singleton Decorator
"""

__all__ = ["Singleton"]

import threading


class Singleton(type):
    """Metaclass caching instances per (class, args, kwargs) — NOT one-per-class.

    Same args return the same instance; distinct args (e.g. different plugin
    search paths, as TestSpecManager/CustomPluginManager rely on) get separate
    instances. Args must be hashable; instances are never evicted. First
    construction is thread-safe.
    """

    _instances = {}
    # RLock（可重入）：单例 __init__ 内构造另一个单例是合法嵌套（如
    # OperatorInterface.__init__ 构造 Opc）——非重入 Lock 会自死锁
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        key = (cls, args, tuple(sorted(kwargs.items())))
        with Singleton._lock:
            if key not in cls._instances:
                cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]
