#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""op_name → 三方 API 搜索的注册制 resolver 与采集结果桥。

仅 Kernel/ACLNN 通路经 resolver 翻译（E2E 用 resolve_api——api 本身就是
torch.xxx，无需翻译）。
"""

from abc import ABC, abstractmethod
from typing import Callable


class XpuResolver(ABC):
    """op_name → 三方 API 搜索（仅 Kernel/ACLNN 通路；E2E 用 resolve_api——
    api 本身就是 torch.xxx，无需翻译）。"""

    @abstractmethod
    def resolve(self, op_name: str, op_type: str) -> tuple: ...  # (callable, resolved_name)


_XPU_RESOLVER_REGISTRY: dict = {}


def register_xpu_resolver(provider: str) -> Callable[[type], type]:
    def decorator(cls):
        _XPU_RESOLVER_REGISTRY[provider] = cls()
        return cls

    return decorator


def get_xpu_resolver(provider: str) -> XpuResolver:
    try:
        return _XPU_RESOLVER_REGISTRY[provider]
    except KeyError:
        raise ValueError(f"Unknown XPU provider {provider!r}; registered: {sorted(_XPU_RESOLVER_REGISTRY)}") from None


def extract_third_party(xpu_results, priority):
    """从 priority provider 取 outputs（采集 → 比对的数据桥）。

    fail-closed：无 results / 无 priority / 非 PASS / 无 outputs → None
    （cross_check → GOLDEN_FAILURE）。priority 为 str 或 tuple（多 provider
    优先级序），逐个尝试首个 PASS 且有 outputs 的 provider。
    """
    if not xpu_results or priority is None:
        return None
    if isinstance(priority, str):
        priority = (priority,)
    for provider in priority:
        entry = xpu_results.get(provider, {})
        if entry.get("status") == "PASS" and "outputs" in entry:
            return entry["outputs"]
    return None
