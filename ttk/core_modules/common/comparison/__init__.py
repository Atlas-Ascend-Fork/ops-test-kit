#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""comparison 对外面（执行入口已收口为 compare_with_hooks）：

- compare_with_hooks：唯一比对执行入口（四通路共用；钩子可选——不声明即纯标准比对）
- register_comparison / get_comparison：判据扩展点（新增判据 = 新模块 + 装饰器）
- compare_batch_consistency：cleanup 阶段组装比对（接线待通路 result construct 接入）

compare / resolve_tolerance / validate_outputs 为内部件（compare() 是
compare_with_hooks 的数值回落层；resolve_tolerance 是判据路由，由各通路
golden 编排层经 ..common.comparison.resolve 子模块引入）。"""

__all__ = [
    "compare_with_hooks",
    "CompareHooks",
    "invoke_pre_compare",
    "register_comparison",
    "get_comparison",
    "compare_batch_consistency",
]

from .batch_consistency import compare_batch_consistency
from .comparison import CompareHooks, compare_with_hooks, invoke_pre_compare
from .registry import get_comparison, register_comparison
