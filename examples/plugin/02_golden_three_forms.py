# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""golden in three forms: string / function / class

kernel 流程 golden 收到 numpy.ndarray，需内部转 torch 计算后转回 numpy。
ACLNN/E2E 流程 golden 直接收到 torch.Tensor，无需转换。
"""

import torch


class AbsStrSpec:
    """Form 1: string — framework auto-maps. Simplest, for ops with direct API."""

    golden = "torch.abs"


class AbsFuncSpec:
    """Form 2: function — params before * are inputs, after * are attrs."""

    def golden(x, **kwargs):
        return [torch.abs(torch.from_numpy(x)).numpy()]


class _AbsGolden:
    """Form 3: class — for state management. __init__ optional, __call__ required."""

    def __init__(self, **kwargs):
        self.low_precision = kwargs.get("low_precision", False)

    def __call__(self, x, **kwargs):
        result = torch.abs(torch.from_numpy(x))
        if self.low_precision:
            result = result.to(torch.float16)
        return [result.numpy()]


class AbsClassSpec:
    """Form 3: golden points to a golden class (state management)."""

    golden = _AbsGolden


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "abs_str": "AbsStrSpec",
    "abs_func": "AbsFuncSpec",
    "abs_class": "AbsClassSpec",
}
