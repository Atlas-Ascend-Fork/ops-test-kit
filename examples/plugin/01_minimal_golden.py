# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Simplest example: only a golden function, everything else defaults."""

# __spec__ 显式注册（可选；类名符合 PascalCase+TestSpec 约定时无需此行）
__spec__ = {"add": "AddTestSpec"}

import torch


class AddTestSpec:
    """Add operator — minimal spec (kernel flow: numpy in, numpy out)"""

    def golden(x, y, **kwargs):
        return [torch.add(torch.from_numpy(x), torch.from_numpy(y)).numpy()]
