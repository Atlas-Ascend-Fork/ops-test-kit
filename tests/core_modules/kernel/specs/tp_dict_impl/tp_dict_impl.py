# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""third_party style: dict of impl classes (spec mode)."""


class AddDictImplSpec:
    golden = "torch.add"

    class AddTorchImpl:
        """torch impl — called on the XPU server (spec mode)."""

        def __call__(self, x, y, **kwargs):
            import torch

            return [torch.add(x, y)]

    third_party = {"torch": AddTorchImpl}


__spec__ = {"add": "AddDictImplSpec"}
