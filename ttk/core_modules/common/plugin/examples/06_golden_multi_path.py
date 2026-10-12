# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""同一算子在四种测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，需手动转 torch 计算后转回 numpy；
ACLNN/E2E 的 golden 直接收到 torch.Tensor，无需转换。
字符串形式（golden = "torch.abs"）由框架自动处理转换，所有路径通用。
"""

__spec__ = {
    "abs": "AbsKernelSpec",
    "aclnnAbs": "AclnnAbsSpec",
    "torch.abs": "TorchAbsSpec",
}

import torch


class AbsKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor"""

    def golden(x, **kwargs):
        x_t = torch.from_numpy(x)
        return [torch.abs(x_t).numpy()]

    third_party = {"torch": "torch.abs"}
    tolerance = {"float32": {"standard": "binary_equal"}}


class AclnnAbsSpec:
    """ACLNN 流程 — golden 收到 CPU（host）侧 torch.Tensor；third_party 在三方设备上执行"""

    def golden(x, out, **kwargs):
        return [torch.abs(x)]

    third_party = {"torch": "torch.abs"}
    tolerance = {"float32": {"standard": "binary_equal"}}


class TorchAbsSpec:
    """E2E 流程 — golden 收到 CPU（host）侧 torch.Tensor；third_party 在三方设备上执行"""

    def golden(x, **kwargs):
        return [torch.abs(x)]

    third_party = {"torch": "torch.abs"}
    tolerance = {"float32": {"standard": "binary_equal"}}
