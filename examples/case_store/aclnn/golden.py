# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""ACLNN 示例用例（cases.csv）的算子测试规范。

覆盖四个 torch 同名自动映射不适配的 API——参数名、参数序或输出布局与
torch 惯例对不上，须显式编写（其余 API 由框架自动映射 torch 同名 API
生成 golden，如 aclnnAdd→torch.add、aclnnCat→torch.cat）。加载：

    python3 -m ttk aclnn -i cases.csv --plugin examples/case_store/aclnn/golden.py

约定：ACLNN 通路 golden 按位收 torch.Tensor，输入序 = aclnnXXGetWorkspaceSize
形参序（不含 workspaceSize/executor），张量/标量/属性按 C 形参序混排为位置
参数；out 为框架分配的输出张量，可直接忽略。
"""

import torch

__spec__ = {
    "aclnnSplitTensor": "AclnnSplitTensorTestSpec",
    "aclnnClamp": "AclnnClampTestSpec",
    "aclnnIndexSelect": "AclnnIndexSelectTestSpec",
    "aclnnNonzeroV2": "AclnnNonzeroV2TestSpec",
}


class AclnnSplitTensorTestSpec:
    """kwarg 名不匹配：torch.split 的形参是 split_size_or_sections（位置参数），
    自动映射转写的 split_sections 关键字不被识别。splitSections 语义为每份
    大小（不能整除时末份取余）。"""

    def golden(self_t, splitSections, dim, out=None, **kwargs):
        return list(torch.split(self_t, split_size_or_sections=int(splitSections), dim=int(dim)))


class AclnnClampTestSpec:
    """kwarg 名不匹配：torch.clamp 收 min/max，C 接口 clipValueMin/clipValueMax
    自动映射转写为 clip_value_min/clip_value_max 后 torch 不识别。"""

    def golden(self_t, clipValueMin, clipValueMax, out=None, **kwargs):
        return [torch.clamp(self_t, min=float(clipValueMin), max=float(clipValueMax))]


class AclnnIndexSelectTestSpec:
    """参数序不匹配：C 接口 (self, dim, index)，torch.index_select(input, dim,
    index)——index 在 torch 侧位于 dim 之后，按位直传会错位。"""

    def golden(self_t, dim, index, out=None, **kwargs):
        return [torch.index_select(self_t, int(dim), index)]


class AclnnNonzeroV2TestSpec:
    """输出布局不一致：aclnnNonzeroV2 输出为 (ndim, N) 转置布局，
    torch.nonzero 返回 (N, ndim)。"""

    def golden(self_t, out=None, **kwargs):
        return [torch.nonzero(self_t).T.contiguous()]
