# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""third_party in three forms: string / dict / class"""

import torch


class SoftmaxSimpleSpec:
    """Single API string — torch auto-mapped"""

    def golden(x, *, axis=-1, **kwargs):
        x_t = torch.from_numpy(x.astype("float32"))
        return [torch.softmax(x_t, dim=axis).numpy()]

    third_party = "torch.nn.functional.softmax"


class SoftmaxMultiVendorSpec:
    """Dict multi-vendor — supports torch / tf / flash_attn / npu_decompose"""

    def golden(x, *, axis=-1, **kwargs):
        x_t = torch.from_numpy(x.astype("float32"))
        return [torch.softmax(x_t, dim=axis).numpy()]

    third_party = {
        "torch": "torch.nn.functional.softmax",
        "tf": "tf.raw_ops.Softmax",
    }


class SoftmaxComposeSpec:
    """Class form — small-op composition"""

    def golden(x, *, axis=-1, **kwargs):
        x_t = torch.from_numpy(x.astype("float32"))
        return [torch.softmax(x_t, dim=axis).numpy()]

    class NpuDecomposeImpl:
        # 参数绑定契约见 README「类形式参数绑定」: input/attr 喂给声明它的方法
        # (axis 属性→__init__, x 输入→__call__); 同名参数两边都喂; 有默认值可省略。
        def __init__(self, *, axis=-1, **kwargs):
            self.axis = axis

        def __call__(self, x, **kwargs):
            exp_x = torch.exp(x)
            sum_exp = torch.sum(exp_x, dim=self.axis, keepdim=True)
            return [exp_x / sum_exp]

    third_party = {
        "torch": "torch.nn.functional.softmax",
        "npu_decompose": NpuDecomposeImpl,
    }


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "softmax_simple": "SoftmaxSimpleSpec",
    "softmax_multi_vendor": "SoftmaxMultiVendorSpec",
    "softmax_compose": "SoftmaxComposeSpec",
}
