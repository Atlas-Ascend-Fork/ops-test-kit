#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
torch tensor 绑定 —— L2 绑定特化（npu_format_cast）。

L1 张量构造（np → torch.Tensor，非连续 view 经 as_strided 重建）属设备原语，
实现位于 common/backend/torch_backend.np_to_torch_inputs，此处不重复。
"""

from ....utilities import FORMAT_CODES, NZ_FORMATS


def apply_format_cast(tensors, formats):
    """Apply NPU format cast to tensors that require non-default formats."""
    import torch_npu

    return [
        torch_npu.npu_format_cast(t, FORMAT_CODES[fmt]) if fmt and fmt in NZ_FORMATS else t
        for t, fmt in zip(tensors, formats)
    ]


def bind_tensors(testcase, backend, dev_tensors, raw_inputs):  # noqa: ARG001
    """L2 绑定特化（TorchAdapter.bind_tensors 委托）：非默认排布（NZ 系）张量
    在上设备后做 npu_format_cast；是否支持由 backend.supports_format_cast() 决定
    （TfBackend 恒 False）。"""
    if testcase.tensor_formats and backend.supports_format_cast():
        return apply_format_cast(dev_tensors, testcase.flat_tensor_formats)
    return dev_tensors
