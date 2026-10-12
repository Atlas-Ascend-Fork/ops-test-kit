#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
InputGenerator.is_broadcast 的参数化测试。

覆盖广播场景：expand 产生 stride=0 的张量应返回 True。
"""

import pytest
import torch

from ttk.core_modules.aclnn.input_generate import InputGenerator


@pytest.mark.parametrize(
    ("description", "tensor_factory", "expected_bool"),
    [
        pytest.param(
            "row broadcast [32]->[16,32]",
            lambda: torch.zeros(32).unsqueeze(0).expand(16, 32),
            True,
            id="row_broadcast",
        ),
        pytest.param(
            "col broadcast [16,1]->[16,32]",
            lambda: torch.zeros(16, 1).expand(16, 32),
            True,
            id="col_broadcast",
        ),
        pytest.param(
            "3D broadcast [1,4,8]->[2,4,8]",
            lambda: torch.zeros(1, 4, 8).expand(2, 4, 8),
            True,
            id="3d_broadcast",
        ),
    ],
)
def test_is_broadcast_broadcast(description, tensor_factory, expected_bool):
    """广播场景：expand 产生 stride=0 的张量应返回 True。"""
    result = InputGenerator.is_broadcast(tensor_factory())
    assert result is expected_bool, f"Expected {expected_bool}, got {result} ({description})"
