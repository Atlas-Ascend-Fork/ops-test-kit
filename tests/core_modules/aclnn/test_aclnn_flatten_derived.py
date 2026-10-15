# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for TestcaseAclnn flat_tensors/scalars derived properties."""

import torch

from ttk.core_modules.aclnn.testcase import TestcaseAclnn


def test_flat_tensors_derives_from_tensors():
    """flat_tensors 是 deep_flatten(tensors) 的派生缓存,不是独立赋值的 plain attr。"""
    case = TestcaseAclnn()
    t1, t2, t3 = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
    case.tensors = ((t1, t2), t3)  # 嵌套:TensorList (t1,t2) + 单 t3
    flat = case.flat_tensors
    assert list(flat) == [t1, t2, t3]  # deep_flatten 展平,顺序保持
    # 缓存生效:再读同一对象
    assert case.flat_tensors is flat


def test_tensor_list_dist_leading_none_group():
    """首位 None 组的派生元数据顺序无关:dist=(2,0),flat 展开保留 None 槽。"""
    case = TestcaseAclnn()
    case.tensor_view_shapes = ((None, (2,)), (4,))
    assert case.tensor_list_dist == (2, 0)
    assert case.flat_tensor_view_shapes == (None, (2,), (4,))
