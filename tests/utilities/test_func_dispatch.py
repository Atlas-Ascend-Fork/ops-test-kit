#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""func_dispatch 测试：framework_of 框架识别、bind_by_name 按名绑定参数、resolve_callable_str 字符串解析。"""

import numpy

from ttk.utilities.func_dispatch import (
    bind_by_name,
    framework_of,
    resolve_callable_str,
)


def test_framework_of_identifies_numpy_torch_and_custom():
    """framework_of 识别 numpy ufunc / numpy 函数 / torch builtin / torch OpOverloadPacket；
    自定义函数、type、instance 返回 None。"""
    import torch

    assert framework_of(numpy.add) == "numpy"  # ufunc
    assert framework_of(numpy.isposinf) == "numpy"  # 非 ufunc
    assert framework_of(torch.add) == "torch"  # 顶层 builtin
    assert framework_of(torch.ops.aten.amax) == "torch"  # OpOverloadPacket
    assert framework_of(lambda x: x) is None  # 自定义 lambda

    def g(x, **kw):
        return x

    assert framework_of(g) is None

    class C:
        def __call__(self, x):
            return x

    assert framework_of(C) is None  # type(class)
    assert framework_of(C()) is None  # instance


def test_bind_by_name_partial_split_init_and_call():
    """__init__ 和 __call__ 分阶段绑定：init 从 pool 取 x/axis，call 从 pool 取 y。"""

    class G:
        def __init__(self, x, *, axis):
            self.x, self.axis = x, axis

        def __call__(self, y):
            return [self.x + y + self.axis]

    pool = {"x": 1, "y": 2, "axis": 9}
    ia, ik = bind_by_name(G.__init__, pool)
    inst = G(*ia, **ik)
    ca, ck = bind_by_name(inst.__call__, pool)
    assert inst(*ca, **ck) == [12]
    assert inst.x == 1
    assert inst.axis == 9


def test_bind_by_name_omitted_earlier_optional_keeps_slot_alignment():
    """省略前序可选属性时，后面的值按名落位不得左移错位（类形式 golden 绑定场景）。"""

    class G:
        def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
            self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    pool = {"alpha": -1.6e38, "reduction": "none", "pred": "P", "target": "T"}
    ia, ik = bind_by_name(G.__init__, pool)
    inst = G(*ia, **ik)
    assert (inst.gamma, inst.alpha, inst.reduction) == (2.0, -1.6e38, "none")


def test_resolve_callable_str_from_dotted_path():
    """resolve_callable_str 把 'numpy.add' / 'torch.mm' 等字符串解析为 callable。"""
    import torch

    assert resolve_callable_str("numpy.add") is numpy.add
    assert resolve_callable_str("np.negative") is numpy.negative
    assert resolve_callable_str("torch.mm") is torch.mm
    assert callable(resolve_callable_str("torch.ops.aten.amax"))
