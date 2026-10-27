# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""self 名输入（aten 惯例 schema，如 aclnnReal 的唯一输入参数名）的服务端执行契约。

类形式（spec third_party 插件）经 bind_params、函数形式经 _invoke_function
三相位，self 名输入都必须送达。回归：三相位与 leftover 曾全部 ``k != "self"``
过滤——远端三方执行单目 aclnn 算子（aclnnReal/aclnnAbs 自动解析 torch 同名
API）时输入被丢弃，服务端 500 "cannot be parameter-bound from inputs ['self']"。
"""

import inspect

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from ttk.remote.server import executor  # noqa: E402


def test_invoke_delivers_self_named_input():
    """类形式插件（绑定 __call__ 声明 *args）经 *args 位置送达；函数形式
    def real(self, out=None) 的 self 是真实形参名，phase 1 按名绑定；api
    模式（无插件，server 自动解析 torch 同名 API）的内建无 Python 签名、
    拒收 self= 关键字，phase 2 按 schema（C 头文件）序位置送达，双输入 +
    属性不错位；phase 3 助手保留 self 参与签名 zip、属性仍走关键字。"""

    class TorchRealLike:
        def __call__(self, *args, input=None, **kwargs):
            if not args:
                raise TypeError("no input received")
            return [torch.real(args[0])]

    t = torch.tensor([1 + 2j, 3 + 4j])
    out = executor._invoke(
        TorchRealLike,
        named={"self": t},
        attrs={},
        provider="torch",
        device_id="cpu",
        use_device=False,
    )
    assert out[0].tolist() == [1.0, 3.0]

    def real(self, out=None):
        return torch.real(self)

    assert executor._invoke_function(real, {"self": t}, {}, None).tolist() == [1.0, 3.0]
    assert executor._invoke_function(torch.real, {"self": t}, {}, None).tolist() == [1.0, 3.0]

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([10.0, 20.0])
    out = executor._invoke_function(torch.add, {"self": a, "other": b}, {"alpha": 2.0}, ["self", "other", "alpha"])
    assert out.tolist() == [21.0, 42.0]

    assert executor._function_param_names(inspect.signature(real)) == ["self", "out"]
    merged = {"self": "S", "other": "O", "alpha": 2.0}
    assert executor._ordered_values(["self", "other", "alpha"], merged, {"alpha": 2.0}) == ["S", "O"]
