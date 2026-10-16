# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for server-side _resolve_3party_api (KERNEL dual-form port).

torch/tf are available on the XPU server. torch_npu 与 tensorflow 的 C 扩展曾在同进程导入时冲突崩溃;
现由 tests/conftest.py 的受控预载(RTLD_DEEPBIND 装 tensorflow)消除, 无需再跳过。
本机未装 tensorflow 时 tf 用例按仓内惯例 skip（aclnn 用例走 torch-only 路径不受影响）。
"""

import importlib.util

import pytest


def test_torch_resolve_snake():
    """torch snake op_name 命中（torch.add）。"""
    from ttk.remote.server import executor

    f, name = executor._resolve_3party_api("add", "Add", "torch")
    assert callable(f)
    assert name == "add"


def test_tf_resolve_camel():
    """tf resolves relu（op_name first）。"""
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("tensorflow not installed")
    from ttk.remote.server import executor

    f, name = executor._resolve_3party_api("relu", "Relu", "tf")
    assert callable(f)


def test_aclnn_resolve_strips_prefix_and_finds_torch():
    """aclnnAdd → strip aclnn → add → torch.add。"""
    from ttk.remote.server import executor

    f, name = executor._resolve_3party_api("aclnnAdd", "Add", "aclnn")
    assert callable(f)
    assert name == "add"
