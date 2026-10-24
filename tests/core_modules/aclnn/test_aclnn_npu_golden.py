#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""aclnn golden-mode NPU 测试（不触设备）。

- gen_npu 先把 ACLNN_GOLDEN 映射的点路径字符串解析为 callable 再进 NPU 闭包
- _call_torch_api(device) / _call_custom_golden(device) 的输入张量由框架上板，
  声明 device 的 golden 额外收到 device 字符串
- _golden_value_to_device：torch 张量转换、嵌套递归、其余原样
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ttk.core_modules.aclnn.golden import GoldenGenerator, _golden_value_to_device
from ttk.core_modules.aclnn.testcase import TestcaseAclnn


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "NPU"
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


def _make_testcase(api_name="aclnnMatmul"):
    case = TestcaseAclnn()
    case.testcase_name = f"test_{api_name}_npu_golden"
    case.api_name = api_name
    # gen_npu 的 is_torch_dtype_support() 门内 tensors 必为 torch（输入生成
    # 已统一转换），测试按同一契约构造
    case.tensors = [torch.ones((2, 2), dtype=torch.float32)]
    case.attributes = {}
    case.output_tensor_indexes = (0,)
    case._pure_output_indexes = ()
    case.manual_golden_binaries = None
    return case


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
    monkeypatch.setattr("ttk.core_modules.aclnn.golden.get_global_storage", lambda: _mock_switches())
    # 无 opp 头文件环境：_package_golden_torch_kwargs 等消费点不触真实单例
    monkeypatch.setattr("ttk.core_modules.aclnn.golden.OpApiInfoKeeper", MagicMock())


@pytest.fixture
def fake_torch_npu(monkeypatch):
    """替身 torch_npu：gen_npu 会 set_device 初始化设备，测试进程不得触真卡
    （真初始化会向本进程 plog 写 [ERROR]，污染后续读取 plog 的用例）。"""
    import sys

    calls = []
    module = SimpleNamespace(npu=SimpleNamespace(set_device=calls.append))
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return calls


def test_gen_npu_resolves_str_carrier_before_npu_closure(monkeypatch, fake_torch_npu):
    """gen_npu 先把字符串载体解析为 callable，NPU 闭包收到 torch 原生函数。"""
    case = _make_testcase("aclnnMatmul")
    monkeypatch.setattr(GoldenGenerator, "_import_golden_funcs", lambda self: "torch.relu")
    # is_torch_dtype_support 是 gen_npu 的前置条件，本测试不构造完整声明态
    monkeypatch.setattr(TestcaseAclnn, "is_torch_dtype_support", lambda self: True)

    seen = []

    def _fake_invoke_golden(self, gf, device=None):
        seen.append((gf, device))
        return ["golden"]

    monkeypatch.setattr(GoldenGenerator, "_invoke_golden", _fake_invoke_golden)

    def _capture_golden(testcase, func, switches, invoke=None, npu_invoke=None, dev_id=None, warmup_count=None):
        assert npu_invoke is not None, "NPU closure must be provided"
        return npu_invoke()

    monkeypatch.setattr("ttk.core_modules.aclnn.golden.Invoke.golden", _capture_golden)

    GoldenGenerator(case).gen_npu(0)

    assert seen == [(torch.relu, "npu:0")], f"NPU closure must see resolved callable, got {seen!r}"
    assert fake_torch_npu == [0], "gen_npu must initialize torch_npu device before execution"
    assert case.golden_tensors == ["golden"]


def test_gen_npu_flattens_cpu_fallback_result(monkeypatch, fake_torch_npu):
    """NPU 执行失败回落 CPU 时 _invoke_golden 可能返回裸 Tensor：gen_npu 须
    flatten 成输出槽列表（否则 results[:1] 切的是张量首行）。"""
    case = _make_testcase("aclnnAdd")
    monkeypatch.setattr(GoldenGenerator, "_import_golden_funcs", lambda self: torch.add)
    monkeypatch.setattr(TestcaseAclnn, "is_torch_dtype_support", lambda self: True)

    bare = torch.arange(8).reshape(2, 4)
    monkeypatch.setattr(GoldenGenerator, "_invoke_golden", lambda self, gf, device=None: bare)

    def _fallback_golden(testcase, func, switches, invoke=None, npu_invoke=None, dev_id=None, warmup_count=None):
        # 模拟 NPU 执行失败后的 CPU 回落：走 invoke 闭包
        return invoke(func)

    monkeypatch.setattr("ttk.core_modules.aclnn.golden.Invoke.golden", _fallback_golden)

    GoldenGenerator(case).gen_npu(0)

    tensors = case.golden_tensors
    assert isinstance(tensors, list), f"expected list, got {type(tensors)}"
    assert len(tensors) == 1, f"expected 1 element, got {len(tensors)}"
    assert tensors[0] is bare, "bare Tensor must be wrapped, not row-sliced"


def test_call_torch_api_uploads_inputs_for_native_carrier():
    """device 非 None 时：输入张量上板（numpy 转 torch），声明 device 的
    golden 收到 device 字符串，返回值回转 numpy。"""
    case = _make_testcase("aclnnRelu")
    gen = GoldenGenerator(case)

    received = {}

    def _native_gf(x):
        received["x"] = x
        return [x * 2]

    out = gen._call_torch_api(_native_gf, device="cpu")

    assert isinstance(received["x"], torch.Tensor), (
        f"native torch golden must receive uploaded torch tensor, got {type(received['x'])}"
    )
    np.testing.assert_array_equal(out[0], np.full((2, 2), 2.0, dtype="float32"))
    assert isinstance(out[0], np.ndarray), f"device path must convert outputs back to numpy, got {type(out[0])}"

    def _device_gf(x, device=None):
        received["device"] = device
        return [x]

    gen._call_torch_api(_device_gf, device="cpu")
    assert received["device"] == "cpu", "device-declaring golden must receive the device string"


def test_call_custom_golden_preuploads_inputs(monkeypatch):
    """device 非 None 时：输入张量上板（.to(device) 拷贝），device 字符串
    注入给声明它的 golden，返回值回转 numpy。"""
    case = _make_testcase("aclnnCustom")
    monkeypatch.setattr(TestcaseAclnn, "is_torch_dtype_support", lambda self: True)

    received = {}

    def _golden(self_t, alpha=None, device=None, **kwargs):
        received["self_t"] = self_t
        received["device"] = device
        return [self_t * alpha]

    fake_plan = SimpleNamespace(build_args=lambda tensors, scalars, attributes: ((tensors[0], 3), {}))
    case._param_plan_cache = fake_plan

    out = GoldenGenerator(case)._call_custom_golden(_golden, device="cpu")

    assert isinstance(received["self_t"], torch.Tensor), (
        f"custom golden must receive pre-uploaded torch tensor, got {type(received['self_t'])}"
    )
    assert received["device"] == "cpu", "device string must stay injected (new-tensor creation)"
    assert isinstance(out[0], np.ndarray), f"device path must convert outputs back to numpy, got {type(out[0])}"
    np.testing.assert_array_equal(out[0], np.full((2, 2), 3.0, dtype="float32"))


def test_golden_value_to_device_recurses_and_passes_through():
    """torch 张量 → .to(device)；嵌套 Sequence 递归；非 torch 值原样返回。"""
    t = torch.ones(2)
    np_leaf = np.ones(3, dtype="float32")
    nested = [t, (np_leaf, None, 5), "attr"]

    moved = _golden_value_to_device(nested, "cpu")

    assert isinstance(moved[0], torch.Tensor)
    assert moved[1][0] is np_leaf, "non-torch values must pass through unchanged (no numpy conversion)"
    assert moved[1][1] is None
    assert moved[1][2] == 5
    assert moved[2] == "attr"
    assert isinstance(moved, list), "container types must be preserved"
    assert isinstance(moved[1], tuple), "container types must be preserved"
