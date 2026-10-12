# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""e2e -b/--binary 三种模式消费端测试：eager set_compile_mode 接线、-c/-d torchair
ge.jit_compile 注入、aclgraph static_kernel_compile 取值。CLI 侧翻转断言见
tests/cli/test_four_modes_entry.py::TestE2eMode。"""

import sys
from types import SimpleNamespace

import pytest


def _fake_torch_npu(monkeypatch):
    calls = []
    fake = SimpleNamespace(npu=SimpleNamespace(set_compile_mode=lambda jit_compile: calls.append(jit_compile)))
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    return calls


class _NpuBackend:
    @staticmethod
    def is_npu():
        return True


class _CpuBackend:
    @staticmethod
    def is_npu():
        return False


def _adapter(framework):
    return SimpleNamespace(framework=framework)


class TestEagerCompileMode:
    def test_binary_disables_jit_compile(self, monkeypatch):
        from ttk.core_modules.framework_api.execute import _apply_e2e_compile_mode

        calls = _fake_torch_npu(monkeypatch)
        _apply_e2e_compile_mode(_NpuBackend(), _adapter("torch"), SimpleNamespace(e2e_binary=True))
        assert calls == [False]

    def test_binary_false_enables_jit_compile(self, monkeypatch):
        from ttk.core_modules.framework_api.execute import _apply_e2e_compile_mode

        calls = _fake_torch_npu(monkeypatch)
        _apply_e2e_compile_mode(_NpuBackend(), _adapter("torch"), SimpleNamespace(e2e_binary=False))
        assert calls == [True]

    def test_cpu_backend_and_tf_adapter_skip(self, monkeypatch):
        from ttk.core_modules.framework_api.execute import _apply_e2e_compile_mode

        calls = _fake_torch_npu(monkeypatch)
        _apply_e2e_compile_mode(_CpuBackend(), _adapter("torch"), SimpleNamespace(e2e_binary=True))
        _apply_e2e_compile_mode(_NpuBackend(), _adapter("tf"), SimpleNamespace(e2e_binary=True))
        assert calls == []


class TestTorchairBinaryConfig:
    def test_binary_config_injects_ge_jit_compile(self):
        pytest.importorskip("torchair")
        from ttk.core_modules.framework_api.torch.graph_runner import _build_torchair_config

        _, global_options = _build_torchair_config(True).as_dict()
        assert global_options["ge.jit_compile"] == "0"

    def test_online_config_keeps_default(self):
        pytest.importorskip("torchair")
        from ttk.core_modules.framework_api.torch.graph_runner import _build_torchair_config

        _, global_options = _build_torchair_config(False).as_dict()
        assert "ge.jit_compile" not in global_options


class TestAclgraphOnlineCompileSource:
    def test_compile_model_aclgraph_maps_online_flag(self, monkeypatch):
        import torch

        from ttk.core_modules.framework_api.torch import graph_runner

        captured = {}

        def fake_compile(model, *, fullgraph, backend, dynamic, **kwargs):
            captured.update(options=kwargs.get("options"))
            return model

        monkeypatch.setattr(torch, "compile", fake_compile)

        graph_runner._compile_model_aclgraph(object(), "npugraph_ex", 0, False, online_compile=False)
        assert captured["options"]["static_kernel_compile"] is False
        graph_runner._compile_model_aclgraph(object(), "npugraph_ex", 0, False, online_compile=True)
        assert captured["options"]["static_kernel_compile"] is True
