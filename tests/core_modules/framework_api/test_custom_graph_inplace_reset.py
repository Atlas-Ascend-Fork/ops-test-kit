# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""custom torch_graph + in-place 组合的逐轮输入复位契约。

custom 分支 self_mutable 写死 None、被写位按 overload 参数名对齐自定义
forward 形参会落空（self vs input），逐轮复位空转，--dl=1 下 3 轮 MD5
误报不一致。修复后按张量身份对齐 fwd_kwargs 键，未声明时缺省覆盖
in-place 的槽 0。
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _XlogyInplaceGraph(torch.nn.Module):
    """模拟 golden 注册的自定义 torch_graph：forward 原位改写首参并返回。"""

    def forward(self, input, other):
        return input.xlogy_(other)


class _FakeResolver:
    def __init__(self, *args, **kwargs):
        self._cls = _XlogyInplaceGraph

    def resolve_torch_graph(self):
        return self._cls


def _plan(*names):
    return SimpleNamespace(overload_params=[SimpleNamespace(name=n, is_keyword_only=False) for n in names])


def test_custom_mutable_kwargs_method_style_undeclared_defaults_to_slot0():
    from ttk.core_modules.framework_api.torch import graph_runner

    t_self = object()
    t_other = object()
    fwd_kwargs = {"input": t_self, "other": t_other}
    mapped = graph_runner._custom_mutable_kwargs((), True, [t_self, t_other], fwd_kwargs)
    assert mapped == {0: "input"}


def test_custom_mutable_kwargs_declared_slots_and_self():
    from ttk.core_modules.framework_api.torch import graph_runner

    tensors = [object(), object(), object()]
    fwd_kwargs = {"a": tensors[0], "b": tensors[1], "c": tensors[2]}
    mapped = graph_runner._custom_mutable_kwargs((1,), True, tensors, fwd_kwargs)
    assert mapped == {0: "a", 1: "b"}


def test_custom_mutable_kwargs_unmapped_returns_none_not_empty_dict():
    from ttk.core_modules.framework_api.torch import graph_runner

    tensor = object()
    assert graph_runner._custom_mutable_kwargs((0,), False, [tensor], {}) is None
    assert graph_runner._custom_mutable_kwargs((), False, [tensor], {"x": tensor}) is None


def test_build_graph_model_custom_method_inplace_maps_self_slot(monkeypatch):
    from ttk.core_modules.framework_api.torch import graph_runner

    monkeypatch.setattr(graph_runner, "SpecResolver", lambda *a, **k: _FakeResolver())
    tensor_input = torch.arange(4, dtype=torch.float32)
    tensor_other = torch.ones(4, dtype=torch.float32)
    testcase = SimpleNamespace(api_name="torch.Tensor.xlogy_", inplace_input_indexes=())
    switches = SimpleNamespace(plugin_path=None)

    bundle = graph_runner._build_graph_model(
        testcase,
        None,
        switches,
        _plan("self", "other"),
        None,
        [tensor_input, tensor_other],
        {},
        is_tensor_method=True,
        is_inplace=True,
    )

    assert bundle.is_custom is True
    assert bundle.self_mutable is None
    assert bundle.fwd_kwargs.keys() == {"input", "other"}
    assert bundle.mutable_kwargs == {0: "input"}


@pytest.mark.parametrize("kind", ["undeclared_self", "declared_slots"])
def test_run_compiled_custom_kwargs_inplace_md5_consistent(tmp_path, monkeypatch, kind):
    """custom bundle 形态（args 空 + kwargs 键 + 无 self_mutable + 有 backups）
    下 --dl=1 三轮 MD5 必须一致：循环顶复位生效。"""
    from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend
    from ttk.core_modules.common.profiler.base import DisabledProfiler
    from ttk.core_modules.framework_api.torch import graph_runner

    monkeypatch.setattr(graph_runner, "get_profiler", lambda *args: DisabledProfiler())
    backend = CpuTorchBackend()
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    other = torch.full_like(value, 2.5)
    backup = value.clone()

    def compiled(*args, **kwargs):
        target = kwargs["input"]
        assert torch.equal(target, backup), "逐轮复位失效：输入带着上轮改写进入本轮"
        return target.xlogy_(kwargs["other"])

    switches = SimpleNamespace(TASK_PROFILING=False, run_time=3, root_path=str(tmp_path), warmup=False)
    if kind == "undeclared_self":
        inplace_backups, mutable_kwargs, kwargs_call = {0: backup}, {0: "input"}, {"input": value, "other": other}
    else:
        inplace_backups = {0: backup, 1: other.clone()}
        mutable_kwargs = {0: "input", 1: "other"}
        kwargs_call = {"input": value, "other": other}

    result, _, status = graph_runner._run_compiled(
        compiled,
        [],
        kwargs_call,
        backend,
        0,
        switches,
        is_inplace=True,
        self_mutable=None,
        api_name="torch.Tensor.xlogy_",
        inplace_backups=inplace_backups,
        mutable_kwargs=mutable_kwargs,
        deterministic_level=1,
        is_aclgraph=True,
    )
    assert status == "PASS"
    assert torch.equal(value, torch.from_numpy(result[0]))
