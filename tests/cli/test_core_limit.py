# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""--core-limit 传递链测试：解析格式、kernel 侧按 core_type 生效分量、启动期物理上限校验、aclgraph 下发。"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8", (8, None)),
        ("8,48", (8, 48)),
        (",48", (None, 48)),
    ],
)
def test_parse_core_limit_formats(raw, expected):
    from ttk.cli.bridge import args_to_switches

    sw = args_to_switches(_wiring_args(core_limit=raw))
    assert sw.core_limit == expected


@pytest.mark.parametrize("raw", ["abc", "8|48"])
def test_parse_core_limit_rejects_invalid(raw):
    from ttk.cli.bridge import args_to_switches

    with pytest.raises(ValueError, match="--core-limit"):
        args_to_switches(_wiring_args(core_limit=raw))


def _make_opc_with_core_limit(core_type, core_limit):
    from ttk.core_modules.kernel.compile.compiler import Opc
    from ttk.switches import SWITCHES, set_global_storage

    sw = SWITCHES()
    sw.core_limit = core_limit
    set_global_storage(sw)
    opc = object.__new__(Opc)
    opc._core_type = core_type
    calls = []
    opc._all_opc_invoke = lambda func, *a, **k: calls.append((func, a))
    return opc, calls


@pytest.mark.parametrize(
    ("core_type", "core_limit", "expected_res"),
    [
        ("AiCore", (8, 48), {"ai_core_cnt": "8"}),
        ("VectorCore", (8, 48), {"vector_core_cnt": "48"}),
        ("MIX", (None, 48), {"vector_core_cnt": "48"}),
        ("AiCore", (None, 48), None),
    ],
)
def test_apply_core_limit_per_core_type(core_type, core_limit, expected_res):
    opc, calls = _make_opc_with_core_limit(core_type, core_limit)
    opc._apply_core_limit()
    if expected_res is None:
        assert calls == []
    else:
        assert calls == [("set_platform_info_res", (0, expected_res))]


_HW = {"full_soc_version": "Ascend910B3", "ai_core_cnt": 20, "vector_core_cnt": 40}


def test_validate_core_limit_bounds():
    from ttk.core_modules.common.device import validate_core_limit

    validate_core_limit((20, 40), _HW)
    validate_core_limit((8, None), _HW)
    with pytest.raises(ValueError, match="ai_core_cnt"):
        validate_core_limit((21, 32), _HW)


def test_e2e_instance_validates_core_limit(monkeypatch):
    """e2e detect_platform 在启动期校验 core_limit；CPU 后端跳过。"""
    from ttk.core_modules.framework_api import instance as fai
    from ttk.switches import SWITCHES, set_global_storage

    sw = SWITCHES()
    sw.dev_plat = "AUTO"
    sw.core_limit = (8, 99)
    set_global_storage(sw)

    backend = SimpleNamespace(
        device_name=lambda: "Ascend910B3",
        soc_series=lambda: "Ascend910B",
        is_npu=lambda: True,
    )
    monkeypatch.setattr(fai, "get_backend", lambda *a, **k: backend)
    monkeypatch.setattr(
        "ttk.core_modules.common.device.get_npu_hw_info",
        lambda plat: dict(_HW),
    )
    inst = fai.FrameworkApiInstance.__new__(fai.FrameworkApiInstance)
    inst.backend = backend
    with pytest.raises(ValueError, match="vector_core_cnt"):
        inst.detect_platform()

    inst.backend = SimpleNamespace(device_name=lambda: "cpu", soc_series=lambda: "cpu", is_npu=lambda: False)
    inst.detect_platform()  # CPU: 不校验不抛错


# ----------------------------------------------------------------------------
# CLI 接线回归：--core-limit 历史上仅 e2e 侧解析，geir/kernel 注册后静默丢弃
# （args_to_switches 收口修复的守护）。add_perf_args 分组注册后三通路同源。
# ----------------------------------------------------------------------------
def _wiring_args(**kwargs):
    base = {
        "provider": None,
        "config": None,
        "input": "x.csv",
        "output": None,
        "run": 1,
        "core_limit": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("mode", ["e2e", "geir", "kernel"])
def test_core_limit_wired_via_args_to_switches_for_all_modes(mode):
    """e2e/geir/kernel 的 --core-limit 均经 args_to_switches 落到 sw.core_limit。"""
    from ttk.cli.bridge import args_to_switches

    sw = args_to_switches(_wiring_args(core_limit="8,48"))
    assert sw.core_limit == (8, 48)


def test_core_limit_absent_leaves_switch_untouched():
    from ttk.cli.bridge import args_to_switches
    from ttk.switches import SWITCHES

    sw = args_to_switches(_wiring_args())
    assert sw.core_limit is SWITCHES().core_limit is None


def test_perf_args_group_registration_matrix():
    """add_perf_args 注册矩阵：xpu-perf 四通路；core-limit 三通路（aclnn 无编译面不注册）。"""
    import argparse

    from ttk.cli.aclnn import register_aclnn_command
    from ttk.cli.e2e import register_e2e_command
    from ttk.cli.geir import register_geir_command
    from ttk.cli.kernel import register_kernel_command

    sub = argparse.ArgumentParser().add_subparsers()
    for register, has_core_limit in (
        (register_e2e_command, True),
        (register_geir_command, True),
        (register_kernel_command, True),
        (register_aclnn_command, False),
    ):
        parser = register(sub)
        opts = {o for a in parser._actions for o in a.option_strings}  # noqa: SLF001
        assert "--xpu-perf" in opts, register.__module__
        assert ("--core-limit" in opts) is has_core_limit, register.__module__


def test_aclgraph_core_limit_scopes_preprocess_and_captured_model(monkeypatch):
    """scope 机制：eager 伴随算子段 + 每次图捕获分别限核。"""

    from ttk.core_modules.framework_api.torch import graph_runner as graph_execution

    events = []

    class FakeScope:
        @staticmethod
        @contextmanager
        def limit_core_num(ai_limit, vec_limit):
            events.append(("enter", ai_limit, vec_limit))
            yield
            events.append(("exit", ai_limit, vec_limit))

    monkeypatch.setattr(graph_execution, "_get_aclgraph_scope", lambda: FakeScope)
    monkeypatch.setattr(
        graph_execution.torch.npu,
        "get_device_properties",
        lambda _dev_id: SimpleNamespace(cube_core_num=20, vector_core_num=40),
    )

    with graph_execution._aclgraph_core_limit_context((1, None), 0):
        events.append(("preprocess",))

    model = graph_execution._AclGraphCoreLimitModel(torch.nn.Identity(), (1, 2), 0)
    assert torch.equal(model(torch.tensor([3])), torch.tensor([3]))
    assert events == [
        ("enter", 1, 40),
        ("preprocess",),
        ("exit", 1, 40),
        ("enter", 1, 2),
        ("exit", 1, 2),
    ]
