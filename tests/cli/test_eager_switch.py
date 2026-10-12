# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""--eager 传递链测试：开关接线（对齐 -c/-d 的 nargs="?" 惯例）、启动期校验。
worker 级行为（门/降级/aclgraph 不内置关）见
tests/core_modules/framework_api/test_no_executable_leg.py。

`--eager=false` 提供"只跑图模式"能力（eager 腿可独立关闭），精度主判据经
ref_nps 回退链迁至图模式。"""

from types import SimpleNamespace

import pytest


def _args(**kwargs):
    base = {
        "dynamic": None,
        "const": None,
        "backend": None,
        "fullgraph": 0,
        "aclgraph": False,
        "super_kernel": False,
        "core_limit": None,
        "eager": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_eager_false_wires_switch_off():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    apply_e2e_args(sw, _args(const=True, eager="false"))
    assert sw.eager_enabled is False
    assert sw.cst_switches.enabled is True


@pytest.mark.parametrize("eager_value", [True, "true", "TRUE", "1"])
def test_eager_truthy_forms_keep_leg_enabled(eager_value):
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    apply_e2e_args(sw, _args(eager=eager_value))
    assert sw.eager_enabled is True


def test_eager_absent_leaves_default_enabled():
    """缺省不动 switches（eager_enabled 默认 True），不触发图模式校验。"""
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    apply_e2e_args(sw, _args())
    assert sw.eager_enabled is True


def test_eager_false_requires_graph_mode():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    sw.dyn_switches.enabled = False
    with pytest.raises(ValueError, match="--eager=false requires a graph mode"):
        apply_e2e_args(sw, _args(eager="false"))


def test_eager_false_with_aclgraph_is_allowed():
    """--aclgraph 不内置关 eager（想不跑 eager 需显式 --eager=false），同开合法。"""
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    apply_e2e_args(sw, _args(aclgraph=True, eager="false"))
    assert sw.eager_enabled is False
    assert sw.aclgraph_enabled is True


def test_cli_parser_accepts_eager_switch():
    import argparse

    from ttk.cli.e2e import register_e2e_command

    subparsers = argparse.ArgumentParser().add_subparsers()
    parser = register_e2e_command(subparsers)
    option_strings = {
        opt
        for action in parser._actions
        for opt in action.option_strings  # noqa: SLF001
    }
    assert "--eager" in option_strings
    assert "--no-eager" not in option_strings
