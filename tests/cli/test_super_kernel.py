# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""--super-kernel CLI 旗标接线与启动期校验（apply_e2e_args 公开面）；graph_runner 内部件用例在 framework_api/test_super_kernel_graph_runner.py。"""

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
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_super_kernel_flag_wires_switch():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    apply_e2e_args(sw, _args(const=True, super_kernel=True))
    assert sw.super_kernel_enabled is True
    assert sw.cst_switches.enabled is True


def test_super_kernel_requires_graph_mode():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    sw.dyn_switches.enabled = False
    with pytest.raises(ValueError, match="graph mode"):
        apply_e2e_args(sw, _args(super_kernel=True))


def test_super_kernel_rejects_cpu():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    sw.dyn_switches.enabled = False
    with pytest.raises(ValueError, match="--backend cpu"):
        apply_e2e_args(sw, _args(const=True, backend="cpu", super_kernel=True))
