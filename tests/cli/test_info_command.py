# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for `ttk info` device table: Util column must query the real DSMI API."""

import argparse
from types import SimpleNamespace


class _FakeDSMI:
    def __init__(self, util):
        self.util = util

    def get_device_count(self):
        return 1

    def get_chip_info(self, device_id):
        return SimpleNamespace(get_complete_platform=lambda: "Ascend910B", get_ver=lambda: "v1")

    def get_device_temperature(self, device_id):
        return 42

    def get_device_util(self, device_id, device_type, soc=None):
        assert device_type == 2
        return self.util


def _install_dsmi(monkeypatch, util):
    import ttk.core_modules.common.device as device_mod

    monkeypatch.setattr(device_mod, "DSMIInterface", lambda: _FakeDSMI(util))


def _run_info(capsys):
    from ttk.cli.info import _handle_info

    _handle_info(argparse.Namespace())
    return capsys.readouterr().out


def test_info_util_column_shows_aicore_util(monkeypatch, capsys):
    _install_dsmi(monkeypatch, util=37)
    out = _run_info(capsys)
    assert "37%" in out
    assert "N/A" not in out


def test_info_util_column_none_shows_na(monkeypatch, capsys):
    _install_dsmi(monkeypatch, util=None)
    out = _run_info(capsys)
    assert "N/A" in out
    assert "%" not in out.replace("N/A", "")
