# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ttk help 二级帮助：分组 help 复用 + 参数深潜 topic。"""

import pytest

from ttk.cli import build_parser
from ttk.cli.help_topics import HELP_TOPICS


def _run(argv, capsys):
    parser, _ = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return capsys.readouterr()


def test_help_without_args_prints_main_help(capsys):
    out = _run(["help"], capsys).out
    assert "TTK — Ascend NPU Single Operator Test Framework" in out
    assert "deep-dive help:" in out


def test_help_command_prints_grouped_subparser_help(capsys):
    out = _run(["help", "kernel"], capsys).out
    # 分组标题（美化主体）：公共组 + 通路专有组
    for section in ("input & output:", "case selection:", "precision & data:", "device:", "kernel mode:"):
        assert section in out
    assert "--devices EXPR" in out


@pytest.mark.parametrize("argv", [["help", "devices"], ["help", "kernel", "devices"]])
def test_help_param_topic_lookup(argv, capsys):
    out = _run(argv, capsys).out
    assert "--devices / --device-whitelist" in out
    assert "interplay:" in out


def test_all_topics_have_required_sections():
    for name, topic in HELP_TOPICS.items():
        assert topic["title"], name
        assert "applies to:" in topic["body"], name


def test_help_unknown_command_and_topic(capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run(["help", "bogus"], capsys)
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit):
        _run(["help", "kernel", "bogus"], capsys)


def test_e2e_help_groups(capsys):
    out = _run(["help", "e2e"], capsys).out
    assert "e2e mode:" in out
    assert "--backend {npu,npusim,cpu}" in out


def _run_cli_main(argv, monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["ttk", *argv])
    from ttk.cli import _cli_main

    _cli_main()
    return capsys.readouterr()


def test_mode_help_alias_routes_before_argparse(monkeypatch, capsys):
    """`ttk kernel help devices`：argparse required(-i) 检查前路由到二级帮助。"""
    out = _run_cli_main(["kernel", "help", "devices"], monkeypatch, capsys).out
    assert "--devices / --device-whitelist" in out


def test_mode_help_bare_shows_grouped_help(monkeypatch, capsys):
    out = _run_cli_main(["kernel", "help"], monkeypatch, capsys).out
    assert "kernel mode:" in out


def test_bare_mode_prints_help_instead_of_required_error(monkeypatch, capsys):
    """裸 `ttk kernel` 打分组 help（git/docker 惯例），不再抛 required 报错。"""
    out = _run_cli_main(["kernel"], monkeypatch, capsys).out
    assert "kernel mode:" in out
    assert "required" not in out


def test_mode_help_unknown_topic_exits(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run_cli_main(["kernel", "help", "bogus"], monkeypatch, capsys)
    assert exit_info.value.code == 2


def test_help_suffix_after_parameter(monkeypatch, capsys):
    """`ttk kernel --devices --help`：--help 后缀形式 → 参数深潜 topic。"""
    out = _run_cli_main(["kernel", "--devices", "--help"], monkeypatch, capsys).out
    assert "--devices / --device-whitelist" in out
    out = _run_cli_main(["kernel", "--help", "--core-limit"], monkeypatch, capsys).out
    assert "--core-limit — cap cores visible to compilation" in out


def test_help_suffix_falls_back_with_real_args(monkeypatch, capsys):
    """带普通参数时（-i x.csv --help）回落 argparse 的模式级 --help。"""
    with pytest.raises(SystemExit):
        _run_cli_main(["kernel", "-i", "x.csv", "--help"], monkeypatch, capsys)
    assert "kernel mode:" in capsys.readouterr().out


def test_info_list_help_alias(monkeypatch, capsys):
    """`ttk info help` / `ttk list help`：别名覆盖全部子命令（list 同有 -i 陷阱）。"""
    assert "usage: ttk list" in _run_cli_main(["list", "help"], monkeypatch, capsys).out
    assert "usage: ttk info" in _run_cli_main(["info", "help"], monkeypatch, capsys).out
