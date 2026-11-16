#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# SPDX-License-Identifier: CANN-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""--rerun 值语法解析测试：轮数内嵌 --rerun 值（冒号/逗号分隔，纯整数 token=轮数）。"""

from argparse import Namespace

import pytest


def _switches(rerun):
    from ttk.cli.bridge import args_to_switches

    return args_to_switches(
        Namespace(input="x.csv", output=None, run=1, provider=None, config=None, plugin=None, rerun=rerun)
    )


@pytest.mark.parametrize(
    ("value", "targets", "rounds"),
    [
        ("", ["precision_status"], 1),  # 裸参数 const / 显式空串
        ("precision_status, timeout, ", ["precision_status", "timeout"], 1),  # 旧写法：剔空格滤空段
        ("precision_status:10", ["precision_status"], 10),
        ("precision_status,perf_status:10", ["precision_status", "perf_status"], 10),
        ("10", None, 10),  # 只给轮数 → 全量复检（不按列圈选）
        (":10", None, 10),
        ("Precision_Status:3", ["precision_status"], 3),  # 列名小写化
    ],
)
def test_parse_rerun_value(value, targets, rounds):
    from ttk.cli.bridge import _parse_rerun_value

    assert _parse_rerun_value(value) == (targets, rounds)


@pytest.mark.parametrize("value", ["5:10", "precision_status:0"])
def test_parse_rerun_value_rejects_invalid(value):
    from ttk.cli.bridge import _parse_rerun_value

    with pytest.raises(ValueError, match="--rerun"):
        _parse_rerun_value(value)


def test_parse_rerun_value_rounds_upper_bound(caplog):
    # 轮数上限 MAX_RERUN_ROUNDS=1000：含边界通过；超限告警并钳到上限（不报错）
    import logging

    from ttk.cli.bridge import MAX_RERUN_ROUNDS, _parse_rerun_value

    assert MAX_RERUN_ROUNDS == 1000
    assert _parse_rerun_value("precision_status:1000") == (["precision_status"], 1000)
    with caplog.at_level(logging.WARNING):
        assert _parse_rerun_value("1001") == (None, 1000)
        assert _parse_rerun_value("precision_status:999999999999999999999") == (["precision_status"], 1000)
    assert "clamping to 1000" in caplog.text


def test_rerun_args_landing():
    # 空串 → 默认列 × 1 轮；只给轮数 → 全量（targets=None）；不传 → rerun 关闭
    sw = _switches("")
    assert (sw.rerun_targets, sw.rerun_rounds) == (["precision_status"], 1)
    sw = _switches("precision_status,perf_status:10")
    assert (sw.rerun_targets, sw.rerun_rounds) == (["precision_status", "perf_status"], 10)
    sw = _switches("10")
    assert (sw.rerun_targets, sw.rerun_rounds) == (None, 10)
    sw = _switches(None)
    assert sw.rerun_targets is None


def test_bare_rerun_takes_const_and_keeps_next_option():
    # nargs="?"：裸 --rerun 后随 -o 不被吞（取 const=""）；不传 → None
    import argparse

    from ttk.cli.kernel import register_kernel_command

    parser = argparse.ArgumentParser(prog="ttk")
    sub = parser.add_subparsers(dest="command")
    register_kernel_command(sub)
    args = parser.parse_args(["kernel", "-i", "cases.csv", "--rerun", "-o", "out.csv"])
    assert args.rerun == ""
    assert args.output == "out.csv"
    assert parser.parse_args(["kernel", "-i", "cases.csv"]).rerun is None
