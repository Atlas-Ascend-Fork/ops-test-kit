#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# SPDX-License-Identifier: CANN-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from argparse import Namespace

import pytest


def test_testcase_filter_trims_spaces_and_ignores_empty_items():
    from ttk.cli.bridge import args_to_switches

    switches = args_to_switches(
        Namespace(
            input="x.csv",
            output=None,
            run=1,
            provider=None,
            config=None,
            testcase="case_a, case_b,",
            testcase_index=None,
            testcase_count=None,
        )
    )

    assert switches.selected_testcases == ["case_a", "case_b"]


# ----------------------------------------------------------------------------
# --tc 非正数回归：曾以 type=int 放行 0/负数，随后绕过 instance 侧
# 0 < N < 总数 的抽样条件（-1 为未启用哨兵），静默退化为全量执行。
# ----------------------------------------------------------------------------


def _parse(argv):
    from ttk.cli import build_parser

    parser, _ = build_parser()
    return parser.parse_args(argv)


@pytest.mark.parametrize("raw", ["0", "-1", "-100"])
def test_testcase_count_rejects_non_positive(raw):
    with pytest.raises(SystemExit) as exc_info:
        _parse(["kernel", "-i", "cases.csv", f"--tc={raw}"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("raw", ["0.5", "abc"])
def test_testcase_count_rejects_non_int(raw):
    with pytest.raises(SystemExit) as exc_info:
        _parse(["kernel", "-i", "cases.csv", f"--tc={raw}"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("option", ["--tc", "--testcase-count"])
def test_testcase_count_accepts_positive(option):
    assert _parse(["kernel", "-i", "cases.csv", option, "5"]).testcase_count == 5
