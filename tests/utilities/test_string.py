# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.utilities.string: kernel-name mangling, Camel->snake
conversion, prefix matching and table rendering."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ttk.utilities.string import (
    camel_to_snake,
    longest_match,
    process_kernel_string,
    render_frameless_table,
    render_table,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Add", "add"),
        ("AddN", "add_n"),
        ("Conv2D", "conv2d"),
        ("BatchNorm2D", "batch_norm2d"),
        ("LSTM", "l_stm"),
        ("aclnnAdd", "aclnn_add"),
        ("Add_S", "add__s"),
    ],
)
def test_camel_to_snake(name, expected):
    # digit-then-upper inserts no underscore; consecutive uppers chain ("LSTM" -> "l_stm");
    # '_' does not reset the state — pre-existing semantics, pinned to force conscious change
    assert camel_to_snake(name) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc_123", "abc_123"),
        ("add-1", "addneg1"),
        ("add$x", "add_x"),
    ],
    ids=["valid-unchanged", "dash-to-neg", "invalid-to-underscore"],
)
def test_process_kernel_string(value, expected):
    assert process_kernel_string(value) == expected


def test_process_kernel_string_rejects_empty_and_non_str():
    with pytest.raises(RuntimeError):
        process_kernel_string("")
    with pytest.raises(TypeError):
        process_kernel_string(123)


def test_process_kernel_string_truncation_preserves_prefix():
    # 120 is the threshold: at it the name passes through unchanged
    assert process_kernel_string("a" * 120) == "a" * 120
    result = process_kernel_string("a" * 130)
    assert result.startswith("a" * 80)
    assert len(result) > 80


def test_process_kernel_string_truncation_deterministic_across_processes():
    # PYTHONHASHSEED must not leak into the truncated suffix (md5-based)
    root = str(Path(__file__).resolve().parents[2])
    snippet = (
        f"import sys; sys.path.insert(0, {root!r}); "
        "from ttk.utilities.string import process_kernel_string; "
        "print(process_kernel_string('a' * 130))"
    )
    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env, check=True)
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("match_str", "match_list", "expected"),
    [
        ("AtlasA1", ["AtlasA1", "AtlasA2"], "AtlasA1"),
        ("AtlasA1B", ["AtlasA1", "AtlasA2"], "AtlasA1"),
        ("AtlasAB", ["AtlasA1"], ""),
        ("Foo", ["Bar"], ""),
        ("Foo", [], ""),
    ],
    ids=["exact-member", "prefix-member", "prefix-not-member", "no-match", "empty-list"],
)
def test_longest_match(match_str, match_list, expected):
    # the longest common prefix must itself be a full member of match_list
    assert longest_match(match_str, match_list) == expected


def test_render_table_golden():
    assert render_table([("hello",)]) == "+--------+\n| hello  |\n+--------+"
    assert (
        render_table([("op", "result"), ("add", "PASS")])
        == "+----------------+\n| op   | result  |\n+----------------+\n| add  | PASS    |\n+----------------+"
    )


def test_render_table_multiline_cells_golden():
    # multi-column: sub-lines pad missing cells with blanks
    assert (
        render_table([("op\nmetrics", "result"), ("add\npass:1", "PASS")])
        == "+--------------------+\n| op       | result  |\n| metrics  |         |\n"
        "+--------------------+\n| add      | PASS    |\n| pass:1   |         |\n+--------------------+"
    )
    # single-column rows span the full width across sub-lines
    assert render_table([("line1\nline2\nline3",)]) == "+--------+\n| line1  |\n| line2  |\n| line3  |\n+--------+"


def test_render_frameless_table_golden():
    assert (
        render_frameless_table([["Name", "avg"], ["item1", "1.1"]])
        == "\n---------- ----------\n      Name        avg\n---------- ----------\n     item1        1.1\n---------------------"
    )
    # column width grows beyond the 10-char minimum
    assert (
        render_frameless_table([["Name", "avg"], ["long_item_name", "1.1"]])
        == "\n-------------- ----------\n          Name        avg\n-------------- ----------\n"
        "long_item_name        1.1\n-------------------------"
    )


def test_render_frameless_table_accepts_non_str_header():
    result = render_frameless_table([[123, "avg"], ["item1", "1.1"]])
    assert "123" in result
