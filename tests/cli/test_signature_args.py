#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""--sig/--signature 参数测试：e2e 子命令注册（append/别名）、bridge 校验与收集。"""

import argparse

import pytest

from ttk.cli.bridge import _apply_signature_args
from ttk.cli.e2e import register_e2e_command
from ttk.switches import SWITCHES


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_e2e_command(subparsers)
    return parser


def _args(**overrides):
    values = {"signature": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_e2e_exposes_signature_with_short_alias_and_append():
    args = _parser().parse_args(["e2e", "-i", "case.csv", "--sig", "a.py", "--signature", "b.pyi"])
    assert args.signature == ["a.py", "b.pyi"]


def test_apply_stores_resolved_absolute_paths(tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text("def f(x: torch.Tensor) -> torch.Tensor: ...\n")
    other = tmp_path / "other.pyi"
    other.write_text("def g(x: torch.Tensor) -> torch.Tensor: ...\n")
    switches = SWITCHES()

    _apply_signature_args(switches, _args(signature=[str(stub), str(other)]))

    assert switches.signature_sources == (str(stub.resolve()), str(other.resolve()))


def test_apply_rejects_missing_file(tmp_path):
    switches = SWITCHES()

    with pytest.raises(ValueError, match="not found"):
        _apply_signature_args(switches, _args(signature=[str(tmp_path / "missing.py")]))


def test_apply_rejects_non_stub_suffix(tmp_path):
    yaml = tmp_path / "signatures.yaml"
    yaml.write_text("- func: ...\n")
    switches = SWITCHES()

    with pytest.raises(ValueError, match=r"expects an annotated .py/.pyi"):
        _apply_signature_args(switches, _args(signature=[str(yaml)]))


def test_apply_noop_without_argument():
    switches = SWITCHES()

    _apply_signature_args(switches, _args())

    assert switches.signature_sources == ()
