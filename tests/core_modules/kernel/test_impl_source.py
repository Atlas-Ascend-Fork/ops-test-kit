#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""算子实现来源可见性测试：impl_source_of 逐算子来源描述、
--expect-impl-source 圈选校验（不符显式失败）、CLI 参数注册与解析。"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.kernel.info_keeper import OpInfoKeeper
from ttk.core_modules.kernel.instance import KernelInstance
from ttk.core_modules.kernel.testcase import TestcaseKernel


class _FakeKeeper:
    """impl_source_of 桩：按 op_name 返回预设来源。"""

    def __init__(self, sources):
        self.sources = sources

    def impl_source_of(self, op_name):
        return self.sources.get(op_name)


def _kernel_case(op_name):
    case = TestcaseKernel()
    case.testcase_name = f"case_{op_name}"
    case.op_name = op_name
    return case


def _make_instance(sw, cases):
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        ins = KernelInstance()
    ins.flatten_testcases = cases
    return ins


def test_impl_source_of_returns_source_fields():
    keeper = OpInfoKeeper()
    keeper._ops_info = {
        "add": {
            "_impl_source": "vendor",
            "_impl_vendor": "acme",
            "_impl_prefix": "acme_",
            "_impl_base_path": "/vendors/acme",
        },
        "sub": {},  # 旧格式信息（无来源字段）→ builtin 默认
    }
    src = keeper.impl_source_of("add")
    assert src == {"source": "vendor", "vendor": "acme", "impl_prefix": "acme_", "impl_base_path": "/vendors/acme"}
    assert keeper.impl_source_of("sub")["source"] == "builtin"
    assert keeper.impl_source_of("missing") is None


def test_expect_impl_source_pass(monkeypatch):
    sw = MagicMock()
    sw.expect_impl_source = "vendor=acme"
    sources = {"add": {"source": "vendor", "vendor": "acme", "impl_prefix": "acme_", "impl_base_path": "/vendors/acme"}}
    monkeypatch.setattr("ttk.core_modules.kernel.info_keeper.OpInfoKeeper", lambda: _FakeKeeper(sources))
    ins = _make_instance(sw, [_kernel_case("add"), _kernel_case("add")])
    ins._check_expected_impl_source()  # 不抛即通过


def test_expect_impl_source_mismatch_raises(monkeypatch):
    """自定义包未生效（静默回落 builtin）时显式失败——核心危害场景。"""
    sw = MagicMock()
    sw.expect_impl_source = "vendor=acme"
    sources = {"add": {"source": "builtin", "vendor": "", "impl_prefix": "", "impl_base_path": "/builtin"}}
    monkeypatch.setattr("ttk.core_modules.kernel.info_keeper.OpInfoKeeper", lambda: _FakeKeeper(sources))
    ins = _make_instance(sw, [_kernel_case("add")])
    with pytest.raises(RuntimeError, match="IMPL_SOURCE_MISMATCH") as excinfo:
        ins._check_expected_impl_source()
    assert "add" in str(excinfo.value)
    assert "builtin" in str(excinfo.value)


def test_expect_impl_source_vendor_name_mismatch_raises(monkeypatch):
    """同 source 不同 vendor（旧包抢占场景）也判不符。"""
    sw = MagicMock()
    sw.expect_impl_source = "vendor=acme"
    sources = {
        "add": {
            "source": "vendor",
            "vendor": "other_vendor",
            "impl_prefix": "other_",
            "impl_base_path": "/vendors/other_vendor",
        }
    }
    monkeypatch.setattr("ttk.core_modules.kernel.info_keeper.OpInfoKeeper", lambda: _FakeKeeper(sources))
    ins = _make_instance(sw, [_kernel_case("add")])
    with pytest.raises(RuntimeError, match="IMPL_SOURCE_MISMATCH"):
        ins._check_expected_impl_source()


def test_expect_impl_source_no_info_raises(monkeypatch):
    sw = MagicMock()
    sw.expect_impl_source = "custom"
    monkeypatch.setattr("ttk.core_modules.kernel.info_keeper.OpInfoKeeper", lambda: _FakeKeeper({}))
    ins = _make_instance(sw, [_kernel_case("ghost")])
    with pytest.raises(RuntimeError, match="no op info found"):
        ins._check_expected_impl_source()


def test_expect_impl_source_invalid_value(monkeypatch):
    sw = MagicMock()
    sw.expect_impl_source = "foo"
    ins = _make_instance(sw, [])
    with pytest.raises(ValueError, match="Invalid --expect-impl-source"):
        ins._check_expected_impl_source()


def test_expect_impl_source_unset_is_noop():
    sw = MagicMock()
    sw.expect_impl_source = None
    ins = _make_instance(sw, [_kernel_case("add")])
    ins._check_expected_impl_source()  # 未配置时不做任何校验


def test_expect_impl_source_cli_argument_registered():
    from ttk.cli.kernel import register_kernel_command

    parser = argparse.ArgumentParser(prog="ttk")
    sub = parser.add_subparsers(dest="command")
    register_kernel_command(sub)
    args = parser.parse_args(["kernel", "-i", "x.csv", "--expect-impl-source", "vendor=acme"])
    assert args.expect_impl_source == "vendor=acme"
    # 未传时默认 None（不改变既有行为）
    args = parser.parse_args(["kernel", "-i", "x.csv"])
    assert args.expect_impl_source is None


def test_hash_cases_to_groups_preserves_input_order():
    """同 hash 组内保持传入序（set 迭代序不稳定的回归锚）。"""
    cases = []
    for name in ("c1", "a1", "b1"):
        case = MagicMock()
        case.testcase_name = name
        case.get_compilation_hash.return_value = 42
        cases.append(case)
    groups = TestcaseKernel.hash_cases_to_groups(cases)
    assert list(groups) == [42]
    assert [c.testcase_name for c in groups[42]] == ["c1", "a1", "b1"]
    # 重复用例（同对象）仍去重并告警
    dup = cases + [cases[0]]
    groups = TestcaseKernel.hash_cases_to_groups(dup)
    assert [c.testcase_name for c in groups[42]] == ["c1", "a1", "b1"]
