#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for attributes1~attributes9 extension columns merged into attributes.
"""

import pytest

from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.testcase.base import TestcaseBase
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.core_modules.kernel.testcase import TestcaseKernel

CLS_LIST = [TestcaseKernel, TestcaseAclnn, TestcaseE2e]


def _make(cls, **attrs):
    case = cls()
    case.is_valid = True
    case.fail_reason = None
    case.testcase_name = "ext_attr_case"
    case.attributes = attrs.pop("attributes", None)
    for k, v in attrs.items():
        setattr(case, k, v)
    return case


@pytest.mark.parametrize("cls", CLS_LIST)
def test_headers_declared(cls):
    """验证 complete_headers 中包含 attributes 与 attributes1~9。"""
    hdrs = cls.complete_headers
    assert "attributes" in hdrs
    for i in range(1, 10):
        assert f"attributes{i}" in hdrs


@pytest.mark.parametrize("cls", CLS_LIST)
def test_merge_basic(cls):
    """attributes + attributes1 无冲突键 → 简单合并。"""
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2})
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 1, "b": 2}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_merge_override_with_base(cls):
    """attributes 与 attributes1 同键 → attributes1 覆盖。"""
    case = _make(cls, attributes={"a": 1}, attributes1={"a": 2})
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 2}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_override_across_extended_columns(cls):
    """attributes1 与 attributes2 同键 → 后者覆盖。"""
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2}, attributes2={"b": 3})
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 1, "b": 3}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_override_chain_last_wins(cls):
    """多列同键链式覆盖 → 最后一列（attributes9）胜出。"""
    case = _make(cls, attributes={"k": 0}, attributes1={"k": 1}, attributes5={"k": 5}, attributes9={"k": 9})
    TestcaseBase.validate(case)
    assert case.attributes == {"k": 9}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_multi_column_merge(cls):
    """多列无冲突键 → 全部合并。"""
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2}, attributes2={"c": 3}, attributes3={"d": 4})
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 1, "b": 2, "c": 3, "d": 4}


@pytest.mark.parametrize("cls", CLS_LIST)
def test_no_extension_leaves_attributes(cls):
    """无扩展列 → attributes 保持不变。"""
    case = _make(cls, attributes={"a": 1})
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 1}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_merge_when_attributes_none(cls):
    """attributes 为 None + attributes1 有值 → 合并后取 attributes1。"""
    case = _make(cls, attributes=None, attributes1={"b": 2})
    TestcaseBase.validate(case)
    assert case.attributes == {"b": 2}


@pytest.mark.parametrize("cls", CLS_LIST)
def test_skip_when_invalid(cls):
    """is_valid 为 False → 不执行合并，attributes 保持原值。"""
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2})
    case.is_valid = False
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 1}
