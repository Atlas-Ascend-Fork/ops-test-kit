#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for aclnn npu_memory (NPU workspace size) result column:
- AclnnReturnStructure exposes npu_memory as a result title (column).
- ApiProfilingResult carries npu_memory; construct() propagates it from prof_result.
"""

import types

from ttk.core_modules.aclnn.result import (
    AclnnReturnStructure,
    ApiComparisonResult,
    ApiProfilingResult,
)


def test_npu_memory_is_result_column():
    """npu_memory 应为结果列（出现在 get_titles() 中）。"""
    titles = AclnnReturnStructure.get_titles()
    assert "npu_memory" in titles


def test_npu_memory_default_none():
    """未执行 profiling 时 npu_memory 默认为 None。"""
    prs = AclnnReturnStructure()
    assert prs.npu_memory is None
    # context=None 路径（早期失败）保持 None
    prs.construct(None, ApiComparisonResult(None))
    assert prs.npu_memory is None


def test_npu_memory_flows_through_construct():
    """aclnn GetWorkspaceSize 返回的 workspace_size 经 ApiProfilingResult → construct() 写入 npu_memory。"""
    prof_result = ApiProfilingResult(True, npu_memory=4096)
    context = types.SimpleNamespace(prof_result=prof_result, api_name="aclnnXxx")
    prs = AclnnReturnStructure()
    prs.construct(context, ApiComparisonResult(None))
    assert prs.npu_memory == 4096


def test_npu_memory_pick_data():
    """pick_data 能按 title 取出 npu_memory，验证结果表格行可填充该列。"""
    prof_result = ApiProfilingResult(True, npu_memory=8192)
    context = types.SimpleNamespace(prof_result=prof_result, api_name="aclnnXxx")
    prs = AclnnReturnStructure()
    prs.construct(context, ApiComparisonResult(None))
    titles = AclnnReturnStructure.get_titles()
    row = prs.pick_data(titles)
    assert row[titles.index("npu_memory")] == 8192
