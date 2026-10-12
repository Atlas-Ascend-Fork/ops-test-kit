#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""test_mode 是分发唯一权威：解析器类型按命令分发，错配输入按三态预期拦截。

issue: https://gitcode.com/cann/ops-test-kit/issues/158

预期行为四档（探针实证）:
  ok      解析成功（匹配格/放行格）
  raise   解析期 RuntimeError: Could not determine value of missing fields（必填列缺失）
  typeerr 解析期 TypeError: None is not a valid shapelike value（output_shapes 无列走 default 分支）
  invalid 解析成功但 validate 后 is_valid=False（软失败）
"""

import pytest

from ttk.switches import SWITCHES, get_global_storage, set_global_storage

KERNEL_CSV = "examples/case_store/kernel/cases.csv"
ACLNN_CSV = "examples/case_store/aclnn/cases.csv"
E2E_CSV = "examples/case_store/e2e/torch_cases.csv"
GEIR_CSV = "examples/case_store/geir/cases.csv"

# (test_mode, 输入, 预期解析器类型, 预期行为)
MATRIX = (
    # 匹配格
    ("op", KERNEL_CSV, "TestcaseKernel", "ok"),
    ("aclnn", ACLNN_CSV, "TestcaseAclnn", "ok"),
    ("framework-api", E2E_CSV, "TestcaseE2e", "ok"),
    ("geir", GEIR_CSV, "GeirTestcase", "ok"),
    # 硬报错格: 必填列缺失 -> RuntimeError
    ("aclnn", KERNEL_CSV, None, "raise"),
    ("aclnn", GEIR_CSV, None, "raise"),
    ("framework-api", KERNEL_CSV, None, "raise"),
    ("framework-api", GEIR_CSV, None, "raise"),
    # 硬报错格: output_shapes 无列 default 分支 -> TypeError
    ("op", ACLNN_CSV, None, "typeerr"),
    ("op", E2E_CSV, None, "typeerr"),
    ("geir", ACLNN_CSV, None, "typeerr"),
    ("geir", E2E_CSV, None, "typeerr"),
    # 软失败格: 解析成功, validate 拒绝
    ("aclnn", E2E_CSV, "TestcaseAclnn", "invalid"),  # _check_api_name: torch.add 不在 OpApiInfo
    ("framework-api", ACLNN_CSV, "TestcaseE2e", "invalid"),  # aclnnAdd 非合法框架 API
    # 放行格（表头超集关系, spec 已知局限）
    ("geir", KERNEL_CSV, "GeirTestcase", "ok"),
    ("op", GEIR_CSV, "TestcaseKernel", "ok"),
)


@pytest.fixture
def mode_env():
    original = get_global_storage()
    yield set_global_storage  # 测试体直接调用设值
    set_global_storage(original)


@pytest.fixture
def mock_keepers(monkeypatch):
    """隔离 CANN 环境依赖: pytest conftest 删 ASCEND_* 后,
    OpApiInfoKeeper 构造即抛错(info_keeper.py:98-101),
    OpInfoKeeper 首次 info_of 时抛 ASCEND_OPP_PATH is not set(platform.py:65),
    匹配格/放行格/软失败格无法到达断言。

    - OpApiInfoKeeper: 模块级 import, patch 模块级名（先例 test_testcase_aclnn.py:594）
    - OpInfoKeeper: testcase_op.py 函数内 import, patch 源类
    """
    from unittest.mock import MagicMock

    from ttk.core_modules.aclnn import testcase as ta_mod
    from ttk.core_modules.kernel import info_keeper as oik_mod

    op_info = MagicMock()
    op_info.info_of.return_value = MagicMock()  # 非None -> _check_op_name 通过
    monkeypatch.setattr(oik_mod, "OpInfoKeeper", MagicMock(return_value=op_info))

    api_keeper = MagicMock()
    api_keeper.has_api.return_value = False  # 软失败格: api 不存在 -> is_valid=False
    monkeypatch.setattr(ta_mod, "OpApiInfoKeeper", MagicMock(return_value=api_keeper))


def _factory(path, test_mode):
    """testcase_class 参数化后的构造（模式推断已删——调用方显式传类）。"""
    from ttk.core_modules.common.testcase import UniversalTestcaseFactory
    from ttk.core_modules.common.testcase.reader import iter_tables

    cls = {
        "op": __import__("ttk.core_modules.kernel.testcase", fromlist=["TestcaseKernel"]).TestcaseKernel,
        "aclnn": __import__("ttk.core_modules.aclnn.testcase", fromlist=["TestcaseAclnn"]).TestcaseAclnn,
        "framework-api": __import__("ttk.core_modules.framework_api.testcase", fromlist=["TestcaseE2e"]).TestcaseE2e,
        "geir": __import__("ttk.core_modules.geir.testcase", fromlist=["GeirTestcase"]).GeirTestcase,
    }[test_mode]
    table = next(iter_tables(path))
    return UniversalTestcaseFactory(table, cls)


@pytest.mark.parametrize(("test_mode", "case_file", "parser_type", "behavior"), MATRIX)
def test_dispatch_matrix(test_mode, case_file, parser_type, behavior, mode_env, mock_keepers):
    if test_mode == "framework-api":
        pytest.importorskip("torch")  # e2e 解析依赖 torch 探测 API 签名
    sw = SWITCHES()
    sw.test_mode = test_mode
    mode_env(sw)

    if behavior == "raise":
        with pytest.raises(RuntimeError, match="Could not determine value of missing fields"):
            _factory(case_file, test_mode)
        return
    if behavior == "typeerr":
        with pytest.raises(TypeError, match="is not a valid shapelike value"):
            _factory(case_file, test_mode)
        return

    factory = _factory(case_file, test_mode)
    assert type(factory.testcase_instance).__name__ == parser_type
    if behavior == "invalid":
        assert factory.testcases  # 防空列表空转通过
        for tc in factory.testcases:
            assert tc.is_valid is False
