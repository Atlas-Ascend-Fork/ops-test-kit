# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""rerun 筛选 pin 测试（聚合层上移的变更对象）。

现状：筛选在 Factory._parse（staticmethod 读 get_global_storage）；按设计将上移
parse_cases 聚合层（选择器随迁 instance_base）。本文件钉模块级 _check_testcase_rerun
的现状语义（迁移后逻辑等价性对照）：
  - 未设 rerun_targets → 全保留
  - 目标列原值 "FAIL"（非数值非 PASS）→ 保留重跑
  - 目标列原值 "PASS" → 跳过
  - 目标列原值为数值（如 precision 0.9987）→ 跳过
  - 不支持的列名 → RuntimeError
"""

import pytest

from ttk.core_modules.common.instance import _check_testcase_rerun
from ttk.switches import SWITCHES, get_global_storage, set_global_storage


class _PinCase:
    """最小用例桩：rerun 判定只消费 supported_rerun_title + original_dict。"""

    def __init__(self, original_dict):
        self.original_dict = original_dict

    @staticmethod
    def supported_rerun_title():
        return ("precision_status", "dyn_kernel_name")


@pytest.fixture
def storage():
    prev = get_global_storage()
    sw = SWITCHES()
    set_global_storage(sw)
    yield sw
    set_global_storage(prev)


def test_no_rerun_targets_keeps_all(storage):
    storage.rerun_targets = None
    assert _check_testcase_rerun(_PinCase({})) is True


def test_fail_value_is_rerun(storage):
    storage.rerun_targets = ["precision_status"]
    case = _PinCase({"precision_status": "FAIL"})
    assert _check_testcase_rerun(case) is True


def test_pass_value_is_skipped(storage):
    storage.rerun_targets = ["precision_status"]
    case = _PinCase({"precision_status": "PASS"})
    assert _check_testcase_rerun(case) is False


def test_numeric_value_is_skipped(storage):
    storage.rerun_targets = ["precision_status"]
    case = _PinCase({"precision_status": "0.9987"})
    assert _check_testcase_rerun(case) is False


def test_unsupported_title_raises(storage):
    storage.rerun_targets = ["not_a_column"]
    with pytest.raises(RuntimeError, match="Unsupported rerun targets"):
        _check_testcase_rerun(_PinCase({}))
