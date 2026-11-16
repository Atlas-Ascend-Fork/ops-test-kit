# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""复检展开测试：rounds=1 直通（兼容）、多轮全后缀展开（命名/交错序/行号
继承/副本独立）、混名链式叠加、重名兜底、reseed 每轮种子语义、四通路
supported_rerun_title。"""

import random

import pytest

from ttk.core_modules.common.instance import InstanceBase
from ttk.switches import SWITCHES, get_global_storage, set_global_storage


class _FakeInstance(InstanceBase):
    """最小可实例化通路（同 test_instance_base 范式）。"""

    name = "fake"

    def prepare_env(self):
        pass

    def detect_platform(self):
        pass

    def detect_device_count(self):
        pass

    def setup(self):
        pass

    def build_tasks(self, testcases):
        return []

    def handle_result(self, task, result):
        return ("PASS", "PASS")

    def result_titles(self):
        return ("precision_status",)

    def get_device_info(self, dev_id):
        return ""


class SimpleCase:
    """deepcopy 友好的最小用例桩（声明面 struct）。"""

    def __init__(self, name):
        self.testcase_name = name
        self.op_name = "Add"
        self._csv_row_index = 0


@pytest.fixture
def storage():
    prev = get_global_storage()
    sw = SWITCHES()
    set_global_storage(sw)
    yield sw
    set_global_storage(prev)


def test_expand_rounds_one_passthrough(storage):
    storage.rerun_rounds = 1
    inst = _FakeInstance()
    inst.flatten_testcases = [SimpleCase("A"), SimpleCase("B")]
    inst._expand_recheck_cases()
    assert [t.testcase_name for t in inst.flatten_testcases] == ["A", "B"]


def test_expand_rounds_three(storage):
    storage.rerun_rounds = 3
    original = SimpleCase("A")
    original._csv_row_index = 7
    inst = _FakeInstance()
    inst.flatten_testcases = [original, SimpleCase("B")]
    inst._expand_recheck_cases()
    # 全后缀交错序；_csv_row_index 继承；轮 0 为原件原地改名、轮 1+ 为独立副本
    names = [t.testcase_name for t in inst.flatten_testcases]
    assert names == ["A__rc0", "A__rc1", "A__rc2", "B__rc0", "B__rc1", "B__rc2"]
    assert all(t._csv_row_index == 7 for t in inst.flatten_testcases[:3])
    assert inst.flatten_testcases[0] is original
    copy_tc = inst.flatten_testcases[1]
    assert copy_tc is not original
    copy_tc.op_name = "Mul"
    assert original.op_name == "Add"


def test_expand_mixed_rc_names_stack(storage):
    # 复检输出链式再检（同含 A 与 A__rc1）→ 后缀叠加，不冲突
    storage.rerun_rounds = 2
    inst = _FakeInstance()
    inst.flatten_testcases = [SimpleCase("A"), SimpleCase("A__rc1")]
    inst._expand_recheck_cases()
    assert [t.testcase_name for t in inst.flatten_testcases] == [
        "A__rc0",
        "A__rc1",
        "A__rc1__rc0",
        "A__rc1__rc1",
    ]


def test_expand_duplicate_original_names_raise(storage):
    # 重名原件正常由 _check_duplicate_case 前置拦截，seen 断言兜底报错
    storage.rerun_rounds = 2
    inst = _FakeInstance()
    inst.flatten_testcases = [SimpleCase("A"), SimpleCase("A")]
    with pytest.raises(RuntimeError, match="recheck copy name collision"):
        inst._expand_recheck_cases()


class _NamedCase:
    def __init__(self, name):
        self.testcase_name = name


def test_reseed_rounds_seed_semantics():
    from ttk.core_modules.common.testcase.input_data import reseed_per_testcase

    sw = SWITCHES()
    sw.random_seed = None
    assert reseed_per_testcase(_NamedCase("A"), sw) is False
    sw.random_seed = 42

    def draw(name):
        assert reseed_per_testcase(_NamedCase(name), sw) is True
        return random.getrandbits(64)

    # 各轮种子互异、同轮名跨运行可复现、轮 0（__rc0）与原名种子不同（新采样）
    assert len({draw("A__rc0"), draw("A__rc1"), draw("A__rc2")}) == 3
    assert draw("A__rc1") == draw("A__rc1")
    assert draw("A") != draw("A__rc0")


def test_supported_rerun_titles():
    from ttk.core_modules.aclnn.testcase import TestcaseAclnn
    from ttk.core_modules.framework_api.testcase import TestcaseE2e

    assert set(TestcaseAclnn.supported_rerun_title()) == {"precision_status", "perf_status", "deterministic_status"}
    # e2e 精度值列（eager/graph_*_precision）不纳入
    assert set(TestcaseE2e.supported_rerun_title()) == {"precision_status", "deterministic_status"}
