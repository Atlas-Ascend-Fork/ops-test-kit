# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""KernelInstance 编译错误路径与双池 task 结构回归：编译模式经
task.compile_mode 还原；build_tasks 前置 profile task + depends_on。"""

from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.common.process import SimpleCommandProcess, Task
from ttk.core_modules.kernel.compile.result import DynamicCompilationResult
from ttk.core_modules.kernel.instance import KernelInstance
from ttk.core_modules.kernel.testcase import TestcaseKernel


@pytest.fixture
def switches():
    sw = MagicMock()
    sw.random_seed = None
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        yield sw


def _make_case(name="t0"):
    case = MagicMock(spec=TestcaseKernel)
    case.testcase_name = name
    case.fail_reason = "invalid testcase spec"
    case.ready_for_profile.return_value = False
    case.output_shape_unknown_indexes = ()  # MagicMock 自动属性为 truthy，显式置空
    return case


def _make_instance(switches):
    with patch("ttk.core_modules.kernel.instance.get_global_storage", return_value=switches):
        return KernelInstance()


class TestCompileInvalidCase:
    def test_dyn_mode_crash_result_applied(self, switches):
        ki = _make_instance(switches)
        case = _make_case()
        task = Task(case, int, (), task_type="compile_dyn", pool="compile", is_final=False)
        out = ki.handle_error(task, "none", None)
        case.apply_compile_result.assert_called_once()
        assert isinstance(case.apply_compile_result.call_args[0][0], DynamicCompilationResult)
        assert out is None  # ready_for_profile=False → 不转入 profiling

    def test_profile_none_result_raises(self, switches):
        ki = _make_instance(switches)
        task = Task(_make_case(), int, (), task_type="profile", pool="profile")
        with pytest.raises(RuntimeError, match="returned None"):
            ki.handle_error(task, "none", None)


class TestBuildTasksDualPool:
    """build_tasks：编译 + profile task 一次性建齐，profile depends_on
    全部 compile task——替代旧 compile_done 回调式 _send_to_profiling。"""

    def _enable_modes(self, switches, dyn=True, cst=True, bin_=True):
        switches.dyn_switches.enabled = dyn
        switches.cst_switches.enabled = cst
        switches.bin_switches.enabled = bin_

    def test_profile_task_depends_on_all_compile_tasks(self, switches):
        self._enable_modes(switches)
        ki = _make_instance(switches)
        case = _make_case()
        ki.task_keeper.append(ki.build_tasks([case]))
        keeper = ki.task_keeper
        assert keeper.count("compile") == 3
        assert keeper.count("profile") == 1
        assert keeper.pop("profile") is None  # 依赖未就绪不放行
        compile_ids = []
        while (ct := keeper.pop("compile")) is not None:
            compile_ids.append(ct.task_id)
        for tid in compile_ids:
            keeper.done(tid)
        pt = keeper.pop("profile")
        assert sorted(pt.depends_on) == sorted(compile_ids)
        assert pt.is_final is True
        assert pt.params[0] is case
        assert pt.params[1] is SimpleCommandProcess._device_grant_events
        assert pt.params[2] is SimpleCommandProcess._device_granted_indices

    def test_all_modes_disabled_skips_case(self, switches):
        self._enable_modes(switches, dyn=False, cst=False, bin_=False)
        ki = _make_instance(switches)
        case = _make_case()
        ki.task_keeper.append(ki.build_tasks([case]))
        assert ki.skipped_cases == 1
        assert ki.task_keeper.count() == 0
        assert case.apply_compile_result.call_count == 3  # 三个 *_OFF 结果落位

    def test_same_hash_sibling_waits_for_anchor(self, switches):
        """同 hash 兄弟用例：编译 task 依赖锚点（弹出即取 semaphore 结果），
        消除 worker 内 sleep(2) 自旋占池；profile 各依赖本用例编译。"""
        self._enable_modes(switches, cst=False, bin_=False)
        ki = _make_instance(switches)
        c1, c2 = _make_case("a"), _make_case("b")
        c1.get_compilation_hash = lambda: 42
        c2.get_compilation_hash = lambda: 42
        ki.task_keeper.append(ki.build_tasks([c1, c2]))
        keeper = ki.task_keeper
        anchor = keeper.pop("compile")
        assert anchor is not None
        assert anchor.depends_on == []
        assert keeper.pop("compile") is None  # 兄弟被锚点闸住
        keeper.done(anchor.task_id)
        sibling = keeper.pop("compile")
        assert sibling is not None
        assert sibling.depends_on == [anchor.task_id]
        keeper.done(sibling.task_id)
        profiles = [keeper.pop("profile"), keeper.pop("profile")]
        assert {p.depends_on[0] for p in profiles} == {anchor.task_id, sibling.task_id}

    def test_distinct_hash_no_cross_dependency(self, switches):
        self._enable_modes(switches, cst=False, bin_=False)
        ki = _make_instance(switches)
        c1, c2 = _make_case("a"), _make_case("b")
        c1.get_compilation_hash = lambda: 1
        c2.get_compilation_hash = lambda: 2
        ki.task_keeper.append(ki.build_tasks([c1, c2]))
        keeper = ki.task_keeper
        first = keeper.pop("compile")
        second = keeper.pop("compile")  # 不同 hash：不被锚点闸住
        assert first.depends_on == []
        assert second.depends_on == []


class TestHandleResult:
    def test_compile_result_applied_and_none_returned(self, switches):
        ki = _make_instance(switches)
        case = _make_case()
        task = Task(case, int, (), task_type="compile_dyn", pool="compile", is_final=False)
        result = MagicMock(spec=DynamicCompilationResult)
        assert ki.handle_result(task, result) is None
        case.apply_compile_result.assert_called_once_with(result)

    def test_compile_result_type_checked(self, switches):
        ki = _make_instance(switches)
        case = _make_case()
        task = Task(case, int, (), task_type="compile_dyn", pool="compile", is_final=False)
        with pytest.raises(RuntimeError, match="BaseCompilationResult"):
            ki.handle_result(task, object())

    def test_recycle_worker_only_for_profile_failure(self, switches):
        from ttk.core_modules.kernel.result import KernelReturnStructure

        ki = _make_instance(switches)
        profile_task = Task(_make_case(), int, (), task_type="profile", pool="profile")
        compile_task = Task(_make_case(), int, (), task_type="compile_dyn", pool="compile", is_final=False)

        ok = MagicMock(spec=KernelReturnStructure)
        ok.kernel_execute_failed.return_value = False
        bad = MagicMock(spec=KernelReturnStructure)
        bad.kernel_execute_failed.return_value = True

        assert ki.recycle_worker(profile_task, ok) is False
        assert ki.recycle_worker(profile_task, bad) is True
        assert ki.recycle_worker(compile_task, bad) is False
