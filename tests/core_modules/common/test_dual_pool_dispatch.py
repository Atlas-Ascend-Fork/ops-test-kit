# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""双池调度单测：pop(pool=) 池键路由 + 池间独立 +
depends_on 依赖闸门 + 完成即 done() 解锁。进程组以桩替代（不 spawn 真进程）。"""

from types import SimpleNamespace

from ttk.core_modules.common.instance import InstanceBase
from ttk.core_modules.common.process.task import Task, TaskKeeper


class _FakePG:
    def __init__(self, key, pool, idle=1):
        self.key = key
        self.pool = pool
        self._idle = idle
        self.pushed = []
        self._completed = []

    def idle_count(self):
        return self._idle

    def push(self, task):
        self.pushed.append(task)

    def completed(self):
        ret = tuple(self._completed)
        self._completed = []
        return ret


def _task(pool="profile", **kw):
    defaults = {
        "testcase": SimpleNamespace(testcase_name="t0", is_enabled=True, is_valid=True),
        "func": print,
        "params": (),
        "task_type": "compile_dyn" if pool == "compile" else "profile",
    }
    defaults.update(kw)
    return Task(pool=pool, **defaults)


class _DispatchHarness:
    """把 InstanceBase 的调度方法挂到轻量桩上（同 test_device_id_validation 范式）。"""

    _push_task_to_process = InstanceBase._push_task_to_process
    _most_idle_group_in_pool = InstanceBase._most_idle_group_in_pool
    _try_push_multi_device_task = InstanceBase._try_push_multi_device_task
    _fail_task_without_worker = InstanceBase._fail_task_without_worker
    _handle_completed_process = InstanceBase._handle_completed_process
    _warn_incomplete_batch_relation = InstanceBase._warn_incomplete_batch_relation

    def __init__(self, groups):
        self.process_groups = groups
        self.task_keeper = TaskKeeper()
        self._multi_device_running = False
        self.device_locks = ()
        self.profile_object = SimpleNamespace(handle_task_result_runtime_error=lambda *a, **k: ("row",))
        self.completed = 0
        self.flushed = []
        self.handled = []
        # 派发 choke point 的 dl=3 缺字段提示所需最小契约
        self.switches = SimpleNamespace(no_prof=False, manual_data_dirs=())
        self._batch_relation_warned = set()

    def testcase_complete(self):
        self.completed += 1

    def _flush(self, row):
        self.flushed.append(row)

    def _handle_task_result(self, task, proc):
        self.handled.append(task)

    def _handle_multi_device_task_result(self, task, proc):
        self.handled.append(task)


class TestPoolRouting:
    def test_compile_task_routed_to_compile_pool_only(self):
        compile_pg = _FakePG("compile", "compile")
        prof_pg = _FakePG(0, "profile")
        h = _DispatchHarness({"compile": compile_pg, 0: prof_pg})
        h.task_keeper.append(_task(pool="compile"))
        h._push_task_to_process()
        assert [t.pool for t in compile_pg.pushed] == ["compile"]
        assert prof_pg.pushed == []

    def test_profile_task_never_enters_compile_pool(self):
        """执行 task 只能进执行池——编译池空闲也不放行（物理隔离）。"""
        compile_pg = _FakePG("compile", "compile", idle=2)
        prof_pg = _FakePG(0, "profile", idle=0)
        h = _DispatchHarness({"compile": compile_pg, 0: prof_pg})
        h.task_keeper.append(_task(pool="profile"))
        h._push_task_to_process()
        assert compile_pg.pushed == []
        assert h.task_keeper.count() == 1  # 留队等执行池空闲

    def test_unmet_dependency_blocks_profile_not_compile(self):
        """池间独立：profile 依赖未就绪不阻塞编译池派发。"""
        compile_pg = _FakePG("compile", "compile")
        prof_pg = _FakePG(0, "profile")
        h = _DispatchHarness({"compile": compile_pg, 0: prof_pg})
        ct = _task(pool="compile")
        pt = _task(pool="profile", depends_on=[ct.task_id])
        h.task_keeper.append([ct, pt])
        h._push_task_to_process()
        assert compile_pg.pushed == [ct]
        assert prof_pg.pushed == []

    def test_profile_dispatches_after_dependency_done(self):
        prof_pg = _FakePG(0, "profile")
        h = _DispatchHarness({0: prof_pg})
        ct = _task(pool="compile")
        pt = _task(pool="profile", depends_on=[ct.task_id])
        h.task_keeper.append([ct, pt])
        h.task_keeper.done(ct.task_id)
        h._push_task_to_process()
        assert prof_pg.pushed == [pt]

    def test_multi_device_dispatch_sets_gate_same_tick(self):
        """多卡派发置闸门后本轮不再派发其它池（执行池优先序保证确定性）。"""
        prof_pg = _FakePG(0, "profile")
        compile_pg = _FakePG("compile", "compile")
        h = _DispatchHarness({0: prof_pg, "compile": compile_pg})
        h.device_locks = (True, True)
        md = _task(pool="profile", device_ids=[0, 1])
        ct = _task(pool="compile")
        h.task_keeper.append([md, ct])
        h._push_task_to_process()
        assert h._multi_device_running is True
        assert prof_pg.pushed == [md]
        assert compile_pg.pushed == []  # 闸门置位即返回，编译 task 下轮再派


class TestHandleCompletedDone:
    def test_completed_task_marks_done_unlocking_dependents(self):
        compile_pg = _FakePG("compile", "compile")
        h = _DispatchHarness({"compile": compile_pg, 0: _FakePG(0, "profile")})
        ct = _task(pool="compile")
        pt = _task(pool="profile", depends_on=[ct.task_id])
        h.task_keeper.append([ct, pt])
        assert h.task_keeper.pop("profile") is None  # 未解锁前取不到
        compile_pg._completed.append((object(), ct))
        h._handle_completed_process()
        assert h.handled == [ct]
        assert h.task_keeper.pop("profile") is pt  # done() 已解锁
