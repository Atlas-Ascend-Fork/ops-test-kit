# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Task / TaskKeeper 单测：调度不变式三条。

不变式：
1. pop() 返回的 task 的 depends_on 已全部 done
2. task_id 全局唯一（自动生成）
3. empty() 为真 ⇔ 队列内无未弹出 task
"""

from unittest.mock import MagicMock

from ttk.core_modules.common.process import Task, TaskKeeper


def _task(name="t0", **kw):
    defaults = {
        "testcase": MagicMock(testcase_name=name),
        "func": lambda: None,
        "params": (),
    }
    defaults.update(kw)
    return Task(**defaults)


class TestTask:
    def test_default_is_final_profile(self):
        t = _task()
        assert t.task_type == "profile"
        assert t.pool == "profile"
        assert t.is_final is True
        assert t.depends_on == []
        assert t.task_id  # 自动生成

    def test_task_id_auto_unique(self):
        ids = {_task().task_id for _ in range(100)}
        assert len(ids) == 100

    def test_explicit_task_id_preserved(self):
        assert _task(task_id="deadbeef").task_id == "deadbeef"

    def test_is_multi_device_property(self):
        assert _task(device_ids=None).is_multi_device is False
        assert _task(device_ids=[3]).is_multi_device is False  # 指定单卡
        assert _task(device_ids=[0, 1, 2]).is_multi_device is True

    def test_label(self):
        assert _task().label == "profile:t0"
        assert _task(task_type="compile_dyn").label == "compile_dyn:t0"

    def test_compile_mode_derives_from_task_type(self):
        # sub_type 删除后编译模式从 task_type 还原（kernel 构造约定 compile_{mode.lower()}）
        assert _task(task_type="compile_dyn").compile_mode == "Dyn"
        assert _task(task_type="compile_cst").compile_mode == "Cst"
        assert _task(task_type="compile_bin").compile_mode == "Bin"


class TestTaskKeeperInvariants:
    def test_invariant_1_pop_only_returns_dependency_satisfied(self):
        keeper = TaskKeeper()
        a, b, c = _task("a"), _task("b"), _task("c")
        b.depends_on = [a.task_id]
        c.depends_on = [a.task_id]
        keeper.append([a, b, c])
        popped = [keeper.pop(), keeper.pop(), keeper.pop()]
        # b/c 依赖 a，a 未 done 前不可弹出
        assert popped[0].task_id == a.task_id
        assert popped[1] is None
        assert popped[2] is None
        keeper.done(a.task_id)
        assert keeper.pop().task_id in {b.task_id, c.task_id}
        assert keeper.pop().task_id in {b.task_id, c.task_id}

    def test_invariant_1_unmet_dependency_does_not_block_later(self):
        keeper = TaskKeeper()
        blocked, free = _task("blocked"), _task("free")
        blocked.depends_on = ["never-done"]
        keeper.append([blocked, free])
        assert keeper.pop().task_id == free.task_id  # 越过未就绪者
        assert keeper.pop() is None

    def test_invariant_2_same_testcase_multi_task_unique_ids(self):
        keeper = TaskKeeper()
        case = MagicMock(testcase_name="one")
        tasks = [Task(case, int, (), task_type=f"compile_{m}", pool="compile") for m in ("dyn", "cst", "bin")]
        tasks += [Task(case, int, (), task_type="profile", pool="profile")]
        keeper.append(tasks)
        assert len({t.task_id for t in tasks}) == 4

    def test_invariant_3_empty_semantics(self):
        keeper = TaskKeeper()
        assert keeper.empty() is True
        keeper.append(_task())
        assert keeper.empty() is False
        keeper.pop()
        assert keeper.empty() is True
        # done 集不参与 empty 判定
        keeper.done("whatever")
        assert keeper.empty() is True

    def test_done_is_idempotent(self):
        keeper = TaskKeeper()
        a, b = _task("a"), _task("b")
        b.depends_on = [a.task_id]
        keeper.append([a, b])
        assert keeper.pop().task_id == a.task_id
        keeper.done(a.task_id)
        keeper.done(a.task_id)  # 幂等
        assert keeper.pop().task_id == b.task_id


class TestPoolFiltering:
    def test_pop_filters_by_pool(self):
        keeper = TaskKeeper()
        compile_t = _task("c", task_type="compile_dyn", pool="compile")
        profile_t = _task("p", task_type="profile", pool="profile")
        keeper.append([compile_t, profile_t])
        assert keeper.pop("profile").task_id == profile_t.task_id
        assert keeper.pop("profile") is None
        assert keeper.pop("compile").task_id == compile_t.task_id

    def test_count_by_pool(self):
        keeper = TaskKeeper()
        keeper.append([_task(pool="compile"), _task(pool="compile"), _task(pool="profile")])
        assert keeper.count() == 3
        assert keeper.count("compile") == 2
        assert keeper.count("profile") == 1


class TestInsertFront:
    def test_insert_puts_front(self):
        keeper = TaskKeeper()
        keeper.append(_task("first"))
        keeper.insert(_task("front"))
        assert keeper.pop().testcase.testcase_name == "front"
