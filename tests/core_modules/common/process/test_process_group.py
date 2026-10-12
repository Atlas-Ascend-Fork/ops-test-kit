# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ProcessGroup 单测：push 单参装配 / busy_types / completed / 生命周期
（close_idles/close_all/info）。进程本体以桩替代（不 spawn 真进程）。"""

import multiprocessing
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.common.process import ProcessGroup, Task


def _task(**kw):
    defaults = {
        "testcase": MagicMock(testcase_name="t0"),
        "func": int,
        "params": (),
        "task_type": "profile",
        "pool": "profile",
    }
    defaults.update(kw)
    return Task(**defaults)


class _ResultsStub:
    """SimpleQueue 桩：empty/get 语义（排水用）。"""

    def __init__(self, items=()):
        self._items = list(items)

    def empty(self):
        return not self._items

    def get(self):
        return self._items.pop(0)


class _ProcStub:
    """SimpleCommandProcess 桩：状态可控。"""

    def __init__(self, idle=True, completed=False, dead=False):
        self.is_idle_ret = idle
        self.completed_ret = completed
        self.dead_ret = dead
        self.sent = None
        self.closed = False
        self.get_pid = lambda: 4242
        self.name = "D0P0"
        self.status = SimpleNamespace(name="IDLE")
        self.process_status_timestamp = time.time()
        self.resurrected = 0
        self.rpc_results = _ResultsStub()

    def is_idle(self):
        return self.is_idle_ret

    def is_completed(self):
        return self.completed_ret

    def is_dead(self):
        return self.dead_ret

    def update(self):
        pass

    def resurrect(self):
        self.resurrected += 1
        self.dead_ret = False

    def send_action(self, func, args, kwargs):
        self.sent = (func, args, kwargs)

    def close(self):
        self.closed = True

    def current_stage(self):
        return "OnProfiling"


def _group(procs, pool="default", dev_id=0):
    with patch.object(ProcessGroup, "__init__", lambda self, *a, **k: None):
        pg = ProcessGroup.__new__(ProcessGroup)
    pg.pool = pool
    pg.dev_id = dev_id
    pg.processes = tuple(procs)
    pg._process_to_task = {}
    return pg


class TestPushSingleParam:
    def test_push_assembles_dev_id_kwarg(self):
        proc = _ProcStub()
        pg = _group([proc])
        task = _task()
        pg.push(task)
        assert proc.sent[2] == {"dev_id": 0}
        assert pg._process_to_task[proc] is task

    def test_push_multi_device_reads_task_self_description(self):
        proc = _ProcStub()
        pg = _group([proc], dev_id=7)
        task = _task(device_ids=[0, 1, 2])
        pg.push(task)
        # 旧 (is_multi_device=True, rank_dev_id=primary_dev) 三参形态删除：
        # dev_id = device_ids[0]，device_ids 随 kwargs 注入
        assert proc.sent[2] == {"dev_id": 0, "is_multi_device": True, "device_ids": [0, 1, 2]}

    def test_push_no_idle_raises_bug(self):
        pg = _group([_ProcStub(idle=False)])
        with pytest.raises(RuntimeError, match="BUG"):
            pg.push(_task())


class TestCompletedAndBusyTypes:
    def test_completed_returns_pairs_and_clears(self):
        proc = _ProcStub(completed=True)
        pg = _group([proc])
        task = _task()
        pg._process_to_task[proc] = task
        pairs = pg.completed()
        assert pairs == ((proc, task),)
        assert pg.completed() == ()  # 取走自动清占用映射

    def test_busy_types_returns_running_task_types(self):
        busy, idle = _ProcStub(idle=False), _ProcStub(idle=True)
        pg = _group([busy, idle])
        pg._process_to_task[busy] = _task(task_type="compile_dyn")
        assert pg.busy_types() == {"compile_dyn"}

    def test_idle_count(self):
        assert _group([_ProcStub(), _ProcStub(idle=False)]).idle_count() == 1


class TestConstructionSmoke:
    """双池构造冒烟：process_count=0 不 spawn 真进程，仅验证构造共存。"""

    def test_dual_pool_groups_coexist_per_device(self):
        mp_ctx = multiprocessing.get_context("forkserver")
        compile_pool = ProcessGroup("compile", 0, 0, mp_ctx)
        profile_pool = ProcessGroup("profile", 0, 0, mp_ctx)
        assert compile_pool.pool == "compile"
        assert profile_pool.pool == "profile"
        assert compile_pool.dev_id == profile_pool.dev_id == 0


class TestLifecycle:
    def test_close_idles_only_closes_idle(self):
        idle, busy = _ProcStub(idle=True), _ProcStub(idle=False)
        pg = _group([idle, busy])
        pg.close_idles()
        assert idle.closed
        assert not busy.closed

    def test_close_all_closes_busy_too(self):
        busy = _ProcStub(idle=False)
        pg = _group([busy])
        pg.close_all()
        assert busy.closed

    def test_info_lists_only_alive_procs(self):
        alive, dead = _ProcStub(), _ProcStub(dead=True)
        pg = _group([alive, dead])
        info = pg.info()
        assert "4242" in info
        assert "OnProfiling" in info
        assert "\n" not in info  # dead 进程不占行


class TestReapDeadIdleWorkers:
    """update() 回收意外死亡的空闲 worker（OOM kill 场景，防 run_loop 死等）。

    判据：dead + 无在跑 task + rpc_results 里留有未消费的 SystemError
    （update() 的 EOF 死亡路径放入）；主动关停 results 为空，不回收。
    """

    def test_dead_idle_worker_is_respawned_with_stale_result_drained(self):
        dead_idle = _ProcStub(idle=True, dead=True)
        dead_idle.rpc_results = _ResultsStub([SystemError("D0P0 exited unexpectedly")])
        alive = _ProcStub()
        pg = _group([dead_idle, alive])
        pg.update()
        assert dead_idle.resurrected == 1
        assert dead_idle.rpc_results.empty()  # 遗留 SystemError 已排水，不留脏结果

    def test_dead_worker_holding_task_is_left_to_scheduler(self):
        """带 task 死亡走 completed() → _handle_task_result 的 resurrect，不在此回收。"""
        dead_busy = _ProcStub(idle=False, dead=True)
        dead_busy.rpc_results = _ResultsStub([SystemError("crashed")])
        pg = _group([dead_busy])
        pg._process_to_task[dead_busy] = _task()
        pg.update()
        assert dead_busy.resurrected == 0

    def test_cleanly_closed_worker_is_not_respawned(self):
        """主动关停（close_idles/close_all）results 为空 → 不回收（无乒乓）。"""
        closed = _ProcStub(idle=True, dead=True)
        pg = _group([closed])
        pg.update()
        assert closed.resurrected == 0

    def test_alive_worker_untouched(self):
        alive = _ProcStub()
        pg = _group([alive])
        pg.update()
        assert alive.resurrected == 0
        assert not alive.closed
