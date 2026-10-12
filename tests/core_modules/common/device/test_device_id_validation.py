# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for device-pool composition: dispatch-time per-case device-id validation,
--devices vs hardware check, and DSMI health filtering."""

from types import SimpleNamespace

import pytest

from ttk.core_modules.common.device.dsmi.info import DSMI_HEALTH_STATE
from ttk.core_modules.common.instance import InstanceBase
from ttk.core_modules.common.process.task import Task, TaskKeeper
from ttk.switches import MODE


class _FakePG:
    def __init__(self, dev_id, idle=1, pool="profile"):
        self.dev_id = dev_id
        self._idle = idle
        self.pool = pool
        self.pushed = []

    def idle_count(self):
        return self._idle

    def push(self, task):
        self.pushed.append(task)


class _DispatchHarness:
    def __init__(self, grantable, pool_devs):
        self.device_locks = tuple(True if i in set(grantable) else None for i in range(max(grantable, default=-1) + 1))
        self.process_groups = {d: _FakePG(d) for d in pool_devs}
        self.task_keeper = TaskKeeper()
        self.failures = []
        self.completed = 0
        self.flushed = []
        self._multi_device_running = False
        # 派发 choke point 的 dl=3 缺字段提示所需最小契约
        self.switches = SimpleNamespace(no_prof=False, manual_data_dirs=())
        self._batch_relation_warned = set()

    def testcase_complete(self):
        self.completed += 1

    def handle_error(self, task, kind, detail):
        if kind == "fail":
            self.failures.append((task, str(detail[1])))
            return ("row",)
        return None

    def write_row(self, row):
        self.flushed.append(row)

    _fail_task_without_worker = InstanceBase._fail_task_without_worker
    _try_push_multi_device_task = InstanceBase._try_push_multi_device_task
    _most_idle_group_in_pool = InstanceBase._most_idle_group_in_pool
    _grantable_devices = InstanceBase._grantable_devices
    _push_task_to_process = InstanceBase._push_task_to_process
    _warn_incomplete_batch_relation = InstanceBase._warn_incomplete_batch_relation


def _task(ids):
    return Task(
        SimpleNamespace(testcase_name="hccl_case", is_enabled=True, is_valid=True),
        print,
        (),
        task_type="profile",
        pool="profile",
        device_ids=list(ids),
    )


class TestIntersectDeviceIds:
    """CSV device_ids ∩ 可用池 的纯决策函数。"""

    def test_full_overlap_passthrough(self):
        from ttk.core_modules.common.instance import intersect_device_ids

        effective, failure = intersect_device_ids([0, 1, 2], {0, 1, 2, 3})
        assert effective == [0, 1, 2]
        assert failure is None

    def test_partial_ge2_shrinks(self):
        from ttk.core_modules.common.instance import intersect_device_ids

        effective, failure = intersect_device_ids([0, 1, 2], {0, 1})
        assert effective == [0, 1]
        assert failure is None

    def test_one_left_fails_with_clear_reason(self):
        from ttk.core_modules.common.instance import intersect_device_ids

        effective, failure = intersect_device_ids([0, 7], {0, 1})
        assert effective == [0]
        assert failure is not None
        assert "requires >= 2 devices" in failure
        assert "only 1 device left" in failure

    def test_empty_intersection_fails(self):
        from ttk.core_modules.common.instance import intersect_device_ids

        effective, failure = intersect_device_ids([5, 7], {0, 1})
        assert effective == []
        assert failure is not None
        assert "= ∅" in failure
        assert "available devices [0, 1]" in failure

    def test_order_preserved(self):
        from ttk.core_modules.common.instance import intersect_device_ids

        effective, _ = intersect_device_ids([2, 0, 3], {0, 1, 2})
        assert effective == [2, 0]


class TestTryPushMultiDeviceTask:
    def test_intersection_one_left_fails_case_only(self):
        """CSV (0,7) ∩ 可用 {0,1} = [0]：不降级单卡，整例判死且信息明确。"""
        h = _DispatchHarness(grantable=[0, 1], pool_devs=[0, 1])
        task = _task((0, 7))
        assert InstanceBase._try_push_multi_device_task(h, task) is False
        assert h.completed == 1
        assert len(h.flushed) == 1
        msg = h.failures[0][1]
        assert "requires >= 2 devices" in msg
        assert "[0, 7]" in msg
        assert "available devices [0, 1]" in msg
        assert h.task_keeper.count() == 0

    def test_intersection_empty_fails_case(self):
        h = _DispatchHarness(grantable=[0, 1], pool_devs=[0, 1])
        task = _task((5, 7))
        assert InstanceBase._try_push_multi_device_task(h, task) is False
        assert h.completed == 1
        msg = h.failures[0][1]
        assert "∩ available devices [0, 1] = ∅" in msg
        assert h.task_keeper.count() == 0

    def test_intersection_shrinks_and_dispatches(self):
        """CSV (0,1,2) ∩ 可用 {0,1} = [0,1]：按交集截断继续多卡。"""
        h = _DispatchHarness(grantable=[0, 1], pool_devs=[0])
        task = _task((0, 1, 2))
        task.testcase = SimpleNamespace(device_ids=(0, 1, 2), testcase_name="hccl_case")
        assert InstanceBase._try_push_multi_device_task(h, task) is True
        assert task.device_ids == [0, 1]
        assert h.process_groups[0].pushed == [task]
        assert h.completed == 0
        assert task.testcase.device_ids == (0, 1)  # 结果 CSV 回显实跑设备集

    def test_primary_outside_dispatch_pool_fails_case(self):
        h = _DispatchHarness(grantable=list(range(8)), pool_devs=[0, 1])
        task = _task((5, 6))
        assert InstanceBase._try_push_multi_device_task(h, task) is False
        assert h.completed == 1
        assert "primary device 5 outside dispatch pool [0, 1]" in h.failures[0][1]
        assert h.task_keeper.count() == 0

    def test_busy_pool_requeues_without_failure(self):
        h = _DispatchHarness(grantable=[0, 1], pool_devs=[0])
        h.process_groups[0]._idle = 0
        task = _task((0, 1))
        assert InstanceBase._try_push_multi_device_task(h, task) is False
        assert h.completed == 0
        assert h.failures == []
        assert h.task_keeper.count() == 1

    def test_member_grantable_outside_pool_dispatches(self):
        h = _DispatchHarness(grantable=list(range(8)), pool_devs=[0])
        task = _task((0, 1, 2, 3, 4, 5, 6, 7))
        assert InstanceBase._try_push_multi_device_task(h, task) is True
        assert h.process_groups[0].pushed == [task]
        assert h.completed == 0


class TestPushTaskToProcessPinnedSingle:
    """CSV 单值 device_ids（"3" 指定单卡）的 pin 路由（文档语义恢复）。"""

    def test_pinned_routes_to_pinned_group(self):
        h = _DispatchHarness(grantable=[0, 3], pool_devs=[0, 3])
        h.process_groups[0]._idle = 2  # 最闲组是 0，但 pin 决定路由
        task = _task((3,))
        h.task_keeper.append([task])
        h._push_task_to_process()
        assert h.process_groups[3].pushed == [task]
        assert h.process_groups[0].pushed == []
        assert h.task_keeper.count() == 0

    def test_pinned_busy_defers_without_blocking(self):
        """pin 组忙：延后回队首，不阻塞同池其它 task 派发。"""
        h = _DispatchHarness(grantable=[0, 3], pool_devs=[0, 3])
        h.process_groups[3]._idle = 0
        pinned, normal = _task((3,)), _task(())
        normal.device_ids = None
        h.task_keeper.append([pinned, normal])
        h._push_task_to_process()
        assert h.process_groups[0].pushed == [normal]
        assert h.process_groups[3].pushed == []
        assert h.task_keeper.count() == 1  # pinned 回队

    def test_pinned_not_in_pool_fails_with_clear_reason(self):
        h = _DispatchHarness(grantable=[0, 1], pool_devs=[0, 1])
        task = _task((5,))
        h.task_keeper.append([task])
        h._push_task_to_process()
        assert h.completed == 1
        msg = h.failures[0][1]
        assert "pinned device 5" in msg
        assert "available devices: [0, 1]" in msg
        assert h.task_keeper.count() == 0

    def test_unpinned_goes_to_most_idle(self):
        h = _DispatchHarness(grantable=[0, 1], pool_devs=[0, 1])
        h.process_groups[1]._idle = 2
        task = _task(())
        task.device_ids = None
        h.task_keeper.append([task])
        h._push_task_to_process()
        assert h.process_groups[1].pushed == [task]


def test_e2e_build_tasks_passes_single_device_through():
    """e2e build_tasks 单值透传（原实现丢弃 → dispatch 无法 pin）。"""
    from ttk.core_modules.framework_api.instance import FrameworkApiInstance
    from ttk.core_modules.framework_api.testcase import TestcaseE2e

    case = TestcaseE2e()
    case.testcase_name = "pin_case"
    case.device_ids = (3,)
    tasks = FrameworkApiInstance.build_tasks(SimpleNamespace(), [case])
    assert tasks[0].device_ids == [3]


class _FakeDSMI:
    physical = 0

    def __init__(self):
        pass

    def get_device_count(self):
        return _FakeDSMI.physical


def _hw_inst(**overrides):
    values = {"mode": MODE.ASCEND_ONBOARD, "compile_only": False, "backend": None, "device_count": 5}
    values.update(overrides)
    return SimpleNamespace(switches=SimpleNamespace(**values))


class TestValidateDeviceCountAgainstHardware:
    @pytest.fixture(autouse=True)
    def _fake_dsmi(self, monkeypatch):
        monkeypatch.setattr("ttk.core_modules.common.instance.DSMIInterface", _FakeDSMI)
        _FakeDSMI.physical = 0

    def test_count_exceeds_physical_raises(self):
        _FakeDSMI.physical = 2
        with pytest.raises(RuntimeError) as exc:
            InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5))
        assert "valid device IDs: 0-1" in str(exc.value)

    def test_count_within_physical_passes(self):
        _FakeDSMI.physical = 8
        InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5))

    def test_no_physical_devices_skipped(self):
        _FakeDSMI.physical = 0
        InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5))

    def test_auto_detect_count_skipped(self):
        _FakeDSMI.physical = 2
        InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=-1))

    def test_compile_only_skipped(self):
        _FakeDSMI.physical = 2
        InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5, compile_only=True))

    def test_npusim_skipped(self):
        _FakeDSMI.physical = 2
        InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5, backend="npusim"))

    def test_model_mode_skipped(self):
        _FakeDSMI.physical = 2
        InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5, mode=MODE.ASCEND_CAMODEL))

    def test_dsmi_failure_skipped(self):
        class _BrokenDSMI:
            def __init__(self):
                raise OSError("no driver")

        import ttk.core_modules.common.instance as inst_mod

        orig = inst_mod.DSMIInterface
        inst_mod.DSMIInterface = _BrokenDSMI
        try:
            InstanceBase._validate_device_count_against_hardware(_hw_inst(device_count=5))
        finally:
            inst_mod.DSMIInterface = orig


class _HealthDSMI:
    states = {}

    def __init__(self):
        pass

    def get_device_health_states(self, device_ids):
        if any(d not in self.states for d in device_ids):
            return None
        return {d: self.states[d] for d in device_ids}


def _health_inst(**overrides):
    values = {
        "mode": MODE.ASCEND_ONBOARD,
        "compile_only": False,
        "backend": None,
        "no_prof": False,
        "manual_data_dirs": (),
    }
    values.update(overrides)
    return SimpleNamespace(switches=SimpleNamespace(**values))


@pytest.fixture
def _fake_health_dsmi(monkeypatch):
    monkeypatch.setattr("ttk.core_modules.common.instance.DSMIInterface", _HealthDSMI)
    _HealthDSMI.states = {}


class TestFilterUnhealthyDevices:
    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_mixed_states_excludes_only_broken(self):
        _HealthDSMI.states = {
            0: DSMI_HEALTH_STATE.OK,
            1: DSMI_HEALTH_STATE.WARNING,
            2: DSMI_HEALTH_STATE.IMPORTANT,
            3: DSMI_HEALTH_STATE.CRITICAL,
            4: DSMI_HEALTH_STATE.OFF,
            5: 0x99,
        }
        excluded = InstanceBase._filter_unhealthy_devices(_health_inst(), [0, 1, 2, 3, 4, 5])
        assert excluded == (2, 3, 4, 5)

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_none_query_failure_keeps_device(self):
        _HealthDSMI.states = {0: DSMI_HEALTH_STATE.OK, 1: None}
        assert InstanceBase._filter_unhealthy_devices(_health_inst(), [0, 1]) == ()

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_dsmi_unavailable_warns_and_skips(self, caplog):
        assert InstanceBase._filter_unhealthy_devices(_health_inst(), [0, 1]) == ()
        assert any("Device health check skipped" in r.message for r in caplog.records)

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_compile_only_skipped(self):
        assert InstanceBase._filter_unhealthy_devices(_health_inst(compile_only=True), [0, 1]) == ()

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_npusim_skipped(self):
        assert InstanceBase._filter_unhealthy_devices(_health_inst(backend="npusim"), [0, 1]) == ()

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_manual_dump_skipped(self):
        assert (
            InstanceBase._filter_unhealthy_devices(_health_inst(no_prof=True, manual_data_dirs=("out",)), [0, 1]) == ()
        )

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_model_mode_skipped(self):
        assert InstanceBase._filter_unhealthy_devices(_health_inst(mode=MODE.ASCEND_CAMODEL), [0, 1]) == ()

    @pytest.mark.usefixtures("_fake_health_dsmi")
    def test_empty_device_list_skipped(self):
        assert InstanceBase._filter_unhealthy_devices(_health_inst(), []) == ()
