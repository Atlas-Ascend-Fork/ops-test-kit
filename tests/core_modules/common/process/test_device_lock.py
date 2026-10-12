# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#!/usr/bin/env python3
"""
Device lock integration tests against production DeviceLockManager + DeviceLock
(simple_command.py). Real subprocesses; scenarios:
1. Basic: 1 kill + recovery
2. Multi-device: 2 devices with separate lock state
3. Stress: 8 children, kill 4 sequentially
4. Normal release: no kill, plain acquire/release
5. No device: use_device=False, no lock acquired, all run concurrently
6. No device + kill: use_device=False with child kill
"""

import multiprocessing as mp
import time

mp.set_start_method("forkserver", force=True)

from ttk.core_modules.common.process.simple_command import (  # noqa: E402
    DeviceLock,
    DeviceLockManager,
)


class MockProcessContext:
    """Mock ProcessContext for testing — simulates get_lock/release_lock RPC via pipe."""

    def __init__(self, pipe_w):
        self.pipe_w = pipe_w

    def get_lock(self, device_id, lock_id, grant_event, granted_idx):
        self.pipe_w.send(("REQUEST", lock_id, device_id))

    def release_lock(self, device_id):
        self.pipe_w.send(("RELEASED", device_id))


def child_work(pipe_w, idx, device_id, use_device=True, grant_event=None, granted_idx=None, work_time=2):
    """Acquire lock via DeviceLock, work, release."""
    try:
        mock_ctx = MockProcessContext(pipe_w)
        lock = DeviceLock(mock_ctx, device_id, use_device=use_device, grant_event=grant_event, granted_idx=granted_idx)
        with lock:
            pipe_w.send(("ACQUIRED", idx, device_id))
            time.sleep(work_time)
    except Exception as e:
        pipe_w.send(("ERROR", idx, str(e)))


def run_test(test_name, num_children, num_devices, kill_plan, use_device=True, timeout=15):
    """启动若干子进程并按 kill_plan 杀进程，验证 DeviceLockManager 的获取/释放/回收语义。"""
    print(f"{'=' * 60}")
    print(f"Test: {test_name}")
    print(f"  children={num_children}, devices={num_devices}, use_device={use_device}, kill_plan={kill_plan}")
    print(f"{'=' * 60}")

    DeviceLockManager.lock_holders.clear()
    DeviceLockManager.pending.clear()

    manager = mp.Manager()

    # Initialize lock state per device
    grant_events = {}
    granted_indices = {}
    if use_device:
        for d in range(num_devices):
            grant_events[d] = manager.Event()
            granted_indices[d] = manager.Value("i", -1)
            DeviceLockManager.lock_holders[d] = None
            DeviceLockManager.pending[d] = []

    # Start children — assign device round-robin
    pipes = []
    procs = []
    for i in range(num_children):
        dev = i % max(num_devices, 1)
        pipe_r, pipe_w = mp.Pipe()
        p = mp.Process(
            target=child_work, args=(pipe_w, i, dev, use_device, grant_events.get(dev), granted_indices.get(dev))
        )
        p.start()
        pipes.append((pipe_r, pipe_w))
        procs.append({"proc": p, "dev": dev, "idx": i, "dead_processed": False, "lock_id": None})

    acquired_order = []
    released_order = []
    killed = set()
    killed_plan = set()

    start = time.time()

    while time.time() - start < timeout:
        elapsed = time.time() - start

        # Execute kill plan
        if kill_plan:
            for delay, child_idx in kill_plan:
                if child_idx not in killed_plan and elapsed >= delay:
                    print(f"  [{elapsed:.1f}s] >>> KILL child {child_idx} (pid={procs[child_idx]['proc'].pid}) <<<")
                    procs[child_idx]["proc"].kill()
                    killed_plan.add(child_idx)

        # Process messages and detect dead children
        for i, (pipe_r, _) in enumerate(pipes):
            if procs[i]["proc"].exitcode is not None and not procs[i]["dead_processed"]:
                procs[i]["dead_processed"] = True
                if i not in killed:
                    killed.add(i)
                if use_device:
                    DeviceLockManager.on_process_dead(procs[i], grant_events, granted_indices)
                continue

            if procs[i]["proc"].exitcode is not None:
                continue

            if pipe_r.poll(0.01):
                msg = pipe_r.recv()
                if msg[0] == "REQUEST":
                    _, lock_id, dev_id = msg
                    procs[i]["lock_id"] = lock_id
                    DeviceLockManager.try_grant(
                        procs[i], dev_id, lock_id, grant_events[dev_id], granted_indices[dev_id]
                    )
                elif msg[0] == "ACQUIRED":
                    acquired_order.append(msg[1])
                    print(f"  [{elapsed:.1f}s] child {msg[1]} acquired dev {msg[2]}")
                elif msg[0] == "RELEASED":
                    dev_id = msg[1]
                    released_order.append(dev_id)
                    print(f"  [{elapsed:.1f}s] child {i} released dev {dev_id}")
                    DeviceLockManager.release(procs[i], dev_id, grant_events[dev_id], granted_indices[dev_id])
                elif msg[0] == "ERROR":
                    print(f"  [{elapsed:.1f}s] child {msg[1]} ERROR: {msg[2]}")

        # Check if all done
        alive = sum(1 for p in procs if not p["dead_processed"])
        if alive == 0:
            break
        if use_device and len(acquired_order) >= num_children and len(released_order) + len(killed) >= num_children:
            break

    # Collect results
    for p in procs:
        if p["proc"].is_alive():
            p["proc"].kill()
        p["proc"].join()

    manager.shutdown()

    if not use_device:
        success = len(set(acquired_order)) >= (num_children - len(killed_plan))
        print(f"\n  Acquired (no lock): {acquired_order}")
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        print()
        return success

    # Verify: all non-killed children eventually acquired
    non_killed_acquired = [i for i in acquired_order if i not in killed_plan]
    success = len(set(non_killed_acquired)) >= (num_children - len(killed_plan))

    print(f"\n  Acquired order: {acquired_order}")
    print(f"  Released: {released_order}")
    print(f"  Killed: {sorted(killed_plan)}")
    print(f"  Non-killed acquisitions: {len(set(non_killed_acquired))} (need >= {num_children - len(killed_plan)})")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    print()
    return success


def test_1_basic():
    """基础：杀 1 个持锁子进程后，下一个等待者能获取锁。"""
    assert run_test(
        "Basic: kill 1 child, verify next acquires",
        num_children=4,
        num_devices=1,
        kill_plan=[(1.5, 0)],
    )


def test_2_multi_device():
    """多设备：2 个设备各自独立锁状态，互不干扰。"""
    assert run_test(
        "Multi-device: 2 devices, kill 1 on each",
        num_children=4,
        num_devices=2,
        kill_plan=[(1.0, 0), (1.0, 1)],
    )


def test_3_stress():
    """压力：8 个子进程 1 设备，连续杀前 4 个，剩余均能获取锁（覆盖多次
    kill→回收循环，与单次 kill 同路径的规模化验证）。"""
    assert run_test(
        "Stress: 8 children, 1 device, kill first 4 sequentially",
        num_children=8,
        num_devices=1,
        kill_plan=[(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3)],
        timeout=20,
    )


def test_4_normal():
    """正常路径：无杀进程，所有子进程依次获取并释放锁。"""
    assert run_test(
        "Normal: no kills, all acquire and release",
        num_children=4,
        num_devices=1,
        kill_plan=None,
    )


def test_5_no_device():
    """无设备模式：use_device=False，所有子进程并发运行无需锁。"""
    assert run_test(
        "No device: use_device=False, all children run concurrently without lock",
        num_children=4,
        num_devices=1,
        kill_plan=None,
        use_device=False,
    )


def test_6_no_device_with_kill():
    """无设备 + 杀进程：use_device=False 下杀 1 个子进程，其余正常完成。"""
    assert run_test(
        "No device + kill: use_device=False, kill child 0, others complete",
        num_children=4,
        num_devices=1,
        kill_plan=[(1.0, 0)],
        use_device=False,
    )
