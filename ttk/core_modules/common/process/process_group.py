#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""ProcessGroup：一池长驻 worker 进程的管理员。

职责：进程创建、task 分发、结果收取、进程回收。主循环只面对本类的公开
方法即可操控任意多个进程池（同 dev_id 双池 = 两个实例）。

不做的事：不决定下一个发哪个 task（TaskKeeper+调度器）、不关心 task 内容
（通路段逻辑在 worker 函数）、不感知通路（pool/dev_id 只是构造参数）。
"""

__all__ = ["ProcessGroup"]

# Standard Packages
import logging
import time
from multiprocessing.context import BaseContext
from typing import Dict, Set, Tuple

from .simple_command import SimpleCommandProcess
from .task import Task


class ProcessGroup:
    """一个进程池的管理器：N 个长驻 worker 进程的创建、分发、收取、回收。
    不感知通路与 task 语义——池键（pool）与设备绑定只是构造参数。"""

    def __init__(self, pool: str, dev_id: int, process_count: int, mp_context: BaseContext, timeout: int = 0):
        """pool:          池路由键（与 Task.pool 匹配；同 dev_id 双池各一组）
        dev_id:        绑卡号；编译池传组序号（不绑卡）
        process_count: worker 数（进程数算术在调用方，本类只管"给我 N 个进程"）
        timeout:       worker 无响应超时（秒，0=不判）
        """
        self.pool = pool
        self.dev_id = dev_id
        self.processes = tuple(
            SimpleCommandProcess(mp_context, name=f"D{dev_id}P{i}", timeout=timeout) for i in range(process_count)
        )
        self._process_to_task: Dict[SimpleCommandProcess, Task] = {}

    # ── 分发与收取（run_loop 主调用面）──

    def push(self, task: Task) -> None:
        """交给空闲 worker。多卡语义读 task.device_ids 装配 kwargs
        （dev_id/device_ids/is_multi_device）；无空闲 worker 抛 RuntimeError。"""
        for proc in self.processes:
            if proc.is_idle():
                logging.debug(f"Sending {task.label} task to process pid {proc.get_pid()}")
                self._process_to_task[proc] = task
                kwargs = {"dev_id": self.dev_id}
                if task.is_multi_device:
                    kwargs = {
                        "dev_id": task.device_ids[0],
                        "is_multi_device": True,
                        "device_ids": task.device_ids,
                    }
                proc.send_action(task.func, task.params, kwargs)
                break
        else:
            raise RuntimeError(f"[BUG] no idle process in pool {self.pool!r} dev{self.dev_id}. It maybe a BUG of TTK.")

    def completed(self) -> Tuple[Tuple[SimpleCommandProcess, Task], ...]:
        """返回本组已完成且尚未收取的 [(proc, task)]，取走的自动清占用映射。"""
        ret = []
        for proc in self.processes:
            if proc.is_completed() and proc in self._process_to_task:
                ret.append((proc, self._process_to_task[proc]))
                del self._process_to_task[proc]
        return tuple(ret)

    def idle_count(self) -> int:
        return sum(proc.is_idle() for proc in self.processes)

    # ── 生命周期 ──

    def is_ready(self) -> bool:
        return all(proc.is_ready() for proc in self.processes)

    def update(self) -> None:
        """刷新 worker 状态，并回收意外死亡的空闲 worker。

        OOM kill 等意外死亡的空闲 worker（无在跑 task）不在 completed()
        的收取范围（那里只收"有 task"的），若不回收则池容量永久损失——
        task 饥饿 → completed_case_count 永不达标 → run_loop 死等。

        判据用现成信号：意外死亡的 worker 的 rpc_results 里留有 update()
        放入的 SystemError；主动关停（close_idles/close_all）results 为空，
        天然不会被回收（无乒乓）。
        """
        for proc in self.processes:
            proc.update()
            if proc in self._process_to_task or not proc.is_dead() or proc.rpc_results.empty():
                continue
            while not proc.rpc_results.empty():
                stale = proc.rpc_results.get()
            logging.warning(
                "Worker %s died idle while not holding a task (discarded: %s); respawning", proc.name, stale
            )
            proc.resurrect()

    def close_idles(self) -> None:
        for proc in self.processes:
            if proc.is_idle():
                proc.close()

    def close_all(self):
        for proc in self.processes:
            proc.close()

    # ── 观测 ──

    def busy_types(self) -> Set[str]:
        """在跑 task 的 task_type 集合（task 完成并被 completed() 收取后移出）。"""
        return {task.task_type for task in self._process_to_task.values()}

    def info(self) -> str:
        return "\n".join(
            f"{proc.get_pid()} "
            f"{proc.name.ljust(16) if len(proc.name) < 16 else proc.name[-16:]} "
            f"{proc.current_stage().ljust(20)} "
            f"{proc.status.name.ljust(8)} "
            f"{int(time.time() - proc.process_status_timestamp)}s"
            for proc in self.processes
            if not proc.is_dead()
        )
