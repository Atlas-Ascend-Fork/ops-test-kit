#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""任务与进程基础设施：Task/TaskKeeper（任务单元与队列）、ProcessGroup
（一池一组 worker 的生命周期与收发）、SimpleCommandProcess（worker RPC）、
worker 入口（bootstrap + RPC 循环 + 崩溃诊断）与子进程侧设备锁。

mp_context 必须是 forkserver：worker fork 自 forkserver 时父进程尚未
import torch/tf 等重框架，规避 fork 已初始化框架多线程状态的死锁；禁止
改回 fork/spawn。initialize_device_locks()（锁表 + Manager Event/Value）
必须先于建池执行——worker 经任务参数持有这些共享对象。
"""

__all__ = [
    "Task",
    "TaskKeeper",
    "ProcessGroup",
    "describe_exit",
    "process_context",
    "ProcessContext",
    "DeviceLock",
    "MultiDeviceLock",
    "gate_host_memory",
]

from .memory_gate import gate_host_memory
from .process_group import ProcessGroup
from .simple_command import (
    DeviceLock,
    MultiDeviceLock,
    ProcessContext,
    SimpleCommandProcess,
    describe_exit,
    intermediate_func,
    process_context,
    worker_bootstrap,
)
from .task import Task, TaskKeeper
