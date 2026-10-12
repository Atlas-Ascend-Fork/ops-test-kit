#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""worker 子进程 RPC 基建。

- SimpleCommandProcess：长驻 worker 的父进程侧代理（pipe RPC 客户端）。两类
  用途：① worker 池内执行 task（ProcessGroup 管理）；② 常驻服务型进程
  （如 Kernel 的 KnowledgeBase server，主进程直连）
- intermediate_func：worker 子进程入口——崩溃诊断装配 + bootstrap 后进入
  while True RPC 循环；异常边界：worker 函数抛任何异常 → traceback 打包
  RuntimeError 回传 + 进程自杀（BroadException 有意为之，RPC 边界必须全捕）
- worker_bootstrap：worker 进程初始化（activate + load_config + 日志 +
  随机种子 + keeper 快照恢复）
- ProcessContext：worker 进程侧上下文（notify_status 进度上报/设备锁）
- DeviceLock / MultiDeviceLock：子进程侧设备锁（与主进程 DeviceLockManager
  经 pipe 交互；多卡一次锁全卡）
- describe_exit：worker 退出码的信号解码与成因提示（崩溃上报消费）

keeper 快照通道：intermediate_func 第三参 keeper_state（纯数据 dict，键 =
keeper 模块路径，值 = 可 pickle 快照），worker_bootstrap 内经模块级
restore_keeper_snapshot() 恢复；恢复失败降级 debug 日志，worker 惰性重解析
兜底。主进程建池前经 SimpleCommandProcess._bootstrap_keeper_state 注入。
"""

__all__ = [
    "SimpleCommandProcess",
    "ProcessContext",
    "DeviceLock",
    "MultiDeviceLock",
    "describe_exit",
    "process_context",
    "worker_bootstrap",
    "intermediate_func",
]


# Standard Packages
import atexit
import faulthandler
import gc
import importlib
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from enum import Enum, auto
from queue import SimpleQueue
from typing import Any, Callable, Dict, List, Optional

import numpy

# Third-Party Packages
import psutil

from ttk.switches import get_global_storage

from ....utilities import set_process_name, set_thread_name, signal_registered
from ..logging import default_logging_config


class PROCESS_STATUS_CODE(Enum):
    CREATED = 0
    LAUNCHED = 114
    IDLE = 1919
    RUNNING = 810
    WAITING = 1551
    DEAD = 1145141919810


# --proc-timeout 判定：除协调性等待外，task 执行期任何 stage 停滞超时均判死。
# 状态门控（LAUNCHED/WAITING/RUNNING）保证空闲 worker 永不受罚；stage 采用
# 豁免白名单——stage 是通路自定义串，白名单式豁免让新增 stage 默认受超时保护
_TIMEOUT_ELIGIBLE_STATES = frozenset(
    {PROCESS_STATUS_CODE.LAUNCHED, PROCESS_STATUS_CODE.WAITING, PROCESS_STATUS_CODE.RUNNING}
)
# 设备锁获取是协调性等待：同卡他 worker 跑长用例时锁等待可合法超时——
# 锁持有方死亡由 on_process_dead 回收兜底（四通路锁 stage 统一 "OnAcquireLock"）
_TIMEOUT_EXEMPT_STAGES = frozenset({"OnAcquireLock"})


class PROCESS_RPC(Enum):
    EXECUTE_FUNCTION = auto()
    FUNCTION_RETURN = auto()
    ACQUIRE_SEMAPHORE = auto()
    SET_SEMAPHORE = auto()
    GET_SEMAPHORE = auto()
    RELEASE_SEMAPHORE = auto()
    GET_ACQUIRED_SEMAPHORES = auto()
    STORE_DATA = auto()
    GET_DATA = auto()
    CHANGE_NAME = auto()
    SUICIDE = auto()
    SEMAPHORE_DEAD_SEQUENCE = auto()
    GET_LOCK = auto()
    RELEASE_LOCK = auto()
    GET_MULTI_LOCK = auto()
    RELEASE_MULTI_LOCK = auto()


_process_context: Optional["ProcessContext"] = None


# 常见致命信号成因提示（exit -N 上报排障入口）。负退出码 = 被信号杀死；
# 正退出码是普通 exit——不得误标为信号（如 exit 1 ≠ SIGHUP）
_EXIT_SIGNAL_HINTS = {
    -6: "C-level assert/abort, often a driver or native library assertion",
    -7: "invalid memory access (bus error / alignment issue)",
    -9: "process killed forcibly, commonly the cgroup OOM killer or a manual kill; "
    "check `dmesg | grep -iE 'oom|killed process'` and the node memory limit",
    -11: "native library segfault, often a tbe/torch/tf symbol clash or a CANN version "
    "mismatch; faulthandler is enabled on workers and dumps the crash site to stderr",
    -15: "terminated, either the TTK timeout path (--proc-timeout) or an external kill; "
    "a terminated worker first dumps all-thread Python stacks to stderr",
}


def describe_exit(exitcode: Optional[int]) -> str:
    """worker 退出码的人类可读描述（负值 = 致命信号）。

    裸数字（exit -11 / -9）无诊断价值——解码信号名并附常见成因，
    供 update() 的 EOFError 分支与 KBS 死亡上报消费。"""
    if exitcode is None:
        return ""
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            return f"{exitcode}（未知信号）"
        hint = _EXIT_SIGNAL_HINTS.get(exitcode)
        return f"{exitcode} ({name})：{hint}" if hint else f"{exitcode} ({name})"
    return f"exit code {exitcode}"


def _install_crash_diagnostics() -> None:
    """worker 崩溃现场采集（exit -11/-15 类问题的"具体报错"来源）：

    - faulthandler：SIGSEGV/SIGABRT/SIGBUS/FPE/ILL 时在 stderr 打印
      全线程 Python 栈（进入致命数的原生调用前的最后 Python 帧）
    - SIGTERM 转储：父进程 terminate（超时/强杀路径）先 dump 全线程栈
      再按默认语义赴死——挂死类问题留下"卡在哪"的现场

    边界：原生调用内部的挂死不回 Python 信号处理器（彼时需 py-spy/gdb
    attach）；本机制只覆盖 Python 层可中断的挂死与崩溃。"""

    def _dump_and_default(signum, frame):
        faulthandler.dump_traceback()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    faulthandler.enable(all_threads=True)
    signal.signal(signal.SIGTERM, _dump_and_default)


class DeviceLockManager:
    """Parent-side device lock state manager."""

    lock_holders: Dict[int, "SimpleCommandProcess"] = {}
    pending: Dict[int, List["SimpleCommandProcess"]] = {}

    @classmethod
    def try_grant(cls, proc, device_id, lock_id, grant_event, granted_idx):
        if cls.lock_holders.get(device_id) is None:
            cls.lock_holders[device_id] = proc
            granted_idx.value = lock_id
            grant_event.set()
        else:
            cls.pending.setdefault(device_id, []).append((proc, lock_id))

    @classmethod
    def try_grant_multi(cls, proc, device_ids, lock_id, grant_events, granted_indices):
        for dev_id in device_ids:
            if cls.lock_holders.get(dev_id) is not None:
                for d in device_ids:
                    cls.pending.setdefault(d, []).append((proc, lock_id))
                return False
        for dev_id in device_ids:
            cls.lock_holders[dev_id] = proc
            granted_indices[dev_id].value = lock_id
            grant_events[dev_id].set()
        return True

    @classmethod
    def release(cls, proc, device_id, grant_event, granted_idx):
        if cls.lock_holders.get(device_id) is proc:
            cls.lock_holders[device_id] = None
            cls._grant_next(device_id, grant_event, granted_idx)

    @classmethod
    def release_multi(cls, proc, device_ids, grant_events, granted_indices):
        for dev_id in device_ids:
            if cls.lock_holders.get(dev_id) is proc:
                cls.lock_holders[dev_id] = None
                cls._grant_next(dev_id, grant_events[dev_id], granted_indices[dev_id])

    @classmethod
    def on_process_dead(cls, proc, grant_events, granted_indices):
        for dev_id, holder in list(cls.lock_holders.items()):
            if holder is proc:
                cls.lock_holders[dev_id] = None
                cls._grant_next(dev_id, grant_events[dev_id], granted_indices[dev_id])
        for dev_id in list(cls.pending.keys()):
            cls.pending[dev_id] = [(p, lid) for p, lid in cls.pending[dev_id] if p is not proc]

    @classmethod
    def _grant_next(cls, device_id, grant_event, granted_idx):
        waiters = cls.pending.get(device_id, [])
        if waiters:
            nxt_proc, nxt_lock_id = waiters.pop(0)
            cls.lock_holders[device_id] = nxt_proc
            granted_idx.value = nxt_lock_id
            grant_event.set()

    @classmethod
    def initialize(cls, available_devices):
        cls.lock_holders = dict.fromkeys(available_devices)
        cls.pending = {dev_id: [] for dev_id in available_devices}


class DeviceLock:
    """
    Context manager for child-side device lock.

    Args:
        process_ctx: ProcessContext instance (child-side)
        device_id: which device to lock
        use_device: if False, skip lock entirely (no-device / CPU mode)
        grant_event: Manager().Event() for grant signal
        granted_idx: Manager().Value('i', -1) for grant identity verification
    """

    def __init__(self, process_ctx, device_id, use_device=True, grant_event=None, granted_idx=None):
        self.process_ctx = process_ctx
        self.device_id = device_id
        self.use_device = use_device
        self.grant_event = grant_event
        self.granted_idx = granted_idx
        self._lock_id = os.getpid()

    def __enter__(self):
        if not self.use_device:
            return self
        self.process_ctx.get_lock(self.device_id, self._lock_id, self.grant_event, self.granted_idx)
        self.grant_event.wait()
        while self.granted_idx.value != self._lock_id:
            self.grant_event.wait()
        self.grant_event.clear()
        return self

    def __exit__(self, *args):
        if not self.use_device:
            return
        self.process_ctx.release_lock(self.device_id)


class MultiDeviceLock:
    """Context manager for child-side multi-device lock.

    Acquires locks for multiple devices simultaneously to support HCCL
    collective communication. device_ids are sorted ascending and granted
    all-or-nothing by the parent (no partial holding), so concurrent
    multi-device requests cannot deadlock on lock order.
    """

    def __init__(self, process_ctx, device_ids, use_device=True, grant_events=None, granted_indices=None):
        self.process_ctx = process_ctx
        self.device_ids = sorted(device_ids)
        self.use_device = use_device
        self.grant_events = grant_events or {}
        self.granted_indices = granted_indices or {}
        self._lock_id = os.getpid()

    def __enter__(self):
        if not self.use_device:
            return self
        sorted_events = {d: self.grant_events[d] for d in self.device_ids if d in self.grant_events}
        sorted_indices = {d: self.granted_indices[d] for d in self.device_ids if d in self.granted_indices}
        for dev_id in self.device_ids:
            if dev_id not in sorted_events or dev_id not in sorted_indices:
                raise RuntimeError(f"MultiDeviceLock: no grant event/idx for device {dev_id}")
        self.process_ctx.get_multi_lock(self.device_ids, self._lock_id, sorted_events, sorted_indices)
        for dev_id in self.device_ids:
            evt = sorted_events[dev_id]
            evt.wait()
            while sorted_indices[dev_id].value != self._lock_id:
                evt.wait()
            evt.clear()
        return self

    def __exit__(self, *args):
        if not self.use_device:
            return
        self.process_ctx.release_multi_lock(self.device_ids)


def process_context() -> "Optional[ProcessContext]":
    """获取本 worker 进程的上下文（worker 函数内使用；主进程返回 None）。"""
    return _process_context


def on_exit():
    if atexit._ncallbacks() > 1:  # 1 for Python internal callback
        atexit._run_exitfuncs()


def worker_bootstrap(global_storage, keeper_state: Optional[dict] = None):
    """Worker 进程初始化：设全局存储 + 加载配置 + 日志 + 随机种子 +
    keeper 快照恢复。

    从 intermediate_func 的 prologue 提取，可独立单测（intermediate_func 是
    while True RPC 循环，无法直接测）。load_config 让 forkserver worker
    拿到 --config 的配置（含 TLS）。"""
    global_storage.activate()
    from ttk.config.loader import load_config  # lazy import（函数体内，避循环依赖）

    load_config(global_storage.config_path)
    default_logging_config()  # 引导期 stdout only——worker 接到用例后重配 per-case 文件
    if global_storage.random_seed:
        numpy.random.seed(global_storage.random_seed)
    if keeper_state:
        _restore_keeper_snapshots(keeper_state)


ORPHAN_GUARD_INTERVAL_S = 2.0

# SUICIDE 后等待 worker 有序退出的宽限（秒）。须覆盖 atexit 清理链
# （rtDeviceReset 裸测 ~5s）；terminate 仅作真挂死兜底
SUICIDE_JOIN_TIMEOUT_S = 15


def _start_orphan_guard() -> None:
    """守护线程: 父进程一死就自杀, 避免 worker 变孤儿长期残留。

    已有保护只覆盖**空闲态** —— worker 阻塞在 ``pipe.recv()`` 时父死会让 recv 抛错从而自杀;
    但正在跑长任务(编译/派发, 可达数百秒)时感知不到, 要等任务结束才发现。历史上出现过
    孤儿 worker/forkserver 堆积吃满内存(``kill -9`` 下 atexit 钩子不会执行)。

    判据用"ppid 是否变化"而非 ``== 1``: 容器/systemd 这类 subreaper 会接管孤儿, ppid 不会变成 1
    (与 ttk/remote/heartbeat.py 的做法一致)。检查本身极廉价(实测 os.getppid() 约 0.09 微秒),
    2 秒间隔一天累计开销不到 10 毫秒。
    """
    original_ppid = os.getppid()

    def _guard():
        while True:
            if os.getppid() != original_ppid:
                # 父已死: 不走清理钩子(它们可能依赖已断的管道而卡住), 直接退出。
                os._exit(1)
            time.sleep(ORPHAN_GUARD_INTERVAL_S)

    threading.Thread(target=_guard, name="orphan-guard", daemon=True).start()


def _restore_keeper_snapshots(keeper_state: dict) -> None:
    """按模块路径恢复 keeper 快照（通路 info_keeper 的模块级入口）。

    键 = 通路 keeper 模块路径，值 = 可 pickle 快照；模块须提供
    restore_keeper_snapshot(snapshot)。恢复失败降级为 debug 日志——
    worker 侧惰性重解析兜底（快照是性能优化不是正确性前提）。"""

    for mod_path, snapshot in keeper_state.items():
        try:
            mod = importlib.import_module(mod_path)
            mod.restore_keeper_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001  # 快照恢复降级：worker 惰性重解析兜底
            logging.debug(f"Keeper snapshot restore skipped ({mod_path}): {exc}")


def intermediate_func(pipe: "mp.connection.Connection", global_storage, keeper_state: Optional[dict] = None) -> None:
    global _process_context
    _start_orphan_guard()
    _process_context = ProcessContext(pipe)
    _process_context.report_status(PROCESS_STATUS_CODE.LAUNCHED)  # 0x0114 -> Launched
    _install_crash_diagnostics()
    worker_bootstrap(global_storage, keeper_state)
    while True:
        _process_context.report_status(PROCESS_STATUS_CODE.IDLE)  # 0x1919 -> Ready for Command
        # noinspection PyBroadException
        try:
            message = pipe.recv()
        except Exception:
            # Silently suicide
            on_exit()
            raise RuntimeError("Subprocess is still alive but main process is dead.") from None
        if isinstance(message, tuple):
            rpc_command = message[0]
            if rpc_command == PROCESS_RPC.EXECUTE_FUNCTION:
                rpc_args = message[1]
                _process_context.report_status(PROCESS_STATUS_CODE.RUNNING)
                func: Callable = rpc_args[0]
                func_args: tuple = rpc_args[1]
                func_kwargs: dict = rpc_args[2]
                function_return = None
                # noinspection PyBroadException
                try:
                    function_return = func(*func_args, **func_kwargs)
                    _process_context.rpc_call((PROCESS_RPC.FUNCTION_RETURN, (function_return,)))
                except Exception:
                    _process_context.notify_status("OnExceptionReport")
                    function_return = RuntimeError(traceback.format_exc())
                    _process_context.rpc_call((PROCESS_RPC.FUNCTION_RETURN, (function_return,)))
                    _process_context.semaphore_dead_process(function_return)
                    _process_context.report_status(PROCESS_STATUS_CODE.DEAD)
                    pipe.close()
                    on_exit()
                    return
                finally:
                    del func, func_args, func_kwargs, function_return
                    gc.collect()
            elif rpc_command == PROCESS_RPC.SUICIDE:
                on_exit()
                if signal_registered(34):
                    os.kill(os.getpid(), 34)  # send signal to HDT for code coverage.
                sys.exit(-1)
            else:
                logging.warning(f"SimpleCommandProcess command pipe received invalid rpc call, ignored: {message}")
        else:
            logging.warning(f"SimpleCommandProcess command pipe received invalid command, ignored: {message}")


class ProcessContext:
    def __init__(self, pipe):
        self.pipe: mp.connection.Connection = pipe
        self.storage = {}

    def report_status(self, status: PROCESS_STATUS_CODE):
        self.pipe.send(status)

    def rpc_call(self, args):
        self.pipe.send(args)

    def acquire_semaphore(self, name) -> bool:
        self.pipe.send((PROCESS_RPC.ACQUIRE_SEMAPHORE, (name,)))
        return self.pipe.recv()

    def set_semaphore(self, name, value) -> None:
        self.pipe.send((PROCESS_RPC.SET_SEMAPHORE, (name, value)))

    def get_semaphore(self, name):
        self.pipe.send((PROCESS_RPC.GET_SEMAPHORE, (name,)))
        return self.pipe.recv()

    def get_acquired_semaphore(self) -> tuple:
        self.pipe.send((PROCESS_RPC.GET_ACQUIRED_SEMAPHORES, ()))
        return self.pipe.recv()

    def semaphore_dead_process(self, value):
        self.pipe.send((PROCESS_RPC.SEMAPHORE_DEAD_SEQUENCE, (value,)))

    def send_data(self, name, value) -> None:
        self.pipe.send((PROCESS_RPC.STORE_DATA, (name, value)))

    def get_data(self, name):
        self.pipe.send((PROCESS_RPC.GET_DATA, (name,)))
        return self.pipe.recv()

    def change_name(self, name: str) -> None:
        set_process_name(name)
        self.pipe.send((PROCESS_RPC.CHANGE_NAME, (name,)))

    def get_lock(self, device_id, lock_id, grant_event, granted_idx):
        self.pipe.send((PROCESS_RPC.GET_LOCK, (device_id, lock_id, grant_event, granted_idx)))

    def release_lock(self, device_id):
        self.pipe.send((PROCESS_RPC.RELEASE_LOCK, (device_id,)))

    def get_multi_lock(self, device_ids, lock_id, grant_events, granted_indices):
        self.pipe.send((PROCESS_RPC.GET_MULTI_LOCK, (device_ids, lock_id, grant_events, granted_indices)))

    def release_multi_lock(self, device_ids):
        self.pipe.send((PROCESS_RPC.RELEASE_MULTI_LOCK, (device_ids,)))

    def notify_status(self, status: str):
        set_thread_name(status)
        self.send_data("stage", status)


class SimpleCommandProcess:
    semaphore_to_holder: Dict[Any, "SimpleCommandProcess"] = {}
    holder_to_semaphores: Dict["SimpleCommandProcess", List[Any]] = {}
    semaphores: Dict[Any, Any] = {}
    all_processes: List["SimpleCommandProcess"] = []
    # Parent-managed device lock state
    _device_grant_events: Dict[int, Any] = {}
    _device_granted_indices: Dict[int, Any] = {}

    # 引导载荷（keeper 快照等，主进程建池前注入）
    _bootstrap_keeper_state: Optional[dict] = None

    def __init__(self, context, name="TBESimpleCommandProcess", daemon=None, timeout: int = 0):
        """context 必填（须传 forkserver 上下文——fork 会复制已初始化框架的
        多线程状态导致 worker 死锁）。"""
        self.original_input_params = (context, name, daemon, timeout)
        self.status: PROCESS_STATUS_CODE = PROCESS_STATUS_CODE.CREATED
        self.rpc_queue: SimpleQueue = SimpleQueue()
        self.rpc_results: SimpleQueue = SimpleQueue()
        self.locks: list = []
        self.process_status_timestamp = time.time()
        self.data: Dict[str, Any] = {}
        self.parent_pipe, self.child_pipe = context.Pipe()
        self.name = name
        self.timeout = timeout
        self.parent = context.Process(
            target=intermediate_func,
            name=name,
            args=(self.child_pipe, get_global_storage(), SimpleCommandProcess._bootstrap_keeper_state),
            daemon=daemon,
        )
        self.all_processes.append(self)
        self.parent.start()
        logging.debug(f"Process created with name {self.parent.name}")

    @classmethod
    def initialize_device_locks(cls, available_devices, mp_manager):
        """Create Event+Value per device and initialize DeviceLockManager."""
        for dev_id in available_devices:
            cls._device_grant_events[dev_id] = mp_manager.Event()
            cls._device_granted_indices[dev_id] = mp_manager.Value("i", -1)
        DeviceLockManager.initialize(available_devices)

    def _update(self):
        while self.parent_pipe.poll():
            message = self.parent_pipe.recv()
            if isinstance(message, PROCESS_STATUS_CODE):
                self.status = message
                self.process_status_timestamp = time.time()
                if message == PROCESS_STATUS_CODE.DEAD:
                    return
            elif isinstance(message, tuple):
                rpc_command = message[0]
                rpc_args = message[1]
                self._rpc_call(rpc_command, rpc_args)
            else:
                raise RuntimeError(f"SimpleCommandProcess received invalid command: {message}")
        if self.get_exitcode() is not None:
            raise EOFError()
        if (
            0 < self.timeout < int(time.time() - self.process_status_timestamp)
            and self.status in _TIMEOUT_ELIGIBLE_STATES
            and self.current_stage() not in _TIMEOUT_EXEMPT_STAGES
        ):
            raise TimeoutError()

    def _parent_send_rpc(self, rpc_command: PROCESS_RPC, rpc_args: tuple):
        self.parent_pipe.send((rpc_command, rpc_args))

    def _clear_data(self):
        self.data = {}

    def _rpc_call(self, rpc_command: PROCESS_RPC, rpc_args: tuple):
        if rpc_command == PROCESS_RPC.FUNCTION_RETURN:
            function_return = rpc_args[0]
            self.rpc_results.put(function_return)
            self._clear_data()
            self.name = self.original_input_params[1]
            self.parent.name = self.name
        elif rpc_command == PROCESS_RPC.STORE_DATA:
            name: str = rpc_args[0]
            value = rpc_args[1]
            self.data[name] = value
            if name == "stage":
                self._update_timestamp()
        elif rpc_command == PROCESS_RPC.GET_DATA:
            name: str = rpc_args[0]
            if name in self.data:
                self.parent_pipe.send(self.data[name])
            else:
                self.parent_pipe.send(None)
        elif rpc_command == PROCESS_RPC.CHANGE_NAME:
            name: str = rpc_args[0]
            self.name = name
            self.parent.name = name
        elif rpc_command == PROCESS_RPC.ACQUIRE_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphore_to_holder:
                self.parent_pipe.send(False)
            else:
                self.semaphore_to_holder[name] = self
                self.holder_to_semaphores.setdefault(self, []).append(name)
                self.semaphores[name] = None
                self.parent_pipe.send(True)
        elif rpc_command == PROCESS_RPC.SET_SEMAPHORE:
            name = rpc_args[0]
            value = rpc_args[1]
            if self.semaphore_to_holder[name] == self:
                self.semaphores[name] = value
            else:
                logging.warning(f"{self.name} trying to access semaphore of another process: {name}")
        elif rpc_command == PROCESS_RPC.GET_SEMAPHORE:
            name = rpc_args[0]
            value = self.semaphores.get(name, None)
            self.parent_pipe.send(value)
        elif rpc_command == PROCESS_RPC.RELEASE_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphores and self.semaphore_to_holder[name] == self:
                del self.semaphores[name]
                del self.semaphore_to_holder[name]
                self.holder_to_semaphores[self].remove(name)
            else:
                logging.warning(f"{self.name} trying to release invalid semaphore: {name}")
        elif rpc_command == PROCESS_RPC.GET_ACQUIRED_SEMAPHORES:
            if self in self.holder_to_semaphores:
                self.parent_pipe.send(self.holder_to_semaphores[self])
            else:
                self.parent_pipe.send(())
        elif rpc_command == PROCESS_RPC.SEMAPHORE_DEAD_SEQUENCE:
            value = rpc_args[0]
            self._semaphore_dead_sequence(value)
        elif rpc_command == PROCESS_RPC.GET_LOCK:
            device_id, lock_id, grant_event, granted_idx = rpc_args
            logging.debug(f"Get lock for device {device_id}")
            DeviceLockManager.try_grant(self, device_id, lock_id, grant_event, granted_idx)
        elif rpc_command == PROCESS_RPC.RELEASE_LOCK:
            device_id = rpc_args[0]
            logging.debug(f"Release lock for device {device_id}")
            DeviceLockManager.release(
                self,
                device_id,
                SimpleCommandProcess._device_grant_events[device_id],
                SimpleCommandProcess._device_granted_indices[device_id],
            )
        elif rpc_command == PROCESS_RPC.GET_MULTI_LOCK:
            device_ids, lock_id, grant_events, granted_indices = rpc_args
            logging.debug(f"Get multi-lock for devices {device_ids}")
            DeviceLockManager.try_grant_multi(self, device_ids, lock_id, grant_events, granted_indices)
        elif rpc_command == PROCESS_RPC.RELEASE_MULTI_LOCK:
            device_ids = rpc_args[0]
            logging.debug(f"Release multi-lock for devices {device_ids}")
            DeviceLockManager.release_multi(
                self,
                device_ids,
                SimpleCommandProcess._device_grant_events,
                SimpleCommandProcess._device_granted_indices,
            )
        else:
            raise NotImplementedError(f"SimpleCommandProcess Master received invalid rpc call: {rpc_command, rpc_args}")

    def _semaphore_dead_sequence(self, value):
        if self in self.holder_to_semaphores:
            for sem in self.holder_to_semaphores[self]:
                if self.semaphores[sem] is None:
                    self.semaphores[sem] = value

    def _update_timestamp(self):
        self.process_status_timestamp = time.time()

    def get_pid(self):
        try:
            return self.parent.pid
        except ValueError:
            return None

    def get_exitcode(self):
        return self.parent.exitcode

    def get_result(self):
        return self.rpc_results.get()

    def get_memory_usage_percent(self):
        return psutil.Process(self.get_pid()).memory_percent()

    def current_stage(self) -> str:
        return self.data.get("stage", "UNKNOWN")

    def resurrect(self):
        # 先清本对象旧引用再重建（__init__ 会 append）——no_reuse 长跑下
        # all_processes 不随复活无界增长
        self.all_processes[:] = [p for p in self.all_processes if p is not self]
        self.__init__(*self.original_input_params)

    def recycle(self) -> None:
        """close + resurrect 原子组合：多卡 task 完成后 / 错误残留后强制回收
        进程并原地重建——hccl 通信圈/异常状态会污染进程，复用会串台。"""
        self.close()
        self.resurrect()

    def close(self, no_update=False):
        if not no_update:
            self.update()
        if self.status != PROCESS_STATUS_CODE.DEAD:
            # noinspection PyBroadException
            try:
                self._parent_send_rpc(PROCESS_RPC.SUICIDE, ())
                # 宽限须覆盖 worker 的有序退出清理：rtDeviceReset 在
                # Ascend950PR 上裸测即 ~5s（与负载无关），3s 会在正当清理
                # 中途 terminate → SIGTERM 全栈转储噪音；真挂死由 terminate 兜底
                self.parent.join(SUICIDE_JOIN_TIMEOUT_S)
            finally:
                if self.get_exitcode() is None:
                    self.parent.terminate()
                    self.parent.join(3)
                    self.kill()
                try:
                    self.parent.close()
                except ValueError:
                    self.parent.terminate()
                    self.parent.join()
                    self.parent.close()
                self.status = PROCESS_STATUS_CODE.DEAD

    def kill(self):
        self.parent.kill()

    def update(self) -> None:
        if self.status == PROCESS_STATUS_CODE.DEAD:
            return
        try:
            self._update()
        except (EOFError, TimeoutError) as e:
            if isinstance(e, EOFError):
                if self.get_exitcode() is not None:
                    detail = describe_exit(self.get_exitcode())
                    logging.warning(
                        f"Process {self.name} exited unexpectedly with code {detail} "
                        f"(last stage: {self.current_stage()})"
                    )
                    exception = SystemError(f"Process {self.name} exited unexpectedly with code {detail}")
                    self.rpc_results.put(exception)
                else:
                    logging.warning(f"Process {self.name} lost connection with the parent process")
                    exception = SystemError(f"Process {self.name} lost connection with the parent process")
                    self.rpc_results.put(exception)
                    self.kill()
            else:
                logging.warning(f"Process {self.name} Timeout")
                exception = SystemError("Process Timeout")
                self.rpc_results.put(exception)
            for lock in self.locks:
                # noinspection PyBroadException
                try:
                    lock.release()
                except Exception:
                    logging.exception("Release lock failed")
            # Release device locks held by dead process
            DeviceLockManager.on_process_dead(
                self, SimpleCommandProcess._device_grant_events, SimpleCommandProcess._device_granted_indices
            )
            self._semaphore_dead_sequence(exception)
            self.close(True)
        except Exception:
            self.close(True)
            raise
        if self.status == PROCESS_STATUS_CODE.IDLE and not self.rpc_queue.empty():
            func_call = self.rpc_queue.get()
            self._parent_send_rpc(*func_call)
            self.status = PROCESS_STATUS_CODE.WAITING

    def send_action(self, target: Callable, args: tuple, kwargs: dict):
        self.rpc_queue.put((PROCESS_RPC.EXECUTE_FUNCTION, (target, args, kwargs)))

    def is_ready(self):
        return self.status == PROCESS_STATUS_CODE.IDLE

    def is_idle(self):
        return self.status == PROCESS_STATUS_CODE.IDLE and self.rpc_queue.empty()

    def is_dead(self):
        return self.status == PROCESS_STATUS_CODE.DEAD

    def is_completed(self):
        return not self.rpc_results.empty()
