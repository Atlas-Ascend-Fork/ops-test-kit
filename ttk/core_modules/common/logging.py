#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/logging：日志配置——per-case 双文件 + stdout。

- **ttk.log**：除 COMPARE 外全部级别混写（级别由 formatter 前缀承载）
- **ttk-compare.log**：COMPARE 级（debugc）专属——比对明细（Index/Diff 行）独立成档，
  不混入执行日志
- worker 每任务开头重配（运行内追加，复跑覆写由主进程截断实现）
- 打屏通道：stdout 仅一个 INFO 级 StreamHandler，与文件日志同根但分离
- CANN slog 打屏开启（ASCEND_SLOG_PRINT_TO_STDOUT=1）时：fd 1 重定向到
  当前用例 ttk.log（C 层直写进档），打屏由镜像线程还原（release 时排干）
- 主进程 / worker 引导 / 编译池 / KB server：不落文件（stdout only）——
  用例级排查以 per-case 文件为准，主流程结果以结果 CSV 为准

文件布局（默认，无开关）：
    {root}/log/{mode}/{testcase_name}/ttk.log          # 每用例：非 COMPARE 全级别（运行内追加——
                                                      #   编译/执行多任务共写；主进程建任务时
                                                      #   截断 = 复跑覆写）
    {root}/log/{mode}/{testcase_name}/ttk-compare.log  # 每用例：比对明细（追加——跨复跑
                                                      #   保留比对历史，每次运行写 run start 分隔标记）
"""

__all__ = [
    "COMPARE_LOG_FILE_NAME",
    "LOG_FILE_NAME",
    "build_single_log_dir",
    "cann_stdout_capture_enabled",
    "default_logging_config",
    "print_error",
    "release_case_stdout",
    "reset_case_log",
]

# Standard Packages
import contextlib
import logging
import os
import sys
import threading
import time
import warnings

# Third-Party Packages
import numpy

_MODE_LOG_DIR = {
    "op": "kernel",
    "aclnn": "aclnn",
    "geir": "geir",
    "framework-api": "e2e",
}

LOG_FILE_NAME = "ttk.log"
COMPARE_LOG_FILE_NAME = "ttk-compare.log"

_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] [%(process)d %(processName)s %(threadName)s] [%(filename)s:%(lineno)d]: %(message)s "
)

_COMPARE_LEVEL = logging.DEBUG - 5


class _ExcludeCompare(logging.Filter):
    def filter(self, record):
        return record.levelno != _COMPARE_LEVEL


class _OnlyCompare(logging.Filter):
    def filter(self, record):
        return record.levelno == _COMPARE_LEVEL


def build_single_log_dir(test_mode, testcase_name, root_path):
    """Construct per-testcase log directory: {root}/log/{mode}/{testcase_name}.

    Mirrors msprof per-case path layout (msprof/op/{testcase_name}/...).
    Creates the directory if missing. Returns the absolute path.
    """
    mode_dir = _MODE_LOG_DIR.get(test_mode, test_mode or "unknown")
    testcase_name = testcase_name or "unknown"
    log_dir = os.path.join(root_path, "log", mode_dir, testcase_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


_owned_handlers: list = []  # 本模块挂载的 handler——重配时随移随关（防误关外来 handler）

_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


class _ErrorRedFormatter(logging.Formatter):
    """ERROR 及以上整行红色；仅用于 tty 流（管道/重定向不注入 ANSI 码）。"""

    def format(self, record):
        text = super().format(record)
        if record.levelno >= logging.ERROR:
            return f"{_ANSI_RED}{text}{_ANSI_RESET}"
        return text


def _formatter_for(handler) -> logging.Formatter:
    stream = getattr(handler, "stream", None)
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    return _ErrorRedFormatter(_LOG_FORMAT) if is_tty else logging.Formatter(_LOG_FORMAT)


def print_error(message: str) -> None:
    """stderr 直写错误行：tty 上整行红色（与 _ErrorRedFormatter 同口径）。"""
    if getattr(sys.stderr, "isatty", lambda: False)():
        message = f"{_ANSI_RED}{message}{_ANSI_RESET}"
    print(message, file=sys.stderr)


def _attach(handler, level=None, flt=None):
    if level is not None:
        handler.setLevel(level)
    if flt is not None:
        handler.addFilter(flt)
    handler.setFormatter(_formatter_for(handler))
    _owned_handlers.append(handler)
    logging.getLogger().addHandler(handler)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).addHandler(handler)


def add_level(name: str, visible_name: str, level: int):
    """Add a new logging level（COMPARE/debugc 注册，保持 logging.log(COMPARE, ...) 可用）."""

    def _logging_method(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    def _logging_method_root(message, *args, **kwargs):
        logging.log(level, message, *args, **kwargs)

    logging.addLevelName(level, visible_name)
    setattr(logging, name, _logging_method_root)
    setattr(logging.getLoggerClass(), name, _logging_method)


# ═══ CANN stdout 捕获（ASCEND_SLOG_PRINT_TO_STDOUT=1 时）═══
# CANN slog 打屏是 C 层直写 fd 1。捕获方式：**fd 1 重定向到当前用例 ttk.log**
# （O_APPEND 直写，CANN/子进程/print 输出与 TTK 日志同档），打屏由 tail 跟随
# 线程转发真实终端——两需求（进档 + 打屏）都保，且无管道。
#
# 不用管道 tee 的原因（真机实录）：fd 1 指向管道时 GE 在 op_impl .so 装载期
# 死锁（AddSoToRegister 停滞）；fd 1 指向文件同场景 23s 通过——管道语义
# （写满阻塞/EOF 语义/fork 继承写端）与 GE 装载路径不相容。

_CAPTURE_STATE = {
    "installed": False,
    "real_fd": None,  # 真实终端
    "target_path": None,  # 当前 fd 1 指向的用例日志（None = 任务结束，镜像挂起）
    "start_offset": 0,  # 本次指向时的文件偏移——镜像从此跟起（安装竞态不漏行）
    "thread": None,
    "stop": False,
}


def cann_stdout_capture_enabled() -> bool:
    """CANN slog 打屏开关（环境变量，随 CANN 组件语义）——打屏关则捕获不装。"""
    return os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT", "0") == "1"


def _write_all(fd: int, data: bytes) -> None:
    """写满（终端/管道 write 可能部分写）。"""
    while data:
        data = data[os.write(fd, data) :]


def release_case_stdout() -> None:
    """任务结束释放镜像跟随（fd 1 保持指向——同进程下一任务重指向前，残余
    CANN 缓冲仍进档；镜像置 None 挂起并排干旧档）。

    消除双镜像重复：同用例档案可能先后被编译/执行 worker 指向，若双方镜像
    同时跟随，新增行会被转发两次——任务边界释放，保证任一时刻至多一个
    镜像跟随同一档案。
    """
    if _CAPTURE_STATE["installed"]:
        _CAPTURE_STATE["target_path"] = None


def _emit_visible(data: bytes) -> bytes:
    """转发完整行到终端（过滤 TTK DEBUG/COMPARE 级），返回不完整尾段。

    尾段由调用方留存拼入下一轮——跨读块的行不丢。
    """
    parts = data.split(b"\n")
    pending = parts.pop()
    visible = [ln for ln in parts if b" [DEBUG] " not in ln and b" [COMPARE] " not in ln]
    if visible:
        _write_all(_CAPTURE_STATE["real_fd"], b"\n".join(visible) + b"\n")
    return pending


def _switch_target(read_fd, path, pending):
    """换档：旧档排干（窗口内尾部行不丢）后打开新档；None = 挂起。

    返回 (read_fd, pending)。"""
    if read_fd is not None:
        pending = _drain(read_fd, pending)
        os.close(read_fd)
        read_fd = None
    if path:
        read_fd = os.open(path, os.O_RDONLY)
        # 从指向时刻的偏移跟起（非 END）——线程启动晚于数据写入时不漏行
        os.lseek(read_fd, _CAPTURE_STATE["start_offset"], os.SEEK_SET)
    return read_fd, pending


def _drain(read_fd, pending):
    """读至 EOF 转发完整行，残余尾段补一个换行发出（不丢字节）。"""
    while True:
        data = os.read(read_fd, 65536)
        if not data:
            break
        pending = _emit_visible(pending + data)
    if pending:
        _write_all(_CAPTURE_STATE["real_fd"], pending + b"\n")
        pending = b""
    return pending


def _mirror_screen():
    """用例日志新增段镜像到真实终端（fd 1 重定向进档后的打屏还原）。

    不漏保证：跨读块的不完整行经 pending 拼接（_emit_visible）；换用例档时
    旧档先排干再切换（_switch_target）；终端写部分写时补写（_write_all）。
    """
    read_fd = None
    path = None
    pending = b""
    while not _CAPTURE_STATE["stop"]:
        with contextlib.suppress(Exception):  # 镜像线程不因偶发 IO 异常退出（退出即打屏冻结）
            target = _CAPTURE_STATE["target_path"]
            if target != path:
                read_fd, pending = _switch_target(read_fd, target, pending)
                path = target
            if read_fd is not None:
                while True:
                    data = os.read(read_fd, 65536)
                    if not data:
                        break
                    pending = _emit_visible(pending + data)
        time.sleep(0.05)


def _point_stdout_to_case_log(path: str) -> None:
    """fd 1 → 用例 ttk.log（首装捕获 + 每用例重指向）。"""
    if not _CAPTURE_STATE["installed"]:
        saved_fd = os.dup(1)
        _CAPTURE_STATE["installed"] = True
        _CAPTURE_STATE["real_fd"] = saved_fd
        _CAPTURE_STATE["thread"] = threading.Thread(target=_mirror_screen, name="CannScreenMirror", daemon=True)
        _CAPTURE_STATE["thread"].start()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    _CAPTURE_STATE["start_offset"] = os.fstat(fd).st_size
    os.dup2(fd, 1)
    os.close(fd)
    _CAPTURE_STATE["target_path"] = path


def _reset_stdout_capture():
    """还原 fd 1 并停止镜像（测试/进程收尾用）。"""
    if not _CAPTURE_STATE["installed"]:
        return
    _CAPTURE_STATE["stop"] = True
    thread = _CAPTURE_STATE["thread"]
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=1)  # 等 mirror 停止再关 real_fd——防 fd 复用后误写
    if _CAPTURE_STATE["real_fd"] is not None:
        try:
            os.dup2(_CAPTURE_STATE["real_fd"], 1)
            os.close(_CAPTURE_STATE["real_fd"])
        except OSError:
            pass
    _CAPTURE_STATE.update(installed=False, real_fd=None, target_path=None, start_offset=0, thread=None, stop=False)


def reset_case_log(log_dir: str) -> None:
    """重置用例 ttk.log（主进程 create_tasks 时逐用例调用——复跑覆写语义）。

    只重置 ttk.log；ttk-compare.log 是跨复跑比对历史（run start 标记分档），不动。
    """
    path = os.path.join(log_dir, LOG_FILE_NAME)
    with open(path, "w", encoding="utf-8"):
        pass


def _clear_handlers():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    # 自有 FileHandler 随移随关：长驻 worker 每任务重配，不关则 fd 逐任务
    # 泄漏；只关 _attach 挂载的（外来 handler——如测试夹具快照——不碰）
    for handler in _owned_handlers:
        if isinstance(handler, logging.FileHandler):
            with contextlib.suppress(OSError, ValueError):
                handler.close()
    _owned_handlers.clear()


def default_logging_config(file_handler: bool = False, testcase_name: str = None, log_dir: str = None):
    """进程日志初始化：注册 COMPARE 级别（debugc）、清空既有 handler、numpy/warning 静音。

    - worker 每任务开头调用（testcase_name 非空 + file_handler）：
      log_dir（build_single_log_dir 产物）下——ttk.log **追加**（O_APPEND 行级
      原子写：同用例的 dyn/cst/bin 编译 + 执行多任务/多 worker 共写互不截断；
      复跑覆写由主进程 create_tasks 时的 reset_case_log 实现）；
      ttk-compare.log 追加（COMPARE 专属，跨复跑保留历史，任务开头写
      "===== run start" 分隔标记——read_compare_log_failures 只读末段）。
    - 主进程 / worker_bootstrap（testcase_name=None）：stdout only，不落文件。
    """
    add_level("debugc", "COMPARE", _COMPARE_LEVEL)
    _clear_handlers()
    # Ignore numpy RuntimeWarnings
    numpy.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    # 三方噪音 logger 静音：torch_npu 的 getenv 补丁逐键打 INFO（每进程各记
    # 一遍"once"），打屏无信息量；压 WARNING，保留告警及以上
    logging.getLogger("torch_npu.env").setLevel(logging.WARNING)
    if file_handler and testcase_name:
        base = log_dir if log_dir else "."
        os.makedirs(base, exist_ok=True)
        # CANN slog 打屏开启：fd 1 重定向到用例 ttk.log（C 层直写进档），
        # 打屏由 tail 线程代理（此时不再挂 stdout handler——防 TTK 行双重上屏）
        _cann_capture = cann_stdout_capture_enabled()
        if _cann_capture:
            _point_stdout_to_case_log(os.path.join(base, LOG_FILE_NAME))
        # ttk.log 追加（运行内多任务共写；复跑覆写由主进程 reset_case_log 实现）；
        # ttk-compare.log 追加（跨复跑保留比对历史）
        _attach(logging.FileHandler(os.path.join(base, LOG_FILE_NAME), mode="a"), flt=_ExcludeCompare())
        _attach(logging.FileHandler(os.path.join(base, COMPARE_LOG_FILE_NAME), mode="a"), flt=_OnlyCompare())
        if not _cann_capture:
            _attach(logging.StreamHandler(sys.stdout), logging.INFO)
    else:
        # 打屏口径 INFO：进程级（主进程/worker 引导）无文件时，DEBUG 只能丢——
        # NOTSET 会把 DSMI 轮询等 DEBUG 刷满终端（真机实录）
        _attach(logging.StreamHandler(sys.stdout), logging.INFO)
    # Enable all logging
    logging.getLogger().setLevel(logging.NOTSET)
    if testcase_name:
        logging.debug(f"Logging single testcase log for testcase {testcase_name}")
        # 追加档的运行分隔标记（仅进 compare 档）：增量打印按末段读取的锚点
        logging.debugc(f"===== run start (pid {os.getpid()}) =====")
