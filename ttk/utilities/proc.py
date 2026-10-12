#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Process Related Utilities
"""

__all__ = [
    "set_process_name",
    "is_main_process",
    "set_thread_name",
    "signal_registered",
    "cpu_count",
    "msdebug_runtime_injection_enabled",
    "kernel_debug_compile_enabled",
    "get_loaded_so_path",
]


# Standard Packages
import multiprocessing
import os
import subprocess
import threading
from ctypes import CDLL


def set_process_name(name: str = "MP"):
    """Set current process name for logging."""
    multiprocessing.current_process().name = name


def set_thread_name(name: str = "MT"):
    """Set current thread name for threading."""
    threading.current_thread().name = name


def is_main_process() -> bool:
    """True in the TTK parent process (compares against TTK_PARENT_PID)."""
    return os.getpid() == int(os.getenv("TTK_PARENT_PID", "0"))


def signal_registered(signum: int) -> bool:
    """True if @signum is caught/ignored/blocked by this process (via /proc status)."""
    pid = os.getpid()
    status_file = f"/proc/{pid}/status"
    signal_fields = ["SigCgt", "SigIgn", "SigBlk"]

    if not os.path.exists(status_file):
        return False

    signal_data = {}
    with open(status_file) as f:
        for line in f:
            for field in signal_fields:
                if line.startswith(field + ":"):
                    hex_str = line.split(":")[1].strip()
                    bitmask = int(hex_str, 16)
                    signal_data[field] = bitmask

    # signum starts from 1. bit is `signum - 1`
    return any((bitmask >> (signum - 1)) & 1 for bitmask in signal_data.values())


def cpu_count() -> int:
    """Number of CPUs available to this process (affinity-aware)."""
    return len(os.sched_getaffinity(0))


def msdebug_runtime_injection_enabled() -> bool:
    """
    Return whether the current process runs under msdebug runtime injection.

    msdebug launches the target process with:
      - MSOP_SOCKET_PATH=<socket> (stub<->lldb communication)
      - LD_PRELOAD=<...>/libruntime_stub.so (runtime hijack)
    Either signal indicates RTS calls should resolve through the process-global
    symbol table so that the preloaded stub intercepts them.
    """
    if os.getenv("MSOP_SOCKET_PATH"):
        return True
    return _msdebug_runtime_stub_preloaded()


def _msdebug_runtime_stub_preloaded() -> bool:
    """Return whether msdebug's runtime interposition library is preloaded."""
    preload = os.getenv("LD_PRELOAD", "")
    return any(os.path.basename(path) == "libruntime_stub.so" for path in preload.replace(":", " ").split())


def kernel_debug_compile_enabled() -> bool:
    """
    Return whether TTK should compile kernels with source-level debug info (-g -O0).

    MSOP_SOCKET_PATH is shared by other runtime interposers (for example
    msopprof), so only msdebug's preloaded runtime stub is a sufficiently
    specific signal for changing kernel compilation to -O0 -g.
    """
    return _msdebug_runtime_stub_preloaded()


def get_loaded_so_path(loaded_cdll: CDLL) -> str:
    """On-disk path(s) of a loaded CDLL, via lsof on this process; "UNKNOWN" if unavailable."""
    try:
        # noinspection PyProtectedMember
        so_name = os.path.basename(loaded_cdll._name)
        lines = subprocess.check_output(["lsof", "-p", str(os.getpid())], encoding="UTF-8").split("\n")
        # match the trailing path column by basename — whole-line substring would
        # false-hit other libs (e.g. "librt.so" against "librt.so.1")
        results = [
            path for path in (line.split(" ")[-1] for line in lines) if path and os.path.basename(path) == so_name
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return "|".join(results) if results else "UNKNOWN"
