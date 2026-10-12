#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Public utility for loading C extension libraries with on-demand compilation.
"""

__all__ = ["load_cext"]


import ctypes
import fcntl
import logging
import os
import subprocess


def _get_ttk_root():
    # __file__ = ttk/core_modules/kernel/compile/cext_loader.py → 上溯 5 级 = 仓库根
    path = os.path.abspath(__file__)
    for _ in range(5):
        path = os.path.dirname(path)
    return path


def _find_so(ttk_root, so_name, src_subdir):
    candidates = [
        os.path.join(ttk_root, "ttk", "lib", so_name),
        os.path.join(ttk_root, "csrc", src_subdir, "build", so_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _build_cext(src_dir, so_name):
    build_dir = os.path.join(src_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    so_path = os.path.join(build_dir, so_name)
    done_marker = os.path.join(build_dir, f".build.{so_name}.done")
    lock_path = os.path.join(build_dir, f".build.{so_name}.lock")

    # Fast path: already built and verified (most workers hit this)
    if os.path.isfile(so_path) and os.path.isfile(done_marker):
        return

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Double-check after acquiring lock
            if os.path.isfile(so_path) and os.path.isfile(done_marker):
                return

            # Stale marker from crashed previous build
            if os.path.isfile(done_marker):
                os.remove(done_marker)

            nproc = os.cpu_count() or 1
            subprocess.check_call(["cmake", "-S", src_dir, "-B", build_dir])
            subprocess.check_call(["cmake", "--build", build_dir, "-j", str(nproc)])

            if not os.path.isfile(so_path):
                raise RuntimeError(f"Build succeeded but {so_name} not found")

            # Atomically write completion marker: tmp file -> rename
            tmp = done_marker + ".tmp"
            with open(tmp, "w") as f:
                f.write("1")
            os.rename(tmp, done_marker)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def load_cext(so_name, src_subdir):
    """
    Load a C extension library with on-demand compilation.

    Search order:
      1. ttk/lib/{so_name}                    (whl install)
      2. csrc/{src_subdir}/build/{so_name}     (source build)
      3. On-demand compile csrc/{src_subdir}/

    Build protection: per-target flock + atomic marker, multi-process safe.

    Args:
        so_name: Library filename, e.g. "libttk_op_registry_accessor.so"
        src_subdir: Subdirectory under csrc/, e.g. "op_registry_accessor"

    Returns:
        ctypes.CDLL handle
    """
    ttk_root = _get_ttk_root()

    so_path = _find_so(ttk_root, so_name, src_subdir)
    if so_path:
        return ctypes.CDLL(so_path)

    # Source first-use: on-demand build
    src_dir = os.path.join(ttk_root, "csrc", src_subdir)
    if os.path.isdir(src_dir):
        logging.info(f"On-demand build: compiling {so_name} from {src_dir}")
        _build_cext(src_dir, so_name)
        so_path = os.path.join(src_dir, "build", so_name)
        if os.path.isfile(so_path):
            return ctypes.CDLL(so_path)

    logging.critical(f"Cannot find {so_name}")
    raise FileNotFoundError(f"{so_name} not found in any expected location")
