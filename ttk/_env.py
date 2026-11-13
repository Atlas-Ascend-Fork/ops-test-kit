#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
import ctypes
import ctypes.util
import glob
import importlib.util
import logging
import os
import resource
import subprocess
import sys
import time


def setup_env():
    ascend_root = _find_ascend_root()
    if ascend_root:
        _source_setenv_bash(ascend_root)
        _setup_cann_paths(ascend_root)
        _setup_ascend_logging()

    _setup_ulimit()
    _preload_libgomp()
    _setup_runtime_env()
    _cleanup_old_logs()
    _ensure_log_dirs()


def _find_ascend_root():
    candidates = []
    for env_var in ["ASCEND_CUSTOM_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"]:
        val = os.getenv(env_var)
        if val:
            candidates.append(val)

    opp = os.getenv("ASCEND_OPP_PATH")
    if opp:
        normalized_opp = os.path.normpath(opp)
        if os.path.basename(normalized_opp) == "opp":
            normalized_opp = os.path.dirname(normalized_opp)
        candidates.append(normalized_opp)

    if not candidates:
        for base in (os.path.expanduser("~/Ascend"), "/usr/local/Ascend"):
            for sub in ("cann", "ascend-toolkit/latest"):
                root = os.path.join(base, sub)
                if os.path.isdir(root):
                    candidates.append(root)

    for root in candidates:
        normalized_root = root.rstrip("/")
        if os.path.isdir(os.path.join(normalized_root, "compiler")) and os.path.isdir(
            os.path.join(normalized_root, "opp")
        ):
            return normalized_root
    return None


def _sim_ld_paths():
    """LD_LIBRARY_PATH segments pointing into a camodel/simulator install.

    The NPUSim camodel runtime is injected via ``LD_LIBRARY_PATH`` (by cannsim
    record or the E2E npusim backend). CANN ``setenv.bash`` rebuilds
    ``LD_LIBRARY_PATH`` and would drop those segments, so they are recorded
    before sourcing and restored afterwards.
    """
    return [
        p
        for p in (os.environ.get("LD_LIBRARY_PATH", "") or "").split(":")
        if p and ("camodel" in p or "/simulator/" in p)
    ]


def _restore_ld_paths(segments):
    """Prepend ``segments`` to LD_LIBRARY_PATH unless already present."""
    if not segments:
        return
    existing = (os.environ.get("LD_LIBRARY_PATH", "") or "").split(":")
    missing = [p for p in segments if p not in existing]
    if missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + existing)


def _source_setenv_bash(ascend_root):
    setenv = None
    for candidate in (os.path.join(ascend_root, "set_env.sh"), os.path.join(ascend_root, "bin", "setenv.bash")):
        if os.path.isfile(candidate):
            setenv = candidate
            break
    if setenv is None:
        return

    # A sourced parent shell already owns the complete CANN environment.  Do
    # not source it again: setenv.bash rebuilds LD_LIBRARY_PATH and can move an
    # ESL/NPUSim runtime behind the real CANN libraries.
    if os.getenv("ASCEND_TOOLKIT_HOME") and os.getenv("ASCEND_OPP_PATH"):
        return

    logging.getLogger(__name__).warning(
        "CANN environment is incomplete (ASCEND_TOOLKIT_HOME/ASCEND_OPP_PATH); "
        "TTK is automatically sourcing %s. Source the CANN environment in your shell before running TTK.",
        setenv,
    )
    sim_paths = _sim_ld_paths()
    try:
        result = subprocess.run(
            ["bash", "-c", f'source "{setenv}" && env -0'],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return

        for entry in result.stdout.split("\0"):
            if "=" in entry:
                key, _, val = entry.partition("=")
                if key and "\n" not in key:
                    os.environ[key] = val
        # setenv.bash rebuilds LD_LIBRARY_PATH; keep the camodel runtime visible.
        _restore_ld_paths(sim_paths)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _get_custom_impl_parent_paths():
    """Get op_impl/ai_core/tbe paths from ASCEND_CUSTOM_OPP_PATH."""
    env_val = os.getenv("ASCEND_CUSTOM_OPP_PATH", "")
    if not env_val:
        return []
    paths = []
    for p in env_val.split(":"):
        if not p:
            continue
        tbe = os.path.join(p, "op_impl", "ai_core", "tbe")
        if os.path.isdir(tbe):
            paths.append(tbe)
    return paths


def _get_vendor_impl_parent_paths(opp_path):
    """Get op_impl/ai_core/tbe paths from vendors/config.ini."""
    config_file = os.path.join(opp_path, "vendors", "config.ini")
    if not os.path.isfile(config_file):
        return []
    paths = []
    with open(config_file) as f:
        for line in f:
            if line.strip().startswith("load_priority="):
                for name in line.split("=", 1)[1].split(","):
                    vendor = name.strip()
                    if vendor:
                        tbe = os.path.join(opp_path, "vendors", vendor, "op_impl", "ai_core", "tbe")
                        if os.path.isdir(tbe):
                            paths.append(tbe)
                break
    return paths


def _get_builtin_impl_parent_path(opp_path):
    """Get built-in op_impl/ai_core/tbe path."""
    tbe = os.path.join(opp_path, "built-in", "op_impl", "ai_core", "tbe")
    return tbe if os.path.isdir(tbe) else None


def _prepend_to_pythonpath(paths):
    """Prepend paths to PYTHONPATH env var and sys.path."""
    if not paths:
        return
    existing = os.getenv("PYTHONPATH", "")
    new = ":".join(paths)
    os.environ["PYTHONPATH"] = new + ":" + existing if existing else new
    path_set = set(paths)
    sys.path[:] = list(paths) + [p for p in sys.path if p not in path_set]


def _setup_cann_paths(ascend_root):
    drv_info = "/etc/ascend_install.info"
    if os.path.isfile(drv_info):
        with open(drv_info) as f:
            for line in f:
                if line.startswith("Driver_Install_Path_Param="):
                    drv_path = line.strip().split("=", 1)[1]
                    drv_lib = os.path.join(drv_path, "driver", "lib64", "driver")
                    if os.path.isdir(drv_lib):
                        existing = os.getenv("LD_LIBRARY_PATH", "")
                        if drv_lib not in existing:
                            os.environ["LD_LIBRARY_PATH"] = drv_lib + ":" + existing
                    break

    opp_path = os.path.join(ascend_root, "opp")
    if os.path.isdir(opp_path):
        os.environ.setdefault("ASCEND_OPP_PATH", opp_path)

        # Collect tbe paths in priority order: custom > vendors > built-in
        tbe_paths = []
        tbe_paths.extend(_get_custom_impl_parent_paths())
        tbe_paths.extend(_get_vendor_impl_parent_paths(opp_path))
        builtin_tbe = _get_builtin_impl_parent_path(opp_path)
        if builtin_tbe:
            tbe_paths.append(builtin_tbe)
        _prepend_to_pythonpath(tbe_paths)


def _setup_ascend_logging():
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    os.environ.setdefault("ASCEND_GLOBAL_EVENT_ENABLE", "0")
    os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")


def _setup_ulimit():
    for res, soft_limit in [
        (resource.RLIMIT_MEMLOCK, 65535),
        (resource.RLIMIT_NOFILE, 655300),
        (resource.RLIMIT_STACK, 81920),
    ]:
        try:
            soft, hard = resource.getrlimit(res)
            resource.setrlimit(res, (min(soft_limit, hard), hard))
        except (ValueError, OSError):
            pass


def _preload_libgomp():
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        torch_dir = os.path.dirname(spec.origin)
        for subdir in ["lib", ".libs"]:
            lib_dir = os.path.join(torch_dir, subdir)
            if os.path.isdir(lib_dir):
                matches = glob.glob(os.path.join(lib_dir, "libgomp*.so*"))
                if matches:
                    ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                    return

    for pkg in ["tensorflow_cpu_aws", "tensorflow", "tensorflow-cpu"]:
        try:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
                for suffix in [pkg + ".libs", ".libs"]:
                    lib_dir = os.path.join(pkg_dir, suffix)
                    if os.path.isdir(lib_dir):
                        matches = glob.glob(os.path.join(lib_dir, "libgomp*.so*"))
                        if matches:
                            ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                            return
        except (ModuleNotFoundError, ValueError):
            continue

    path = ctypes.util.find_library("gomp")
    if path:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _setup_runtime_env():
    os.environ["PYTHONHASHSEED"] = "0"


def _cleanup_old_logs():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for f in glob.glob(os.path.join(base_dir, "ttk-*.log")):
        with contextlib.suppress(OSError):
            os.remove(f)


def _ensure_log_dirs():
    log_root = os.environ.get("ASCEND_PROCESS_LOG_PATH") or (
        os.path.join(os.environ["ASCEND_WORK_PATH"], "log")
        if os.environ.get("ASCEND_WORK_PATH")
        else os.path.expanduser("~/ascend/log")
    )
    log_root = os.path.abspath(os.path.expanduser(log_root))
    try:
        os.makedirs(log_root, exist_ok=True)
    except OSError as e:
        logging.warning("Cannot create log dir %s: %s. Continue without it.", log_root, e)

    cutoff = time.time() - 15 * 86400
    for d in [os.path.join(log_root, "plog"), os.path.join(log_root, "debug/plog")]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                try:
                    if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                except OSError:
                    pass
