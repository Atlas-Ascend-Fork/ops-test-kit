#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os
import re
import subprocess
from typing import Optional

_DT_ENUM_RE = re.compile(r"^\s*(DT_[A-Za-z0-9_]+)\s*=", re.MULTILINE)
_CANN_FEATURES = None


def detect_cann_features():
    """Probe installed CANN headers for symbols referenced by the GEIR template.

    Returns None when the Ascend installation or graph/types.h cannot be located
    (callers then keep the unfiltered template defaults). Otherwise returns
    {"ge_enums": set of DT_* names defined in graph/types.h, "ge_prof_ok": whether
    ge_prof.h exists and defines GE_PROF_SUCCESS}. Newer dtypes (DT_HIFLOAT4 etc.)
    and GE_PROF_SUCCESS are absent on older CANN (e.g. 8.5.0) and must not be
    emitted for those installations.
    """
    global _CANN_FEATURES
    if _CANN_FEATURES is not None:
        return _CANN_FEATURES
    from ttk._env import _find_ascend_root

    asc_path = _find_ascend_root()
    if not asc_path:
        return None
    from ttk.core_modules.common.device import get_ascend_scene_info

    try:
        scene_os, scene_arch = get_ascend_scene_info()
    except (RuntimeError, OSError):
        # ASCEND_OPP_PATH 未配置时 get_opp_paths 会 raise；此处按 arch_dir 的
        # 既有缺省语义降级为 x86_64-linux，而不是让探测整体崩溃。
        scene_os, scene_arch = "", ""
    arch_dir = f"{scene_arch}-{scene_os}" if scene_arch and scene_os else "x86_64-linux"

    def _read(*rel):
        for base in (os.path.join(asc_path, arch_dir), asc_path):
            path = os.path.join(base, *rel)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        return f.read()
                except OSError:
                    return None
        return None

    types_h = _read("include", "graph", "types.h")
    if types_h is None:
        return None
    ge_prof_h = _read("include", "ge", "ge_prof.h")

    _CANN_FEATURES = {
        "ge_enums": set(_DT_ENUM_RE.findall(types_h)),
        "ge_prof_ok": ge_prof_h is not None and "GE_PROF_SUCCESS" in ge_prof_h,
    }
    return _CANN_FEATURES


class GeirCompiler:
    def __init__(self, switches, build_dir=None):
        self._switches = switches
        self._build_dir = build_dir or os.path.join(getattr(switches, "root_path", os.getcwd()), "geir")
        os.makedirs(self._build_dir, exist_ok=True)

    @property
    def build_dir(self):
        return self._build_dir

    _CXX_FLAGS = ["g++", "-std=c++17", "-O0", "-D_GLIBCXX_USE_CXX11_ABI=0"]

    def _resolve_ascend_env(self):
        """Resolve Ascend include dirs, lib dirs, and libs. Returns (include_dirs, lib_dirs, libs)."""
        from ttk._env import _find_ascend_root

        asc_path = _find_ascend_root()
        if not asc_path:
            raise RuntimeError("Ascend installation not found. Please check Ascend installation.")

        from ttk.core_modules.common.device import get_ascend_scene_info

        try:
            scene_os, scene_arch = get_ascend_scene_info()
        except (RuntimeError, OSError):
            # 同 detect_cann_features：OPP 路径缺失时降级默认 arch 目录。
            scene_os, scene_arch = "", ""
        arch_dir = f"{scene_arch}-{scene_os}" if scene_arch and scene_os else "x86_64-linux"

        include_dirs = [
            os.path.join(asc_path, arch_dir, "include"),
            os.path.join(asc_path, arch_dir, "include", "graph"),
            os.path.join(asc_path, arch_dir, "include", "ge"),
            os.path.join(asc_path, arch_dir, "include", "external"),
            os.path.join(asc_path, "opp", "built-in", "op_proto", "inc"),
            os.path.join(asc_path, "opp", "built-in", "op_graph", "inc"),
        ]

        from ttk.core_modules.common.device import get_opp_paths

        for copp in get_opp_paths("custom"):
            d = os.path.join(copp, "op_proto", "inc")
            if os.path.isdir(d):
                include_dirs.append(d)
        for vopp in get_opp_paths("vendor"):
            d = os.path.join(vopp, "op_proto", "inc")
            if os.path.isdir(d):
                include_dirs.append(d)

        lib_dirs = [
            os.path.join(asc_path, arch_dir, "lib64"),
            os.path.join(asc_path, arch_dir, "lib64", "stub"),
        ]
        libs = ["graph", "ge_runner", "graph_base", "ge_compiler", "msprofiler"]
        return include_dirs, lib_dirs, libs

    def _build_compile_cmd(self, source_path, binary_path, include_dirs, lib_dirs, libs):
        cmd = self._CXX_FLAGS + ["-o", binary_path, source_path]
        for d in include_dirs:
            if os.path.isdir(d):
                cmd.extend(["-I", d])
        for d in lib_dirs:
            if os.path.isdir(d):
                cmd.extend(["-L", d])
        for lib in libs:
            cmd.append(f"-l{lib}")
        return cmd

    def compile(self, source_path: str, binary_name: str = "ttk_geir_test") -> Optional[str]:
        include_dirs, lib_dirs, libs = self._resolve_ascend_env()

        binary_path = os.path.join(self._build_dir, binary_name)
        cmd = self._build_compile_cmd(source_path, binary_path, include_dirs, lib_dirs, libs)

        logging.info(f"GEIR compile: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=self._build_dir, check=False)
            if result.returncode != 0:
                logging.error(f"GEIR compile failed:\n{result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logging.error("GEIR compile timed out (300s)")
            return None

        self._binary_path = binary_path
        return binary_path

    def _compute_cache_key(self, source_path, include_dirs, lib_dirs, libs):
        """Compute a cache key covering source mtime, compile flags, and Ascend SDK path."""
        import hashlib

        parts = [str(os.path.getmtime(source_path)), " ".join(self._CXX_FLAGS)]
        from ttk._env import _find_ascend_root

        asc_path = _find_ascend_root() or ""
        parts.append(asc_path)
        h = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]  # noqa: S324  # 非安全用途：缓存键校验和
        return f"{os.path.getmtime(source_path)}:{h}"

    def compile_op(
        self,
        source_path: str,
        op_name: str,
        op_dir: str,
    ) -> Optional[str]:
        """Compile op-level binary with caching (flock + stamp keyed on source mtime + SDK)."""
        os.makedirs(op_dir, exist_ok=True)
        binary_path = os.path.join(op_dir, op_name)
        stamp_path = os.path.join(op_dir, op_name + ".stamp")
        lock_path = os.path.join(op_dir, op_name + ".lock")

        include_dirs, lib_dirs, libs = self._resolve_ascend_env()
        cache_key = self._compute_cache_key(source_path, include_dirs, lib_dirs, libs)

        import fcntl

        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            if os.path.isfile(binary_path) and os.path.isfile(stamp_path):
                try:
                    with open(stamp_path) as sf:
                        cached_key = sf.read().strip()
                except OSError:
                    cached_key = ""
                if cached_key == cache_key:
                    logging.info("GEIR op binary cached: %s", binary_path)
                    self._binary_path = binary_path
                    return binary_path
                logging.info("GEIR op cache miss (key changed), recompiling %s", op_name)

            cmd = self._build_compile_cmd(source_path, binary_path, include_dirs, lib_dirs, libs)

            logging.info(f"GEIR op compile: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=op_dir, check=False)
                if result.returncode != 0:
                    logging.error(f"GEIR op compile failed:\n{result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                logging.error("GEIR op compile timed out (300s)")
                return None

            try:
                with open(stamp_path, "w") as sf:
                    sf.write(cache_key)
            except OSError:
                pass

        self._binary_path = binary_path
        return binary_path

    def cleanup(self, input_prefix=None):
        if input_prefix:
            import contextlib
            import glob

            for f in glob.glob(f"{input_prefix}_*.bin"):
                with contextlib.suppress(OSError):
                    os.remove(f)
