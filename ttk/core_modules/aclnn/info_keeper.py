#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Op api info keeper.

Parses aclnn op_api C headers (regex) plus the libopapi symbol table into
``OpApiInfo`` entries, lazily, once per process. The parsed cache is pure
data, so it can be snapshotted in the main process after testcase parsing
and shipped to forkserver workers as part of their bootstrap payload:
workers restore it via :func:`restore_keeper_snapshot` and never re-parse
the header tree.
"""

__all__ = ["OpApiInfoKeeper", "OpApiInfo", "snapshot_keeper", "restore_keeper_snapshot", "ensure_api_so_path"]


# Standard Packages
import glob
import logging
import os
import re
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third-Party Packages
from ...utilities import Singleton
from ..common.device import get_ascend_lib64_path, get_ascend_scene_info, get_opp_paths

# Module-level built-in SO path resolution (lazy, nm scanning)
_builtin_api_so_map: Optional[Dict[str, str]] = None


def _ensure_builtin_so_map():
    global _builtin_api_so_map
    if _builtin_api_so_map is not None:
        return
    lib64 = get_ascend_lib64_path()
    so_files = sorted(glob.glob(f"{lib64}/libopapi.so")) + sorted(glob.glob(f"{lib64}/libopapi_*.so"))
    _builtin_api_so_map = {}
    for so_path in so_files:
        try:
            output = subprocess.check_output(["nm", "-D", so_path], text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in output.splitlines():
            if "GetWorkspaceSize" not in line:
                continue
            symbol = line.strip().split()[-1]
            name = symbol[: -len("GetWorkspaceSize")]
            _builtin_api_so_map[name] = so_path


def _get_builtin_so_path(api_name: str) -> str:
    _ensure_builtin_so_map()
    return (_builtin_api_so_map or {}).get(api_name, "")


# Fallback probe order when symbol-scan resolution comes up empty:
# libopapi_transformer.so carries the transformer subset alongside the base
# libopapi.so, so it is probed first.
_LIBOPAPI_FALLBACKS = ("libopapi_transformer.so", "libopapi.so")


def ensure_api_so_path(api_name: str) -> str:
    """Resolve a loadable SO path for an aclnn API, single- and multi-device.

    Generic resolution shared by every consumer (runtime._get_op_api and the
    multi-device thread launcher): an existing loadable ``so_path`` (header-derived
    libcust_opapi.so / TTK_OP_API_SO_PATH, or the builtin nm symbol-scan
    result) is kept; only empty or dangling paths fall back to probing
    ``_LIBOPAPI_FALLBACKS`` in the arch-specific lib64 dir
    (``get_ascend_lib64_path()``, x86_64-linux on x86 hosts — the legacy
    inline probe in the multi-device thread launcher hardcoded 'aarch64-linux',
    resolved to '' and caused "SO path is empty" errors plus HCCL deadlocks)
    and cache the hit back into the info entry. A probe is skipped when the
    CANN scene is unresolvable (no ASCEND_OPP_PATH). Returns "" when the API
    is unknown or nothing resolves.
    """
    info = OpApiInfoKeeper().info_of(api_name)
    if info is None:
        return ""
    try:
        current = info.so_path
    except RuntimeError:
        return ""
    if current and os.path.isfile(current):
        return current
    try:
        lib64 = get_ascend_lib64_path()
    except RuntimeError:
        return current
    for so_name in _LIBOPAPI_FALLBACKS:
        candidate = os.path.join(lib64, so_name)
        if os.path.isfile(candidate):
            info._so_path = candidate
            return candidate
    return current


@dataclass
class OpApiInfo:
    params: OrderedDict
    # tensor/tensor-list names including input/output
    tensors: List[str] = field(default_factory=list, init=False)
    # scalar/scalar-list names
    scalars: List[str] = field(default_factory=list, init=False)
    # other param names except tensor/scalar/tensor-list/scalar-list
    others: List[str] = field(default_factory=list, init=False)
    category: str = field(default="", init=False, repr=False)
    _api_name: str = field(default="", init=False, repr=False)
    _so_path: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        for k, v in self.params.items():
            if v["type"] in ("aclTensor*", "aclTensorList*"):
                self.tensors.append(k)
            elif v["type"] in ("aclScalar*", "aclScalarList*"):
                self.scalars.append(k)
            else:
                self.others.append(k)

    @property
    def so_path(self) -> str:
        if not self._so_path and self.category == "builtin":
            self._so_path = _get_builtin_so_path(self._api_name)
        return self._so_path


class OpApiInfoKeeper(metaclass=Singleton):
    """
    Singleton Class to keep op api info
    """

    FUNC_PATTERN = r"\b([A-Za-z0-9_]+)\s+\b([A-Za-z0-9_]+)\s*\(([^){]*?)\)(?=\s*\;)"
    PARAM_PATTERN = r"(\w+[\s\*\s]*)(\w+)(\s*=\s*(.*?))?(,|$)"

    def __init__(self):
        self._api_info = None
        self._hdr_dirs: list = self._collect_header_dirs()
        if not self._hdr_dirs:
            raise RuntimeError(
                "OpApi header file path does not exist. Make sure opp-kernel package has been installed !!!"
            )

    def _collect_header_dirs(self) -> list:
        """Collect header directories in priority order: built-in → vendors(reversed) → custom(reversed).
        Each entry is (category, hdr_path, opp_root). opp_root is used to derive SO path."""
        dirs = []

        # Developer builds may expose an ACLNN symbol from libopapi before its
        # public header is installed into the CANN scene directory.  Allow TTK
        # to parse one or more source op_api directories in that situation.
        # These are treated as built-in APIs so the owning SO is still resolved
        # from the installed libopapi symbol table.
        source_so = os.getenv("TTK_OP_API_SO_PATH", "").strip()
        source_dirs = []
        for raw_dir in os.getenv("TTK_OP_API_HEADER_PATH", "").split(":"):
            source_dir = raw_dir.strip()
            if os.path.isdir(source_dir):
                category = "source" if source_so else "builtin"
                source_dirs.append((category, source_dir, source_so))

        # 1. Built-in: scene-based path
        builtin_dir = self._find_builtin_header_dir()
        if builtin_dir:
            dirs.append(("builtin", builtin_dir, ""))

        # 2. Vendor dirs (reversed for dict-overwrite semantics: later wins)
        for vopp in reversed(get_opp_paths("vendor")):
            hdr = os.path.join(vopp, "op_api", "include", "aclnnop")
            if not os.path.isdir(hdr):
                hdr = os.path.join(vopp, "op_api", "include")
            if os.path.isdir(hdr):
                dirs.append(("vendor", hdr, vopp))

        # 3. Custom dirs (reversed for dict-overwrite semantics: later wins)
        for copp in reversed(get_opp_paths("custom")):
            hdr = os.path.join(copp, "op_api", "include", "aclnnop")
            if not os.path.isdir(hdr):
                hdr = os.path.join(copp, "op_api", "include")
            if os.path.isdir(hdr):
                dirs.append(("custom", hdr, copp))

        # Explicit source headers describe the library selected by
        # TTK_OP_API_SO_PATH and must win over installed headers with the same
        # API name (for example, a newer generated MegaMoe signature).
        dirs.extend(source_dirs)

        return dirs

    @staticmethod
    def _find_builtin_header_dir() -> Optional[str]:
        """Find built-in aclnnop header directory via scene.info path."""
        try:
            opp_paths = get_opp_paths("builtin")
        except RuntimeError:
            return None
        if not opp_paths:
            return None
        opp_path = opp_paths[0]
        os_scene, os_arch = get_ascend_scene_info()
        if not os_scene:
            return None
        hdr_path = os.path.abspath(os.path.join(opp_path, "..", f"{os_arch}-{os_scene}", "include", "aclnnop"))
        return hdr_path if os.path.isdir(hdr_path) else None

    @property
    def api_info(self) -> dict:
        if self._api_info is None:
            self._api_info = self._parse_op_api_header_files()
        return self._api_info

    def has_api(self, api_name: str) -> bool:
        return api_name in self.api_info

    def info_of(self, api_name: str) -> OpApiInfo:
        return self.api_info.get(api_name, None)

    def _parse_op_api_header_files(self) -> dict:
        api_dict: Dict[str, OpApiInfo] = {}
        for category, hdr_path, opp_root in self._hdr_dirs:
            files = [
                f
                for f in os.listdir(hdr_path)
                if os.path.isfile(os.path.join(hdr_path, f)) and f.endswith((".h", ".hpp"))
            ]
            for f in files:
                with open(os.path.join(hdr_path, f)) as hf:
                    content = hf.read()
                for match in re.finditer(self.FUNC_PATTERN, content):
                    func_name = match.group(2).strip()
                    if not (func_name.startswith("aclnn") and func_name.endswith("GetWorkspaceSize")):
                        continue
                    api_name = func_name[:-16]
                    func_params = re.sub(r"\bconst\b(?!_)", "", match.group(3).strip())
                    params = OrderedDict({})
                    for param_match in re.finditer(self.PARAM_PATTERN, func_params):
                        param_type = param_match.group(1).strip().replace(" ", "")
                        param_name = param_match.group(2).strip()
                        if param_name in ("workspaceSize", "executor"):
                            break
                        default = param_match.group(4)
                        params.update({param_name: {"type": param_type, "default": default}})
                    info = OpApiInfo(params)
                    info._api_name = api_name
                    info.category = category
                    if category in ("custom", "vendor"):
                        info._so_path = os.path.join(opp_root, "op_api", "lib", "libcust_opapi.so")
                    elif category == "source":
                        info._so_path = opp_root
                    api_dict[api_name] = info
        return api_dict


# ── Cache snapshot transfer (pure-data keeper: main process snapshot →
#    forkserver worker bootstrap payload → restore below) ──────────────────


def snapshot_keeper() -> dict:
    """Picklable cache snapshot, taken after testcase parsing filled the cache."""
    keeper = OpApiInfoKeeper()
    if keeper._api_info is None:
        keeper.api_info  # noqa: B018  # property access triggers the lazy parse
    return {"api_info": keeper._api_info}


def restore_keeper_snapshot(snapshot: dict) -> None:
    """Restore a snapshot into this process' keeper cache.

    Injects the instance into the Singleton registry without running
    ``__init__`` (workers may restore before any header directory is
    resolvable). Snapshot hits are answered from the restored dict; lookups
    outside the snapshot miss to None, matching an unparseable environment.
    """
    if not snapshot or "api_info" not in snapshot:
        return
    key = (OpApiInfoKeeper, (), ())
    keeper = Singleton._instances.get(key)
    if keeper is None:
        keeper = OpApiInfoKeeper.__new__(OpApiInfoKeeper)
        keeper._hdr_dirs = []
        keeper._api_info = None
        Singleton._instances[key] = keeper
    keeper._api_info = snapshot["api_info"]
    logging.debug(f"OpApiInfoKeeper snapshot restored ({len(keeper._api_info)} entries)")
