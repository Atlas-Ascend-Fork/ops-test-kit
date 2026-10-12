#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Auto-discover and import ``torch.ops`` extension packages on demand.

A ``torch.ops.<namespace>.<op>`` API lives in an installed extension package
whose importable name equals the namespace (the torch extension convention).
This loader derives the package name from the api_name itself -- there is no
hardcoded namespace registry -- and imports it lazily the first time a matching
api_name is consulted at the registry match point (``FrameworkApiInfoKeeper.get``).
"""

import importlib
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import threading


class TorchOpsPackageRegistrationError(RuntimeError):
    """An installed custom-op package could not be discovered or imported."""


class TorchOpsPackageLoader:
    """Import ``torch.ops`` extension packages on demand, by namespace name."""

    CANN_ROOT_ENV_VARS = (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_AICPU_PATH",
    )
    LOCK = threading.RLock()
    _cann_dirs = None
    _paths_inserted = False

    @classmethod
    def _extension_namespace(cls, api_name):
        """Return the namespace for ``torch.ops.<ns>.<op>``, else None.

        Built-in torch namespaces (aten, prim, ...) have no importable package
        and are left for torch itself to register.
        """
        parts = api_name.split(".")
        if len(parts) < 4 or parts[0] != "torch" or parts[1] != "ops":
            return None
        return parts[2]

    @classmethod
    def cann_site_packages(cls):
        """CANN ``<root>/python/site-packages`` dirs from the environment."""
        if cls._cann_dirs is None:
            dirs, seen = [], set()
            for env_name in cls.CANN_ROOT_ENV_VARS:
                root = os.environ.get(env_name)
                if not root:
                    continue
                candidate = pathlib.Path(root).expanduser().joinpath("python", "site-packages").resolve()
                if candidate.is_dir() and candidate not in seen:
                    dirs.append(str(candidate))
                    seen.add(candidate)
            cls._cann_dirs = dirs
        return cls._cann_dirs

    @classmethod
    def environment_summary(cls):
        return ", ".join(f"{name}={os.environ.get(name) or '<unset>'}" for name in cls.CANN_ROOT_ENV_VARS)

    @classmethod
    def ensure_registered(cls, api_name):
        """Import the extension package for ``torch.ops.<ns>.<op>`` if needed.

        Idempotent and thread-safe. The package name is the namespace itself
        (torch extension convention). CANN site-packages are probed first so a
        namespace name never collides with an unrelated global package; a
        global ``find_spec`` fallback covers pip-installed extensions. Built-in
        namespaces (aten/prim/...) resolve to nothing and are skipped.
        """
        ns = cls._extension_namespace(api_name)
        if ns is None or ns in sys.modules:
            return

        with cls.LOCK:
            cann_dirs = cls.cann_site_packages()
            spec = importlib.machinery.PathFinder.find_spec(ns, path=cann_dirs) if cann_dirs else None
            if spec is not None:
                cls._insert_paths(cann_dirs)
            else:
                spec = importlib.util.find_spec(ns)
                if spec is None:
                    return  # built-in namespace, or genuinely missing
            try:
                importlib.import_module(ns)
            except Exception as error:
                checked = ", ".join(cann_dirs) or "<none>"
                raise TorchOpsPackageRegistrationError(
                    f"Cannot register {api_name!r}: importing torch.ops namespace "
                    f"package {ns!r} failed with {type(error).__name__}: {error}. "
                    f"CANN environment: {cls.environment_summary()}. "
                    f"Checked site-packages: {checked}. Source the target CANN "
                    "environment and install the matching extension package."
                ) from error

    @classmethod
    def _insert_paths(cls, cann_dirs):
        if cls._paths_inserted:
            return
        for candidate in reversed(cann_dirs):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        importlib.invalidate_caches()
        cls._paths_inserted = True
