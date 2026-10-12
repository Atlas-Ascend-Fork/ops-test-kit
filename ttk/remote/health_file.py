#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Atomic JSON file I/O for cross-process health state sharing.

Used by the heartbeat subprocess (writer) and EndpointView (reader).
Health file schema is a plain dict: {"endpoints": {"host:port": {alive, last_seen,
providers, hardware, ts}}} — written by heartbeat_loop, read by EndpointView.
"""

import json
import os
import tempfile
from typing import Optional


def atomic_write_json(path: str, data: dict) -> None:
    """Write dict as JSON to path atomically via tempfile + os.replace.

    Creates parent directories if missing. Uses os.replace for POSIX atomicity:
    readers never see a partially-written file.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".xpu_health_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_health_file(path: str = "") -> Optional[dict]:
    """Read health state dict from disk. Returns None on any failure.

    Tolerates: missing file, missing env var, corrupted JSON, permission errors.
    Workers call this on every dispatch — must never raise.
    """
    if not path:
        path = os.environ.get("TTK_XPU_HEALTH_PATH", "")
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError, OSError):
        return None
