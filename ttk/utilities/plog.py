#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Ascend plog error extraction utilities.
"""

__all__ = ["extract_plog_errors"]

import glob
import os

from .file_utils import read_file


def extract_plog_errors(max_lines=10, pid=None):
    """Extract recent error lines from the given process's plog file.

    Scans ``~/ascend/log/debug/plog/plog-<pid>_*.log`` for ``[ERROR]`` lines
    and returns the last *max_lines* entries.  Falls back to the last
    *max_lines* raw lines when no ``[ERROR]`` entries exist, or a placeholder
    message when no plog file is found or it cannot be read.

    Args:
        max_lines: Maximum number of log lines to return.
        pid: Process ID to look up. Defaults to ``os.getpid()``.

    Returns:
        list[str]: Extracted error / log lines.
    """
    if pid is None:
        pid = os.getpid()
    if max_lines <= 0:
        return []
    plog_home = os.path.expanduser("~/ascend/log/debug/plog")
    plog_pattern = f"plog-{pid}_*.log"
    candidates = glob.glob(f"{plog_home}/{plog_pattern}")
    if not candidates:
        return ["No plog file found for current process, please check ascend log manually for details"]
    # glob returns arbitrary order — pick the most recently modified plog
    log_path = max(candidates, key=os.path.getmtime)
    try:
        content = read_file(log_path).decode("UTF-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ["Failed to read plog file, please check ascend log manually for details"]
    filtered = [line for line in content if "[ERROR]" in line]
    return (filtered if filtered else content)[-max_lines:]
