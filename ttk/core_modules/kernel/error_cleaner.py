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
CANN ErrorManager residual error cleaner.

Clears ErrorManager errors at the start of each testcase to prevent
cross-testcase contamination (e.g., profiling errors affecting subsequent tiling).
"""

__all__ = ["clear_error_manager"]


import ctypes
import logging

from .compile.cext_loader import load_cext

_cleaner = None


def _ensure_loaded():
    global _cleaner
    if _cleaner is not None:
        return
    # Preload liberror_manager.so globally so the cleaner cext and the CANN runtime
    # resolve the same ErrorManager instance; otherwise ClearErrorManager would clear
    # a different symbol copy and residual errors would survive.
    ctypes.CDLL("liberror_manager.so", mode=ctypes.RTLD_GLOBAL)
    _cleaner = load_cext("libttk_error_manager_cleaner.so", "error_manager_cleaner")
    _cleaner.ClearErrorManager.restype = ctypes.c_int


def clear_error_manager():
    """Clear CANN ErrorManager residual errors. Call before each testcase."""
    try:
        _ensure_loaded()
        _cleaner.ClearErrorManager()
    except Exception as e:
        logging.debug(f"clear_error_manager skipped: {e}")
