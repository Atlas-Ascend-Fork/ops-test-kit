#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for the e2e npu_memory field (torch_npu runtime peak, MB) and _capture_peak_memory.
"""

import types
from unittest.mock import MagicMock

import pytest

from ttk.core_modules.framework_api import execute as e2e_profiling
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure


def test_npu_memory_is_result_column():
    """npu_memory is a result column and defaults to None."""
    titles = FrameworkApiReturnStructure.get_titles()
    assert "npu_memory" in titles
    assert FrameworkApiReturnStructure().npu_memory is None


def _npu_backend(peak_bytes=805306368):
    backend = MagicMock()
    backend.is_npu.return_value = True
    backend.max_memory_allocated.return_value = peak_bytes
    return backend


def test_peak_memory_passthrough_on_non_npu():
    """Non-NPU backend passes through; npu_memory stays None."""
    backend = MagicMock()
    backend.is_npu.return_value = False
    rs = types.SimpleNamespace(npu_memory=None)
    calls = []

    def fn(x):
        calls.append(x)
        return x + 1

    assert e2e_profiling._capture_peak_memory(backend, 0, rs, fn, 41) == 42
    assert calls == [41]
    backend.reset_peak_memory_stats.assert_not_called()
    assert rs.npu_memory is None


def test_peak_memory_records_mb():
    """NPU backend: run fn after reset, record peak /1e6 into npu_memory."""
    backend = _npu_backend(peak_bytes=805306368)
    rs = MagicMock()
    e2e_profiling._capture_peak_memory(backend, 3, rs, lambda: "ok")
    backend.reset_peak_memory_stats.assert_called_once_with(3)
    backend.max_memory_allocated.assert_called_once_with(3)
    assert rs.npu_memory == pytest.approx(805.306368)


def test_peak_memory_recorded_even_if_fn_raises():
    """Peak is still recorded (finally) when fn raises; the exception propagates."""
    backend = _npu_backend(peak_bytes=1_000_000)
    rs = MagicMock()

    def boom():
        raise RuntimeError("op failed")

    with pytest.raises(RuntimeError):
        e2e_profiling._capture_peak_memory(backend, 0, rs, boom)
    assert rs.npu_memory == pytest.approx(1.0)


def test_peak_memory_unsupported_backend_keeps_none():
    """npu_memory stays None when max_memory_allocated returns None (unsupported)."""
    backend = _npu_backend(peak_bytes=None)
    rs = types.SimpleNamespace(npu_memory=None)
    e2e_profiling._capture_peak_memory(backend, 0, rs, lambda: None)
    assert rs.npu_memory is None


def test_peak_memory_reset_failure_degrades():
    """reset API failure degrades to no capture; fn still runs and returns."""
    backend = _npu_backend()
    backend.reset_peak_memory_stats.side_effect = RuntimeError("reset boom")
    rs = types.SimpleNamespace(npu_memory=None)
    assert e2e_profiling._capture_peak_memory(backend, 0, rs, lambda: "ok") == "ok"
    assert rs.npu_memory is None


def test_peak_memory_read_failure_degrades():
    """max_memory_allocated failure degrades to no capture and never masks fn's result."""
    backend = _npu_backend()
    backend.max_memory_allocated.side_effect = RuntimeError("read boom")
    rs = types.SimpleNamespace(npu_memory=None)
    assert e2e_profiling._capture_peak_memory(backend, 0, rs, lambda: "ok") == "ok"
    assert rs.npu_memory is None
