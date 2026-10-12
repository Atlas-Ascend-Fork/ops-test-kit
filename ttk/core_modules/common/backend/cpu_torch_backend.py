#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU backend (for golden baseline or no-device testing).

Renamed from CpuBackend. CPU has no ``torch.cpu.get_device_name`` (torch exposes
no ``torch.cpu`` module), so device_name is permanently overridden to alias()
('cpu') — this is the contract, not a placeholder. CPU backends never go
through the profiler ``_build`` path (which would inject torch_lib/profile), so
both are provided as class attributes here for state self-consistency.
"""

from __future__ import annotations

from ....utilities import numpy_to_torch_tensor
from .torch_backend import TorchBackend


class CpuTorchBackend(TorchBackend):
    """CPU backend (for golden baseline or no-device testing)."""

    torch_lib = "cpu"
    # torch_lib key mirrors what _build injects for config-driven backends, so
    # TorchProfiler.__init__ reading profile["torch_lib"] works uniformly.
    profile = {"torch_lib": "cpu", "profiler": {"activities": ["CPU"]}}
    # alias = segment name. CPU never goes through _build, so set the class
    # attribute directly (mirrors the torch_lib/profile class-attr pattern).
    _segment_name = "cpu"

    def device_name(self, dev_id: int = 0) -> str:
        return self.device_type()  # CPU has no torch.cpu.get_device_name

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def to_device(self, tensor, dev_id=0, preserve_stride=False):
        if preserve_stride:
            return self._to_device_preserving_stride(tensor, dev_id)
        return self.from_numpy(tensor)

    def synchronize(self, dev_id=0):
        pass

    def from_numpy(self, arr):
        return numpy_to_torch_tensor(arr)

    def has_device(self) -> bool:
        return False
