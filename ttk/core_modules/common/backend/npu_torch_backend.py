#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import numpy as np

from ....utilities import (
    is_4bit_dtype,
    is_torch_native_dtype,
)
from ..device import get_npu_hw_info
from .torch_backend import TorchBackend


class NpuTorchBackend(TorchBackend):
    """NPU backend.

    Inherits the torch-generic device API (is_available/device_count) from
    TorchBackend — those go through ``torch.npu.*``.

    device_name is inherited from TorchBackend (``torch.npu.get_device_name``
    returns the SoC model, e.g. 'Ascend910B3'). soc_series derives the short
    series from that model via get_npu_hw_info (no more asys dependency).
    """

    torch_lib = "npu"
    _segment_name = "npu"
    # profiler 工厂消费（profiler/factory.py）：builtin → NpuProfiler
    profile = {"torch_lib": "npu", "profiler": "builtin"}

    def is_npu(self) -> bool:
        return True

    def set_device(self, dev_id: int = 0):
        import torch_npu  # NPU-only: keep import in method body (lazy)

        torch_npu.npu.set_device(dev_id)

    def to_device(self, tensor, dev_id=0, preserve_stride=False):
        if preserve_stride:
            return self._to_device_preserving_stride(tensor, dev_id)
        import torch_npu  # NPU-only: keep import in method body (lazy)

        str_dtype = str(tensor.dtype).split(".")[-1]
        if is_torch_native_dtype(str_dtype):
            torch_tensor = self.from_numpy(tensor)
            return torch_tensor.npu(dev_id)
        if is_4bit_dtype(str_dtype):
            torch_tensor = self.from_numpy(tensor)
            return torch_tensor.npu(dev_id)
        if str_dtype == "float8_e8m0":
            return self.from_numpy(tensor).npu(dev_id)
        np_fp32 = tensor.astype(np.float32)
        npu_torch_tensor = torch_npu.npu_dtype_cast(
            self.from_numpy(np_fp32).npu(dev_id), dtype=getattr(torch_npu, str_dtype)
        )
        return npu_torch_tensor

    # to_numpy/from_numpy inherited from TorchBackend (was a dead byte-identical
    # override that also dropped the .contiguous() guard the parent applies).

    def soc_series(self):
        try:
            hw_info = get_npu_hw_info(self.device_name())
            return hw_info["short_soc_version"]
        except (FileNotFoundError, RuntimeError, KeyError):
            return self.device_name()

    def set_deterministic_level(self, level):
        import torch_npu

        torch_npu.npu.set_deterministic_level(level)

    def reset_peak_memory_stats(self, dev_id: int = 0):
        import torch_npu  # NPU-only: keep import in method body (lazy)

        torch_npu.npu.reset_peak_memory_stats(dev_id)

    def max_memory_allocated(self, dev_id: int = 0):
        import torch_npu  # NPU-only: keep import in method body (lazy)

        return int(torch_npu.npu.max_memory_allocated(dev_id))
