#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""XPU (hardware accelerator) backend.

The torch-level device API (is_available /
device_count / to_device / synchronize) all come from TorchBackend via
``getattr(torch, torch_lib)``. This class is the generic accelerator backend used
for any torch_lib that is not 'npu' or 'cpu' (cuda/mlu/musa/...); the segment
identity (alias) is config-driven via _segment_name injected by _build.
"""

from __future__ import annotations

from .torch_backend import TorchBackend


class XpuTorchBackend(TorchBackend):
    """Generic accelerator backend (torch_lib bound at config time)."""

    def set_device(self, dev_id: int = 0):
        import torch

        getattr(torch, self.torch_lib).set_device(dev_id)

    def third_party_provider(self, framework: str) -> str:
        """本后端为三方采集方：产物按三方槽位（xpu_{framework}_*）落盘。"""
        return framework or "torch"
