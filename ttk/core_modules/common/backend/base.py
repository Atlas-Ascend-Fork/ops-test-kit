#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Hardware backend abstraction for NPU / XPU / CPU.

    Framework-neutral contract layer: no torch/tf import lives here (asserted
    by tests/core_modules/common/backend/test_backends.py). Torch-family
    specifics (``torch_lib`` injection, ``torch.<lib>.get_device_name``) are
    implemented in TorchBackend; TF backends override the identity hooks.

    Device-capability contract (6 first-principle methods):
      is_available / device_count / to_device / synchronize / from_numpy / to_numpy

    Identity contract:
      device_name()  -- hardware MODEL name (e.g. 'Ascend910B3'). Default
                         degrades to device_type(); TorchBackend overrides
                         via torch.<torch_lib>.get_device_name, CpuTorchBackend
                         keeps the degrade (no torch.cpu.get_device_name exists).
      device_type()  -- the backend alias; class attribute _segment_name
                        ('npu'/'cpu'/'npu'(tf)/'cpu'(tf)).
      is_npu()       -- convenience predicate (default False; NpuTorchBackend overrides).
      has_device()   -- whether device resources are used.
      soc_series()   -- short SoC series; default degrades to device_name() (model);
                        NpuTorchBackend overrides via get_npu_hw_info(model).

    profiler config:
      profile        -- class attribute dict (torch family reads torch_lib
                        out of it; see TorchBackend / profiler factory).
    """

    profile: dict
    # Backend alias（device_type() 返回值）；子类按硬件设类属性。
    _segment_name: str = ""

    def device_name(self, dev_id: int = 0) -> str:
        """Hardware MODEL name; default degrades to the config-segment name.

        TorchBackend overrides this via ``torch.<torch_lib>.get_device_name``
        (e.g. 'Ascend910B3'); CpuTorchBackend keeps the degrade (no
        ``torch.cpu.get_device_name`` exists).
        """
        return self.device_type()

    def device_type(self) -> str:
        """Return the backend alias（类属性 _segment_name：'npu'/'cpu'/...）。"""
        return self._segment_name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if device is available."""

    @abstractmethod
    def device_count(self) -> int:
        """Return number of available devices."""

    def to_device(self, tensor, dev_id: int = 0, preserve_stride: bool = False):
        """Move tensor to target device.

        preserve_stride=True: preserve non-contiguous stride (torch only).
        Default: framework's standard device move.
        """
        raise NotImplementedError

    @abstractmethod
    def synchronize(self, dev_id: int = 0):
        """Synchronize device (for accurate timing)."""

    @abstractmethod
    def from_numpy(self, arr: np.ndarray):
        """Convert numpy array to framework tensor (zero-copy preferred)."""

    def to_numpy(self, tensor, safe: bool = False) -> np.ndarray:
        """Convert framework tensor to numpy array.

        safe=True: extra detach/clone for inplace-extracted tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def has_device(self) -> bool:
        """Whether use device resource."""

    def is_npu(self) -> bool:
        """Whether this is an NPU backend. Default False; NpuTorchBackend overrides."""
        return False

    def soc_series(self) -> str:
        """Default: degrade to device_name() (model). NpuTorchBackend overrides short."""
        return self.device_name()

    # ========== Framework tensor-lifecycle methods ==========

    def clone(self, tensor):
        """Make a copy of a framework tensor (for inplace backup)."""
        import copy

        return copy.copy(tensor)

    def stage_mutable_rounds(self, args, kwargs, keys, total):
        """预热/计时多轮的可变槽载体预置：返回逐轮装载函数 stage(i)。

        键约定：int = args 下标；str = kwargs 参数名。默认实现逐轮换新克隆
        （torch 原地算子需要干净载体）；keys 为空时 stage 为 no-op。
        """
        backups = {key: self.clone(args[key] if isinstance(key, int) else kwargs[key]) for key in keys}
        clones = {key: [self.clone(backup) for _ in range(total)] for key, backup in backups.items()}

        def stage(i):
            for key, round_carriers in clones.items():
                value = round_carriers[i]
                if isinstance(key, int):
                    args[key] = value
                else:
                    kwargs[key] = value

        return stage

    def is_npu_only(self, api_name: str) -> bool:
        """Whether the API can only run on device (not CPU). Default: False."""
        return False

    def supports_graph_mode(self) -> bool:
        """Whether this backend supports graph mode compilation."""
        return True

    def supports_format_cast(self) -> bool:
        """Whether this backend supports NPU format cast."""
        return False

    def set_deterministic_level(self, level):  # noqa: B027
        """Set deterministic computation level. Default: no-op."""

    def reset_peak_memory_stats(self, dev_id: int = 0):  # noqa: B027
        """Reset peak memory stats. Default: no-op; NpuTorchBackend overrides."""

    def max_memory_allocated(self, dev_id: int = 0):  # noqa: B027
        """Current peak allocated device memory in bytes, or None if unsupported."""

    def set_device(self, dev_id: int = 0):  # noqa: B027
        """Set the current device for eager execution. Default: no-op.

        Framework-specific (torch_npu needs a process-global device; TF's
        npu_device.open().as_default() in __init__ already pinned the device).
        """

    def device_scope(self, dev_id=0):
        """Context manager for device-scoped execution. Default: nullcontext."""
        from contextlib import nullcontext

        return nullcontext()

    def wrap_eager_callable(self, resolved, api_name=None):
        """Wrap an API callable for eager execution. Default: no-op."""
        return resolved

    def needs_numpy_fallback(self, testcase) -> bool:
        """Whether plugins should receive numpy arrays (instead of framework tensors).

        True when the framework cannot natively represent the testcase's dtype
        (e.g. torch with float8/int4), so data is passed as numpy arrays instead.
        """
        return False

    def inputs_from_numpy(self, testcase, raw_inputs):
        """Convert flat numpy arrays to framework tensors.

        When needs_numpy_fallback() is True (framework cannot natively
        represent the dtype, e.g. torch int4/float8), raw numpy arrays
        are returned as-is so plugins receive numpy instead of crashing
        in the framework tensor conversion.
        """
        if self.needs_numpy_fallback(testcase):
            return list(raw_inputs)
        return [self.from_numpy(arr) if arr is not None else None for arr in raw_inputs]

    def result_to_numpy(self, result, copy=False):
        """Convert an API result (Tensor/tuple/scalar) to list of numpy arrays."""
        if result is None:
            return [None]
        if isinstance(result, (tuple, list)):
            nps = []
            for r in result:
                if r is None:
                    nps.append(None)
                elif hasattr(r, "numpy") and callable(r.numpy):
                    arr = self.to_numpy(r)
                    nps.append(arr.copy() if copy else arr)
                else:
                    nps.append(np.array(r))
            return nps
        if hasattr(result, "numpy") and callable(result.numpy):
            arr = self.to_numpy(result)
            return [arr.copy() if copy else arr]
        return [np.array(result)]
