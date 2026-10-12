#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MC2 multi-device dispatcher — framework entry point.

All operator-specific golden logic has been migrated to per-operator
tests/assets/ directories in ops-transformer. This module retains only:
  - golden_multi_device_compare: migration dispatcher: SKIP if no plugin, delegate if plugin present
  - Public framework_* stubs for backward compatibility

TTK = multi-device test framework only. Golden computation lives in
ops-transformer/mc2/<op>/tests/assets/impl/golden.py, loaded via --plugin.
"""

import logging
from typing import Dict, List

import numpy

from .testcase import TestcaseAclnn

# ---------------------------------------------------------------------------
# Migration dispatcher
# ---------------------------------------------------------------------------

_MIGRATED_APIS = (
    "AlltoAllAllGatherBatchMatMul",
    "BatchMatMulReduceScatterAlltoAll",
    "GroupedMatMulAlltoAllv",
    "AlltoAllvGroupedMatMul",
    "MoeDistributeDispatch",
    "MoeDistributeCombine",
    "AllGatherMatmul",
    "MatmulReduceScatter",
    "MatmulAlltoAll",
    "AlltoAllMatmul",
    # A5 operators (quant / V2 / add_rms_norm variants)
    "MatmulAllReduce",
    "WeightQuantMatmulAllReduce",
    "QuantMatmulAllReduce",
    "AllGatherMatmulV2",
    "MatmulReduceScatterV2",
    "InplaceMatmulAllReduceAddRmsNorm",
    "MatmulAllReduceAddRmsNorm",
    "GroupedMatMulAllReduce",
    "QuantAllReduce",
    "QuantReduceScatter",
    "QuantGroupedMatMulAlltoAllv",
    "AlltoAllvQuantGroupedMatMul",
)


def golden_multi_device_compare(thread_contexts: Dict[int, TestcaseAclnn], device_ids: List[int], all_precision: list):
    """Multi-device golden dispatcher.

    All MC2 operators have been migrated to ops-transformer. When --plugin
    is provided, the plugin's multi_device_golden is called by the framework
    before reaching here. If we reach here without a plugin, the operator
    is marked SKIP(MIGRATED).

    DistributeBarrier is a synchronization-only API with no value output.
    """
    api_name = next(iter(thread_contexts.values())).api_name

    if api_name.startswith("aclnnDistributeBarrier"):
        for did in device_ids:
            thread_contexts[did].golden_tensors = []
            all_precision.append(f"rank{did}:PASS(EXECUTED)")
        return

    if any(m in api_name for m in _MIGRATED_APIS):
        logging.warning(
            f"{api_name}: golden has been migrated to ops-transformer; "
            f"use --plugin <op>/tests/assets to run precision comparison"
        )
        for did in device_ids:
            thread_contexts[did].golden_tensors = []
            all_precision.append(f"rank{did}:SKIP(MIGRATED)")
        return

    # Non-migrated APIs: mark as no golden available
    logging.warning(f"{api_name}: no multi-device golden implementation")
    for did in device_ids:
        thread_contexts[did].golden_tensors = []
        all_precision.append(f"rank{did}:SKIP(NO_GOLDEN)")


# ---------------------------------------------------------------------------
# Public framework API stubs (backward compatibility)
# ---------------------------------------------------------------------------
# These are kept for any external code that may still import them.
# The actual implementations now live in ops-transformer's golden_utils.py.


def framework_to_torch_f32(t):
    import torch

    if t is None:
        return None
    if isinstance(t, numpy.ndarray):
        dtype_str = str(t.dtype)
        if "e8m0" in dtype_str or "float8_e8m0" in dtype_str:
            raw = t.view(numpy.uint8).astype(numpy.float64)
            arr = numpy.power(2.0, raw - 127).astype(numpy.float32)
        else:
            try:
                arr = t.astype(numpy.float32, copy=False)
            except (TypeError, ValueError):
                arr = t.view(numpy.uint8).astype(numpy.float32, copy=False)
        return torch.from_numpy(arr)
    if hasattr(t, "dtype") and str(t.dtype).replace("torch.", "") in ("float8_e4m3fn", "float8_e5m2", "hifloat8"):
        return t.float()
    return t.float()
