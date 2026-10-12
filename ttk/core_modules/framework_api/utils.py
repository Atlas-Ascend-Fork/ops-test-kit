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
Profiling utility functions shared between profiling.py and graph runners.

Framework-neutral — framework-specific fixups are delegated to the
FrameworkAdapter（bind_tensors / eager 钩子）与 backend 方法。
"""

import hashlib
import logging

import numpy as np


def compute_output_md5(arrays):
    """Compute MD5 hash of numpy output arrays for deterministic check."""
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]
    return hashlib.md5(  # noqa: S324  # 非安全用途：确定性校验和
        b"".join(arr.tobytes() if isinstance(arr, np.ndarray) else b"" for arr in arrays)
    ).hexdigest()


def finalize_det_status(md5_list, testcase_name):
    """Compare MD5 list and return deterministic status string."""
    if not md5_list:
        return None
    if len(md5_list) > 1:
        if len(set(md5_list)) != 1:
            logging.error(f"[{testcase_name}] E2E MD5 mismatch across {len(md5_list)} runs: {md5_list}")
            return "FAIL"
        logging.info(f"[{testcase_name}] E2E MD5 consistent across {len(md5_list)} runs: {md5_list[0]}")
    return "PASS"


def resolve_npu_preprocess_hook(testcase, switches):
    """TestSpec.npu_preprocess hook 的 e2e resolve（SpecResolver 统一前端）。"""
    from ..common.plugin import SpecResolver

    return SpecResolver(testcase.api_name, "e2e", switches.plugin_path).resolve_npu_preprocess()


def prepare_device_args(testcase, backend, dev_id, plan, raw_inputs, adapter=None):
    """Prepare device tensors and build args/kwargs for API execution.

    Shared logic between eager and graph execution modes:
    1. Convert raw_inputs to device tensors (preserving stride for torch)
    2. Framework-specific L2 fixups via adapter.bind_tensors
       (torch: npu format cast; tf: mutable-ref reset + const-value slots)
    3. Apply tensor list distribution if needed
    4. Build args/kwargs using plan

    Args:
        testcase: TestcaseE2e
        backend: Backend instance
        dev_id: device ID
        plan: ParamPlan
        raw_inputs: numpy input arrays
        adapter: FrameworkAdapter（None 时按 api_name 前缀解析）

    Returns:
        tuple: (args, kwargs) ready for API call
    """
    from ttk.utilities.container import nest_by_distribution

    if adapter is None:
        from .adapter import adapter_for_testcase

        adapter = adapter_for_testcase(testcase)
    use_framework_tensors = getattr(testcase, "tensors", None) is not None
    if use_framework_tensors:
        flat_tensors = testcase.flat_tensors
        preserve_stride = not backend.needs_numpy_fallback(testcase)
        dev_tensors = [
            backend.to_device(t, dev_id, preserve_stride=preserve_stride) if t is not None else None
            for t in flat_tensors
        ]
    else:
        dev_tensors = [backend.to_device(x, dev_id) if x is not None else None for x in raw_inputs]
    # mutable-ref(OpDef is_ref)变量在 testcase.tensors 上只创建一次, eager /
    # graph-const / graph-dynamic 三个模式共用同一个对象; 图模式计时阶段按设计不复位,
    # 后跑的模式会从前一个模式改过的状态起跑。每个模式入口按 raw_inputs 复位 ——
    # raw_inputs 是原始 numpy 输入, 不被任何执行阶段改写。复位/const 位/format_cast
    # 均经 adapter.bind_tensors（torch/tf 各自实现）。
    dev_tensors = adapter.bind_tensors(testcase, backend, dev_tensors, raw_inputs)
    dist = testcase.tensor_list_dist
    nested_tensors = nest_by_distribution(dev_tensors, dist) if dist else dev_tensors
    args, kwargs, _ = plan.build_args(nested_tensors)
    return args, kwargs


def unpack_4bit_outputs(testcase, result_nps):
    """Unpack uint8 packed 4-bit outputs to float4/int4 numpy arrays.

    When torch has no native float4 dtype, the NPU op may return uint8
    (packed) data. This post-processing step unpacks it based on the
    testcase's declared output dtypes, mirroring the _decode_output_bytes
    logic in ACLNN comparison.py and Kernel comparison.py.
    """
    if not result_nps:
        return result_nps
    output_dtypes = getattr(testcase, "flat_output_dtypes", None)
    if not output_dtypes:
        return result_nps

    from ttk.utilities import unpack_4bits

    for idx, dtype_str in enumerate(output_dtypes):
        if idx >= len(result_nps) or result_nps[idx] is None:
            continue
        if not isinstance(result_nps[idx], np.ndarray):
            continue
        ds = str(dtype_str)
        if ("float4" in ds or "int4" in ds) and result_nps[idx].dtype == np.uint8:
            try:
                result_nps[idx] = unpack_4bits(result_nps[idx], ds)
            except Exception as e:
                # 解包失败保留 uint8 原样送比对（比对必挂，此处留指向性线索）
                logging.debug(f"[unpack_4bit_outputs] unpack output#{idx} as {ds} failed: {e}")
    return result_nps
