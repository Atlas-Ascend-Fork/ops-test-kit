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
Profiling utility functions shared between worker.py and graph runners.

Framework-neutral — framework-specific fixups are delegated to the
FrameworkAdapter（bind_tensors / eager 钩子）与 backend 方法。
"""

import hashlib
import logging

import numpy as np

from ttk.core_modules.common.testcase.batch import batch_relation_kwargs
from ttk.utilities.dtypes import is_4bit_dtype, unpack_4bits

from ..common.invoke import accepts_var_keyword, sig_params

# eager（execute.py）与图模式（torch/graph_runner.py）共用的 warmup 轮数
WARMUP_COUNT = 1


def build_plugin_call(
    func, testcase, switches, plan, nested_inputs, *, backend, use_numpy, device=None, input_ranges=None
):
    """按 ParamPlan 装配插件入参（input/golden 插件共用）：build_args + 上下文注入 + 签名过滤。

    extra = testcase_name/tensor_formats/tensor_dtypes/use_numpy/short_soc_version/
    backend（device 与 input_ranges 可选）+ attributes 未消耗项 + batch_relation_kwargs；
    func 无 **kwargs 时只传已声明形参。返回 (args, kwargs)。
    """
    args, kwargs, extra_attrs = plan.build_args(nested_inputs)
    extra = {
        "backend": backend,
        "tensor_formats": testcase.tensor_formats,
        "tensor_dtypes": testcase.tensor_dtypes,
        "use_numpy": use_numpy,
        "short_soc_version": getattr(switches, "short_soc_version", None),
        "testcase_name": testcase.testcase_name,
    }
    if device is not None:
        extra["device"] = device
    if input_ranges is not None:
        extra["input_ranges"] = input_ranges
    extra.update(extra_attrs)
    extra.update(batch_relation_kwargs(testcase))
    params = sig_params(func)
    if accepts_var_keyword(params):
        kwargs.update(extra)
    else:
        kwargs.update({k: v for k, v in extra.items() if k in params})
    return args, kwargs


def compute_output_md5(arrays):
    """输出数组确定性 MD5：每位带 [idx:dtype:shape:nbytes] 边界，None 位显式留痕。

    仅拼 tobytes 时异形同字节（如 ([1,2],[3]) 与 ([1],[2,3])）会得到相同摘要。"""
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]
    h = hashlib.md5()  # noqa: S324  # 非安全用途：确定性校验和
    for idx, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray):
            h.update(f"[{idx}:None]".encode())
            continue
        h.update(f"[{idx}:{arr.dtype.str}:{arr.shape}:{arr.nbytes}]".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


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


def _unpack_4bit_guard(src, dst_dtype, label):
    """unpack_4bits 统一守卫：失败保留原样送比对（比对必挂，此处留指向性线索）。"""
    try:
        return unpack_4bits(src, dst_dtype)
    except Exception as exc:
        logging.debug(f"[4bit] unpack {label} as {dst_dtype} failed: {exc}")
        return None


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

    for idx, dtype_str in enumerate(output_dtypes):
        if idx >= len(result_nps) or result_nps[idx] is None:
            continue
        if not isinstance(result_nps[idx], np.ndarray):
            continue
        ds = str(dtype_str)
        if ("float4" in ds or "int4" in ds) and result_nps[idx].dtype == np.uint8:
            unpacked = _unpack_4bit_guard(result_nps[idx], ds, f"output#{idx}")
            if unpacked is not None:
                result_nps[idx] = unpacked
    return result_nps


def align_4bit_golden(output_nps, golden_nps):
    """golden 为 4bit dtype 而输出 dtype 更宽时的比对前对齐。

    两种封装形态统一先 view 成 uint8 再 unpack 到 golden 的 4bit dtype：
      1. 输出 uint8/int8 —— 每字节 2 个 4bit 值；
      2. 输出 int32/fp32 等宽 dtype —— 每 32bit 元素 8 个 4bit 值。
    消除纯封装 dtype 差异，让数值比对与容差解析落在 golden 的 4bit dtype 上。
    奇数元素数时打包尾部的填充 nibble（pack_4bits 奇数补零约定）裁齐 golden
    元素数；仅当多出尾部在打包粒度内（< 2*itemsize 个 nibble，即只可能是末
    元素填充位）且全为零时才裁——超出粒度或尾部非零属输出尺寸异常，保留
    unpack 结果让长度比对诚实暴露（无条件截断会把尺寸 bug 放行成假 PASS）。
    CSV 已声明 4bit 输出的场景在执行段 unpack_4bit_outputs 完成，此处为
    golden 侧驱动的兜底（golden 来自插件/回放时 dtype 以 golden 为准）。
    """
    if not output_nps or not golden_nps:
        return output_nps

    for i in range(min(len(output_nps), len(golden_nps))):
        res, gld = output_nps[i], golden_nps[i]
        if res is None or gld is None:
            continue
        if not isinstance(res, np.ndarray) or not isinstance(gld, np.ndarray):
            continue
        if not is_4bit_dtype(gld.dtype) or res.dtype == gld.dtype:
            continue
        unpacked = _unpack_4bit_guard(res, gld.dtype, f"output#{i}")
        if unpacked is None:
            continue
        surplus = unpacked.size - gld.size
        if 0 < surplus < 2 * res.itemsize and not unpacked[gld.size :].any():
            unpacked = unpacked[: gld.size]
        output_nps[i] = unpacked
    return output_nps
