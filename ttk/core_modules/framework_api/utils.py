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
from dataclasses import dataclass

import numpy as np

from ttk.core_modules.common.testcase.batch import batch_relation_kwargs
from ttk.utilities.dtypes import is_4bit_dtype, unpack_4bits

from ..common.invoke import accepts_var_keyword, golden_mode, sig_params

# eager（adapter.run_eager）与图模式（torch/graph_runner.py）共用的 warmup 轮数
WARMUP_COUNT = 1


@dataclass
class RunCtx:
    """eager 与图执行器共用的执行上下文（worker 构建，run_eager/run_graph 消费）。

    run_count/warmup_count 目前只有 eager 用（图执行器从 switches 自行推导），
    其余字段两路共用。"""

    backend: object
    dev_id: int
    switches: object
    resolved: object
    is_tensor_method: bool
    deterministic_level: int
    run_count: int
    warmup_count: int


def build_plugin_call(
    func, testcase, switches, plan, nested_inputs, *, backend, use_numpy, device=None, input_ranges=None
):
    """按 ParamPlan 装配插件入参（input/golden 插件共用）：build_args + 上下文注入 + 签名过滤。

    extra = testcase_name/tensor_formats/tensor_dtypes/use_numpy/short_soc_version/
    backend/golden_mode（device 与 input_ranges 可选）+ attributes 未消耗项 +
    batch_relation_kwargs；golden_mode 为本例生效值（override 优先于全局）；
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
        "golden_mode": golden_mode(testcase, switches),
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


def round_nps(
    backend, r, args, inplace_input_indexes, *, api_name=None, kwargs=None, adapter=None, backward_mode=False
):
    """确定性 MD5 取回（backward 时 (r, grads) 打平展开）。"""
    run_nps = []
    if r is not None:
        run_nps.extend(_result_to_numpy(backend, r, backward=backward_mode))
    elif api_name is not None:
        # 无返回值有状态算子(TF raw Resource*): 逐轮回读当轮变量(克隆)的更新值
        if adapter is None:
            from .adapter import adapter_for_api_name

            adapter = adapter_for_api_name(api_name)
        run_nps.extend(adapter.mutable_readback(api_name, args, kwargs))
    if inplace_input_indexes:
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                run_nps.append(backend.to_numpy(args[idx], safe=True))
    return run_nps


def _result_to_numpy(backend, result, backward=False):
    """结果展开成 numpy 列表；backward=True 时把 (forward, grads) 包装打平。

    该元组启发式必须受 backward 门控：原生返回 (Tensor, list[Tensor]) 的
    API 不带门控会被误拆（普通 result_to_numpy 对 list 项做 np.array 堆叠）。"""
    if backward and (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], list)
        and (result[1] and hasattr(result[1][0], "numpy") or result[1] == [])
    ):
        forward, grads = result
        nps = backend.result_to_numpy(forward) if forward is not None else []
        for g in grads:
            nps.extend(
                backend.result_to_numpy(g)
                if not isinstance(g, list)
                else [x for g2 in g for x in backend.result_to_numpy(g2)]
            )
        return nps
    return backend.result_to_numpy(result)


def output_nps(
    backend, result, args, kwargs, inplace_input_indexes, *, api_name=None, adapter=None, backward_mode=False
):
    """轮末输出取回：有返回值取返回值，否则回读被写参数位；声明被写位追加。

    TF 两阶段在正确性轮（Phase 1）末调用本函数定格结果——Phase 2 计时段
    不复位，别名 buffer 会继续累积，事后取读已被污染。"""
    if result is not None:
        result_nps = _result_to_numpy(backend, result, backward=backward_mode)
    else:
        result_nps = adapter.mutable_readback(api_name, args, kwargs) or None
    if inplace_input_indexes:
        if result_nps is None:
            result_nps = []
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                result_nps.append(backend.to_numpy(args[idx], safe=True))
    return result_nps


def _eager_mutable_keys(adapter, api_name, args, kwargs, is_inplace):
    """可变槽轮换键（int = args 下标，str = kwargs 参数名），mutable_slots 派生。

    torch 解析降级（命名原地但模型无被写位）时回退槽 0。"""
    positions, names = adapter.mutable_slots(api_name)
    keys = {i for i in positions if i < len(args) and args[i] is not None}
    keys |= {name for name in names if kwargs.get(name) is not None}
    if not keys and is_inplace and args and args[0] is not None:
        keys = {0}
    return keys


def _declared_inplace_keys(adapter, api_name, args, kwargs, inplace_input_indexes):
    """CSV 声明的被写位 → 轮换键。

    下标落在 args 且非 None 用下标；否则按张量序名映射（TF raw_ops 全走
    kwargs，经 adapter.tensor_param_names 取参数名）。无该钩子（torch/onnx
    默认）时声明只走 args 位。"""
    keys = set()
    names_hook = getattr(adapter, "tensor_param_names", None)
    param_names = names_hook(api_name) or () if names_hook else ()
    for i in inplace_input_indexes:
        if 0 <= i < len(args):
            if args[i] is not None:
                keys.add(i)
            continue
        if 0 <= i < len(param_names) and kwargs.get(param_names[i]) is not None:
            keys.add(param_names[i])
    return keys


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
