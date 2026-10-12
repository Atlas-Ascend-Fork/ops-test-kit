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
Golden data generation with three-level fallback:
1. CSV golden_api column (per-testcase override)
2. Custom plugin via plugin_loader (level_type="e2e")
3. Default: same API on CPU

All paths use testcase.get_param_plan() to get consistent arg ordering
with profiling.py — same plan, same (*args, **kwargs) layout.
"""

import logging

from ttk.core_modules.common.testcase.batch import batch_relation_kwargs
from ttk.core_modules.framework_api.api_param import build_positional_args
from ttk.utilities import DTYPE_PROMOTE_MAP
from ttk.utilities.container import nest_by_distribution

from ..common.backend import detect_framework, resolve_api
from ..common.invoke import accepts_var_keyword, golden_mode, sig_params
from ..common.plugin import SpecResolver
from .adapter import adapter_for_api_name
from .info_keeper import FrameworkApiInfoKeeper

_cpu_backend_cache = {}


def _get_cpu_backend(framework="torch"):
    """Get cached CPU backend for the given framework."""
    if framework not in _cpu_backend_cache:
        if framework == "tf":
            from ..common.backend.cpu_tf_backend import CpuTfBackend

            _cpu_backend_cache[framework] = CpuTfBackend()
        else:
            from ..common.backend.cpu_torch_backend import CpuTorchBackend

            _cpu_backend_cache[framework] = CpuTorchBackend()
    return _cpu_backend_cache[framework]


def _promote_raw_inputs(testcase, raw_inputs, switches=None):
    """golden_mode=Promote 时按 DTYPE_PROMOTE_MAP 抬高浮点输入精度（fp16/bf16/float8→fp32、
    fp32→fp64），整型不动。

    提升为精确转换、不改值，仅抬高高精度 golden 的中间计算精度——cross_check /
    mix_tolerance 判据要求 golden 是独立于被测双方的高精度真值。对任意 API 一致
    适用（mutable/有状态算子的输出以赋值语义为主，提升后与被测逐位一致）。
    """
    # 模式判定放在此处而非调用点:让 generate_golden 少一层分支。
    # golden_mode_override 由 profiling 侧在 cross_check/mix_tolerance 判据下设置。
    if golden_mode(testcase, switches) != "Promote":
        return raw_inputs
    # TestcaseE2e 暴露的是 flat_tensor_dtypes(与 raw_inputs 一一对齐);
    # flat_input_dtypes 在该类上并不存在,取到 None 会让本函数直接原样返回、Promote 空转。
    flat_dtypes = getattr(testcase, "flat_tensor_dtypes", None)
    if not flat_dtypes:
        return raw_inputs
    promoted = list(raw_inputs)
    for idx, array in enumerate(promoted):
        if array is None or idx >= len(flat_dtypes):
            continue
        target = DTYPE_PROMOTE_MAP.get(flat_dtypes[idx])
        if target is None:
            continue
        try:
            promoted[idx] = array.astype(target)
        except (TypeError, ValueError):
            # 只兜 numpy astype 实际会抛的两类:目标 dtype 非法(TypeError)、值无法转换(ValueError)。
            # 其余异常(如 array 非 ndarray)属调用方传参错误,不在此吞掉。
            logging.warning(f"[golden] promote input#{idx} {flat_dtypes[idx]}->{target} failed, keep original")
    return promoted


def generate_golden(testcase, raw_inputs, plugin_path=None, switches=None, backend="cpu"):
    """
    Generate golden data using three-level fallback.

    Args:
        testcase: TestcaseE2e
        raw_inputs: list of numpy arrays (flat, one per tensor)
        plugin_path: custom plugin path from SWITCHES
        switches: SWITCHES object
        backend: backend name string (e.g. 'cpu', 'npu')

    Returns:
        list of numpy arrays (golden outputs)
    """
    golden_api = getattr(testcase, "golden_api", None)
    dist = testcase.tensor_list_dist
    api_name = testcase.api_name

    # mutable-ref 算子原地改写变量, golden 生成在设备执行之后、与被测共用同一个
    # tf.Variable —— 不复位等于在被测改过 N 遍的脏状态上再算一遍。复位经
    # adapter 钩子（tf 实现；torch no-op），共享层不持框架分支。
    try:
        adapter_for_api_name(api_name).reset_mutable_inputs(
            testcase, getattr(testcase, "flat_tensors", None), raw_inputs
        )
    except Exception as _e:  # pragma: no cover
        logging.warning(f"[golden] mutable-ref reset skipped: {_e}")
    framework = detect_framework(api_name)
    cpu_backend = _get_cpu_backend(framework)

    # 抬高浮点输入精度后再算 golden,使其成为独立于被测与竞品的高精度真值。
    # 覆盖面:golden_api 路径与「同 API 跑 CPU」路径(两者都吃这里的 raw_inputs)。
    # 自定义插件路径(Priority 2)不经 raw_inputs、由插件自行取数,故不受此处影响。
    raw_inputs = _promote_raw_inputs(testcase, raw_inputs, switches)

    # --- Priority 1: CSV golden_api column (different API) ---
    if golden_api:
        if golden_api.lower() == "disable":
            return ["SUPPRESSED"]
        if cpu_backend.is_npu_only(golden_api):
            logging.warning(f"[golden] golden_api {golden_api} is NPU-only (no CPU kernel)")
            return ["UNSUPPORTED"]
        if not testcase.is_dtype_support():
            logging.debug(f"[golden] Skip golden_api={golden_api} for {api_name}: non-native dtype detected")
            return ["UNSUPPORTED"]
        if golden_api == api_name:
            golden_api_info = testcase.get_api_info()
        else:
            golden_api_info = FrameworkApiInfoKeeper().get(golden_api)
        return _run_api_on_cpu(
            golden_api, raw_inputs, testcase, dist, api_info=golden_api_info, cpu_backend=cpu_backend
        )

    # --- Priority 2: Custom plugin via plugin_loader ---
    func = SpecResolver(api_name, "e2e", plugin_path).resolve_golden()
    if func is not None:
        return _call_plugin_with_plan(testcase, func, switches, backend, cpu_backend)

    # --- Priority 3: Same API on CPU ---
    if not testcase.is_dtype_support():
        logging.debug(f"[golden] Skip CPU golden for {api_name}: non-native dtype detected, no custom plugin found")
        return ["UNSUPPORTED"]
    if cpu_backend.is_npu_only(api_name):
        # 快速失败：NPU-only 前缀无 CPU kernel，不再白跑 resolve → 构造 CPU 输入
        # → 调用失败整链；UNSUPPORTED ∉ PLACEHOLDER_PASS → 比对 FAIL（真值缺失可见）。
        logging.warning(
            f"[golden] {api_name} is NPU-only (no CPU kernel); "
            "provide a custom plugin via --plugin or specify golden_api in CSV"
        )
        return ["UNSUPPORTED"]
    try:
        return _run_api_on_cpu(api_name, raw_inputs, testcase, dist, cpu_backend=cpu_backend)
    except Exception as e:
        raise RuntimeError(
            f"{api_name} cannot run on CPU and has no e2e custom plugin. "
            f"Please provide a custom plugin via --plugin or specify golden_api in CSV."
        ) from e


def _run_backward_on_cpu(api_name, raw_inputs, testcase, dist, cpu_backend):
    """backward golden：CPU forward + autograd.grad（与被测同构）。"""

    import torch

    cpu_inputs = [cpu_backend.from_numpy(x.copy()) if x is not None else None for x in raw_inputs]
    for t in cpu_inputs:
        if t is not None and hasattr(t, "requires_grad_") and t.is_floating_point():
            t.requires_grad_(True)
    golden_api_name = api_name + ".backward"
    golden_api_info = FrameworkApiInfoKeeper().get(golden_api_name)
    if golden_api_info is not None:
        logging.warning(
            f"[golden] registered backward entry {golden_api_name} is not executable "
            f"(no EagerExecutor implementation); falling back to manual forward + grad"
        )
    # keeper 后缀条目仅作登记提示（无独立执行器）：手工 forward + grad
    args = list(cpu_inputs)
    r = _invoke_plain(api_name, args, testcase)
    grad_inputs = [t for t in args if getattr(t, "requires_grad", False)]
    outputs = r if isinstance(r, (tuple, list)) else [r]
    outputs = [o for o in outputs if getattr(o, "requires_grad", False)]
    if outputs and grad_inputs:
        grad_outputs = [torch.ones_like(o) for o in outputs]
        grads = torch.autograd.grad(outputs, grad_inputs, grad_outputs=grad_outputs, allow_unused=True)
        return _flatten_cpu_outputs(r) + [g.numpy() for g in grads if g is not None]
    return _flatten_cpu_outputs(r)


def _invoke_plain(api_name, args, testcase):
    """位置直调（优先复用 testcase 的 param_plan；无 plan 时按位直调——仅 CPU 参照）。"""
    import importlib

    module_path, _, attr = api_name.rpartition(".")
    fn = getattr(importlib.import_module(module_path), attr)
    plan = testcase.get_param_plan()
    if plan is not None:
        pos_args, kwargs, _ = plan.build_args(args)
        return fn(*pos_args, **kwargs)
    return fn(*args)


def _flatten_cpu_outputs(result):
    out = []
    items = result if isinstance(result, (tuple, list)) else [result]
    for o in items:
        if hasattr(o, "numpy"):
            out.append(o.detach().numpy())
        elif isinstance(o, (tuple, list)):
            out.extend(x.detach().numpy() for x in o if hasattr(x, "numpy"))
    return out


def _run_api_on_cpu(api_name, raw_inputs, testcase, dist, api_info=None, cpu_backend=None):
    """Execute API on CPU, return numpy result list.

    api_info=None: reuse testcase's param plan (same API path).
    api_info=...: independently match overload (golden_api path).
    """
    if cpu_backend is None:
        framework = detect_framework(api_name)
        cpu_backend = _get_cpu_backend(framework)
    from ttk.utilities import get

    flat_dtypes = getattr(testcase, "flat_tensor_dtypes", None) or ()
    from .tf.param_extractor import mutable_flat_indexes

    # CPU 参照的设备作用域与 npu_device.open 解耦：golden 侧张量/Variable 显式
    # 落 CPU（CpuTfBackend 走 context.device，不受 as_default 的 ops.device
    # patch 影响；tf.Variable 无显式设备时随默认设备——NPU worker 里默认是
    # NPU，mutable 变量会被 placer 连带算子 colocate 到 NPU、撞内核 dtype 域）
    mutable_idx = mutable_flat_indexes(api_name) if not any(d > 0 for d in (dist or ())) else ()
    with cpu_backend.device_scope(0):
        cpu_inputs = []
        for idx, x in enumerate(raw_inputs):
            if x is None:
                cpu_inputs.append(None)
                continue
            # TF 量化 dtype 的 numpy 存储是普通整型，golden 侧同样须按声明 dtype 显式转换
            if hasattr(cpu_backend, "tf_device_type"):
                from ttk.utilities.dtypes import tf_quantized_dtype

                tf_dtype = tf_quantized_dtype(get(flat_dtypes, idx))
                if tf_dtype is not None:
                    import tensorflow as tf

                    cpu_inputs.append(tf.convert_to_tensor(x.copy(), dtype=tf_dtype))
                    continue
            cpu_inputs.append(cpu_backend.from_numpy(x.copy()))
        # mutable 槽（ParamInfo.is_mutable，张量序位次）与张量参数对位;
        # TensorList 会造成 flat 下标偏移, 此时不用启用
        # (mutable-ref 算子没有 TensorList 参数; dist 编码: 0=普通张量, >0=TensorList)。
        # 非 TF 前缀在 _api_info 内短路: torch 进程后加载 TF 的 C 扩展会段错误
        if mutable_idx:
            import tensorflow as tf

            for idx in mutable_idx:
                if idx < len(cpu_inputs) and cpu_inputs[idx] is not None:
                    cpu_inputs[idx] = tf.Variable(cpu_inputs[idx])
    nested = nest_by_distribution(cpu_inputs, dist) if dist else cpu_inputs

    if api_info is not None:
        args, kwargs, oidx = build_positional_args(
            api_name,
            nested,
            testcase.attributes or {},
            testcase.output_tensor_indexes,
            tensor_distribution=[d > 0 for d in dist] if dist else None,
            api_info=api_info,
        )
    else:
        plan = testcase.get_param_plan()
        args, kwargs, _ = plan.build_args(nested)
        oidx = plan.overload_index

    return _exec_and_convert(api_name, args, kwargs, oidx, cpu_backend)


def _call_plugin_with_plan(testcase, func, switches=None, backend="cpu", cpu_backend=None):
    """Call golden plugin using testcase's param plan — same arg order as profiling.
    Interface is identical to input plugin, except golden returns values.
    Uses testcase.tensors (CPU nested tensors) directly, like aclnn.
    """
    plan = testcase.get_param_plan()
    if plan is None:
        raise RuntimeError(f"No param plan for {testcase.api_name}, cannot call golden plugin")
    if cpu_backend is None:
        framework = detect_framework(testcase.api_name)
        cpu_backend = _get_cpu_backend(framework)
    use_numpy = cpu_backend.needs_numpy_fallback(testcase)

    args, kwargs, extra_attrs = plan.build_args(testcase.tensors)
    extra = {
        "backend": backend,
        "tensor_formats": testcase.tensor_formats,
        "tensor_dtypes": testcase.tensor_dtypes,
        "use_numpy": use_numpy,
        "short_soc_version": getattr(switches, "short_soc_version", None),
        "testcase_name": testcase.testcase_name,
    }
    extra.update(extra_attrs)

    extra.update(batch_relation_kwargs(testcase))

    params = sig_params(func)
    if accepts_var_keyword(params):
        kwargs.update(extra)
    else:
        kwargs.update({k: v for k, v in extra.items() if k in params})
    result = func(*args, **kwargs)
    return cpu_backend.result_to_numpy(result)


def _exec_and_convert(api_name, args, kwargs, overload_index=0, cpu_backend=None):
    """Execute API and convert result to list of numpy arrays."""
    from .execute import call_api

    resolved, is_tensor_method = resolve_api(api_name)
    if cpu_backend is None:
        framework = detect_framework(api_name)
        cpu_backend = _get_cpu_backend(framework)
    # tf.raw_ops.Resource* 的 DT_RESOURCE 位: tf.Variable 须显式传 .handle
    # (op 边界会自动解引用为值); args 按下标、kwargs 按名对位, 就地替换。
    # 无输出有状态算子的输出回读: 须在替换前快照变量(替换后 args/kwargs 里
    # 只剩 handle, Variable 引用丢失), 调用后读取更新值 —— 与设备侧回读同序
    from .tf.param_extractor import collect_mutable_variables, convert_variables_to_handles, read_mutable_values

    mutable_vars = collect_mutable_variables(api_name, args, kwargs)
    convert_variables_to_handles(api_name, args, kwargs)
    with cpu_backend.device_scope(0):
        if is_tensor_method:
            if args[0] is None:
                return [None]
            result = call_api(api_name, overload_index, getattr(args[0], resolved), args[1:], kwargs)
        else:
            result = call_api(api_name, overload_index, resolved, args, kwargs)
    nps = cpu_backend.result_to_numpy(result)
    if result is None and mutable_vars:
        nps = read_mutable_values(mutable_vars)
    return nps
