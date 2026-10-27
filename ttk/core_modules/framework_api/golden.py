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
E2E golden 生成。

- generate_golden_data：模式分发入口（worker 消费），按 XPU/NPU/Disable/Enable 分发
- generate_golden：四级回退——CSV golden_api 列 → e2e 插件 → 同 API 跑 NPU
  （exec_backend 就绪时）→ 同 API 跑 CPU
- _run_api：按 API 名字执行，CPU/NPU 由 backend 参数决定

设备由调用方经 exec_backend 指定（None = CPU；NPU 后端 = 上板），不读
golden_mode。

All paths use testcase.get_param_plan() to get consistent arg ordering
with profiling.py — same plan, same (*args, **kwargs) layout.
"""

import functools
import logging

from ttk.core_modules.common.testcase.batch import batch_relation_kwargs
from ttk.core_modules.framework_api.api_param import build_positional_args
from ttk.utilities import DTYPE_PROMOTE_MAP
from ttk.utilities.container import nest_by_distribution

from ..common.backend import detect_framework, resolve_api
from ..common.invoke import Invoke, accepts_var_keyword, golden_mode, sig_params
from ..common.plugin import SpecResolver, get_plugin_function
from ..common.process import process_context
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


def _reset_mutable_inputs(api_name, testcase, raw_inputs):
    """复位 mutable-ref 算子的输入变量：golden 与被测共用同一个 tf.Variable，
    不复位会在被测改写后的状态上计算。复位经 adapter 钩子实现（tf 有效，
    torch 为 no-op）。"""
    try:
        adapter_for_api_name(api_name).reset_mutable_inputs(
            testcase, getattr(testcase, "flat_tensors", None), raw_inputs
        )
    except Exception as _e:  # pragma: no cover
        logging.warning(f"[golden] mutable-ref reset skipped: {_e}")


def _cpu_fast_fail(testcase, api_target, cpu_backend):
    """CPU 执行前置检查：API 无 CPU kernel 或 dtype 不受支持时返回原因串，
    可执行返回 None。"""
    if cpu_backend.is_npu_only(api_target):
        return f"{api_target} is NPU-only (no CPU kernel)"
    if not testcase.is_dtype_support():
        return f"{api_target} skipped: non-native dtype detected"
    return None


def _golden_from_csv_api(testcase, golden_api, raw_inputs, exec_backend, dev_id):
    """按 CSV golden_api 列指定的 API 计算真值。

    exec_backend 为 NPU 后端时在 NPU 上执行（跳过 CPU 前置检查，NPU-only
    的 API 可用）；否则在 CPU 上执行。api_info：golden_api 等于被测 API 时
    复用本例信息，否则独立匹配 overload。
    """
    if golden_api.lower() == "disable":
        return ["SUPPRESSED"]
    api_info = testcase.get_api_info() if golden_api == testcase.api_name else FrameworkApiInfoKeeper().get(golden_api)
    if exec_backend is not None and exec_backend.is_npu():
        return _run_api(golden_api, raw_inputs, testcase, exec_backend, dev_id, api_info=api_info)
    cpu_backend = _get_cpu_backend(detect_framework(testcase.api_name))
    reason = _cpu_fast_fail(testcase, golden_api, cpu_backend)
    if reason:
        logging.warning(f"[golden] golden_api {reason}")
        return ["UNSUPPORTED"]
    return _run_api(golden_api, raw_inputs, testcase, cpu_backend, api_info=api_info)


def generate_golden(testcase, raw_inputs, plugin_path=None, switches=None, backend="cpu", exec_backend=None, dev_id=0):
    """
    Generate golden data using three-level fallback.

    Args:
        testcase: TestcaseE2e
        raw_inputs: list of numpy arrays (flat, one per tensor)
        plugin_path: custom plugin path from SWITCHES
        switches: SWITCHES object
        backend: backend name string (e.g. 'cpu', 'npu')，按字符串透传给插件
        exec_backend: 执行后端对象，设备由它决定：None = 全程 CPU；NPU 后端 =
            插件收 device 串、无插件时同 API 改跑 NPU
        dev_id: NPU 执行时的设备 id（可与被测不同卡）

    Returns:
        list of numpy arrays (golden outputs)
    """
    golden_api = getattr(testcase, "golden_api", None)
    api_name = testcase.api_name

    _reset_mutable_inputs(api_name, testcase, raw_inputs)
    cpu_backend = _get_cpu_backend(detect_framework(api_name))
    # exec_backend 决定设备：传入 NPU 后端则上板，否则全程 CPU
    run_on_npu = exec_backend is not None and exec_backend.is_npu()

    # Promote：抬高浮点输入精度。golden_api 与同 API 路径共用 raw_inputs，
    # 插件路径自行取数不受影响；Promote 与 NPU/XPU 模式互斥
    raw_inputs = _promote_raw_inputs(testcase, raw_inputs, switches)

    # --- Priority 1: CSV golden_api column ---
    if golden_api:
        return _golden_from_csv_api(testcase, golden_api, raw_inputs, exec_backend, dev_id)

    # --- Priority 2: Custom plugin via plugin_loader ---
    func = SpecResolver(api_name, "e2e", plugin_path).resolve_golden()
    if func is not None:
        # NPU 时插件收 device 串（输入由框架上板），CPU 时 device=None
        device = f"npu:{dev_id}" if run_on_npu else None
        return _call_plugin_with_plan(testcase, func, switches, backend, device=device)

    # --- Priority 2.5: run_on_npu → same API on NPU ---
    if run_on_npu:
        return _run_api(api_name, raw_inputs, testcase, exec_backend, dev_id)

    # --- Priority 3: Same API on CPU ---
    # 前置检查失败返回 UNSUPPORTED（比对按真值缺失落 FAIL，不执行整链）
    reason = _cpu_fast_fail(testcase, api_name, cpu_backend)
    if reason:
        logging.warning(f"[golden] {reason}; provide a custom plugin via --plugin or specify golden_api in CSV")
        return ["UNSUPPORTED"]
    try:
        return _run_api(api_name, raw_inputs, testcase, cpu_backend)
    except Exception as e:
        raise RuntimeError(
            f"{api_name} cannot run on CPU and has no e2e custom plugin. "
            f"Please provide a custom plugin via --plugin or specify golden_api in CSV."
        ) from e


def generate_golden_data(testcase, raw_inputs, switches, backend, dev_id=0, return_struct=None):
    """golden 生成入口（worker 消费）：按模式分发——XPU 占位（真值由三方
    回填）、NPU 设备执行（含性能采集，失败回落 CPU）、Disable 抑制、其余
    CPU 生成；产出后统一 dump_goldens。

    dev_id/return_struct 仅 NPU 分支使用。dump 由 dump_goldens → DataDumper
    统一门控（dump_golden 开关 + no-prof 豁免）。
    """
    from .compare import dump_goldens

    process_ctx = process_context()
    process_ctx.notify_status("OnGenGolden")
    if golden_mode(testcase, switches) == "XPU":
        # XPU：真值由三方执行产出，此处占位；三方采集后回填，缺失整例 FAIL
        golden_nps = ["GOLDEN_FAILURE"]
    elif golden_mode(testcase, switches) == "NPU":
        golden_nps = _generate_npu_golden_with_perf(testcase, raw_inputs, switches, backend, dev_id, return_struct)
    elif switches.golden_mode == "Disable":
        # override=Promote 叠加全局 Disable 时仍抑制
        golden_nps = ["SUPPRESSED"]
    else:
        # Invoke.golden 收敛占位与异常语义（含 golden_api="disable" → SUPPRESSED）
        golden_nps = Invoke.golden(
            testcase,
            functools.partial(
                generate_golden, testcase, raw_inputs, switches.plugin_path, switches, backend.device_type()
            ),
            switches,
        )
    process_ctx.notify_status("OnDumpGolden")
    dump_goldens(testcase, golden_nps, switches)
    return golden_nps


def _generate_npu_golden_with_perf(testcase, raw_inputs, switches, backend, dev_id, return_struct):
    """golden-mode NPU：golden 在设备上执行并采集性能。

    _form 传 exec_backend 时在 NPU 上执行（msprof 采集窗内，经 Invoke.golden），
    不传时在 CPU 上执行；NPU 执行或采集失败回落 CPU。性能经
    testcase.golden_perf/golden_source 回写 return_struct。无 NPU 设备时
    直接在 CPU 上执行。
    """
    from ..common.golden import has_npu_device

    def _form(exec_backend=None, dev_id=0):
        # exec_backend 决定执行设备（None = CPU）
        return generate_golden(
            testcase,
            raw_inputs,
            switches.plugin_path,
            switches,
            backend.device_type(),
            exec_backend=exec_backend,
            dev_id=dev_id,
        )

    def _cpu_placeholder():
        # CPU 执行；包一层 Invoke.golden 把异常转为占位串（无卡路径没有
        # 外层转换点）
        return Invoke.golden(testcase, functools.partial(_form), switches)

    if not has_npu_device():
        # 无 NPU 设备：直接 CPU 生成，不采性能
        logging.warning(
            "[%s] golden-mode NPU: no NPU device capability (torch_npu device detection "
            "failed); CPU golden used, device perf unavailable",
            testcase.testcase_name,
        )
        golden_nps = _cpu_placeholder()
        if return_struct is not None:
            return_struct.set_golden_perf(None, source="cpu")
        return golden_nps

    npu_form = functools.partial(_form, exec_backend=backend, dev_id=dev_id)
    golden_nps = Invoke.golden(
        testcase,
        npu_form,
        switches,
        invoke=lambda _func: _cpu_placeholder(),
        npu_invoke=npu_form,
        dev_id=dev_id,
    )
    if return_struct is not None:
        return_struct.set_golden_perf(getattr(testcase, "golden_perf", None), getattr(testcase, "golden_source", None))
    return golden_nps


def has_dynamic_golden_source(testcase, switches, manual_case=None):
    """Return whether a dynamic-slot case can avoid the CPU default Golden."""
    if switches.golden_mode == "Disable" or str(testcase.golden_api or "").lower() == "disable":
        return True
    if manual_case is not None and manual_case.has_goldens:
        return True
    # A non-disabled golden_api takes priority over plugins and receives the
    # unresolved None slot, so it is not a valid dynamic-slot Golden source.
    if testcase.golden_api:
        return False
    return get_plugin_function(testcase.api_name, "golden", "e2e", switches.plugin_path) is not None


def _run_api(api_name, raw_inputs, testcase, backend, dev_id=0, api_info=None):
    """按 API 名字执行并返回 numpy 列表；backend 决定 CPU/NPU 落点。

    - NPU：先 set_device(dev_id)（TF 侧会 open npu_device 并设为默认设备，
      此后 TF 张量落在 NPU；torch 幂等），torch 输入 to_device 拷贝上板，
      执行在 device_scope(dev_id) 内
    - CPU：device_scope(0) 显式落 CPU（TF 侧进程默认设备可能是 NPU，不
      钉住时 mutable 变量会被 placer 连带上板）
    - TF 量化 dtype 须显式转换；mutable 槽转 tf.Variable；TF 裸 eager 落
      CPU，NPU 执行经 adapter.wrap_eager 用 tf.function 触发 GE 分发
    - api_info 给定时按指定 API 的签名绑定参数（golden_api 路径），否则用
      本例 param plan（同 API 路径）；TensorList 嵌套以本例声明为准
    """
    from ttk.utilities import get

    from .tf.param_extractor import mutable_flat_indexes

    dist = testcase.tensor_list_dist
    on_npu = backend.is_npu()
    scope_dev = dev_id if on_npu else 0
    mutable_idx = mutable_flat_indexes(api_name) if not any(d > 0 for d in (dist or ())) else ()
    flat_dtypes = getattr(testcase, "flat_tensor_dtypes", None) or ()
    if on_npu:
        backend.set_device(dev_id)
    with backend.device_scope(scope_dev):
        inputs = []
        for idx, x in enumerate(raw_inputs):
            if x is None:
                inputs.append(None)
                continue
            if hasattr(backend, "tf_device_type"):
                # TF：量化 dtype 的 numpy 存储是普通整型，须按声明 dtype 显式
                # 转换；落点随 backend 的默认设备
                from ttk.utilities.dtypes import tf_quantized_dtype

                tf_dtype = tf_quantized_dtype(get(flat_dtypes, idx))
                if tf_dtype is not None:
                    import tensorflow as tf

                    inputs.append(tf.convert_to_tensor(x.copy(), dtype=tf_dtype))
                    continue
                inputs.append(backend.from_numpy(x.copy()))
                continue
            # torch：to_device 收 numpy 数组（内部 from_numpy + 上板）；
            # CPU 直接 from_numpy
            if on_npu:
                inputs.append(backend.to_device(x.copy(), dev_id))
            else:
                inputs.append(backend.from_numpy(x.copy()))
        # mutable 槽与张量参数按位对位；TensorList 会造成 flat 下标偏移，此时
        # 不启用（mutable-ref 算子没有 TensorList 参数；dist 编码：0=普通张量，
        # >0=TensorList）
        # 非 TF 前缀在 _api_info 内短路：torch 进程后加载 TF 的 C 扩展会段错误
        if mutable_idx:
            import tensorflow as tf

            for idx in mutable_idx:
                if idx < len(inputs) and inputs[idx] is not None:
                    inputs[idx] = tf.Variable(inputs[idx])
    nested = nest_by_distribution(inputs, dist) if dist else inputs

    if api_info is not None:
        # 按指定 API 的签名绑定参数
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

    return _exec_and_convert(api_name, args, kwargs, oidx, backend, dev_id=scope_dev)


def _call_plugin_with_plan(testcase, func, switches=None, backend="cpu", device=None):
    """Call golden plugin using testcase's param plan — same arg order as profiling.
    Interface is identical to input plugin, except golden returns values.
    Uses testcase.tensors (CPU nested tensors) directly.

    device 非 None 时（NPU）：args/kwargs 中 torch 张量 .to(device) 上板，
    extra 注入 device 字符串（声明 device 形参或 **kwargs 的插件才收到）；
    返回值经 result_to_numpy 回转。cpu_backend 按本例框架自取（模块级缓存）。"""
    plan = testcase.get_param_plan()
    if plan is None:
        raise RuntimeError(f"No param plan for {testcase.api_name}, cannot call golden plugin")
    cpu_backend = _get_cpu_backend(detect_framework(testcase.api_name))
    use_numpy = cpu_backend.needs_numpy_fallback(testcase)

    args, kwargs, extra_attrs = plan.build_args(testcase.tensors)
    if isinstance(func, type):
        return _call_class_golden(testcase, func, plan, switches, cpu_backend, device=device)
    if device is not None:
        import torch

        def _to_dev(v):
            if isinstance(v, torch.Tensor):
                return v.to(device)
            if isinstance(v, (list, tuple)):
                return type(v)(_to_dev(x) for x in v)
            return v

        args = [_to_dev(a) for a in args]
        kwargs = {k: _to_dev(v) for k, v in kwargs.items()}
    extra = {
        "backend": backend,
        "device": device or "cpu",
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


def _call_class_golden(testcase, func, plan, switches, cpu_backend, device=None):
    """class 形插件 golden：按名池绑定 __init__/__call__。

    池 = 张量参数名绑定值（plan.tensor_param_bindings，TensorList 参数为
    列表）+ testcase.attributes + device（class_accepts_device 声明才注入）。
    device 非 None 时池内 torch 张量 .to(device) 上板；返回值经
    result_to_numpy 回转。
    """
    from ttk.core_modules.common.invoke import class_accepts_device, instantiate_and_call

    args, kwargs, _extra = plan.build_args(testcase.tensors)
    pool = {}
    for name, entries in plan.tensor_param_bindings(testcase.tensors).items():
        vals = [kwargs[key] if container == "kwargs" else args[key] for _top_index, container, key, _vararg in entries]
        pool[name] = vals[0] if len(vals) == 1 else vals
    pool.update(testcase.attributes or {})
    if device is not None:
        import torch

        pool = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in pool.items()}
    if class_accepts_device(func):
        pool["device"] = device or "cpu"
    return cpu_backend.result_to_numpy(instantiate_and_call(func, pool))


def _exec_and_convert(api_name, args, kwargs, overload_index=0, cpu_backend=None, dev_id=0):
    """Execute API and convert result to list of numpy arrays.

    dev_id 仅 NPU 分支使用（CPU 恒 0）；backend 为 NPU 时经
    adapter.wrap_eager 包装（TF 裸 eager 落 CPU，GE 分发须 tf.function），
    调用在 backend.device_scope(dev_id) 内进行。
    """
    from .execute import call_api

    resolved, is_tensor_method = resolve_api(api_name)
    if cpu_backend is None:
        framework = detect_framework(api_name)
        cpu_backend = _get_cpu_backend(framework)
    if cpu_backend.is_npu():
        from .adapter import adapter_for_api_name

        resolved = adapter_for_api_name(api_name).wrap_eager(resolved, api_name, cpu_backend)
    # tf.raw_ops.Resource* 的 DT_RESOURCE 位: tf.Variable 须显式传 .handle
    # (op 边界会自动解引用为值); args 按下标、kwargs 按名对位, 就地替换。
    # 无输出有状态算子的输出回读: 须在替换前快照变量(替换后 args/kwargs 里
    # 只剩 handle, Variable 引用丢失), 调用后读取更新值 —— 与设备侧回读同序
    from .tf.param_extractor import collect_mutable_variables, convert_variables_to_handles, read_mutable_values

    mutable_vars = collect_mutable_variables(api_name, args, kwargs)
    convert_variables_to_handles(api_name, args, kwargs)
    with cpu_backend.device_scope(dev_id):
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
