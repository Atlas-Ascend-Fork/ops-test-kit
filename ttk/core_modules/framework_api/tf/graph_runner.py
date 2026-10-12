#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TF 图模式执行器（tf/ 域）。

via tf.function — TF's native graph compilation; npu_device handles NPU
dispatch internally, no separate compiler backend needed.
Corresponds to torch's graph_runner.py (torch.compile + torchair).
"""

import logging

from ttk.core_modules.common.testcase.batch import resolve_deterministic_level

from ...common.invoke import Invoke
from ..utils import compute_output_md5, finalize_det_status, prepare_device_args, resolve_npu_preprocess_hook

WARMUP_COUNT = 1


def _build_input_signature(testcase, dynamic):
    """Build tf.TensorSpec list from testcase tensor_view_shapes/dtypes.

    static (dynamic=False): fixed shapes → corresponds to -c/--const
    dynamic (dynamic=True):  None dimensions → corresponds to -d/--dynamic

    Returns (signature, sig_idx): signature is the TensorSpec list (or None),
    sig_idx lists the flat tensor index each signature entry binds to.
    Raises ValueError when a dtype cannot be mapped to tf.dtype — fail-fast,
    不静默降级为无签名自动 trace（否则静态模式语义变成动态、对外仍报 static）。
    """
    import tensorflow as tf

    from ttk.utilities.dtypes import str_to_tf_dtype

    from .param_extractor import mutable_flat_indexes

    sig = []
    sig_idx = []
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = testcase.flat_tensor_dtypes
    # const 来源的 0-D 位不进 signature：GE infershape 要求其以 Const 节点
    # 进入图，由 TfGraphWrapper 闭包（Python 标量）而非 Placeholder 提供
    const_indexes = getattr(testcase, "const_input_indexes", None) or set()
    # mutable-ref 位不进 signature：tf.function 按 TensorSpec 追踪会把
    # tf.Variable 降级为 SymbolicTensor，state_ops.*scatter 系列随即在
    # ref.handle/_lazy_read 上崩溃；ref 只能经闭包捕获保持 Variable 身份
    # (见 param_extractor.py 与 TfGraphWrapper._build_kw_function)
    dist = testcase.tensor_list_dist or ()
    mutable_idx = set(mutable_flat_indexes(testcase.api_name)) if not any(d > 0 for d in dist) else set()
    for idx, (shape, dtype_str) in enumerate(zip(flat_shapes, flat_dtypes)):
        if shape is None or idx in const_indexes or idx in mutable_idx:
            continue
        dims = list(shape) if not dynamic else [None] * len(shape)
        tf_dtype = str_to_tf_dtype(dtype_str)
        if tf_dtype is None:
            raise ValueError(f"Cannot map dtype {dtype_str} to tf.dtype for input_signature")
        sig.append(tf.TensorSpec(dims, tf_dtype))
        sig_idx.append(idx)
    return (sig if sig else None), sig_idx


def execute_tf_graph(
    testcase,
    backend,
    dev_id,
    switches,
    plan,
    resolved,
    is_tensor_method,
    is_inplace,
    raw_inputs,
    dynamic,
    is_aclgraph=False,
    deterministic_level=None,
):
    """Execute API in TF graph mode via tf.function with profiling.

    Args:
        testcase: TestcaseE2e
        backend: Backend instance (NpuTfBackend or CpuTfBackend)
        dev_id: device ID
        switches: SWITCHES
        plan: ParamPlan
        resolved: resolved API callable
        is_tensor_method: unused placeholder (always False for TF; kept for
            signature parity with torch's execute_graph so the caller can
            use a single graph_fn variable for both frameworks)
        is_inplace: unused placeholder (always False for TF; same reason)
        raw_inputs: numpy input arrays
        dynamic: True for dynamic shape graph, False for static shape graph

    Returns:
        (list of numpy arrays, ProfileResult, det_status) on success, or ([], None, None) on failure
    """
    if is_aclgraph:
        logging.warning("aclgraph mode not supported for TF, skipping")
        return [], None, None
    if tuple(getattr(testcase, "dyn_input_slot_indexes", ()) or ()):
        logging.error(f"TF graph mode does not support -1 dynamic slots: {testcase.testcase_name}")
        return [], None, None

    mode_str = "dynamic" if dynamic else "static"
    logging.info(f"Executing TF graph mode: {mode_str}")

    try:
        args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)
        if backend.is_npu():
            preprocess_result = Invoke.npu_preprocess(
                testcase,
                resolve_npu_preprocess_hook(testcase, switches),
                switches,
                device_args=(args, kwargs),
                device_scope=lambda: backend.device_scope(dev_id),
            )
            if preprocess_result is not None:
                raise RuntimeError(
                    "NPU_PREPROCESS_FAILURE: TF graph mode does not support npu_preprocess tensor materialization"
                )

        input_signature, sig_idx = _build_input_signature(testcase, dynamic)
        wrapper = TfGraphWrapper(
            resolved,
            input_signature=input_signature,
            api_name=testcase.api_name,
            call_args=args,
            call_kwargs=kwargs,
            sig_idx=sig_idx,
        )

        from .param_extractor import (
            collect_mutable_variables,
            is_ref_variable,
            read_mutable_values,
        )

        # mutable-ref 变量经闭包捕获，resource 句柄在 trace 时固化，无法像 eager
        # 路径那样在窗口外换新克隆、窗口内纯引用替换。改为两阶段(基准测试常规
        # 模式)：Phase 1 带复位跑正确性(窗口外)，Phase 2 纯图执行跑计时(窗口内
        # 无复位 op，采集天然干净，无需事后按 op 名剔除)。计时有效性依据：kernel
        # 耗时只与 shape/dtype 相关，与数据值无关，Phase 2 的状态累积不影响计时
        mutable_backups = [(v, v.read_value()) for v in (*args, *kwargs.values()) if is_ref_variable(v)]

        def reset_mutable_refs():
            for var, backup in mutable_backups:
                var.assign(backup)

        profiling_enabled = bool(getattr(switches, "task_profiling", True))

        def _collect_outputs(res, call_args, call_kwargs):
            # 无输出有状态算子(TF raw Resource*): 输出 = mutable 变量位的更新后值
            # (对齐 torch inplace 的"原地位即输出位"); 与 eager/golden 路径同序
            # (mutable 下标升序)。有返回值时不回读(snake_case 封装已含 lazy-read)
            nps = backend.result_to_numpy(res)
            if res is None:
                readbacks = read_mutable_values(collect_mutable_variables(testcase.api_name, call_args, call_kwargs))
                if readbacks:
                    nps = readbacks
            return nps

        if deterministic_level is None:
            deterministic_level = resolve_deterministic_level(switches, testcase)
        deterministic = deterministic_level > 0
        run_count = switches.run_time
        if switches.warmup and profiling_enabled:
            for _ in range(WARMUP_COUNT):
                reset_mutable_refs()
                wrapper(*args, **kwargs)
            backend.synchronize(dev_id)

        from ...common.profiler import ProfilerConfig, get_profiler

        profiler = get_profiler(
            testcase.api_name,
            backend,
            ProfilerConfig(
                testcase_name=testcase.testcase_name,
                root_path=switches.root_path,
                dev_id=dev_id,
                enabled=profiling_enabled,
            ),
        )
        md5_list = []
        result = None
        result_nps = None
        if mutable_backups:
            # Phase 1 正确性: 每轮复位回初值, 产出比对结果与确定性 MD5
            for _ in range(run_count):
                reset_mutable_refs()
                result = wrapper(*args, **kwargs)
                if deterministic:
                    backend.synchronize(dev_id)
                    md5_list.append(compute_output_md5(_collect_outputs(result, args, kwargs)))
            backend.synchronize(dev_id)
            # 比对结果定格于 Phase 1 末尾: 无输出有状态算子的回读取变量当前值,
            # Phase 2(窗口内不复位)会在其上继续累积, 事后回读必被污染
            result_nps = _collect_outputs(result, args, kwargs)
            # Phase 2 性能: 窗口内仅图执行
            if profiling_enabled and run_count:
                with profiler:
                    for _ in range(run_count):
                        wrapper(*args, **kwargs)
                    backend.synchronize(dev_id)
        else:
            with profiler:
                for _ in range(run_count):
                    result = wrapper(*args, **kwargs)
                    if deterministic:
                        backend.synchronize(dev_id)
                        run_nps = _collect_outputs(result, args, kwargs)
                        md5_list.append(compute_output_md5(run_nps))
                backend.synchronize(dev_id)

        perf = profiler.result(run_count)
        if result_nps is None:
            result_nps = _collect_outputs(result, args, kwargs)
        det_status = finalize_det_status(md5_list, testcase.testcase_name)
    except Exception as e:
        logging.error(f"TF graph {mode_str} execution failed: {e}", exc_info=True)
        return [], None, None

    del args, kwargs
    return result_nps, perf, det_status


# ── 图网络构造与执行 ────────────────────────────────────────────


class TfGraphWrapper:
    """Wrap a TF API callable in tf.function for graph-mode execution.

    For static shape (-c): input_signature with fixed TensorSpec shapes.
    For dynamic shape (-d): input_signature with None dimensions.

    tf.raw_ops.* ops require keyword args; the wrapper binds positional
    inputs to the API's tensor parameter names via inspect.signature,
    so tf.function tracing passes them as kwargs.
    """

    def __init__(
        self,
        api_func,
        input_signature=None,
        api_name=None,
        call_args=None,
        call_kwargs=None,
        sig_idx=None,
    ):
        import tensorflow as tf

        from .param_extractor import resource_param_names

        self._api_func = api_func
        self._api_name = api_name
        self._input_signature = input_signature
        self._param_names = self._extract_tensor_param_names(api_func, api_name)
        self._sig_param_names = None
        # tf.raw_ops.Resource* 的 DT_RESOURCE 位: trace 内 tf.Variable 须显式
        # 取 .handle(Variable 在 op 边界被解引用为值, resource 校验必挂)
        self._resource_names = set(resource_param_names(api_name)) if api_name else set()

        if self._param_names and input_signature is not None:
            self._tf_func, sig_param_names = self._build_kw_function(
                api_func,
                self._param_names,
                input_signature,
                call_args,
                call_kwargs,
                sig_idx,
                resource_names=self._resource_names,
            )
            self._sig_param_names = set(sig_param_names)
        elif input_signature is not None:
            self._tf_func = tf.function(api_func, input_signature=input_signature, autograph=False)
            self._tf_func.get_concrete_function()
        else:
            self._tf_func = tf.function(api_func, autograph=False)

    @staticmethod
    def _build_kw_function(
        api_func, param_names, input_signature, call_args=None, call_kwargs=None, sig_idx=None, resource_names=None
    ):
        """Build tf.function with explicit named params matching input_signature.

        tf.raw_ops.* require keyword args; we generate a wrapper with explicit
        parameter names (matching param_names) so input_signature binds correctly,
        and the wrapper forwards them as kwargs to the API. Params outside the
        signature are baked into the closure: Python scalars trace as Const
        nodes (what GE infershape requires), tf.Variable keeps its identity via
        resource capture (what state_ops mutable-ref APIs require — a Variable
        bound to a TensorSpec slot degrades to a SymbolicTensor and crashes on
        ref._lazy_read).

        sig_idx maps signature entry j to param_names[sig_idx[j]] (flat tensor
        index == tensor param position; mutable/const positions are excluded
        from the signature by the caller). Falls back to binding the first
        len(input_signature) params when absent.

        resource_names (tf.raw_ops.Resource* 的 DT_RESOURCE 位): call 时把
        tf.Variable 换成 .handle; 无输出 raw op 在 trace 下返回 Operation,
        tf.function 拒绝 — 降级为 None(eager 下本就返回 None)。
        """
        import tensorflow as tf

        call_args = list(call_args or [])
        if sig_idx and all(i < len(param_names) for i in sig_idx):
            tensor_names = [param_names[i] for i in sig_idx]
            sig_pos = set(sig_idx)
        else:
            tensor_names = param_names[: len(input_signature)]
            sig_pos = set(range(len(input_signature)))
        closure_values = {}
        for i, name in enumerate(param_names):
            if i in sig_pos:
                continue
            if i < len(call_args):
                closure_values[name] = call_args[i]
            elif call_kwargs and name in call_kwargs:
                closure_values[name] = call_kwargs[name]

        def wrapper(*args):
            kwargs = dict(zip(tensor_names, args))
            kwargs.update(closure_values)
            if resource_names:
                for name in resource_names:
                    v = kwargs.get(name)
                    if isinstance(v, tf.Variable):
                        kwargs[name] = v.handle
            r = api_func(**kwargs)
            return None if isinstance(r, tf.Operation) else r

        tf_func = tf.function(wrapper, input_signature=input_signature, autograph=False)
        tf_func.get_concrete_function()
        return tf_func, tensor_names

    @staticmethod
    def _extract_tensor_param_names(api_func, api_name):
        """Extract tensor parameter names from the API signature."""
        import inspect

        try:
            sig = inspect.signature(api_func)
            names = []
            for name, p in sig.parameters.items():
                if name == "name":
                    continue
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    names.append(name)
                elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                    break
            return names if names else None
        except (ValueError, TypeError):
            return None

    def __call__(self, *args, **kwargs):
        if self._param_names:
            call_kwargs = {}
            for i, name in enumerate(self._param_names):
                if i < len(args) and args[i] is not None:
                    call_kwargs[name] = args[i]
                elif name in kwargs and kwargs[name] is not None:
                    call_kwargs[name] = kwargs[name]
            # 有签名时只传签名位（其余位已烘进闭包）；无签名时全量透传
            if self._sig_param_names is not None:
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in self._sig_param_names}
            return self._tf_func(*call_kwargs.values())
        return self._tf_func(*args, **kwargs)
