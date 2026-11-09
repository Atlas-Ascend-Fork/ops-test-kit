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
torch 图模式执行器（torch/ 域）。

Wraps API in GraphNetwork + torch.compile for GE graph mode testing.
"""

import functools
import inspect
import logging
import os
import shlex
import subprocess
from contextlib import contextmanager, nullcontext
from typing import NamedTuple
from unittest.mock import patch

import torch

from ttk.core_modules.common.plugin import SpecResolver
from ttk.core_modules.common.testcase.batch import resolve_deterministic_level
from ttk.switches import get_global_storage

from ...common.invoke import Invoke
from ...common.profiler import ProfilerConfig, get_profiler
from ..utils import (
    WARMUP_COUNT,
    compute_output_md5,
    finalize_det_status,
    prepare_device_args,
    resolve_npu_preprocess_hook,
)

# debug_per_op_max_core_num（单算子满核验证）仅 aclgraph/npugraph_ex 的
# super_kernel_debug_options 支持；GE 图模式 scope options 走 op 编译白名单
# （CANN 9.2.0 不含该选项，传入会导致 Compilation_Error E40021）。
_ACLGRAPH_SUPER_KERNEL_OPTIONS = {
    "static_kernel_compile": True,
    "super_kernel_optimize": True,
    "super_kernel_debug_options": {"debug_per_op_max_core_num": 1},
    "clone_input": False,
}


@contextmanager
def _require_kernel_compile(enabled):
    """Track this worker's compiler calls, including failures followed by fallback.

    The backend catches CalledProcessError and may return valid binary outputs.
    Remember that failure independently so precision cannot turn it into a PASS.
    No compiler invocation is required: a valid compiled cache may be reused.
    """
    if not enabled:
        yield
        return

    original_run = subprocess.run
    failures = []

    def checked_run(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", ())
        parts = shlex.split(command) if isinstance(command, str) else command
        tracked = bool(parts) and os.path.basename(os.fsdecode(parts[0])) == "op_compiler"
        if not tracked:
            return original_run(*args, **kwargs)
        output_dir = parts[parts.index("-o") + 1] if "-o" in parts else "(not specified)"
        try:
            result = original_run(*args, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            failures.append(f"{exc}; compile output: {output_dir}")
            raise
        if result.returncode:
            failures.append(f"op_compiler exit code {result.returncode}; compile output: {output_dir}")
        return result

    # Each TTK graph worker executes one case at a time. Restore even on errors.
    with patch.object(subprocess, "run", checked_run):
        yield
    if failures:
        raise RuntimeError("KERNEL_COMPILE_FAILURE: " + "; ".join(failures))


def _build_torchair_config(binary):
    """torchair CompilerConfig（-c/-d GE 图模式）。

    binary=True 时经 as_dict 注入 ge.jit_compile=0（GE 图编译复用预编译 kernel）——
    当前 torchair CompilerConfig 未暴露该字段，注入走 global options 通道（与
    --core-limit 的 ge.aicoreNum 同路）。"""
    from torchair.configs.compiler_config import CompilerConfig

    if not binary:
        return CompilerConfig()

    class _BinaryCompilerConfig(CompilerConfig):
        def as_dict(self):
            local_options, global_options = super().as_dict()
            global_options["ge.jit_compile"] = "0"
            return local_options, global_options

    return _BinaryCompilerConfig()


@functools.lru_cache(maxsize=1)
def _get_npu_backend():
    import torchair

    config = _build_torchair_config(getattr(get_global_storage(), "e2e_binary", False))
    core_limit = get_global_storage().core_limit
    if core_limit and isinstance(core_limit, tuple) and any(core_limit):
        ai_limit, vec_limit = core_limit
        props = torch.npu.get_device_properties(torch.npu.current_device())
        ai_num = ai_limit if ai_limit is not None else props.cube_core_num
        vec_num = vec_limit if vec_limit is not None else props.vector_core_num
        config.ge_config.aicore_num = f"{ai_num}|{vec_num}"
    return torchair.get_npu_backend(compiler_config=config)


@functools.lru_cache(maxsize=1)
def _get_npu_backend_aclgraph():
    """获取 aclgraph 模式的 NPU backend。"""
    npu_backend = "npugraph_ex"
    return npu_backend


def _get_aclgraph_scope():
    """Load the graph scope API lazily so CPU/GE users need no torch_npu import."""
    try:
        from torch_npu.npu.npugraph_ex import scope
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "--core-limit with --aclgraph requires torch_npu.npu.npugraph_ex.scope.limit_core_num"
        ) from exc
    return scope


def _resolve_aclgraph_core_limit(core_limit, dev_id):
    """Fill an omitted side of --core-limit with that device's physical count."""
    if not core_limit or not isinstance(core_limit, tuple) or not any(core_limit):
        return None
    ai_limit, vec_limit = core_limit
    props = torch.npu.get_device_properties(dev_id)
    return (
        ai_limit if ai_limit is not None else props.cube_core_num,
        vec_limit if vec_limit is not None else props.vector_core_num,
    )


def _aclgraph_core_limit_context(core_limit, dev_id):
    """Limit eager companion calls (for example metadata) on the active stream."""
    resolved = _resolve_aclgraph_core_limit(core_limit, dev_id)
    if resolved is None:
        return nullcontext()
    ai_limit, vec_limit = resolved
    return _get_aclgraph_scope().limit_core_num(ai_limit, vec_limit)


def _compile_model(model, backend, dynamic, fullgraph):
    """Compile model with torch.compile. Returns compiled callable or raises."""
    compiled = torch.compile(
        model,
        fullgraph=fullgraph,
        backend=backend,
        dynamic=dynamic,
    )
    return compiled


def _compile_model_aclgraph(model, backend, fullgraph, super_kernel=False, online_compile=True):
    """以 aclgraph 模式编译模型"""
    options = _ACLGRAPH_SUPER_KERNEL_OPTIONS if super_kernel else {"static_kernel_compile": online_compile}
    compiled = torch.compile(
        model,
        fullgraph=fullgraph,
        backend=backend,
        dynamic=False,
        options=options,
    )
    return compiled


class _SuperKernelScopeModel(torch.nn.Module):
    """Mark the target forward with the scope understood by its graph backend."""

    def __init__(self, model, scope_name, is_aclgraph=False):
        super().__init__()
        self._model = model
        self._scope_name = scope_name
        self._is_aclgraph = is_aclgraph
        if is_aclgraph and not all(
            callable(getattr(torch.npu, name, None)) for name in ("super_kernel_scope_begin", "super_kernel_scope_end")
        ):
            raise RuntimeError("--super-kernel with --aclgraph requires torch_npu SuperKernel scope APIs")
        if not is_aclgraph:
            import torchair

            # Resolve the lazy module outside Dynamo's traced forward.
            self._super_kernel_scope = torchair.scope.super_kernel

    def forward(self, *args, **kwargs):
        if self._is_aclgraph:
            torch.npu.super_kernel_scope_begin(self._scope_name)
            result = self._model(*args, **kwargs)
            torch.npu.super_kernel_scope_end(self._scope_name)
            return result
        with self._super_kernel_scope(self._scope_name, ""):
            return self._model(*args, **kwargs)


class _AclGraphCoreLimitModel(torch.nn.Module):
    """Encode an explicit core limit in every ACLGraph capture."""

    def __init__(self, model, core_limit, dev_id):
        super().__init__()
        self._model = model
        resolved = _resolve_aclgraph_core_limit(core_limit, dev_id)
        if resolved is None:
            raise ValueError("ACLGraph core-limit wrapper requires a non-empty core limit")
        self._ai_limit, self._vec_limit = resolved
        # Resolve this outside Dynamo tracing; the scope call itself is a graph marker.
        self._limit_core_num = _get_aclgraph_scope().limit_core_num

    def forward(self, *args, **kwargs):
        with self._limit_core_num(self._ai_limit, self._vec_limit):
            return self._model(*args, **kwargs)


def _restore_after_first_call(args, kwargs, is_inplace, self_mutable, inplace_backups, mutable_kwargs):
    """首调改写后按 mutable 位复位输入（by_kwargs 走 kwargs 名，否则 args 位）。"""
    if mutable_kwargs is not None:
        for idx, key in mutable_kwargs.items():
            if key in kwargs and kwargs[key] is not None:
                kwargs[key][:] = inplace_backups[idx]
        return
    if is_inplace and self_mutable is not None and args:
        args[0][:] = self_mutable
    for idx, bak in (inplace_backups or {}).items():
        if idx < len(args) and args[idx] is not None:
            args[idx][:] = bak


def _prepare_mutable_rounds(
    args, kwargs, backend, is_inplace, self_mutable, inplace_backups, mutable_kwargs, clone_count
):
    """mutable 位逐轮预置：original_tensors（末轮原始张量）与 inplace_clones（轮克隆）。

    aclgraph 不用克隆（clone_count=0，循环顶 copy_ 复位）。
    """
    original_tensors = {}
    inplace_clones = {}
    if mutable_kwargs is not None:
        for idx, key in mutable_kwargs.items():
            if key in kwargs and kwargs[key] is not None:
                original_tensors[idx] = (key, kwargs[key])
                inplace_clones[idx] = (key, [backend.clone(kwargs[key]) for _ in range(clone_count)])
        return original_tensors, inplace_clones
    for idx in inplace_backups or {}:
        if idx < len(args) and args[idx] is not None:
            original_tensors[idx] = args[idx]
            inplace_clones[idx] = [backend.clone(args[idx]) for _ in range(clone_count)]
    if is_inplace and self_mutable is not None and 0 not in original_tensors and args and args[0] is not None:
        original_tensors[0] = args[0]
        inplace_clones[0] = [backend.clone(args[0]) for _ in range(clone_count)]
    return original_tensors, inplace_clones


def _round_md5(args, kwargs, backend, r, inplace_backups, mutable_kwargs):
    """确定性轮的输出 MD5（含 mutable 位回读）。"""
    run_nps = []
    if r is not None:
        run_nps.extend(backend.result_to_numpy(r))
    if mutable_kwargs is not None:
        for key in mutable_kwargs.values():
            if kwargs.get(key) is not None:
                run_nps.append(backend.to_numpy(kwargs[key], safe=True))
    else:
        for idx in sorted(inplace_backups or ()):
            if idx < len(args) and args[idx] is not None:
                run_nps.append(backend.to_numpy(args[idx], safe=True))
    return compute_output_md5(run_nps)


def _run_compiled(
    compiled,
    args,
    kwargs,
    backend,
    dev_id,
    switches,
    is_inplace,
    self_mutable,
    api_name,
    testcase_name="",
    inplace_backups=None,
    mutable_kwargs=None,
    deterministic_level=None,
    is_aclgraph=False,
    round_reset=None,
):
    """执行编译图：首调定精度 → warmup + 逐轮计时 → 末轮结果与确定性 MD5。

    mutable 位三来源：self_mutable=通用 inplace 路径 args[0] 的克隆；
    inplace_backups=CSV 声明位的 {下标: 克隆}；mutable_kwargs=custom 模块的
    {下标: kwargs 参数名}。round_reset=声明式对象参数的轮前复位（generator
    spec，见 object_spec.py），每次 compiled() 调用前触发。
    """
    profiling_enabled = bool(getattr(switches, "task_profiling", True))
    deterministic = bool(deterministic_level)
    run_count = switches.run_time
    by_kwargs = mutable_kwargs is not None

    if round_reset is not None:
        round_reset()
    result = compiled(*args, **kwargs)
    backend.synchronize(dev_id)
    result_nps = backend.result_to_numpy(result, copy=is_inplace)
    _restore_after_first_call(args, kwargs, is_inplace, self_mutable, inplace_backups, mutable_kwargs)

    # 图模式 warmup 与 eager 对齐：在 profiler 窗口内、schedule 的 warmup
    # step 中执行（按 step 语义丢弃）。torchair 图首次 kernel 事件与
    # step-tag 建立存在竞态（首行 StepId=''），由本就要丢弃的 warmup
    # 调用吸收，计时轮全部落在 active step 且归因完整。
    warmup_in_window = switches.warmup and profiling_enabled
    # 非 aclgraph 的可变位逐轮克隆多备一份给 warmup（末轮用原始张量）；
    # aclgraph 走循环顶 copy_ 复位，无逐轮克隆。
    clone_count = 0 if is_aclgraph else run_count - 1 + (1 if warmup_in_window else 0)
    original_tensors, inplace_clones = _prepare_mutable_rounds(
        args, kwargs, backend, is_inplace, self_mutable, inplace_backups, mutable_kwargs, clone_count
    )

    profiler = get_profiler(
        api_name,
        backend,
        ProfilerConfig(
            testcase_name=testcase_name,
            root_path=switches.root_path,
            dev_id=dev_id,
            enabled=profiling_enabled,
            aclgraph=is_aclgraph,
            warmup_count=WARMUP_COUNT if warmup_in_window else 0,
            graph=True,
        ),
    )
    md5_list = []
    with profiler:
        if warmup_in_window:
            if round_reset is not None:
                round_reset()
            if is_aclgraph:
                compiled(*args, **kwargs)
            elif by_kwargs:
                for _idx, (key, clones) in inplace_clones.items():
                    if clones:
                        kwargs[key] = clones[-1]
                compiled(*args, **kwargs)
            else:
                for _idx, clones in inplace_clones.items():
                    if clones:
                        args[_idx] = clones[-1]
                compiled(*args, **kwargs)
            backend.synchronize(dev_id)
            profiler.step()
        for i in range(run_count):
            if round_reset is not None:
                round_reset()
            if is_aclgraph:
                # Mutated-input address changes force recapture inside the measured loop.
                for idx, original in original_tensors.items():
                    target = original[1] if by_kwargs else original
                    backup = (inplace_backups or {}).get(idx, self_mutable)
                    target.copy_(backup)
            elif i < run_count - 1:
                if by_kwargs:
                    for _idx, (key, clones) in inplace_clones.items():
                        kwargs[key] = clones[i]
                else:
                    for idx, clones in inplace_clones.items():
                        args[idx] = clones[i]
            elif by_kwargs:
                for _idx, (key, orig) in original_tensors.items():
                    kwargs[key] = orig
            else:
                for idx, orig in original_tensors.items():
                    args[idx] = orig
            r = compiled(*args, **kwargs)
            if not is_inplace:
                result = r
            backend.synchronize(dev_id)
            if deterministic:
                md5_list.append(_round_md5(args, kwargs, backend, r, inplace_backups, mutable_kwargs))
        backend.synchronize(dev_id)

    perf = profiler.result(max(run_count, 1))

    if not is_inplace:
        result_nps = backend.result_to_numpy(result, copy=is_inplace)

    det_status = finalize_det_status(md5_list, testcase_name)

    return result_nps, perf, det_status


def _resolve_generator_round_reset(args, kwargs, backend, dev_id, testcase, is_aclgraph):
    """generator spec 的图模式处理，返回默认 generator 轮前复位句柄。

    Dynamo 无法 proxy Generator（断图），图腿摘除显式 generator 回落默认
    generator、逐 replay 重播种（见 object_spec.py）；aclgraph 捕获期禁读
    RNG 状态（随机算子本身不支持），快速失败（worker 层收敛为
    GRAPH_EXEC_FAILURE）。
    """
    from ..object_spec import make_default_generator_reset, strip_specs

    gen_specs = strip_specs(args, kwargs)
    if not gen_specs:
        return None
    if is_aclgraph:
        raise RuntimeError(
            "GENERATOR_GRAPH_UNSUPPORTED: explicit generator spec is not "
            "supported in aclgraph mode; use the eager leg or -c/-d"
        )
    if len(gen_specs) > 1:
        logging.warning(
            "[%s] multiple generator specs in graph mode; reseeding with the first",
            testcase.testcase_name,
        )
    return make_default_generator_reset(gen_specs[0], backend, dev_id)


def _snapshot_inplace_backups(args, backend, inplace_input_indexes):
    """CSV 声明 mutable 位的克隆快照 {下标: 克隆}。"""
    backups = {}
    for idx in sorted(inplace_input_indexes or ()):
        if idx < len(args) and args[idx] is not None:
            backups[idx] = backend.clone(args[idx])
    return backups


class _GraphModel(NamedTuple):
    """图模型构建产物（execute_graph 主流程消费）。"""

    model: object
    fwd_args: list
    fwd_kwargs: dict
    is_custom: bool
    self_mutable: object
    mutable_kwargs: dict


def _build_graph_model(testcase, backend, switches, plan, resolved, args, kwargs, is_tensor_method, is_inplace):
    """custom torch_graph 模块（TestSpec 注册）优先，缺省回落通用 GraphNetwork。"""
    inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
    custom_cls = SpecResolver(testcase.api_name, "e2e", switches.plugin_path).resolve_torch_graph()
    if custom_cls:
        logging.info(f"Using custom graph module: {custom_cls.__name__}")
        init_kwargs, fwd_kwargs = _split_params(custom_cls, plan.overload_params, args, kwargs)
        mutable_kwargs = None
        if inplace_input_indexes:
            positional_param_names = [p.name for p in plan.overload_params if not p.is_keyword_only]
            mutable_kwargs = {}
            for idx in inplace_input_indexes:
                if idx < len(positional_param_names) and positional_param_names[idx] in fwd_kwargs:
                    mutable_kwargs[idx] = positional_param_names[idx]
        return _GraphModel(custom_cls(**init_kwargs), [], fwd_kwargs, True, None, mutable_kwargs)

    if is_tensor_method:

        def api_caller(*call_args, **call_kwargs):
            return getattr(call_args[0], resolved)(*call_args[1:], **call_kwargs)

    else:
        api_caller = resolved
    self_mutable = backend.clone(args[0]) if is_inplace and args and args[0] is not None else None
    return _GraphModel(GraphNetwork(api_caller), args, kwargs, False, self_mutable, None)


def _compile_graph_model(model, npu_backend, switches, dynamic, is_aclgraph, mode_str, dev_id, testcase):
    """SuperKernel / ACLGraph core-limit 包装后编译。"""
    if switches.super_kernel_enabled:
        logging.info("SuperKernel enabled: mode=%s scope=%s", mode_str, testcase.api_name)
        model = _SuperKernelScopeModel(model, testcase.api_name, is_aclgraph=is_aclgraph)
    if is_aclgraph and switches.core_limit:
        logging.info(
            "ACLGraph core limit enabled: aicore=%s vectorcore=%s",
            switches.core_limit[0] or "physical",
            switches.core_limit[1] or "physical",
        )
        model = _AclGraphCoreLimitModel(model, switches.core_limit, dev_id)
    use_fullgraph = bool(switches.fullgraph)
    if is_aclgraph:
        return _compile_model_aclgraph(
            model,
            npu_backend,
            use_fullgraph,
            switches.super_kernel_enabled,
            not getattr(switches, "e2e_binary", False),
        )
    return _compile_model(model, npu_backend, dynamic, use_fullgraph)


def execute_graph(
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
    """
    Execute API in GE graph mode via torch.compile with profiling.

    Args:
        testcase: TestcaseE2e
        backend: Backend instance
        dev_id: device ID
        switches: SWITCHES
        plan: ParamPlan
        resolved: resolved API callable (or method name string for tensor methods)
        is_tensor_method: True if API is a Tensor method
        is_inplace: True if API modifies input in-place
        raw_inputs: numpy input arrays
        dynamic: True for dynamic shape graph, False for static shape graph

    Returns:
        (nps, ProfileResult, det_status, err_msg)：执行失败时 nps=[]、
        err_msg 带异常描述；成功或跳过时 err_msg 为 None。
    """
    if not backend.is_npu():
        logging.warning("Graph mode only supports NPU backend, skipping")
        return [], None, None, None

    if deterministic_level is None:
        deterministic_level = resolve_deterministic_level(switches, testcase)

    import torch_npu

    torch_npu.npu.set_device(dev_id)

    mode_str = "aclgraph" if is_aclgraph else ("dynamic" if dynamic else "static")
    logging.info(f"Executing graph mode: {mode_str}")

    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)
    round_reset = _resolve_generator_round_reset(args, kwargs, backend, dev_id, testcase, is_aclgraph)
    # Companion operators such as metadata run eagerly before ACLGraph capture.
    with _aclgraph_core_limit_context(switches.core_limit if is_aclgraph else None, dev_id):
        result = Invoke.npu_preprocess(
            testcase,
            resolve_npu_preprocess_hook(testcase, switches),
            switches,
            device_args=(args, kwargs),
            device_scope=lambda: backend.device_scope(dev_id),
        )
    testcase.absorb_preprocess(result, plan, args, kwargs)

    inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
    inplace_backups = _snapshot_inplace_backups(args, backend, inplace_input_indexes)
    bundle = _build_graph_model(testcase, backend, switches, plan, resolved, args, kwargs, is_tensor_method, is_inplace)

    try:
        npu_backend = _get_npu_backend_aclgraph() if is_aclgraph else _get_npu_backend()
    except Exception as e:
        logging.error(f"Failed to get TorchAir NPU backend: {e}")
        return [], None, None, f"Failed to get TorchAir NPU backend: {e}"

    try:
        with _require_kernel_compile(True):
            compiled = _compile_graph_model(
                bundle.model, npu_backend, switches, dynamic, is_aclgraph, mode_str, dev_id, testcase
            )
            result_nps, perf, det_status = _run_compiled(
                compiled,
                bundle.fwd_args,
                bundle.fwd_kwargs,
                backend,
                dev_id,
                switches,
                is_inplace,
                bundle.self_mutable if is_inplace else None,
                testcase.api_name,
                testcase_name=testcase.testcase_name,
                inplace_backups=inplace_backups if inplace_input_indexes else None,
                mutable_kwargs=bundle.mutable_kwargs,
                deterministic_level=deterministic_level,
                is_aclgraph=is_aclgraph,
                round_reset=round_reset,
            )

    except Exception as e:
        logging.error(f"Graph {mode_str} execution failed: {e}", exc_info=True)
        return [], None, None, f"Graph {mode_str} execution failed: {e}"

    if result_nps and inplace_input_indexes and not bundle.is_custom:
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                result_nps.append(backend.to_numpy(args[idx].detach().clone()))

    return result_nps, perf, det_status, None


# ── 图网络构造与执行 ────────────────────────────────────────────


class GraphNetwork(torch.nn.Module):
    """
    Generic torch.nn.Module wrapper for torch.compile graph mode testing.

    api_func is always a callable — either the resolved API function directly,
    or a closure that handles tensor method dispatch.
    """

    def __init__(self, api_func):
        super().__init__()
        self.api_func = api_func

    def forward(self, *args, **kwargs):
        return self.api_func(*args, **kwargs)


def _split_params(module_cls, overload_params, args, kwargs):
    """Split args/kwargs into __init__ and forward kwargs based on module signatures.

    Uses overload_params (ParamInfo list) to map positional args to parameter names,
    then inspects module_cls.__init__ and .forward to split them.

    For tensor methods, overload_params[0] is "self" (the input tensor).
    It is mapped to the first forward parameter automatically.

    Args:
        module_cls: torch.nn.Module subclass
        overload_params: list of ParamInfo from plan.overload_params (positional order)
        args: positional args from prepare_device_args
        kwargs: keyword args from prepare_device_args

    Returns:
        (init_kwargs, fwd_kwargs) — dicts for __init__ and forward respectively
    """
    full_kwargs = dict(kwargs)
    positional_param_names = [p.name for p in overload_params if not p.is_keyword_only]
    for i, val in enumerate(args):
        if i < len(positional_param_names):
            full_kwargs[positional_param_names[i]] = val

    init_params = set(inspect.signature(module_cls.__init__).parameters.keys()) - {"self"}
    fwd_params = set(inspect.signature(module_cls.forward).parameters.keys()) - {"self"}

    if "self" in full_kwargs:
        fwd_ordered = [
            p.name
            for p in inspect.signature(module_cls.forward).parameters.values()
            if p.name != "self" and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        for name in fwd_ordered:
            if name not in init_params and name not in full_kwargs:
                full_kwargs[name] = full_kwargs.pop("self")
                break

    fwd_has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(module_cls.forward).parameters.values()
    )

    init_kwargs = {k: v for k, v in full_kwargs.items() if k in init_params}
    if fwd_has_var_kw:
        fwd_kwargs = {k: v for k, v in full_kwargs.items() if k not in init_params}
    else:
        fwd_kwargs = {k: v for k, v in full_kwargs.items() if k in fwd_params}

    return init_kwargs, fwd_kwargs
