#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""executor — runs ONE /v1/run inside a fresh child process (isolation).

Module-level so it is picklable for multiprocessing. execute_request returns an
envelope dict and never raises for 400/424/500; only a hard crash (segfault/OOM)
kills the child, which the parent sees as a nonzero exitcode -> 500.

整仓部署：契约代码（bind_params/UnknownParamError）与 XPU profiler 共享引擎
经 ttk.core_modules.common.* 同源消费（单份实现，client/server 不漂移）；
三方依赖仅 numpy + 框架（torch/tensorflow 懒加载）。
"""

import importlib
import inspect
import logging
import os
import sys
from collections.abc import Callable as _Callable
from functools import partial

import numpy as np

# bind_params/UnknownParamError 与 client 同源（ttk.core_modules.common.xpu 单份）；
# has_data/has_perf/format_device/match_params_v1/resolve_callable 留驻本文件（server 专属语义）
from ttk.core_modules.common.xpu import UnknownParamError, bind_params  # noqa: F401
from ttk.utilities.dtypes import is_4bit_dtype, resolve_numpy_dtype

_FRAMEWORK_RESERVED = {"device"}
_PROVIDER_MODULE_ALIASES = {"tf": "tensorflow"}

# npz 无法承载的自定义 dtype：wire 只传原始字节，元数据由 X-Input-Schema 声明。
# 与 dispatcher._UINT8_WIRE_DTYPES 同口径。
_UINT8_WIRE_DTYPES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0",
    "hifloat8",
    "int4",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat4",
)
# float8 家族中 torch 有原生存储的类型（其余 numpy 直通）。
_TORCH_FLOAT8_DTYPES = ("float8_e4m3fn", "float8_e5m2")

DATA = 0b01  # return device outputs
PERF = 0b10  # collect performance


# ── server 专属符号 ──


def has_data(mode: int) -> bool:
    return bool(mode & DATA)


def has_perf(mode: int) -> bool:
    return bool(mode & PERF)


def format_device(provider, profile, device_id):
    """Build the framework device spec. cpu short-circuits
    before reading profile; otherwise the lib/type comes from profile."""
    if str(device_id) == "cpu":
        return "cpu"
    if provider == "torch":
        return f"{profile['torch_lib']}:{device_id}"
    if provider == "tf":
        tf_type = profile.get("tf_device_type")
        if tf_type is None:
            raise ValueError("hardware has no tf_device_type, tf provider unavailable")
        return f"/device:{tf_type}:{device_id}"
    raise ValueError(f"unknown provider: {provider}")


def resolve_callable(api_str: str):
    """Resolve a dotted ``module.attr.sub`` string to a callable (api mode).

    Rejects classes: api mode is defined for stateless functions;
    a class (e.g. ``torch.nn.Softmax``) is a misuse -> ValueError.

    Raises:
        ValueError: if the string has no dot, or resolves to a class.
        ImportError/AttributeError: if the module/attr does not exist
            (propagated; the executor maps these to the appropriate status).
    """
    parts = api_str.split(".")
    if len(parts) < 2:
        raise ValueError(f"api '{api_str}' must be a dotted 'module.attr' path")
    # Map short names (e.g. "tf") to real package names (e.g. "tensorflow")
    if parts[0] in _PROVIDER_MODULE_ALIASES:
        parts[0] = _PROVIDER_MODULE_ALIASES[parts[0]]
    obj = importlib.import_module(parts[0])
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    if inspect.isclass(obj):
        raise ValueError(f"api mode requires a function, got a class: {api_str}")
    return obj


def match_params_v1(schema: list, flat_arrays: list) -> dict:
    """Restore named inputs from X-Input-Schema and flat arrays.

    Args:
        schema: List of {"name": str, "index"|"indices"|None}
        flat_arrays: Flat list of numpy arrays (no None gaps)

    Returns:
        Dict mapping name to value (single ndarray, list of ndarrays, or None)
    """
    result = {}
    for entry in schema:
        name = entry["name"]
        if "indices" in entry:
            result[name] = [flat_arrays[i] for i in entry["indices"]]
        elif "index" in entry:
            idx = entry["index"]
            result[name] = flat_arrays[idx] if idx is not None else None
    return result


class _MissingSpecDependency(ImportError):
    """Spec-module import failed -> 424 (syncable), not 500."""


# Module-level cache for the torch backend extension module (the object
# returned by getattr(torch, profile["torch_lib"])). executor runs in a
# per-request throwaway child process, so a module-global is safe (one request
# == one process lifetime). Set ONCE in execute_request after the backend
# import (Step 4a); read by device_available and _run_perf pass-2 instead of
# repeating `getattr(torch, profile["torch_lib"])`.
_TORCH_DEV_MODULE = None

# third_party key aliases — mirrors client-side aliases (client.py) so spec
# writers can use "tensorflow"/"np"/"npu"; duplicated here (separate process).
# Keep in sync with the client copy.
_TP_ALIASES = {"tensorflow": "tf", "np": "numpy", "npu": "torch"}


def _api_from_kwargs(kwargs):
    """从请求 kwargs 取 client 携带的 api 标识（X-API 或 spec_class）。

    用于 server-side 错误路径回传 X-API：成功路径用
    executor 推导的 resolved api（api_label），错误路径只能回传 client 发来的原始值。
    """
    return kwargs.get("api") or kwargs.get("spec_class")


def _ok(output_path, count, shapes, dtypes, perf, api=None, schema=None):
    return {
        "ok": True,
        "http_status": 200,
        "output_path": output_path,
        "output_count": count,
        "shapes": shapes,
        "dtypes": dtypes,
        "perf": perf,
        "missing": None,
        "error": None,
        "api": api,
        "schema": schema,
    }


def _err(status, error, missing=None, api=None):
    return {
        "ok": False,
        "http_status": status,
        "output_path": None,
        "output_count": 0,
        "shapes": [],
        "dtypes": [],
        "perf": None,
        "missing": missing,
        "error": error,
        "api": api,
    }


def _client_error(e) -> str:
    """Client-facing error string for a 500: exception TYPE + MESSAGE only.

    Security (OWASP Improper Error Handling / OTG-ERR-002): the full traceback
    carries server FS paths, line numbers and source lines, so it is logged
    server-side only (logging.exception) and NEVER put in the wire envelope.
    The exception message can still embed a path (residual risk, accepted). To
    tighten the contract later (generic message, add error_id, ...), change ONLY
    this function — both 500 exit points (execute_request, child_main) route
    through it, so it is the single control point.
    """
    return f"{type(e).__name__}: {e}"


def _resolve_attr_path(module, dotted):
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


# --- KERNEL api resolve: 算子名（snake/Camel 双形式）→ framework callable ---
# Special-case dict literals reference torch/tf symbols, so they are built
# inside each resolver where the lazy import has run. Proven mappings —
# do not paraphrase or trim entries.
# aclnn-prefixed names take the ACLNN branch; this section covers the
# torch/tf dual-form KERNEL path only.


def _torch_resolve(name):
    """Resolve a snake-form op name to a torch callable.

    Only invoked when provider == 'torch' (gated by _resolve_3party_api's
    dispatch); a TF request never reaches this import.
    """
    import torch

    # === special-case mapping (proven — do not trim entries) ===
    torch_funcs = {
        "kl_div": torch.nn.functional.kl_div,
        "ctc_loss_v2": torch.nn.functional.ctc_loss,
        "equal": torch.eq,
        "gelu": torch.nn.GELU(),
        "floor_mod": torch.fmod,
        "floor_div": partial(torch.div, rounding_mode="floor"),
        "pows": torch.pow,
        "real_div": torch.div,
        "top_k_v2": torch.topk,
        "select": torch.where,
        "select_v2": torch.where,
        "is_inf": torch.isinf,
        "is_nan": torch.isnan,
        "is_close": torch.isclose,
        "is_finite": torch.isfinite,
    }
    if name in torch_funcs:
        return torch_funcs[name]
    # 候选源含 torch.nn.functional:服务端是【按参数名绑参】的,而 torch.<name> 多为 C 内置
    # (inspect.signature 抛 "no signature found for builtin")、torch.ops.aten.<name> 是
    # OpOverloadPacket(签名只有泛型 *args/**kwargs),两者都无法按名绑参,且底层重载常只收
    # 枚举 int(如 reduction 传 "mean" 报 failed to match any schema)。故按【能否绑命名参数】
    # 择优,而不是按算子名逐个开特例。都不可绑参时维持原有的 torch -> aten 兜底次序不变。
    sources = [torch, torch.nn.functional, torch.ops.aten]
    candidates = []
    for source in sources:
        try:
            found = getattr(source, name, None)
        except RuntimeError:
            found = None
        if found is not None:
            candidates.append(found)
    for cand in candidates:
        try:
            params = inspect.signature(cand).parameters.values()
        except (ValueError, TypeError):
            continue
        if any(p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in params):
            return cand
    if candidates:
        return candidates[0]
    result = None
    # suffix stripping (serial-if, NOT elif). The bare `if`
    # chain does last-match-wins: each matching suffix UNCONDITIONALLY reassigns
    # `result`, so a later matching suffix overwrites an earlier one. In practice
    # a name ends in exactly ONE of _d/_grad/_v2 (they are mutually exclusive
    # endings), so last-match-wins == short-circuit return — behaviorally
    # equivalent. EXCEPTION: a name like `xxx_d_v2` matches BOTH `_d` (step 1,
    # result=resolve(xxx_v2)) AND `_v2` (step 3, result=resolve(xxx_d)); the `_v2`
    # step runs last and wins, so `xxx_d_v2` resolves via the `_v2` path. The
    # if-chain structure is kept verbatim (proven behavior; don't second-guess).
    if name.endswith("_d"):
        result = _torch_resolve(name[:-2])
    if name.endswith("_grad"):
        result = _torch_resolve(name[:-5] + "_backward")
    if name.endswith("_v2"):
        result = _torch_resolve(name[:-3])
    return result


def _tf_resolve(name):
    """Resolve a Camel-form op name to a tf callable."""
    import tensorflow as tf
    from tensorflow import raw_ops
    from tensorflow.python.ops import (
        gen_array_ops,
        gen_bitwise_ops,
        gen_image_ops,
        gen_math_ops,
        gen_nn_ops,
        gen_state_ops,
        nn_impl,
    )

    # === special-case mapping (proven — do not trim entries) ===
    tf_func_map = {
        "batch_norm_v3": nn_impl.fused_batch_norm,
        "spence": tf.math.special.spence,
        "space_to_batch": gen_array_ops.SpaceToBatch,
    }
    if name in tf_func_map:
        return tf_func_map[name]
    sources = [
        tf,
        tf.math,
        gen_image_ops,
        gen_nn_ops,
        tf.nn,
        gen_state_ops,
        gen_array_ops,
        gen_math_ops,
        nn_impl,
        gen_bitwise_ops,
        raw_ops,
    ]

    try:
        from tensorflow.python.training import training_ops

        sources.append(training_ops)
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        from tensorflow.python.ops import gen_stateless_random_ops_v2

        sources.append(gen_stateless_random_ops_v2)
    except (ModuleNotFoundError, ImportError):
        pass

    for source in sources:
        result = getattr(source, name, None)
        if result is not None:
            return result
    # suffix stripping (replace-based; order matters)
    if "_d" in name:
        r = _tf_resolve(name.replace("_d", ""))
        if r is not None:
            return r
    if "_v2" in name:
        r = _tf_resolve(name.replace("_v2", ""))
        if r is not None:
            return r
    if "_v1" in name:
        r = _tf_resolve(name.replace("_v1", ""))
        if r is not None:
            return r
    if "_v3" in name:
        r = _tf_resolve(name.replace("_v3", ""))
        if r is not None:
            return r
    if "_1d" in name:
        r = _tf_resolve(name.replace("_1d", "1d"))
        if r is not None:
            return r
    if "_2d" in name:
        r = _tf_resolve(name.replace("_2d", "2d"))
        if r is not None:
            return r
    if "_3d" in name:
        r = _tf_resolve(name.replace("_3d", "3d"))
        if r is not None:
            return r
    if "_" in name:
        r = _tf_resolve(name.replace("_", ""))
        if r is not None:
            return r
    return None


# --- ACLNN api resolve: aclnn 前缀算子名的 torch 搜索段 ---
# camel_to_snake duplicates ttk/utilities/string.py (kept in sync with it);
# _find_torch_api / _auto_import_from_torch search torch / torch.nn.functional /
# torch.ops.aten with suffix stripping. The plugin lookup that precedes the
# torch fallback in TTK's golden chain is not portable (needs plugin_path /
# OpApiInfoKeeper), so only the torch-search segment lives here.


def camel_to_snake(camel_name: str) -> str:
    """
    Operator Registered Camel name convert to snake name.
    Duplicates ttk/utilities/string.py (kept in sync with it).
    """
    snake_name = ""
    sub_head = False
    name_list = list(camel_name)
    for _idx, _char in enumerate(name_list):
        if _char.islower():
            sub_head = False
        if _char.isdigit():
            sub_head = True
        if _char.isupper() and _idx != 0:
            if not sub_head:
                snake_name += "_"
                sub_head = True
            else:
                _idx_next = _idx + 1
                if _idx_next < len(name_list) and name_list[_idx_next].islower():
                    snake_name += "_"
        snake_name += _char

    return snake_name.lower()


def _find_torch_api(torch_module, snake_name: str):
    # find torch.snake_name (getattr + suffix stripping recursion below).
    try:
        torch_api = getattr(torch_module, snake_name, None)
    except RuntimeError:
        # older torch version, 1.11.0 for exp,
        # will throw RuntimeError when absent.
        torch_api = None
    if torch_api and isinstance(torch_api, _Callable):
        return torch_api
    if snake_name.startswith("inplace_"):
        return _find_torch_api(torch_module, snake_name[len("inplace_") :])
    if snake_name.endswith("_scalar"):
        return _find_torch_api(torch_module, snake_name[: -len("_scalar")])
    if snake_name.endswith("_tensor"):
        return _find_torch_api(torch_module, snake_name[: -len("_tensor")])
    if snake_name.endswith("_v2"):
        return _find_torch_api(torch_module, snake_name[: -len("_v2")])
    if snake_name.endswith("s"):
        return _find_torch_api(torch_module, snake_name[: -len("s")])
    if not snake_name.startswith("_"):  # _add_relu
        return _find_torch_api(torch_module, "_" + snake_name)
    return None


def _auto_import_from_torch(snake_name: str):
    """Search torch / torch.nn.functional / torch.ops.aten.

    ACLNN-only path (provider=='torch'): sole caller is _aclnn_resolve; a TF
    request never reaches here.
    """
    import torch

    # torch api first
    torch_api = _find_torch_api(torch, snake_name)
    if torch_api is not None:
        return torch_api
    torch_nn_api = _find_torch_api(torch.nn.functional, snake_name)
    if torch_nn_api is not None:
        return torch_nn_api
    return _find_torch_api(torch.ops.aten, snake_name)


def _aclnn_resolve(api_name):
    """ACLNN: strip 'aclnn' prefix + camel_to_snake + torch search.

    Returns (callable, name) | (None, None). name = strip+snake 后的实际搜索 key
    （如 aclnnAdd → add），供 _resolve_callable 拼 provider.name（如 torch.add）——
    X-API 回传的是 server 实际执行的 API，不是请求的原始 op_name。
    """
    if not api_name or not api_name.startswith("aclnn"):
        return None, None
    snake = camel_to_snake(api_name[5:])  # strip "aclnn" (5 chars) + Camel→snake
    f = _auto_import_from_torch(snake)
    return (f, snake) if f is not None else (None, None)


def _resolve_3party_api(op_name, op_type, provider):
    """KERNEL/ACLNN: resolve op_name(snake)/op_type(Camel) -> (callable, name).

    aclnn-prefixed op_name routes through _aclnn_resolve (strip + torch search),
    returning (callable, op_name). Otherwise dual-form KERNEL port: try op_name
    then op_type, first hit wins (tf benefits from Camel; torch resolves via
    snake). name = 命中的搜索入参（op_name 或 op_type），非 callable 的
    __name__（suffix stripping 脱钩）。Returns (None, None) if unresolvable.
    """
    # ACLNN branch: aclnn-prefixed op_name -> torch search. ACLNN is torch-only;
    # provider is assumed torch (not asserted — _aclnn_resolve does a torch
    # search regardless of provider, which is correct only for torch).
    if op_name and op_name.startswith("aclnn"):
        return _aclnn_resolve(op_name)
    resolve = _torch_resolve if provider == "torch" else _tf_resolve
    for name in (op_name, op_type):
        if not name:
            continue
        f = resolve(name)
        if f is not None:
            return f, name  # name = 命中的搜索入参
    return None, None


def _resolve_callable(exec_type, provider, api, op_name, op_type, spec_module, spec_class):
    """Resolve the callable to execute. Returns (callable, api_label).

    api_label = 实际执行 api 的标识，供 X-API response header 回传：
      - spec 模式: spec_class（dotted 路径，e.g. "MySpec.AddSpec"）
      - api 模式 (explicit api): api 字符串本身（e.g. "torch.nn.functional.softmax"）
      - api=None 推导模式: provider.命中name（e.g. "torch.add"、"tf.Relu"、"torch.aclnnAdd"）
    """
    if exec_type == "spec":
        if not spec_class:
            raise ValueError("spec mode requires spec_class")
        try:
            mod = importlib.import_module(spec_module)
        except ImportError as e:
            name = getattr(e, "name", None) or spec_module
            err = _MissingSpecDependency(name)
            err.name = name  # ImportError.name isn't set by manual construction
            raise err from e
        cls = _resolve_attr_path(mod, spec_class) if spec_class else mod
        tp = getattr(cls, "third_party", None)
        if tp is None:
            return cls, spec_class
        if isinstance(tp, str):
            return resolve_callable(tp), spec_class
        if isinstance(tp, dict):
            tp = {_TP_ALIASES.get(k, k): v for k, v in tp.items()}
            if provider not in tp:
                raise ValueError(f"provider '{provider}' not in third_party")
            v = tp[provider]
            return (resolve_callable(v) if isinstance(v, str) else v), spec_class
        raise ValueError("unsupported third_party format")
    # api mode
    if api:
        return resolve_callable(api), api
    # api=None -> KERNEL/ACLNN: server derives via _resolve_3party_api
    f, name = _resolve_3party_api(op_name, op_type, provider)
    if f is None:
        raise ValueError(f"cannot resolve api for op_name={op_name!r} op_type={op_type!r} provider={provider!r}")
    return f, f"{provider}.{name}"  # 推导 api_label = provider.命中name


def _bind(func, named, attrs, device, warn_leftover=True, param_order=None):
    merged = dict(attrs or {})
    merged.update(named)
    if param_order:
        from collections import OrderedDict

        ordered = OrderedDict()
        for name in param_order:
            if name in merged:
                ordered[name] = merged[name]
        for k, v in merged.items():
            if k not in ordered:
                ordered[k] = v
        merged = ordered
    return bind_params(func, merged, device=device, warn_leftover=warn_leftover)


def _invoke(callable_fn, named, attrs, provider, device_id, use_device, profile=None, param_order=None):
    """Run the callable once, return RAW outputs (no numpy cast)."""
    if inspect.isclass(callable_fn):
        cls = callable_fn
        dev = format_device(provider, profile, "cpu" if not use_device else device_id)
        # 契约: __init__/__call__ 参数(除kwargs)并集 ⊆ inputs∪attrs; input/attr
        # 喂给声明它的方法(都声明则都喂)。device 保留注入, 有默认值用默认。
        # warn_leftover=False: 一个方法没消耗的可能喂给另一个, 不报 leftover 误报。
        if cls.__init__ is object.__init__:  # 无自定义 __init__: 直接实例化
            inst = cls()
        else:
            ia, ik = _bind(cls.__init__, named, attrs, dev, warn_leftover=False, param_order=param_order)
            inst = cls(*ia, **ik)
        ca, ck = _bind(inst.__call__, named, attrs, dev, warn_leftover=False, param_order=param_order)
        return inst(*ca, **ck)
    return _invoke_function(callable_fn, named, attrs, param_order)


def _function_param_names(sig):
    """Non-self, non-*args/**kwargs parameter names, in declaration order."""
    names = []
    for p, v in sig.parameters.items():
        if v.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL) and p != "self":
            names.append(p)
    return names


def _ordered_values(param_order, merged, attrs):
    """``param_order`` entries present in ``merged`` but not declared as attrs.

    Keeps attrs out of the positional zip so an attr never consumes a tensor
    slot; attrs are re-supplied as keywords by the caller.
    """
    values = []
    for k in param_order:
        if k in merged and k not in (attrs or {}) and k != "self":
            values.append(merged[k])
    return values


def _invoke_function(callable_fn, named, attrs, param_order):
    """Three-phase invoke for a plain function (not a class).

    1. kwargs as-is (when input names match API param names)
    2. positional (torch uses input/other which don't match schema names)
    3. inspect signature, zip positional values to real param names (TF raw_ops)

    param_order (when available) provides C-header order so 'self' and other
    params are zipped correctly without exclusion-based hacks.
    """
    merged = dict(attrs or {})
    merged.update(named)
    try:
        call_kwargs = {k: v for k, v in merged.items() if k != "self"}
        return callable_fn(**call_kwargs)
    except TypeError:
        pass

    try:
        # Positional values = named INPUTS only; attrs go by keyword (their
        # names match the API params). Folding attrs into pos_vals duplicated
        # them (pos_vals + pos_kwargs) and mis-bound them positionally.
        pos_vals = [v for k, v in named.items() if k != "self"]
        pos_kwargs = {k: v for k, v in (attrs or {}).items() if k != "self"}
        return callable_fn(*pos_vals, **pos_kwargs)
    except TypeError as exc:
        bind_error = exc

    # No straightforward call matched: zip positional values to the real
    # parameter names from the signature (TF raw_ops-style). torch C builtins
    # carry no Python signature (ValueError) — surface the phase-2 binding
    # error instead of crashing with an opaque 500.
    try:
        sig = inspect.signature(callable_fn)
    except ValueError:
        raise TypeError(
            f"{getattr(callable_fn, '__name__', callable_fn)} cannot be parameter-bound "
            f"from inputs {list(named.keys())}: no Python signature, "
            f"and both kwargs and positional invocation failed"
        ) from bind_error
    param_names = _function_param_names(sig)
    if param_order:
        # attrs are already re-supplied as keywords below; keep them out
        # of the positional zip so an attr never consumes a tensor slot.
        ordered_vals = _ordered_values(param_order, merged, attrs)
        if len(ordered_vals) <= len(param_names):
            zip_kwargs = {k: v for k, v in (attrs or {}).items() if k != "self"}
            return callable_fn(**dict(zip(param_names, ordered_vals), **zip_kwargs))
    vals = [v for k, v in named.items() if k != "self"]
    if len(vals) > len(param_names):
        raise TypeError(
            f"{getattr(callable_fn, '__name__', callable_fn)} expects "
            f"{len(param_names)} params {param_names}, "
            f"got {len(vals)} inputs {list(named.keys())}"
        ) from bind_error
    zip_kwargs = {k: v for k, v in (attrs or {}).items() if k != "self"}
    return callable_fn(**dict(zip(param_names, vals), **zip_kwargs))


def _tf_bf16_wire(v):
    """tf bf16 张量位级 bitcast 成 int16 按位落盘；非 bf16 张量/不可用返回 None。

    绕开 numpy __array__ 协议（部分 TF/numpy 组合下协议产物不可识别），
    wire 位与通用 bf16 分支等价。
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None
    if tf.is_tensor(v) and v.dtype == tf.bfloat16:
        try:
            return np.asarray(tf.bitcast(v, tf.int16).numpy()), "bfloat16"
        except Exception:  # noqa: S110
            logging.debug("tf.bitcast bf16→int16 failed; falling back to generic asarray path", exc_info=True)
    return None


def _to_numpy_pair(v, provider):
    """Convert one output to (numpy_array, semantic_dtype_name), provider-aware.

    bfloat16 can't round-trip through numpy savez, so it's stored as raw int16
    bits with 'bfloat16' declared (client reinterprets). float8 likewise has no
    numpy storage class — stored as raw uint8 bits with the float8 dtype name
    declared. complex32 also has no numpy storage class — stored as interleaved
    fp16 with a trailing [2] dim (TTK storage convention: [..., 0]=real,
    [..., 1]=imag). torch is imported ONLY for the torch path; tf/other paths
    never touch it (tf bfloat16 ships a numpy bf16 dtype via
    tensorflow.bfloat16.as_numpy_dtype).
    """
    if provider == "torch":
        try:
            import torch

            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bfloat16:
                    return v.contiguous().view(torch.int16).cpu().numpy(), "bfloat16"
                if "complex32" in str(v.dtype):
                    fp16 = v.contiguous().view(torch.float16)
                    shape = list(v.shape)
                    out_shape = shape[:-1] + [shape[-1], 2] if shape else [2]
                    return fp16.cpu().numpy().reshape(out_shape), "complex32"
                if "float8" in str(v.dtype):
                    return v.contiguous().view(torch.uint8).cpu().numpy(), str(v.dtype).replace("torch.", "")
                return v.detach().cpu().numpy(), str(v.dtype).replace("torch.", "")
        except ImportError:
            pass
    # tf tensor / numpy / array-like — np.asarray handles it; never imports torch.
    tf_out = _tf_bf16_wire(v) if provider == "tf" else None
    if tf_out is None:
        a = np.asarray(v)
        dt = a.dtype.name
        if dt in ("bfloat16", "bf16"):
            return np.ascontiguousarray(a).reshape(a.shape).view(np.int16), "bfloat16"
        if dt in _UINT8_WIRE_DTYPES:
            # NumPy/TF float8/4bit dtypes use non-standard 1-byte descriptors that
            # NumPy cannot read back from NPZ. Keep the semantic dtype in schema
            # and serialize only the raw byte representation, like torch outputs.
            return np.ascontiguousarray(a).reshape(a.shape).view(np.uint8), dt
        return a, dt
    return tf_out


def _outputs_to_numpy(outputs, provider):
    """保留嵌套（不 flatten）。遍历顶层 slots，构造 schema。
    返回 (schema, arrays_叶子)。schema: [{index|indices|null, dtype}, ...]"""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    schema = []
    arrays = []
    npz_idx = 0
    for slot in outputs:
        if slot is None:
            schema.append({"index": None, "dtype": None})  # None 只在顶层 slot（op 输出占位）
            continue
        if isinstance(slot, (list, tuple)):
            # tensor-list slot：叶子均为实际 tensor（op 输出约束；None 只在顶层 slot 处理）
            leaves = []
            dt = None
            for leaf in slot:
                arr, dt = _to_numpy_pair(leaf, provider)
                arrays.append(arr)
                leaves.append(npz_idx)
                npz_idx += 1
            schema.append(
                {"indices": leaves, "dtype": dt}
            )  # dt 取末叶子；list 内假设 dtype 同质（op 输出约束；异质 list 未支持）
        else:
            arr, dt = _to_numpy_pair(slot, provider)
            arrays.append(arr)
            schema.append({"index": npz_idx, "dtype": dt})
            npz_idx += 1
    return schema, arrays


def _restore_wire_dtype(value, dtype_name):
    """按 schema 声明还原 npz 存不回的自定义 dtype（4bit / float8 家族）。

    wire 上只有原始字节（dtype 元数据已退化），重铸后才能参与运算或
    numpy 直通。非自定义 dtype 返回 None（走常规转换）；依赖包缺失时
    原样返回（兜底直通）。
    """
    if not isinstance(value, np.ndarray) or not dtype_name:
        return None
    if not (is_4bit_dtype(dtype_name) or dtype_name in _UINT8_WIRE_DTYPES):
        return None
    try:
        return value.view(resolve_numpy_dtype(dtype_name))
    except (RuntimeError, ValueError, TypeError):
        return value


def _to_torch_float8(value, dtype_name, device_str):
    """float8 numpy → torch 张量；失败返回 None（回落 numpy 直通）。"""
    if dtype_name not in _TORCH_FLOAT8_DTYPES:
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        torch_dtype = getattr(torch, dtype_name)
        return torch.from_numpy(np.ascontiguousarray(value).view(np.uint8)).view(torch_dtype).to(device_str)
    except Exception:
        return None


def _to_vendor_tensor(value, provider, device_str, dtype_name=None, logical_dtype=None):
    """Framework H2D: numpy input -> provider tensor on device.

    Inputs arrive as numpy (restored from the tmp_in savez). torch/tf callables
    need tensors, so convert before binding. ``dtype_name`` is the physical dtype
    declared in X-Input-Schema by the client (whose numpy knows the real dtype) —
    used to convert dtypes the server's numpy can't represent (bfloat16) without
    guessing. ``logical_dtype`` is the CSV-declared dtype (e.g. 'complex32'):
    complex32 ships as interleaved fp16 with a trailing [2] dim (real, imag) —
    reinterpreted into a real torch.complex32 tensor with the logical shape so
    vendor APIs see the dtype they expect. None / lists recurse; non-numpy /
    unknown provider pass through.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(_to_vendor_tensor(v, provider, device_str, dtype_name, logical_dtype) for v in value)
    if isinstance(value, np.ndarray):
        restored = _restore_wire_dtype(value, dtype_name)
        if restored is not None:
            # torch/tf 无 4bit 存储；torch 能表示的 float8 转张量，
            # 其余以 numpy 直通，由三方 compose 自行消费。
            if provider == "torch":
                as_torch = _to_torch_float8(restored, dtype_name, device_str)
                if as_torch is not None:
                    return as_torch
            return restored
    if provider == "torch":
        try:
            import torch
        except ImportError:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(device_str)
        if isinstance(value, np.ndarray):
            if dtype_name == "bfloat16":
                # numpy 无原生 bf16：wire 是原始 int16 位，重铸回 bf16。
                return torch.from_numpy(value.view(np.int16)).view(torch.bfloat16).to(device_str)
            if logical_dtype == "complex32" and value.dtype == np.float16:
                # complex32 以 fp16 交错位传输，还原逻辑张量。
                if not hasattr(torch, "complex32"):
                    raise RuntimeError(
                        f"Current pytorch version [{torch.__version__}] is too old. Please update to at least v1.13.1"
                    )
                return torch.from_numpy(value).view(torch.complex32).squeeze(-1).to(device_str)
            return torch.from_numpy(value).to(device_str)
        return value
    if provider == "tf":
        try:
            import tensorflow as tf

            if dtype_name == "bfloat16" and isinstance(value, np.ndarray):
                # wire 是原始 int16 位，重铸 tf bf16。
                value = value.view(tf.bfloat16.as_numpy_dtype)
            # 钉在分配的设备上；TF2 软放置使设备缺失时 tf.device 不产生副作用。
            if device_str and device_str != "cpu":
                with tf.device(device_str):
                    return tf.convert_to_tensor(value)
            return tf.convert_to_tensor(value)
        except ImportError:
            return value
    return value


def device_available(provider, profile) -> bool:
    """Provider-aware device availability — never imports torch for a tf request.

    profile-driven: torch -> getattr(torch, torch_lib).is_available();
    tf -> tf_type = profile.get("tf_device_type"); False when tf_type is missing
    (graceful degrade, NOT a config error — distinguishes from format_device's
    ValueError on the same missing field). The torch_{lib} extension is imported
    upstream in execute_request before this is reached.
    """
    if provider == "torch":
        # Defensive on torch_lib: tf branch degrades via .get(); mirror that so a
        # missing key degrades to False rather than KeyError. Prefer the cached
        # backend module (set in execute_request) to avoid repeating getattr.
        # (import torch sits right under the provider=='torch' gate so the static
        # guard recognizes it as vendor-gated.)
        try:
            import torch
        except ImportError:
            return False
        lib_name = profile.get("torch_lib")
        if lib_name is None:
            return False
        dev = _TORCH_DEV_MODULE if _TORCH_DEV_MODULE is not None else getattr(torch, lib_name, None)
        if dev is None:
            return False
        return dev.is_available()
    if provider == "tf":
        tf_type = profile.get("tf_device_type")
        if tf_type is None:
            return False
        try:
            import tensorflow as tf

            return bool(tf.config.list_physical_devices(tf_type))
        except ImportError:
            return False
    return False


def _run_perf(callable_fn, named, attrs, provider, device_id, use_device, profile=None, runtime=3, param_order=None):
    """PERF timing via profiler (torch: Self device time; TF: xplane.pb device plane).

    Two passes: (1) profiler pass for device_us (no empty_cache); (2) peak pass
    for peak_memory_mb (separate, reset_peak then re-invoke). device_us is the
    per-iteration average over ``runtime`` active iterations (μs, 3 sig figs).

    CPU / no-device -> device_us=NA. Device EXCLUSIVITY is enforced by the PARENT
    (it holds a lock around this child); this function does NOT lock. reset_peak
    is per-device global, so the parent's lock keeps concurrent PERF from
    corrupting it.
    """
    profile = profile or {}
    perf = {"device_us": "NA", "peak_memory_mb": "NA"}
    # use_device is first in the `and` chain so device_available is NOT called
    # when profile={} (cpu short-circuit upstream + and-use_device double-guard).
    torch_dev = use_device and provider == "torch" and device_available(provider, profile)
    tf_dev = use_device and provider == "tf" and device_available(provider, profile)

    # --- Pass 1: profiler (timing) ---
    outputs = None
    device_us = 0.0  # init before branch: CPU branch never assigns it
    if torch_dev:
        outputs, device_us = _torch_profiler_pass(
            callable_fn, named, attrs, provider, device_id, use_device, profile, runtime, param_order=param_order
        )
    elif tf_dev:
        outputs, device_us = _tf_profiler_pass(
            callable_fn, named, attrs, provider, device_id, use_device, profile, runtime, param_order=param_order
        )
    else:
        # CPU or no device: single invoke for outputs only
        outputs = _invoke(
            callable_fn, named, attrs, provider, device_id, use_device, profile=profile, param_order=param_order
        )

    if device_us > 0:
        perf["device_us"] = float(f"{device_us:.3g}")

    # --- Pass 2: peak (memory) ---
    # Pass 2 is best-effort: the op already executed in pass 1 (an op error would
    # have propagated -> FAIL), so any failure here is memory-API breakage ->
    # peak_memory_mb=NA (logged).
    if torch_dev:
        try:
            lib = _TORCH_DEV_MODULE
            lib.reset_peak_memory_stats(device_id)
            _invoke(
                callable_fn, named, attrs, provider, device_id, use_device, profile=profile, param_order=param_order
            )
            lib.synchronize()
            perf["peak_memory_mb"] = lib.max_memory_allocated(device_id) / 1e6
        except Exception:
            logging.exception("torch peak_memory measure failed; peak_memory_mb=NA")
    elif tf_dev:
        try:
            import tensorflow as tf

            tf_device = f"{profile['tf_device_type']}:{device_id}"
            tf.config.experimental.reset_memory_stats(tf_device)
            _invoke(
                callable_fn, named, attrs, provider, device_id, use_device, profile=profile, param_order=param_order
            )
            info = tf.config.experimental.get_memory_info(tf_device)
            perf["peak_memory_mb"] = info.get("peak", 0) / 1e6
        except Exception:
            logging.exception("tf peak_memory measure failed; peak_memory_mb=NA")

    return outputs, perf


def _torch_profiler_pass(
    callable_fn, named, attrs, provider, device_id, use_device, profile, runtime, param_order=None
):
    """pass 结构映射本包 xpu_profiler.torch_xpu_profiler_pass。"""
    from .xpu_profiler import torch_xpu_profiler_pass

    device = profile["torch_lib"]
    try:
        activities_cfg = profile["torch_profiler"]["activities"]
    except KeyError:
        raise RuntimeError("profile missing torch_profiler/activities") from None

    def invoke():
        return _invoke(
            callable_fn, named, attrs, provider, device_id, use_device, profile=profile, param_order=param_order
        )

    return torch_xpu_profiler_pass(invoke, runtime, device, activities_cfg, torch_lib=profile["torch_lib"])


def _tf_profiler_pass(callable_fn, named, attrs, provider, device_id, use_device, profile, runtime, param_order=None):
    """pass 结构映射本包 xpu_profiler.tf_xpu_profiler_pass。"""
    from .xpu_profiler import tf_xpu_profiler_pass

    def invoke():
        return _invoke(
            callable_fn, named, attrs, provider, device_id, use_device, profile=profile, param_order=param_order
        )

    return tf_xpu_profiler_pass(invoke, runtime)


def _ensure_torch_backend(provider, use_device, profile):
    """按需导入 torch 后端扩展并缓存到 _TORCH_DEV_MODULE。

    非默认后端是扩展包：import torch_{lib} 注册 device_available / 内存查询 /
    ProfilerActivity 用到的 torch.<lib> 命名空间。默认后端内建（hasattr
    True → 跳过）；扩展缺失 raise RuntimeError → 500。torch 专属（TF 请求
    不触 torch 命名空间）；cpu 短路（use_device=False）。
    """
    if not (use_device and provider == "torch" and profile):
        return
    lib = profile.get("torch_lib")
    if lib is None:
        raise RuntimeError("profile missing torch_lib")
    import torch

    if not hasattr(torch, lib):
        try:
            importlib.import_module(f"torch_{lib}")
        except ModuleNotFoundError as e:
            raise RuntimeError(f"torch backend '{lib}' unavailable: {e}") from e
    # 一次性请求子进程内每请求写一次（见模块级 _TORCH_DEV_MODULE）
    global _TORCH_DEV_MODULE
    _TORCH_DEV_MODULE = getattr(torch, lib)


def _load_named_inputs(input_schema, tmp_in_path, input_count, provider, device_str):
    """npz 回解 → 按 X-Input-Schema 还原命名输入 → H2D 转框架 tensor。

    dtype 用 schema 声明的物理 dtype（client 侧 numpy 认识真实 dtype，
    server 侧可能不认识，如 bfloat16）；logical_dtype 为 CSV 声明值——
    complex32 以 fp16+尾维[2] 到达，据此还原成 torch.complex32 逻辑形状。
    """
    if not (input_count and tmp_in_path):
        return {}
    npz = np.load(tmp_in_path)
    flat = [npz[f"a{i}"] for i in range(input_count)]
    named = match_params_v1(input_schema, flat)
    dtypes = {e.get("name"): e.get("dtype") for e in (input_schema or [])}
    logical = {e.get("name"): e.get("logical_dtype") for e in (input_schema or [])}
    return {k: _to_vendor_tensor(v, provider, device_str, dtypes.get(k), logical.get(k)) for k, v in named.items()}


def _compose_kwargs(attrs, input_schema, callable_fn):
    """三方执行入参：attrs 原样；仅类形态实现（compose）注入 input_formats。

    format 已内嵌在 X-Input-Schema 每个条目——api 模式函数（如 torch.asinh）
    的形参与框架合成 kwargs 无关，注入会让 phase-2 变成
    torch.asinh(t, input_formats=['ND']) → TypeError（绑定失败）。
    """
    compose_kwargs = dict(attrs or {})
    schema_formats = [e.get("format") for e in (input_schema or []) if e.get("format")]
    if schema_formats and inspect.isclass(callable_fn):
        compose_kwargs["input_formats"] = schema_formats
    return compose_kwargs


def execute_request(
    *,
    tenant_sync_dir,
    exec_type,
    provider,
    api,
    spec_module,
    spec_class,
    mode,
    input_schema,
    attrs,
    tmp_in_path,
    input_count,
    device_id,
    use_device,
    output_dir,
    profile=None,
    op_name=None,
    op_type=None,
    runtime=3,
    param_order=None,
    **_extra,
):
    """Run one request. Returns an envelope dict (never raises for 4xx/5xx)."""
    try:
        if tenant_sync_dir and tenant_sync_dir not in sys.path:
            sys.path.insert(0, tenant_sync_dir)
        importlib.invalidate_caches()

        _ensure_torch_backend(provider, use_device, profile)

        device_str = format_device(provider, profile, "cpu" if not use_device else device_id)
        named = _load_named_inputs(input_schema, tmp_in_path, input_count, provider, device_str)

        callable_fn, api_label = _resolve_callable(exec_type, provider, api, op_name, op_type, spec_module, spec_class)
        compose_kwargs = _compose_kwargs(attrs, input_schema, callable_fn)
        if has_perf(mode):
            raw_outputs, perf = _run_perf(
                callable_fn,
                named,
                compose_kwargs,
                provider,
                device_id,
                use_device,
                profile=profile,
                runtime=runtime,
                param_order=param_order,
            )
        else:
            raw_outputs = _invoke(
                callable_fn,
                named,
                compose_kwargs,
                provider,
                device_id,
                use_device,
                profile=profile,
                param_order=param_order,
            )
            perf = None
        schema, outs = _outputs_to_numpy(raw_outputs, provider)

        if has_data(mode):
            path = os.path.join(output_dir, "out.npz")
            np.savez(path, **{f"a{i}": o for i, o in enumerate(outs)})
            return _ok(
                path, len(schema), [list(o.shape) for o in outs], dtypes=None, perf=perf, api=api_label, schema=schema
            )
        return _ok(None, 0, [], [], perf=perf, api=api_label, schema=[])

    except _MissingSpecDependency as e:
        # 424 is syncable: client will _sync_missing_dependency + retry. Not a
        # failure — info, not warning (avoids noise on the expected first miss).
        logging.info("spec dependency missing: %s (awaiting client sync)", e.name)
        return _err(424, f"missing spec dependency: {e}", missing=e.name, api=api or spec_class)
    except ImportError as e:
        logging.exception("request failed: import error (api=%s)", api or spec_class)
        return _err(500, f"import failed: {e}", api=api or spec_class)
    except (UnknownParamError, ValueError) as e:
        logging.warning("request failed: bad params (api=%s): %s", api or spec_class, e)
        return _err(400, str(e), api=api or spec_class)
    except Exception as e:
        logging.exception("request failed (api=%s)", api or spec_class)
        return _err(500, _client_error(e), api=api or spec_class)


def _die_with_parent():
    """向内核注册"父进程一死就杀掉我"(PR_SET_PDEATHSIG)。

    每个请求一个子进程, 若服务端被 ``kill -9``/段错误终止, 这些子进程会被 reparent
    而继续活着(atexit 之类的钩子在 SIGKILL 下不会执行), 历史上出现过孤儿堆积吃满内存。
    PDEATHSIG 是内核级保证, 不依赖父进程还能执行任何代码。

    这里用它是安全的: 子进程入口是单线程, 不存在"注册线程先退出导致误杀"的那个已知坑。
    不可用(非 Linux / prctl 缺失)时静默跳过, 绝不影响请求处理。
    """
    try:
        import ctypes
        import signal as _signal

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, _signal.SIGKILL)
        # 注册与父进程退出之间存在竞态: 若父已先死, 现在补一次自检。
        if os.getppid() == 1:
            os._exit(1)
    except Exception as e:  # noqa: BLE001  仅为兜底清理, 失败不应影响正常请求
        logging.debug("PDEATHSIG unavailable (%s); orphan cleanup falls back to parent-side kill", e)


def child_main(conn, kwargs):
    """Child-process entry point: run execute_request, send the envelope back.

    A hard crash (segfault/OOM) kills the child before conn.send -> the parent
    sees a nonzero exitcode and no message -> 500.
    """
    # pop env BEFORE execute_request: sandbox=none execution isolation passes the
    # assigned device via env; if it leaked through, **_extra would silently
    # swallow it and isolation would silently fail. os imported
    # at module top.
    _die_with_parent()
    os.environ.update(kwargs.pop("env", {}))
    try:
        conn.send(execute_request(**kwargs))
    except Exception as e:
        logging.exception("child_main: execute_request raised (api=%s)", _api_from_kwargs(kwargs))
        conn.send(_err(500, _client_error(e), api=_api_from_kwargs(kwargs)))
    finally:
        conn.close()
