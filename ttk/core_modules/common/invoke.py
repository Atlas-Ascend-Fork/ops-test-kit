#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/invoke.py：统一执行器——resolve 之后的执行层。

SpecResolver 决定"调谁"（callable/类/None），invoke 决定"怎么调"（占位
语义 / str 解析 / 元信息注入 / 异常收敛）。golden / npu_preprocess 的
执行入口集中于全静态类 Invoke；三方执行入口为 common/xpu.py 的
collect_third_party 门面。

- bind_by_name / resolve_callable_str：模块级公开（跨通路直调）
- sig_params / accepts_var_keyword / instantiate_and_call / golden_flatten：
  golden 通路共享小件（签名探测 / class 双段调用 / 输出规整）
- framework_of：仅 golden 分发用——私有
"""

__all__ = [
    "Invoke",
    "resolve_callable_str",
    "golden_mode",
    "sig_params",
    "accepts_var_keyword",
    "instantiate_and_call",
    "golden_flatten",
    "apply_npu_preprocess_result",
    "apply_aclnn_npu_preprocess_result",
]

# Standard Packages
import gc
import inspect
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy

from ...utilities import deep_flatten
from .testcase.batch import BATCH_RELATION_FIELDS, batch_relation_kwargs, has_complete_batch_relation


def resolve_callable_str(s: str):
    """Resolve a dotted-path string like 'numpy.abs' / 'torch.mm' / 'tf.raw_ops.Add' into a callable.

    Lazy import：只 import 字符串引用的框架（避免解析 'numpy.add' 时 eager-import
    torch/tensorflow —— 某些环境的 tensorflow C 扩展 import 即 segfault）。
    np 别名 = numpy；torch/tf 按前缀按需加载。

    逐段 getattr 解析（纯点路径）：s 可能来自用例 CSV（如 TF 的 api_name），
    不允许任意表达式求值。"""
    parts = s.split(".")
    if len(parts) < 2 or not all(parts):
        raise ValueError(f"Cannot resolve golden callable {s!r}: expected a dotted path")
    root = parts[0]
    if root in ("numpy", "np"):
        obj = numpy
    elif root == "torch":
        import torch

        obj = torch
    elif root in ("tf", "tensorflow"):
        import tensorflow as tf

        obj = tf
    elif root in ("npu_device", "npu_bridge"):
        # TF Adapter 暴露的 NPU 自定义算子。这些包把 npu_device.compat 重映射到
        # npu_device._api.compat, 从根包逐级 getattr 取不到(_api 下没有 tbe), 所以改成
        # "导入最深的可导入模块, 余下部分再 getattr"。
        import importlib

        for k in range(len(parts) - 1, 0, -1):
            try:
                mod = importlib.import_module(".".join(parts[:k]))
            except ImportError:
                continue
            obj = mod
            try:
                for attr in parts[k:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            return obj
        raise ValueError(f"Cannot resolve golden callable {s!r}: no importable module prefix")
    else:
        raise ValueError(f"Cannot resolve golden callable {s!r}: unsupported root module {root!r}")
    try:
        for attr in parts[1:]:
            obj = getattr(obj, attr)
    except AttributeError as e:
        raise ValueError(f"Cannot resolve golden callable {s!r}: {e}") from e
    return obj


def golden_mode(testcase, switches) -> str:
    """golden 模式解析（单份）：用例级 golden_mode_override 优先于全局 --golden-mode。

    缺省回 "Enable"；switches 为 None / 缺字段的鸭子类型（测试 SimpleNamespace）
    同样安全。仅 Promote/Disable/XPU 比较消费此值。
    """
    return getattr(testcase, "golden_mode_override", None) or getattr(switches, "golden_mode", None) or "Enable"


def _framework_of(func) -> Optional[str]:
    """Classify a callable by its framework: 'numpy'/'torch'/'tf'/None (custom)。

    Caveats (verified empirically — do NOT simplify):
    - numpy ufunc has no ``__module__`` → must use getattr fallback.
    - torch top-level functions: ``type(func).__module__ == 'builtins'`` →
      check func BEFORE type(func).
    - Do NOT use ``inspect.getmodule``: None for ufuncs / aten packets.
    """
    for obj in (func, type(func)):
        mod = getattr(obj, "__module__", "") or ""
        if mod.startswith("numpy"):
            return "numpy"
        if mod.startswith("torch"):
            return "torch"
        if mod.startswith(("tensorflow", "tf")):
            return "tf"
    if isinstance(func, numpy.ufunc):
        return "numpy"
    return None


def _bind_by_name(func, pool: dict) -> Tuple[list, dict]:
    """Bind parameters of ``func`` from ``pool`` by name.

    - Parameters before ``*`` → positional args；after ``*`` → keyword kwargs
    - ``*args`` collects unconsumed pool entries（Python 保留名冲突用，如 self）
    - ``**kwargs`` absorbs leftover entries（excluding ``self``）
    - missing & no default → UnknownParamError
    """
    import inspect

    from .xpu import UnknownParamError

    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_kw = False
    has_var_pos = False
    consumed = set()
    for name, p in sig.parameters.items():
        # self 仅在未绑定函数上是实例参数；绑定方法签名已去实例参，残留 self 是
        # API 真实形参名（aten 惯例）——一律跳过会把第一个输入静默丢掉
        if name == "self" and not inspect.ismethod(func):
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            seen_star = True
            has_var_pos = True
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
            continue
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            seen_star = True
        if name in pool:
            consumed.add(name)
            if seen_star:
                kwargs[name] = pool[name]
            else:
                args.append(pool[name])
        elif p.default is not inspect.Parameter.empty:
            if seen_star:
                # Post-star optional param missing from pool: leave it to
                # Python (skip, keyword-only 位置走默认值)
                continue
            # Pre-star optional param missing from pool: append the default to
            # hold the positional slot — skipping it would shift every later
            # value one param to the left (e.g. alpha's value landing on gamma)
            args.append(p.default)
        else:
            raise UnknownParamError(f"parameter {name!r} of {func} has no value in pool")
    leftover = {k: v for k, v in pool.items() if k not in consumed and k != "self"}
    if leftover:
        if has_var_kw:
            kwargs.update(leftover)
        elif has_var_pos:
            args.extend(leftover.values())
        else:
            logging.warning("bind_by_name: unconsumed pool entries %s", sorted(leftover))
    return args, kwargs


def sig_params(func):
    """inspect.signature(func).parameters；异常返回 {}（与正常值同为 mapping，in/.values() 统一可用）。"""
    try:
        return inspect.signature(func).parameters
    except Exception:
        return {}


def accepts_var_keyword(params) -> bool:
    """签名（sig_params 产物）是否声明 **kwargs。"""
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def instantiate_and_call(cls, pool: dict):
    """class 形态 golden 双段调用：__init__ 与 __call__ 各自按名绑定 pool。"""
    if cls.__init__ is object.__init__:
        inst = cls()
    else:
        args, kwargs = _bind_by_name(cls.__init__, pool)
        inst = cls(*args, **kwargs)
    args, kwargs = _bind_by_name(inst.__call__, pool)
    return inst(*args, **kwargs)


def golden_flatten(results, deep: bool = False) -> list:
    """golden 输出规整为扁平列表：非 Sequence 包装、numpy.generic → 1 元素数组。

    deep=True 递归展开嵌套 Sequence（kernel TensorList 输出）；False 仅顶层（aclnn）。
    """
    if not isinstance(results, Sequence):
        results = [results]
    flat = deep_flatten(results) if deep else list(results)
    return [numpy.array((x,)) if isinstance(x, numpy.generic) else x for x in flat]


def _npu_preprocess_param_names(plan):
    """Names declared by a param plan (aclnn param_layout / e2e overload_params)."""
    overload_params = getattr(plan, "overload_params", None)
    if overload_params is not None:
        return {parameter.name for parameter in overload_params}
    return {entry[1] for entry in (getattr(plan, "param_layout", None) or ()) if len(entry) > 1}


def _npu_preprocess_extras(testcase, switches, plan):
    """Meta-info kwargs offered to the hook (batch relation stays all-or-nothing)."""
    parameter_names = _npu_preprocess_param_names(plan)
    attributes = dict(getattr(testcase, "attributes", None) or {})
    extra = {
        name: value
        for name, value in attributes.items()
        if name not in parameter_names and name not in (*BATCH_RELATION_FIELDS, "context")
    }
    extra.update(
        {
            "testcase_name": getattr(testcase, "testcase_name", None),
            "short_soc_version": getattr(switches, "short_soc_version", None),
            "tensor_formats": getattr(testcase, "tensor_formats", None),
            "tensor_dtypes": getattr(testcase, "tensor_dtypes", None),
            "scalar_dtypes": getattr(testcase, "scalar_dtypes", None),
            "input_ranges": getattr(testcase, "input_data_ranges", None),
        }
    )
    # Batch relation fields form one contract.  Do not expose a partial triple
    # to an operator hook when this is only a deterministic execution case.
    extra.update(batch_relation_kwargs(testcase))
    # Framework-side slice replacement marker (input generation already ran).
    extra["batch_applied"] = has_complete_batch_relation(testcase)
    return extra


def _npu_preprocess_assemble(testcase, device_args):
    """Assemble device-side (args, kwargs) plus the plan used for meta filtering.

    device_args wins when given (E2E device tensors are execution-local and
    shared with the main API call); otherwise the testcase's cached plan
    assembles from its own queues (ACLNN: tensors/scalars/attributes).
    """
    plan_getter = getattr(testcase, "get_param_plan", None)
    plan = plan_getter() if plan_getter is not None else None
    if device_args is not None:
        return device_args[0], device_args[1], plan
    if plan is None:
        raise RuntimeError(f"{Invoke.NPU_PREPROCESS_FAILURE} no assembled device args and no param plan to build them")
    args, kwargs = plan.build_args(testcase.tensors, testcase.scalars, testcase.attributes)
    return args, kwargs, plan


def _npu_preprocess_hook_kwargs(parameters, kwargs, extra):
    """Merge caller kwargs with meta extras, trimmed to the hook signature."""
    hook_kwargs = {name: value for name, value in dict(kwargs or {}).items() if name != "context"}
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    for name, value in extra.items():
        if accepts_kwargs or name in parameters:
            hook_kwargs.setdefault(name, value)
    return hook_kwargs


class Invoke:
    """统一执行器（全静态，无状态）。resolve 之后的执行层命名空间。"""

    # hook 契约失败前缀：消费方（aclnn worker）以此区分"契约失败落行"与"其他异常上抛"
    NPU_PREPROCESS_FAILURE = "NPU_PREPROCESS_FAILURE:"

    @staticmethod
    def golden(testcase, func, switches, invoke=None) -> list:
        """统一执行 golden（纯 CPU 语义，三通路共享）。

        收敛三通路 golden 执行的共性语义：占位（None/Disable）、str 形态
        resolve、Timeout/Exception 占位与日志、gc。通路专有部分经 invoke
        闭包注入——func（str 已解析）传入闭包执行（kernel/aclnn：装配池 /
        class 双阶段绑定 / 框架转换 / Promote 包裹）；invoke 缺省直接调用
        func()（e2e：func 为绑定好上下文的 runner）。

        Args:
            testcase: 通路 testcase（golden_mode_override 优先于全局）
            func: resolve 链产物（callable / class / str / None）
            switches: SWITCHES
            invoke: (func) -> results 的通路执行闭包，None → func()
        Returns: golden 输出列表（通路侧 flatten/reshape 之前），或占位字符串列表：
            func is None            → ["UNSUPPORTED"]
            golden_mode == "Disable" → ["SUPPRESSED"]
            TimeoutError             → ["GOLDEN_TIMEOUT"]
            Exception                → ["GOLDEN_FAILURE"]
        golden_mode == "XPU" 时编排层跳过本入口（golden 由三方执行产出，
        入口为 common/xpu.collect_third_party 门面）。
        manual_golden_binaries CSV 列的装载在通路 golden 模块侧。
        """
        if func is None:
            return ["UNSUPPORTED"]
        mode = golden_mode(testcase, switches)
        if mode == "Disable":
            return ["SUPPRESSED"]
        try:
            if isinstance(func, str):
                func = resolve_callable_str(func)
            return invoke(func) if invoke is not None else func()
        except TimeoutError:
            logging.exception("Golden generation timeout")
            return ["GOLDEN_TIMEOUT"]
        except Exception:
            logging.exception("Golden generation failure")
            return ["GOLDEN_FAILURE"]
        finally:
            # gen-0 足够：golden 大数组走引用计数即时释放，此处只清 torch
            # 内部等循环垃圾；全量收集每例 10-100ms，千例套件可感知
            gc.collect(0)

    @staticmethod
    def npu_preprocess(testcase, func, switches, device_args=None, *, device_scope=None):
        """前预处理执行（ACLNN/E2E，设备锁内、H2D 后、主 API 前调用）。

        Args:
            testcase: 输入张量与属性来源；plan 经 get_param_plan() 自取
            func: resolve_npu_preprocess 的返回值（callable / None）
            switches: SWITCHES
            device_args: (args, kwargs) 执行现场已装配的设备侧入参（E2E——
                        设备张量为执行局部量，与主 API 共享同一对象，invoke
                        无法从 testcase 复装配）；None → 经 plan 自装配
                        （ACLNN：tensors/scalars/attributes）
            device_scope: 设备上下文工厂（如 lambda: backend.device_scope(dev_id)），
                          None 则裸跑
        Returns: func is None → None（未注册，跳过）；hook 返回 None → None
            （纯副作用）；hook 返回 dict{参数名: 设备张量} → 校验后原样返回
            （动态 -1 槽物化结果，由调用方经 apply_*_npu_preprocess_result 回填）。

        hook 签名约束（违反抛 NPU_PREPROCESS_FAILURE）：
          - 不得声明 context 参数（框架无关契约）
          - 必须返回 None 或 dict{参数名: 张量}（张量需带 shape/device）
        调用约定：
          - 位置参数 = 设备侧输入张量；属性按关键字传入
          - 自动注入元信息 kwargs（testcase_name / short_soc_version /
            tensor_formats / tensor_dtypes / scalar_dtypes / input_ranges /
            batch_*，仅在 hook 声明同名参数或 **kwargs 时）
          - 在 device_scope() 内执行
        """
        if func is None:
            return None
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} cannot inspect TestSpec.npu_preprocess: {exc}"
            ) from exc
        if "context" in signature.parameters:
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} TestSpec.npu_preprocess must not declare a framework context parameter"
            )
        args, kwargs, plan = _npu_preprocess_assemble(testcase, device_args)
        hook_kwargs = _npu_preprocess_hook_kwargs(
            signature.parameters, kwargs, _npu_preprocess_extras(testcase, switches, plan)
        )
        scope = device_scope() if device_scope is not None else nullcontext()
        try:
            with scope:
                result = func(*args, **hook_kwargs)
        except Exception as exc:
            if str(exc).startswith(Invoke.NPU_PREPROCESS_FAILURE):
                raise
            raise RuntimeError(f"{Invoke.NPU_PREPROCESS_FAILURE} {exc}") from exc
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} TestSpec.npu_preprocess must return None or dict{{param_name: tensor}}"
            )
        for name, tensor in result.items():
            if (
                not isinstance(name, str)
                or tensor is None
                or not hasattr(tensor, "shape")
                or not hasattr(tensor, "device")
            ):
                raise RuntimeError(f"{Invoke.NPU_PREPROCESS_FAILURE} returned slot '{name}' is not a tensor")
        return result


def _tensor_param_bindings(testcase, plan, args, kwargs):
    """Map tensor parameter names to flat slots and their built call locations."""
    top_slots = []
    flat_idx = 0
    for shape in testcase.tensor_view_shapes or ():
        count = len(shape) if shape and isinstance(shape[0], (tuple, list)) else 1
        top_slots.append(tuple(range(flat_idx, flat_idx + count)))
        flat_idx += count

    bindings = {}
    for name, entries in plan.tensor_param_bindings(testcase.tensor_view_shapes or ()).items():
        if len(entries) != 1:
            # A named *args region cannot be addressed by one dict key.
            top_index, container, key, _is_vararg = entries[0]
            bindings[name] = (top_slots[top_index], container, key, True)
            continue
        top_index, container, key, is_vararg = entries[0]
        bindings[name] = (top_slots[top_index], container, key, is_vararg)
    return bindings


def _tensor_device(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _tensor_device(item)
            if device is not None:
                return device
        return None
    return getattr(value, "device", None)


def apply_npu_preprocess_result(testcase, plan, args, kwargs, returned):
    """Patch device-side call arguments materialized by npu_preprocess（E2E 动态槽回填）."""
    if returned is None:
        return []
    bindings = _tensor_param_bindings(testcase, plan, args, kwargs)
    dyn_indexes = set(testcase.dyn_input_slot_indexes)

    dynamic_names = {
        name
        for name, (slots, _container, _key, is_vararg) in bindings.items()
        if not is_vararg and len(slots) == 1 and slots[0] in dyn_indexes
    }
    for name in returned:
        binding = bindings.get(name)
        if binding is None or binding[3] or len(binding[0]) != 1 or binding[0][0] not in dyn_indexes:
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} returned slot '{name}' is not declared as a supported -1 slot"
            )
    missing = sorted(dynamic_names - set(returned))
    if missing:
        raise RuntimeError(
            f"{Invoke.NPU_PREPROCESS_FAILURE} dynamic slots {missing} not materialized by npu_preprocess"
        )
    mapped_dyn = {
        slots[0] for slots, _container, _key, is_vararg in bindings.values() if not is_vararg and len(slots) == 1
    }
    unmapped = sorted(dyn_indexes - mapped_dyn)
    if unmapped:
        raise RuntimeError(
            "NPU_PREPROCESS_FAILURE: dynamic slot(s) "
            f"{unmapped} have no supported API tensor parameter (TensorList and *args slots are unsupported)"
        )

    target_device = _tensor_device(args) or _tensor_device(tuple(kwargs.values()))
    patched = []
    for name, tensor in returned.items():
        returned_device = _tensor_device(tensor)
        if target_device is not None and returned_device is not None and str(returned_device) != str(target_device):
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} returned slot '{name}' is on {returned_device}, expected {target_device}"
            )
        slots, container, key, _is_vararg = bindings[name]
        if container == "args":
            args[key] = tensor
        else:
            kwargs[key] = tensor
        patched.append((slots[0], name, tensor))

    logging.info(
        "[%s] npu_preprocess resolved dynamic shapes: %s",
        getattr(testcase, "testcase_name", ""),
        {name: tuple(tensor.shape) for _idx, name, tensor in patched},
    )
    return patched


def _aclnn_tensor_param_bindings(testcase, plan):
    """Map ACLNN tensor parameter names to top-level and flat testcase slots."""
    bindings = {}
    top = 0
    flat = 0
    for kind, name, _acl_type, _default in plan.param_layout:
        if kind != "tensor":
            continue
        shape = testcase.tensor_view_shapes[top]
        count = len(shape) if shape and isinstance(shape[0], (tuple, list)) else 1
        bindings[name] = (top, tuple(range(flat, flat + count)))
        top += 1
        flat += count
    return bindings


def apply_aclnn_npu_preprocess_result(testcase, plan, returned):
    """Materialize dynamic ACLNN inputs on host after their NPU preprocess（动态槽主机侧物化）."""
    if returned is None:
        return []

    import torch

    from ...utilities import str_to_torch_dtype

    bindings = _aclnn_tensor_param_bindings(testcase, plan)
    dyn_indexes = set(testcase.dyn_input_slot_indexes)
    dynamic_names = {name for name, (_top, slots) in bindings.items() if len(slots) == 1 and slots[0] in dyn_indexes}
    for name in returned:
        binding = bindings.get(name)
        if binding is None or len(binding[1]) != 1 or binding[1][0] not in dyn_indexes:
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} returned ACLNN slot '{name}' is not declared as a supported -1 slot"
            )
    missing = sorted(dynamic_names - set(returned))
    if missing:
        raise RuntimeError(f"{Invoke.NPU_PREPROCESS_FAILURE} dynamic ACLNN slots {missing} not materialized")
    mapped_dyn = {slots[0] for _top, slots in bindings.values() if len(slots) == 1}
    if dyn_indexes - mapped_dyn:
        raise RuntimeError("NPU_PREPROCESS_FAILURE: dynamic ACLNN slot inside TensorList is not supported")

    dynamic_tensors = {}
    patched = []
    for name, tensor in returned.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{Invoke.NPU_PREPROCESS_FAILURE} returned ACLNN slot '{name}' is not a torch.Tensor")
        top, slots = bindings[name]
        flat = slots[0]
        host_tensor = tensor.detach().to("cpu").contiguous()
        expected_dtype = str_to_torch_dtype(testcase.flat_tensor_dtypes[flat])
        if expected_dtype is None or host_tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"{Invoke.NPU_PREPROCESS_FAILURE} returned ACLNN slot '{name}' has dtype {host_tensor.dtype}, "
                f"expected {expected_dtype}"
            )
        dynamic_tensors[flat] = host_tensor
        patched.append((flat, name, host_tensor))

    # Keep the CPU-side None marker intact.  ACLNN needs a host copy only to
    # create its device tensor; golden, dump, and third-party paths must not
    # mistake a NPU-derived metadata buffer for a user input.
    testcase._aclnn_dynamic_tensors = dynamic_tensors
    logging.info(
        "[%s] npu_preprocess resolved ACLNN dynamic shapes: %s",
        getattr(testcase, "testcase_name", ""),
        {name: tuple(tensor.shape) for _idx, name, tensor in patched},
    )
    return patched
