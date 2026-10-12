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
Parameter plan (E2E): overload matching, value coercion, and argument assembly.

framework_api 的 Python 签名参数机器，对 torch / tf 参数解析器产出的
ParamInfo/APIParamInfo 契约（api_model.py）做重载匹配与调用参数装配。
match_overload / coerce_value 已上提至 common/testcase/api_param.py
（公开面），此处经 import 复用并 re-export；aclnn 不走本模块：
C 签名变体见 aclnn/api_param.py，共用 ABC 契约 ResolveApiParamBase 与
match_overload/coerce_value 见 common/testcase/api_param.py。

Usage:
    plan = ParamPlan(api_name, overload_params, oidx, output_tensor_indexes, attributes)
    args, kwargs, extra = plan.build_args(nested_tensors)
"""

import logging

from ttk.core_modules.common.testcase.api_param import coerce_value, match_overload

__all__ = [
    "ParamPlan",
    "build_positional_args",
    "coerce_value",
    "match_overload",
]


class ParamPlan:
    """Cached parameter assembly plan — resolved once, reused by profiling/golden/input.

    Decides overload matching and param layout based on testcase configuration
    (tensor count, attributes, output_tensor_indexes, distribution).
    The actual tensor objects (numpy vs device tensor) are plugged in at build time.

    Usage:
        plan = testcase.get_param_plan()
        args, kwargs, _ = plan.build_args(nested_tensors)
        result = resolved_api(*args, **kwargs)
    """

    __slots__ = (
        "api_name",
        "overload_params",
        "overload_index",
        "output_tensor_indexes",
        "attributes",
    )

    def __init__(self, api_name, overload_params, overload_index, output_tensor_indexes, attributes):
        self.api_name = api_name
        self.overload_params = overload_params
        self.overload_index = overload_index
        self.output_tensor_indexes = output_tensor_indexes
        self.attributes = attributes

    def build_args(self, nested_tensors):
        """Assemble (*args, **kwargs) from tensors using this plan.

        Tensor positions are filled from nested_tensors (any type: numpy, torch, etc).
        Scalar positions use values from attributes/defaults, coerced to correct types.

        Returns:
            (args_list, kwargs_dict, extra_attrs) where extra_attrs contains
            attributes not matched by any API parameter name.
        """
        args, kwargs, extra_attrs, _bindings = self._build_args(nested_tensors)
        return args, kwargs, extra_attrs

    def tensor_param_bindings(self, nested_tensors):
        """Return tensor parameter bindings produced by the normal argument builder.

        Each value is a list of ``(top_index, container, key, is_vararg)``
        entries.  Keeping this alongside ``build_args`` prevents dynamic-slot
        patching from maintaining a second copy of the parameter-consumption
        rules.
        """
        _args, _kwargs, _extra_attrs, bindings = self._build_args(nested_tensors)
        return bindings

    def _place_out_param(self, param, out_iter, kwargs, args, record_binding):
        """'out' 参数：从输出张量迭代器取值，落 kwargs；无值且非 kw-only 时占位 None。"""
        if param.is_tensor_list:
            collected_items = [(i, t) for i, t in out_iter if t is not None]
            collected = [t for _i, t in collected_items]
            if collected:
                kwargs["out"] = collected
                for top_index, _value in collected_items:
                    record_binding(param.name, top_index, "kwargs", "out")
            elif not param.is_keyword_only:
                args.append(None)
        else:
            item = next(out_iter, None)
            top_index, val = item if item is not None else (None, None)
            if val is not None:
                kwargs["out"] = val
                record_binding(param.name, top_index, "kwargs", "out")
            elif not param.is_keyword_only:
                args.append(None)

    def _place_keyword_only_param(self, param, tensor_queue, attrs, kwargs, record_binding):
        """kw-only 参数：张量队列优先，其次标量属性（raw 透传），最后默认值。"""
        if param.is_tensor_like:
            if tensor_queue:
                top_index, value = tensor_queue.pop(0)
                kwargs[param.name] = value
                record_binding(param.name, top_index, "kwargs", param.name)
            elif param.name in attrs:
                # Tensor params beyond the shape-covered prefix may be
                # satisfied by scalar attributes (match_overload
                # scalar_cover). raw_ops attr params follow the
                # raw_ops convention of classifying every non-'name'
                # param as tensor. Values are passed raw — param.type
                # is "Tensor" and coercion would fail.
                kwargs[param.name] = attrs[param.name]
        elif param.name in attrs:
            kwargs[param.name] = coerce_value(attrs[param.name], param.type)
        elif param.default is not None:
            val = coerce_value(param.default, param.type)
            kwargs[param.name] = val

    def _place_tensor_param(self, param, tensor_queue, attrs, args, record_binding):
        """位置张量参数：变参吸收 / 标量回退（含 Number 强转与原值透传）/ 队列出队 / 占位。"""
        if getattr(param, "is_var_positional", False):
            for top_index, value in tensor_queue:
                arg_index = len(args)
                args.append(value)
                record_binding(param.name, top_index, "args", arg_index, is_vararg=True)
            tensor_queue.clear()
        elif param.name in attrs and param.name != "self" and not tensor_queue:
            raw = attrs[param.name]
            try:
                args.append(coerce_value(raw, param.type))
            except (ValueError, TypeError):
                logging.warning(
                    f"{self.api_name}: scalar fallback for param '{param.name}' "
                    f"(declared type={param.type}, value={raw!r})"
                )
                try:
                    args.append(coerce_value(raw, "Number"))
                except (ValueError, TypeError):
                    # TF 无注解签名里的枚举/字符串参数(如 dequantize 的 mode)会被
                    # 归类为 Tensor，数值强转也失败时按原值透传，由 API 自行校验
                    args.append(raw)
        elif tensor_queue:
            top_index, val = tensor_queue.pop(0)
            if param.is_tensor and isinstance(val, list) and len(val) == 1:
                val = val[0]
            arg_index = len(args)
            args.append(val)
            record_binding(param.name, top_index, "args", arg_index)
        elif param.is_optional:
            args.append(None)
        else:
            raise ValueError(
                f"{self.api_name}: not enough tensors for param '{param.name}' "
                f"(queue empty, {len(args)} args built so far)"
            )

    def _build_args(self, nested_tensors):
        """Build call arguments and record where each tensor parameter lands."""
        out_indices = set(self.output_tensor_indexes or ())
        from ttk.core_modules.common.backend import is_inplace_tensor_method

        is_inplace = is_inplace_tensor_method(self.api_name) if self.api_name else False
        if is_inplace:
            input_tensors = list(enumerate(nested_tensors))
        else:
            input_tensors = [(i, t) for i, t in enumerate(nested_tensors) if i not in out_indices]
        out_tensors = [(i, nested_tensors[i]) for i in sorted(out_indices)]
        out_iter = iter(out_tensors)

        tensor_queue = list(input_tensors)
        attrs = dict(self.attributes) if self.attributes else {}
        args = []
        kwargs = {}
        bindings = {}

        def record_binding(param_name, top_index, container, key, is_vararg=False):
            bindings.setdefault(param_name, []).append((top_index, container, key, is_vararg))

        for param in self.overload_params:
            if param.is_tensor_like and param.name == "out":
                self._place_out_param(param, out_iter, kwargs, args, record_binding)
            elif param.is_keyword_only:
                self._place_keyword_only_param(param, tensor_queue, attrs, kwargs, record_binding)
            elif param.is_tensor_like:
                self._place_tensor_param(param, tensor_queue, attrs, args, record_binding)
            elif param.name in attrs:
                args.append(coerce_value(attrs[param.name], param.type))
            elif param.default is not None:
                args.append(coerce_value(param.default, param.type))
            else:
                args.append(None)

        param_names = {p.name for p in self.overload_params}
        extra_attrs = {k: v for k, v in attrs.items() if k not in param_names}
        return args, kwargs, extra_attrs, bindings


def build_positional_args(
    api_name, nested_tensors, attributes, output_tensor_indexes, tensor_distribution=None, api_info=None
):
    """Build (positional_args, kwargs) based on matched API signature.

    Convenience wrapper — creates a one-shot ParamPlan.
    For repeated calls (profiling/golden), prefer testcase.get_param_plan() + plan.build_args().

    Args:
        api_info: **Required**. APIParamInfo for the target API.

    Returns:
        (args_list, kwargs_dict, matched_overload_index) or raises ValueError
    """
    out_set = set(output_tensor_indexes or ())
    # 对齐 get_param_plan 口径（testcase.py 剔声明 None 槽）：None 槽是省略的
    # 可选张量占位，不计入输入张量数——否则 golden_api 路径的重载匹配会因
    # 计数偏大而错配或失败，与主路径选出不同重载
    input_tensor_count = sum(1 for i, t in enumerate(nested_tensors) if i not in out_set and t is not None)
    overload_params, oidx = match_overload(
        api_name,
        input_tensor_count=input_tensor_count,
        attributes=attributes,
        tensor_distribution=tensor_distribution,
        api_info=api_info,
    )
    if overload_params is None:
        raise ValueError(f"Cannot match overload for {api_name} with {input_tensor_count} input tensors")

    plan = ParamPlan(api_name, overload_params, oidx, output_tensor_indexes, attributes)
    args, kwargs, _ = plan.build_args(nested_tensors)
    return args, kwargs, oidx
