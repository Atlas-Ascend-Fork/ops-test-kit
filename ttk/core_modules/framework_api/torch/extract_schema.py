#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""aten 算子注册表（_schemas）参数解析源。"""

from typing import List, Optional

from ..api_model import APIParamInfo, ParamInfo, resolve_function
from .extract_declaration import _normalize_npu_type


def _extract_params_from_aten_schemas(
    api_name: str,
) -> Optional[APIParamInfo]:
    """从 torch 算子注册表解析（链上第一优先级源，FunctionSchema 权威元数据）。

    OpOverloadPacket 带复数 _schemas；api_name 钉死具名重载时 resolve_function 返回
    OpOverload 只有单数 _schema（不取则退 inspect 得 (*args, **kwargs)）。钉死重载时
    只收该重载——防整个 packet 混入 *_out 变体（is_out_required 会让按返回值取输出的
    用例误判 MISSING_REQUIRED_OUTPUT）。返回 APIParamInfo。
    """
    obj = resolve_function(api_name)
    if obj is None:
        return None
    if hasattr(obj, "_schemas"):
        schemas = obj._schemas
    elif hasattr(obj, "_schema"):
        # api_name 钉死具名重载时 resolve_function 返回 OpOverload, 它只有单数 _schema;
        # 不取这一支会退到 inspect.signature, 而 OpOverload.__call__ 的签名是
        # (*args, **kwargs), 只能解析出一个 "args", 下游按形参名派发时长度对不上。
        schemas = {api_name.rsplit(".", 1)[-1]: obj._schema}
    else:
        return None
    if not schemas:
        return None
    # api_name 若钉死了具体重载(如 torch.ops.aten._foreach_addcmul.Tensor)，只解析该重载。
    # resolve_function 会退回到 OpOverloadPacket(丢掉重载后缀)，若不在此过滤，整个
    # packet 的重载都会被收进来——其中的 *_out 变体带 is_out_required=True，会让本来
    # 靠返回值取输出的用例被误判为 MISSING_REQUIRED_OUTPUT；多重载并存也会导致
    # CASE_FIELD_AMBIGUOUS。
    overload_tail = api_name.rsplit(".", 1)[-1]
    if overload_tail in schemas:
        schemas = {overload_tail: schemas[overload_tail]}
    all_overloads = []
    return_counts = []
    for schema in schemas.values():
        params = _parse_aten_schema_arguments(schema)
        if params:
            all_overloads.append(params)
            return_counts.append(len(schema.returns) if hasattr(schema, "returns") else 0)
    if not all_overloads:
        return None
    primary = max(all_overloads, key=lambda ov: sum(1 for p in ov if p.is_optional))
    _first_schema = next(iter(schemas.values()))
    namespace = getattr(_first_schema, "name", api_name.rsplit(".", 1)[0]).split("::")[0]
    source = (
        f"{namespace}._schemas({len(all_overloads)} overloads)" if len(all_overloads) > 1 else f"{namespace}._schemas"
    )
    return APIParamInfo(
        api_name=api_name, params=primary, source=source, overloads=all_overloads, _return_counts=return_counts
    )


def _parse_aten_schema_arguments(schema) -> Optional[List[ParamInfo]]:
    """FunctionSchema.arguments → List[ParamInfo]（默认值 / Optional[...] 可选性 / kwarg_only）。

    写别名（alias_info.is_write，如 add_ 的 Tensor(a!) self）逐参数落 is_mutable ——
    参数级可变性与签名同源，解析期即确定。
    """
    params = []
    for arg in schema.arguments:
        name = arg.name
        if not name:
            continue
        type_str = _normalize_aten_schema_type(str(arg.type))
        default = None
        is_optional = False
        if arg.has_default_value():
            default = arg.default_value
            is_optional = True
            if isinstance(default, (list, tuple)):
                default = list(default)
        elif str(arg.type).startswith("Optional["):
            is_optional = True
        is_keyword_only = arg.kwarg_only if hasattr(arg, "kwarg_only") else False
        alias = getattr(arg, "alias_info", None)
        params.append(
            ParamInfo(
                name=name,
                type=type_str,
                default=default,
                is_optional=is_optional,
                is_keyword_only=is_keyword_only,
                is_mutable=bool(alias and alias.is_write),
            )
        )
    return params if params else None


def _normalize_aten_schema_type(t: str) -> str:
    """schema 类型词归一：Optional[...] 剥壳、List[...] → tuple of xxx，余者走 npu 词表。"""
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_aten_schema_type(inner)
    if t.startswith("List[") and t.endswith("]"):
        inner = t[len("List[") : -1]
        inner_mapped = _normalize_aten_schema_type(inner)
        if inner_mapped == "Tensor":
            return "tuple of Tensors"
        return f"tuple of {inner_mapped}s"
    return _normalize_npu_type(t)


def _torch_tensor_type():
    """torch.Tensor;torch 不可用时返回一个不会与任何注解相等的哨兵。"""
    try:
        import torch

        return torch.Tensor
    except ImportError:
        return object()


def _torch_dtype_type():
    """torch.dtype;torch 不可用时返回一个不会与任何注解相等的哨兵。"""
    try:
        import torch

        return torch.dtype
    except ImportError:
        return object()
