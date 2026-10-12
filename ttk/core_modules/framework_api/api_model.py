#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
API 签名数据模型 —— torch / tf 两个参数解析器的共用契约。

ParamInfo / OverloadInfo / APIParamInfo 由 torch / tf 参数解析器与
api_param（ParamPlan/match_overload）共同消费，独立成模块、不寄居在
单一框架实现文件内。

resolve_function 惰性 import torch；torch/param_extractor 的 schema 解析
与测试侧均从本模块引用。

ParamInfo.is_mutable / is_resource —— 参数级可变性（接口确定即可知的静态事实，
与 is_keyword_only 同类）：该参数槽会被算子原地修改。共享一个字段、两框架按
各自元数据填充（torch: aten schema alias_info.is_write / 尾下划线命名投影；
tf: OpDef is_ref / DT_RESOURCE，见 tf/param_extractor）。输出契约（返回被改值
vs 无返回需回读）不落参数级——运行期按 res is None 分派。
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ParamInfo:
    name: str
    type: str = ""
    default: Any = None
    is_optional: bool = False
    is_keyword_only: bool = False
    is_var_positional: bool = False
    is_mutable: bool = False
    is_resource: bool = False

    @property
    def is_tensor(self):
        """True if any member type is a single Tensor."""
        return any(t in ("Tensor", "tensor") for t in self.type.split("|"))

    @property
    def is_tensor_list(self):
        """True if any member type is a sequence of Tensors."""
        return any(
            t in ("tuple of Tensors", "list of Tensors", "Tensor[]", "List[Tensor]") for t in self.type.split("|")
        )

    @property
    def is_scalar(self):
        """True if any member type is a scalar (Number, int, float, bool, str)."""
        return any(t in ("Number", "Scalar", "int", "float", "bool", "str") for t in self.type.split("|"))

    @property
    def is_scalar_list(self):
        """True if any member type is a list of scalars."""
        return any(t in ("Scalar[]", "List[Scalar]", "list of Numbers") for t in self.type.split("|"))

    @property
    def is_tensor_like(self):
        """True if the param accepts Tensor data (single tensor or tensor list)."""
        return self.is_tensor or self.is_tensor_list


@dataclass
class OverloadTensorLayout:
    """Tensor parameter layout for a single overload, split by input/out role."""

    input_params: List[ParamInfo]
    required_input_count: int
    input_count: int
    has_var_input: bool
    out_param: Optional[ParamInfo]
    is_out_required: bool
    is_out_tensor_list: bool
    out_expected_count: int

    @staticmethod
    def build(overload_params: List[ParamInfo], return_count: int = 0) -> "OverloadTensorLayout":
        input_params = [p for p in overload_params if p.is_tensor_like and p.name != "out"]
        has_var_input = any(getattr(p, "is_var_positional", False) for p in input_params)
        required_input_count = sum(
            1 for p in input_params if not p.is_optional and not getattr(p, "is_var_positional", False)
        )

        out_param = next((p for p in overload_params if p.name == "out" and p.is_tensor_like), None)
        is_out_required = out_param is not None and not out_param.is_optional
        is_out_tensor_list = out_param.is_tensor_list if out_param else False

        if out_param is None:
            out_expected_count = 0
        elif is_out_tensor_list:
            out_expected_count = return_count if return_count > 0 else 0
        else:
            out_expected_count = 1

        return OverloadTensorLayout(
            input_params=input_params,
            required_input_count=required_input_count,
            input_count=len(input_params),
            has_var_input=has_var_input,
            out_param=out_param,
            is_out_required=is_out_required,
            is_out_tensor_list=is_out_tensor_list,
            out_expected_count=out_expected_count,
        )


@dataclass
class OverloadInfo:
    """Single overload: its parameter list and pre-computed tensor layout."""

    params: List[ParamInfo]
    return_count: int = 0
    layout: OverloadTensorLayout = field(init=False, repr=False)

    def __post_init__(self):
        self.layout = OverloadTensorLayout.build(self.params, self.return_count)


def _normalize_to_overload_infos(overloads, return_counts=None):
    """Convert List[List[ParamInfo]] or List[OverloadInfo] → List[OverloadInfo]."""
    if return_counts is None:
        return_counts = []
    result = []
    for i, ov in enumerate(overloads):
        if isinstance(ov, OverloadInfo):
            result.append(ov)
        elif isinstance(ov, list):
            rc = return_counts[i] if i < len(return_counts) else 0
            result.append(OverloadInfo(params=ov, return_count=rc))
    return result


@dataclass
class APIParamInfo:
    api_name: str
    params: List[ParamInfo] = field(default_factory=list)
    source: str = ""
    overloads: list = field(default_factory=list)
    is_tensor_method: bool = False
    _return_counts: List[int] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.overloads:
            self.overloads = _normalize_to_overload_infos(self.overloads, self._return_counts)
            best = max(self.overloads, key=lambda oi: sum(1 for p in oi.params if p.is_optional))
            self.params = best.params
        elif self.params and not self.overloads:
            self.overloads = _normalize_to_overload_infos([self.params], self._return_counts)

    def _resolve_api_flags(self):
        """torch 侧解析标志（tf 侧不经过本入口，is_mutable 在 extractor 内填充）。"""
        parts = self.api_name.split(".") if self.api_name else []
        self.is_tensor_method = len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor"
        func_name = parts[-1] if parts else ""
        if not (func_name.endswith("_") and not func_name.startswith("_")):
            return
        # 尾下划线命名约定（torch inplace）：schema 源已在 _parse_aten_schema_arguments
        # 按 alias_info.is_write 逐参数落 is_mutable；非 schema 源（docstring/pyi）或
        # 解析降级时投影到首张量参数（与既有"槽 0"惯例等价）。逐 overload 投影——
        # match_overload 选中的不一定是 best overload。
        for ov in self.overloads or ():
            if any(p.is_mutable for p in ov.params):
                continue
            tensor_params = [i for i, p in enumerate(ov.params) if p.is_tensor_like]
            if tensor_params:
                ov.params[tensor_params[0]].is_mutable = True

    @property
    def tensors(self):
        return [p for p in self.params if p.is_tensor_like]

    @property
    def scalars(self):
        return [p for p in self.params if p.is_scalar or p.is_scalar_list]

    @property
    def tensor_count(self):
        return len(self.tensors)

    @property
    def scalar_count(self):
        return len(self.scalars)

    @property
    def tensor_distribution(self):
        """Return distribution tuple: -1 for TensorList, 0 for single Tensor."""
        dist = []
        for p in self.tensors:
            if p.is_tensor_list:
                dist.append(-1)
            else:
                dist.append(0)
        return tuple(dist)

    def match_overload(self, tensor_count, nested_flags=None, skip_flags=None):
        """Check if testcase matches ANY overload signature.

        Args:
            tensor_count: Number of top-level input tensors (excluding out).
            nested_flags: Optional list of bools — True if position is nested (TensorList).
            skip_flags: Optional list of bools — True to skip type check at that position
                        (used for None elements which match any type).
        Returns:
            (matched, input_params_list, overload_index) or (False, None, -1).
        """
        for oidx, ov in enumerate(self.overloads):
            layout = ov.layout
            if layout.has_var_input:
                if tensor_count < layout.required_input_count:
                    continue
            elif tensor_count < layout.required_input_count or tensor_count > layout.input_count:
                continue
            if nested_flags is not None:
                type_ok = True
                non_var = [p for p in layout.input_params if not getattr(p, "is_var_positional", False)]
                for idx in range(tensor_count):
                    if skip_flags and idx < len(skip_flags) and skip_flags[idx]:
                        continue
                    if idx < len(non_var):
                        param = non_var[idx]
                    else:
                        continue
                    is_nested = nested_flags[idx]
                    if param.is_tensor_list and not is_nested:
                        type_ok = False
                        break
                    if param.is_tensor and is_nested:
                        type_ok = False
                        break
                if not type_ok:
                    continue
            return True, layout.input_params, oidx
        return False, None, -1


def resolve_function(api_name: str):
    try:
        parts = api_name.split(".")
        if len(parts) < 2:
            return None
        if api_name.startswith("torch.ops."):
            import torch

            if len(parts) >= 4:
                namespace = parts[2]
                op_name = parts[3]
                packet = getattr(torch.ops, namespace).__getattr__(op_name)
                # api_name 钉死了具名重载(如 ...._foreach_addcdiv.Tensor)时取该重载:
                # OpOverloadPacket 只在存在 default 重载时才有 _schema, 全具名重载的
                # 算子(Scalar/ScalarList/Tensor)取不到, 会导致形参解析为空。
                if len(parts) >= 5 and packet is not None:
                    overload = getattr(packet, parts[4], None)
                    if overload is not None:
                        return overload
                return packet
        if len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor":
            import torch

            method_name = ".".join(parts[2:])
            return getattr(torch.Tensor, method_name, None)
        mod_name = ".".join(parts[:-1])
        func_name = parts[-1]
        import importlib

        mod = importlib.import_module(mod_name)
        return getattr(mod, func_name, None)
    except (ImportError, AttributeError):
        return None


def split_by_comma(s: str) -> List[str]:
    parts = []
    current = []
    depth = 0
    for char in s:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts
