#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn parameter assembly: C header signature order.

Unlike framework_api's Python-signature plan (overloads, keyword-only
arguments, ``out=`` separation), aclnn exposes a single C function with
all-positional parameters. The resolver captures the ordered layout once
and assembles arguments from the testcase's tensor/scalar/attribute
queues on every call.
"""

__all__ = ["AclnnApiParamResolver", "AclnnParamPlan"]


# Standard Packages
from typing import Any, List, Tuple

# Third-Party Packages
from ..common.testcase import ResolveApiParamBase
from .info_keeper import OpApiInfo


class AclnnApiParamResolver(ResolveApiParamBase):
    """Cached parameter assembly plan for aclnn (op_api) testcases.

    Single source of truth for param layout, resolved once per testcase,
    reused by custom input, custom golden, and profiling.
    """

    TENSOR = "tensor"
    SCALAR = "scalar"
    OTHER = "other"

    __slots__ = (
        "api_name",
        "param_layout",
        "tensor_count",
        "scalar_count",
    )

    def __init__(self, api_name: str, op_api_info: OpApiInfo):
        self.api_name = api_name
        self.param_layout: List[Tuple[str, str, str, Any]] = []
        self.tensor_count = 0
        self.scalar_count = 0

        for param_name, param_info in op_api_info.params.items():
            acl_type = param_info["type"]
            default = param_info["default"]
            if "aclTensor" in acl_type:
                self.param_layout.append((self.TENSOR, param_name, acl_type, default))
                self.tensor_count += 1
            elif "aclScalar" in acl_type:
                self.param_layout.append((self.SCALAR, param_name, acl_type, default))
                self.scalar_count += 1
            else:
                self.param_layout.append((self.OTHER, param_name, acl_type, default))

    def resolve_args(self, tensors, scalars, attributes) -> tuple:
        """Assemble positional args in C header signature order.

        Iterates param_layout, consuming tensors and scalars from their
        respective queues, and looking up attribute values.

        Args:
            tensors: nested tensor list (from context.tensors)
            scalars: nested scalar list (from context.scalars)
            attributes: dict of API parameters

        Returns:
            (args_list, (), extra_attrs) — args in C header signature order;
            extra_attrs contains attributes not matched by any param name.
        """
        args = []
        tensor_queue = list(tensors)
        scalar_queue = list(scalars)
        param_names = set()
        for kind, name, _acl_type, default in self.param_layout:
            param_names.add(name)
            if kind == self.TENSOR:
                args.append(tensor_queue.pop(0))
            elif kind == self.SCALAR:
                args.append(scalar_queue.pop(0))
            else:
                args.append(attributes.get(name, default))
        extra_attrs = {k: v for k, v in attributes.items() if k not in param_names}
        return args, (), extra_attrs

    def build_args(self, tensors, scalars, attributes):
        """Two-tuple compatibility view of :meth:`resolve_args`."""
        args, _, extra_attrs = self.resolve_args(tensors, scalars, attributes)
        return args, extra_attrs


# Compatibility alias for the historical plan name.
AclnnParamPlan = AclnnApiParamResolver
