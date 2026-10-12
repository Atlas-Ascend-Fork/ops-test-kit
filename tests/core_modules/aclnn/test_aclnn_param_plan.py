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
Tests for AclnnApiParamResolver and TestcaseAclnn.get_param_plan().
"""

from collections import OrderedDict

from ttk.core_modules.aclnn.api_param import AclnnApiParamResolver
from ttk.core_modules.aclnn.info_keeper import OpApiInfo


def _make_op_api_info(params_dict):
    return OpApiInfo(params=OrderedDict(params_dict))


class TestAclnnApiParamResolverInit:
    """Tests for AclnnApiParamResolver initialization — param_layout / tensor_count / scalar_count."""

    def test_tensor_only(self):
        """纯 aclTensor* 参数 → tensor_count == len(params)，scalar_count == 0。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("y", {"type": "aclTensor*", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnAdd", info)
        assert plan.tensor_count == 2
        assert plan.scalar_count == 0
        assert len(plan.param_layout) == 2
        assert plan.param_layout[0][0] == "tensor"
        assert plan.param_layout[1][0] == "tensor"

    def test_tensor_and_scalar(self):
        """aclTensor* 与 aclScalar* 交错 → layout 保留声明顺序。"""
        info = _make_op_api_info(
            [
                ("self", {"type": "aclTensor*", "default": None}),
                ("dim", {"type": "aclScalar*", "default": None}),
                ("out", {"type": "aclTensor*", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnSoftmax", info)
        assert plan.tensor_count == 2
        assert plan.scalar_count == 1
        assert plan.param_layout[0] == ("tensor", "self", "aclTensor*", None)
        assert plan.param_layout[1] == ("scalar", "dim", "aclScalar*", None)
        assert plan.param_layout[2] == ("tensor", "out", "aclTensor*", None)

    def test_with_attributes(self):
        """非 Tensor/Scalar 类型（如 float/aclDataType）→ 归入 'other' (attribute) 桶。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("alpha", {"type": "float", "default": "1.0"}),
                ("dtype", {"type": "aclDataType", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnCast", info)
        assert plan.tensor_count == 1
        assert plan.scalar_count == 0
        assert len(plan.param_layout) == 3
        assert plan.param_layout[1][0] == "other"
        assert plan.param_layout[1][1] == "alpha"
        assert plan.param_layout[2][0] == "other"
        assert plan.param_layout[2][1] == "dtype"

    def test_tensor_list(self):
        """aclTensorList* → 计入 tensor_count，layout 保留 TensorList 类型。"""
        info = _make_op_api_info(
            [
                ("tensors", {"type": "aclTensorList*", "default": None}),
                ("dim", {"type": "int64_t", "default": "0"}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnCat", info)
        assert plan.tensor_count == 1
        assert plan.param_layout[0] == ("tensor", "tensors", "aclTensorList*", None)

    def test_scalar_list(self):
        """aclScalarList* → 计入 scalar_count。"""
        info = _make_op_api_info(
            [
                ("self", {"type": "aclTensor*", "default": None}),
                ("scalarList", {"type": "aclScalarList*", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnAddScalarList", info)
        assert plan.tensor_count == 1
        assert plan.scalar_count == 1
        assert plan.param_layout[1][0] == "scalar"

    def test_full_signature(self):
        """混合 Tensor/Scalar/attribute 的完整 Conv2d-like 签名。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("weight", {"type": "aclTensor*", "default": None}),
                ("bias", {"type": "aclTensor*", "default": None}),
                ("scale", {"type": "aclScalar*", "default": None}),
                ("groups", {"type": "int64_t", "default": "1"}),
                ("format", {"type": "aclDataType", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnConv2d", info)
        assert plan.tensor_count == 3
        assert plan.scalar_count == 1
        assert len(plan.param_layout) == 6


class TestAclnnApiParamResolverBuildArgs:
    """Tests for AclnnApiParamResolver.build_args — 位置参数组装与 attribute 提取。"""

    def test_tensor_only_args(self):
        """纯 Tensor 参数 → args 即输入 tensors，extra 为空。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("y", {"type": "aclTensor*", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnAdd", info)
        tensors = ["T_X", "T_Y"]
        args, extra = plan.build_args(tensors, [], {})
        assert args == ["T_X", "T_Y"]
        assert extra == {}

    def test_interleaved_tensor_scalar_attr(self):
        """Tensor 与 Scalar 交错 → args 按声明顺序插入 scalar。"""
        info = _make_op_api_info(
            [
                ("self", {"type": "aclTensor*", "default": None}),
                ("dim", {"type": "aclScalar*", "default": None}),
                ("out", {"type": "aclTensor*", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnSoftmax", info)
        tensors = ["T_SELF", "T_OUT"]
        scalars = ["S_DIM"]
        args, extra = plan.build_args(tensors, scalars, {})
        assert args == ["T_SELF", "S_DIM", "T_OUT"]
        assert extra == {}

    def test_attribute_from_dict(self):
        """attribute 从 attrs dict 取值，按声明顺序追加到 args。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("alpha", {"type": "float", "default": "1.0"}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnAddAlpha", info)
        args, extra = plan.build_args(["T_X"], [], {"alpha": 2.5})
        assert args == ["T_X", 2.5]
        assert extra == {}

    def test_attribute_default_fallback(self):
        """attribute 未在 attrs 提供 → 使用 default。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("alpha", {"type": "float", "default": "1.0"}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnAddAlpha", info)
        args, extra = plan.build_args(["T_X"], [], {})
        assert args == ["T_X", "1.0"]
        assert extra == {}

    def test_attribute_missing_no_default(self):
        """attribute 无 default 且未提供 → None。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("dtype", {"type": "aclDataType", "default": None}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnCast", info)
        args, extra = plan.build_args(["T_X"], [], {})
        assert args == ["T_X", None]
        assert extra == {}

    def test_full_conv2d_like(self):
        """完整 Conv2d-like 签名：Tensor + Scalar + attribute 混合。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("weight", {"type": "aclTensor*", "default": None}),
                ("bias", {"type": "aclTensor*", "default": None}),
                ("scale", {"type": "aclScalar*", "default": None}),
                ("groups", {"type": "int64_t", "default": "1"}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnConv2d", info)
        tensors = ["T_X", "T_W", "T_B"]
        scalars = ["S_SCALE"]
        attrs = {"groups": 4}
        args, extra = plan.build_args(tensors, scalars, attrs)
        assert args == ["T_X", "T_W", "T_B", "S_SCALE", 4]
        assert extra == {}

    def test_tensor_list_preserved(self):
        """TensorList 作为单一位置参数保留列表结构。"""
        info = _make_op_api_info(
            [
                ("tensors", {"type": "aclTensorList*", "default": None}),
                ("dim", {"type": "int64_t", "default": "0"}),
            ]
        )
        plan = AclnnApiParamResolver("aclnnCat", info)
        tensor_list = [["T_A", "T_B", "T_C"]]
        args, extra = plan.build_args(tensor_list, [], {"dim": 1})
        assert args[0] == ["T_A", "T_B", "T_C"]
        assert args[1] == 1
        assert extra == {}


class TestGetParamPlan:
    """Tests for TestcaseAclnn.get_param_plan() — 缓存与 None 兜底。"""

    def test_plan_cached(self):
        """已缓存的 _param_plan_cache → 直接返回，不重新构建。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
            ]
        )
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn

        case = TestcaseAclnn()
        case.api_name = "aclnnAdd"
        plan = AclnnApiParamResolver("aclnnAdd", info)
        case._param_plan_cache = plan
        assert case.get_param_plan() is plan
        assert case.get_param_plan() is plan

    def test_plan_none_for_no_api(self):
        """api_name 为 None → get_param_plan 返回 None。"""
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn

        case = TestcaseAclnn()
        case.api_name = None
        assert case.get_param_plan() is None

    def test_plan_build_args_integration(self):
        """get_param_plan + build_args 端到端集成。"""
        info = _make_op_api_info(
            [
                ("x", {"type": "aclTensor*", "default": None}),
                ("alpha", {"type": "float", "default": "1.0"}),
            ]
        )
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn

        case = TestcaseAclnn()
        case.api_name = "aclnnAddAlpha"
        plan = AclnnApiParamResolver("aclnnAddAlpha", info)
        case._param_plan_cache = plan
        args, extra = plan.build_args(["T_X"], [], {"alpha": 3.0})
        assert args == ["T_X", 3.0]
