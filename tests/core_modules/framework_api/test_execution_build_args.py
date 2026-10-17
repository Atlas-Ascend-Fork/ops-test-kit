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
Tests for ttk.core_modules.framework_api.execution:
_match_overload, _coerce_value, build_positional_args, ParamPlan.build_args,
and ttk.core_modules.framework_api.testcase: get_param_plan.
"""

import importlib.util

import pytest

from ttk.core_modules.framework_api.api_model import ParamInfo
from ttk.core_modules.framework_api.api_param import (
    build_positional_args as _real_build_positional_args,
)
from ttk.core_modules.framework_api.api_param import (
    coerce_value,
    match_overload,
)
from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper
from ttk.core_modules.framework_api.torch.param_extractor import _MANUAL_OVERRIDES

_TORCH_NPU_INSTALLED = importlib.util.find_spec("torch_npu") is not None


@pytest.fixture(autouse=True)
def _clean_keeper():
    FrameworkApiInfoKeeper().clear_cache()
    _MANUAL_OVERRIDES.clear()
    yield
    FrameworkApiInfoKeeper().clear_cache()
    _MANUAL_OVERRIDES.clear()


def _register(api_name, params, source="test"):
    FrameworkApiInfoKeeper().register(api_name, params, source=source)


def _match_overload(api_name, input_tensor_count, attributes=None, tensor_distribution=None):
    api_info = FrameworkApiInfoKeeper().get(api_name)
    return match_overload(api_name, input_tensor_count, attributes, tensor_distribution, api_info)


def _build_positional_args(api_name, nested_tensors, attributes, output_tensor_indexes, tensor_distribution=None):
    api_info = FrameworkApiInfoKeeper().get(api_name)
    return _real_build_positional_args(
        api_name, nested_tensors, attributes, output_tensor_indexes, tensor_distribution, api_info
    )


class TestMatchOverload:
    def test_single_overload_match(self):
        _register(
            "torch.t1",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int"),
            ],
        )
        params, oidx = _match_overload("torch.t1", 1, {"dim": "0"})
        assert params is not None
        assert oidx == 0

    def test_multi_overload_select_by_tensor_count(self):
        _register(
            "torch.t2",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="input", type="Tensor")],
            ],
        )
        params1, oidx1 = _match_overload("torch.t2", 1)
        assert oidx1 == 1
        params2, oidx2 = _match_overload("torch.t2", 2)
        assert oidx2 == 0

    def test_multi_overload_select_by_attribute_score(self):
        _register(
            "torch.t3",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="min", type="Tensor")],
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="min", type="Number")],
            ],
        )
        params, oidx = _match_overload("torch.t3", 1, {"min": "0.5"})
        assert oidx == 1

    def test_no_match_returns_none(self):
        _register(
            "torch.t4",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        params, oidx = _match_overload("torch.t4", 3)
        assert params is None
        assert oidx == -1

    def test_tensor_distribution_filter(self):
        _register(
            "torch.t5",
            [
                [ParamInfo(name="tensors", type="tuple of Tensors")],
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            ],
        )
        params_nested, oidx_nested = _match_overload("torch.t5", 1, tensor_distribution=[True])
        assert oidx_nested == 0
        params_flat, oidx_flat = _match_overload("torch.t5", 2, tensor_distribution=[False, False])
        assert oidx_flat == 1

        # AccumulateNV2 形态：N*T 列表输入 + shape 属性（raw_ops 惯例归 Tensor 位，
        # 由属性值常量折叠覆盖）——列表分布声明 [True] 须命中，单张量误声明须落空
        _register(
            "tf.raw_ops.AccumulateNV2",
            [
                ParamInfo(name="inputs", type="list of Tensors", is_keyword_only=True),
                ParamInfo(name="shape", type="Tensor", is_keyword_only=True),
                ParamInfo(name="name", type="Number", default=None, is_optional=True, is_keyword_only=True),
            ],
        )
        attrs = {"N": 1, "shape": [7302]}
        params, oidx = _match_overload("tf.raw_ops.AccumulateNV2", 1, attrs, tensor_distribution=[True])
        assert params is not None
        assert oidx == 0
        params_flat, oidx_flat = _match_overload("tf.raw_ops.AccumulateNV2", 1, attrs, tensor_distribution=[False])
        assert params_flat is None
        assert oidx_flat == -1

    def test_unknown_api_returns_none(self):
        params, oidx = _match_overload("nonexistent.api", 1)
        assert params is None


class TestScoreAttrTypeCompatibility:
    @pytest.mark.parametrize(
        ("api_suffix", "attr_name", "attr_value", "expected_oidx"),
        [
            ("str", "ord", "fro", 1),
            ("int", "ord", 2, 0),
            ("bool", "flag", True, 1),
            ("tuple", "size", (2, 3), 1),
        ],
        ids=[
            "str-prefers-str-overload",
            "int-prefers-number-overload",
            "bool-prefers-bool-overload",
            "tuple-prefers-tuple-overload",
        ],
    )
    def test_attr_type_prefers_matching_overload(self, api_suffix, attr_name, attr_value, expected_oidx):
        if api_suffix == "str":
            _register(
                "torch.norm_like_str",
                [
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="Number"),
                        ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
                        ParamInfo(name="keepdim", type="bool", default="False", is_optional=True),
                    ],
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="str", default="fro", is_optional=True),
                        ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
                        ParamInfo(name="keepdim", type="bool", default="False", is_optional=True),
                    ],
                ],
            )
            params, oidx = _match_overload(
                "torch.norm_like_str", 1, {attr_name: attr_value, "dim": (-2, -1), "keepdim": False}
            )
        elif api_suffix == "int":
            _register(
                "torch.norm_like_int",
                [
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="Number"),
                        ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
                        ParamInfo(name="keepdim", type="bool", default="False", is_optional=True),
                    ],
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="str", default="fro", is_optional=True),
                        ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
                        ParamInfo(name="keepdim", type="bool", default="False", is_optional=True),
                    ],
                ],
            )
            params, oidx = _match_overload(
                "torch.norm_like_int", 1, {attr_name: attr_value, "dim": (-2, -1), "keepdim": False}
            )
        elif api_suffix == "bool":
            _register(
                "torch.bool_overload_test",
                [
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="str"),
                        ParamInfo(name="dim", type="int"),
                    ],
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="bool"),
                        ParamInfo(name="dim", type="int"),
                    ],
                ],
            )
            params, oidx = _match_overload("torch.bool_overload_test", 1, {attr_name: attr_value, "dim": 0})
        else:
            _register(
                "torch.tuple_overload_test",
                [
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="int"),
                        ParamInfo(name="mode", type="str"),
                    ],
                    [
                        ParamInfo(name="input", type="Tensor"),
                        ParamInfo(name=attr_name, type="tuple of ints"),
                        ParamInfo(name="mode", type="str"),
                    ],
                ],
            )
            params, oidx = _match_overload("torch.tuple_overload_test", 1, {attr_name: attr_value, "mode": "bilinear"})
        assert oidx == expected_oidx

    def test_key_score_takes_priority_over_value_score(self):
        _register(
            "torch.key_priority_test",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="alpha", type="Number")],
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="alpha", type="Number"),
                    ParamInfo(name="extra", type="str"),
                ],
            ],
        )
        params, oidx = _match_overload("torch.key_priority_test", 1, {"alpha": 0.5})
        assert oidx == 0

    def test_live_matrix_norm_overload_selection(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        params, oidx = _match_overload("torch.linalg.matrix_norm", 1, {"ord": "fro", "dim": (-2, -1), "keepdim": False})
        assert oidx == 1
        ord_param = next(p for p in params if p.name == "ord")
        assert ord_param.type == "str"


class TestCoerceValue:
    """Tests for coerce_value.

    单一参数化入口：None 透传、字符串解析、原生类型透传、tuple/list 回退、
    bool 字符串/原生、标量 int/float 等场景均收在 test_coerce_value 的
    参数表内；test_unknown_type_raises 覆盖未知目标类型 → ValueError。
    """

    @pytest.mark.parametrize(
        ("value", "target_type", "expected"),
        [
            (None, "int", None),
            ("None", "int", None),
            ("3", "int", 3),
            ("3.5", "float", 3.5),
            ("2.0", "Number", 2.0),
            ("hello", "str", "hello"),
            ((2, 2), "int", (2, 2)),
            ([2, 3], "int", (2, 3)),
            ((1.0, 2.0), "float", (1.0, 2.0)),
            ([1, 2], "float", (1.0, 2.0)),
            ("true", "bool", True),
            ("false", "bool", False),
            (True, "bool", True),
            (False, "bool", False),
            (2, "int", 2),
            (1.0, "float", 1.0),
        ],
        ids=[
            "none-passthrough",
            "string-none",
            "int-string",
            "float-string",
            "number-coerces-float",
            "str-passthrough",
            "int-tuple-fallback",
            "int-list-fallback",
            "float-tuple-fallback",
            "float-list-fallback",
            "bool-string-true",
            "bool-string-false",
            "bool-native-true",
            "bool-native-false",
            "int-native",
            "float-native",
        ],
    )
    def test_coerce_value(self, value, target_type, expected):
        assert coerce_value(value, target_type) == expected

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unsupported type"):
            coerce_value(42, "UnknownType")


class TestBuildPositionalArgs:
    def test_simple_two_tensors(self):
        _register(
            "torch.add",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
                ParamInfo(name="alpha", type="Number", default="1", is_optional=True, is_keyword_only=True),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
            ],
        )
        t1, t2 = "T1", "T2"
        args, kwargs, oidx = _build_positional_args("torch.add", [t1, t2], {"alpha": "2.0"}, ())
        assert args == ["T1", "T2"]
        assert kwargs == {"alpha": 2.0}
        assert oidx == 0

    def test_interleaved_tensor_scalar(self):
        _register(
            "torch.gather",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int"),
                ParamInfo(name="index", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
                ParamInfo(name="sparse_grad", type="bool", default="False", is_optional=True, is_keyword_only=True),
            ],
        )
        t1, t2 = "T_INPUT", "T_INDEX"
        args, kwargs, oidx = _build_positional_args("torch.gather", [t1, t2], {"dim": "1"}, ())
        assert args == ["T_INPUT", 1, "T_INDEX"]
        assert kwargs == {"sparse_grad": False}

    def test_out_tensor_from_output_indexes(self):
        _register(
            "torch.abs_out",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
            ],
        )
        t_in, t_out = "T_IN", "T_OUT"
        args, kwargs, oidx = _build_positional_args("torch.abs_out", [t_in, t_out], {}, (1,))
        assert args == ["T_IN"]
        assert kwargs == {"out": "T_OUT"}

    def test_tensorlist_param(self):
        _register(
            "torch.cat",
            [
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="dim", type="int", default="0"),
            ],
        )
        tl = ["T_A", "T_B"]
        args, kwargs, oidx = _build_positional_args("torch.cat", [tl], {"dim": "0"}, ())
        assert args == [["T_A", "T_B"], 0]
        assert kwargs == {}

    def test_multi_overload_tensor_scalar(self):
        _register(
            "torch.div",
            [
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="other", type="Tensor"),
                    ParamInfo(name="rounding_mode", type="str", default="None", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
                ],
                [
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="other", type="Number"),
                    ParamInfo(name="rounding_mode", type="str", default="None", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="out", type="Tensor", default="None", is_optional=True, is_keyword_only=True),
                ],
            ],
        )
        t1 = "T_IN"
        args, kwargs, oidx = _build_positional_args("torch.div", [t1], {"other": "3.0"}, ())
        assert oidx == 1
        assert args == ["T_IN", 3.0]

    def test_raises_when_overload_not_matched(self):
        _register(
            "torch.raise_test",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int", default="0"),
                ParamInfo(name="index", type="Tensor"),
            ],
        )
        with pytest.raises(ValueError, match="Cannot match overload"):
            _build_positional_args("torch.raise_test", ["T1"], {}, ())

    def test_none_slot_excluded_from_count(self):
        """None 槽（省略的可选张量占位）不计入输入张量数——对齐 get_param_plan
        口径。修复前 None 计数导致重载匹配失败（golden_api 路径错配/报错）。"""
        _register(
            "torch.none_slot_test",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int", default="0"),
            ],
        )
        # 1 实张量 + 1 个 None 占位：计 1 而非 2，匹配单张量重载（dim 取默认值）
        args, kwargs, oidx = _build_positional_args("torch.none_slot_test", ["T1", None], {}, ())
        assert args == ["T1", 0]
        assert kwargs == {}
        assert oidx == 0


class TestVarPositional:
    """Tests for *args tensor parameter support (is_var_positional=True)."""

    def test_var_positional_consumes_all_remaining_tensors(self):
        _register(
            "torch.block_diag",
            [
                ParamInfo(name="tensors", type="Tensor", is_var_positional=True),
            ],
        )
        args, kwargs, oidx = _build_positional_args("torch.block_diag", ["T1", "T2", "T3"], {}, ())
        assert args == ["T1", "T2", "T3"]
        assert kwargs == {}
        assert oidx == 0

    def test_match_overload_var_positional_accepts_any_count(self):
        _register(
            "torch.vp_match",
            [
                [ParamInfo(name="tensors", type="Tensor", is_var_positional=True)],
            ],
        )
        params1, oidx1 = _match_overload("torch.vp_match", 2)
        assert params1 is not None
        assert oidx1 == 0
        params2, oidx2 = _match_overload("torch.vp_match", 5)
        assert params2 is not None
        assert oidx2 == 0

    def test_match_overload_non_var_still_enforces_upper_bound(self):
        _register(
            "torch.vp_no_var",
            [
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            ],
        )
        params, oidx = _match_overload("torch.vp_no_var", 3)
        assert params is None


class TestParamPlanBuildArgs:
    """Tests for ParamPlan.build_args (reusable plan)."""

    def test_plan_reused_produces_same_result(self):
        _register(
            "torch.plan_reuse",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int", default="0"),
            ],
        )
        from ttk.core_modules.framework_api.api_param import ParamPlan

        params, oidx = _match_overload("torch.plan_reuse", 1)
        plan = ParamPlan("torch.plan_reuse", params, oidx, (), {})

        args1, kwargs1, _ = plan.build_args(["T_A"])
        args2, kwargs2, _ = plan.build_args(["T_B"])
        assert args1 == ["T_A", 0]
        assert args2 == ["T_B", 0]
        assert kwargs1 == {}
        assert kwargs2 == {}

    def test_plan_with_out_tensor(self):
        _register(
            "torch.plan_out",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor", is_keyword_only=True, is_optional=True),
            ],
        )
        from ttk.core_modules.framework_api.api_param import ParamPlan

        params, oidx = _match_overload("torch.plan_out", 1)
        plan = ParamPlan("torch.plan_out", params, oidx, (0,), {})

        t_in, t_out = "T_IN", "T_OUT"
        args, kwargs, _ = plan.build_args([t_in, t_out])
        assert args == ["T_OUT"]  # T_IN filtered to out_tensors, T_OUT is input
        assert kwargs == {"out": "T_IN"}  # T_IN placed as out kwarg


class TestGetParamPlan:
    """Tests for testcase.get_param_plan() -- cached on testcase."""

    def _make_testcase(self, api_name, shapes, dtypes, attrs=None, output_indexes=()):
        from ttk.core_modules.framework_api.testcase import TestcaseE2e

        case = TestcaseE2e()
        case.api_name = api_name
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = dtypes
        case.attributes = attrs or {}
        case.output_tensor_indexes = output_indexes
        case.validate()
        return case

    def test_plan_cached(self):
        case = self._make_testcase("torch.abs", ((2, 3), (2, 3)), ("float32", "float32"), output_indexes=(1,))
        plan1 = case.get_param_plan()
        plan2 = case.get_param_plan()
        assert plan1 is plan2

    def test_plan_none_for_no_api(self):
        case = self._make_testcase("nonexistent.api", ((2, 3),), ("float32",))
        plan = case.get_param_plan()
        assert plan is None

    def test_plan_build_args_integration(self):
        import numpy as np

        _register(
            "torch.np_test",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        case = self._make_testcase("torch.np_test", ((2, 3), (2, 3)), ("float32", "float32"))
        plan = case.get_param_plan()
        assert plan is not None
        tensors = [np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32)]
        args, kwargs, _ = plan.build_args(tensors)
        assert len(args) == 2
        assert isinstance(args[0], np.ndarray)


@pytest.mark.skipif(not _TORCH_NPU_INSTALLED, reason="torch_npu 不可导入，npu_weight_quant_batchmatmul 签名无法解析")
class TestNpuWeightQuantBatchmatmul:
    """Tests for torch_npu.npu_weight_quant_batchmatmul with optional tensors."""

    @staticmethod
    def _make_case(variant, shapes, dtypes, attrs=None):
        from ttk.core_modules.framework_api.testcase import TestcaseE2e

        case = TestcaseE2e()
        case.api_name = "torch_npu.npu_weight_quant_batchmatmul"
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = dtypes
        case.attributes = attrs or {}
        case.validate()
        return case

    def test_per_channel_4_tensors_3_none(self):
        import numpy as np

        case = self._make_case(
            "per_channel",
            ((4, 16), (16, 8), (1, 8), (1, 8), None, None, None),
            ("float16", "int8", "float16", "float16", None, None, None),
        )
        assert case.is_valid

        plan = case.get_param_plan()
        assert plan is not None
        tensors = [
            np.zeros((4, 16), np.float16),
            np.zeros((16, 8), np.int8),
            np.zeros((1, 8), np.float16),
            np.zeros((1, 8), np.float16),
            None,
            None,
            None,
        ]
        args, kwargs, _ = plan.build_args(tensors)

        assert args[0].shape == (4, 16)
        assert args[1].shape == (16, 8)
        assert args[2].shape == (1, 8)
        assert args[3].shape == (1, 8)
        assert args[4] is None
        assert args[5] is None
        assert args[6] is None
        assert args[7] == 0
        assert args[8] == 0
        assert kwargs == {}

    def test_with_bias_5_tensors_2_none(self):
        import numpy as np

        case = self._make_case(
            "with_bias",
            ((4, 16), (16, 8), (1, 8), (1, 8), None, None, (4, 8)),
            ("float16", "int8", "float16", "float16", None, None, "float16"),
        )
        assert case.is_valid

        plan = case.get_param_plan()
        tensors = [
            np.zeros((4, 16), np.float16),
            np.zeros((16, 8), np.int8),
            np.zeros((1, 8), np.float16),
            np.zeros((1, 8), np.float16),
            None,
            None,
            np.zeros((4, 8), np.float16),
        ]
        args, kwargs, _ = plan.build_args(tensors)

        assert args[0].shape == (4, 16)
        assert args[1].shape == (16, 8)
        assert args[2].shape == (1, 8)
        assert args[3].shape == (1, 8)
        assert args[4] is None
        assert args[5] is None
        assert args[6].shape == (4, 8)
        assert args[7] == 0
        assert args[8] == 0
        assert kwargs == {}

    def test_per_group_with_attr(self):
        import numpy as np

        case = self._make_case(
            "per_group",
            ((4, 16), (16, 8), (2, 8), (2, 8), None, None, None),
            ("float16", "int8", "float16", "float16", None, None, None),
            attrs={"antiquant_group_size": 8},
        )
        assert case.is_valid

        plan = case.get_param_plan()
        tensors = [
            np.zeros((4, 16), np.float16),
            np.zeros((16, 8), np.int8),
            np.zeros((2, 8), np.float16),
            np.zeros((2, 8), np.float16),
            None,
            None,
            None,
        ]
        args, kwargs, _ = plan.build_args(tensors)

        assert args[0].shape == (4, 16)
        assert args[1].shape == (16, 8)
        assert args[2].shape == (2, 8)
        assert args[3].shape == (2, 8)
        assert args[4] is None
        assert args[5] is None
        assert args[6] is None
        assert args[7] == 8
        assert args[8] == 0
        assert kwargs == {}


class TestInplaceTensorMethodBuildArgs:
    def test_inplace_build_args_includes_self(self):
        import numpy as np

        _register(
            "torch.Tensor.fake_add_",
            [
                ParamInfo(name="self", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
                ParamInfo(name="alpha", type="Number", default="1", is_optional=True, is_keyword_only=True),
            ],
        )
        from ttk.core_modules.framework_api.api_param import ParamPlan

        plan = ParamPlan(
            api_name="torch.Tensor.fake_add_",
            overload_params=_MANUAL_OVERRIDES["torch.Tensor.fake_add_"].params,
            overload_index=0,
            output_tensor_indexes=(0,),
            attributes={"alpha": "2"},
        )
        tensors = [
            np.zeros((2, 3), np.float32),
            np.ones((2, 3), np.float32),
        ]
        args, kwargs, _ = plan.build_args(tensors)
        assert len(args) == 2
        assert args[0].shape == (2, 3)
        assert args[1].shape == (2, 3)
        assert kwargs.get("alpha") == 2.0


class TestCheckInputCountExceededVarPos:
    """Tests for _check_input_count_exceeded skipping VAR_POSITIONAL overloads."""

    def _make_testcase(self, api_name, shapes, dtypes, attrs=None, output_indexes=()):
        from ttk.core_modules.framework_api.testcase import TestcaseE2e

        case = TestcaseE2e()
        case.api_name = api_name
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = dtypes
        case.attributes = attrs or {}
        case.output_tensor_indexes = output_indexes
        case.validate()
        return case

    def test_var_pos_overload_skips_input_count_check(self):
        _register(
            "torch.block_diag",
            [
                ParamInfo(name="tensors", type="Tensor", is_var_positional=True),
            ],
        )
        case = self._make_testcase("torch.block_diag", ((4, 4), (4, 4)), ("float32", "float32"))
        assert case.is_valid
        assert case.fail_reason != "INPUT_COUNT_EXCEEDED"

    def test_non_var_pos_overload_still_checked(self):
        _register(
            "torch.strict_two_tensors",
            [
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        case = self._make_testcase(
            "torch.strict_two_tensors", ((4, 4), (4, 4), (4, 4)), ("float32", "float32", "float32")
        )
        assert not case.is_valid
        assert case.fail_reason == "INPUT_COUNT_EXCEEDED"

    def test_live_block_diag_passes(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        case = self._make_testcase("torch.block_diag", ((4, 4), (4, 4)), ("float32", "float32"))
        assert case.is_valid


class TestTensorListOutBuildArgs:
    def test_tensor_list_out_assembles_tuple(self):
        import numpy as np

        from ttk.core_modules.framework_api.api_param import ParamPlan

        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
        ]
        t0 = np.zeros((2, 3))
        t1 = np.ones((2, 3))
        t2 = np.full((2, 3), 2.0)
        t3 = np.full((2, 3), 3.0)
        t4 = np.full((2, 3), 4.0)
        plan = ParamPlan(
            api_name="torch_npu.test_encode",
            overload_params=params,
            overload_index=0,
            output_tensor_indexes=(1, 2, 3, 4),
            attributes={},
        )
        args, kwargs, extra = plan.build_args([t0, t1, t2, t3, t4])
        assert len(args) == 1
        assert "out" in kwargs
        assert len(kwargs["out"]) == 4
        assert kwargs["out"][0] is t1
        assert kwargs["out"][3] is t4

    def test_tensor_list_out_with_attrs(self):
        import numpy as np

        from ttk.core_modules.framework_api.api_param import ParamPlan

        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="statistic", type="bool", default=False, is_keyword_only=True),
            ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
        ]
        tensors = [np.zeros((2, 3))] + [np.ones((2, 3)) for _ in range(4)]
        plan = ParamPlan(
            api_name="torch_npu.npu_hans_encode",
            overload_params=params,
            overload_index=0,
            output_tensor_indexes=(1, 2, 3, 4),
            attributes={"statistic": True},
        )
        args, kwargs, extra = plan.build_args(tensors)
        assert len(args) == 1
        assert kwargs["statistic"] is True
        assert "out" in kwargs
        assert len(kwargs["out"]) == 4
