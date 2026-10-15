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
Tests for ttk.core_modules.common.testcase.field_parser: nested parsers.
"""

from functools import partial

import pytest

from ttk.core_modules.common.testcase.field_parser import (
    _eval,
    _is_inference,
    process_dict,
    process_eval,
    scalar_nested,
    shape_stride,
    shapelike_dyn_nested,
    shapelike_float_nested,
    shapelike_float_signed_nested,
    shapelike_stc_ex,
    shapelike_stc_nested,
)

_int_container_nested = partial(scalar_nested, allowed_type=int)


class TestShapelikeStcNested:
    """Tests for shapelike_stc_nested parser.

    分两组：
    - test_parse: 输入字符串 → 期望解析结果
    - test_parse_raises: 非法 shape（float / string）→ TypeError
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("((3,3),(3,5))", ((3, 3), (3, 5))),
            ("(((3,3),(3,2)),(3,5))", (((3, 3), (3, 2)), (3, 5))),
            ("((3,5),)", ((3, 5),)),
            ("(((3,3),(3,2)),None,(3,5))", (((3, 3), (3, 2)), None, (3, 5))),
            ("((None, (2,)), (4,))", ((None, (2,)), (4,))),
            ("(((2,), None), (4,))", (((2,), None), (4,))),
            ("((None, (2,), None),)", ((None, (2,), None),)),
            ("()", ()),
            ("((3,),)", ((3,),)),
            ("((5,),)", ((5,),)),
        ],
        ids=[
            "flat-two-tensors",
            "tensor-list-plus-tensor",
            "single-tensor",
            "with-none",
            "tensor-list-leading-none",
            "tensor-list-trailing-none",
            "tensor-list-multi-none",
            "empty-tuple",
            "scalar-shape",
            "1d-shape",
        ],
    )
    def test_parse(self, text, expected):
        """验证 shapelike_stc_nested 对各类合法输入的解析结果。"""
        assert shapelike_stc_nested(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "((3.5,),)",
            "('abc',)",
            "((None, None),)",
            "((3, (2,)),)",
        ],
        ids=["invalid-shape-float", "invalid-shape-string", "all-none-group-ambiguous", "mixed-scalar-and-shape"],
    )
    def test_parse_raises(self, text):
        """验证非法 shape（float / string）→ TypeError。"""
        with pytest.raises(TypeError):
            shapelike_stc_nested(text)


class TestScalarNested:
    """Tests for scalar_nested parser.

    每行参数：输入字符串与期望解析结果。
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("('float32','float32')", ("float32", "float32")),
            ("(('float32','float32'),'float32')", (("float32", "float32"), "float32")),
            ("('float32',)", ("float32",)),
            ("(('float32','float32'),None)", (("float32", "float32"), None)),
            ("('ND','ND','ND')", ("ND", "ND", "ND")),
            ("(('ND',),'ND')", (("ND",), "ND")),
            ("(1e-08, 1e-08, 1e-08)", (1e-08, 1e-08, 1e-08)),
            ("1e-08", (1e-08,)),
        ],
        ids=[
            "flat-dtypes",
            "nested-dtypes",
            "single-dtype",
            "with-none",
            "formats",
            "compressed-tensor-list-format",
            "float-tuple-no-double-wrap",
            "single-float",
        ],
    )
    def test_parse(self, text, expected):
        """验证 scalar_nested 对各类输入的解析结果。"""
        result = scalar_nested(text)
        assert result == expected
        # float tuple 不应被二次包装为嵌套
        if text == "(1e-08, 1e-08, 1e-08)":
            for val in result:
                assert isinstance(val, float)


class TestIntContainerNested:
    """Tests for int_container_nested (scalar_nested with allowed_type=int).

    每行参数：输入字符串与期望解析结果。
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("(0, 1)", (0, 1)),
            ("((0, 1), 2)", ((0, 1), 2)),
            ("0", (0,)),
            ("", ()),
            ("(0, None, 2)", (0, None, 2)),
        ],
        ids=["flat-offsets", "nested-offsets", "single-value", "empty", "with-none"],
    )
    def test_parse(self, text, expected):
        """验证 int_container_nested 对各类输入的解析结果。"""
        assert _int_container_nested(text) == expected


class TestShapelikeFloatSignedNested:
    """Tests for shapelike_float_signed_nested parser.

    分两组：
    - test_parse: 输入字符串 → 期望解析结果
    - test_parse_raises: 非法字符串值 → TypeError
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("((None, 1.0), (-1.0, 1.0))", ((None, 1.0), (-1.0, 1.0))),
            ("(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))", (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))),
            ("((-1.0, 1.0),)", ((-1.0, 1.0),)),
            ("(((0.0, 1.0), None), (0.0, 1.0))", (((0.0, 1.0), None), (0.0, 1.0))),
            ("((None, (None, 1.0)), (-1.0, 1.0))", ((None, (None, 1.0)), (-1.0, 1.0))),
            ("(((None, 1.0), None), (0.0, 5.0))", (((None, 1.0), None), (0.0, 5.0))),
            ("((None, None),)", ((None, None),)),
            ("()", ()),
            ("((0, 1),)", ((0, 1),)),
        ],
        ids=[
            "flat-ranges",
            "nested-tensor-list",
            "single-range",
            "with-none-element",
            "tensor-list-leading-none",
            "tensor-list-trailing-none",
            "all-none-group-stays-leaf",
            "empty-tuple",
            "int-values",
        ],
    )
    def test_parse(self, text, expected):
        """验证 shapelike_float_signed_nested 对各类合法输入的解析结果。"""
        assert shapelike_float_signed_nested(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "('abc',)",
            "(('abc',),)",
        ],
        ids=["invalid-string-value", "invalid-nested-string"],
    )
    def test_parse_raises(self, text):
        """验证非法字符串值 → TypeError。"""
        with pytest.raises(TypeError):
            shapelike_float_signed_nested(text)


class TestShapelikeDynNested:
    """Tests for shapelike_dyn_nested parser（动态 shape 嵌套，允许 -1/-2）。

    - test_parse: 组内 None 槽位与顺序无关
    - test_parse_raises: 首位 None 组内非法动态维（非 -1/-2 负数）→ ValueError
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("((None, (-1, 2)), (4,))", ((None, (-1, 2)), (4,))),
            ("(((2, -1), None), (4,))", (((2, -1), None), (4,))),
        ],
        ids=["tensor-list-leading-none", "tensor-list-trailing-none"],
    )
    def test_parse(self, text, expected):
        """验证 shapelike_dyn_nested 对组内 None 槽位（任意位置）的解析结果。"""
        assert shapelike_dyn_nested(text) == expected

    def test_parse_raises(self):
        with pytest.raises(ValueError, match="dynamic shape dim"):
            shapelike_dyn_nested("((None, (-3,)),)")


class TestShapelikeFloatNested:
    """Tests for shapelike_float_nested parser（正浮点 range 嵌套）。

    - test_parse: 组内 None 槽位与顺序无关
    - test_parse_raises: 组内标量与 range 混合 → TypeError
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("((None, (0.1, 0.2)), (0.3, 0.4))", ((None, (0.1, 0.2)), (0.3, 0.4))),
            ("(((0.1, 0.2), None), (0.3, 0.4))", (((0.1, 0.2), None), (0.3, 0.4))),
        ],
        ids=["tensor-list-leading-none", "tensor-list-trailing-none"],
    )
    def test_parse(self, text, expected):
        """验证 shapelike_float_nested 对组内 None 槽位（任意位置）的解析结果。"""
        assert shapelike_float_nested(text) == expected

    def test_parse_raises(self):
        with pytest.raises(TypeError):
            shapelike_float_nested("((3, (0.1,)),)")


class TestShapeStride:
    """Tests for shape_stride parser.

    分两组：
    - test_parse: 输入字符串 → 期望解析结果（含 stride=0 广播步长）
    - test_parse_raises: 负步长 → ValueError
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("(4,1)", ((4, 1),)),
            ("((4,1),(2,1))", ((4, 1), (2, 1))),
            ("(0,1)", ((0, 1),)),
            ("((0,1),(4,1))", ((0, 1), (4, 1))),
            ("None", None),
        ],
        ids=["positive", "multi-tensor", "zero-broadcast", "mixed-zero-positive", "none"],
    )
    def test_parse(self, text, expected):
        """验证 shape_stride 对合法输入（含 stride=0 广播）的解析结果。"""
        assert shape_stride(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "(-1,1)",
            "((0,1),(-1,1))",
        ],
        ids=["negative", "mixed-zero-negative"],
    )
    def test_parse_raises(self, text):
        """验证负步长 → ValueError（stride=0 合法但负值不允许）。"""
        with pytest.raises(ValueError, match="negative dim"):
            shape_stride(text)


class TestShapelikeStcExNoneDims:
    """嵌套分支的 None 子维度与扁平分支同语义（allow_sub_value_none）。"""

    def test_nested_none_dim_parses(self):
        assert shapelike_stc_ex("((None, 3),)") == ((None, 3),)

    def test_nested_negative_dim_raises_valueerror(self):
        with pytest.raises(ValueError, match="negative dim"):
            shapelike_stc_ex("((-1, 3),)")


class TestIsInference:
    """推理串识别用前缀匹配，不做子串误判。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ELEWISE", True),
            ("REDUCE", True),
            ("ELEWISE(2, None)", True),
            ("REDUCE((0, 1), 2, None)", True),
            ("(3, 3)", False),
            ("myELEWISE(1,2)", False),
            ("SOMEREDUCE(1,2)", False),
        ],
        ids=["elewise", "reduce", "elewise-limited", "reduce-limited", "plain", "elewise-suffix", "reduce-suffix"],
    )
    def test_prefix_match(self, text, expected):
        assert _is_inference(text) is expected

    def test_invalid_limited_inference_raises(self):
        with pytest.raises(ValueError, match="not a valid limited inference"):
            _is_inference("ELEWISE(not_a_tuple)")


class TestRestrictedEval:
    """字段求值走受限命名空间：白名单函数可用，任意代码执行被拒。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("(1, 2)", (1, 2)),
            ("None", None),
            ("True", True),
            ("1e-08", 1e-08),
            ("'float32'", "float32"),
            ("float('inf')", float("inf")),
            ("-float(0)", -0.0),
            ("{'a': 1}", {"a": 1}),
        ],
        ids=["tuple", "none", "true", "float-literal", "string", "float-inf", "neg-zero", "dict"],
    )
    def test_whitelisted_expressions(self, text, expected):
        assert _eval(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "__import__('os').getpid()",
            "open('/etc/passwd')",
            "().__class__.__bases__[0].__subclasses__()",
            "().__class__",
            "[x for x in (1, 2)]",
            "(1, 2)[0]",
        ],
        ids=["import", "open", "class-chain", "attribute", "comprehension", "subscript"],
    )
    def test_arbitrary_code_blocked(self, text):
        with pytest.raises((NameError, ValueError)):
            _eval(text)

    def test_arithmetic_still_works(self):
        assert _eval("2 * 1024") == 2048
        assert _eval("-0.5") == -0.5

    def test_process_eval_and_dict_still_work(self):
        assert process_eval("(1, 'a')") == (1, "a")
        assert process_dict("{'mode': 'bilinear'}") == {"mode": "bilinear"}
        assert process_dict("{'scale': float('inf')}") == {"scale": float("inf")}
