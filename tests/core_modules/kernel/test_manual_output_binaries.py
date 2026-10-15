# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Tests for manual_golden_binaries / manual_output_data_dtypes
normalize + validate + reshape in testcase_op.py.

Mirrors TestNormalizeManualBinaries but for output side.
"""

from unittest.mock import patch

import pytest

from ttk.core_modules.kernel.testcase import TestcaseKernel


def _make_testcase(
    op_name="Add",
    input_shapes=((8,), (8,)),
    input_dtypes=("float16", "float16"),
    output_shapes=((8,),),
    output_dtypes=("float16",),
    **kwargs,
):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name or 'None'}"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = kwargs.pop("input_ori_shapes", input_shapes)
    case.output_ori_shapes = kwargs.pop("output_ori_shapes", output_shapes)
    case.attributes = kwargs.pop("attributes", {})
    n_in = len(input_shapes)
    n_out = len(output_shapes or ())
    case.input_formats = kwargs.pop("input_formats", ("ND",) * n_in)
    case.input_ori_formats = kwargs.pop("input_ori_formats", ("ND",) * n_in)
    case.output_formats = kwargs.pop("output_formats", ("ND",) * n_out)
    case.output_ori_formats = kwargs.pop("output_ori_formats", ("ND",) * n_out)
    case.input_data_ranges = kwargs.pop("input_data_ranges", (None,) * n_in)
    for k, v in kwargs.items():
        setattr(case, k, v)
    return case


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


def _validate(case):
    n_in = len(case.input_shapes) if case.input_shapes else 0
    n_out = len(case.output_shapes) if case.output_shapes and not isinstance(case.output_shapes, str) else 0
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock:
        mock.return_value.info_of.return_value = {
            "coreType.value": "AiCore",
            "inputs": [{"name": f"i{i}"} for i in range(n_in)],
            "outputs": [{"name": f"o{i}"} for i in range(n_out)],
        }
        case.validate()


# =====================================================================
# Normalize: basic type conversions
# =====================================================================


class TestNormalizeManualOutputBinaries:
    """Tests for _normalize_manual_binaries on output side.

    分三组：
    - test_normalize_preserves_value: validate 后字段被规范化为 expected
    - test_list_converted_to_tuple: list → tuple 转换 + isinstance 断言
    - test_invalid_type_rejected: 非法类型 → MANUAL_OUTPUT_BINARIES_INVALID
    """

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value", "expected"),
        [
            (((8,), (8,)), ("float16", "float16"), (), ()),
            (((8,), (8,)), ("float16", "float16"), None, None),
            (((8,),), ("float16",), "out.bin", ("out.bin",)),
            (
                ((8,), (8,), (8,)),
                ("float16", "float16", "float16"),
                ("o1.bin", "o2.bin", "o3.bin"),
                ("o1.bin", "o2.bin", "o3.bin"),
            ),
            (
                ((8,), None, (8,)),
                ("float16", "float16", "float32"),
                ("o1.bin", "None", "o3.bin"),
                ("o1.bin", None, "o3.bin"),
            ),
        ],
        ids=[
            "empty-not-modified",
            "none-not-modified",
            "single-string-wrapped",
            "flat-tuple-preserved",
            "none-quoted-converted",
        ],
    )
    def test_normalize_preserves_value(self, output_shapes, output_dtypes, value, expected):
        """验证 validate 后 manual_golden_binaries 等于 expected。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid
        assert case.manual_golden_binaries == expected

    def test_list_converted_to_tuple(self):
        """list 被 normalize 转换为 tuple，保留 isinstance 断言。"""
        case = _make_testcase(output_shapes=((8,), (8,)), output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = ["o1.bin", "o2.bin"]
        _validate(case)
        assert case.is_valid
        assert isinstance(case.manual_golden_binaries, tuple)

    def test_invalid_type_rejected(self):
        """非 tuple/list/None/str 类型 → MANUAL_OUTPUT_BINARIES_INVALID。"""
        case = _make_testcase(output_shapes=((8,),), output_dtypes=("float16",))
        case.manual_golden_binaries = 123
        _validate(case)
        assert case.is_valid is False
        assert case.fail_reason == "MANUAL_OUTPUT_BINARIES_INVALID"


# =====================================================================
# Validation: flat outputs
# =====================================================================


class TestValidateOutputBinariesFlat:
    """Tests for flat output binaries validation.

    分两组：
    - test_flat_valid: binaries 与 outputs 匹配 → is_valid 为真
      （expected 为 tuple 时额外检查字段值）
    - test_flat_rejected: binaries 与 outputs 不匹配 → is_valid 为 False
      （check_reason 为 True 表示同时检查 fail_reason）
    """

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value", "expected"),
        [
            (((8,), (8,)), ("float16", "float16"), ("o1.bin", "o2.bin"), True),
            (((8,), None, (8,)), ("float16", "float16", "float32"), ("o1.bin", None, "o3.bin"), True),
            (((8,), None, None), ("float16", "float16", "float16"), ("o1.bin",), ("o1.bin", None, None)),
        ],
        ids=["flat-count-matches", "flat-with-none-output", "flat-trailing-none-padded"],
    )
    def test_flat_valid(self, output_shapes, output_dtypes, value, expected):
        """验证 flat binaries 与 outputs 匹配时 is_valid 为真（expected 为 tuple 时额外检查字段值）。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid
        if expected is not True:
            assert case.manual_golden_binaries == expected

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value", "check_reason"),
        [
            (((8,), (8,)), ("float16", "float16"), ("o1.bin", "o2.bin", "o3.bin"), True),
            (((8,), None, (8,)), ("float16", "float16", "float32"), ("o1.bin", "unexpected.bin", "o3.bin"), False),
            (((8,), (8,), (8,)), ("float16", "float16", "float16"), ("o1.bin", None, "o3.bin"), False),
        ],
        ids=["flat-exceeds-outputs-rejected", "file-for-none-output-rejected", "missing-file-for-non-none-rejected"],
    )
    def test_flat_rejected(self, output_shapes, output_dtypes, value, check_reason):
        """验证 flat binaries 与 outputs 不匹配时 is_valid 为 False。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid is False
        if check_reason:
            assert case.fail_reason == "MANUAL_OUTPUT_BINARIES_INVALID"


# =====================================================================
# Validation: nested (TensorList) outputs
# =====================================================================


class TestValidateOutputBinariesNested:
    """Tests for nested (TensorList) output binaries validation.

    分两组：
    - test_nested_preserved: 嵌套结构与 output_shapes 匹配 → 保留原值
    - test_nested_rejected: 嵌套结构不匹配 → is_valid 为 False
    """

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value", "expected"),
        [
            ((((8,), (8,)),), (("float16", "float16"),), (("o1.bin", "o2.bin"),), (("o1.bin", "o2.bin"),)),
            ((((8,), None),), (("float16", "float16"),), (("o1.bin", None),), (("o1.bin", None),)),
        ],
        ids=["nested-preserved", "nested-with-none-in-tensorlist"],
    )
    def test_nested_preserved(self, output_shapes, output_dtypes, value, expected):
        """验证嵌套 binaries 与 TensorList output_shapes 匹配时被保留。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid
        assert case.manual_golden_binaries == expected

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value"),
        [
            (((8,), (8,)), ("float16", "float16"), (("o1.bin", "o2.bin"),)),
            ((((8,), (8,)), (4,)), ("float16", "float32"), (("o1.bin", "o2.bin"),)),
            ((((8,), (8,)),), (("float16", "float16"),), ("o1.bin",)),
            ((((8,), (8,)), (4,)), ("float16", "float32"), (("o1.bin", "o2.bin"), ("o3.bin",))),
            ((((8,), None),), (("float16", "float16"),), (("o1.bin", "unexpected.bin"),)),
            ((((8,), (8,)),), (("float16", "float16"),), (("o1.bin", None),)),
        ],
        ids=[
            "nested-rejected-without-tensorlist",
            "nested-top-level-count-mismatch",
            "nested-tensorlist-position-is-str-rejected",
            "nested-non-tensorlist-position-is-tuple-rejected",
            "file-for-none-in-tensorlist-rejected",
            "missing-file-for-non-none-tensorlist-rejected",
        ],
    )
    def test_nested_rejected(self, output_shapes, output_dtypes, value):
        """验证嵌套结构不匹配时 is_valid 为 False。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid is False


# =====================================================================
# Reshape: flat → nested
# =====================================================================


class TestReshapeOutputBinaries:
    """Tests for reshape: flat → nested binaries.

    每行参数：output_shapes/output_dtypes/value/expected，验证 flat binaries 被正确 reshape。
    """

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value", "expected"),
        [
            ((((8,), (8,)),), (("float16", "float16"),), ("o1.bin", "o2.bin"), (("o1.bin", "o2.bin"),)),
            ((((8,), None),), (("float16", "float16"),), ("o1.bin", None), (("o1.bin", None),)),
            (
                (((8,), (8,)), (4,)),
                ("float16", "float32"),
                ("o1.bin", "o2.bin", "o3.bin"),
                (("o1.bin", "o2.bin"), "o3.bin"),
            ),
            (((8,), (8,)), ("float16", "float16"), ("o1.bin", "o2.bin"), ("o1.bin", "o2.bin")),
        ],
        ids=["flat-to-nested", "flat-with-none-to-nested", "mixed-tensorlist-and-flat", "no-tensorlist-stays-flat"],
    )
    def test_reshape(self, output_shapes, output_dtypes, value, expected):
        """验证 flat binaries 被 reshape 为匹配 output_shapes 的结构。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid
        assert case.manual_golden_binaries == expected


# =====================================================================
# Flat properties
# =====================================================================


class TestFlatOutputBinariesProperties:
    """Tests for flat_manual_golden_binaries property.

    分两组：
    - test_flat_property: 嵌套/混合结构 → flat 属性返回扁平结果
    - test_flat_output_none_when_not_set: 未设置时 flat 属性为 None
    """

    @pytest.mark.parametrize(
        ("output_shapes", "output_dtypes", "value", "expected"),
        [
            (((8,), (8,)), ("float16", "float16"), ("o1.bin", "o2.bin"), ("o1.bin", "o2.bin")),
            ((((8,), (8,)),), (("float16", "float16"),), (("o1.bin", "o2.bin"),), ("o1.bin", "o2.bin")),
            (
                (((8,), (8,)), (4,)),
                ("float16", "float32"),
                (("o1.bin", "o2.bin"), "o3.bin"),
                ("o1.bin", "o2.bin", "o3.bin"),
            ),
        ],
        ids=["flat", "nested", "mixed"],
    )
    def test_flat_property(self, output_shapes, output_dtypes, value, expected):
        """验证 flat_manual_golden_binaries 在 flat/nested/mixed 下的返回值。"""
        case = _make_testcase(output_shapes=output_shapes, output_dtypes=output_dtypes)
        case.manual_golden_binaries = value
        _validate(case)
        assert case.is_valid
        assert case.flat_manual_golden_binaries == expected

    def test_flat_output_none_when_not_set(self):
        """未设置 manual_golden_binaries → flat 属性为 None。"""
        case = _make_testcase()
        _validate(case)
        assert case.flat_manual_golden_binaries is None
