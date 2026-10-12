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
Tests for dyn_* nested refactor:
- dyn_inputs/dyn_outputs return nested format matching stc_* structure
- dyn_input_dtypes etc. return nested format matching stc_* structure
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
    const_input_indexes=None,
    **kwargs,
):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name or 'None'}"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    if const_input_indexes is not None:
        case.const_input_indexes = const_input_indexes
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
def _mock_op_info(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


def _validate(case):
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock:
        mock.return_value.info_of.return_value = None
        case.validate()


class TestDynInputsNested:
    """Tests for dyn_inputs nested refactor.

    分两组：
    - test_dyn_inputs: input_shapes 与 dyn_inputs 期望值的参数化验证
    - test_dyn_inputs_none_element / test_dynamicize_nested_negative_dims:
      特殊断言（None 元素 / 直接调用 classmethod），保留独立测试
    """

    @pytest.mark.parametrize(
        ("input_shapes", "expected"),
        [
            (((8,), (8,)), ((-1,), (-1,))),
            ((((3, 4), (5, 4)), (8,)), (((-1, -1), (-1, -1)), (-1,))),
            ((), ()),
        ],
        ids=["flat-no-tensorlist", "nested-with-tensorlist", "empty"],
    )
    def test_dyn_inputs(self, input_shapes, expected):
        """验证 dyn_inputs 在 flat/nested/empty 下的返回值。"""
        case = _make_testcase(input_shapes=input_shapes)
        _validate(case)
        assert case.dyn_inputs == expected

    def test_dyn_inputs_none_element(self):
        """input_shapes 含 None → dyn_inputs 对应位置为 None。"""
        case = _make_testcase(input_shapes=((8,), None))
        _validate(case)
        assert case.dyn_inputs[0] == (-1,)
        assert case.dyn_inputs[1] is None

    def test_dynamicize_nested_negative_dims(self):
        """_dynamicize_nested: -2 保持，>0 替换为 -1。"""
        result = TestcaseKernel._dynamicize_nested(((-2,), (8,)), (0, 0))
        assert result == ((-2,), (-1,))


class TestDynInputDtypesNested:
    """Tests for dyn_input_dtypes / dyn_input_formats / dyn_outputs / dyn_ori_inputs.

    每行参数：property_name 指定待检查的 dyn_* 属性，kwargs 为 _make_testcase 参数，
    expected 为期望返回值。
    """

    @pytest.mark.parametrize(
        ("property_name", "kwargs", "expected"),
        [
            ("dyn_input_dtypes", {"input_dtypes": ("float16", "float32")}, ("float16", "float32")),
            (
                "dyn_input_dtypes",
                {"input_shapes": (((3, 4), (5, 4)), (8,)), "input_dtypes": (("float16", "float16"), "float32")},
                (("float16", "float16"), "float32"),
            ),
            (
                "dyn_input_formats",
                {"input_shapes": (((3, 4), (5, 4)), (8,)), "input_formats": (("ND", "ND"), "NCHW")},
                (("ND", "ND"), "NCHW"),
            ),
            ("dyn_outputs", {"output_shapes": (((8,), (8,)),)}, (((-1,), (-1,)),)),
            (
                "dyn_ori_inputs",
                {"input_shapes": (((3, 4), (5, 4)), (8,)), "input_ori_shapes": (((3, 4), (5, 4)), (8,))},
                (((-1, -1), (-1, -1)), (-1,)),
            ),
            (
                "dyn_ori_outputs",
                {"output_shapes": (((3, 4),),), "output_ori_shapes": (((3, 4),),)},
                (((-1, -1),),),
            ),
        ],
        ids=[
            "dtypes-flat",
            "dtypes-nested",
            "formats-nested",
            "outputs-nested",
            "ori-inputs-nested",
            "ori-outputs-nested",
        ],
    )
    def test_dyn_property_nested(self, property_name, kwargs, expected):
        """验证各 dyn_* 属性在嵌套结构下返回与 stc 同构的结果。"""
        case = _make_testcase(**kwargs)
        _validate(case)
        assert getattr(case, property_name) == expected


class TestFlatDynProperties:
    """Tests for flat_dyn_* properties.

    每行参数：property_name 指定待检查的 flat_dyn_* 属性，kwargs 为 _make_testcase 参数，
    expected 为期望扁平化结果。
    """

    @pytest.mark.parametrize(
        ("property_name", "kwargs", "expected"),
        [
            ("flat_dyn_inputs", {"input_shapes": ((8,), (8,))}, ((-1,), (-1,))),
            ("flat_dyn_inputs", {"input_shapes": (((3, 4), (5, 4)), (8,))}, ((-1, -1), (-1, -1), (-1,))),
            (
                "flat_dyn_input_dtypes",
                {"input_shapes": (((3, 4), (5, 4)), (8,)), "input_dtypes": (("float16", "float16"), "float32")},
                ("float16", "float16", "float32"),
            ),
            ("flat_dyn_outputs", {"output_shapes": (((8,), (8,)),)}, ((-1,), (-1,))),
            ("flat_dyn_inputs", {"input_shapes": (), "input_dtypes": ()}, ()),
        ],
        ids=[
            "inputs-no-tensorlist",
            "inputs-with-tensorlist",
            "input_dtypes-with-tensorlist",
            "outputs-with-tensorlist",
            "inputs-empty",
        ],
    )
    def test_flat_dyn_property(self, property_name, kwargs, expected):
        """验证 flat_dyn_* 属性返回扁平化结果。"""
        case = _make_testcase(**kwargs)
        _validate(case)
        assert getattr(case, property_name) == expected


class TestDynTensorDictNested:
    def test_dyn_tensor_dict_no_tensorlist(self):
        case = _make_testcase(input_shapes=((8,), (8,)), output_shapes=((8,),))
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(inputs) == 2
        assert len(outputs) == 1
        assert inputs[0]["shape"] == (-1,)
        assert inputs[0]["dtype"] == "float16"
        assert inputs[0]["format"] == "ND"
        assert inputs[0]["ori_shape"] == (-1,)
        assert outputs[0]["shape"] == (-1,)

    def test_dyn_tensor_dict_with_tensorlist(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
            output_shapes=((8,),),
        )
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(inputs) == 2
        # Position 0 is TensorList
        assert isinstance(inputs[0], tuple)
        assert len(inputs[0]) == 2
        assert inputs[0][0]["shape"] == (-1, -1)
        assert inputs[0][0]["dtype"] == "float16"
        assert inputs[0][1]["shape"] == (-1, -1)
        assert inputs[0][1]["dtype"] == "float16"
        # Position 1 is normal
        assert not isinstance(inputs[1], tuple)
        assert inputs[1]["dtype"] == "float32"

    def test_dyn_tensor_dict_output_tensorlist(self):
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(outputs) == 1
        assert isinstance(outputs[0], tuple)
        assert len(outputs[0]) == 2
        assert outputs[0][0]["shape"] == (-1,)
        assert outputs[0][0]["dtype"] == "float16"
        assert outputs[0][1]["shape"] == (-1,)

    def test_dyn_tensor_dict_none_element(self):
        case = _make_testcase(input_shapes=(None, (8,)), input_dtypes=("float16", "float16"))
        _validate(case)
        inputs, _ = case.dyn_tensor_dict
        assert inputs[0] is None
        assert inputs[1]["shape"] == (-1,)

    def test_dyn_tensor_dict_cached(self):
        case = _make_testcase()
        _validate(case)
        td1 = case.dyn_tensor_dict
        td2 = case.dyn_tensor_dict
        assert td1 is td2

    def test_dyn_tensor_dict_scalar_shape_eliminated(self):
        """Scalar shapes () should be converted to (1,) via promote_scalar_shapes."""
        case = _make_testcase(input_shapes=((), (8,)), input_dtypes=("float16", "float16"))
        _validate(case)
        inputs, _ = case.dyn_tensor_dict
        # () is dynamicized to () then promote_scalar_shapes → (1,)
        assert inputs[0]["shape"] == (1,)

    def test_promote_scalar_shapes_nested_static(self):
        """_promote_scalar_shapes_nested 处理 TensorList 嵌套：() → (1,)。"""
        shapes = (((), (8,)), ((),))
        result = TestcaseKernel._promote_scalar_shapes_nested(shapes)
        assert result == (((1,), (8,)), ((1,),))

    def test_promote_scalar_shapes_nested_none(self):
        """_promote_scalar_shapes_nested: None 元素保持 None。"""
        shapes = (None, (8,))
        result = TestcaseKernel._promote_scalar_shapes_nested(shapes)
        assert result[0] is None
        assert result[1] == (8,)

    def test_promote_scalar_shapes_nested_empty(self):
        """_promote_scalar_shapes_nested: 空输入返回空。"""
        result = TestcaseKernel._promote_scalar_shapes_nested(())
        assert result == ()
