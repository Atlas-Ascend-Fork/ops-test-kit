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
Coverage boost for ttk.core_modules.kernel.operator.op_interface
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ttk.core_modules.kernel.compile.compiler as _tbe_mod
from ttk.core_modules.kernel.testcase import TestcaseKernel

_real_opc = _tbe_mod.Opc
_tbe_mod.Opc = MagicMock

from ttk.core_modules.kernel.operator.op_interface import (  # noqa: E402, I001
    OperatorInterface,
    OperatorNotFoundError,
    REGBASE_V2_SOURCE_DEBUG_OPTION,
    _ASCENDC_COMMON_UTILITY_MODULES,
)

# 恢复真实 Opc：op_interface 已在上方导入时绑定了 mock，但 tbe_interface 模块属性
# 不能永久污染——后续其他测试文件延迟导入 Opc 时仍应拿到真实类。
_tbe_mod.Opc = _real_opc


def _make_testcase(
    op_name="Add",
    input_shapes=((8,), (8,)),
    input_dtypes=("float16", "float16"),
    output_shapes=((8,),),
    output_dtypes=("float16",),
    **kwargs,
):
    """构造一个填好默认字段的 TestcaseKernel，便于各用例快速定制。"""
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


def _mock_opc():
    """构造一个上下文管理器齐全的 Opc mock，避免触及真实 CANN 环境。"""
    opc = MagicMock()
    opc.op_info.OpInfo.return_value = MagicMock()
    opc.build_config.return_value.__enter__ = MagicMock(return_value=None)
    opc.build_config.return_value.__exit__ = MagicMock(return_value=False)
    opc.api_config.bit_width_32.return_value.__enter__ = MagicMock(return_value=None)
    opc.api_config.bit_width_32.return_value.__exit__ = MagicMock(return_value=False)
    opc.api_config.bit_width_64.return_value.__enter__ = MagicMock(return_value=None)
    opc.api_config.bit_width_64.return_value.__exit__ = MagicMock(return_value=False)
    opc.op_context.OpContext.return_value.__enter__ = MagicMock(return_value=None)
    opc.op_context.OpContext.return_value.__exit__ = MagicMock(return_value=False)
    opc.get_computes.return_value = []
    opc.get_compile_info.return_value = {}
    opc.get_tiling_op_type.return_value = "Add"
    return opc


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    """清掉可能干扰编译的 CANN 环境变量，保证用例隔离。"""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@pytest.fixture(autouse=True)
def mock_singleton():
    """提供一个装配好 mock Opc 的 OperatorInterface 单例。"""
    p = patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper")
    mock_cls = p.start()
    mock_cls.return_value.info_of.return_value = None
    with patch("ttk.core_modules.kernel.compile.compiler.Opc", return_value=_mock_opc()):
        oi = OperatorInterface()
        oi._opc = _mock_opc()
    yield oi
    p.stop()


class TestWithCoreType:
    """with_core_type 链式设置算子核型并返回自身。"""

    def test_sets_core_type(self, mock_singleton):
        oi = mock_singleton
        result = oi.with_core_type("VectorCore")
        assert result is oi


class TestGetOpGeneralizeFunc:
    """get_op_generalize_func 在空/None/合法 op_type 下的返回值。"""

    @pytest.mark.parametrize(
        ("op_type", "expected"),
        [("", None), (None, None), ("Add", "func")],
    )
    def test_get_op_generalize_func(self, mock_singleton, op_type, expected):
        if op_type == "Add":
            mock_singleton._opc.get_param_generalization.return_value = "func"
        assert mock_singleton.get_op_generalize_func(op_type) == expected


class TestSwitchOpc:
    """_switch_opc 仅对已知后端（tbe）切换，未知后端保持不变。"""

    def test_switch_tbe(self, mock_singleton):
        mock_singleton._switch_opc("tbe")
        mock_singleton._opc.switch_opc.assert_called_with("tbe")

    def test_switch_invalid(self, mock_singleton):
        mock_singleton._switch_opc("invalid")
        mock_singleton._opc.switch_opc.assert_not_called()


class TestEnableShapeInt64:
    """_enable_shape_int64 根据 shape*numel*dtype 字节数判断是否需要 int64 索引。"""

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ([{"shape": (np.iinfo(np.int32).max + 1,), "dtype": "float16"}], True),
            ([{"shape": (65536, 32768), "dtype": "float32"}], True),
            ([{"shape": (0,), "dtype": "float16"}], False),
            ([[{"shape": (8,), "dtype": "float16"}]], False),
        ],
    )
    def test_enable_shape_int64(self, data, expected):
        assert OperatorInterface._enable_shape_int64(data) is expected


class TestDtypeStrToType:
    """dtype_str_to_type 将字符串映射为 Python 类型对象。"""

    def test_list_type(self):
        assert OperatorInterface.dtype_str_to_type("listInt") is list

    def test_int_type(self):
        assert OperatorInterface.dtype_str_to_type("int") is int


class TestGetOpTypeFromSourceCode:
    """get_op_type_from_source_code 从源码 @register_operator 装饰器提取 op 类型。"""

    def test_with_register_operator(self):
        func = MagicMock()
        with patch("inspect.getsource", return_value='@register_operator("AddOp")\ndef add(x, y): pass'):
            assert OperatorInterface.get_op_type_from_source_code(func) == "AddOp"

    def test_os_error(self):
        func = MagicMock()
        with patch("inspect.getsource", side_effect=OSError):
            assert OperatorInterface.get_op_type_from_source_code(func) is None

    def test_not_callable(self):
        assert OperatorInterface.get_op_type_from_source_code("not_a_func") is None


class TestGetOpFuncParams:
    """get_op_func_params 在 OpInfo 缺失时抛 RuntimeError。"""

    def test_runtime_error(self):
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as m:
            m.return_value.info_of.return_value = None
            with pytest.raises(RuntimeError, match="not configured"):
                OperatorInterface.get_op_func_params(op_name="MissingOp")

    def test_from_callable(self):
        def my_op(x, y, axis):
            pass

        result = OperatorInterface.get_op_func_params(operator_func=my_op)
        assert result == ("x", "y", "axis")

    def test_from_op_name(self):
        op_info = {
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
            "attr": [{"name": "axis"}],
        }
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.info_of.return_value = op_info
            result = OperatorInterface.get_op_func_params(op_name="Add")
        assert result == ("x", "y", "z", "axis")

    def test_no_args_returns_empty(self):
        result = OperatorInterface.get_op_func_params()
        assert result == ()


class TestGetOpFuncParameterDict:
    """get_op_func_parameter_dict 从 op_info 构建参数字典（含默认值）。"""

    def test_from_op_info(self):
        op_info = {
            "inputs": [{"name": "x", "paramType": "required"}, {"name": "y", "paramType": "optional"}],
            "outputs": [{"name": "z"}],
            "attr": [
                {"name": "axis", "type": "int", "defaultValue": 1},
                {"name": "keep_dims", "type": "bool", "defaultValue": None},
            ],
        }
        mock_oik = MagicMock()
        mock_oik.info_of.return_value = op_info
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper", return_value=mock_oik):
            result = OperatorInterface.get_op_func_parameter_dict(op_name="FakeOp_param")
        assert len(result) == 5
        assert result["y"].default is None
        assert result["axis"].default == 1

    def test_returns_none(self):
        assert OperatorInterface.get_op_func_parameter_dict() is None


class TestAddAdditionToOpContext:
    """add_addition_to_op_context 将 op_name 与 @ 前缀键写入编译上下文。"""

    def test_basic(self):
        cxt = MagicMock()
        op_info = MagicMock(op_name="Add")
        OperatorInterface.add_addition_to_op_context(cxt, {"!key": "val"}, op_info)
        cxt.add_addition.assert_any_call("op_name", "Add")
        cxt.add_addition.assert_any_call("key", "val")


class TestAddPrivateAttrToOpInfo:
    """add_private_attr_to_op_info 按 @ 前缀名筛选私有属性写入 op_info。"""

    def test_no_private_attrs(self):
        op_info = MagicMock(spec=[])
        OperatorInterface.add_private_attr_to_op_info((), {}, op_info)

    def test_with_private_attrs(self):
        op_info = MagicMock()
        OperatorInterface.add_private_attr_to_op_info(({"name": "axis"}, {"name": "scale"}), {"@scale": 0.5}, op_info)
        op_info.private_attrs = {"name": "scale"}


class TestConstructCompileContextOpInfo:
    """_construct_compile_context_op_info 解析 op_type、回退 UNKNOWN、应用 impl_mode。"""

    def test_with_op_type(self, mock_singleton):
        oi = mock_singleton
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as m:
            m.return_value.op_type_of.return_value = "AddOp"
            oi._construct_compile_context_op_info(lambda: None, "Add", "kernel", {})
        oi._opc.op_info.OpInfo.assert_called_with("AddOp", "AddOp")

    def test_unknown_op_type(self, mock_singleton):
        oi = mock_singleton
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as m, patch.object(
            OperatorInterface, "get_op_type_from_source_code", return_value=None
        ):
            m.return_value.op_type_of.return_value = None
            oi._construct_compile_context_op_info(lambda: None, "Add", "kernel", {})
        oi._opc.op_info.OpInfo.assert_called_with("UNKNOWN", "UNKNOWN")

    def test_with_impl_mode(self, mock_singleton):
        oi = mock_singleton
        mock_oi = MagicMock()
        oi._opc.op_info.OpInfo.return_value = mock_oi
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as m:
            m.return_value.op_type_of.return_value = "AddOp"
            oi._construct_compile_context_op_info(lambda: None, "Add", "kernel", {"impl_mode": "hp"})
        assert mock_oi.precision_mode == "hp"


class TestCompileOp:
    """_compile_op 对合法 func 返回编译耗时；不可调用或执行抛错时各自报错。"""

    @pytest.mark.parametrize(
        ("mode", "func", "expect_raises", "match"),
        [
            ("Dyn", MagicMock(), False, None),
            ("Dyn", MagicMock(side_effect=RuntimeError("err")), True, None),
            ("Cst", "str", True, "not callable"),
        ],
    )
    def test_compile_op(self, mock_singleton, mode, func, expect_raises, match):
        oi = mock_singleton
        mock_gs = MagicMock(kernel_compile_options=[], macros=[])
        with patch("ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs):
            if expect_raises:
                with pytest.raises(RuntimeError, match=match):
                    oi._compile_op(mode, "Add", func, ("x",), [], {})
            else:
                result = oi._compile_op(mode, "Add", func, ("x",), [], {})
                assert isinstance(result, float)

    def test_compile_op_injects_macros(self, mock_singleton, monkeypatch):
        """--compile-opts macros 注入、与 regbase 补参数合并、退出还原、无栈可打时报错。"""
        import sys
        import types

        class FakeCommonUtility:
            @staticmethod
            def check_debug_options(compile_options):
                compile_options.append("ORIGINAL")
                return True

        mod = types.ModuleType("fake_ascendc_common_utility")
        mod.CommonUtility = FakeCommonUtility
        monkeypatch.setitem(sys.modules, _ASCENDC_COMMON_UTILITY_MODULES[0], mod)

        oi = mock_singleton
        mock_gs = MagicMock(kernel_compile_options=[], macros=["-DFOO=1", "-DBAR=2"], compile_options={})

        # 1. 基础注入 + 退出后还原（后续编译不再注入）
        observed = []

        def op_func(*_args, **_kwargs):
            options = []
            FakeCommonUtility.check_debug_options(options)
            observed.append(options)

        with patch("ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs):
            oi._compile_op("Dyn", "Add", op_func, ("x",), [], {})
        assert observed == [["ORIGINAL", "-DFOO=1", "-DBAR=2"]]
        restored = []
        FakeCommonUtility.check_debug_options(restored)
        assert restored == ["ORIGINAL"]

        # 2. 与 regbase 补参（debug + Ascend950）合并在同一 patch 回合
        monkeypatch.setattr("ttk.core_modules.kernel.operator.op_interface.kernel_debug_compile_enabled", lambda: True)
        oi._opc.get_soc_spec.return_value = "Ascend950"
        combined = []

        def op_func_debug(*_args, **_kwargs):
            options = ["-O0", "-g"]
            FakeCommonUtility.check_debug_options(options)
            combined.extend(options)

        with patch("ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs):
            oi._compile_op("Dyn", "Add", op_func_debug, ("x",), [], {})
        assert combined == ["-O0", "-g", "ORIGINAL", REGBASE_V2_SOURCE_DEBUG_OPTION, "-DFOO=1", "-DBAR=2"]

        # 3. 两套栈都打不上补丁时，显式传 macros 必须报错
        for name in _ASCENDC_COMMON_UTILITY_MODULES:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        with patch(
            "ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs
        ), pytest.raises(RuntimeError, match="macros requested"):
            oi._compile_op("Dyn", "Add", MagicMock(), ("x",), [], {})

    def test_compile_op_without_macros_leaves_options(self, mock_singleton, monkeypatch):
        """无 macros 时 check_debug_options 不追加任何编译选项。"""
        import sys
        import types

        class FakeCommonUtility:
            @staticmethod
            def check_debug_options(compile_options):
                compile_options.append("ORIGINAL")
                return True

        mod = types.ModuleType("fake_ascendc_common_utility")
        mod.CommonUtility = FakeCommonUtility
        monkeypatch.setitem(sys.modules, _ASCENDC_COMMON_UTILITY_MODULES[0], mod)

        observed = []

        def op_func(*_args, **_kwargs):
            options = []
            FakeCommonUtility.check_debug_options(options)
            observed.append(options)

        oi = mock_singleton
        mock_gs = MagicMock(kernel_compile_options=[], macros=[])
        with patch("ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs):
            oi._compile_op("Dyn", "Add", op_func, ("x",), [], {})
        assert observed == [["ORIGINAL"]]


class TestSetCommonCompileContext:
    """set_common_compile_context 写入 master_pid 等公共编译附加项。"""

    def test_basic(self, mock_singleton):
        oi = mock_singleton
        cxt = MagicMock()
        cxt.get_op_mode.return_value = "dynamic"
        case = _make_testcase()
        case.kb_pid = 42
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as m:
            m.return_value.op_type_of.return_value = "AddOp"
            oi.set_common_compile_context(cxt, case, MagicMock(), "kernel")
        cxt.add_addition.assert_any_call("master_pid", 42)


class TestSetDynamicCompileContext:
    """set_dynamic_compile_context 在静态模式下装配输入信息。"""

    def test_static_mode(self, mock_singleton):
        oi = mock_singleton
        cxt = MagicMock()
        cxt.get_op_mode.return_value = "static"
        mock_op_info = MagicMock()
        cxt.get_op_info.return_value = [MagicMock(), mock_op_info]
        case = _make_testcase(attributes={"axis": 1})
        dyn_params = ({"shape": (-1,)},) * 3
        with patch.object(oi, "set_common_compile_context"), patch(
            "ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper"
        ) as m:
            m.return_value.info_of.return_value = None
            oi.set_dynamic_compile_context(cxt, case, MagicMock(), "kernel", dyn_params)
        assert mock_op_info.inputs is not None


class TestGetDynOperator:
    """get_dyn_operator 找不到动态算子函数时返回 None。"""

    def test_not_found(self, mock_singleton):
        oi = mock_singleton
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as m:
            m.return_value.get_operator_function.return_value = None
            assert oi.get_dyn_operator(_make_testcase()) is None


class TestCompileDynamicShape:
    """compile_dynamic_shape 在算子缺失时返回 None，命中时走通编译流程。"""

    def test_operator_not_found(self, mock_singleton):
        oi = mock_singleton
        with patch.object(oi, "get_dyn_operator", return_value=None):
            assert oi.compile_dynamic_shape((), _make_testcase(), "k") is None

    def test_compile_success(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase()
        mock_gs = MagicMock(auto_switch=False, kernel_compile_options=[], macros=[])
        dyn_params = (
            {"shape": (-1,), "dtype": "float16"},
            {"shape": (-1,), "dtype": "float16"},
            {"shape": (-1,), "dtype": "float16"},
        )
        with patch.object(oi, "get_dyn_operator", return_value=MagicMock()), patch.object(
            OperatorInterface, "get_op_func_params", return_value=("x",)
        ), patch("ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs), patch(
            "ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper"
        ), patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock_oi2, patch.object(
            oi, "set_dynamic_compile_context"
        ), patch.object(oi, "_compile_op", return_value=0.1), patch.object(
            OperatorInterface, "construct_optiling_attrs", return_value=()
        ):
            mock_oi2.return_value.info_of.return_value = None
            result = oi.compile_dynamic_shape(dyn_params, case, "k")
        assert result is not None


class TestPrepareOperatorParametersConst:
    """prepare_operator_parameters_const 覆盖普通/None/常量/TensorList 输入与输出组合。"""

    def _run(self, oi, case, op_info=None, op_output_defined=False, func_params=("x", "y"), param_transform=None):
        """Helper to run prepare_operator_parameters_const with mocks."""
        mocks = [
            patch.object(oi, "get_dyn_operator", return_value=MagicMock()),
            patch.object(OperatorInterface, "get_op_func_params", return_value=func_params),
            patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper"),
            patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper"),
        ]
        if param_transform is not None:
            mocks.append(
                patch(
                    "ttk.core_modules.kernel.operator.op_interface.adapt_pickup_by_names", return_value=param_transform
                )
            )
        with mocks[0], mocks[1], mocks[2] as m_oi, mocks[3] as m_oi2:
            m_oi.return_value.op_output_defined.return_value = op_output_defined
            m_oi2.return_value.info_of.return_value = op_info
            if param_transform is not None:
                with mocks[4]:
                    return oi.prepare_operator_parameters_const(case)
            return oi.prepare_operator_parameters_const(case)

    def test_operator_not_found(self, mock_singleton):
        oi = mock_singleton
        with patch.object(oi, "get_dyn_operator", return_value=None), pytest.raises(OperatorNotFoundError):
            oi.prepare_operator_parameters_const(_make_testcase())

    def test_normal_inputs_no_output(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            output_shapes=((3, 6),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=False)
        assert len(ipt) == 2
        assert ipt[0]["shape"] == (3, 4)
        assert ipt[1]["shape"] == (5, 6)
        assert opt == ()

    def test_normal_inputs_with_output(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            output_shapes=((3, 6),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True)
        assert len(ipt) == 2
        assert len(opt) == 1
        assert opt[0]["shape"] == (3, 6)
        assert opt[0]["dtype"] == "float16"

    def test_none_input(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase(input_shapes=(None, (3, 4)))
        ipt, opt = self._run(oi, case, op_output_defined=False)
        assert ipt[0] is None
        assert ipt[1]["shape"] == (3, 4)

    def test_const_input(self, mock_singleton):
        oi = mock_singleton
        op_info = {
            "inputs": [{"name": "x"}, {"name": "y", "valueDepend": "required"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(input_shapes=((3, 4), (2,)), attributes={"y": [1, 2]})
        ipt, opt = self._run(oi, case, op_info=op_info, op_output_defined=False, param_transform={"y": [1, 2]})
        assert "const_value" in ipt[1]
        assert ipt[1]["name"] == "y"
        assert ipt[1]["dtype"] == "float16"
        assert ipt[0]["shape"] == (3, 4)

    def test_tl_input(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=((7, 6),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True, func_params=("x", "y"))
        assert isinstance(ipt[0], tuple)
        assert len(ipt[0]) == 2
        assert ipt[0][0]["shape"] == (3, 4)
        assert ipt[0][0]["dtype"] == "float16"
        assert ipt[0][1]["shape"] == (5, 6)
        assert ipt[0][1]["dtype"] == "float16"
        assert ipt[1]["shape"] == (7, 8)

    def test_tl_output(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase(
            input_shapes=((3, 4),),
            input_dtypes=("float16",),
            output_shapes=(((2, 3), (4, 5)),),
            output_dtypes=(("float16", "float16"),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True, func_params=("x",))
        assert isinstance(opt[0], tuple)
        assert len(opt[0]) == 2
        assert opt[0][0]["shape"] == (2, 3)
        assert opt[0][1]["shape"] == (4, 5)

    def test_const_plus_tl_mixed(self, mock_singleton):
        oi = mock_singleton
        op_info = {
            "inputs": [{"name": "x", "valueDepend": "required"}, {"name": "y"}, {"name": "z"}],
            "outputs": [{"name": "out"}],
        }
        case = _make_testcase(
            input_shapes=((2, 3), ((3, 4), (5, 6)), (7, 8)),
            input_dtypes=("float16", ("float16", "float16"), "float16"),
            output_shapes=((7, 6),),
            attributes={"x": [[1, 2, 3], [4, 5, 6]]},
        )
        ipt, opt = self._run(
            oi,
            case,
            op_info=op_info,
            op_output_defined=False,
            func_params=("x", "y", "z"),
            param_transform={"x": [[1, 2, 3], [4, 5, 6]]},
        )
        # pos0: const
        assert "const_value" in ipt[0]
        assert ipt[0]["name"] == "x"
        # pos1: TL (non-const)
        assert isinstance(ipt[1], tuple)
        assert len(ipt[1]) == 2
        assert ipt[1][0]["shape"] == (3, 4)
        assert ipt[1][1]["shape"] == (5, 6)
        # pos2: non-const normal
        assert ipt[2]["shape"] == (7, 8)

    def test_tl_input_and_tl_output(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((2, 3), (4, 5)),),
            output_dtypes=(("float16", "float16"),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True, func_params=("x", "y"))
        assert isinstance(ipt[0], tuple)
        assert len(ipt[0]) == 2
        assert isinstance(opt[0], tuple)
        assert len(opt[0]) == 2
        assert opt[0][0]["shape"] == (2, 3)
        assert opt[0][1]["shape"] == (4, 5)


class TestPrepareTilingParams:
    """prepare_tiling_params 装配输入/输出/属性三元组（属性为空）。"""

    def test_basic(self, mock_singleton):
        oi = mock_singleton
        case = _make_testcase()
        with patch.object(OperatorInterface, "construct_optiling_attrs", return_value=()), patch.object(
            oi,
            "prepare_operator_parameters_const",
            return_value=(({"shape": (8,), "range": (None, None)},), ({"shape": (8,)},)),
        ):
            inputs, outputs, attrs = oi.prepare_tiling_params(case)
        assert len(attrs) == 0


class TestCallConstOpTiling:
    """call_const_op_tiling 成功返回含 tiling_time 的字典；undefined symbol 直接上抛，其余失败转 OPTILING_FAILURE。"""

    @pytest.mark.parametrize(
        ("tiling_side_effect", "expected"),
        [
            ({"tiling_key": 0}, None),
            (RuntimeError("some error"), "OPTILING_FAILURE"),
            (RuntimeError("undefined symbol in foo"), "undefined symbol"),
        ],
    )
    def test_call_const_op_tiling(self, mock_singleton, tiling_side_effect, expected):
        oi = mock_singleton
        mock_cr = MagicMock(tiling_op_type="Add", compile_info={})
        if isinstance(tiling_side_effect, Exception):
            oi._opc.do_op_tiling.side_effect = tiling_side_effect
        else:
            oi._opc.do_op_tiling.return_value = tiling_side_effect
        mock_gs = MagicMock(tiling_run_time=1)
        with patch(
            "ttk.core_modules.kernel.operator.op_interface.get_global_storage", return_value=mock_gs
        ), patch.object(oi, "prepare_tiling_params", return_value=(({"shape": (8,)},), ({"shape": (8,)},), ())), patch(
            "ttk.core_modules.kernel.operator.op_interface.adapter_before_tiling"
        ), patch("os.path.expanduser", return_value="/tmp/plog"), patch("glob.glob", return_value=[]):
            if expected is None:
                result = oi.call_const_op_tiling(mock_cr, _make_testcase())
                assert "tiling_time" in result
            else:
                with pytest.raises(RuntimeError, match=expected):
                    oi.call_const_op_tiling(mock_cr, _make_testcase())


class TestAdapterBeforeTiling:
    """adapter_before_tiling 对 Conv2D 输入格式做 FRACTAL_Z 转换，非 Conv2D 不变。"""

    @pytest.mark.parametrize(
        ("op_type_list", "attributes", "fi", "expect_transform"),
        [
            (["Matmul"], {}, (), False),
            (["Conv2D"], {"groups": 1}, [None, {"format": "NC1HWC0", "shape": (1, 1, 8, 8, 16)}], True),
            (["Conv2D"], {"groups": 1}, [None, {"format": "NCHW", "shape": (16, 3, 3, 3)}], True),
        ],
    )
    def test_adapter_before_tiling(self, op_type_list, attributes, fi, expect_transform):
        from ttk.core_modules.kernel.operator.op_interface import adapter_before_tiling

        case = _make_testcase(attributes=attributes)
        cr = MagicMock(compile_info={"tiling_type": "binary", "op_type_list": op_type_list})
        fo = [None] if fi else ()
        adapter_before_tiling(case, cr, fi, fo)
        if expect_transform:
            assert fi[1]["format"] == "FRACTAL_Z"


_DEFAULT_OP_INFO = {
    "coreType.value": "AiCore",
    "inputs": [{"name": "x"}, {"name": "y"}],
    "outputs": [{"name": "z"}],
}


def _validate(case, op_info=_DEFAULT_OP_INFO):
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock:
        mock.return_value.info_of.return_value = op_info
        case.validate()


class TestPrepareOperatorParameters:
    def test_dyn_mode(self):
        case = _make_testcase()
        _validate(case)
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.op_output_defined.return_value = True
            ipt, opt = OperatorInterface.prepare_operator_parameters(case, "dyn")
        assert len(ipt) == 2
        assert ipt[0]["shape"] == (-1,)

    def test_bin_mode(self):
        case = _make_testcase()
        _validate(case)
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi, patch(
            "ttk.core_modules.kernel.info_keeper.OpInfoKeeper"
        ) as mock_oi2:
            mock_oi.return_value.op_output_defined.return_value = True
            mock_oi2.return_value.info_of.return_value = None
            ipt, opt = OperatorInterface.prepare_operator_parameters(case, "bin")
        assert len(ipt) == 2

    def test_unsupported_mode_raises(self):
        case = _make_testcase()
        with pytest.raises(RuntimeError, match="not supported"):
            OperatorInterface.prepare_operator_parameters(case, "invalid")

    def test_op_output_not_defined(self):
        case = _make_testcase()
        _validate(case)
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.op_output_defined.return_value = False
            ipt, opt = OperatorInterface.prepare_operator_parameters(case, "dyn")
        assert opt == ()


class TestConstructOptilingAttrs:
    def test_from_attr_dictionary_no_op_info(self):
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.info_of.return_value = None
            result = OperatorInterface.construct_optiling_attrs("Add", {"axis": 1, "keep_dims": True})
        assert len(result) == 2
        assert result[0]["name"] == "axis"
        assert result[0]["dtype"] == "int"

    def test_from_op_info(self):
        op_info = {
            "attr": [
                {"name": "axis", "type": "Int", "defaultValue": None},
                {"name": "keep_dims", "type": "Bool", "defaultValue": False},
            ]
        }
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.info_of.return_value = op_info
            result = OperatorInterface.construct_optiling_attrs("Add", {"axis": 1})
        assert len(result) == 2
        assert result[0]["value"] == 1

    def test_required_attr_missing_raises(self):
        op_info = {
            "attr": [
                {"name": "axis", "type": "Int", "defaultValue": None},
            ]
        }
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.info_of.return_value = op_info
            with pytest.raises(RuntimeError, match="Required attribute"):
                OperatorInterface.construct_optiling_attrs("Add", {})

    def test_private_attrs(self):
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.info_of.return_value = None
            result = OperatorInterface.construct_optiling_attrs("Add", {"@private_key": 42})
        assert any(r["name"] == "private_key" for r in result)

    def test_skip_special_prefixes(self):
        with patch("ttk.core_modules.kernel.operator.op_interface.OpInfoKeeper") as mock_oi:
            mock_oi.return_value.info_of.return_value = None
            result = OperatorInterface.construct_optiling_attrs("Add", {"!skip": 1, "#skip": 2, "keep": 3})
        assert len(result) == 1
        assert result[0]["name"] == "keep"


class TestPrintFuncParams:
    def test_basic(self):
        result = OperatorInterface.print_func_params(("x", "y"), {"axis": 1}, [{"shape": (8,)}])
        assert len(result) > 2
        assert "***Params***" in result


class TestAddCompileInfoToOpContext:
    def test_dict_context(self):
        cxt = {}
        case = _make_testcase()
        case.core_type = "AiCore"
        OperatorInterface.add_compile_info_to_op_context(cxt, case)
        assert cxt["_sgt_cube_vector_core_type"] == "AiCore"

    def test_dict_context_with_hash_param(self):
        cxt = {}
        case = _make_testcase(attributes={"#custom_key": "value"})
        OperatorInterface.add_compile_info_to_op_context(cxt, case)
        assert cxt["custom_key"] == "value"

    def test_object_context(self):
        cxt = MagicMock()
        case = _make_testcase(attributes={"#custom_key": "value"})
        OperatorInterface.add_compile_info_to_op_context(cxt, case)
        cxt.add_compile_info.assert_called()
