# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""static_compilation 入口行为：开关关闭/用例无效短路 + normalize_mode 归一化。"""

from unittest.mock import MagicMock, patch

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


class TestStaticCompilation:
    @patch("ttk.core_modules.kernel.compile.static.process_context")
    @patch("ttk.core_modules.kernel.compile.static.get_global_storage")
    @patch("ttk.core_modules.kernel.compile.static.OperatorInterface")
    def test_switch_disabled(self, mock_oi_cls, mock_gs, mock_pc):
        mock_gs.return_value.kernel_meta = "/tmp/test_kernel_meta"
        mock_sw = MagicMock()
        mock_sw.enabled = False
        mock_gs.return_value.cst_switches = mock_sw
        mock_pc.return_value.notify_status = MagicMock()

        from ttk.core_modules.kernel.compile.static import static_compilation

        case = _make_testcase()
        result = static_compilation(case, "Cst")
        assert result.compile_result == "CST_OFF"

    @patch("ttk.core_modules.kernel.compile.static.process_context")
    @patch("ttk.core_modules.kernel.compile.static.get_global_storage")
    @patch("ttk.core_modules.kernel.compile.static.OperatorInterface")
    def test_cst_dyn_invalid(self, mock_oi_cls, mock_gs, mock_pc):
        mock_gs.return_value.kernel_meta = "/tmp/test_kernel_meta"
        mock_sw = MagicMock()
        mock_sw.enabled = True
        mock_sw.realtime = True
        mock_gs.return_value.cst_switches = mock_sw
        mock_pc.return_value.notify_status = MagicMock()

        from ttk.core_modules.kernel.compile.static import static_compilation

        case = _make_testcase()
        case.is_valid = False
        case.fail_reason = "DYN_INPUT_MISSING"
        result = static_compilation(case, "Cst")
        assert result.compile_result == "DYN_INPUT_MISSING"


class TestNormalizeMode:
    """Tests for normalize_mode — 大小写归一化为 'Cst'，非法模式抛 NotImplementedError。"""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("cst", "Cst"),
            ("CST", "Cst"),
        ],
        ids=["lowercase", "uppercase"],
    )
    def test_normalize_mode(self, mode, expected):
        """验证 cst/CST 均归一化为 'Cst'。"""
        from ttk.core_modules.kernel.compile.common import normalize_mode

        assert normalize_mode(mode) == expected

    def test_invalid_raises(self):
        """非法 mode → NotImplementedError。"""
        from ttk.core_modules.kernel.compile.common import normalize_mode

        with pytest.raises(NotImplementedError):
            normalize_mode("invalid_long")
