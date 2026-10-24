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
Regression test for the KERNEL builtin-golden dispatch path in
output_generation._generate_golden:

    get_plugin_function(op, "golden", "kernel") -> None
    -> KERNEL_GOLDEN[op] resolves (e.g. "neg" -> numpy.negative)
    -> framework_of(numpy.negative) == "numpy"
    -> _call_numpy_api
    -> _invoke_golden returns numeric numpy arrays

This guards the refactor that unified KERNEL dispatch through
_invoke_golden + framework_of. If the dispatch regressed such that a
builtin op fell through to "UNSUPPORTED" / "GOLDEN_FAILURE", the
assertion that the first golden element is a numeric numpy.ndarray
(not a sentinel string) would fail.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.kernel import golden as _mod
from ttk.core_modules.kernel.testcase import TestcaseKernel

# Access module-level private functions via getattr to dodge class-style name mangling.
_generate_golden = _mod._generate_golden
KERNEL_GOLDEN = _mod.KERNEL_GOLDEN


def _make_testcase(
    op_name="neg", input_shapes=((4,),), input_dtypes=("float32",), output_shapes=((4,),), output_dtypes=("float32",)
):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name}_builtin_golden"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = input_shapes
    case.output_ori_shapes = output_shapes
    n_in = len(input_shapes)
    n_out = len(output_shapes or ())
    case.input_formats = ("ND",) * n_in
    case.input_ori_formats = ("ND",) * n_in
    case.output_formats = ("ND",) * n_out
    case.output_ori_formats = ("ND",) * n_out
    case.input_data_ranges = (None,) * n_in
    case.attributes = {}
    case.input_arrays = tuple(np.ones(s, dtype=d) for s, d in zip(input_shapes, input_dtypes))
    case.original_input_arrays = None
    return case


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Enable"  # not Promote → __golden_mode is a no-op
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    # Avoid real ASCEND toolkit lookups during construction/validate.
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch("ttk.core_modules.kernel.golden.OpInfoKeeper")
@patch("ttk.core_modules.kernel.golden.get_global_storage")
@patch("ttk.core_modules.common.plugin.resolver.get_plugin_function")
class TestKernelBuiltinGoldenDispatch:
    """KERNEL_GOLDEN[op] → numpy ufunc → numeric result (not a sentinel)."""

    def test_neg_resolves_and_returns_numeric_array(self, mock_get_plugin, mock_sw, mock_op_info):
        """`neg` (numpy.negative, unary) end-to-end through _generate_golden."""
        # 1. No custom plugin → forces KERNEL_GOLDEN fallback.
        mock_get_plugin.return_value = None
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            op_name="neg",
            input_shapes=((4,),),
            input_dtypes=("float32",),
            output_shapes=((4,),),
            output_dtypes=("float32",),
        )
        # Sanity: pre-conditions for the path under test.
        assert "neg" in KERNEL_GOLDEN
        assert KERNEL_GOLDEN["neg"] is np.negative

        out = np.array([1.0, -2.0, 3.0, -4.0], dtype="float32")
        case.input_arrays = (out,)

        golden = _generate_golden(case, ["float32"])

        # The path must succeed (not "GOLDEN_FAILURE" / "UNSUPPORTED") and
        # produce a numeric numpy array equal to -input.
        assert len(golden) >= 1
        assert isinstance(golden[0], np.ndarray), f"expected ndarray, got sentinel/string: {golden[0]!r}"
        np.testing.assert_array_equal(golden[0], np.array([-1.0, 2.0, -3.0, 4.0], dtype="float32"))
        # Confirm get_plugin_function was consulted (proves the None→KERNEL_GOLDEN branch ran).
        mock_get_plugin.assert_called_once()

    def test_generate_npu_golden_numpy_carrier_stays_cpu(self, mock_get_plugin, mock_sw, mock_op_info, monkeypatch):
        """golden-mode NPU：numpy golden 在 CPU 现算（source="cpu"、perf 为
        None），不构造 MsProfiler。"""
        mock_get_plugin.return_value = None
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        def _no_profiler(*_args, **_kwargs):
            raise AssertionError("MsProfiler must not be constructed for numpy-carrier golden")

        monkeypatch.setattr("ttk.core_modules.common.golden.npu_golden.MsProfiler", _no_profiler)

        case = _make_testcase(
            op_name="neg",
            input_shapes=((4,),),
            input_dtypes=("float32",),
            output_shapes=((4,),),
            output_dtypes=("float32",),
        )
        case.input_arrays = (np.array([1.0, -2.0, 3.0, -4.0], dtype="float32"),)

        results, perf, source = _mod.generate_npu_golden(case, 0)

        assert source == "cpu"
        assert perf is None
        assert isinstance(results[0], np.ndarray), f"expected ndarray, got sentinel/string: {results[0]!r}"
        np.testing.assert_array_equal(results[0], np.array([-1.0, 2.0, -3.0, 4.0], dtype="float32"))

    def test_generate_npu_golden_flattens_cpu_fallback_result(
        self, mock_get_plugin, mock_sw, mock_op_info, monkeypatch
    ):
        """NPU 执行失败回落 CPU 时 class 形 golden 返回裸数组：generate_npu_golden
        须 flatten 成输出槽列表（否则按声明 shape 截齐时槽位错乱）。同时钉住
        class/custom 分支的 torch_npu 设备初始化。"""
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        import sys
        import types

        set_device_calls = []
        fake_torch_npu = types.SimpleNamespace(npu=types.SimpleNamespace(set_device=set_device_calls.append))
        monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

        class _BareArray:
            def __call__(self):
                return np.arange(6, dtype=np.float32).reshape(2, 3)

        mock_get_plugin.return_value = _BareArray

        case = _make_testcase(
            op_name="neg",
            input_shapes=((4,),),
            input_dtypes=("float32",),
            output_shapes=((2, 3),),
            output_dtypes=("float32",),
        )
        case.input_arrays = (np.ones(4, dtype="float32"),)

        from unittest.mock import patch

        with patch("ttk.core_modules.kernel.golden.Invoke.golden") as mock_golden:
            # 模拟 NPU 执行失败后的 CPU 回落：走 invoke 闭包（真实 _invoke_golden）
            mock_golden.side_effect = (
                lambda context, func, switches, invoke=None, npu_invoke=None, dev_id=None, **_kw: invoke(func)
            )
            results, _perf, _source = _mod.generate_npu_golden(case, 0)

        assert set_device_calls == [0], "class-form golden must initialize torch_npu device"
        assert isinstance(results, list), f"expected list, got {type(results)}"
        assert len(results) == 1, f"expected 1 element, got {len(results)}"
        assert results[0].shape == (2, 3)


@patch("ttk.core_modules.common.plugin.resolver.get_plugin_function")
class TestKernelGoldenFallbackBranchCoverage:
    """Lighter-weight unit on the fallback decision itself.

    Even without a fully-validated TestcaseKernel we want to prove that, when
    get_plugin_function yields None, KERNEL_GOLDEN still resolves the op
    and the resolved callable is the documented numpy ufunc — i.e. the
    regression surface (the dict + its wiring) is exercised directly.
    """

    def test_builtin_ufunc_routes_to_numpy(self, mock_get_plugin):
        # framework_of classification is what steers numpy ufuncs into
        # _call_numpy_api inside _invoke_golden.
        from ttk.core_modules.common.invoke import _framework_of as framework_of

        for op in ("neg", "acos", "floor_div"):
            assert framework_of(KERNEL_GOLDEN[op]) == "numpy", f"{op} must classify as 'numpy' to hit the builtin path"
