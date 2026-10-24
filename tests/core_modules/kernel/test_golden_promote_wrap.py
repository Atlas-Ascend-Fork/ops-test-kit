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
Regression test for the golden Promote-context wrap lift.

Before the fix, `__golden_mode` (the dtype-promote context for
`--golden-mode=promote`) only wrapped the numpy/torch **builtin** and **class**
golden paths inside KERNEL `_invoke_golden`. The custom + torch/tf-adapter
paths ran UNGUARDED, so under `golden_mode=Promote`, bfloat16/float16 inputs
were NOT promoted before those goldens ran → inaccurate "true value".

The fix lifts the `__golden_mode` wrap to the OUTER dispatch so ALL forms
(class / numpy / torch / tf / custom) are guarded by a single wrap point.

float16 is used (also in DTYPE_PROMOTE_MAP → promotes to float32) because it is
simpler to construct than bfloat16; the promote path under test is identical.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.kernel import golden as _mod
from ttk.core_modules.kernel.testcase import TestcaseKernel
from ttk.core_modules.kernel.worker import _resolve_tolerance as resolve_tolerance_routing

_generate_golden = _mod._generate_golden


def _make_custom_golden(seen_dtypes):
    """构造 custom golden 函数，捕获实际接收的 dtype。"""

    def golden(x, **kwargs):
        if isinstance(x, np.ndarray):
            seen_dtypes.append(x.dtype)
        elif isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, np.ndarray):
                    seen_dtypes.append(t.dtype)
        return x

    return golden


def _make_testcase(
    op_name="custom_promote_op",
    input_shapes=((4,),),
    input_dtypes=("float16",),
    output_shapes=((4,),),
    output_dtypes=("float16",),
):
    case = TestcaseKernel()
    case.testcase_name = f"test_{op_name}_promote_wrap"
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


def _mock_switches_promote():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Promote"  # <-- the mode under test
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch("ttk.core_modules.kernel.golden.OpInfoKeeper")
@patch("ttk.core_modules.kernel.golden.get_global_storage")
@patch("ttk.core_modules.common.plugin.resolver.get_plugin_function")
class TestKernelPromoteWrapCoversAllForms:
    """Under golden_mode=Promote, ALL dispatch forms must see promoted inputs."""

    def test_golden_receives_promoted_float32(self, mock_get_plugin, mock_sw, mock_op_info):
        """Promote 模式下，custom golden 路径应收到提升后的 float32 输入。"""
        seen_dtypes = []

        mock_get_plugin.return_value = _make_custom_golden(seen_dtypes)
        mock_sw.return_value = _mock_switches_promote()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(op_name="custom_promote_op", input_dtypes=("float16",), output_dtypes=("float16",))

        _generate_golden(case, ["float16"])

        assert len(seen_dtypes) >= 1, "golden was not invoked"
        for d in seen_dtypes:
            # float16 ∈ DTYPE_PROMOTE_MAP → must be promoted to float32
            assert d == np.dtype("float32"), (
                f"golden received UN-promoted dtype {d!r}; expected float32 (promoted from float16 under Promote mode)"
            )


class TestKernelResolveToleranceSetsPromote:
    """判据要求升精度（mixed/mix_tolerance）时 _resolve_tolerance 须设 golden_mode_override=Promote。"""

    @staticmethod
    def _switches():
        sw = MagicMock()
        sw.compare_method = None
        sw.plugin_path = None
        return sw

    @classmethod
    def _resolve(cls, case, compare_method=None):
        sw = cls._switches()
        sw.compare_method = compare_method
        spec = MagicMock()
        spec.resolve_tolerance.return_value = None
        spec.resolve_pre_compare.return_value = None
        spec.resolve_compare.return_value = None
        with patch("ttk.core_modules.kernel.worker.get_global_storage", return_value=sw), patch(
            "ttk.core_modules.kernel.worker.SpecResolver", return_value=spec
        ):
            return resolve_tolerance_routing(case)

    def test_mix_tolerance_default_sets_promote(self):
        """float32 输出默认路由 mix_tolerance（单标杆）→ golden_mode_override=Promote。"""
        case = _make_testcase(output_dtypes=("float32",))

        tol_state, early = self._resolve(case)

        assert early is None
        assert case.golden_mode_override == "Promote"
        assert tol_state.need_3party is False

    def test_cli_mixed_alias_sets_promote(self):
        """CLI 简写 mixed 与 mix_tolerance 同判据，同样升精度。"""
        case = _make_testcase(output_dtypes=("float32",))

        _, early = self._resolve(case, compare_method="mixed")

        assert early is None
        assert case.golden_mode_override == "Promote"

    def test_int_output_no_promote(self):
        """int32 输出路由 binary_equal，不设 override。"""
        case = _make_testcase(output_dtypes=("int32",))

        _, early = self._resolve(case)

        assert early is None
        assert not hasattr(case, "golden_mode_override")
