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
Tests for ttk.core_modules.kernel.compare:
- comparing() function: 4-group comparison logic (dyn/golden, cst/golden, bin/golden, bin/dyn)
- ComparisonResult output
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.kernel.compare import __outputs_to_numpy_arrays as _decode_outputs
from ttk.core_modules.kernel.compare import comparing


@pytest.fixture(autouse=True)
def _mock_outputs_to_numpy():
    with patch("ttk.core_modules.kernel.compare.__outputs_to_numpy_arrays", lambda *args, **kwargs: None):
        yield


def _call_comparing(mock_compare_side_effect):
    with patch("ttk.core_modules.common.comparison.comparison.compare") as mock_compare:
        mock_compare.side_effect = mock_compare_side_effect
        result = comparing(
            "dyn_k",
            "cst_k",
            "bin_k",
            (np.array([1.0]),),
            (np.array([1.0]),),
            (np.array([1.0]),),
            (np.array([1.0]),),
            ("float32",),
            standards=[MagicMock()],
        )
        return result, mock_compare


class TestComparingPass:
    def test_all_pass(self):
        result, _ = _call_comparing(
            [
                ("1.0", "", True, {}),
                ("1.0", "", True, {}),
                ("1.0", "", True, {}),
            ]
        )
        assert result.passed == "PASS"
        assert result.dyn_precision == "1.0"
        assert result.cst_precision == "1.0"
        assert result.bin_precision == "1.0"

    def test_dyn_fail_means_overall_fail(self):
        result, _ = _call_comparing(
            [
                ("0.5", "", False, {}),
                ("1.0", "", True, {}),
                ("1.0", "", True, {}),
            ]
        )
        assert result.passed == "FAIL"
        assert result.dyn_precision == "0.5"

    def test_3_comparisons_made(self):
        result, mock_compare = _call_comparing(
            [
                ("1.0", "", True, {}),
                ("1.0", "", True, {}),
                ("1.0", "", True, {}),
            ]
        )
        assert mock_compare.call_count == 3

    def test_comparison_order(self):
        call_log = []

        def track_compare(outputs, goldens, output_dtypes, *, standards, third_parties=None, _verdict=None):
            first = outputs[0]
            label = "dyn" if first is dyn_out else "cst" if first is cst_out else "bin"
            if goldens[0] is golden:
                label += "_vs_golden"
            elif goldens[0] is dyn_out:
                label += "_vs_dyn"
            call_log.append(label)
            return ("1.0", "", True, {})

        dyn_out = np.array([1.0])
        cst_out = np.array([1.0])
        bin_out = np.array([1.0])
        golden = np.array([1.0])

        with patch("ttk.core_modules.common.comparison.comparison.compare", side_effect=track_compare):
            comparing(
                "dyn_k",
                "cst_k",
                "bin_k",
                (dyn_out,),
                (cst_out,),
                (bin_out,),
                (golden,),
                ("float32",),
                standards=[MagicMock()],
            )

        assert call_log == [
            "dyn_vs_golden",
            "cst_vs_golden",
            "bin_vs_golden",
        ]

    def test_exception_returns_compare_failure(self):
        with patch("ttk.core_modules.common.comparison.comparison.compare", side_effect=RuntimeError("boom")):
            result = comparing(
                "dyn_k",
                "cst_k",
                "bin_k",
                (np.array([1.0]),),
                (np.array([1.0]),),
                (np.array([1.0]),),
                (np.array([1.0]),),
                ("float32",),
                standards=[MagicMock()],
            )
        assert result.passed == "COMPARE_FAILURE"


class TestRestoreOutputShape:
    """数据源头形状还原（自 compare_with_hooks._reshape_outputs_for_hooks 上移）。"""

    def _decode(self, n=6):
        return [np.frombuffer(np.arange(n, dtype=np.float32).tobytes(), np.float32)]

    def test_declared_shape_wins(self):
        outputs = self._decode()
        _decode_outputs(outputs, ("float32",), [(2, 3)], [np.ones((3, 2), dtype=np.float32)])  # 声明 ≠ golden 排布
        assert outputs[0].shape == (2, 3)  # 输出真形，不按 golden 排布误导

    def test_declared_missing_falls_back_to_golden(self):
        outputs = self._decode()
        _decode_outputs(outputs, ("float32",), [None], [np.ones((3, 2), dtype=np.float32)])
        assert outputs[0].shape == (3, 2)  # 动态 shape 未知 → golden 形状

    def test_no_shape_keeps_flat(self):
        outputs = self._decode()
        _decode_outputs(outputs, ("float32",), None, None)
        assert outputs[0].shape == (6,)  # 均无形状信息 → 保持扁平

    def test_numel_mismatch_keeps_flat(self):
        outputs = self._decode()
        _decode_outputs(outputs, ("float32",), [(4,)], [np.ones((5,), dtype=np.float32)])  # 声明/golden numel 均不符
        assert outputs[0].shape == (6,)

    def test_sentinels_skipped(self):
        outputs = ["DYN_OFF", None]
        _decode_outputs(outputs, ("float32",), [(2, 3)], [np.ones((2, 3))] * 2)
        assert outputs == ["DYN_OFF", None]
