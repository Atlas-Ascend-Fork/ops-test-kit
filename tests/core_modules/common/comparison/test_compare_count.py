# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UT for compare() 接口（comparison.py 核心分发逻辑）。"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ttk.core_modules.common.comparison import CompareHooks, compare_with_hooks


class _Tc:
    testcase_name = "t"
    api_name = "a"


def _hooks(dtypes):
    return CompareHooks(output_dtypes=dtypes)


def _stds(n):
    return [MagicMock() for _ in range(n)]


class TestOutputSentinels:
    @pytest.mark.parametrize(
        ("sentinel", "check_precision"),
        [
            pytest.param("DYN_OFF", True, id="dyn_off"),
            pytest.param("BIN_OFF", False, id="bin_off"),
        ],
    )
    def test_pass_sentinels(self, sentinel, check_precision):
        """各 fake-pass 哨兵 output → is_pass=True（分类在 compare_with_hooks）。"""
        r = compare_with_hooks(_Tc(), [sentinel], [np.array([1.0])], _stds(1), _hooks(("float32",)))
        assert r[2] is True
        if check_precision:
            assert sentinel in r[0]

    @pytest.mark.parametrize(
        ("output", "expected_precision_substr"),
        [
            pytest.param(None, "NO_OUTPUT", id="none_output"),
            pytest.param("SOMETHING_ELSE", None, id="non_fake_string"),
        ],
    )
    def test_fail_sentinels(self, output, expected_precision_substr):
        """非 fake-pass output → is_pass=False。"""
        r = compare_with_hooks(_Tc(), [output], [np.array([1.0])], _stds(1), _hooks(("float32",)))
        assert r[2] is False
        if expected_precision_substr is not None:
            assert expected_precision_substr in r[0]


class TestMultiOutput:
    def test_mix_dyn_off_and_none_fails(self):
        """多输出混合 fake-pass 与 None → 整体 FAIL（占位路透传 + VALID 子集比对）。"""
        r = compare_with_hooks(
            _Tc(),
            ["DYN_OFF", None],
            [np.array([1.0]), np.array([2.0])],
            _stds(2),
            _hooks(("float32", "float32")),
        )
        assert r[2] is False


class TestThirdPartyCount:
    def test_none_third_parties_ok(self):
        """third_parties=None → 正常流程（不报错）。"""
        r = compare_with_hooks(_Tc(), ["DYN_OFF"], [np.array([1.0])], _stds(1), _hooks(("float32",)))
        assert r[2] is True
