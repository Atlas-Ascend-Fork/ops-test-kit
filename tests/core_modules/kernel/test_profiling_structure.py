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
Tests for ttk.core_modules.kernel.result:
- ComparisonResult: set/get
- KernelExecuteResult: oob_status
"""

import pytest

from ttk.core_modules.kernel.result import (
    ComparisonResult,
    KernelExecuteResult,
)


class TestComparisonResult:
    def test_set_and_get(self):
        cr = ComparisonResult(None)
        result = cr.set("d", "c", "b", "PASS")
        assert result is cr
        assert cr.dyn_precision == "d"
        assert cr.cst_precision == "c"
        assert cr.bin_precision == "b"
        assert cr.passed == "PASS"


class TestKernelExecuteResult:
    @pytest.mark.parametrize(
        ("oob_value", "expected_status"),
        [
            pytest.param("", "PASS", id="empty"),
            pytest.param(None, "PASS", id="none"),
            pytest.param("PASS,SOMETHING", "PASS", id="pass_normal"),
            pytest.param("FAIL,SOMETHING", "FAIL", id="fail"),
        ],
    )
    def test_oob_status(self, oob_value, expected_status):
        """oob_status 属性：空/None/PASS 开头返回 PASS，FAIL 开头返回 FAIL。"""
        r = KernelExecuteResult(oob=oob_value)
        assert r.oob_status == expected_status
