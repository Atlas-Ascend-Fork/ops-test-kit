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
GeirTestcase — GEIR 模式的用例结构，继承自 TestcaseKernel（共享 Golden 生成逻辑）
"""

__all__ = ["GeirTestcase"]

from typing import Dict, Optional

from ttk.core_modules.common.testcase.field_types import FIELD_TYPES
from ttk.core_modules.kernel.testcase import TestcaseKernel


class GeirTestcase(TestcaseKernel):
    __slots__ = ("dyn_input_shapes",)

    non_platform_static_property_headers: Dict[str, tuple] = {
        **TestcaseKernel.non_platform_static_property_headers,
        "dyn_input_shapes": (FIELD_TYPES.SHAPELIKE_DYN_NESTED, None, None),
    }
    property_headers: Dict[str, tuple] = {
        **non_platform_static_property_headers,
        **TestcaseKernel.static_property_headers,
        **TestcaseKernel.special_property_headers,
    }
    complete_headers: Dict[str, tuple] = {
        **TestcaseKernel.identity_headers,
        **property_headers,
        **TestcaseKernel.option_headers,
    }

    def __init__(self):
        super().__init__()
        self.dyn_input_shapes: Optional[tuple] = None

    @staticmethod
    def supported_rerun_title() -> tuple:
        # GEIR 结果列不含 kernel 专属的 bin_perf_us/perf_status（二进制复用走
        # cst_bin_perf_us/dyn_bin_perf_us 列名），rerun 支持面以本通路结果列为准。
        return "dyn_perf_us", "cst_perf_us", "precision_status"
