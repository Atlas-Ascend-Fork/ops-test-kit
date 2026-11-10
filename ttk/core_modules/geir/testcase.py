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

import logging
from typing import Dict, Optional

from ttk.core_modules.common.testcase.field_types import FIELD_TYPES
from ttk.core_modules.kernel.testcase import TestcaseKernel


class GeirTestcase(TestcaseKernel):
    __slots__ = ("dyn_input_shapes",)

    describe_pathway = "geir"

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

    def _check_op_name(self):
        # 已判无效的用例不救助：基类在 ops-info 缺失时会覆写 fail_reason
        was_valid = self.is_valid
        super()._check_op_name()
        if not was_valid or self.is_valid or self.fail_reason != "SOC_NOT_SUPPORT":
            return
        # 纯图级算子（如被 fusion pass 消除的老 IR 算子）无 ops-info 条目属正常，
        # proto 有 REG_OP 声明即可建图编译；查不到才维持 SOC_NOT_SUPPORT
        from ttk.core_modules.geir.proto_loader import ProtoLoader

        if ProtoLoader().get_op_info(self.op_name):
            logging.warning(
                f"Operator [{self.op_name}] has no kernel ops-info entry on this SOC; "
                "allowed by IR registration only. If no fusion pass eliminates it, "
                "graph compile will fail loudly."
            )
            self.is_valid = True
            self.fail_reason = None
