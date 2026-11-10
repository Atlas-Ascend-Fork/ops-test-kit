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
        # base 归一化阶段可能已因其他原因判无效（如字段歧义、非法 device_ids），而
        # 基类 _check_op_name 在 ops-info 缺失时无条件覆写 fail_reason=SOC_NOT_SUPPORT——
        # 回落只能救"此前有效、仅 SOC 判定失败"的用例，先快照 is_valid 防止复活。
        was_valid = self.is_valid
        super()._check_op_name()
        if not was_valid or self.is_valid or self.fail_reason != "SOC_NOT_SUPPORT":
            return
        # GEIR 测的是图编译通路：fusion pass 等图优化会在编译前消除部分老 IR 节点
        # （如 Histogram 经 HistogramFusionPass 转 HistogramV2），这类纯图级算子按
        # 官方实践不登记 kernel 信息库（canndev 各 arch ops-info ini 均无其条目，
        # GE 也从不按条目初始化它们）。kernel 血统的 SOC 检查在此检查对象错位——
        # 准入条件应是"GE 认识该 IR 算子"而非"该算子有 kernel 实现"。
        # 回落 proto 注册表（含 built-in op_graph/inc 的 REG_OP 声明）；
        # proto 也查不到才是真不支持，保留 SOC_NOT_SUPPORT。
        from ttk.core_modules.geir.proto_loader import ProtoLoader

        if ProtoLoader().get_op_info(self.op_name):
            logging.warning(
                f"Operator [{self.op_name}] has no kernel ops-info entry on this SOC; "
                "allowed by IR registration only (expected for graph-level ops eliminated "
                "by fusion passes, e.g. Histogram). If no pass eliminates it, graph "
                "compile will fail loudly."
            )
            self.is_valid = True
            self.fail_reason = None
