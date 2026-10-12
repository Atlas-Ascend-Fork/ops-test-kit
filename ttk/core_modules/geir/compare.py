#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir/compare.py —— GEIR 比对调度。

SpecResolver 钩子解析（直用，无吞错包装）+
compare_with_hooks + dump 触发（含 dump_on_fail 补漏）。
"""

__all__ = ["CompareInputs", "compare_case"]

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ttk.switches import get_global_storage
from ttk.utilities import deep_flatten, resolve_numpy_dtypes

from ..common.comparison import CompareHooks, compare_with_hooks
from ..common.dump import DataDumper
from ..common.plugin import SpecResolver
from .testcase import GeirTestcase


@dataclass
class CompareInputs:
    """比对段输入（worker 编排层装配）。pre_compare 由 worker 判据解析阶段一并
    resolve（双消费点：比对段一次 + execute 逐轮规范化）。"""

    outputs: tuple
    goldens: tuple
    standards: Any
    third_parties: Optional[Sequence] = None
    pre_compare: Optional[Any] = None


def compare_case(testcase: GeirTestcase, segment: str, inputs: CompareInputs, skip_stored_data: bool = False) -> tuple:
    """比对 + dump。返回 (precision, log_str, passed, metrics)。

    skip_stored_data：manual-data store 已写输入/golden 时置 True——常规
    数据 dump 跳过（store 替代，kernel/e2e/aclnn 同惯例），输出 dump 保留。"""
    switches = get_global_storage()
    dump_cfg = switches.dump_config
    if not dump_cfg.dump_input and not dump_cfg.dump_on_fail:
        testcase.input_arrays = None

    # 走带 TestSpec 钩子的公共比对入口（与 kernel/aclnn/e2e 对齐）。
    # 【为何要做】GEIR 原先直接调 compare()，Spec.compare / Spec.pre_compare 在本通路
    # 完全不生效——而对某些算子自实现 compare 是刚需：NonZeroWithValue 的三个输出
    # 都是静态 max-size buffer，有效长度由 count 给出，尾部预留区在 NPU 上未定义，
    # 默认整块比对会把这段未定义内存算进判定。未声明钩子的算子，
    # compare_with_hooks 内部直接回落到原来的 compare()，行为逐位不变。
    resolver = SpecResolver(testcase.op_name, "geir", getattr(switches, "plugin_path", None))
    flat_out_dtypes = tuple(resolve_numpy_dtypes(testcase.flat_output_dtypes))
    precision, log_str, passed, metrics = compare_with_hooks(
        testcase,
        inputs.outputs,
        inputs.goldens,
        inputs.standards,
        CompareHooks(
            output_dtypes=flat_out_dtypes,
            third_parties=inputs.third_parties,
            pre_compare=inputs.pre_compare,
            custom_compare=resolver.resolve_compare(),
        ),
    )

    _std_tokens = sorted({str(s.token) for s in inputs.standards}) if inputs.standards else []
    _std = ",".join(_std_tokens) if _std_tokens else "unknown"
    logging.debugc(f"\nComparing geir_{testcase.testcase_name} with {_std}\n{log_str}")

    _dump_case(testcase, switches, segment, inputs, passed, skip_stored_data)
    return precision, log_str, passed, metrics


def _dump_case(testcase, switches, segment, inputs, passed, skip_stored_data=False):
    """dump（{root}/dump/{case}/ + store 命名协议；segment = const/dynamic，
    修正旧两模式同名覆盖；golden 逐模式重生成，每模式独立落盘）。

    skip_stored_data：输入/golden 已由 manual-data store 落盘时跳过（输出与
    dump-on-fail 仍落——输出来自执行，store 表达不了）。"""
    dump_cfg = switches.dump_config
    dumper = DataDumper(switches, testcase.testcase_name)
    if not skip_stored_data:
        # TensorList 输入嵌套组先展平（对齐 kernel/worker.py 的 dump 口径），
        # 嵌套序列直写会 f.write(list) 崩溃。
        dumper.dump_inputs(tuple(deep_flatten(testcase.input_arrays or ())), segment=segment)
        dumper.dump_goldens([g for g in inputs.goldens if isinstance(g, np.ndarray)])

    dumper.dump_outputs(inputs.outputs, segment=segment)

    if dump_cfg.dump_on_fail and not passed:
        # 现状语义：只补未开的槽位（已开的不重复写）；补漏产物与常规 dump 同名
        dumper.dump_on_fail(
            testcase.input_arrays or (),
            inputs.outputs,
            [g for g in inputs.goldens if isinstance(g, np.ndarray)],
            segment=segment,
        )
