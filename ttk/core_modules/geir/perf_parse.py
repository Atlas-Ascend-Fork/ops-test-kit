#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GEIR msprof task_time 解析（msprof 导出与 CSV 读取在 common/profiler/ms_profiler，
此处只保留 GE 图的 median 聚合语义）。"""

import logging

import numpy as np

from ..common.profiler.ms_profiler import export_msprof_summary, read_task_durations


def _parse_msprof_task_duration(prof_dirs):
    """本轮新建 PROF（执行前后差集）的 Task Duration 中位数（µs）。

    msprof -dir 可直接传 PROF 目录，导出/解析逐 PROF 限定在差集上，历史
    PROF 不参与；无数据/工具缺失返回 None。
    """
    if not prof_dirs:
        return None
    durations = []
    for prof_dir in prof_dirs:
        # msprof.py export summary：原始数据转 CSV（仅本轮 PROF）
        export_msprof_summary(prof_dir, exports=("summary",))
        durations.extend(read_task_durations(prof_dir))
    if not durations:
        return None
    median_us = round(float(np.median(durations)), 3)
    logging.debugc("[GEIR] device perf: median=%.3f us, all=%s", median_us, durations)
    return median_us
