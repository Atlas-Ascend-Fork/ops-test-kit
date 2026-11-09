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
此处只保留 GE 图的求和均摊语义）。"""

import logging

from ..common.profiler.ms_profiler import export_msprof_summary, read_task_durations


def _parse_msprof_task_duration(prof_dirs, rounds: int):
    """会话内全部 launch 的 Task Duration 求和后按采集轮数均摊（µs）。

    每轮图执行的设备耗时 = 该轮全部 kernel launch 之和（单算子多 kernel /
    多算子图不再被 median 低估）；总和 / 采集轮数 = 平均每轮耗时。无数据/
    工具缺失返回 None。
    """
    if not prof_dirs or rounds <= 0:
        return None
    durations = []
    for prof_dir in prof_dirs:
        # msprof.py export summary：原始数据转 CSV（仅本轮 PROF）
        export_msprof_summary(prof_dir, exports=("summary",))
        durations.extend(read_task_durations(prof_dir))
    if not durations:
        return None
    avg_us = round(sum(durations) / rounds, 3)
    logging.debugc("[GEIR] device perf: rounds=%s, launches=%s, avg=%.3f us", rounds, len(durations), avg_us)
    return avg_us
