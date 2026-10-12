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


def _parse_msprof_task_duration(prof_path: str):
    """Run msprof.py export summary and parse op_summary CSV for Task Duration(us).

    Returns median device time in µs, or None when profiling data is
    unavailable / empty / msprof tool missing.
    """
    # 1. msprof.py export summary (converts binary dumps to CSV)
    export_msprof_summary(prof_path, exports=("summary",))

    # 2. Per-launch Task Duration (op_summary_/task_time_，KERNEL_TYPE 过滤，
    #    mtime 择新 + 空文件回退——kernel 通路同源)
    durations = read_task_durations(prof_path)
    if durations:
        median_us = round(float(np.median(durations)), 3)
        logging.debugc("[GEIR] device perf: median=%.3f us, all=%s", median_us, durations)
        return median_us
    return None
