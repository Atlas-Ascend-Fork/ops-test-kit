#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""比对失败日志采集与增量打印。

日志布局：比对明细独立成档——
Index:/Diff: 行随比对发生在 worker、落在该用例的 COMPARE 专属日志文件
（{root}/log/{mode}/{testcase_name}/ttk-compare.log，**追加式**——跨复跑保留
比对历史，每次运行写 "===== run start" 分隔标记）；主进程 handle_result 后
按用例路径读取**末段**（最后一个标记之后）、Index/Diff 行过滤，≤ MAX 条随结果打屏。
"""

__all__ = [
    "MAX_COMPARE_FAILURE_PRINT",
    "compare_log_size",
    "read_compare_log_failures",
    "print_compare_log_failures",
]

import logging
import os
from typing import Optional

MAX_COMPARE_FAILURE_PRINT = 20
RUN_START_MARKER = "===== run start"


def compare_log_size(log_path: str) -> int:
    try:
        return os.path.getsize(log_path)
    except OSError:
        return 0


def read_compare_log_failures(log_path: str, start_size: int = 0) -> tuple:
    """读 log_path 中自 start_size 起新出现的 Index:/Diff: 行。

    per-case 覆写布局下文件即本用例日志，调用方通常传 start_size=0 全读；
    start_size 语义保留（增量读场景兼容）。
    """
    if not os.path.exists(log_path):
        return [], start_size
    try:
        with open(log_path, "rb") as f:
            if start_size:
                f.seek(start_size)
            data = f.read()
    except OSError:
        return [], start_size
    # 追加档只读本运行段（最后一个 run start 标记之后——跨复跑历史不重打）
    if RUN_START_MARKER.encode("utf-8") in data:
        data = data.rsplit(RUN_START_MARKER.encode("utf-8"), 1)[1]
    diff_lines = []
    for line in data.splitlines():
        text = line.decode("utf-8", errors="replace")
        # 两种差异行格式：标准判据 "Expected/Actual/Diff:" 与 cross_check 三方
        # "|t-g|=/|b-g|="（后者无 "Diff:" 字样）
        if "Index:" in text and ("Diff:" in text or "|t-g|" in text):
            diff_lines.append(text.rstrip("\n"))
    return diff_lines, start_size + len(data)


def print_compare_log_failures(diff_lines, testcase_name: Optional[str] = None):
    diff_lines = diff_lines[:MAX_COMPARE_FAILURE_PRINT]
    if diff_lines:
        header = testcase_name
        logging.info(
            "\n=========== %s (first %d) ===========\n%s",
            header,
            len(diff_lines),
            "\n".join(diff_lines),
        )
