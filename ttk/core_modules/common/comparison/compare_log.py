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
按用例路径读取**末段**（最后一个标记之后）、差异行过滤，≤ MAX 条随结果打屏。
自定义比对（Spec.compare）失败详情不走日志扫描：结构化条目经
precision_metrics 随返回结构回传，print_structured_compare_failures 直接消费。
"""

__all__ = [
    "MAX_COMPARE_FAILURE_PRINT",
    "compare_log_size",
    "read_compare_log_failures",
    "print_compare_log_failures",
    "custom_compare_failure_lines",
    "print_structured_compare_failures",
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
        # 三种差异行格式：标准判据 "Expected/Actual/Diff:"、cross_check 三方
        # "|t-g|=/|b-g|="（后者无 "Diff:" 字样）、判据摘要行
        # "Output N Compare Difference length M (排序)"（带排序标签，随行回显）
        if ("Index:" in text and ("Diff:" in text or "|t-g|" in text)) or "Compare Difference length" in text:
            diff_lines.append(text.rstrip("\n"))
    return diff_lines, start_size + len(data)


def custom_compare_failure_lines(metrics) -> list:
    """precision_metrics 中的自定义比对失败条目 → 回显行。

    通用嵌套遍历（e2e 按模式嵌套 {"eager": {0: ...}}，kernel/aclnn 按输出
    平铺 {0: ...}）；条目特征 standard=="custom" 且 pass is False。
    """
    lines = []

    def _walk(node, path):
        for key, entry in (node or {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("standard") == "custom" and entry.get("pass") is False:
                scope = f"[{' '.join(map(str, path))}] " if path else ""
                detail = f"{scope}Output {key}: {entry.get('reason', '')}"
                if "diff_indices" in entry:
                    detail += f"; diff_indices={entry['diff_indices']}"
                    if "diff_indices_total" in entry:
                        detail += f" (total={entry['diff_indices_total']})"
                extras = {
                    k: v
                    for k, v in entry.items()
                    if k not in ("standard", "pass", "reason", "diff_indices", "diff_indices_total")
                }
                if extras:
                    detail += f"; metrics={extras}"
                lines.append(detail)
            else:
                _walk(entry, path + (key,))

    _walk(metrics or {}, ())
    return lines


def print_structured_compare_failures(metrics, testcase_name: Optional[str] = None):
    lines = custom_compare_failure_lines(metrics)[:MAX_COMPARE_FAILURE_PRINT]
    if lines:
        logging.info(
            "\n=========== %s custom compare failures (first %d) ===========\n%s",
            testcase_name or "unknown",
            len(lines),
            "\n".join(lines),
        )


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
