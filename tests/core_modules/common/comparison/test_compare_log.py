# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""compare_log 过滤器：标准判据行（Diff:）、cross_check 三方行（|t-g|=）与
判据摘要行（Compare Difference length，带排序标签）都命中；结构化自定义
比对失败条目（precision_metrics）→ 回显行。"""

import os
import tempfile

from ttk.core_modules.common.comparison.compare_log import (
    custom_compare_failure_lines,
    read_compare_log_failures,
)

STD_LINE = "Index: 000 RealIndex: 000012 Expected: 2.000000 Actual: 1.000000 Diff: -0.500000"
CROSS_LINE = (
    "Index: 001 RealIndex: 000013 [L] t=1.000000e+00 g=2.000000e+00 b=1.500000e+00 "
    "|t-g|=1.000000e+00 |b-g|=5.000000e-01"
)
SUMMARY_LINE = "Output 0 Compare Difference length 100 (worst-first by error)"
NOISE_LINE = "some other log line without diff markers"


def _write(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        return f.name


def test_filter_captures_both_formats():
    """三种差异行格式都命中（cross_check 行无 "Diff:" 字样，此前被漏过滤）。"""
    path = _write([NOISE_LINE, STD_LINE, SUMMARY_LINE, CROSS_LINE])
    try:
        lines, end = read_compare_log_failures(path)
        assert lines == [STD_LINE, SUMMARY_LINE, CROSS_LINE]
        assert end > 0
    finally:
        os.unlink(path)


def test_custom_failure_lines_walk_nested_metrics():
    """结构化条目提取：e2e 按模式嵌套 / kernel·aclnn 按输出平铺；非 custom 条目跳过。"""
    entry = {
        "standard": "custom",
        "pass": False,
        "reason": "deliberate mismatch",
        "diff_indices": [0],
        "diff_indices_total": 10000,
        "max_error": 100,
    }
    flat = {0: entry, 1: {"standard": "mix_tolerance", "pass": False, "reason": "x"}}
    nested = {"eager": flat, "graph_cst": {0: dict(entry, reason="graph fail")}}
    lines = custom_compare_failure_lines(nested)
    assert lines == [
        "[eager] Output 0: deliberate mismatch; diff_indices=[0] (total=10000); metrics={'max_error': 100}",
        "[graph_cst] Output 0: graph fail; diff_indices=[0] (total=10000); metrics={'max_error': 100}",
    ]
    assert custom_compare_failure_lines(flat) == [lines[0].replace("[eager] ", "")]
    assert custom_compare_failure_lines({}) == []
    assert custom_compare_failure_lines(None) == []


def test_missing_file_returns_empty():
    lines, end = read_compare_log_failures("/nonexistent/path.log", 0)
    assert lines == []
    assert end == 0
