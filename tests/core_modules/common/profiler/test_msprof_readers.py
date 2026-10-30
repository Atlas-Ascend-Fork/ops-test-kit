# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ms_profiler 域 CSV 读取器与 MsProfiler.result 视图测试（谁采集谁解析）。

read_task_durations（kernel/geir 消费）/ read_op_statistic（e2e 消费）/
find_newest_csv（mtime 择新）此前散落在三个通路 + e2e profiler 里各有一份
分叉实现，本文件锚定共享实现的语义。
"""

import logging
import os

from ttk.core_modules.common.profiler.ms_profiler import (
    MsProfiler,
    find_newest_csv,
    read_op_statistic,
    read_task_durations,
)
from ttk.core_modules.common.profiler.types import TtkMsProfType

_OP_SUMMARY_DATA = (
    "Task Type,Task Duration(us)\n"
    "AI_CORE,10.0\n"
    "AI_CORE,20.0\n"
    "MIX_AIV,5.0\n"
    "AI_CPU,999.0\n"  # 非 KERNEL_TASK_TYPES → 过滤
)
_OP_SUMMARY_EMPTY = "Task Type,Task Duration(us)\nSDMA,1.0\n"  # 无 KERNEL 行
_TASK_TIME_DATA = "kernel_type,task_time(us)\nAI_CORE,7.5\nSDMA,1.0\n"

_OP_STATISTIC_DATA = (
    "OP Type,Avg Time(us),Total Time(us),Count,Min Time(us),Max Time(us)\n"
    "Add,2.0,20.0,10,1.0,3.0\n"
    "Mul,1.5,15.0,10,0.5,2.5\n"
    "ZeroAvg,0,0,0,0,0\n"  # Avg<=0 → 剔除
    "   ,1.0,2.0,2,0.5,1.5\n"  # 空 OP Type → 剔除
    "Broken,not-a-number,1.0,1,0,0\n"  # 坏值 → 剔除
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _utime(path, ns):
    os.utime(str(path), ns=(ns, ns))


# --- find_newest_csv ---


def test_find_newest_csv_picks_newest_mtime(tmp_path):
    old = tmp_path / "op_statistic_20230101000000.csv"
    new = tmp_path / "op_statistic_20240101000000.csv"
    old.write_text("old")
    new.write_text("new")
    _utime(old, 1_000_000_000)
    _utime(new, 2_000_000_000)
    assert find_newest_csv(str(tmp_path), ("op_statistic_",)) == str(new)


def test_find_newest_csv_no_match(tmp_path):
    (tmp_path / "other.csv").write_text("x")
    assert find_newest_csv(str(tmp_path), ("op_statistic_",)) is None


# --- read_task_durations ---


def test_read_task_durations_filters_kernel_types(tmp_path):
    _write(tmp_path / "op_summary_20240101000000.csv", _OP_SUMMARY_DATA)
    assert read_task_durations(str(tmp_path)) == [10.0, 20.0, 5.0]


def test_read_task_durations_task_time_columns(tmp_path):
    _write(tmp_path / "task_time_20240101000000.csv", _TASK_TIME_DATA)
    assert read_task_durations(str(tmp_path)) == [7.5]


def test_read_task_durations_newest_first_and_fall_through(tmp_path):
    """mtime 降序；最新的 op_summary 无 KERNEL 行时回退到下一候选
    （kernel 通路的 fall-through 语义）。"""
    oldest = tmp_path / "run1" / "op_summary_20230101000000.csv"
    middle = tmp_path / "task_time_20230102000000.csv"
    newest = tmp_path / "run2" / "op_summary_20240101000000.csv"
    _write(oldest, _OP_SUMMARY_DATA)
    _write(middle, _TASK_TIME_DATA)
    _write(newest, _OP_SUMMARY_EMPTY)
    _utime(oldest, 1_000_000_000)
    _utime(middle, 2_000_000_000)
    _utime(newest, 3_000_000_000)
    assert read_task_durations(str(tmp_path)) == [7.5]


def test_read_task_durations_missing_dir(tmp_path):
    assert read_task_durations(str(tmp_path / "nope")) == []


# --- read_op_statistic ---


def test_read_op_statistic_maps_and_filters_rows(tmp_path):
    csv_path = tmp_path / "op_statistic_20240101000000.csv"
    _write(csv_path, _OP_STATISTIC_DATA)
    kernels = read_op_statistic(str(csv_path))
    assert [k.name for k in kernels] == ["Add", "Mul"]
    add = kernels[0]
    assert add.device_us == 20.0
    assert add.calls == 10
    assert add.avg_us == 2.0
    assert add.max_us == 3.0
    assert add.min_us == 1.0


# --- MsProfiler.result 视图 ---


_ACTIVE_DLL = object()  # non-None sentinel: 采集链路在位


def _msprof_for_result(result_path, prof_type, dll=_ACTIVE_DLL):
    prof = MsProfiler.__new__(MsProfiler)
    prof._msprof_dll = dll
    prof._prof_type = prof_type
    prof._result_path = str(result_path)
    return prof


def test_ms_profiler_result_none_type_is_disabled(tmp_path):
    res = _msprof_for_result(tmp_path, TtkMsProfType.API, dll=None).result(1)
    assert res.source == "disabled"
    assert res.elapsed_us == 0.0
    assert res.kernel_details is None


def test_ms_profiler_result_api_view(tmp_path):
    _write(tmp_path / "op_statistic_20240101000000.csv", _OP_STATISTIC_DATA)
    res = _msprof_for_result(tmp_path, TtkMsProfType.API).result(repeat_count=10)
    assert res.source == "msprof"
    assert res.elapsed_us == 3.5  # Σ Avg（单轮口径）
    assert res.kernel_details.total_device_us == 35.0  # Σ Avg × repeat
    assert res.kernel_details.total_cpu_us == 0.0
    assert [k.name for k in res.kernel_details.kernels] == ["Add", "Mul"]


def test_ms_profiler_result_api_view_no_data(tmp_path):
    res = _msprof_for_result(tmp_path, TtkMsProfType.API).result(1)
    assert res.source == "msprof"
    assert res.elapsed_us == 0.0
    assert res.kernel_details is None


def test_ms_profiler_result_op_view_median(tmp_path):
    _write(tmp_path / "op_summary_20240101000000.csv", _OP_SUMMARY_DATA)  # [10, 20, 5]
    res = _msprof_for_result(tmp_path, TtkMsProfType.OP).result(1)
    assert res.source == "msprof"
    assert res.elapsed_us == 10.0  # median([5, 10, 20])；warmup 剔除归通路编排
    assert res.kernel_details is None


def test_ms_profiler_result_parse_failure_degrades_to_empty(tmp_path, monkeypatch, caplog):
    _write(tmp_path / "op_statistic_20240101000000.csv", _OP_STATISTIC_DATA)

    def _boom(path):
        raise OSError("disk on fire")

    monkeypatch.setattr("ttk.core_modules.common.profiler.ms_profiler.read_op_statistic", _boom)
    prof = _msprof_for_result(tmp_path, TtkMsProfType.API)
    with caplog.at_level(logging.WARNING):
        res = prof.result(1)  # 不得抛——通路按 elapsed==0 走自己的降级
    assert res.source == "msprof"
    assert res.elapsed_us == 0.0
    assert "parse failed" in caplog.text
