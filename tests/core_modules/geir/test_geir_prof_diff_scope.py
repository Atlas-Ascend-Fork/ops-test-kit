# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GEIR PROF 快照差集作用域测试（#185 回归）。

覆盖 _snapshot_prof_entries/_new_prof_dirs 差集语义与
_parse_msprof_task_duration 的导出/解析作用域。
"""

import os
from unittest import mock

import pytest

from ttk.core_modules.geir.execute import _new_prof_dirs, _snapshot_prof_entries
from ttk.core_modules.geir.perf_parse import _parse_msprof_task_duration

_OP_SUMMARY_HEADER = "Task Duration(us),Task Type\n"
_OP_SUMMARY_ROW = "{duration},AI_VECTOR_CORE\n"


def _make_prof_with_summary(root, prof_name, duration, mtime=None):
    prof_dir = root / prof_name
    out_dir = prof_dir / "mindstudio_profiler_output"
    out_dir.mkdir(parents=True)
    csv_path = out_dir / "op_summary_20260101000000.csv"
    csv_path.write_text(_OP_SUMMARY_HEADER + _OP_SUMMARY_ROW.format(duration=duration))
    if mtime is not None:
        os.utime(csv_path, (mtime, mtime))
    return prof_dir


class TestSnapshotDiff:
    """执行前后差集 = 本轮子进程新建的 PROF 目录。"""

    def test_diff_returns_only_new_dirs(self, tmp_path):
        (tmp_path / "PROF_old").mkdir()
        (tmp_path / "PROF_new").mkdir()
        before = _snapshot_prof_entries(str(tmp_path))
        (tmp_path / "PROF_round_new").mkdir()
        (tmp_path / "stray_file.txt").write_text("x")
        assert before == frozenset({"PROF_old", "PROF_new"})
        assert _new_prof_dirs(str(tmp_path), before) == [str(tmp_path / "PROF_round_new")]

    def test_empty_or_missing_path(self, tmp_path):
        assert _snapshot_prof_entries("") == frozenset()
        assert _snapshot_prof_entries(str(tmp_path / "nope")) == frozenset()
        assert _new_prof_dirs("", frozenset()) == []
        assert _new_prof_dirs(str(tmp_path / "nope"), frozenset()) == []


class TestScopedPerfParse:
    """_parse_msprof_task_duration 导出/解析均只作用于本轮新建 PROF。"""

    def test_empty_diff_returns_none_without_export(self, tmp_path):
        with mock.patch("ttk.core_modules.geir.perf_parse.export_msprof_summary") as exp:
            assert _parse_msprof_task_duration([]) is None
            exp.assert_not_called()

    def test_reads_only_new_prof_ignoring_stale_reexport(self, tmp_path):
        _make_prof_with_summary(tmp_path, "PROF_old", 14.885, mtime=2_000_000_000)
        new_prof = _make_prof_with_summary(tmp_path, "PROF_new", 145.001, mtime=1_000_000_000)
        with mock.patch("ttk.core_modules.geir.perf_parse.export_msprof_summary") as exp:
            result = _parse_msprof_task_duration([str(new_prof)])
        exp.assert_called_once_with(str(new_prof), exports=("summary",))
        assert result == pytest.approx(145.001)

    def test_no_kernel_rows_returns_none(self, tmp_path):
        new_prof = tmp_path / "PROF_new" / "mindstudio_profiler_output"
        new_prof.mkdir(parents=True)
        (new_prof / "op_summary_20260101000000.csv").write_text(_OP_SUMMARY_HEADER + "1.0,SDMA\n")
        with mock.patch("ttk.core_modules.geir.perf_parse.export_msprof_summary"):
            assert _parse_msprof_task_duration([str(new_prof.parent)]) is None

    def test_multiple_new_profs_merged_median(self, tmp_path):
        prof_a = _make_prof_with_summary(tmp_path, "PROF_a", 10.0)
        prof_b = _make_prof_with_summary(tmp_path, "PROF_b", 20.0)
        with mock.patch("ttk.core_modules.geir.perf_parse.export_msprof_summary") as exp:
            result = _parse_msprof_task_duration([str(prof_a), str(prof_b)])
        assert [call.args[0] for call in exp.call_args_list] == [str(prof_a), str(prof_b)]
        assert result == pytest.approx(15.0)
