# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""统计复检（--rerun 多轮）结果聚合测试。

明细行恒直写；多轮时按用例采样，组到齐后终判行写 {原始文件名}_rc_summary.csv
（统计列：用例名/终判/原因/轮数/失败直方图/逐路 median 与 CI）；无 cross_check
样本判 SKIP 不产终判行；终判 FAIL 计入 fail_count。
"""

import csv
from types import SimpleNamespace

import pytest

from ttk.core_modules.common.instance import InstanceBase
from ttk.switches import SWITCHES, get_global_storage, set_global_storage


class _FakeInstance(InstanceBase):
    """最小可实例化通路（同 test_instance_base 范式）。"""

    name = "fake"

    def prepare_env(self):
        pass

    def detect_platform(self):
        pass

    def detect_device_count(self):
        pass

    def setup(self):
        pass

    def build_tasks(self, testcases):
        return []

    def handle_result(self, task, result):
        return ("PASS", "{}", "compare log")

    def result_titles(self):
        return ("precision_status", "precision_metrics", "log")

    def get_device_info(self, dev_id):
        return ""


@pytest.fixture
def storage():
    prev = get_global_storage()
    sw = SWITCHES()
    set_global_storage(sw)
    yield sw
    set_global_storage(prev)


def _task(name):
    return SimpleNamespace(testcase=SimpleNamespace(testcase_name=name))


def _row(name, status="FAIL"):
    return (name, "add", status, "{}", "compare log")


def _cross_check_metrics(mare=0.9):
    return {"dyn": {0: {"standard": "cross_check", "result": {"mare": mare, "mere": 0.5, "rmse": 0.8}}}}


def _stat_inst(storage, rounds, tmp_path):
    storage.rerun_rounds = rounds
    storage.input_files = [str(tmp_path / "cases.csv")]  # 原始文件名 → cases_rc_summary.csv
    inst = _FakeInstance()
    inst._rerun_stat_mode = True
    inst.result_path = str(tmp_path / "result.csv")
    inst._precision_status_idx = 2  # 生产由表头首写解析；此处行布局第 3 列
    written = []
    inst.write_row = written.append
    return inst, written


def _summary_rows(tmp_path):
    path = tmp_path / "cases_rc_summary.csv"  # 原始文件名拼接 _rc_summary
    if not path.exists():
        return None
    return list(csv.reader(path.open()))


def test_stat_mode_gate(storage):
    storage.rerun_rounds = 3
    assert _FakeInstance()._rerun_stat_mode is True  # 多轮即开采样；是否有终判取决于样本
    storage.rerun_rounds = 1
    assert _FakeInstance()._rerun_stat_mode is False  # 单轮不采样
    storage.rerun_rounds = None
    assert _FakeInstance()._rerun_stat_mode is False  # 缺省直通


def test_extract_ratio_samples_dual_shapes():
    """metrics 双形状：带模式层（kernel/geir/e2e）与单层（aclnn，mode 记 ""）。"""
    from ttk.core_modules.common.comparison.recheck import RecheckCollector

    layered = RecheckCollector()
    layered.add_run({"dyn": {0: {"standard": "cross_check", "result": {"mare": 0.9}}}})
    assert layered._samples == {("dyn", 0, "mare"): [0.9]}

    single = RecheckCollector()
    single.add_run({0: {"standard": "cross_check", "result": {"mare": 0.9}}})
    assert single._samples == {("", 0, "mare"): [0.9]}

    neither = RecheckCollector()
    neither.add_run({0: {"standard": "mix_tolerance", "result": {"mare": 0.9}}})
    assert neither._samples == {}
    assert neither.failed_rounds == 1  # 非 cross_check 采不到样本 → 计失败轮


def test_detail_rows_written_immediately(storage, tmp_path):
    """明细行不缓冲：逐轮即时落盘；统计只采样，不影响输出。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    inst._dispatch_result_row(_task("A__rc0"), _row("A__rc0"), _cross_check_metrics())
    assert [r[0] for r in written] == ["A__rc0"]
    assert inst._rerun_groups["A"]["count"] == 1
    inst._dispatch_result_row(_task("A__rc1"), _row("A__rc1"), None)
    assert [r[0] for r in written] == ["A__rc0", "A__rc1"]


def test_summary_written_on_group_complete(storage, tmp_path):
    """组到齐 → 统计列终判行写 {原始文件名}_rc_summary.csv。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    assert getattr(storage, "compare_method", None) is None  # Spec 路由 cross_check（无 CLI --compare）
    for k in range(3):
        inst._dispatch_result_row(_task(f"A__rc{k}"), _row(f"A__rc{k}"), _cross_check_metrics())
    assert [r[0] for r in written] == ["A__rc0", "A__rc1", "A__rc2"]  # 明细不受影响
    summary = _summary_rows(tmp_path)
    assert summary is not None
    assert len(summary) == 2  # 表头 + 1 终判行
    row = dict(zip(summary[0], summary[1]))
    assert row["testcase_name"] == "A"
    assert row["precision_status"] == "PASS"  # 比值 0.9 < 1.0 → CI 下限 < 1.0
    assert row["valid_rounds"] == "3"
    assert row["reason"] == ""  # PASS 无判定原因
    assert "median" in row["recheck_metrics"]
    assert "ci_lower" in row["recheck_metrics"]
    assert inst._rerun_groups == {}


def test_verdict_fail_gates_exit_code(storage, tmp_path):
    """终判 FAIL 计入 fail_count（明细行 PASS 但系统性劣化也暴露为退出码 1）。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    for k in range(3):
        inst._dispatch_result_row(_task(f"A__rc{k}"), _row(f"A__rc{k}", status="PASS"), _cross_check_metrics(mare=1.5))
    assert inst.fail_count == 1  # 比值 1.5 → CI 下限 > 1.0 → 统计 FAIL
    row = dict(zip(*_summary_rows(tmp_path)))
    assert row["precision_status"] == "FAIL"
    assert "systematic" in row["reason"]
    assert "dyn/out0/mare=1.5" in row["reason"]  # 失败路明细：mode/输出序/比值与 CI 下限


def test_no_samples_no_summary_row(storage, tmp_path):
    """全轮无 cross_check 样本（三方不可用/错误行）→ 不产终判行，明细照常。"""
    inst, written = _stat_inst(storage, 2, tmp_path)
    inst._dispatch_result_row(_task("A__rc0"), _row("A__rc0", status="FAILURE"), None)
    inst._dispatch_result_row(_task("A__rc1"), _row("A__rc1", status="FAILURE"), None)
    assert [r[0] for r in written] == ["A__rc0", "A__rc1"]
    assert _summary_rows(tmp_path) is None
    assert inst.fail_count == 0
    assert inst._rerun_groups == {}


def test_error_round_counts_as_failed_sample(storage, tmp_path):
    """混合轮：成功轮采样，错误轮计失败直方图；终判按 CI 出。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    inst._dispatch_result_row(_task("A__rc0"), _row("A__rc0"), _cross_check_metrics())
    inst._dispatch_result_row(_task("A__rc2"), _row("A__rc2", status="PROFILE_CRASH"), None)
    inst._dispatch_result_row(_task("A__rc1"), _row("A__rc1"), _cross_check_metrics())
    row = dict(zip(*_summary_rows(tmp_path)))
    assert row["precision_status"] == "PASS"  # 2 有效轮比值 0.9 → CI 下限 < 1.0
    assert row["valid_rounds"] == "2"
    assert row["failed_rounds"] == "1"
    assert "PROFILE_CRASH" in row["fail_markers"]
    assert "median" in row["recheck_metrics"]


def test_incomplete_group_logged_no_verdict(storage, tmp_path):
    """轮数不齐的组（异常中断）cleanup 提示且无终判。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    inst._dispatch_result_row(_task("A__rc0"), _row("A__rc0"), _cross_check_metrics())
    assert "A" in inst._rerun_groups
    inst._log_incomplete_rerun_groups()
    assert inst._rerun_groups == {}
    assert _summary_rows(tmp_path) is None


def test_non_suffixed_name_not_sampled(storage, tmp_path):
    """无 __rc 后缀的行（防御路径）不进统计采样。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    inst._dispatch_result_row(_task("B"), _row("B"), None)
    assert [r[0] for r in written] == ["B"]
    assert inst._rerun_groups == {}


def test_summary_append_mode_matches_result_behavior(storage, tmp_path):
    """-a 追加：表头一致 → 续写；不一致 → 降级覆盖（与 result 打开行为一致）。"""
    inst, written = _stat_inst(storage, 3, tmp_path)
    summary_path = tmp_path / "cases_rc_summary.csv"
    from ttk.core_modules.common.instance import _RECHECK_SUMMARY_TITLES

    storage.append_mode = True
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_RECHECK_SUMMARY_TITLES)
        w.writerow(("old_case", "FAIL", "", 1, 1, 0, "", {}))
    for k in range(3):
        inst._dispatch_result_row(_task(f"A__rc{k}"), _row(f"A__rc{k}"), _cross_check_metrics())
    rows = _summary_rows(tmp_path)
    assert len(rows) == 3  # 旧表头 + 旧行 + 新终判行（追加，不覆盖）
    assert rows[1][0] == "old_case"

    # 表头不一致 → 降级覆盖
    inst2, _ = _stat_inst(storage, 3, tmp_path)
    with summary_path.open("w", newline="") as f:
        csv.writer(f).writerow(["wrong", "header"])
    for k in range(3):
        inst2._dispatch_result_row(_task(f"A__rc{k}"), _row(f"A__rc{k}"), _cross_check_metrics())
    rows = _summary_rows(tmp_path)
    assert len(rows) == 2  # 新表头 + 新终判行
    assert rows[0] == list(_RECHECK_SUMMARY_TITLES)
