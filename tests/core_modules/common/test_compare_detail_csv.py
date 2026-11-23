#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for compare_detail.csv（宽表）：pop_ulp_payloads 嵌套与 (payload, leaf) 契约 /
iter_detail_rows 单行渲染（三对 max_ulp 作列 + precision_metrics 参数并入 + XPU golden 标注）/
InstanceBase.write_row 摘出钩子（端到端：result.csv 互斥 + 明细落盘 + append 对账 +
路径冲突守卫 + 自定义插件 metrics 保留）。"""

import ast
import csv
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.common.comparison.ulp import (
    COMPARE_DETAIL_HEADER,
    iter_detail_rows,
    pop_ulp_payloads,
    scan_ulp_triplet,
)
from ttk.core_modules.common.instance import InstanceBase


def fp16_from_hex(*values):
    return np.array(values, dtype=np.uint16).view(np.float16)


def _sample_payload():
    """三方齐全、含已知差异位的 fp16 载荷（端到端用例的数据源）。"""
    t = fp16_from_hex(0x3C00, 0x3C00, 0x3C02, 0x3C00)
    g = fp16_from_hex(0x3C00, 0x3C01, 0x3C00, 0x3C00)
    b = fp16_from_hex(0x3C00, 0x3C01, 0x3C02, 0x3C01)
    return scan_ulp_triplet(t, g, b, "float16")


def _cross_check_metrics(payload, passed=True, reason=None):
    """cross_check._build_metrics 同构的叶 metrics（宽表参数列的数据源）。"""
    m = {
        "standard": "cross_check",
        "level": "L2",
        "pass": passed,
        "config": {"mare": 2.0, "mere": 1.2, "rmse": 1.2, "small_value": 2**-11, "small_value_atol": 2**-16},
        "result": {
            "mare": 1.0,
            "mere": 0.83,
            "rmse": 0.91,
            "small_err_cnt_target": 3,
            "small_err_cnt_third": 4,
            "small_err_ratio": 0.75,
        },
    }
    if payload is not None:
        m["ulp"] = payload
    if reason:
        m["reason"] = reason
    return m


# ---------------------------------------------------------------------------
# pop_ulp_payloads：嵌套形态 / (payload, leaf) 契约 / 自定义插件兼容
# ---------------------------------------------------------------------------
def test_pop_flat_nesting_aclnn():
    """平铺 {out_idx: leaf}（aclnn）：mode=''，逐叶摘出，值为 (payload, leaf)。"""
    leaf0 = _cross_check_metrics({"pairs": {}})
    leaf1 = {"standard": "cross_check", "pass": True}
    payloads = pop_ulp_payloads({0: leaf0, 1: leaf1})
    assert set(payloads) == {("", 0)}
    payload, leaf = payloads[("", 0)]
    assert payload == {"pairs": {}}
    assert leaf is leaf0
    assert "ulp" not in leaf0
    assert "ulp" not in leaf1


def test_pop_mode_nesting_kernel():
    """mode 层嵌套（kernel dyn/cst/bin、geir、e2e）：(mode, out_idx) 定位。"""
    metrics = {
        "dyn": {0: _cross_check_metrics({"pairs": {}}), 1: {"standard": "cross_check"}},
        "cst": {0: {"standard": "cross_check"}},
    }
    payloads = pop_ulp_payloads(metrics)
    assert set(payloads) == {("dyn", 0)}
    assert "ulp" not in metrics["dyn"][0]
    assert isinstance(payloads[("dyn", 0)], tuple)
    assert payloads[("dyn", 0)][1] is metrics["dyn"][0]


def test_pop_ignores_placeholders():
    """非比对 metrics（占位/错误行/custom 条目）自然跳过、零副作用。"""
    assert pop_ulp_payloads({"reason": "third_party unavailable"}) == {}
    custom = {0: {"standard": "custom", "pass": False, "reason": "x"}}
    assert pop_ulp_payloads(custom) == {}
    assert custom[0]["reason"] == "x"
    assert pop_ulp_payloads(None) == {}
    assert pop_ulp_payloads("SUPPRESSED") == {}


def test_pop_preserves_custom_plugin_ulp_scalar():
    """自定义插件的 ulp 标量指标保留在主表（检视意见 P2：不误摘不丢失）。"""
    custom = {0: {"standard": "custom", "pass": False, "reason": "x", "ulp": 7}}
    assert pop_ulp_payloads(custom) == {}
    assert custom[0]["ulp"] == 7  # 主表保留，不触发明细文件


# ---------------------------------------------------------------------------
# iter_detail_rows：宽表渲染契约（三对列 / N/A / 参数并入 / reason / XPU golden）
# ---------------------------------------------------------------------------
def test_iter_rows_wide_merge():
    """三方齐全 → 单行：三对 max_ulp 各一列 + precision_metrics 参数平铺。

    _sample_payload 手算：npu_cpu dists=[0,1,2,0]→max 2；npu_xpu=[0,1,0,1]→max 1；
    xpu_cpu=[0,0,2,1]→max 2。
    """
    payload = _sample_payload()
    metrics = _cross_check_metrics(payload)
    rows = list(iter_detail_rows(payload, metrics, "case_a", "dyn", 0))
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == len(COMPARE_DETAIL_HEADER) == 23
    assert row[0] == "case_a"
    assert row[1] == "dyn"
    assert row[2] == 0
    assert row[3] == "float16"
    assert row[4] is False  # golden_is_third_party
    # 三对 max_ulp 列（需求 a/b/c）
    assert (row[5], row[6], row[7]) == (2, 1, 2)
    # precision_metrics 参数并入
    assert row[8] == "cross_check"
    assert row[9] == "L2"
    assert row[10] is True
    assert row[11] == 1.0
    assert row[12] == 0.83
    assert row[13] == 0.91
    assert row[14] == 3
    assert row[15] == 4
    assert row[16] == 0.75
    assert row[17] == 2.0
    assert row[18] == 1.2
    assert row[19] == 1.2
    assert row[20] == 2**-11
    assert row[21] == 2**-16
    assert row[22] == ""  # PASS 无 reason


def test_iter_rows_xpu_golden():
    """XPU golden 模式（golden≡third_party）：npu_cpu/xpu_cpu 两列 N/A（检视意见 P2）。"""
    t = fp16_from_hex(0x3C00, 0x3C02)
    golden_source = fp16_from_hex(0x3C00, 0x3C03)
    payload = scan_ulp_triplet(t, golden_source, golden_source, "float16", golden_is_third_party=True)
    metrics = _cross_check_metrics(payload)
    (row,) = list(iter_detail_rows(payload, metrics, "case_xpu_golden", "", 0))
    assert row[4] is True
    assert row[5] == "N/A"  # npu_cpu：无 CPU 参与
    assert isinstance(row[6], int)  # npu_xpu：target vs golden 的真实度量
    assert row[7] == "N/A"  # xpu_cpu：自比恒 0，无信息量


def test_iter_rows_reason_on_fail():
    """失败行：reason 列透传 precision_metrics["reason"]。"""
    payload = _sample_payload()
    metrics = _cross_check_metrics(payload, passed=False, reason="ratio exceeded: mare(3.10>2.0)")
    (row,) = list(iter_detail_rows(payload, metrics, "case_f", "", 0))
    assert row[10] is False
    assert row[22] == "ratio exceeded: mare(3.10>2.0)"


def test_iter_rows_all_special_na():
    """全特殊（无 finite 对）：三对列均 N/A。"""
    nan_arr = fp16_from_hex(0x7E00, 0x7E01)
    payload = scan_ulp_triplet(nan_arr, nan_arr.copy(), nan_arr.copy(), "float16")
    metrics = _cross_check_metrics(payload)
    (row,) = list(iter_detail_rows(payload, metrics, "case_nan", "dyn", 0))
    assert (row[5], row[6], row[7]) == ("N/A", "N/A", "N/A")
    assert row[3] == "float16"


def test_iter_rows_unsupported_dtype():
    """dtype 不支持：单行，三对列 N/A，dtype/参数列照常。"""
    t = np.array([1, 2], dtype=np.int32)
    payload = scan_ulp_triplet(t, t.copy(), t.copy(), "int32")
    metrics = _cross_check_metrics(payload)
    (row,) = list(iter_detail_rows(payload, metrics, "case_i32", "dyn", 1))
    assert row[3] == "int32"
    assert (row[5], row[6], row[7]) == ("N/A", "N/A", "N/A")
    assert row[8] == "cross_check"


def test_iter_rows_defensive_missing_metrics_keys():
    """metrics 缺键（config/result 空）：参数列渲染为空串而非抛错。"""
    payload = _sample_payload()
    (row,) = list(iter_detail_rows(payload, {"standard": "cross_check"}, "c", "", 0))
    assert row[8] == "cross_check"
    assert row[11] == ""
    assert row[17] == ""
    assert row[22] == ""


# ---------------------------------------------------------------------------
# InstanceBase.write_row 摘出钩子（端到端） + 路径冲突守卫 + 生命周期
# ---------------------------------------------------------------------------
class _FakeInstance(InstanceBase):
    """最小可实例化通路（8 个 abstract hook 的空实现）。"""

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
        return ("PASS", "PASS")

    def result_titles(self):
        return ("precision_status", "precision_metrics")

    def get_device_info(self, dev_id):
        return ""


@pytest.fixture
def fake_instance(tmp_path):
    sw = MagicMock()
    sw.random_seed = None
    sw.append_mode = False
    sw.input_files = []
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        inst = _FakeInstance()
    inst.result_path = str(tmp_path / "out_result.csv")
    inst.result_csv_file = open(inst.result_path, "w", newline="")  # noqa: SIM115
    inst.result_csv_writer = csv.writer(inst.result_csv_file)
    # 表头 = 身份列 + 结果列（对齐真实 open_output 的组装形态）
    inst.case_result_titles = ("testcase_name", "precision_status", "precision_metrics")
    inst._detail.path = inst._derive_compare_detail_path()
    inst._detail.name_idx = inst._resolve_column_idx("testcase_name")
    inst._detail.metrics_idx = inst._resolve_column_idx("precision_metrics")
    inst._header_flushed = False
    yield inst
    inst.result_csv_file.close()
    if inst._detail.file:
        inst._detail.file.close()


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_write_row_end_to_end(fake_instance):
    """数据行经 write_row：result.csv 单元格无 ulp（互斥）+ 明细 1 行落盘。"""
    inst = fake_instance
    inst.write_row(inst.case_result_titles)  # 表头行（不触发明细文件）

    payload = _sample_payload()
    metrics = {0: _cross_check_metrics(payload)}  # aclnn 平铺形态
    inst.write_row(("case_a", "PASS", metrics))
    inst.write_row(("case_b", "PASS", {0: {"standard": "mix_tolerance", "pass": True}}))  # 非 cross_check 行

    # result.csv：两数据行；case_a 的 precision_metrics 无 ulp，case_b 原样
    # （显式 flush 后再读：断言不依赖产品隐式刷盘——检视意见）
    inst.result_csv_file.flush()
    rows = _read_csv(inst.result_path)
    assert rows[0] == list(inst.case_result_titles)
    assert len(rows) == 3
    assert "ulp" not in ast.literal_eval(rows[1][2])[0]
    assert ast.literal_eval(rows[2][2])[0]["standard"] == "mix_tolerance"

    # compare_detail.csv：表头 + case_a 单行（宽表：一输出一行；case_b 零扰动）
    inst._detail.file.flush()
    detail = _read_csv(inst._detail.path)
    assert detail[0] == list(COMPARE_DETAIL_HEADER)
    assert len(detail) == 2
    assert detail[1][0] == "case_a"
    assert detail[1][2] == "0"  # output_idx
    assert detail[1][3] == "float16"
    assert (detail[1][5], detail[1][6], detail[1][7]) == ("2", "1", "2")
    assert detail[1][8] == "cross_check"
    assert detail[1][9] == "L2"


def test_custom_plugin_metrics_preserved_in_result(fake_instance):
    """自定义插件 metrics 含 ulp:7 → 主表保留、不触发明细文件（检视意见 P2）。"""
    inst = fake_instance
    inst.write_row(inst.case_result_titles)
    inst.write_row(("case_p", "FAIL", {0: {"standard": "custom", "pass": False, "reason": "x", "ulp": 7}}))

    inst.result_csv_file.flush()
    rows = _read_csv(inst.result_path)
    leaf = ast.literal_eval(rows[1][2])[0]
    assert leaf["ulp"] == 7  # 主表保留
    assert not os.path.exists(inst._detail.path)  # 不触发明细文件


def test_write_row_lazy_no_file_without_ulp(fake_instance):
    """无 cross_check 数据行 → compare_detail.csv 不创建（惰性）。"""
    inst = fake_instance
    inst.write_row(inst.case_result_titles)
    inst.write_row(("case_b", "PASS", {"standard": "mix_tolerance", "pass": True}))
    assert not os.path.exists(inst._detail.path)


def test_write_row_pop_nested_modes(fake_instance):
    """mode 嵌套（kernel 形态）：(mode, out_idx) 各自落单行，mode 列生效。"""
    inst = fake_instance
    inst.write_row(inst.case_result_titles)
    metrics = {
        "dyn": {0: _cross_check_metrics(_sample_payload())},
        "cst": {0: _cross_check_metrics(_sample_payload())},
    }
    inst.write_row(("case_k", "PASS", metrics))
    assert "ulp" not in metrics["dyn"][0]
    assert "ulp" not in metrics["cst"][0]

    inst._detail.file.flush()
    detail = _read_csv(inst._detail.path)
    assert len(detail) == 3  # 表头 + dyn 1 行 + cst 1 行
    assert detail[1][1] == "dyn"
    assert detail[2][1] == "cst"


def test_detail_path_never_collides_with_input(tmp_path):
    """检视意见 P1：`-i compare_detail.csv -o report.csv` → 明细名从 report 派生，不覆写输入。"""
    input_csv = tmp_path / "compare_detail.csv"
    input_csv.write_text("testcase_name,op_name\norig_case,Add\n")  # 原始输入内容
    result_csv = tmp_path / "report.csv"
    result_csv.write_text("")  # 空 → 主表新建

    sw = MagicMock()
    sw.random_seed = None
    sw.append_mode = False
    sw.input_files = [str(input_csv)]
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        inst = _FakeInstance()
    inst.result_path = str(result_csv)
    inst.case_result_titles = ("testcase_name", "precision_status", "precision_metrics")

    detail_path = inst._derive_compare_detail_path()
    assert detail_path == str(tmp_path / "report_compare_detail.csv")
    assert os.path.abspath(detail_path) != os.path.abspath(str(input_csv))

    # 端到端：写入 ULP 明细后输入文件内容完好
    inst.result_csv_file = open(result_csv, "w", newline="")  # noqa: SIM115
    inst.result_csv_writer = csv.writer(inst.result_csv_file)
    inst._detail.path = detail_path
    inst._detail.name_idx = inst._resolve_column_idx("testcase_name")
    inst._detail.metrics_idx = inst._resolve_column_idx("precision_metrics")
    inst._header_flushed = False
    inst.write_row(inst.case_result_titles)
    inst.write_row(("c", "PASS", {0: _cross_check_metrics(_sample_payload())}))

    assert input_csv.read_text() == "testcase_name,op_name\norig_case,Add\n"  # 输入完好
    assert os.path.exists(detail_path)  # 明细在专属路径
    inst.result_csv_file.close()
    inst._detail.file.close()


def test_detail_path_result_suffix_not_stripped(tmp_path):
    """检视意见 P2：`jobs_result.csv` 与 `jobs.csv` 派生不同的明细（不共用）。

    剥去 _result 后缀会让两份合法主表都派生 jobs_compare_detail.csv——
    后写覆盖先写。恒用完整 stem 保证互不干扰。
    """
    sw = MagicMock()
    sw.random_seed = None
    sw.append_mode = False
    sw.input_files = []
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        inst = _FakeInstance()
    inst.case_result_titles = ("testcase_name", "precision_status", "precision_metrics")

    inst.result_path = str(tmp_path / "jobs_result.csv")
    detail_a = inst._derive_compare_detail_path()
    inst.result_path = str(tmp_path / "jobs.csv")
    detail_b = inst._derive_compare_detail_path()

    assert detail_a == str(tmp_path / "jobs_result_compare_detail.csv")
    assert detail_b == str(tmp_path / "jobs_compare_detail.csv")
    assert detail_a != detail_b  # 互不干扰


def test_detail_path_stem_derivation_collision_bump(tmp_path):
    """派生名与输入撞车时自动 _1 递增（P1 冲突守卫）。"""
    input_csv = tmp_path / "report_compare_detail.csv"
    input_csv.write_text("x")
    sw = MagicMock()
    sw.random_seed = None
    sw.append_mode = False
    sw.input_files = [str(input_csv)]
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        inst = _FakeInstance()
    inst.result_path = str(tmp_path / "report.csv")
    inst.case_result_titles = ("testcase_name", "precision_status", "precision_metrics")

    assert inst._derive_compare_detail_path() == str(tmp_path / "report_compare_detail_1.csv")


def test_detail_lifecycle_follows_main_rebuild(tmp_path):
    """检视意见 P2：主表重建/新建 → 上轮明细一并删除；主表续写 → 明细保留。"""
    sw = MagicMock()
    sw.random_seed = None
    sw.append_mode = False
    sw.input_files = []
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        inst = _FakeInstance()
    inst.result_path = str(tmp_path / "out_result.csv")
    inst.case_result_titles = ("testcase_name", "precision_status", "precision_metrics")
    inst._detail.path = inst._derive_compare_detail_path()

    # 预置"上轮"明细文件
    with open(inst._detail.path, "w", newline="") as f:
        csv.writer(f).writerow(["previous_run_data"])

    # 主表新建（main_continued=False）→ 上轮明细被删
    inst._open_compare_detail_context(main_continued=False)
    assert not os.path.exists(inst._detail.path)

    # 再预置 + 主表续写（main_continued=True）→ 明细保留
    with open(inst._detail.path, "w", newline="") as f:
        csv.writer(f).writerow(["previous_run_data"])
    inst._open_compare_detail_context(main_continued=True)
    assert os.path.exists(inst._detail.path)


def test_compare_detail_append_reconcile(fake_instance, tmp_path):
    """--append：表头一致续写；不一致覆写（与结果 CSV 同款对账语义）。"""
    inst = fake_instance
    detail_path = inst._detail.path
    # 预置同表头文件 + 一行旧数据（宽表 23 列）
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COMPARE_DETAIL_HEADER)
        writer.writerow(
            (
                "old_case",
                "dyn",
                0,
                "float16",
                False,
                1,
                1,
                0,
                "cross_check",
                "L2",
                True,
                1.0,
                0.9,
                0.8,
                0,
                0,
                0.0,
                2.0,
                1.2,
                1.2,
                2**-11,
                2**-16,
                "",
            )
        )
    inst.switches.append_mode = True

    inst.write_row(inst.case_result_titles)
    inst.write_row(("case_new", "PASS", {0: _cross_check_metrics(_sample_payload())}))

    inst._detail.file.flush()
    detail = _read_csv(detail_path)
    assert len(detail) == 3  # 表头 + 旧 1 行 + 新 1 行（续写）
    assert detail[1][0] == "old_case"
    assert detail[2][0] == "case_new"

    # 表头不一致（模拟旧版本格式）→ 覆写 + 告警
    with open(detail_path, "w", newline="") as f:
        csv.writer(f).writerow(["legacy_header"])
    inst._detail.file.close()
    inst._detail.writer = None
    inst._detail.file = None
    inst2_rows = ("case_x", "PASS", {0: _cross_check_metrics(_sample_payload())})
    inst.write_row(inst2_rows)
    inst._detail.file.flush()
    detail = _read_csv(detail_path)
    assert detail[0] == list(COMPARE_DETAIL_HEADER)
    assert detail[1][0] == "case_x"
    assert not any(r[0] == "legacy_header" for r in detail)
