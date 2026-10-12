# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.common.instance: header assembly & handle_error dispatch.

错误三分支（crash/fail/none）+ 结果行组装的基类回归。"""

from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.common.instance import InstanceBase
from ttk.core_modules.common.process import Task
from ttk.core_modules.common.testcase.base import TestcaseBase


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
        return ("dyn_perf_us", "precision")

    def get_device_info(self, dev_id):
        return ""


@pytest.fixture
def switches():
    sw = MagicMock()
    sw.random_seed = None
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=sw):
        yield sw


def _make_case(name="t0"):
    case = MagicMock(spec=TestcaseBase)
    case.testcase_name = name
    case.pick_data.return_value = (name, "Add")
    case.get_all_visible_headers.return_value = ("testcase_name", "op_name", "api_name", "input_shape")
    return case


def _make_instance(switches):
    ins = _FakeInstance()
    ins.case_input_title = ("testcase_name", "op_name")
    ins.case_result_title = ("dyn_perf_us", "precision")
    return ins


def test_instance_base_is_abstract():
    with pytest.raises(TypeError):
        InstanceBase()


def test_abstract_hooks_required():
    class _Partial(InstanceBase):
        name = "partial"

        def prepare_env(self):
            pass

    with pytest.raises(TypeError):
        _Partial()


def test_reorder_moves_identity_headers_to_front(switches):
    ins = _make_instance(switches)
    out = ins._reorder_input_titles(("testcase_name", "input_shape", "op_name", "api_name", "extra"))
    assert out == ("testcase_name", "op_name", "api_name", "input_shape", "extra")
    assert ins._front_input_count == 3


def test_reorder_noop_when_testcase_name_not_first(switches):
    ins = _make_instance(switches)
    out = ins._reorder_input_titles(("input_shape", "op_name"))
    assert out == ("input_shape", "op_name")
    assert ins._front_input_count == 0


def test_assemble_row_splits_front(switches):
    ins = _make_instance(switches)
    ins._front_input_count = 2
    out = ins._assemble_row(("tc", "op", "extra"), ("perf", "prec"))
    assert out == ("tc", "op", "perf", "prec", "extra")


def test_assemble_row_no_split_when_front_zero(switches):
    ins = _make_instance(switches)
    ins._front_input_count = 0
    out = ins._assemble_row(("tc", "op"), ("perf", "prec"))
    assert out == ("tc", "op", "perf", "prec")


def test_error_row_puts_details_first(switches):
    ins = _make_instance(switches)
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    out = ins._error_row(task, "FAILURE", "boom")
    assert out == ("t0", "Add", "boom", "FAILURE")


def test_error_row_empty_titles(switches):
    """结果列为空时行 = 仅输入列（与旧行为一致：assemble_row(inputs, ())）。"""
    ins = _make_instance(switches)
    ins.case_result_title = ()
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    assert ins._error_row(task, "FAILURE", "boom") == ("t0", "Add")


def test_handle_result_profile_path(switches):
    ins = _make_instance(switches)
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    assert ins.handle_result(task, object()) == ("PASS", "PASS")
    assert ins.recycle_worker(task, object()) is False


def test_handle_task_result_assembles_input_columns(switches):
    """完整结果行 = 输入列 + 结果列（handle_result 只产结果列，组装归基类）。"""
    from unittest.mock import MagicMock

    ins = _make_instance(switches)
    flushed = []
    ins.write_row = flushed.append
    proc = MagicMock()
    proc.get_result.return_value = object()
    proc.get_pid.return_value = 1
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    ins._handle_task_result(task, proc)
    # pick_data → ("t0", "Add")；handle_result → ("PASS", "PASS")
    assert flushed == [("t0", "Add", "PASS", "PASS")]


def test_assemble_result_row_none_passthrough(switches):
    ins = _make_instance(switches)
    task = Task(_make_case(), int, (), task_type="compile_dyn", pool="compile", is_final=False)
    assert ins._assemble_result_row(task, None) is None


def test_handle_error_fail_profile_path(switches):
    ins = _make_instance(switches)
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    out = ins.handle_error(task, "fail", (99, RuntimeError("crashed")))
    assert out == ("t0", "Add", "crashed", "FAILURE")


def test_handle_error_crash_profile_path(switches):
    ins = _make_instance(switches)
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    out = ins.handle_error(task, "crash", ("StageX", 99, SystemError("segv")))
    assert out[2] == "Crashed at profiling stage: segv"
    assert out[3] == "PROFILE_CRASH"


def test_handle_error_none_profile_path_returns_none(switches):
    ins = _make_instance(switches)
    task = Task(_make_case(), int, (), task_type="profile", pool="profile")
    assert ins.handle_error(task, "none", None) is None


def test_handle_error_crash_compile_path_restores_mode(switches):
    """compile 崩溃路径：mode 经 task.compile_mode 还原（原 task.sub_type AttributeError 回归点）。"""
    ins = _make_instance(switches)
    ins.apply_compile_fail_result = MagicMock()
    task = Task(_make_case(), int, (), task_type="compile_bin", pool="compile", is_final=False)
    out = ins.handle_error(task, "crash", ("CompileStage", 99, SystemError("segv")))
    assert ins.apply_compile_fail_result.call_args[0][2] == "Bin"
    assert out is None  # compile_done 默认实现返回 None


def test_handle_error_fail_compile_path_restores_mode(switches):
    """compile 失败路径：mode 经 task.compile_mode 还原。"""
    ins = _make_instance(switches)
    ins.apply_compile_fail_result = MagicMock()
    task = Task(_make_case(), int, (), task_type="compile_cst", pool="compile", is_final=False)
    out = ins.handle_error(task, "fail", (99, RuntimeError("boom")))
    assert ins.apply_compile_fail_result.call_args[0][2] == "Cst"
    assert out is None


def test_output_titles_composes_header(switches):
    ins = _make_instance(switches)
    case = _make_case()
    header = ins.output_titles(case, [])
    assert header == ("testcase_name", "op_name", "api_name", "dyn_perf_us", "precision", "input_shape")
    assert ins.case_result_title == ("dyn_perf_us", "precision")
