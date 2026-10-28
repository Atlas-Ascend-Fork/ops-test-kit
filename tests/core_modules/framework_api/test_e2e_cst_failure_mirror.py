#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""失败路径镜像单测：_mirror_cst_failure_on_dedup 判据与副作用。

镜像块（_execute_phases）位于 eager 执行之后，异常传播先于镜像赋值——本
helper 在 run() 异常出口按同一判据补镜像。测试以鸭子类型替身覆盖：
双开成立镜像 / 负向条件合并（cst 关、eager 关、非 onnx 域）/ 判据解析
失败不掩盖原异常。
"""

import time
from types import SimpleNamespace

import ttk.core_modules.framework_api.worker as worker_mod
import ttk.utilities.plog as plog_mod
from ttk.core_modules.framework_api.worker import _mirror_cst_failure_on_dedup


class _OnnxAdapter:
    framework = "onnx"

    def eager_graph_alias(self):
        return "cst"


class _TorchAdapter:
    framework = "torch"


def _switches(cst_enabled, eager=True):
    return SimpleNamespace(eager_enabled=eager, cst_switches=SimpleNamespace(enabled=cst_enabled))


def _case(api="onnx.Add"):
    return SimpleNamespace(api_name=api, testcase_name="ut_cst_mirror")


def test_mirror_on_onnx_dual_mode(monkeypatch):
    monkeypatch.setattr(worker_mod, "adapter_for_testcase", lambda _tc: _OnnxAdapter())
    rs = SimpleNamespace(graph_cst_precision=None)
    _mirror_cst_failure_on_dedup(_case(), _switches(cst_enabled=True), rs, "ATC_COMPILE_FAILED: atc rc=255")
    assert rs.graph_cst_precision == "ATC_COMPILE_FAILED: atc rc=255"


def test_no_mirror_outside_dual_mode(monkeypatch):
    """镜像成立的负向条件：cst 关 / eager 关 / 非 onnx 域。"""
    monkeypatch.setattr(worker_mod, "adapter_for_testcase", lambda _tc: _OnnxAdapter())
    rs = SimpleNamespace(graph_cst_precision=None)
    _mirror_cst_failure_on_dedup(_case(), _switches(cst_enabled=False), rs, "boom")
    assert rs.graph_cst_precision is None
    _mirror_cst_failure_on_dedup(_case(), _switches(cst_enabled=True, eager=False), rs, "boom")
    assert rs.graph_cst_precision is None

    monkeypatch.setattr(worker_mod, "adapter_for_testcase", lambda _tc: _TorchAdapter())
    rs = SimpleNamespace(graph_cst_precision=None)
    _mirror_cst_failure_on_dedup(_case(api="torch.add"), _switches(cst_enabled=True), rs, "boom")
    assert rs.graph_cst_precision is None


def test_adapter_resolution_failure_swallowed(monkeypatch):
    def _boom(_tc):
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(worker_mod, "adapter_for_testcase", _boom)
    rs = SimpleNamespace(graph_cst_precision=None)
    # 不抛：镜像判据失败不得掩盖原异常路径
    _mirror_cst_failure_on_dedup(_case(), _switches(cst_enabled=True), rs, "boom")
    assert rs.graph_cst_precision is None


def test_run_failure_mirror_writes_same_attached_text(monkeypatch):
    """run() 异常出口：eager 列与镜像列落同一份 _attach_plog_errors 产物
    （单行折叠 + plog 尾缀）——「两列恒同值」在失败路径成立；镜像内二次
    附加会因 plog 逐次累积而异文，故共用调用点单次产物。"""
    monkeypatch.setattr(
        worker_mod,
        "get_global_storage",
        lambda: SimpleNamespace(
            eager_enabled=True,
            cst_switches=SimpleNamespace(enabled=True),
            test_mode="e2e",
            root_path="ut_root",
            logging_to_file=False,
            no_memory_check=True,
        ),
    )
    monkeypatch.setattr(worker_mod, "process_context", lambda: SimpleNamespace(change_name=lambda _n: None))
    monkeypatch.setattr(worker_mod, "build_single_log_dir", lambda *a, **k: "ut_log_dir")
    monkeypatch.setattr(worker_mod, "default_logging_config", lambda **k: None)
    monkeypatch.setattr(worker_mod, "reseed_per_testcase", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "gate_host_memory", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_get_or_create_backend", lambda _sw: object())
    monkeypatch.setattr(worker_mod, "resolve_deterministic_level", lambda _sw, _tc: 0)
    monkeypatch.setattr(worker_mod, "_ensure_deterministic_level", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_profiling_end_print", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "adapter_for_testcase", lambda _tc: _OnnxAdapter())

    def _raise(*_a, **_k):
        raise RuntimeError("boom\ndetail tail")

    monkeypatch.setattr(worker_mod, "_run_profile", _raise)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(plog_mod, "extract_plog_errors", lambda max_lines=10: ["[ERROR] EZ9999 fake plog"])

    testcase = SimpleNamespace(api_name="onnx.Add", testcase_name="ut_cst_mirror_run", is_valid=True)
    rs = worker_mod.run(testcase, {}, {}, 0)

    expected = "boom | detail tail | plog: [ERROR] EZ9999 fake plog"
    assert rs.precision_status == "FAIL"
    assert rs.eager_precision == expected
    assert rs.graph_cst_precision == expected  # 两列同串（修复点）
    assert "\n" not in rs.graph_cst_precision  # CSV 单元格不带换行
