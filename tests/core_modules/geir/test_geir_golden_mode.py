# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""geir golden-mode XPU：三方输出升格 golden 位（与 kernel/aclnn/e2e 同构）。

golden 函数旁路不生成；三方采集强制拉取并回填 golden 位；XPU 不可用在
compile/execute 前显式整例落败（GOLDEN_FAILURE/FAIL）。
"""

import types
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import ttk.core_modules.geir.worker as geir_worker
from ttk.switches import get_global_storage


class _Marker(Exception):
    pass


def _testcase():
    return types.SimpleNamespace(
        is_valid=True,
        testcase_name="tc1",
        op_name="Add",
        input_arrays=[np.ones(2, dtype=np.float32)],
        original_input_arrays=[np.ones(2, dtype=np.float32)],
        flat_input_dtypes=("float32",),
        flat_output_dtypes=("float32",),
        flat_precision_tolerances=[],
        flat_absolute_precision=[],
        attributes={},
        input_formats=None,
        input_dtypes=None,
        golden_arrays=None,
    )


@contextmanager
def _null_lock(*_args, **_kwargs):
    yield


@contextmanager
def _xpu_golden_mode():
    switches = get_global_storage()
    prev = switches.golden_mode
    switches.golden_mode = "XPU"
    try:
        yield switches
    finally:
        switches.golden_mode = prev


def _enter_common(stack, gen_output_mock):
    resolver_stub = mock.MagicMock()
    resolver_stub.resolve_tolerance.return_value = None
    for p in (
        mock.patch.object(geir_worker, "process_context"),
        mock.patch.object(geir_worker, "DeviceLock", _null_lock),
        mock.patch.object(geir_worker, "gen_input", return_value=None),
        mock.patch.object(geir_worker, "gen_output", gen_output_mock),
        mock.patch.object(geir_worker, "SpecResolver", return_value=resolver_stub),
        mock.patch.object(geir_worker, "_geir_param_order", return_value=[]),
        mock.patch.object(geir_worker, "_geir_profiling_end_print", return_value=None),
        mock.patch("ttk.core_modules.geir.proto_loader.ProtoLoader"),
    ):
        stack.enter_context(p)


def test_geir_xpu_golden_guard_fails_without_third_party():
    """XPU golden + 三方不可用 → compile 前整例落败；golden 函数不执行。"""
    gen_output = mock.MagicMock()
    with _xpu_golden_mode(), ExitStack() as stack:
        _enter_common(stack, gen_output)
        stack.enter_context(mock.patch.object(geir_worker, "collect_third_party", return_value=(None, None, None)))
        result = geir_worker.run(_testcase(), {}, {}, 0)
    assert result.precision == "GOLDEN_FAILURE"
    assert result.precision_status == "FAIL"
    assert result.passed is False
    gen_output.assert_not_called()  # golden 旁路


def test_geir_xpu_golden_backfills_third_party():
    """XPU golden + 三方在 → 三方输出回填 golden 位；golden 函数不执行。"""
    gen_output = mock.MagicMock()
    third = [np.array([1.0, 2.0], dtype=np.float32)]
    testcase = _testcase()
    with _xpu_golden_mode(), ExitStack() as stack:
        _enter_common(stack, gen_output)
        builder = mock.MagicMock()
        builder.return_value.generate_op_source.side_effect = _Marker  # 回填后截停
        stack.enter_context(mock.patch.object(geir_worker, "GeirGraphBuilder", builder))
        stack.enter_context(mock.patch.object(geir_worker, "collect_third_party", return_value=(None, third, {})))
        geir_worker.run(testcase, {}, {}, 0)  # Marker 被通路 except 捕获
    assert list(testcase.golden_arrays) == third  # 三方输出已升格 golden 位
    gen_output.assert_not_called()  # golden 旁路


# ══════════════════════════════════════════════════════════════════════════
# manual_xpu 三方短路（CSV 显式路径 / case 目录槽位 → 跳过远端 dispatch）
# ══════════════════════════════════════════════════════════════════════════


def _manual_xpu_case(manual, tmp_path):
    case = SimpleNamespace(
        testcase_name="geir_manual_xpu",
        op_name="Add",
        flat_manual_xpu_binaries=manual,
        flat_output_shapes=((8,),),
        flat_output_dtypes=("float32",),
        original_input_arrays=None,
        input_arrays=(),
        attributes={},
        input_formats=None,
        input_dtypes=None,
    )
    return case


def _switches(golden_mode="Enable"):
    return SimpleNamespace(dump_config=SimpleNamespace(dump_xpu=False), xpu_perf=False, golden_mode=golden_mode)


def _process_ctx(recorder):
    return SimpleNamespace(notify_status=lambda stage, *_: recorder.append(stage))


def _slot_case(paths):
    # manual-data case 目录槽位形态（load_manual_third_parties 的第二来源）
    return SimpleNamespace(has_third_parties=True, third_party_paths=lambda: list(paths))


@pytest.mark.parametrize(
    ("via_slots", "golden_mode", "need_3party"),
    [(False, "Enable", True), (True, "XPU", False)],
    ids=["csv-cross-check", "case-dir-golden-xpu"],
)
def test_manual_xpu_short_circuits_remote_collect(tmp_path, monkeypatch, via_slots, golden_mode, need_3party):
    """manual 已声明 + 需要 DATA（cross_check 判据 / golden-XPU 升格）→ 本地装载，远端门面零调用。"""
    path = tmp_path / "gpu.bin"
    expected = np.arange(8, dtype=np.float32)
    expected.tofile(path)

    case = _manual_xpu_case(None if via_slots else (str(path),), tmp_path)
    manual_case = _slot_case((str(path),)) if via_slots else None
    recorder = []

    def _no_remote(**kwargs):
        raise AssertionError("remote collect must not run when manual_xpu_binaries declared")

    monkeypatch.setattr(geir_worker, "collect_third_party", _no_remote)

    xpu = geir_worker._dispatch_xpu(case, _switches(golden_mode), _process_ctx(recorder), manual_case, need_3party)
    assert not xpu.pending  # manual 源 → ready
    third_parties, xpu_results, provider = xpu.collect()

    np.testing.assert_array_equal(third_parties[0], expected)
    assert xpu_results == {"manual_binary": {"status": "PASS", "api": "manual_xpu_binaries"}}
    assert provider == "manual_binary"
    assert recorder == ["OnLoadManualXpu"]


def test_undeclared_goes_remote(tmp_path, monkeypatch):
    case = _manual_xpu_case(None, tmp_path)
    recorder = []
    sentinel_third = [np.arange(8, dtype=np.float32)]
    sentinel_results = {"remote": {"status": "PASS", "api": "x"}}

    def _remote(**kwargs):
        recorder.append("remote")
        assert kwargs["need_data"] is True
        return "remote", sentinel_third, sentinel_results

    monkeypatch.setattr(geir_worker, "collect_third_party", _remote)
    monkeypatch.setattr(geir_worker, "_proto_input_names", lambda tc: ["x"])  # 免 ProtoLoader 环境依赖
    monkeypatch.setattr(geir_worker, "_geir_param_order", lambda tc: None)  # 免硬件信息探测
    monkeypatch.setattr(geir_worker, "xpu_overlap_enabled", lambda: False)  # 本用例钉同步路径

    xpu = geir_worker._dispatch_xpu(case, _switches(), _process_ctx(recorder), None, True)
    third_parties, xpu_results, provider = xpu.collect()

    assert third_parties is sentinel_third
    assert xpu_results is sentinel_results
    assert provider == "remote"
    assert recorder == ["OnXpuProfiling", "remote"]


def test_no_data_need_with_manual_still_skips(tmp_path):
    """need_data=False（纯 PERF 场景）不短路——manual 无法提供性能数据。"""
    case = _manual_xpu_case(("gpu.bin",), tmp_path)
    recorder = []

    xpu = geir_worker._dispatch_xpu(case, _switches(), _process_ctx(recorder), None, False)
    third_parties, xpu_results, provider = xpu.collect()

    assert third_parties is None
    assert xpu_results is None
    assert provider is None
    assert recorder == []
