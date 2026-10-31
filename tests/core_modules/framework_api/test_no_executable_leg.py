# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""eager/graph 模式执行开关与失败消息回写的编排级行为（worker 入口）。

- --eager=false 且图模式被后端降级（如 CPU 不支持）→ 显式 FAIL 行
  （修复前两模式全空、precision_status 保持 None，用例静默无判定）；
- eager 默认开 + 图模式降级 → 降级仅跳过图执行，eager 照常出结果；
- --aclgraph 不内置关 eager（想关需显式 --eager=false）；
- 执行/比对异常经顶层兜底与 _run_graph_phase 带消息落结果列，
  _attach_plog_errors 拼 plog [ERROR] 行并单行化。
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.common.dump import DumpConfig
from ttk.core_modules.framework_api import worker as e2e_worker
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.switches import SWITCHES, get_global_storage, set_global_storage


@pytest.fixture(autouse=True)
def _restore_global_storage():
    prev = get_global_storage()
    yield
    set_global_storage(prev)


class _NoOpAdapter:
    framework = "torch"

    def load_api(self, _api_name):
        return (lambda *args, **kwargs: None), False

    def is_inplace(self, _resolved, _is_tensor_method):
        return False

    def run_graph(self, *args, **kwargs):
        return [], None, None, None

    def setup_device_state(self, _backend, _testcase, _process_ctx):
        return None


def _case():
    case = TestcaseE2e()
    case.testcase_name = "no_leg"
    case.api_name = "torch.add"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.golden_api = ""
    case._tensor_list_dist = (0, 0)
    case._pure_output_indexes = [1]
    case._param_plan_cache = object()  # 跳过签名解析（非本测试焦点）
    return case


def _switches(**kw):
    sw = SWITCHES()
    sw.dump_config = DumpConfig()
    sw.no_prof = False
    sw.manual_data_dirs = ()
    sw.cst_switches.enabled = kw.pop("cst", True)  # 图模式开（触发降级分支）
    sw.dyn_switches.enabled = False
    sw.aclgraph_enabled = kw.pop("aclgraph", False)
    for k, v in kw.items():
        setattr(sw, k, v)
    return sw


def _stub_orchestration(monkeypatch):
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    backend = SimpleNamespace(
        device_type=lambda: "cpu",
        has_device=lambda: False,
        is_npu=lambda: False,
        set_device=lambda *_: None,
        inputs_from_numpy=lambda tc, ri: ri,
        supports_graph_mode=lambda: False,
    )
    ctx = SimpleNamespace(
        notify_status=lambda *_: None,
        change_name=lambda *_: None,
        storage={"framework_api_backend": backend},
    )

    def generate(testcase, *_a, **_k):
        testcase.np_storages = list(inputs)
        return list(inputs)

    monkeypatch.setattr(e2e_worker, "process_context", lambda: ctx)
    monkeypatch.setattr(e2e_worker, "gate_host_memory", lambda *_a, **_k: None)
    monkeypatch.setattr(e2e_worker, "build_single_log_dir", lambda *_a, **_k: "unused")
    monkeypatch.setattr(e2e_worker, "default_logging_config", lambda **_k: None)
    monkeypatch.setattr(e2e_worker, "generate_inputs", generate)
    monkeypatch.setattr(e2e_worker, "adapter_for_testcase", lambda *_: _NoOpAdapter())
    monkeypatch.setattr(e2e_worker, "DeviceLock", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(e2e_worker, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_worker, "_profiling_end_print", lambda *_a, **_k: None)
    monkeypatch.setattr(e2e_worker, "dump_inputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_worker, "dump_goldens", lambda *_, **__: None)
    monkeypatch.setattr(e2e_worker, "dump_outputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_worker, "_generate_golden", lambda *_a, **_k: list(golden))
    return inputs


def _run(monkeypatch, switches):
    _stub_orchestration(monkeypatch)
    eager_calls = []

    def execute(*_a, **_k):
        eager_calls.append(1)
        return [np.array([3.0, 4.0], np.float32)], None, None

    monkeypatch.setattr(e2e_worker, "_execute_eager", execute)

    def evaluate(testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct, *_rest):
        return_struct.construct("1.0", "PASS", perf)

    monkeypatch.setattr(e2e_worker, "evaluate_eager_precision", evaluate)

    set_global_storage(switches)
    ret = e2e_worker.run(_case(), {}, {}, 0)
    return ret, eager_calls


def test_eager_off_and_graph_downgraded_fails_case(monkeypatch):
    """--eager=false + 图模式被降级：整例显式 FAIL，不再静默无判定。"""
    ret, eager_calls = _run(monkeypatch, _switches(eager_enabled=False))
    assert eager_calls == []
    assert ret.precision_status == "FAIL"
    assert "no executable leg" in ret.eager_precision


def test_eager_default_on_with_graph_downgraded_still_executes(monkeypatch):
    """eager 默认开 + 图模式降级：仅跳过图执行，eager 照常产出结果。"""
    ret, eager_calls = _run(monkeypatch, _switches())
    assert eager_calls == [1]
    assert ret.precision_status == "PASS"


def test_aclgraph_does_not_disable_eager(monkeypatch):
    """--aclgraph 不内置关 eager（想关需显式 --eager=false）。"""
    ret, eager_calls = _run(monkeypatch, _switches(aclgraph=True))
    assert eager_calls == [1]
    assert ret.precision_status == "PASS"


def test_eager_exception_carries_plog_detail(monkeypatch):
    """eager 异常穿透顶层兜底：eager_precision 附带 plog [ERROR] 行（对齐 kernel/aclnn）。"""
    import time

    _stub_orchestration(monkeypatch)
    sw = _switches()
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "ttk.utilities.plog.extract_plog_errors",
        lambda max_lines=10: ["[ERROR] aclnn tiling check failed"],
    )

    def execute(*_a, **_k):
        raise RuntimeError("eager boom")

    monkeypatch.setattr(e2e_worker, "_execute_eager", execute)
    set_global_storage(sw)
    ret = e2e_worker.run(_case(), {}, {}, 0)

    assert ret.precision_status == "FAIL"
    assert "eager boom" in ret.eager_precision
    assert "[ERROR] aclnn tiling check failed" in ret.eager_precision


# ── 图模式执行失败的消息链路 ─────────────────────────────


def test_graph_phase_captures_escaping_exception(monkeypatch):
    """执行器异常不再穿透顶层：转成该模式失败，err 带异常描述。"""

    def boom(**kwargs):
        raise RuntimeError("capture OOM")

    nps, perf, det, err = e2e_worker._run_graph_phase("aclgraph", boom)
    assert (nps, perf, det) == ([], None, None)
    assert err == "Graph aclgraph execution failed: capture OOM"


def test_graph_phase_passes_through_executor_error():
    """执行器按 4 元组回传内部失败：err 原样透传。"""

    def failed(**kwargs):
        return [], None, None, "Graph static execution failed: compile error"

    nps, perf, det, err = e2e_worker._run_graph_phase("static", failed)
    assert (nps, perf, det) == ([], None, None)
    assert err == "Graph static execution failed: compile error"


def test_attach_plog_errors_appends_error_lines(monkeypatch):
    """plog 有 [ERROR] 行：拼到异常消息末尾，整体单行。"""
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "ttk.utilities.plog.extract_plog_errors",
        lambda max_lines=10: ["[ERROR] TILING check failed", "[ERROR] shape [127] should be [128]"],
    )
    out = e2e_worker._attach_plog_errors("Graph aclgraph execution failed: boom")
    assert out == (
        "Graph aclgraph execution failed: boom | "
        "plog: [ERROR] TILING check failed | [ERROR] shape [127] should be [128]"
    )


def test_attach_plog_errors_flattens_multiline_exception(monkeypatch):
    """多行异常文本：单行化折叠后落列，不带换行。"""
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr("ttk.utilities.plog.extract_plog_errors", lambda max_lines=10: [])
    out = e2e_worker._attach_plog_errors("torchair failed:\n  line one\n  line two")
    assert out == "torchair failed: | line one | line two"
    assert "\n" not in out


def test_attach_plog_errors_without_error_lines_unchanged(monkeypatch):
    """plog 无 [ERROR] 行（占位串/raw 行/空）：原样返回。"""
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "ttk.utilities.plog.extract_plog_errors",
        lambda max_lines=10: ["No plog file found, check ascend log manually"],
    )
    assert e2e_worker._attach_plog_errors("boom") == "boom"


def test_attach_plog_errors_short_circuits_without_err(monkeypatch):
    """err 为空（成功路径）：不触碰 plog，也不 sleep。"""
    from unittest.mock import MagicMock

    mocked = MagicMock(return_value=[])
    monkeypatch.setattr("ttk.utilities.plog.extract_plog_errors", mocked)
    assert e2e_worker._attach_plog_errors(None) is None
    mocked.assert_not_called()


def test_attach_plog_errors_extract_failure_unchanged(monkeypatch):
    """plog 提取异常（读取失败）：原样返回，不放大失败。"""
    import time

    def boom(max_lines=10):
        raise OSError("plog unreadable")

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr("ttk.utilities.plog.extract_plog_errors", boom)
    assert e2e_worker._attach_plog_errors("boom") == "boom"
