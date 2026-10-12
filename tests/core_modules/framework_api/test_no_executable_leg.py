# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""eager/graph 模式执行开关的编排级行为（worker.run 对外入口）。

- --eager=false 且图模式被后端降级（如 CPU 不支持）→ 显式 FAIL 行
  （修复前两模式全空、precision_status 保持 None，用例静默无判定）；
- eager 默认开 + 图模式降级 → 降级仅跳过图执行，eager 照常出结果；
- --aclgraph 不内置关 eager（想关需显式 --eager=false）。
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
        return [], None, None


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
