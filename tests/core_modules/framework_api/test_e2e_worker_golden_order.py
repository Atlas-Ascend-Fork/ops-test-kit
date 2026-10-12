# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""framework_api worker 阶段序 pin：golden 先于设备执行生成。

单卡路径的阶段序契约为：输入生成 → golden 生成/静态 replay 装载 → 设备执行
（eager/graph 模式）→ 延迟装载（动态 shape / XPU / 静态装载失败）→ 比对。
本文件用桩替换设备执行与 golden 生成，记录调用顺序防回归：
- golden 生成失败 → 占位落行（Invoke.golden 捕获），执行照常进行；
- 动态 shape 用例（-1 槽）/静态装载失败 → 执行后按实际 references 重装载。
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.common.dump import DumpConfig
from ttk.core_modules.framework_api import worker as e2e_worker
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.switches import SWITCHES


def _process_context():
    return SimpleNamespace(notify_status=lambda *_: None, change_name=lambda *_: None)


def _e2e_case(name, first_view_shape=(2,)):
    case = TestcaseE2e()
    case.testcase_name = name
    case.api_name = "torch.add"
    case.tensor_view_shapes = (first_view_shape, (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = (first_view_shape, (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.golden_api = ""
    case._tensor_list_dist = (0, 0)
    case._pure_output_indexes = [1]
    case._param_plan_cache = object()
    return case


def _switches():
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = False
    switches.manual_data_dirs = ()
    switches.cst_switches.enabled = False
    switches.dyn_switches.enabled = False
    return switches


class _NoOpAdapter:
    framework = "torch"

    def load_api(self, _api_name):
        return (lambda *args, **kwargs: None), False

    def is_inplace(self, _resolved, _is_tensor_method):
        return False

    def run_graph(self, *args, **kwargs):
        return [], None, None


def _backend(is_npu=False):
    return SimpleNamespace(
        device_type=lambda: "npu" if is_npu else "cpu",
        has_device=lambda: is_npu,
        is_npu=lambda: is_npu,
        set_device=lambda *_: None,
        inputs_from_numpy=lambda tc, ri: ri,
        supports_graph_mode=lambda: False,
    )


def _apply_common_stubs(monkeypatch, switches, inputs, result):
    def generate(testcase, _switches, _backend, _plan, stored_inputs=None, **_kw):
        testcase.np_storages = list(inputs)
        return list(inputs)

    monkeypatch.setattr(e2e_worker, "process_context", _process_context)
    monkeypatch.setattr(e2e_worker, "generate_inputs", generate)
    monkeypatch.setattr(e2e_worker, "adapter_for_testcase", lambda *_: _NoOpAdapter())
    monkeypatch.setattr(e2e_worker, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(e2e_worker, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_worker, "_profiling_end_print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(e2e_worker, "dump_inputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_worker, "dump_goldens", lambda *_, **__: None)
    monkeypatch.setattr(e2e_worker, "dump_outputs", lambda *_, **__: None)
    monkeypatch.setattr(
        e2e_worker,
        "evaluate_eager_precision",
        lambda *args, **_k: None,
    )
    return generate


def test_golden_generated_before_execution(monkeypatch):
    """单卡路径：golden 生成先于设备执行，比对消费同一份 golden。"""
    switches = _switches()
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    result = [np.array([3.0, 4.0], np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    _apply_common_stubs(monkeypatch, switches, inputs, result)
    case = _e2e_case("golden_order")
    order = []
    compared = {}

    monkeypatch.setattr(
        e2e_worker,
        "_generate_golden",
        lambda *_args, **_k: order.append("golden") or list(golden),
    )

    def execute(*_args, **_kwargs):
        order.append("execute")
        return list(result), None, None

    monkeypatch.setattr(e2e_worker, "_execute_eager", execute)

    def evaluate(testcase, raw_inputs, result_nps, golden_nps, *_rest):
        compared["golden"] = golden_nps

    monkeypatch.setattr(e2e_worker, "evaluate_eager_precision", evaluate)
    return_struct = FrameworkApiReturnStructure()
    e2e_worker._run_profile(case, _backend(), {}, {}, 0, switches, return_struct)

    assert order == ["golden", "execute"]
    np.testing.assert_array_equal(compared["golden"][0], golden[0])


def test_golden_failure_placeholder_does_not_block_execution(monkeypatch):
    """golden 生成失败 → 占位落行，执行照常进行。"""
    switches = _switches()
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    result = [np.array([3.0, 4.0], np.float32)]
    _apply_common_stubs(monkeypatch, switches, inputs, result)
    case = _e2e_case("golden_failure")
    compared = {}

    from ttk.core_modules.framework_api import compare as e2e_compare
    from ttk.core_modules.framework_api import golden as e2e_golden

    monkeypatch.setattr(e2e_compare, "process_context", _process_context)
    monkeypatch.setattr(e2e_golden, "generate_golden", MagicMock(side_effect=RuntimeError("golden boom")))
    executed = MagicMock(return_value=(list(result), None, None))
    monkeypatch.setattr(e2e_worker, "_execute_eager", executed)

    def evaluate(testcase, raw_inputs, result_nps, golden_nps, *_rest):
        compared["golden"] = golden_nps

    monkeypatch.setattr(e2e_worker, "evaluate_eager_precision", evaluate)
    return_struct = FrameworkApiReturnStructure()
    e2e_worker._run_profile(case, _backend(), {}, {}, 0, switches, return_struct)

    assert executed.call_count == 1
    assert compared["golden"] == ["GOLDEN_FAILURE"]


class _ManualCaseStub:
    """replay 桩：inputs 还原 + load_goldens 计数与装载形态。"""

    def __init__(self, goldens):
        self.inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
        self._goldens = goldens
        self.load_calls = []

    @property
    def has_goldens(self):
        return True

    def load_goldens(self, references=None, shapes=None, dtypes=None):
        self.load_calls.append(("static" if shapes is not None else "references", tuple(shapes or ())))
        return list(self._goldens)


def _stub_manual_case(monkeypatch, manual_case):
    monkeypatch.setattr(e2e_worker, "resolve_manual_case", lambda *a, **k: manual_case)


def test_static_replay_golden_loaded_once_before_execution(monkeypatch):
    """声明输出 shape 已知 → 执行前静态装载一次，不再重装载。"""
    switches = _switches()
    switches.manual_data_dirs = ("replay",)
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    result = [np.array([3.0, 4.0], np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    _apply_common_stubs(monkeypatch, switches, inputs, result)
    case = _e2e_case("static_replay")
    manual_case = _ManualCaseStub(golden)
    _stub_manual_case(monkeypatch, manual_case)
    order = []

    monkeypatch.setattr(
        e2e_worker,
        "generate_golden_data",
        MagicMock(side_effect=AssertionError("replay must not generate golden")),
    )

    def execute(*_args, **_kwargs):
        order.append("execute")
        return list(result), None, None

    monkeypatch.setattr(e2e_worker, "_execute_eager", execute)
    return_struct = FrameworkApiReturnStructure()
    e2e_worker._run_profile(case, _backend(), {}, {}, 0, switches, return_struct)

    assert len(manual_case.load_calls) == 1
    assert manual_case.load_calls[0] == ("static", ((2,),))  # CSV 声明输出 shape
    assert order == ["execute"]


def test_dynamic_case_defers_replay_load_until_after_execution(monkeypatch):
    """动态 shape 用例（-1 槽）→ 执行后按 references 装载 replay golden。"""
    switches = _switches()
    switches.manual_data_dirs = ("replay",)
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    result = [np.array([3.0, 4.0], np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    _apply_common_stubs(monkeypatch, switches, inputs, result)
    case = _e2e_case("dynamic_replay", first_view_shape=(-1,))
    manual_case = _ManualCaseStub(golden)
    _stub_manual_case(monkeypatch, manual_case)
    compared = {}
    order = []

    # 动态槽门：NPU 后端 + 已注册 npu_preprocess hook（门语义，非本测试焦点）
    monkeypatch.setattr(e2e_worker, "resolve_npu_preprocess_hook", lambda *_a, **_k: object())

    def execute(*_args, **_kwargs):
        order.append("execute")
        return list(result), None, None

    monkeypatch.setattr(e2e_worker, "_execute_eager", execute)

    def evaluate(testcase, raw_inputs, result_nps, golden_nps, *_rest):
        compared["golden"] = golden_nps

    monkeypatch.setattr(e2e_worker, "evaluate_eager_precision", evaluate)
    return_struct = FrameworkApiReturnStructure()
    e2e_worker._run_profile(case, _backend(is_npu=True), {}, {}, 0, switches, return_struct)

    assert len(manual_case.load_calls) == 1
    assert manual_case.load_calls[0][0] == "references"  # 延迟路径：按执行返回 references 装载
    assert order == ["execute"]
    np.testing.assert_array_equal(compared["golden"][0], golden[0])


def test_static_load_failure_falls_back_to_execution_references(monkeypatch):
    """静态装载失败（声明与落盘不符等）→ 执行后按 references 重装载。"""
    switches = _switches()
    switches.manual_data_dirs = ("replay",)
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    result = [np.array([3.0, 4.0], np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    _apply_common_stubs(monkeypatch, switches, inputs, result)
    case = _e2e_case("static_load_failure")
    manual_case = _ManualCaseStub(golden)
    _stub_manual_case(monkeypatch, manual_case)

    def load_goldens(references=None, shapes=None, dtypes=None):
        if shapes is not None:
            manual_case.load_calls.append(("static", tuple(shapes)))
            raise RuntimeError("saved shape mismatch")
        manual_case.load_calls.append(("references", None))
        return list(golden)

    manual_case.load_goldens = load_goldens
    monkeypatch.setattr(
        e2e_worker,
        "generate_golden_data",
        MagicMock(side_effect=AssertionError("replay must not generate golden")),
    )

    def execute(*_args, **_kwargs):
        return list(result), None, None

    monkeypatch.setattr(e2e_worker, "_execute_eager", execute)
    compared = {}

    def evaluate(testcase, raw_inputs, result_nps, golden_nps, *_rest):
        compared["golden"] = golden_nps

    monkeypatch.setattr(e2e_worker, "evaluate_eager_precision", evaluate)
    return_struct = FrameworkApiReturnStructure()
    e2e_worker._run_profile(case, _backend(), {}, {}, 0, switches, return_struct)

    assert [call[0] for call in manual_case.load_calls] == ["static", "references"]
    np.testing.assert_array_equal(compared["golden"][0], golden[0])
