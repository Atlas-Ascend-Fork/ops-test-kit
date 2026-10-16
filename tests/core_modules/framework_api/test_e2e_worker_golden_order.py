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
import pytest

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


def test_deferred_golden_early_return_discards_pending_dispatch(monkeypatch):
    """deferred 装载失败早退（dispatch 之后、collect 之前）须弃置预派发句柄。"""
    import threading

    switches = _switches()
    switches.manual_data_dirs = ("replay",)
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    result = [np.array([3.0, 4.0], np.float32)]
    _apply_common_stubs(monkeypatch, switches, inputs, result)
    case = _e2e_case("deferred_discard", first_view_shape=(-1,))
    manual_case = _ManualCaseStub([np.array([3.0, 4.0], np.float32)])
    manual_case.load_goldens = MagicMock(side_effect=RuntimeError("saved golden mismatch"))
    _stub_manual_case(monkeypatch, manual_case)

    # 动态槽门：NPU 后端 + 已注册 npu_preprocess hook（门语义，非本测试焦点）
    monkeypatch.setattr(e2e_worker, "resolve_npu_preprocess_hook", lambda *_a, **_k: object())

    # cross_check + 重叠开启 → _dispatch_xpu 预派发 background 句柄
    monkeypatch.setattr(e2e_worker, "SpecResolver", _FakeSpecResolver)
    monkeypatch.setattr(
        e2e_worker,
        "resolve_tolerance",
        lambda *a, **k: [SimpleNamespace(token="cross_check")],  # noqa: S106  # token 为精度标准名，非口令
    )
    monkeypatch.setattr("ttk.core_modules.common.xpu.xpu_overlap_enabled", lambda: True)
    monkeypatch.setattr(e2e_worker, "e2e_xpu_inputs", lambda tc, ri: [])
    monkeypatch.setattr(e2e_worker, "e2e_xpu_input_names", lambda tc: [])
    monkeypatch.setattr(e2e_worker, "e2e_xpu_input_dtypes", lambda tc: None)

    release = threading.Event()

    def _blocked_remote(**kwargs):
        release.wait(1.5)  # 短于 discard 的 join 预算：早退点观测到 pending 后线程自然结束
        return None, None, None

    monkeypatch.setattr("ttk.core_modules.common.xpu.collect_third_party", _blocked_remote)
    monkeypatch.setattr(e2e_worker, "_execute_eager", lambda *_a, **_k: (list(result), None, None))

    discard_states = []
    _original_discard = e2e_worker._discard_unconsumed

    def _record_discard(xpu_dispatch):
        discard_states.append(xpu_dispatch.pending if xpu_dispatch is not None else None)
        _original_discard(xpu_dispatch)

    monkeypatch.setattr(e2e_worker, "_discard_unconsumed", _record_discard)

    return_struct = FrameworkApiReturnStructure()
    try:
        e2e_worker._run_profile(case, _backend(is_npu=True), {}, {}, 0, switches, return_struct)
    finally:
        release.set()

    assert discard_states == [True]  # 早退点弃置的是仍在执行的 background 句柄


# ══════════════════════════════════════════════════════════════════════════
# manual_xpu_binaries：_dispatch_xpu/_collect_xpu 短路 + e2e inplace 对位
# ══════════════════════════════════════════════════════════════════════════
def test_manual_xpu_header_registered():
    assert "manual_xpu_binaries" in TestcaseE2e.complete_headers


def test_inplace_output_slots_align_with_loader(tmp_path):
    """e2e inplace 对位：output_tensor_indexes 自动回填别名输入槽 (0,)，manual 装载
    与 golden 静态回放同口径（flat_output_view_shapes/flat_output_dtypes）——
    path[0] ↔ 输入槽 0 的 view shape/dtype，与远端三方输出（inplace 返回值）同位。"""
    from ttk.core_modules.common.xpu import load_manual_third_parties

    case = TestcaseE2e()
    case.api_name = "torch.Tensor.relu_"
    case.testcase_name = "e2e_inplace"
    case.tensor_view_shapes = ((8,),)
    case.tensor_dtypes = ("float32",)
    case._auto_fill_inplace_tensor_method()

    assert case.output_tensor_indexes == (0,)  # inplace：输出别名输入槽 0
    assert case.flat_output_view_shapes == ((8,),)
    assert case.flat_output_dtypes == ("float32",)

    path = tmp_path / "gpu.bin"
    expected = np.arange(8, dtype=np.float32)
    expected.tofile(path)
    case.manual_xpu_binaries = (str(path),)

    arrays = load_manual_third_parties(case)

    assert len(arrays) == 1  # 单输出槽（inplace 返回自身），manual 路径与之一一对位
    np.testing.assert_array_equal(arrays[0], expected)


def test_inplace_non_contiguous_view_loads_logical_values(tmp_path):
    """inplace + 非连续 view：装载按逻辑 view shape 对位（XPU 归档写的是返回数组的
    逻辑值，result_nps 亦为逻辑值）——不取存储 shape，防错位。"""
    from ttk.core_modules.common.xpu import load_manual_third_parties

    case = TestcaseE2e()
    case.api_name = "torch.Tensor.relu_"
    case.testcase_name = "e2e_inplace_view"
    case.tensor_view_shapes = ((3, 4),)
    case.tensor_dtypes = ("float32",)
    case._auto_fill_inplace_tensor_method()

    path = tmp_path / "gpu.bin"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected.tofile(path)
    case.manual_xpu_binaries = (str(path),)

    arrays = load_manual_third_parties(case)

    np.testing.assert_array_equal(arrays[0], expected)
    assert arrays[0].shape == (3, 4)  # 逻辑 view shape，非存储展平


def _mx_case(path):
    # 装载器 duck-typing：flat 三件套即可（不依赖完整 TestcaseE2e 解析链）
    return SimpleNamespace(
        testcase_name="e2e_manual_xpu",
        api_name="torch.add",
        flat_manual_xpu_binaries=(str(path),) if path else None,
        flat_output_view_shapes=((8,),),
        flat_output_dtypes=("float32",),
        flat_precision_tolerances=None,
        flat_absolute_precision=None,
        attributes={},
    )


def _mx_switches(golden_mode=None, **overrides):
    values = {
        "golden_mode": golden_mode,
        "plugin_path": (),
        "compare_method": None,
        "dump_config": SimpleNamespace(dump_xpu=False),
        "xpu_perf": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _mx_slot_case(paths):
    # manual-data case 目录槽位形态（load_manual_third_parties 的第二来源）
    return SimpleNamespace(has_third_parties=True, third_party_paths=lambda: list(paths))


def _mx_collect(case, switches, golden_nps=None, ref_nps=None, manual_case=None):
    recorder = []
    process_ctx = SimpleNamespace(notify_status=lambda stage, *_: recorder.append(stage))
    return_struct = SimpleNamespace(xpu_metrics=None)
    third, xpu_results, out_golden, early = e2e_worker._collect_xpu(
        case, switches, process_ctx, manual_case, (), ref_nps, golden_nps, None, return_struct
    )
    return third, xpu_results, out_golden, early, recorder, return_struct


class _FakeSpecResolver:
    def __init__(self, *args, **kwargs):
        pass

    def resolve_tolerance(self):
        return None


@pytest.fixture
def no_remote(monkeypatch):
    def _no_remote(**kwargs):
        raise AssertionError("remote collect must not run when manual_xpu_binaries declared")

    monkeypatch.setattr("ttk.core_modules.common.xpu.collect_third_party", _no_remote)
    monkeypatch.setattr(e2e_worker, "SpecResolver", _FakeSpecResolver)
    monkeypatch.setattr(
        e2e_worker,
        "resolve_tolerance",
        lambda *a, **k: [SimpleNamespace(token="cross_check")],  # noqa: S106  # token 为精度标准名，非凭据
    )
    monkeypatch.setattr(e2e_worker, "dump_goldens", lambda *a, **k: None)
    # manual 装载后的落盘副作用（依赖 root_path/全局 storage），非本组断言对象
    monkeypatch.setattr("ttk.core_modules.common.xpu.dump_xpu_slot_files", lambda *a, **k: None)
    monkeypatch.setattr("ttk.core_modules.common.xpu.dump_manual_xpu_archive", lambda *a, **k: None)


@pytest.mark.parametrize("via_slots", [False, True], ids=["csv-path", "case-dir-slots"])
def test_golden_xpu_mode_short_circuits_remote(tmp_path, no_remote, via_slots):
    """golden-mode XPU + manual（CSV 路径 / case 目录槽位）：golden 位真值源来自本地装载，远端零调用。

    case-dir 形态同时覆盖 e2e 无声明输出槽的 references 取槽路径（shape/dtype
    取设备输出，装载口径同 golden）。
    """
    path = tmp_path / "gpu.bin"
    expected = np.arange(8, dtype=np.float32)
    expected.tofile(path)

    case = _mx_case(None if via_slots else path)
    if via_slots:
        case.flat_output_view_shapes = ()
        case.flat_output_dtypes = ()
    manual_case = _mx_slot_case((str(path),)) if via_slots else None
    third, xpu_results, golden_nps, early, recorder, _rs = _mx_collect(
        case,
        _mx_switches(golden_mode="XPU"),
        golden_nps=[],
        ref_nps=[np.zeros(8, dtype=np.float32)] if via_slots else None,
        manual_case=manual_case,
    )

    np.testing.assert_array_equal(third[0], expected)
    assert xpu_results == {"manual_binary": {"status": "PASS", "api": "manual_xpu_binaries"}}
    np.testing.assert_array_equal(golden_nps[0], expected)  # golden 位回填 = 三方本地装载
    assert early is False
    assert recorder == ["OnLoadManualXpu"]


@pytest.mark.parametrize(
    ("golden_nps", "switches_over", "label"),
    [
        ([np.ones(8, dtype=np.float32)], {}, "cross-check-real-golden"),
        (["SUPPRESSED"], {"dump_config": SimpleNamespace(dump_xpu=True)}, "dump-xpu-placeholder-golden"),
    ],
)
def test_manual_source_short_circuits_remote(tmp_path, no_remote, golden_nps, switches_over, label):
    """manual 声明 + DATA 需求（cross_check 判据 / --dump xpu）：本地装载，远端零调用。"""
    path = tmp_path / "gpu.bin"
    expected = np.arange(8, dtype=np.float32)
    expected.tofile(path)

    third, xpu_results, _golden, early, recorder, _rs = _mx_collect(
        _mx_case(path), _mx_switches(**switches_over), golden_nps=golden_nps
    )

    assert third is not None
    np.testing.assert_array_equal(third[0], expected)
    assert xpu_results["manual_binary"]["status"] == "PASS"
    assert early is False
    assert recorder == ["OnLoadManualXpu"]


@pytest.mark.parametrize(
    ("golden_nps", "ref_nps", "standards", "switches_over", "manual_case_src", "expected_need_data"),
    [
        pytest.param([np.ones(8)], None, ["cross_check"], {}, None, True, id="real-golden-cross-check"),
        pytest.param(["UNSUPPORTED"], [np.ones(8)], ["cross_check"], {}, None, True, id="placeholder-refnps-fallback"),
        pytest.param(["UNSUPPORTED"], [np.ones(8)], [], {"xpu_perf": True}, None, False, id="placeholder-xpu-perf"),
        pytest.param([np.ones(8)], None, [], {}, None, None, id="real-golden-no-trigger"),
        pytest.param(["SUPPRESSED"], None, [], {}, None, None, id="placeholder-no-dtype-source"),
        pytest.param([], None, [], {"golden_mode": "XPU"}, "slots", True, id="golden-xpu-empty-manual-remote"),
    ],
)
def test_undeclared_goes_remote(
    tmp_path, monkeypatch, golden_nps, ref_nps, standards, switches_over, manual_case_src, expected_need_data
):
    """manual 未声明时的采集分流：cross_check 带 DATA；占位 golden 经设备输出兜底
    解析判据；--xpu-perf 与 golden 状态解耦；无触发零采集；golden-XPU 目录无
    xpu 槽回退远端。"""
    case = _mx_case(None)
    sentinel_third = [np.arange(8, dtype=np.float32)]
    called = []

    def _remote(**kwargs):
        called.append(kwargs["need_data"])
        return "remote", sentinel_third, {"remote": {"status": "PASS"}}

    monkeypatch.setattr("ttk.core_modules.common.xpu.collect_third_party", _remote)
    monkeypatch.setattr(e2e_worker, "SpecResolver", _FakeSpecResolver)
    monkeypatch.setattr(
        e2e_worker,
        "resolve_tolerance",
        lambda *a, **k: [SimpleNamespace(token=t) for t in standards],  # noqa: S106  # token 为精度标准名，非凭据
    )
    monkeypatch.setattr(e2e_worker, "dump_goldens", lambda *a, **k: None)
    # 免完整 TestcaseE2e 解析链：远端分支的输入提取器打桩
    monkeypatch.setattr(e2e_worker, "e2e_xpu_inputs", lambda tc, raw: [])
    monkeypatch.setattr(e2e_worker, "e2e_xpu_input_names", lambda tc: [])
    monkeypatch.setattr(e2e_worker, "e2e_xpu_input_dtypes", lambda tc: None)

    manual_case = SimpleNamespace(has_third_parties=False, third_party_paths=lambda: ()) if manual_case_src else None
    third, _xpu_results, _golden, early, recorder, _rs = _mx_collect(
        case, _mx_switches(**switches_over), golden_nps=golden_nps, ref_nps=ref_nps, manual_case=manual_case
    )

    if expected_need_data is None:
        assert called == []
        assert third is None
        assert recorder == []
    else:
        assert third is sentinel_third
        assert early is False
        assert called == [expected_need_data]
        assert recorder == ["OnXpuProfiling"]
