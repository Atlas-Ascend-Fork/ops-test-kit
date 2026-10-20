#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""manual_data prepare/replay 与 e2e/aclnn profiling 流程的集成测试。"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.aclnn import compare as aclnn_comparison
from ttk.core_modules.aclnn import worker as aclnn_profiling
from ttk.core_modules.aclnn.result import (
    ApiComparisonResult,
    ApiProfilingResult,
)
from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.dump import (
    DumpConfig,
    ManualDataStore,
)
from ttk.core_modules.framework_api import worker as e2e_profiling
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.switches import SWITCHES


def _process_context():
    """构造一个空操作的 process context 桩。"""
    return SimpleNamespace(notify_status=lambda *_: None, change_name=lambda *_: None)


def _e2e_case(name):
    """构造一个填好默认字段的 E2E 测试用例。"""
    case = TestcaseE2e()
    case.testcase_name = name
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
    case._param_plan_cache = object()
    return case


def _aclnn_case(name):
    """构造一个填好默认字段的 ACLNN 测试用例。"""
    case = TestcaseAclnn()
    case.testcase_name = name
    case.api_name = "aclnnAdd"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.output_inplace_indexes = ()
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.scalar_dtypes = ("float32",)
    case.scalar_data_ranges = ((0, 1),)
    case._tensor_list_dist = (0, 0)
    case._scalar_list_dist = (0,)
    case._pure_output_indexes = [1]
    case._param_plan_cache = object()
    return case


def _switches(tmp_path, no_prof=False):
    """构造一份带 manual_data 配置的 SWITCHES（no_prof=True 即 dump 方向）。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = no_prof
    switches.manual_data_dirs = (str(tmp_path),)
    switches.cst_switches.enabled = False
    switches.dyn_switches.enabled = False
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    return switches


class _SpecResolverStub:
    """get_spec_attr(name, attr, path) 键控 monkeypatch 的 SpecResolver 等价物。"""

    def __init__(self, get_attr):
        self._get_attr = get_attr

    def __call__(self, name, pathway, path):
        self._name, self._path = name, path
        return self

    def resolve_tolerance(self):
        return self._get_attr(self._name, "tolerance", self._path)

    def resolve_pre_compare(self):
        return self._get_attr(self._name, "pre_compare", self._path)

    def resolve_compare(self):
        return self._get_attr(self._name, "compare", self._path)


class _NoOpAdapter:
    """测试桩 adapter：load_api 返回哑 callable，图模式返回空。"""

    framework = "torch"

    def load_api(self, _api_name):
        return (lambda *args, **kwargs: None), False

    def is_inplace(self, _resolved, _is_tensor_method):
        return False

    def run_graph(self, *args, **kwargs):
        return [], None, None, None


def _no_op_adapter():
    return _NoOpAdapter()


def test_e2e_prepare_stops_before_api_resolution_and_device_execution(monkeypatch, tmp_path):
    """e2e prepare 模式在 API 解析与设备执行前停止，仅落盘输入数据。"""
    case = _e2e_case("e2e_prepare")
    switches = _switches(tmp_path, no_prof=True)
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    resolve_api = MagicMock(
        side_effect=AssertionError("main API must not be resolved")
    )  # 桩在 TorchAdapter.load_api 缝
    execute = MagicMock(side_effect=AssertionError("main API must not execute"))

    def generate(testcase, *_args, **_kw):
        testcase.np_storages = inputs
        return inputs

    monkeypatch.setattr(e2e_profiling, "process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", generate)

    def generate_golden(*_args, **_kwargs):
        inputs[0][:] = -1
        return golden

    monkeypatch.setattr(e2e_profiling, "generate_golden_data", generate_golden)
    # adapter 获取允许发生在 prepare 模式（TF 的 L1 工厂注册须先于输入生成）；
    # 「API 解析」契约锚定在 load_api 缝——prepare 早退于 load_api 之前。
    from ttk.core_modules.framework_api.torch import TorchAdapter

    monkeypatch.setattr(TorchAdapter, "load_api", resolve_api)
    monkeypatch.setattr(e2e_profiling, "_execute_eager", execute)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    result = FrameworkApiReturnStructure()

    e2e_profiling._run_profile(
        case,
        SimpleNamespace(device_type=lambda: "npu", set_device=lambda *_: None, inputs_from_numpy=lambda tc, ri: ri),
        {},
        {},
        0,
        switches,
        result,
    )

    assert result.precision_status == "PASS"
    assert result.eager_precision == "MANUAL_DATA_PREPARED"
    assert not resolve_api.called
    assert not execute.called
    store = ManualDataStore(tmp_path)
    assert (store.case_dir(case.testcase_name) / "input_0_float32.bin").is_file()
    assert not (store.case_dir(case.testcase_name) / "manifest.json").exists()
    np.testing.assert_array_equal(
        store.load_case(case, "e2e").inputs[0],
        np.array([1.0, 2.0], np.float32),
    )


def test_e2e_dynamic_input_only_prepare_records_none_for_replay(monkeypatch, tmp_path):
    """E2E prepare keeps a dynamic slot unresolved until NPU replay."""
    case = _e2e_case("e2e_dynamic_prepare")
    case.tensor_view_shapes = ((2,), (-1,), (2,))
    case.tensor_dtypes = ("float32", "int32", "float32")
    case.tensor_formats = ("ND", "ND", "ND")
    case.tensor_storage_shapes = ((2,), None, (2,))
    case.tensor_view_offsets = (0, 0, 0)
    case.tensor_view_strides = ((1,), None, (1,))
    case.output_tensor_indexes = (2,)
    case.input_data_ranges = ((-1, 1), (None, None), (-1, 1))
    case._tensor_list_dist = (0, 0, 0)
    case._pure_output_indexes = [2]
    inputs = [np.ones(2, np.float32), None, np.zeros(2, np.float32)]
    switches = _switches(tmp_path, "prepare")
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = False

    def generate(testcase, *_args, **_kwargs):
        testcase.np_storages = list(inputs)
        return list(inputs)

    monkeypatch.setattr(e2e_profiling, "process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", generate)
    monkeypatch.setattr(e2e_profiling, "resolve_npu_preprocess_hook", lambda *_args: object())
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    backend = SimpleNamespace(
        is_npu=lambda: False,
        device_type=lambda: "cpu",
        set_device=lambda *_: None,
        inputs_from_numpy=lambda _case, values: values,
    )
    result = FrameworkApiReturnStructure()

    e2e_profiling._run_profile(case, backend, {}, {}, 0, switches, result)

    assert result.precision_status == "PASS"
    assert result.eager_precision == "MANUAL_DATA_PREPARED"
    case_dir = ManualDataStore(tmp_path).case_dir(case.testcase_name)
    assert (case_dir / "input_1_none.bin").stat().st_size == 0
    assert ManualDataStore(tmp_path).load_case(case, "e2e").inputs[1] is None


def test_e2e_failed_reprepare_invalidates_previous_case(monkeypatch, tmp_path):
    """e2e 重准备输入失败时标记 FAIL 并清除已有用例目录。"""
    case = _e2e_case("e2e_failed_reprepare")
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    store = ManualDataStore(tmp_path)
    store.write_case(case, "e2e", inputs, [np.ones(2, np.float32)])
    switches = _switches(tmp_path, no_prof=True)
    monkeypatch.setattr(e2e_profiling, "process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", MagicMock(side_effect=RuntimeError("input failure")))

    result = FrameworkApiReturnStructure()
    e2e_profiling._run_profile(
        case,
        SimpleNamespace(device_type=lambda: "npu", set_device=lambda *_: None, inputs_from_numpy=lambda tc, ri: ri),
        {},
        {},
        0,
        switches,
        result,
    )

    assert result.precision_status == "FAIL"
    assert not store.case_dir(case.testcase_name).exists()


def test_e2e_full_flow_empty_dir_generates_and_dumps(monkeypatch, tmp_path):
    """e2e 全流程 + 空目录首跑：正常生成、执行，随后 input/golden 落盘 store。"""
    case = _e2e_case("e2e_first_run_dump")
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    switches = _switches(empty_root)
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]

    def generate(testcase, _switches, _backend, _plan, stored_inputs=None, **_kw):
        assert stored_inputs is None
        testcase.np_storages = list(inputs)
        return list(inputs)

    monkeypatch.setattr(e2e_profiling, "process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", generate)
    monkeypatch.setattr(e2e_profiling, "_generate_golden", lambda *_, **__: list(golden))
    monkeypatch.setattr(
        e2e_profiling,
        "generate_golden_data",
        MagicMock(side_effect=AssertionError("full-flow dump must not regenerate golden")),
    )
    monkeypatch.setattr(e2e_profiling, "adapter_for_testcase", lambda *_: _no_op_adapter())
    monkeypatch.setattr(e2e_profiling, "_execute_eager", lambda *_: (golden, None, None))
    monkeypatch.setattr(e2e_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(e2e_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "dump_inputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "dump_goldens", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "dump_outputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "evaluate_eager_precision", lambda *_a, **_k: None)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    backend = SimpleNamespace(
        device_type=lambda: "npu",
        has_device=lambda: False,
        is_npu=lambda: False,
        set_device=lambda *_: None,
        inputs_from_numpy=lambda tc, ri: ri,
    )

    e2e_profiling._run_profile(case, backend, {}, {}, 0, switches, FrameworkApiReturnStructure())

    loaded = ManualDataStore(empty_root).load_case(case, "e2e")
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(
        loaded.load_goldens(shapes=case.flat_tensor_view_shapes[-1:], dtypes=case.flat_tensor_dtypes[-1:])[0],
        golden[0],
    )


def test_e2e_replay_skips_input_and_golden_generation(monkeypatch, tmp_path):
    """e2e replay 模式跳过输入与 golden 生成，直接用落盘数据执行。"""
    case = _e2e_case("e2e_replay")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "e2e", inputs, golden)
    switches = _switches(tmp_path)
    generated = MagicMock()
    evaluated = MagicMock()

    def restore(testcase, _switches, _backend, _plan, stored_inputs=None, **_kw):
        assert stored_inputs is not None
        testcase.np_storages = stored_inputs
        return stored_inputs

    generated.side_effect = restore
    monkeypatch.setattr(e2e_profiling, "process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", generated)
    monkeypatch.setattr(
        e2e_profiling,
        "generate_golden_data",
        MagicMock(side_effect=AssertionError("golden generation must be skipped")),
    )
    monkeypatch.setattr(e2e_profiling, "adapter_for_testcase", lambda *_: _no_op_adapter())
    monkeypatch.setattr(e2e_profiling, "_execute_eager", lambda *_: (golden, None, None))
    monkeypatch.setattr(e2e_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(e2e_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "dump_inputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "dump_goldens", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "dump_outputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "evaluate_eager_precision", evaluated)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    backend = SimpleNamespace(
        device_type=lambda: "npu",
        has_device=lambda: False,
        is_npu=lambda: False,
        set_device=lambda *_: None,
        inputs_from_numpy=lambda tc, ri: ri,
    )

    e2e_profiling._run_profile(case, backend, {}, {}, 0, switches, FrameworkApiReturnStructure())

    assert generated.call_count == 1
    np.testing.assert_array_equal(evaluated.call_args.args[3][0], golden[0])


def test_e2e_replay_custom_compare_receives_restored_inputs(monkeypatch, tmp_path):
    """e2e replay 下自定义 compare 收到含还原输入/属性/CSV 字段的上下文。"""
    case = _e2e_case("e2e_replay_compare_context")
    case.attributes = {"alpha": 1}
    case.original_dict = {"remark": "replay-context"}
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "e2e", inputs, golden)
    switches = _switches(tmp_path)
    switches.plugin_path = (str(tmp_path),)
    captured = {}

    def restore(testcase, _switches, _backend, _plan, stored_inputs=None, **_kw):
        testcase.np_storages = list(stored_inputs)
        testcase.tensors = tuple(stored_inputs)
        return list(stored_inputs)

    def custom_compare(output, expected, *, compare_context):
        captured["context"] = compare_context
        return {
            "pass": bool(np.array_equal(output, expected)),
            "precision": "REPLAY_CONTEXT",
        }

    def spec_attr(_api_name, attribute, _plugin_path):
        return custom_compare if attribute == "compare" else None

    monkeypatch.setattr(e2e_profiling, "process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", restore)
    monkeypatch.setattr(
        e2e_profiling,
        "generate_golden_data",
        MagicMock(side_effect=AssertionError("golden generation must be skipped")),
    )
    from ttk.core_modules.framework_api import compare as _e2e_compare

    monkeypatch.setattr(_e2e_compare, "SpecResolver", _SpecResolverStub(spec_attr))
    monkeypatch.setattr(e2e_profiling, "adapter_for_testcase", lambda *_: _no_op_adapter())
    monkeypatch.setattr(e2e_profiling, "_execute_eager", lambda *_: ([golden[0].copy()], None, None))
    monkeypatch.setattr(e2e_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(e2e_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "dump_inputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "dump_goldens", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "dump_outputs", lambda *_, **__: None)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    backend = SimpleNamespace(
        device_type=lambda: "npu",
        has_device=lambda: False,
        is_npu=lambda: False,
        set_device=lambda *_: None,
        inputs_from_numpy=lambda tc, ri: ri,
    )
    result = FrameworkApiReturnStructure()

    e2e_profiling._run_profile(case, backend, {}, {}, 0, switches, result)

    compare_context = captured["context"]
    assert result.precision_status == "PASS"
    assert result.eager_precision == "REPLAY_CONTEXT"
    np.testing.assert_array_equal(compare_context.input_tensors[0], inputs[0])
    assert compare_context.input_scalars == ()
    assert compare_context.attributes == case.attributes
    assert compare_context.csv_fields == case.original_dict


def test_aclnn_prepare_stops_before_device_execution(monkeypatch, tmp_path):
    """aclnn prepare 模式在设备执行前停止，仅落盘输入/标量数据。"""
    case = _aclnn_case("aclnn_prepare")
    switches = _switches(tmp_path, no_prof=True)
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    scalar = np.array(0.5, np.float32)
    golden = [np.array([3.0, 4.0], np.float32)]
    execute = MagicMock(side_effect=AssertionError("ACLNN must not execute"))
    compare = MagicMock(side_effect=AssertionError("ACLNN must not compare"))

    class Inputs:
        def __init__(self, context):
            self.context = context

        def gen(self):
            self.context.np_storages = inputs
            self.context.tensors = inputs
            self.context.scalars = (scalar,)

    class Golden:
        def __init__(self, context):
            self.context = context

        def gen(self):
            self.context.np_storages[0][:] = -1
            self.context.scalars[0][...] = 9
            self.context.golden_tensors = golden

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "process_context", _process_context)
    monkeypatch.setattr(aclnn_profiling, "OpApiInfoKeeper", lambda: SimpleNamespace(has_api=lambda *_: True))
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)
    monkeypatch.setattr(aclnn_profiling, "GoldenGenerator", Golden)
    monkeypatch.setattr(aclnn_profiling, "Comparator", compare)
    monkeypatch.setattr(aclnn_profiling, "execute_on_device", execute)
    monkeypatch.setattr(aclnn_profiling, "_profiling_end_print", lambda *_: None)

    result = aclnn_profiling.run(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert result.precision == "MANUAL_DATA_PREPARED"
    assert not execute.called
    assert not compare.called
    loaded = ManualDataStore(tmp_path).load_case(case, "aclnn")
    np.testing.assert_array_equal(
        loaded.inputs[0],
        np.array([1.0, 2.0], np.float32),
    )
    assert loaded.scalars[0] == np.array(0.5, np.float32)


def test_aclnn_dynamic_input_only_prepare_records_none_for_replay(monkeypatch, tmp_path):
    """ACLNN prepare persists a None marker without running npu_preprocess."""
    case = _aclnn_case("aclnn_dynamic_prepare")
    case.tensor_view_shapes = ((2,), (-1,), (2,))
    case.tensor_dtypes = ("float32", "int32", "float32")
    case.tensor_formats = ("ND", "ND", "ND")
    case.tensor_storage_shapes = ((2,), None, (2,))
    case.tensor_view_offsets = (0, 0, 0)
    case.tensor_view_strides = ((1,), None, (1,))
    case.output_tensor_indexes = (2,)
    case.input_data_ranges = ((-1, 1), (None, None), (-1, 1))
    case._tensor_list_dist = (0, 0, 0)
    case._pure_output_indexes = [2]
    switches = _switches(tmp_path, "prepare")
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = False
    preprocess = MagicMock(side_effect=AssertionError("npu_preprocess must wait for replay"))

    class Inputs:
        def __init__(self, context):
            self.context = context

        def gen(self):
            values = [np.ones(2, np.float32), None, np.zeros(2, np.float32)]
            self.context.np_storages = values
            self.context.tensors = values
            self.context.scalars = (np.array(0.5, np.float32),)

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "process_context", _process_context)
    monkeypatch.setattr(aclnn_profiling, "OpApiInfoKeeper", lambda: SimpleNamespace(has_api=lambda *_: True))
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)
    monkeypatch.setattr(aclnn_profiling, "_invoke_aclnn_npu_preprocess", lambda *_args: preprocess())
    monkeypatch.setattr(aclnn_profiling, "_profiling_end_print", lambda *_args, **_kw: None)

    result = aclnn_profiling.run(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert result.precision == "MANUAL_DATA_PREPARED"
    assert not preprocess.called
    case_dir = ManualDataStore(tmp_path).case_dir(case.testcase_name)
    assert (case_dir / "input_1_none.bin").stat().st_size == 0
    assert ManualDataStore(tmp_path).load_case(case, "aclnn", require_goldens=False).inputs[1] is None


def test_aclnn_failed_reprepare_invalidates_previous_case(monkeypatch, tmp_path):
    """aclnn 重准备输入失败时标记 INPUT_GEN_FAILURE 并清除用例目录。"""
    case = _aclnn_case("aclnn_failed_reprepare")
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    scalar = [np.array(0.5, np.float32)]
    store = ManualDataStore(tmp_path)
    store.write_case(case, "aclnn", inputs, [np.ones(2, np.float32)], scalars=scalar)
    switches = _switches(tmp_path, no_prof=True)

    class Inputs:
        def __init__(self, _context):
            pass

        def gen(self):
            raise RuntimeError("input failure")

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "process_context", _process_context)
    monkeypatch.setattr(
        aclnn_profiling,
        "OpApiInfoKeeper",
        lambda: SimpleNamespace(has_api=lambda *_: True),
    )
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)

    result = aclnn_profiling.run(case, {}, {}, 0)

    assert result.precision_status == "INPUT_GEN_FAILURE"
    assert not store.case_dir(case.testcase_name).exists()


def test_aclnn_replay_skips_input_and_golden_plugins(monkeypatch, tmp_path):
    """aclnn replay 模式跳过输入与 golden 插件，用落盘数据执行 compare。"""
    case = _aclnn_case("aclnn_replay")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    scalar = [np.array(0.5, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "aclnn", inputs, golden, scalars=scalar)
    switches = _switches(tmp_path)
    restore = MagicMock()

    class Inputs:
        def __init__(self, context):
            self.context = context

        def gen(self, stored_inputs=None, stored_scalars=None):
            restore(stored_inputs, stored_scalars)
            self.context.np_storages = stored_inputs
            self.context.tensors = stored_inputs
            self.context.scalars = stored_scalars

    class Comparator:
        def __init__(self, context, standards=None, third_parties=None):
            self.context = context

        def compare(self):
            np.testing.assert_array_equal(self.context.golden_tensors[0], golden[0])
            return ApiComparisonResult(None).set("100%", "PASS")

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "process_context", _process_context)
    monkeypatch.setattr(aclnn_profiling, "OpApiInfoKeeper", lambda: SimpleNamespace(has_api=lambda *_: True))
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)
    monkeypatch.setattr(
        aclnn_profiling, "GoldenGenerator", MagicMock(side_effect=AssertionError("golden plugin must be skipped"))
    )
    monkeypatch.setattr(aclnn_profiling, "Comparator", Comparator)
    monkeypatch.setattr(aclnn_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(aclnn_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_profiling_end_print", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_dump_input", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_dump_golden", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_dump_output", lambda *_: None)
    monkeypatch.setattr(
        aclnn_profiling,
        "execute_on_device",
        lambda *_: ApiProfilingResult(
            True,
            output_bytes=[np.zeros(2, np.float32).tobytes()],
            output_view_shapes=[(2,)],
        ),
    )

    result = aclnn_profiling.run(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert restore.call_count == 1


def test_aclnn_replay_runs_current_custom_compare(monkeypatch, tmp_path):
    """aclnn replay 下当前自定义 compare 收到还原的输入张量与标量。"""
    case = _aclnn_case("aclnn_replay_custom_compare")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    scalar = [np.array(0.5, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "aclnn", inputs, golden, scalars=scalar)
    switches = _switches(tmp_path)
    switches.plugin_path = (str(tmp_path),)
    captured = {}

    def custom_compare(output, expected, *, compare_context):
        captured["context"] = compare_context
        return {
            "pass": bool(np.array_equal(output, expected)),
            "precision": "REPLAY_CUSTOM",
        }

    class Inputs:
        def __init__(self, context):
            self.context = context

        def gen(self, stored_inputs=None, stored_scalars=None):
            self.context.np_storages = stored_inputs
            self.context.tensors = stored_inputs
            self.context.scalars = stored_scalars

    def spec_attr(_api_name, attribute, _plugin_path):
        return custom_compare if attribute == "compare" else None

    def restore_outputs(comparator):
        comparator._ctx.prof_result.output_bytes = [golden[0].copy()]

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_comparison, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_comparison, "SpecResolver", _SpecResolverStub(spec_attr))
    monkeypatch.setattr(aclnn_comparison.Comparator, "_output_bytes_to_tensors", restore_outputs)
    monkeypatch.setattr(aclnn_profiling, "process_context", _process_context)
    monkeypatch.setattr(
        aclnn_profiling,
        "OpApiInfoKeeper",
        lambda: SimpleNamespace(has_api=lambda *_: True),
    )
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)
    monkeypatch.setattr(
        aclnn_profiling,
        "GoldenGenerator",
        MagicMock(side_effect=AssertionError("golden plugin must be skipped")),
    )
    monkeypatch.setattr(aclnn_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(aclnn_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_profiling_end_print", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_dump_input", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_dump_golden", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "_dump_output", lambda *_: None)
    monkeypatch.setattr(
        aclnn_profiling,
        "execute_on_device",
        lambda *_: ApiProfilingResult(
            True,
            output_bytes=[golden[0].tobytes()],
            output_view_shapes=[golden[0].shape],
        ),
    )

    result = aclnn_profiling.run(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert result.precision == "REPLAY_CUSTOM"
    compare_context = captured["context"]
    np.testing.assert_array_equal(compare_context.input_tensors[0], inputs[0])
    np.testing.assert_array_equal(compare_context.input_scalars[0], scalar[0])


def test_loaded_golden_still_uses_custom_compare(monkeypatch):
    """落盘 golden 仍走自定义 compare 并返回其精度结果。"""
    case = _e2e_case("custom_compare")
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    output = np.array([1.0, 2.0], np.float32)
    loaded_golden = np.array([1.0, 2.0], np.float32)
    custom_compare = MagicMock(return_value={"pass": True, "precision": "CUSTOM_PASS"})

    def spec_attr(_api_name, attribute, _plugin_path):
        return custom_compare if attribute == "compare" else None

    from ttk.core_modules.framework_api import compare as _e2e_compare

    monkeypatch.setattr(_e2e_compare, "SpecResolver", _SpecResolverStub(spec_attr))
    result = FrameworkApiReturnStructure()

    from ttk.core_modules.framework_api import compare as e2e_compare

    e2e_compare.evaluate_eager_precision(case, [], [output], [loaded_golden], switches, None, result)

    assert result.precision_status == "PASS"
    assert result.eager_precision == "CUSTOM_PASS"
    np.testing.assert_array_equal(custom_compare.call_args.args[1], loaded_golden)


def test_loaded_golden_still_uses_current_csv_close_tolerance(monkeypatch):
    """落盘 golden 在 CSV 配置的 close tolerance 下判定通过。"""
    from ttk.core_modules.framework_api import compare as _e2e_compare

    case = _e2e_case("csv_tolerance")
    case.precision_tolerances = ((0.0, 0.5),)
    case.absolute_precision = (0.0,)
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.compare_method = "close"
    monkeypatch.setattr(_e2e_compare, "SpecResolver", _SpecResolverStub(lambda *_: None))
    result = FrameworkApiReturnStructure()

    _e2e_compare.evaluate_eager_precision(
        case,
        [],
        [np.array([0.0, 1.0], np.float32)],
        [np.array([0.0, 0.0], np.float32)],
        switches,
        None,
        result,
    )

    assert result.precision_status == "PASS"
    assert result.eager_precision == "50.0%"
