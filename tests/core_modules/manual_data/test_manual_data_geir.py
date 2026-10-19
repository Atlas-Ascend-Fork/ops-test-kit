# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""geir 算子 manual_data 落盘/复跑单元测试（--no-prof / --manual-data-dirs）。"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.common.dump import DumpConfig, ManualDataStore
from ttk.core_modules.geir import worker as geir_worker
from ttk.core_modules.geir.execute import GraphExecResult
from ttk.core_modules.geir.testcase import GeirTestcase
from ttk.switches import SWITCHES


def _geir_case(name="geir_manual_data"):
    """构造一个填好默认字段的 geir 测试用例（GeirTestcase 继承 TestcaseKernel）。"""
    case = GeirTestcase()
    case.testcase_name = name
    case.op_name = "add"
    case.input_shapes = ((2,), None)
    case.input_dtypes = ("float32", "float32")
    case.input_formats = ("ND", "ND")
    case.input_ori_shapes = case.input_shapes
    case.input_ori_formats = case.input_formats
    case.output_shapes = ((2,),)
    case.output_dtypes = ("float32",)
    case.output_formats = ("ND",)
    case.output_ori_shapes = case.output_shapes
    case.output_ori_formats = case.output_formats
    case.output_inplace_indexes = ()
    case.output_shape_unknown_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), None)
    case.precision_tolerances = ((0.001, 0.001),)
    case.absolute_precision = (0.0001,)
    case._input_distribution = (0, 0)
    case._output_distribution = (0,)
    case.is_valid = True
    case.fail_reason = None
    return case


def _process_context():
    return SimpleNamespace(notify_status=lambda *_: None, change_name=lambda *_: None)


class _SpecResolverStub:
    """SpecResolver 等价物：tolerance/pre_compare/compare 均返回 None。"""

    def __init__(self, *_):
        pass

    def resolve_tolerance(self):
        return None

    def resolve_pre_compare(self):
        return None

    def resolve_compare(self):
        return None


def _switches(tmp_path, *, no_prof=False, manual_dirs=()):
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.test_mode = "geir"
    switches.task_profiling = False
    switches.no_prof = no_prof
    switches.plugin_path = ()
    switches.golden_mode = "Enable"
    switches.cst_switches.enabled = True
    switches.dyn_switches.enabled = False
    switches.geir_binary = False
    if manual_dirs:
        switches.manual_data_dirs = tuple(str(d) for d in manual_dirs)
    return switches


def _patch_common(monkeypatch, switches, tmp_path):
    """run() 编排层公共桩：全局存储/进程上下文/日志/设备锁/内存闸。"""
    monkeypatch.setattr(geir_worker, "get_global_storage", lambda: switches)
    monkeypatch.setattr(geir_worker, "process_context", _process_context)
    monkeypatch.setattr(geir_worker, "build_single_log_dir", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(geir_worker, "default_logging_config", lambda *a, **k: None)
    monkeypatch.setattr(geir_worker, "DeviceLock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(geir_worker, "gate_host_memory", lambda *a, **k: None)
    monkeypatch.setattr(geir_worker, "SpecResolver", _SpecResolverStub)
    monkeypatch.setattr(geir_worker, "_reconcile_inputs", lambda *_: None)


def test_geir_no_prof_dumps_and_skips_compile_execute(monkeypatch, tmp_path):
    """geir --no-prof：输入+golden 单遍落盘，图编译与图执行跳过。"""
    case = _geir_case("geir_no_prof")
    switches = _switches(tmp_path, no_prof=True, manual_dirs=[tmp_path])
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    original_input = np.array([1.0, 2.0], np.float32)
    golden = np.array([3.0, 4.0], np.float32)

    def generate_input(context, stored_inputs=None):
        assert stored_inputs is None
        context.input_arrays = (original_input, None)

    def generate_output(context, stored_goldens=None):
        context.golden_arrays = [golden]
        context.output_arrays = (np.ones(2, np.float32),)

    compile_stage = MagicMock(side_effect=AssertionError("geir must not compile"))
    execute_graph = MagicMock(side_effect=AssertionError("geir must not execute"))
    _patch_common(monkeypatch, switches, tmp_path)
    monkeypatch.setattr(geir_worker, "gen_input", generate_input)
    monkeypatch.setattr(geir_worker, "gen_output", generate_output)
    monkeypatch.setattr(geir_worker, "_compile_stage", compile_stage)
    monkeypatch.setattr(geir_worker, "execute_graph", execute_graph)

    result = geir_worker.run(case, {}, {}, 0)

    assert result.precision_status == "MANUAL_DATA_PREPARED"
    assert result.passed is True
    loaded = ManualDataStore(tmp_path).load_case(case, "geir")
    np.testing.assert_array_equal(loaded.inputs[0], original_input)
    assert loaded.inputs[1] is None
    np.testing.assert_array_equal(
        loaded.load_goldens(shapes=case.flat_output_shapes, dtypes=case.flat_output_dtypes)[0],
        golden,
    )


def test_geir_no_prof_xpu_golden_writes_input_only(monkeypatch, tmp_path):
    """golden-mode XPU + --no-prof 对齐 kernel：占位 golden 被过滤，仅落 input。"""
    case = _geir_case("geir_no_prof_xpu")
    switches = _switches(tmp_path, no_prof=True, manual_dirs=[tmp_path])
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    switches.golden_mode = "XPU"
    original_input = np.array([1.0, 2.0], np.float32)
    _patch_common(monkeypatch, switches, tmp_path)

    def generate_input(context, stored_inputs=None):
        context.input_arrays = (original_input, None)

    monkeypatch.setattr(geir_worker, "gen_input", generate_input)
    monkeypatch.setattr(
        geir_worker,
        "_resolve_stage",
        lambda *a: (None, False, True, None),
    )

    result = geir_worker.run(case, {}, {}, 0)

    assert result.passed is True
    assert result.precision_status == "MANUAL_DATA_PREPARED"
    loaded = ManualDataStore(tmp_path).load_case(case, "geir", require_goldens=False)
    np.testing.assert_array_equal(loaded.inputs[0], original_input)
    assert loaded.inputs[1] is None
    assert not loaded.has_goldens


def test_geir_replay_restores_stored_inputs_and_golden(monkeypatch, tmp_path):
    """geir load 方向：存储输入逐模式恢复、golden 装载复用，编译执行照常。"""
    case = _geir_case("geir_replay")
    inputs = [np.array([1.0, 2.0], np.float32), None]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "geir", inputs, golden)
    switches = _switches(tmp_path, manual_dirs=[tmp_path])

    seen = {}

    def generate_input(context, stored_inputs=None):
        seen["stored_inputs"] = stored_inputs
        context.input_arrays = tuple(stored_inputs)

    def generate_output(context, stored_goldens=None):
        seen["stored_goldens"] = stored_goldens
        assert stored_goldens is not None
        context.golden_arrays = list(stored_goldens)
        context.output_arrays = (np.ones(2, np.float32),)

    compile_stage = MagicMock(return_value=(SimpleNamespace(build_dir=str(tmp_path)), "binary", "config"))
    execute_graph = MagicMock(return_value=GraphExecResult(outputs=[np.zeros(2, np.float32)]))
    _patch_common(monkeypatch, switches, tmp_path)
    monkeypatch.setattr(geir_worker, "gen_input", generate_input)
    monkeypatch.setattr(geir_worker, "gen_output", generate_output)
    monkeypatch.setattr(geir_worker, "_compile_stage", compile_stage)
    monkeypatch.setattr(geir_worker, "execute_graph", execute_graph)
    monkeypatch.setattr(geir_worker, "compare_case", lambda *a, **k: ("precision", "log", True, {}))

    result = geir_worker.run(case, {}, {}, 0)

    assert execute_graph.called
    assert result.passed is True
    np.testing.assert_array_equal(seen["stored_inputs"][0], inputs[0])
    assert seen["stored_inputs"][1] is None
    np.testing.assert_array_equal(seen["stored_goldens"][0], golden[0])
    # store 加载方向不被清空（回归 pin：加载前不得 invalidate）
    reloaded = ManualDataStore(tmp_path).load_case(case, "geir")
    np.testing.assert_array_equal(reloaded.inputs[0], inputs[0])


def test_geir_first_run_on_empty_dirs_dumps_dataset(monkeypatch, tmp_path):
    """load 方向目录全空首跑：正常执行并向 store 落盘（后续运行即可加载）。"""
    case = _geir_case("geir_first_run")
    switches = _switches(tmp_path, manual_dirs=[tmp_path])
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    original_input = np.array([1.0, 2.0], np.float32)
    golden = np.array([3.0, 4.0], np.float32)

    def generate_input(context, stored_inputs=None):
        assert stored_inputs is None
        context.input_arrays = (original_input, None)

    def generate_output(context, stored_goldens=None):
        assert stored_goldens is None
        context.golden_arrays = [golden]
        context.output_arrays = (np.ones(2, np.float32),)

    _patch_common(monkeypatch, switches, tmp_path)
    monkeypatch.setattr(geir_worker, "gen_input", generate_input)
    monkeypatch.setattr(geir_worker, "gen_output", generate_output)
    monkeypatch.setattr(
        geir_worker,
        "_compile_stage",
        MagicMock(return_value=(SimpleNamespace(build_dir=str(tmp_path)), "binary", "config")),
    )
    monkeypatch.setattr(geir_worker, "execute_graph", MagicMock(return_value=GraphExecResult(outputs=[np.zeros(2)])))
    monkeypatch.setattr(geir_worker, "compare_case", lambda *a, **k: ("precision", "log", True, {}))

    geir_worker.run(case, {}, {}, 0)

    loaded = ManualDataStore(tmp_path).load_case(case, "geir")
    np.testing.assert_array_equal(loaded.inputs[0], original_input)
    np.testing.assert_array_equal(
        loaded.load_goldens(shapes=case.flat_output_shapes, dtypes=case.flat_output_dtypes)[0],
        golden,
    )


def test_geir_shape_unknown_stored_golden_no_double_append(monkeypatch, tmp_path):
    """shape-unknown 复跑：stored golden 切齐声明槽——逐模式重入不重复追加元数据槽。

    落盘侧数据集含 uint64 元数据 golden 槽；复跑侧 const 模式（flat 未
    扩展）与 dynamic 模式（flat 已扩展）经 _golden_stage 切齐后槽数一致。
    """
    dump_case = _geir_case("shape_unknown_geir")
    dump_case.output_shape_unknown_indexes = (0,)
    dump_case.output_shapes = ((8, 1716),)
    dump_case.output_ori_shapes = dump_case.output_shapes
    dump_case.output_dtypes = ("int32",)
    data_golden = np.arange(8 * 1716, dtype=np.int32).reshape(8, 1716)
    dump_case.append_output_metadata("uint64", "ND", (9,))
    metadata_golden = np.zeros(9, np.uint64)

    store = ManualDataStore(tmp_path)
    store.write_case(
        dump_case,
        "geir",
        [np.array([1.0, 2.0], np.float32), None],
        [data_golden, metadata_golden],
    )

    replay_case = _geir_case("shape_unknown_geir")
    replay_case.output_shape_unknown_indexes = (0,)
    replay_case.output_shapes = ((8, 1716),)
    replay_case.output_ori_shapes = replay_case.output_shapes
    replay_case.output_dtypes = ("int32",)

    loaded = store.load_case(replay_case, "geir")
    stored_goldens = loaded.load_goldens(
        shapes=replay_case.manual_data_golden_shapes,
        dtypes=replay_case.manual_data_golden_dtypes,
    )
    assert len(stored_goldens) == 2  # data + metadata

    switches = SimpleNamespace(golden_mode="Enable", overflow_mode=0, plugin_path=(), short_soc_version="Ascend910_93")
    import ttk.core_modules.kernel.golden as output_generation

    monkeypatch.setattr(output_generation, "get_global_storage", lambda: switches)

    # 首模式（flat 未扩展）
    geir_worker._golden_stage(replay_case, golden_is_xpu=False, stored_goldens=stored_goldens)
    first_pass = list(replay_case.golden_arrays)
    assert len(first_pass) == 2
    np.testing.assert_array_equal(first_pass[0], data_golden)

    # 后续模式（flat 已扩展）：不重复追加元数据槽
    geir_worker._golden_stage(replay_case, golden_is_xpu=False, stored_goldens=stored_goldens)
    assert len(replay_case.golden_arrays) == 2
    np.testing.assert_array_equal(replay_case.golden_arrays[0], data_golden)
    meta = replay_case.golden_arrays[1]
    assert meta.dtype == np.uint64
    assert meta[0] == 0x80000002  # 低 32 位高位标记 | shape 维数（uint64 编码）
    assert tuple(meta[1:3]) == (8, 1716)
