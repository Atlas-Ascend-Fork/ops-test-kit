# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""kernel 算子 manual_data prepare/replay 与存储输入/golden 的单元测试。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.common.dump import DumpConfig, ManualDataStore
from ttk.core_modules.kernel import golden as output_generation
from ttk.core_modules.kernel import input_generate as input_generation
from ttk.core_modules.kernel import worker as profiling
from ttk.core_modules.kernel.testcase import TestcaseKernel
from ttk.switches import SWITCHES


def _kernel_case(name="kernel_manual_data"):
    """构造一个填好默认字段的 kernel 测试用例。"""
    case = TestcaseKernel()
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
    """构造一个空操作的 process context 桩。"""
    return SimpleNamespace(
        notify_status=lambda *_: None,
        change_name=lambda *_: None,
    )


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


def test_kernel_stored_input_skips_random_generation_and_plugin(monkeypatch):
    """kernel 回放时用存储输入跳过随机生成与输入插件。"""
    case = _kernel_case("kernel_stored_input")
    switches = SimpleNamespace(golden_mode="Enable", plugin_path=())
    plugin = MagicMock(side_effect=AssertionError("input plugin must not run"))
    monkeypatch.setattr(input_generation, "get_global_storage", lambda: switches)
    monkeypatch.setattr(input_generation, "get_plugin_function", plugin)
    stored = [np.array([1.0, 2.0], np.float32), None]

    input_generation.gen_input(case, stored_inputs=stored)

    assert not plugin.called
    np.testing.assert_array_equal(case.input_arrays[0], stored[0])
    assert case.input_arrays[1] is None
    np.testing.assert_array_equal(case.original_input_arrays[0], stored[0])


def test_kernel_stored_golden_skips_golden_plugin(monkeypatch):
    """kernel 回放时用存储 golden 跳过 golden 插件。"""
    case = _kernel_case("kernel_stored_golden")
    case.input_arrays = (np.array([1.0, 2.0], np.float32), None)
    switches = SimpleNamespace(
        golden_mode="Enable",
        overflow_mode=0,
        plugin_path=(),
        short_soc_version="Ascend910_93",
    )
    plugin = MagicMock(side_effect=AssertionError("golden plugin must not run"))
    monkeypatch.setattr(output_generation, "get_global_storage", lambda: switches)
    import ttk.core_modules.common.plugin.resolver as _resolver

    monkeypatch.setattr(_resolver, "get_plugin_function", plugin)
    golden = np.array([3.0, 4.0], np.float32)

    output_generation.gen_output(case, stored_goldens=[golden])

    assert not plugin.called
    np.testing.assert_array_equal(case.golden_arrays[0], golden)
    assert case.output_arrays[0].shape == golden.shape


def test_kernel_shape_unknown_replay_restores_golden_shape(monkeypatch, tmp_path):
    """shape-unknown 算子回放：元数据槽声明补齐使 data golden 保持真值 shape。

    prepare 侧 append_output_metadata 追加 uint64 元数据 golden 槽一并落盘；
    回放侧 fresh CSV 只声明数据槽——manual_data_golden_shapes/dtypes 协议按
    output_shape_unknown_indexes 补齐声明，golden 数与输出数匹配，data
    golden 按 CSV 声明 shape 还原（元数据从 data shape 重新推导）。
    """
    prepare_case = _kernel_case("shape_unknown_replay")
    prepare_case.output_shape_unknown_indexes = (0,)
    prepare_case.output_shapes = ((8, 1716),)
    prepare_case.output_ori_shapes = prepare_case.output_shapes
    prepare_case.output_dtypes = ("int32",)
    data_golden = np.arange(8 * 1716, dtype=np.int32).reshape(8, 1716)
    prepare_case.append_output_metadata("uint64", "ND", (9,))
    metadata_golden = np.zeros(9, np.uint64)

    store = ManualDataStore(tmp_path)
    store.write_case(
        prepare_case,
        "kernel",
        [np.array([1.0, 2.0], np.float32), None],
        [data_golden, metadata_golden],
    )

    replay_case = _kernel_case("shape_unknown_replay")
    replay_case.output_shape_unknown_indexes = (0,)
    replay_case.output_shapes = ((8, 1716),)
    replay_case.output_ori_shapes = replay_case.output_shapes
    replay_case.output_dtypes = ("int32",)

    loaded = store.load_case(replay_case, "kernel")
    stored_goldens = loaded.load_goldens(
        shapes=replay_case.manual_data_golden_shapes,
        dtypes=replay_case.manual_data_golden_dtypes,
    )

    assert stored_goldens[0].shape == (8, 1716)

    switches = SimpleNamespace(
        golden_mode="Enable",
        overflow_mode=0,
        plugin_path=(),
        short_soc_version="Ascend910_93",
    )
    monkeypatch.setattr(output_generation, "get_global_storage", lambda: switches)
    output_generation.gen_output(replay_case, stored_goldens=stored_goldens)

    assert replay_case.golden_arrays[0].shape == (8, 1716)
    np.testing.assert_array_equal(replay_case.golden_arrays[0], data_golden)
    meta = replay_case.golden_arrays[1]
    assert meta.dtype == np.uint64
    assert meta[0] == 0x80000002  # 低 32 位高位标记 | shape 维数（uint64 编码）
    assert tuple(meta[1:3]) == (8, 1716)


def test_kernel_no_prof_dump_snapshots_input_and_stops_before_device(monkeypatch, tmp_path):
    """kernel --no-prof 落盘：快照输入与 golden 并在设备执行前停止。"""
    case = _kernel_case("kernel_prepare")
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = True
    switches.manual_data_dirs = (str(tmp_path),)
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    device_execution = MagicMock(side_effect=AssertionError("Kernel must not execute"))
    prepared = object()
    original_input = np.array([1.0, 2.0], np.float32)

    def generate_input(context, stored_inputs=None):
        assert stored_inputs is None
        context.input_arrays = (original_input, None)

    def generate_output(context, stored_goldens=None):
        assert stored_goldens is None
        original_input[:] = -1
        context.golden_arrays = [np.array([3.0, 4.0], np.float32)]
        context.output_arrays = (np.ones(2, np.float32),)

    monkeypatch.setattr(profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(profiling, "process_context", _process_context)
    monkeypatch.setattr(profiling, "_apply_force_overrides", lambda *_: None)
    monkeypatch.setattr(profiling, "_parse_dynamic_tiling_data", lambda *_: None)
    monkeypatch.setattr(profiling, "_parse_binary_tiling_data", lambda *_: None)
    monkeypatch.setattr(TestcaseKernel, "compile_failed", lambda *_: False)
    monkeypatch.setattr(profiling, "gen_input", generate_input)
    monkeypatch.setattr(profiling, "gen_output", generate_output)
    monkeypatch.setattr(profiling, "SpecResolver", _SpecResolverStub(lambda *_: None))
    monkeypatch.setattr(profiling, "execute_on_device", device_execution)
    monkeypatch.setattr(profiling, "_dump_only_end", lambda *_: prepared)

    result = profiling.run(case, {}, {}, 0)

    assert result is prepared
    assert not device_execution.called
    loaded = ManualDataStore(tmp_path).load_case(case, "kernel")
    np.testing.assert_array_equal(loaded.inputs[0], np.array([1.0, 2.0], np.float32))
    np.testing.assert_array_equal(
        loaded.load_goldens(
            shapes=case.flat_output_shapes,
            dtypes=case.flat_output_dtypes,
        )[0],
        np.array([3.0, 4.0], np.float32),
    )


def test_kernel_replay_keeps_store_intact_and_loads(monkeypatch, tmp_path):
    """kernel load 方向：加载已存用例，不先清空 store（回归 pin：加载前不得 invalidate）。"""
    case = _kernel_case("kernel_replay")
    inputs = [np.array([1.0, 2.0], np.float32), None]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "kernel", inputs, golden)
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = False
    switches.manual_data_dirs = (str(tmp_path),)
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    device_execution = MagicMock()

    def generate_input(context, stored_inputs=None):
        assert stored_inputs is not None
        context.input_arrays = tuple(stored_inputs)

    def generate_output(context, stored_goldens=None):
        assert stored_goldens is not None
        context.golden_arrays = list(stored_goldens)
        context.output_arrays = (np.ones(2, np.float32),)

    monkeypatch.setattr(profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(profiling, "process_context", _process_context)
    monkeypatch.setattr(profiling, "_apply_force_overrides", lambda *_: None)
    monkeypatch.setattr(profiling, "_parse_dynamic_tiling_data", lambda *_: None)
    monkeypatch.setattr(profiling, "_parse_binary_tiling_data", lambda *_: None)
    monkeypatch.setattr(TestcaseKernel, "compile_failed", lambda *_: False)
    monkeypatch.setattr(profiling, "gen_input", generate_input)
    monkeypatch.setattr(profiling, "gen_output", generate_output)
    monkeypatch.setattr(profiling, "SpecResolver", _SpecResolverStub(lambda *_: None))
    monkeypatch.setattr(profiling, "execute_on_device", device_execution)
    monkeypatch.setattr(profiling, "_device_profiling", device_execution)
    from ttk.core_modules.common.xpu import XpuDispatch

    monkeypatch.setattr(profiling, "_dispatch_xpu", lambda *_, **__: XpuDispatch.ready((None, None, None)))
    monkeypatch.setattr(profiling, "_gen_workspaces", lambda *_: None)
    monkeypatch.setattr(profiling, "_dump_golden", lambda *_: None)
    monkeypatch.setattr(profiling, "_compare_and_finalize", lambda *_: object())

    profiling.run(case, {}, {}, 0)

    assert device_execution.called
    reloaded = ManualDataStore(tmp_path).load_case(case, "kernel")
    np.testing.assert_array_equal(reloaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(
        reloaded.load_goldens(shapes=case.flat_output_shapes, dtypes=case.flat_output_dtypes)[0],
        golden[0],
    )
