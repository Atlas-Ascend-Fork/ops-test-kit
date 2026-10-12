# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""kernel worker run() 顶层序列：三条早退路由 + 正常流阶段序。

补四通路 worker 级测试缺口（aclnn/geir/e2e 均有顶层 run 测试，kernel 独缺）：
- 用例无效 / --compile-only / 编译失败 → 早退落行，不进输入生成；
- 正常流：输入生成 → golden → workspace → 设备执行 → 比对的调用序 pin。
manual-data 方向（--no-prof 落盘 / 回放）由 manual_data/test_manual_data_kernel.py 覆盖。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.common.dump import DumpConfig
from ttk.core_modules.kernel import worker as profiling
from ttk.core_modules.kernel.result import KernelReturnStructure
from ttk.core_modules.kernel.testcase import TestcaseKernel
from ttk.switches import SWITCHES


def _kernel_case(name="kernel_run_case"):
    case = TestcaseKernel()
    case.testcase_name = name
    case.op_name = "add"
    case.input_shapes = ((2,), (2,))
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
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.precision_tolerances = ((0.001, 0.001),)
    case.absolute_precision = (0.0001,)
    case._input_distribution = (0, 0)
    case._output_distribution = (0,)
    case.is_valid = True
    case.fail_reason = None
    return case


def _compile_result(compile_result="SUCC"):
    return SimpleNamespace(
        compile_result=compile_result,
        kernel_name="test_kernel",
        block_dim=1,
        compile_time=0.1,
        tiling_key=0,
        tiling_result=SimpleNamespace(tiling_time=(0.0,)),
        workspaces=(),
        debug_buf_size=0,
    )


def _switches():
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = False
    switches.manual_data_dirs = ()
    switches.compile_only = False
    return switches


class _StatusRecorder:
    def __init__(self):
        self.statuses = []

    def notify_status(self, status, *args):
        self.statuses.append(status)

    def change_name(self, *args):
        pass


class _SpecResolverStub:
    def __init__(self, *_args):
        pass

    def resolve_tolerance(self):
        return None

    def resolve_pre_compare(self):
        return None

    def resolve_compare(self):
        return None


def _apply_common_stubs(monkeypatch, switches, recorder):
    monkeypatch.setattr(profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(profiling, "process_context", lambda: recorder)
    monkeypatch.setattr(profiling, "default_logging_config", lambda **_kwargs: None)
    monkeypatch.setattr(profiling, "gate_host_memory", lambda *_args: None)
    monkeypatch.setattr(profiling, "__apply_force_overrides", lambda *_args: None)
    monkeypatch.setattr(profiling, "__parse_dynamic_tiling_data", lambda *_args: None)
    monkeypatch.setattr(profiling, "__parse_binary_tiling_data", lambda *_args: None)
    monkeypatch.setattr(profiling, "__profiling_end_print", lambda *_args: None)
    monkeypatch.setattr(profiling, "__compile_only_end_print", lambda *_args: None)
    monkeypatch.setattr(profiling, "__dump_input", lambda *_args: None)
    monkeypatch.setattr(profiling, "__dump_golden", lambda *_args: None)
    monkeypatch.setattr(profiling, "SpecResolver", _SpecResolverStub)


def _apply_compile_results(case, compile_result="SUCC"):
    for mode in ("dyn", "cst", "bin"):
        setattr(case, f"{mode}_compile_result", _compile_result(compile_result))


def test_run_invalid_case_stops_before_input_generation(monkeypatch):
    """用例无效 → OnParseParameters 后早退落行，不进输入生成。"""
    switches = _switches()
    recorder = _StatusRecorder()
    _apply_common_stubs(monkeypatch, switches, recorder)
    case = _kernel_case("run_invalid")
    case.is_valid = False
    case.fail_reason = "OP_NAME_MISSING"
    _apply_compile_results(case)
    gen_input = MagicMock(side_effect=AssertionError("input generation must not run"))

    monkeypatch.setattr(profiling, "gen_input", gen_input)

    result = profiling.run(case, {}, {}, 0)

    assert isinstance(result, KernelReturnStructure)
    assert recorder.statuses == ["OnParseParameters"]
    assert not gen_input.called


def test_run_compile_only_stops_before_input_generation(monkeypatch):
    """--compile-only → 编译结论落行（COMPILE_ONLY），不进输入生成与执行。"""
    switches = _switches()
    switches.compile_only = True
    recorder = _StatusRecorder()
    _apply_common_stubs(monkeypatch, switches, recorder)
    case = _kernel_case("run_compile_only")
    _apply_compile_results(case)
    gen_input = MagicMock(side_effect=AssertionError("input generation must not run"))
    monkeypatch.setattr(profiling, "gen_input", gen_input)
    monkeypatch.setattr(profiling, "handle_profiling_result", lambda *_args: True)

    result = profiling.run(case, {}, {}, 0)

    assert isinstance(result, KernelReturnStructure)
    assert result.perf_status == "COMPILE_ONLY"
    assert recorder.statuses == ["OnParseParameters"]
    assert not gen_input.called


def test_run_compile_failed_stops_before_input_generation(monkeypatch):
    """三模式编译全失败 → 编译失败早退，不进输入生成与执行。"""
    switches = _switches()
    recorder = _StatusRecorder()
    _apply_common_stubs(monkeypatch, switches, recorder)
    case = _kernel_case("run_compile_failed")
    _apply_compile_results(case, compile_result="FAIL")
    assert case.compile_failed() is True
    gen_input = MagicMock(side_effect=AssertionError("input generation must not run"))
    monkeypatch.setattr(profiling, "gen_input", gen_input)
    monkeypatch.setattr(profiling, "handle_profiling_result", lambda *_args: True)

    result = profiling.run(case, {}, {}, 0)

    assert isinstance(result, KernelReturnStructure)
    assert recorder.statuses == ["OnParseParameters"]
    assert not gen_input.called


def test_run_normal_flow_stage_order(monkeypatch):
    """正常流调用序：输入生成 → golden → 设备执行 → 比对；阶段状态依序推进。"""
    switches = _switches()
    recorder = _StatusRecorder()
    _apply_common_stubs(monkeypatch, switches, recorder)
    case = _kernel_case("run_normal")
    _apply_compile_results(case)
    order = []

    def generate_input(context, stored_inputs=None):
        order.append("gen_input")
        context.input_arrays = (np.ones(2, np.float32), np.ones(2, np.float32))

    def generate_output(context, stored_goldens=None):
        assert "gen_input" in order
        order.append("gen_output")
        context.golden_arrays = [np.array([3.0, 4.0], np.float32)]
        context.output_arrays = (np.ones(2, np.float32),)

    def device_profiling(context, *args):
        assert "gen_output" in order
        order.append("device")

    def compare_and_finalize(context, *args, **kwargs):
        assert "device" in order
        order.append("compare")
        return object()

    monkeypatch.setattr(profiling, "gen_input", generate_input)
    monkeypatch.setattr(profiling, "gen_output", generate_output)
    monkeypatch.setattr(profiling, "_device_profiling", device_profiling)
    monkeypatch.setattr(profiling, "_compare_and_finalize", compare_and_finalize)

    profiling.run(case, {}, {}, 0)

    assert order == ["gen_input", "gen_output", "device", "compare"]
    statuses = recorder.statuses
    assert statuses.index("OnParseParameters") < statuses.index("OnGenInput")
    assert statuses.index("OnGenInput") < statuses.index("OnGenGolden")
    assert statuses.index("OnGenGolden") < statuses.index("OnGenWorkspace")
    assert "OnAcquireLock" not in statuses
    assert case.input_arrays is not None
    assert case.golden_arrays[0].shape == (2,)
