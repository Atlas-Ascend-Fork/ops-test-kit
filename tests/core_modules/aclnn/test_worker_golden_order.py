# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""aclnn worker 阶段序 pin：golden 先于设备执行生成。

单卡路径的阶段序契约为：输入生成 → 判据解析 → golden 生成 → 设备执行 →
动态 replay 重装载 → 比对。本文件用桩替换设备执行与 golden 生成，
记录调用顺序防回归：
- golden 生成失败 → 占位落行，执行照常进行；
- 声明输出 shape 与执行返回不一致（动态 shape）→ 执行后按实际 shape
  重新装载 replay golden；
- 执行失败 → 失败原因透传精度列 + FAIL（对齐 kernel，不再 "N/A"）。
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ttk.core_modules.aclnn import worker as aclnn_worker
from ttk.core_modules.aclnn.result import (
    ApiComparisonResult,
    ApiProfilingResult,
)
from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.dump import DumpConfig
from ttk.switches import SWITCHES


def _process_context():
    return SimpleNamespace(notify_status=lambda *_: None, change_name=lambda *_: None)


def _aclnn_case(name, output_view_shape=(2,)):
    case = TestcaseAclnn()
    case.testcase_name = name
    case.api_name = "aclnnAdd"
    case.tensor_view_shapes = ((2,), output_view_shape)
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), output_view_shape)
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.output_inplace_indexes = ()
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.scalar_dtypes = ()
    case.scalar_data_ranges = ()
    case._tensor_list_dist = (0, 0)
    case._scalar_list_dist = ()
    case._pure_output_indexes = [1]
    case._param_plan_cache = object()
    return case


def _switches():
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = False
    switches.manual_data_dirs = ()
    switches.device_count = 1
    return switches


class _Inputs:
    def __init__(self, context):
        self.context = context

    def gen(self, stored_inputs=None, stored_scalars=None):
        self.context.np_storages = [np.ones(2, np.float32), np.zeros(2, np.float32)]
        self.context.tensors = tuple(self.context.np_storages)
        self.context.scalars = ()


def _apply_common_stubs(monkeypatch, switches):
    monkeypatch.setattr(aclnn_worker, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_worker, "process_context", _process_context)
    monkeypatch.setattr(aclnn_worker, "OpApiInfoKeeper", lambda: SimpleNamespace(has_api=lambda *_: True))
    monkeypatch.setattr(aclnn_worker, "InputGenerator", _Inputs)
    monkeypatch.setattr(aclnn_worker, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(aclnn_worker, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(aclnn_worker, "_profiling_end_print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aclnn_worker, "_dump_input", lambda *_: None)
    monkeypatch.setattr(aclnn_worker, "_dump_golden", lambda *_: None)
    monkeypatch.setattr(aclnn_worker, "_dump_output", lambda *_: None)


def test_golden_generated_before_execution(monkeypatch):
    """单卡路径：golden 生成先于设备执行，比对消费同一份 golden。"""
    switches = _switches()
    _apply_common_stubs(monkeypatch, switches)
    case = _aclnn_case("golden_order")
    golden = [np.array([3.0, 4.0], np.float32)]
    order = []
    compared = {}

    class Golden:
        def __init__(self, context):
            self.context = context

        def gen(self):
            order.append("golden")
            self.context.golden_tensors = golden

    class Comparator:
        def __init__(self, context, standards=None, third_parties=None):
            self.context = context

        def compare(self):
            compared["golden"] = self.context.golden_tensors
            return ApiComparisonResult(None).set("100%", "PASS")

    monkeypatch.setattr(aclnn_worker, "GoldenGenerator", Golden)
    monkeypatch.setattr(aclnn_worker, "Comparator", Comparator)
    monkeypatch.setattr(
        aclnn_worker,
        "execute_on_device",
        lambda *_: order.append("execute")
        or ApiProfilingResult(True, output_bytes=[np.zeros(2, np.float32).tobytes()], output_view_shapes=[(2,)]),
    )

    result = aclnn_worker.run(case, {}, {}, 0)

    assert order == ["golden", "execute"]
    assert result.precision_status == "PASS"
    np.testing.assert_array_equal(compared["golden"][0], golden[0])


def test_golden_failure_placeholder_does_not_block_execution(monkeypatch):
    """golden 生成失败 → 占位落行，执行照常进行。"""
    switches = _switches()
    _apply_common_stubs(monkeypatch, switches)
    case = _aclnn_case("golden_failure")
    compared = {}

    class Golden:
        def __init__(self, context):
            self.context = context

        def gen(self):
            raise RuntimeError("golden boom")

    class Comparator:
        def __init__(self, context, standards=None, third_parties=None):
            self.context = context

        def compare(self):
            compared["golden"] = self.context.golden_tensors
            return ApiComparisonResult(None).set("GOLDEN_FAILURE", "FAIL")

    executed = MagicMock(
        return_value=ApiProfilingResult(
            True, output_bytes=[np.zeros(2, np.float32).tobytes()], output_view_shapes=[(2,)]
        )
    )
    monkeypatch.setattr(aclnn_worker, "GoldenGenerator", Golden)
    monkeypatch.setattr(aclnn_worker, "Comparator", Comparator)
    monkeypatch.setattr(aclnn_worker, "execute_on_device", executed)

    result = aclnn_worker.run(case, {}, {}, 0)

    assert executed.call_count == 1
    assert compared["golden"] == ["GOLDEN_FAILURE"]
    assert result.precision_status == "FAIL"


@pytest.mark.parametrize(
    ("api_prof", "expected_precision"),
    [
        ("MULTI_DEVICE_FAILED", "MULTI_DEVICE_FAILED"),
        ([{"API Name": "aclnnAdd", "Avg(us)": 1}], "ACLNN_EXECUTE_FAILED"),
    ],
    ids=["reason-string-passed-through", "profiler-rows-fall-back-to-short-reason"],
)
def test_execute_failure_reason_in_precision_column(monkeypatch, api_prof, expected_precision):
    """执行失败：字符串原因透传精度列；api_prof 为 profiler 行列表时短原因兜底（对齐 kernel）。"""
    switches = _switches()
    _apply_common_stubs(monkeypatch, switches)
    case = _aclnn_case("execute_failure_reason")

    class Golden:
        def __init__(self, context):
            self.context = context

        def gen(self):
            self.context.golden_tensors = [np.zeros(2, np.float32)]

    monkeypatch.setattr(aclnn_worker, "GoldenGenerator", Golden)
    monkeypatch.setattr(
        aclnn_worker,
        "execute_on_device",
        MagicMock(
            return_value=ApiProfilingResult(False, api_prof=api_prof, output_bytes=[b""], output_view_shapes=[(2,)])
        ),
    )

    result = aclnn_worker.run(case, {}, {}, 0)

    assert result.precision == expected_precision
    assert result.precision_status == "FAIL"


def test_execute_failure_plog_detail_in_precision_column(monkeypatch):
    """execute_error（plog 拦截详情）单行化进 precision 列。"""
    switches = _switches()
    _apply_common_stubs(monkeypatch, switches)
    case = _aclnn_case("execute_failure_plog")

    class Golden:
        def __init__(self, context):
            self.context = context

        def gen(self):
            self.context.golden_tensors = [np.zeros(2, np.float32)]

    plog_detail = "[ERROR] TILING check failed\n[ERROR] shape [127] should be [128]"
    monkeypatch.setattr(aclnn_worker, "GoldenGenerator", Golden)
    monkeypatch.setattr(
        aclnn_worker,
        "execute_on_device",
        MagicMock(
            return_value=ApiProfilingResult(
                False,
                api_prof="ACLNN_EXECUTE_FAILED",
                output_bytes=[b""],
                output_view_shapes=[(2,)],
                execute_error=plog_detail,
            )
        ),
    )

    result = aclnn_worker.run(case, {}, {}, 0)

    assert result.precision == "ACLNN_EXECUTE_FAILED: [ERROR] TILING check failed | [ERROR] shape [127] should be [128]"
    assert result.precision_status == "FAIL"


def test_execute_error_channel_wired_in_executor():
    """#178 回归锚：_acl_sequence 6 元组返回 execute_error，do() 两处构造透传（含多卡线程 npu_memory）。"""
    import re
    from pathlib import Path

    from ttk.core_modules.aclnn import execute as aclnn_execute

    source = Path(aclnn_execute.__file__).read_text(encoding="utf-8")
    assert re.search(
        r"return output_byte_arrays, output_view_shapes, status == \"OK\", det_status, npu_memory, execute_error",
        source,
    ), "_acl_sequence must return execute_error"
    assert source.count("execute_error=execute_error") == 2, (
        "both ApiProfilingResult constructions must pass it through"
    )
    assert "execute_error = None" in source, "_acl_sequence must initialize execute_error"
    # 多卡线程路径（skip_context_creation）不得再以 `_` 丢弃 npu_memory
    assert re.search(
        r"output_byte_arrays, output_view_shapes, success, _, npu_memory, execute_error = self\._acl_sequence",
        source,
    ), "thread path must keep npu_memory (was discarded)"


class _ManualCaseStub:
    """replay 桩：inputs/scalars 还原 + load_goldens 计数与装载值。"""

    def __init__(self, goldens):
        self.inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
        self.scalars = []
        self._goldens = goldens
        self.load_calls = []

    @property
    def has_goldens(self):
        return True

    def load_goldens(self, shapes=None, dtypes=None):
        self.load_calls.append((tuple(shapes or ()), tuple(dtypes or ())))
        return list(self._goldens)


def test_dynamic_replay_golden_loaded_statically_once(monkeypatch):
    """声明输出 shape 未知（动态）→ replay golden 按文件名 shape 段装载一次，无执行后二次重载。"""
    switches = _switches()
    switches.manual_data_dirs = ("replay",)
    _apply_common_stubs(monkeypatch, switches)
    case = _aclnn_case("dynamic_replay", output_view_shape=None)
    golden = [np.array([3.0, 4.0], np.float32)]
    manual_case = _ManualCaseStub(golden)
    compared = {}

    class Comparator:
        def __init__(self, context, standards=None, third_parties=None):
            self.context = context

        def compare(self):
            compared["golden"] = self.context.golden_tensors
            return ApiComparisonResult(None).set("100%", "PASS")

    monkeypatch.setattr(aclnn_worker, "resolve_manual_case", lambda *a, **k: manual_case)
    monkeypatch.setattr(
        aclnn_worker,
        "GoldenGenerator",
        MagicMock(side_effect=AssertionError("golden plugin must not run for replay")),
    )
    monkeypatch.setattr(aclnn_worker, "Comparator", Comparator)
    monkeypatch.setattr(
        aclnn_worker,
        "execute_on_device",
        lambda *_: ApiProfilingResult(True, output_bytes=[golden[0].tobytes()], output_view_shapes=[(2,)]),
    )

    result = aclnn_worker.run(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert len(manual_case.load_calls) == 1
    assert manual_case.load_calls[0][0] == (None,)  # 声明 shape 未知；文件自述权威，不再重载
    np.testing.assert_array_equal(compared["golden"][0], golden[0])


def test_static_replay_golden_loaded_once_when_shapes_match(monkeypatch):
    """声明输出 shape 与执行返回一致 → 静态装载一次，不再重装载。"""
    switches = _switches()
    switches.manual_data_dirs = ("replay",)
    _apply_common_stubs(monkeypatch, switches)
    case = _aclnn_case("static_replay")
    golden = [np.array([3.0, 4.0], np.float32)]
    manual_case = _ManualCaseStub(golden)

    class Comparator:
        def __init__(self, context, standards=None, third_parties=None):
            self.context = context

        def compare(self):
            return ApiComparisonResult(None).set("100%", "PASS")

    monkeypatch.setattr(aclnn_worker, "resolve_manual_case", lambda *a, **k: manual_case)
    monkeypatch.setattr(aclnn_worker, "Comparator", Comparator)
    monkeypatch.setattr(
        aclnn_worker,
        "execute_on_device",
        lambda *_: ApiProfilingResult(True, output_bytes=[golden[0].tobytes()], output_view_shapes=[(2,)]),
    )

    result = aclnn_worker.run(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert len(manual_case.load_calls) == 1
    assert manual_case.load_calls[0][0] == ((2,),)
