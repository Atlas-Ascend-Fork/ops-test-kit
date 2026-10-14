#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
测试 ttk.core_modules.framework_api worker/execute/compare（原 profiling.py 拆分）。

覆盖内容：
- to_non_contiguous_view / default_generate_inputs: 非连续视图与默认输入生成；
- invoke_pre_compare/invoke_custom_compare（钩子契约在 common/comparison/comparison.py）：跳过条件、原地/返回模式、返回值适配；
- evaluate_eager_precision（统一比对入口内联处）/ compare_with_hooks：端到端集成。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.framework_api.input_generate import (
    default_generate_inputs,
    to_non_contiguous_view,
)


def _make_switches(**overrides):
    """构造测试用 SWITCHES 对象，默认 input_distribution='uniform'。"""
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    sw.input_distribution = "uniform"
    for k, v in overrides.items():
        setattr(sw, k, v)
    return sw


def _make_testcase(
    shapes,
    dtypes,
    attrs=None,
    output_tensor_indexes=(),
    pure_output_indexes=None,
    storage_shapes=(),
    view_strides=(),
    view_offsets=(),
    input_data_ranges=None,
):
    """构造一个 MagicMock 测试用例，模拟 flat_tensor_* 系列接口。"""
    case = MagicMock()
    case.flat_tensor_view_shapes = shapes
    case.flat_tensor_dtypes = dtypes
    case.flat_input_data_ranges = input_data_ranges or ()
    case.pure_output_indexes = pure_output_indexes or []
    case.flat_tensor_storage_shapes = storage_shapes
    case.flat_tensor_view_strides = view_strides
    case.flat_tensor_view_offsets = view_offsets

    def flat_storage_side_effect(idx):
        if storage_shapes and idx < len(storage_shapes):
            val = storage_shapes[idx]
            if val is not None:
                return val
        return shapes[idx] if idx < len(shapes) else None

    def flat_stride_side_effect(idx):
        if view_strides and idx < len(view_strides):
            s = view_strides[idx]
            if s is not None and s != ():
                return s
        from ttk.utilities.shape import shape_stride

        v = shapes[idx] if idx < len(shapes) else None
        return shape_stride(v) if v is not None else None

    def flat_offset_side_effect(idx):
        if view_offsets and idx < len(view_offsets):
            val = view_offsets[idx]
            if val is not None:
                return val
        return 0

    case.flat_storage_shape.side_effect = flat_storage_side_effect
    case.flat_view_stride.side_effect = flat_stride_side_effect
    case.flat_view_offset.side_effect = flat_offset_side_effect
    return case


# --- 模块级 compare 辅助函数（供参数化测试引用） ---


def _compare_single_dict(*outputs):
    """返回单个 dict 结果。"""
    return {"pass": True, "precision": 99.5}


class TestDeterministicContract:
    """The level-3 option stays independent from an optional batch relation."""

    @pytest.mark.parametrize("cli_level", [0, 1, 2, 3])
    @pytest.mark.parametrize("case_level", [None, 0, 1, 2, 3])
    def test_level_priority_matrix(self, cli_level, case_level):
        from ttk.core_modules.common.testcase.batch import resolve_deterministic_level

        case = SimpleNamespace(attributes={"batch_deterministic_level": case_level})
        assert resolve_deterministic_level(_make_switches(deterministic_level=cli_level), case) == (
            cli_level or case_level or 0
        )

    @pytest.mark.parametrize("case_level", [True, "3", 3.0, -1, 4])
    def test_invalid_case_level_is_rejected(self, case_level):
        from ttk.core_modules.common.testcase.batch import resolve_deterministic_level

        with pytest.raises(ValueError, match="batch_deterministic_level"):
            resolve_deterministic_level(
                _make_switches(), SimpleNamespace(attributes={"batch_deterministic_level": case_level})
            )

    def test_reused_worker_resets_level_and_retries_failed_set(self):
        from ttk.core_modules.framework_api.worker import _ensure_deterministic_level

        process = SimpleNamespace(storage={})
        backend = MagicMock()
        backend.is_npu.return_value = True
        backend.set_deterministic_level.side_effect = [RuntimeError("transient"), None, None]
        case = SimpleNamespace(testcase_name="case")
        with pytest.raises(RuntimeError, match="level=3.*execution stopped.*CANN.*transient"):
            _ensure_deterministic_level(process, backend, case, 3)
        assert "_deterministic_level" not in process.storage
        _ensure_deterministic_level(process, backend, case, 3)
        _ensure_deterministic_level(process, backend, case, 3)
        _ensure_deterministic_level(process, backend, case, 0)
        assert [call.args[0] for call in backend.set_deterministic_level.call_args_list] == [3, 3, 0]
        assert process.storage["_deterministic_level"] == 0

    @pytest.mark.parametrize("level", [0, 3])
    def test_failed_level_stops_case_before_execution(self, level, caplog):
        from ttk.core_modules.framework_api import worker as module

        process = MagicMock(storage={"_deterministic_level": 1})
        backend = MagicMock()
        backend.set_deterministic_level.side_effect = RuntimeError("unsupported runtime")
        case = SimpleNamespace(is_valid=True, testcase_name="bad_environment", api_name="add", attributes={})
        switches = _make_switches(deterministic_level=level)
        with patch.object(module, "get_global_storage", return_value=switches), patch.object(
            module, "process_context", return_value=process
        ), patch.object(module, "_get_or_create_backend", return_value=backend), patch.object(
            module, "build_single_log_dir", return_value="logs"
        ), patch.object(module, "default_logging_config"), patch.object(module, "_run_profile") as execute:
            result = module.run(case, {}, {}, 0)
        execute.assert_not_called()
        assert result.precision_status == "FAIL"
        assert "execution stopped" in result.eager_precision
        assert "CANN environment" in caplog.text
        assert "_deterministic_level" not in process.storage

    @pytest.mark.parametrize("prepare", [False, True])
    @pytest.mark.parametrize("cli_level", [0, 1, 3])
    @pytest.mark.parametrize(
        "missing",
        [(), ("batch_axis",), ("batch_slice_info",), ("batch_seed",), ("batch_axis", "batch_slice_info", "batch_seed")],
    )
    def test_batch_execution_log(self, prepare, cli_level, missing, caplog):
        from ttk.core_modules.common.instance import InstanceBase

        malformed = SimpleNamespace(is_enabled=True, is_valid=True, attributes={"batch_deterministic_level": "bad"})
        case = SimpleNamespace(
            testcase_name="case",
            is_enabled=True,
            is_valid=True,
            attributes={"batch_deterministic_level": 3},
            batch_axis=((0,),),
            batch_slice_info=((((0, 1, 1),),),),
            batch_seed=(((7,),),),
        )
        for name in missing:
            setattr(case, name, None)
        instance = SimpleNamespace(
            switches=_make_switches(
                test_mode="framework-api",
                deterministic_level=cli_level,
                no_prof=prepare,
                manual_data_dirs=("out",) if prepare else (),
            ),
            _batch_relation_warned=set(),
        )
        with caplog.at_level("INFO"):
            # 派发期逐 case 提示：malformed（坏 level）静默跳过；case 按缺字段打
            InstanceBase._warn_incomplete_batch_relation(instance, malformed)
            InstanceBase._warn_incomplete_batch_relation(instance, case)
            # 重派（deferred / kernel 双池）不重复
            InstanceBase._warn_incomplete_batch_relation(instance, case)
        active = cli_level != 1 and not prepare
        warnings = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == int(bool(missing) and active)
        if warnings:
            assert all(name in caplog.text for name in missing)
            assert "continuing ordinary level-3 execution" in caplog.text
        assert "wrote dumps" not in caplog.text

        # 收尾提示：以 _batch_summaries 非空为准（实际比过才提示）
        instance2 = SimpleNamespace(
            switches=_make_switches(test_mode="framework-api"),
            _batch_summaries={},
        )
        with caplog.at_level("INFO"):
            InstanceBase._compare_batch_summaries(instance2)
        assert "scripts/compare_batch_consistency.py" not in caplog.text
        from ttk.core_modules.common.comparison.batch_consistency import BatchSummary

        instance2._batch_summaries = {"case": BatchSummary(testcase_name="case", batch_consistency_id="g1")}
        with caplog.at_level("INFO"):
            InstanceBase._compare_batch_summaries(instance2)
        assert "scripts/compare_batch_consistency.py" in caplog.text

    @pytest.mark.parametrize("complete", [False, True])
    def test_npu_preprocess_forwards_only_complete_batch_relation(self, complete):
        """Keep preprocessing aligned with ordinary input/golden relation forwarding."""
        from ttk.core_modules.common.invoke import _npu_preprocess_extras
        from ttk.core_modules.common.testcase.batch import BATCH_RELATION_FIELDS

        testcase = SimpleNamespace(
            attributes={"batch_axis": "ignored-as-attribute"},
            testcase_name="case",
            batch_axis=((0,),),
            batch_slice_info=((((0, 1, 1),),),) if complete else None,
            batch_seed=(((7,),),),
        )
        extras = _npu_preprocess_extras(testcase, _make_switches(), SimpleNamespace(overload_params=()))
        forwarded = {name: extras[name] for name in BATCH_RELATION_FIELDS if name in extras}
        assert forwarded == ({name: getattr(testcase, name) for name in BATCH_RELATION_FIELDS} if complete else {})


class TestToNonContiguousView:
    """to_non_contiguous_view: 通过 as_strided 构造非连续视图。"""

    def test_stride_with_offset(self):
        """带偏移的非连续视图，验证数据正确读取。"""
        storage = np.arange(12, dtype=np.float32)
        view_shape = (3, 2)
        view_stride = (4, 2)
        view_offset = 1
        result = to_non_contiguous_view(storage, view_shape, view_stride, view_offset)
        assert result.shape == (3, 2)
        expected = np.array([[1, 3], [5, 7], [9, 11]], dtype=np.float32)
        np.testing.assert_array_equal(np.array(result), expected)


class TestDefaultGenerateInputs:
    """default_generate_inputs: 默认输入张量生成逻辑。"""

    def test_basic_input_generation(self):
        """基本场景：两个 float32 张量正确生成。"""
        case = _make_testcase(shapes=((4,), (4,)), dtypes=("float32", "float32"))
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 2
        assert inputs[0].shape == (4,)
        assert inputs[0].dtype == np.float32
        assert inputs[1].shape == (4,)
        assert inputs[1].dtype == np.float32

    def test_complex32_input_keeps_lane_storage(self):
        """complex32：视图即存储（fp16 + 尾维 2），不按声明形状坍缩成半数据。"""
        case = _make_testcase(shapes=((2, 3),), dtypes=("complex32",))
        inputs = default_generate_inputs(case, _make_switches())
        assert inputs[0].shape == (2, 3, 2)
        assert inputs[0].dtype == np.float16


class TestApplyPreCompareSkip:
    """invoke_pre_compare 的跳过条件：func 缺失 / 空 golden（钩子契约在 common/comparison/comparison.py）。"""

    @pytest.mark.parametrize(
        ("func_present", "result_init", "golden_init"),
        [
            pytest.param(False, [np.array([1, 2])], [np.array([3, 4])], id="no_func"),
            pytest.param(True, [np.array([1, 2])], [], id="empty_golden"),
        ],
    )
    def test_skip_conditions(self, func_present, result_init, golden_init):
        """验证各跳过条件下 result/golden 保持不变。"""
        from ttk.core_modules.common.comparison.comparison import invoke_pre_compare

        case = MagicMock()
        case.testcase_name = "test_0"
        result = list(result_init)
        golden = list(golden_init)
        func = MagicMock(side_effect=AssertionError("should not run")) if func_present else None
        invoke_pre_compare(case, result, golden, func)
        if result_init:
            np.testing.assert_array_equal(result[0], result_init[0])
        if golden_init:
            np.testing.assert_array_equal(golden[0], golden_init[0])


class TestApplyPreCompareInplace:
    """invoke_pre_compare 原地修改模式：pre_compare 直接修改传入数组。"""

    def test_inplace_no_tensor_list(self):
        """无 TensorList 时 pre_compare 原地修改所有输出。"""
        from ttk.core_modules.common.comparison.comparison import invoke_pre_compare

        def pre_compare(*outputs):
            for arr in outputs:
                arr[:] = arr * 2

        case = MagicMock()
        case.testcase_name = "test_0"
        case.output_dist = ()
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        invoke_pre_compare(case, result, golden, pre_compare)
        np.testing.assert_array_equal(result[0], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [6.0, 8.0])


class TestApplyPreCompareReturnMode:
    """invoke_pre_compare 返回模式：pre_compare 返回新列表替换 result/golden。"""

    def test_return_mode_no_tensor_list(self):
        """无 TensorList 时返回值直接替换。"""
        from ttk.core_modules.common.comparison.comparison import invoke_pre_compare

        def pre_compare(output, golden):
            return output * 2, golden * 2

        case = MagicMock()
        case.testcase_name = "test_0"
        case.output_dist = ()
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        invoke_pre_compare(case, result, golden, pre_compare)
        np.testing.assert_array_equal(result[0], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [6.0, 8.0])

    def test_exception_propagates(self):
        """pre_compare 抛出的异常原样向上传播。"""
        from ttk.core_modules.common.comparison.comparison import invoke_pre_compare

        def pre_compare(*outputs):
            raise RuntimeError("boom")

        case = MagicMock()
        case.testcase_name = "test_0"
        case.output_dist = ()
        result = [np.array([1.0])]
        golden = [np.array([2.0])]
        with pytest.raises(RuntimeError, match="boom"):
            invoke_pre_compare(case, result, golden, pre_compare)


class TestPreCompareEndToEnd:
    """统一比对入口（evaluate_eager_precision 内联）：pre→custom 顺序、变换到达钩子、调用方隔离。"""

    def test_pre_compare_transforms_then_custom_wins(self):
        """pre_compare 返回模式变换数据 → custom 收到变换后数据；调用方列表不被改写。"""
        from ttk.core_modules.framework_api import compare as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        calls = []

        def pre_compare(output, golden):
            calls.append("pre_compare")
            return output * 2, golden * 2  # 返回模式

        def custom_compare(output, golden):
            calls.append("compare")
            np.testing.assert_array_equal(output, [2.0, 4.0])  # 变换后到达 custom
            np.testing.assert_array_equal(golden, [20.0, 40.0])
            return {"pass": True, "precision": "HOOKED"}

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0, 2.0])]
        golden = [np.array([10.0, 20.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch.object(prof_module, "SpecResolver") as _sr:
            _sr.return_value.resolve_pre_compare.return_value = pre_compare
            _sr.return_value.resolve_compare.return_value = custom_compare
            prof_module.evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert calls == ["pre_compare", "compare"]
        assert return_struct.eager_precision == "HOOKED"
        assert return_struct.precision_status == "PASS"
        # 调用方列表不被改写（统一入口的隔离语义）
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(golden[0], [10.0, 20.0])


class TestTryCustomCompareSkip:
    """invoke_custom_compare 的跳过条件：返回 None 的各种场景（钩子契约在 common/comparison/comparison.py）。"""

    @pytest.mark.parametrize(
        ("func_present", "result_arr", "golden_arr"),
        [
            pytest.param(False, [np.array([1.0, 2.0])], [np.array([3.0, 4.0])], id="no_func"),
            pytest.param(True, [], [np.array([2.0])], id="empty_result_nps"),
        ],
    )
    def test_skip_returns_none(self, func_present, result_arr, golden_arr):
        from ttk.core_modules.common.comparison.comparison import invoke_custom_compare

        case = MagicMock()
        case.testcase_name = "test_0"
        func = MagicMock(side_effect=AssertionError("should not run")) if func_present else None
        assert invoke_custom_compare(case, result_arr, golden_arr, func) is None


class TestTryCustomCompareAdapt:
    """invoke_custom_compare 返回值适配：dict/列表/嵌套、精度格式化、非法返回报错。"""

    def _make_case(self, api_name="add", output_dist=()):
        case = MagicMock()
        case.api_name = api_name
        case.testcase_name = "test_0"
        case.output_dist = output_dist
        return case

    def _make_switches(self):
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        return sw

    @pytest.mark.parametrize(
        ("output_dist", "compare_fn", "result_arr", "golden_arr", "expected_p", "expected_is_pass"),
        [
            pytest.param(
                (),
                _compare_single_dict,
                [np.array([1.0, 2.0])],
                [np.array([1.0, 2.0])],
                "99.5%",
                True,
                id="single_dict",
            ),
        ],
    )
    def test_dict_list_return(self, output_dist, compare_fn, result_arr, golden_arr, expected_p, expected_is_pass):
        """compare 返回 dict 时的精度拼接、pass 判定与 metrics（pass=True 无失败条目）。"""
        from ttk.core_modules.common.comparison.comparison import invoke_custom_compare

        case = self._make_case(output_dist=output_dist)
        p, log, is_pass, metrics = invoke_custom_compare(case, result_arr, golden_arr, compare_fn)
        assert p == expected_p
        assert is_pass == expected_is_pass
        assert metrics == {}

    @pytest.mark.parametrize(
        ("compare_return", "match", "result_arr", "golden_arr"),
        [
            pytest.param(
                {"precision": 99.0},
                r"missing required key\(s\): 'pass'",
                [np.array([1.0])],
                [np.array([1.0])],
                id="missing_pass",
            ),
        ],
    )
    def test_invalid_return_raises(self, compare_return, match, result_arr, golden_arr):
        """compare 返回非法结构时抛出 ValueError。"""
        from ttk.core_modules.common.comparison.comparison import invoke_custom_compare

        def compare(*outputs):
            return compare_return

        case = self._make_case()
        with pytest.raises(ValueError, match=match):
            invoke_custom_compare(case, result_arr, golden_arr, compare)


class TestCustomCompareEndToEnd:
    """evaluate_eager_precision 端到端：自定义 compare / 回退 / 失败路径。"""

    def test_evaluate_precision_uses_custom_compare(self):
        """Verify evaluate_eager_precision uses custom compare when available."""
        from ttk.core_modules.framework_api import compare as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        def custom_compare(*outputs):
            return {"pass": True, "precision": 100.0}

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch.object(prof_module, "SpecResolver") as _sr:
            _sr.return_value.resolve_pre_compare.return_value = None
            _sr.return_value.resolve_compare.return_value = custom_compare
            prof_module.evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert return_struct.eager_precision == "100.0%"
        assert return_struct.precision_status == "PASS"

    def test_evaluate_precision_falls_back_to_builtin(self):
        """Verify evaluate_eager_precision uses built-in compare when no custom compare."""
        from ttk.core_modules.framework_api import compare as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = None  # no plugin path → no custom compare

        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        captured = {}

        def fake_compare(*args, **kwargs):
            captured["called"] = True
            return "100.0%", "log", True, {}

        with patch.object(prof_module, "compare_with_hooks", side_effect=fake_compare):
            prof_module.evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert captured.get("called") is True
        assert return_struct.eager_precision == "100.0%"

    def test_evaluate_precision_custom_compare_failure(self):
        """Verify custom compare exception → COMPARE_FAILURE."""
        from ttk.core_modules.framework_api import compare as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        def custom_compare(*outputs):
            raise RuntimeError("custom boom")

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch.object(prof_module, "SpecResolver") as _sr:
            _sr.return_value.resolve_pre_compare.return_value = None
            _sr.return_value.resolve_compare.return_value = custom_compare
            prof_module.evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert return_struct.eager_precision == "COMPARE_FAILURE"
        assert return_struct.precision_status == "FAIL"
