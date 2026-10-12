# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""geir golden-mode XPU：三方输出升格 golden 位（与 kernel/aclnn/e2e 同构）。

golden 函数旁路不生成；三方采集强制拉取并回填 golden 位；XPU 不可用在
compile/execute 前显式整例落败（GOLDEN_FAILURE/FAIL）。
"""

import types
from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np

import ttk.core_modules.geir.worker as geir_worker
from ttk.switches import get_global_storage


class _Marker(Exception):
    pass


def _testcase():
    return types.SimpleNamespace(
        is_valid=True,
        testcase_name="tc1",
        op_name="Add",
        input_arrays=[np.ones(2, dtype=np.float32)],
        original_input_arrays=[np.ones(2, dtype=np.float32)],
        flat_input_dtypes=("float32",),
        flat_output_dtypes=("float32",),
        flat_precision_tolerances=[],
        flat_absolute_precision=[],
        attributes={},
        input_formats=None,
        input_dtypes=None,
        golden_arrays=None,
    )


@contextmanager
def _null_lock(*_args, **_kwargs):
    yield


@contextmanager
def _xpu_golden_mode():
    switches = get_global_storage()
    prev = switches.golden_mode
    switches.golden_mode = "XPU"
    try:
        yield switches
    finally:
        switches.golden_mode = prev


def _enter_common(stack, gen_output_mock):
    resolver_stub = mock.MagicMock()
    resolver_stub.resolve_tolerance.return_value = None
    for p in (
        mock.patch.object(geir_worker, "process_context"),
        mock.patch.object(geir_worker, "DeviceLock", _null_lock),
        mock.patch.object(geir_worker, "gen_input", return_value=None),
        mock.patch.object(geir_worker, "gen_output", gen_output_mock),
        mock.patch.object(geir_worker, "SpecResolver", return_value=resolver_stub),
        mock.patch.object(geir_worker, "_geir_param_order", return_value=[]),
        mock.patch.object(geir_worker, "_geir_profiling_end_print", return_value=None),
        mock.patch("ttk.core_modules.geir.proto_loader.ProtoLoader"),
    ):
        stack.enter_context(p)


def test_geir_xpu_golden_guard_fails_without_third_party():
    """XPU golden + 三方不可用 → compile 前整例落败；golden 函数不执行。"""
    gen_output = mock.MagicMock()
    with _xpu_golden_mode(), ExitStack() as stack:
        _enter_common(stack, gen_output)
        stack.enter_context(mock.patch.object(geir_worker, "collect_third_party", return_value=(None, None, None)))
        result = geir_worker.run(_testcase(), {}, {}, 0)
    assert result.precision == "GOLDEN_FAILURE"
    assert result.precision_status == "FAIL"
    assert result.passed is False
    gen_output.assert_not_called()  # golden 旁路


def test_geir_xpu_golden_backfills_third_party():
    """XPU golden + 三方在 → 三方输出回填 golden 位；golden 函数不执行。"""
    gen_output = mock.MagicMock()
    third = [np.array([1.0, 2.0], dtype=np.float32)]
    testcase = _testcase()
    with _xpu_golden_mode(), ExitStack() as stack:
        _enter_common(stack, gen_output)
        builder = mock.MagicMock()
        builder.return_value.generate_op_source.side_effect = _Marker  # 回填后截停
        stack.enter_context(mock.patch.object(geir_worker, "GeirGraphBuilder", builder))
        stack.enter_context(mock.patch.object(geir_worker, "collect_third_party", return_value=(None, third, {})))
        geir_worker.run(testcase, {}, {}, 0)  # Marker 被通路 except 捕获
    assert list(testcase.golden_arrays) == third  # 三方输出已升格 golden 位
    gen_output.assert_not_called()  # golden 旁路
