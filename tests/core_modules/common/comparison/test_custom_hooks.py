#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""compare 钩子（pre_compare / compare）的哨兵处理：str 哨兵（输出侧在入口
挡、golden 侧在 invoke 内部挡）与 NO_OUTPUT 槽（output None + golden 真值）
不进钩子；双 None 抑制位交由插件钩子自行处理，不禁用钩子。"""

import numpy as np
import pytest

from ttk.core_modules.common.comparison import CompareHooks, compare_with_hooks
from ttk.core_modules.common.comparison.resolve import resolve_tolerance


class _Case:
    testcase_name = "case_x"
    output_dist = None
    tensors = ()
    scalars = ()
    attributes = {}
    original_dict = {}


def _cmp(o0, o1, g0, g1):
    return [
        {"pass": bool(np.array_equal(o0, g0)), "precision": 100},
        {"pass": bool(np.array_equal(o1, g1)), "precision": 100},
    ]


@pytest.mark.parametrize(
    ("outputs", "goldens", "expected_precision", "expected_pass"),
    [
        pytest.param(
            [np.array([1.0]), np.array([2.0])],
            [np.array([1.0]), "GOLDEN_FAILURE"],
            "100.0%,GOLDEN_FAILURE",
            False,
            id="golden_sentinel",
        ),
        pytest.param(["DYN_OFF"], [np.array([1.0])], "DYN_OFF", True, id="output_sentinel"),
    ],
)
def test_str_sentinel_not_customizable(outputs, goldens, expected_precision, expected_pass):
    """str 哨兵不进钩子（输出侧在入口挡、golden 侧在 invoke 内部挡）：占位透传，钩子零调用。"""
    called = []

    def custom_compare(*args):
        called.append(args)
        return [{"pass": True, "precision": 100}]

    precision, _log, passed, _m = compare_with_hooks(
        _Case(),
        outputs,
        goldens,
        resolve_tolerance(None, None, None, ["float32"] * len(outputs), None),
        CompareHooks(output_dtypes=("float32",) * len(outputs), custom_compare=custom_compare),
    )
    assert called == []
    assert precision == expected_precision
    assert passed is expected_pass


def test_custom_compare_runs_when_all_goldens_real():
    """golden 全为真实数组时钩子正常接管（入口 custom 采信）。"""
    outputs = [np.array([1.0]), np.array([2.0])]
    goldens = [np.array([1.0]), np.array([2.0])]
    precision, _log, passed, _m = compare_with_hooks(
        _Case(),
        outputs,
        goldens,
        resolve_tolerance(None, None, None, ["float32", "float32"], None),
        CompareHooks(output_dtypes=("float32", "float32"), custom_compare=_cmp),
    )
    assert precision == "100%,100%"
    assert passed is True


@pytest.mark.parametrize(
    ("outputs", "goldens", "none_positions"),
    [
        pytest.param([np.array([1.0]), np.array([2.0])], [np.array([1.0]), None], (3,), id="golden_none"),
        pytest.param([np.array([1.0]), None], [np.array([1.0]), None], (1, 3), id="both_none"),
    ],
)
def test_custom_compare_runs_with_suppressed_slot(outputs, goldens, none_positions):
    """双 None 抑制位（golden 侧 / 输出侧双侧同 None，如 return_softmax_lse=False
    的 LSE 位）不禁用钩子：插件接管全部输出位，抑制位由插件自行处理。"""
    called = []

    def custom_compare(o0, o1, g0, g1):
        called.append((o0, o1, g0, g1))
        return [
            {"pass": bool(np.array_equal(o0, g0)), "precision": 100},
            {"pass": g1 is None or bool(np.array_equal(o1, g1)), "precision": 100},
        ]

    precision, _log, passed, _m = compare_with_hooks(
        _Case(),
        outputs,
        goldens,
        resolve_tolerance(None, None, None, ["float32", "float32"], None),
        CompareHooks(output_dtypes=("float32", "float32"), custom_compare=custom_compare),
    )
    assert len(called) == 1
    assert all(called[0][pos] is None for pos in none_positions)
    assert precision == "100%,100%"
    assert passed is True


def test_output_none_with_real_golden_still_blocks_hooks():
    """output None + golden 真值（NO_OUTPUT 槽）仍禁用钩子：执行失败的 FAIL
    语义不经插件改写（单槽死位不禁用整组钩子，但死位须是双 None 抑制位）。"""
    called = []

    def custom_compare(o0, o1, g0, g1):
        called.append(1)
        return [
            {"pass": True, "precision": 100},
            {"pass": True, "precision": 100},
        ]

    precision, _log, passed, _m = compare_with_hooks(
        _Case(),
        [np.array([1.0]), None],
        [np.array([1.0]), np.array([2.0])],
        resolve_tolerance(None, None, None, ["float32", "float32"], None),
        CompareHooks(output_dtypes=("float32", "float32"), custom_compare=custom_compare),
    )
    assert called == []
    assert passed is False
    assert "NO_OUTPUT" in precision


def test_failed_custom_compare_has_bounded_diagnostics(caplog):
    """失败 custom 结果的嵌套诊断有界：error_info 截断、diff/metrics 只报总量。"""
    item = {
        "pass": False,
        "precision": 0,
        "error_info": "LSE mismatch",
        "diff_indices": list(range(10000)),
        "metrics": {f"metric_{i}": list(range(1000)) for i in range(100)},
    }
    _p, detail, passed, _m = compare_with_hooks(
        _Case(),
        [np.zeros(1)],
        [np.ones(1)],
        resolve_tolerance(None, None, None, ["float32"], None),
        CompareHooks(output_dtypes=("float32",), custom_compare=lambda *args: item),
    )
    assert passed is False
    assert "LSE mismatch" in detail
    assert "total=10000" in detail
    assert "9999" not in detail
    assert "total=100" in detail
    assert len(detail) < 5000
    assert len(caplog.records) == 1
    assert detail.strip() in caplog.text
    # 结构化条目（CSV precision_metrics / 父进程回显消费）：容器同界（前 20 + total）
    assert _m[0]["standard"] == "custom"
    assert _m[0]["pass"] is False
    assert _m[0]["reason"] == "LSE mismatch"
    assert _m[0]["diff_indices"] == list(range(20))
    assert _m[0]["diff_indices_total"] == 10000
    assert "total=1000" in _m[0]["metric_0"]
    assert not isinstance(_m[0]["metric_0"], list)


def test_failed_custom_compare_without_reason_is_visible(caplog):
    _p, detail, passed, _m = compare_with_hooks(
        _Case(),
        [np.zeros(1)],
        [np.ones(1)],
        resolve_tolerance(None, None, None, ["float32"], None),
        CompareHooks(
            output_dtypes=("float32",),
            custom_compare=lambda *args: {
                "pass": False,
                "precision": 0,
                "metrics": {"standard": "my_check", "pass": True, "max_error": 5},
            },
        ),
    )
    assert not passed
    assert "without error_info" in detail
    assert "case_x" in caplog.text
    assert len(caplog.records) == 1
    assert _m[0]["reason"] == "Custom comparison returned pass=False without error_info"
    # 插件 metrics 的保留键不覆盖判别标记（standard=="custom" 是回显判别依据）
    assert _m[0]["standard"] == "custom"
    assert _m[0]["pass"] is False
    assert _m[0]["max_error"] == 5
    assert "my_check" not in _m[0]
