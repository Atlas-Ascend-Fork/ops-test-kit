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
End-to-end integration of compare_with_hooks(): spec.tolerance → resolve_tolerance → 入口 → 4-tuple → metrics。
"""

import numpy as np
import pytest

from ttk.core_modules.common.comparison import CompareHooks, compare_with_hooks
from ttk.core_modules.common.comparison.resolve import ResolvedStandard, resolve_tolerance

CLI_TOKENS = ("close", "stat_rel_err", "mixed", "cosine", "binary", "requant", "cross_check")


class _Tc:
    testcase_name = "t"
    output_dist = ()


def _entry(outputs, goldens, dtypes, standards, third=None):
    """公开入口 compare_with_hooks 的无钩子形态（等价于此前的裸 compare 直调）。"""
    return compare_with_hooks(
        _Tc(), outputs, goldens, standards, CompareHooks(output_dtypes=dtypes, third_parties=third)
    )


@pytest.mark.parametrize("method", CLI_TOKENS)
def test_cli_token_end_to_end(method):
    """7 个 --compare CLI 取值全链冒烟：resolve → registry → 入口比对。

    钉两只历史 bug：cosine 死分支 AttributeError（is_torch）、binary 懒加载
    表漏键 ValueError（_TOKEN_MODULES 漏 "binary"）——单元全绿但端到端必炸。"""
    standards = resolve_tolerance(None, None, None, ["float32"], method)
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    third = [np.array([1.0, 2.0], np.float32)] if method == "cross_check" else None
    precision, log, is_pass, metrics = _entry(outputs, goldens, ("float32",), standards, third)
    assert is_pass is True, method
    assert metrics[0]["standard"], method


def test_quant_spec_ptol_flows():
    """Spec 声明 quant + ptol → resolve 透传 → 比对生效（此前被 _resolve_params 静默丢弃）。"""
    tolerance = {"int8": {"standard": "quant", "ptol": 0.5}}
    standards = resolve_tolerance(tolerance, None, None, ["int8"], None, input_dtypes=["float32"])
    assert standards[0].params["ptol"] == 0.5
    outputs = [np.array([0, 0, 0, 0], np.int8)]
    goldens = [np.array([0, 5, 0, 5], np.int8)]  # 50% 超差
    _p, _l, is_pass, metrics = _entry(outputs, goldens, ("int8",), standards)
    assert is_pass is True  # ptol=0.5 容忍 50%
    assert metrics[0]["exceed_count"] == 2


def test_quant_default_ptol_zero_fails():
    """quant 未配 ptol → 默认 0：超差即 FAIL（绝对标准不默认放宽）。"""
    tolerance = {"int8": {"standard": "quant"}}
    standards = resolve_tolerance(tolerance, None, None, ["int8"], None, input_dtypes=["float32"])
    outputs = [np.array([0, 0, 0, 0], np.int8)]
    goldens = [np.array([0, 5, 0, 5], np.int8)]
    _p, _l, is_pass, _m = _entry(outputs, goldens, ("int8",), standards)
    assert is_pass is False


@pytest.mark.parametrize("ph", ["GOLDEN_TIMEOUT", "GOLDEN_FAILURE", "UNSUPPORTED"])
def test_golden_placeholder_fails_at_entry(ph):
    """入口收 golden 占位串（生成失败类）：不抛、precision 透传、FAIL。"""
    standards = resolve_tolerance(None, None, None, ["float32"], None)
    precision, log, is_pass, metrics = _entry([np.ones(2)], [ph], ("float32",), standards)
    assert precision == ph
    assert is_pass is False


@pytest.mark.parametrize("ph", ["SUPPRESSED", "DYN_OFF"])
def test_fake_pass_placeholder_passes_at_entry(ph):
    """豁免/模式开关占位：入口透传 + PASS。"""
    standards = resolve_tolerance(None, None, None, ["float32"], None)
    precision, _log, is_pass, _m = _entry([np.ones(2)], [ph], ("float32",), standards)
    assert precision == ph
    assert is_pass is True


def test_compare_returns_4tuple_with_metrics():
    """入口返回 4-tuple，metrics 含 standard 字段。"""
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    standards = resolve_tolerance(None, None, None, ["float32"], None)  # -> mix_tolerance
    precision, log, is_pass, metrics = _entry(outputs, goldens, ("float32",), standards)
    assert is_pass is True
    assert 0 in metrics
    assert metrics[0]["standard"] == "mix_tolerance"


# —— 端到端:Spec.tolerance → resolve → 入口 → metrics → structure ——


def test_threshold_override_flows_to_metrics():
    """Spec.tolerance threshold override 经 resolve → 入口 → stat_rel_err metrics。"""
    tolerance = {"float32": {"standard": "stat_rel_err", "threshold": 1e-3}}
    standards = resolve_tolerance(tolerance, None, None, ["float32"], None)
    assert standards[0].params["threshold"] == 1e-3

    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    _p, _l, _ip, metrics = _entry(outputs, goldens, ("float32",), standards)
    assert metrics[0]["threshold"] == 1e-3
    assert metrics[0]["standard"] == "stat_rel_err"
    assert "mere" in metrics[0]
    assert "mare" in metrics[0]


def test_metrics_flow_to_comparison_result():
    """入口 4-tuple metrics → ComparisonResult.set(metrics) → .metrics 槽。"""
    from ttk.core_modules.kernel.result import ComparisonResult

    standards = resolve_tolerance(None, None, None, ["float32"], None)
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    _p, _l, _ip, metrics = _entry(outputs, goldens, ("float32",), standards)

    nested = {"dyn": metrics, "cst": metrics, "bin": metrics}
    cr = ComparisonResult(None).set("PASS", "PASS", "PASS", "PASS", nested)
    assert cr.metrics == nested
    assert cr.metrics["dyn"][0]["standard"] == "mix_tolerance"


def test_metrics_flow_to_api_structure():
    """op_api: 入口 metrics → ApiComparisonResult.set(metrics) → AclnnReturnStructure.precision_metrics。"""
    from ttk.core_modules.aclnn.result import AclnnReturnStructure, ApiComparisonResult

    standards = resolve_tolerance(None, None, None, ["float32"], None)
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    _p, _l, _ip, metrics = _entry(outputs, goldens, ("float32",), standards)

    acr = ApiComparisonResult(None).set("PASS", "PASS", metrics)
    assert acr.metrics == metrics

    prs = AclnnReturnStructure()
    prs.construct(None, acr)  # context=None ok for this structure
    assert prs.precision_metrics == metrics
    assert prs.precision_metrics[0]["standard"] == "mix_tolerance"


def test_output_none_fails():
    """output=None + golden 非 None → NO_OUTPUT / FAIL（输出侧缺数据分支）。"""
    precision, _log, is_pass, _m = _entry(
        [None],
        [np.array([1.0])],
        ("float32",),
        [ResolvedStandard("stat_rel_err")],
    )
    assert precision == "NO_OUTPUT"
    assert is_pass is False


# —— SUPPRESSED（golden=None，shape 大小为 0 的输出）位置日志留痕 ——


def test_suppressed_golden_logs_position():
    """golden=None（如 shape 大小为 0 时插件返回 None）→ SUPPRESSED/PASS，且 log 有该位置记录。"""
    outputs = [np.array([1.0]), np.array([], np.float32), np.array([3.0])]
    goldens = [np.array([1.0]), None, np.array([3.0])]
    standards = resolve_tolerance(None, None, None, ["float32"] * 3, None)
    precision, log, is_pass, _m = _entry(outputs, goldens, ("float32",) * 3, standards)
    assert precision.split(",")[1] == "SUPPRESSED"
    assert is_pass is True
    # 原先 SUPPRESSED 位置在比对 log 中完全消失
    assert "Output 1 Precision is SUPPRESSED (golden is None)" in log


class TestSpecialDtypeCompleteness:
    """特殊 dtype × 判据的完备性：fp8 家族经浮点族路由可达 cosine/isclose。"""

    @staticmethod
    def _run(token, output, golden, params):
        from ttk.core_modules.common.comparison.registry import get_comparison

        return get_comparison(token)(output, golden, 0, str(output.dtype), params).compare()

    def test_cosine_fp8_no_overflow(self):
        """cosine + fp8：dot/norm 须提升 fp32 计算——全同向量 precision=1.0（原生计算溢出为 inf）。"""
        from ttk.utilities.dtypes import resolve_numpy_dtype

        f8 = resolve_numpy_dtype("float8_e5m2")
        a = np.full(200, 30.0, dtype=f8)  # 平方和 1.8e5 > e5m2 max 57344
        _p, _l, is_pass, metrics = self._run("cosine", a, a.copy(), {"legacy": {"rtol": None}})
        assert bool(is_pass) is True
        assert metrics["precision"] == pytest.approx(1.0)

    def test_isclose_fp8_dtype_aware_default_rtol(self):
        """isclose + fp8：默认 rtol 分档感知精度（相邻可表示值 ulp≈2^-2~2^-3 须宽限）。"""
        from ttk.utilities.dtypes import resolve_numpy_dtype

        f8 = resolve_numpy_dtype("float8_e5m2")
        b = np.array([1.0], dtype=f8)
        c = np.array([1.25], dtype=f8)  # 1.0 的相邻可表示值
        _p, _l, is_pass, _m = self._run("isclose", c, b, {"legacy": {"rtol": None, "ptol": None, "atol": None}})
        assert bool(is_pass) is True


class TestCompareWithHooksEntry:
    """唯一执行入口 compare_with_hooks 的钩子形态（真实调用，不 mock 内部）。

    通路级测试（test_profiling / test_aclnn_custom_compare / npu/test_comparison）
    在 mock 层验证接线；此处验证入口本身的钩子语义契约。"""

    @staticmethod
    def _hooks(**kw):
        from ttk.core_modules.common.comparison import CompareHooks

        return CompareHooks(output_dtypes=("float32",), **kw)

    @staticmethod
    def _case():
        class _Tc:
            testcase_name = "t"
            output_dist = ()

        return _Tc()

    def test_pre_compare_then_custom_wins(self):
        """pre 变换后 custom 采信（顺序 + 变换到达 + 调用方列表隔离）。"""
        calls = []

        def pre_compare(output, golden):
            calls.append("pre")
            return output * 2, golden * 2  # 返回模式

        def custom_compare(output, golden):
            calls.append("custom")
            np.testing.assert_array_equal(output, [2.0, 4.0])  # 变换后到达
            return {"pass": True, "precision": "HOOKED"}

        out = [np.array([1.0, 2.0])]
        gold = [np.array([1.0, 2.0])]
        p, _l, ok, metrics = compare_with_hooks(
            self._case(),
            out,
            gold,
            resolve_tolerance(None, None, None, ["float32"], None),
            self._hooks(pre_compare=pre_compare, custom_compare=custom_compare),
        )
        assert calls == ["pre", "custom"]
        assert p == "HOOKED"
        assert ok is True
        assert metrics == {}
        np.testing.assert_array_equal(out[0], [1.0, 2.0])  # 隔离

    def test_no_custom_falls_back_to_standard(self):
        """仅声明 pre_compare（custom 未声明）→ 回落标准判据，且 pre 变换对回落比对生效。"""

        def pre_compare(output, golden):
            return output * 2, golden

        out = [np.array([1.0, 2.0])]
        gold = [np.array([2.0, 4.0])]  # 变换后 output == golden → 标准判据须 PASS
        p, _l, ok, metrics = compare_with_hooks(
            self._case(),
            out,
            gold,
            resolve_tolerance(None, None, None, ["float32"], None),
            self._hooks(pre_compare=pre_compare),
        )
        assert ok is True
        assert metrics[0]["standard"] == "mix_tolerance"

    def test_mixed_placeholder_skips_hooks(self):
        """混合占位：钩子不生效，占位透传 + VALID 子集走标准比对，metrics 键为原始输出序号。"""
        called = []

        def custom_compare(output, golden):
            called.append("should-not-run")
            return {"pass": True, "precision": "X"}

        out = [np.array([1.0, 2.0]), np.array([1.0, 2.0])]
        gold = ["GOLDEN_TIMEOUT", np.array([1.0, 2.0])]
        p, _l, ok, metrics = compare_with_hooks(
            self._case(),
            out,
            gold,
            resolve_tolerance(None, None, None, ["float32", "float32"], None),
            self._hooks(custom_compare=custom_compare),
        )
        assert called == []
        assert p == "GOLDEN_TIMEOUT,100.0%"
        assert ok is False
        assert 1 in metrics
        assert 0 not in metrics
