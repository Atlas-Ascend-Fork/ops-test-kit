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
Unit tests for the E2E golden Promote path.

Covers the three functions of the E2E golden Promote mechanism:

  * `golden_generation._promote_raw_inputs` - dtype lift itself
  * `worker._needs_golden_promote`           - "does the standard need Promote?"
  * `worker._generate_golden`                - set / restore the override
"""

from unittest.mock import patch

import numpy as np

from ttk.core_modules.framework_api import worker as _worker
from ttk.core_modules.framework_api.golden import (
    _promote_raw_inputs as promote_raw_inputs,
)
from ttk.core_modules.framework_api.worker import (
    _generate_golden as generate_golden,
)
from ttk.core_modules.framework_api.worker import (
    _needs_golden_promote as needs_golden_promote,
)

# 被测函数以别名导入而非 `模块._私有` 属性访问:后者会被判受保护成员访问(G.CLS.11)。
# 仍保留 _worker 模块引用,供 patch.object 按名字打桩使用(传的是字符串,不构成属性访问)。


class _Testcase:
    """Minimal stand-in for TestcaseE2e (plain attributes, no __slots__)."""

    def __init__(self, flat_tensor_dtypes=None, golden_mode_override=None, flat_output_dtypes=("float32",)):
        self.testcase_name = "ut_case"
        self.api_name = "torch.add"
        self.flat_tensor_dtypes = flat_tensor_dtypes
        self.golden_mode_override = golden_mode_override
        self.flat_output_dtypes = flat_output_dtypes
        self.flat_precision_tolerances = None
        self.flat_absolute_precision = None


class _Switches:
    """Minimal stand-in for the SWITCHES object."""

    def __init__(self, golden_mode=None):
        self.golden_mode = golden_mode
        self.plugin_path = None
        self.compare_method = None


def _standard(token):
    """构造一个只带 token 字段的 tolerance standard 替身。"""
    return type("_Std", (), {"token": token})()


# ----------------------------------------------------------------------------
# _promote_raw_inputs
# ----------------------------------------------------------------------------
def test_promote_lifts_float_dtypes():
    """fp16/bf16 -> fp32, fp32 -> fp64, per DTYPE_PROMOTE_MAP."""
    testcase = _Testcase(flat_tensor_dtypes=["float16", "float32"])
    raw = [np.ones(4, dtype=np.float16), np.ones(4, dtype=np.float32)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32
    assert out[1].dtype == np.float64
    # 原数组不被就地改写
    assert raw[0].dtype == np.float16
    assert raw[1].dtype == np.float32


def test_promote_leaves_integer_dtypes_untouched():
    """整型不在 DTYPE_PROMOTE_MAP 中,必须原样保留(含对象本身)。"""
    testcase = _Testcase(flat_tensor_dtypes=["int32", "int64", "bool"])
    raw = [
        np.ones(2, dtype=np.int32),
        np.ones(2, dtype=np.int64),
        np.ones(2, dtype=bool),
    ]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert [a.dtype for a in out] == [np.dtype(np.int32), np.dtype(np.int64), np.dtype(bool)]
    for got, want in zip(out, raw):
        assert got is want


def test_promote_leaves_extra_inputs_alone():
    """raw_inputs 比 dtype 列表长时,多出的部分保持原样而非报错。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"])
    raw = [np.ones(2, dtype=np.float16), np.ones(2, dtype=np.float16)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32
    assert out[1].dtype == np.float16


def test_promote_applies_to_mutable_apis():
    """升精度只由 golden_mode 与 dtype 决定，对 mutable/有状态 API 一致适用。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"])
    testcase.api_name = "tf.raw_ops.ResourceScatterAdd"
    raw = [np.ones(4, dtype=np.float16)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32


# ----------------------------------------------------------------------------
# _needs_golden_promote
# ----------------------------------------------------------------------------
def test_needs_promote_true_for_cross_check():
    with patch.object(_worker, "SpecResolver") as _sr, patch.object(
        _worker, "resolve_tolerance", return_value=[_standard("cross_check")]
    ):
        _sr.return_value.resolve_tolerance.return_value = None
        assert needs_golden_promote(_Testcase(), _Switches()) is True


def test_needs_promote_true_for_mix_tolerance():
    """mix_tolerance（默认单标杆判据）与 CLI 简写 mixed 同样要求 golden 升精度。"""
    for token in ("mix_tolerance", "mixed"):
        with patch.object(_worker, "SpecResolver") as _sr, patch.object(
            _worker, "resolve_tolerance", return_value=[_standard(token)]
        ):
            _sr.return_value.resolve_tolerance.return_value = None
            assert needs_golden_promote(_Testcase(), _Switches()) is True


def test_needs_promote_false_for_non_promote_standards():
    """stat_rel_err/binary_equal/isclose 判据不强制 golden 升精度。"""
    for token in ("stat_rel_err", "binary_equal", "isclose"):
        with patch.object(_worker, "SpecResolver") as _sr, patch.object(
            _worker, "resolve_tolerance", return_value=[_standard(token)]
        ):
            _sr.return_value.resolve_tolerance.return_value = None
            assert needs_golden_promote(_Testcase(), _Switches()) is False


def test_needs_promote_false_without_declared_output_dtypes():
    """CSV 未声明输出 dtype 时无法解析判据，不升精度（锚点=声明 dtype，不依赖执行结果）。"""
    with patch.object(_worker, "SpecResolver") as _sr, patch.object(
        _worker, "resolve_tolerance", return_value=[_standard("mix_tolerance")]
    ):
        _sr.return_value.resolve_tolerance.return_value = None
        assert needs_golden_promote(_Testcase(flat_output_dtypes=None), _Switches()) is False


def test_needs_promote_anchor_is_csv_declared_dtype():
    """判定锚点 = CSV 声明输出 dtype：resolve_numpy_dtypes(flat_output_dtypes) 供参。"""
    seen = {}

    def _capture(tolerance, precisions, absolutes, dtypes, method):
        seen["dtypes"] = dtypes
        return [_standard("cross_check")]

    with patch.object(_worker, "SpecResolver") as _sr, patch.object(_worker, "resolve_tolerance", side_effect=_capture):
        _sr.return_value.resolve_tolerance.return_value = None
        assert needs_golden_promote(_Testcase(flat_output_dtypes=("float32", "float16")), _Switches()) is True
    assert tuple(str(d) for d in seen["dtypes"]) == ("float32", "float16")


def test_needs_promote_false_under_xpu():
    """golden_mode=XPU 与 Promote 互斥：golden 由三方产出，不做 CPU 抬精度。"""
    assert needs_golden_promote(object(), _Switches(golden_mode="XPU")) is False


def test_generate_golden_data_xpu_placeholder(monkeypatch):
    """XPU 模式本地不生成 golden：统一返回 GOLDEN_FAILURE 占位（真值由远端三方回路给出）。"""
    from types import SimpleNamespace

    import ttk.core_modules.framework_api.compare as prof

    monkeypatch.setattr(
        prof, "process_context", lambda: SimpleNamespace(notify_status=lambda *a, **k: None, storage={})
    )
    testcase = SimpleNamespace(golden_api="", golden_mode_override=None)
    golden = prof.generate_golden_data(testcase, [], _Switches(golden_mode="XPU"), None, dump=False)
    assert golden == ["GOLDEN_FAILURE"]


# ----------------------------------------------------------------------------
# _generate_golden
# ----------------------------------------------------------------------------
def _patch_plain_generator(seen):
    def _fake_generate(case, raw_inputs, switches, backend, dump=True):
        seen["mode"] = case.golden_mode_override
        seen["dump"] = dump
        return ["golden"]

    return patch.object(_worker, "generate_golden_data", side_effect=_fake_generate)


def test_generate_golden_promotes_then_restores():
    testcase = _Testcase(golden_mode_override="Enable")
    seen = {}

    with _patch_plain_generator(seen):
        out = generate_golden(testcase, [], _Switches(), "cpu", promote=True)

    assert out == ["golden"]
    assert seen["mode"] == "Promote"  # 生成期间已抬升
    assert testcase.golden_mode_override == "Enable"  # 结束后恢复原值


def test_generate_golden_without_promote_keeps_mode():
    testcase = _Testcase(golden_mode_override="Enable")
    seen = {}

    with _patch_plain_generator(seen):
        out = generate_golden(testcase, [], _Switches(), "cpu", promote=False)

    assert out == ["golden"]
    assert seen["mode"] == "Enable"  # 未抬升
