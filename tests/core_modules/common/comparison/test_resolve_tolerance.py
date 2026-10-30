# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""resolve_tolerance 单元测试：dtype→standard 路由 + threshold 解析（表默认/spec override/override 优先级）。"""

from ttk.core_modules.common.comparison.resolve import resolve_tolerance


def _tokens(rs):
    """提取 ResolvedStandard 列表的 token 字符串。"""
    return [r.token for r in rs]


def test_int_bool_always_binary_equal():
    """整型/布尔 dtype 始终路由到 binary_equal。"""
    for dtype in ["int32", "bool"]:
        assert _tokens(resolve_tolerance(None, None, None, [dtype], "close")) == ["binary_equal"]


def test_complex_defaults_isclose():
    """complex64 默认 isclose。"""
    assert _tokens(resolve_tolerance(None, None, None, ["complex64"], None)) == ["isclose"]


def test_fp8_defaults_mix_tolerance():
    """float8_e5m2 默认 mix_tolerance（生态算子开源精度标准）。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float8_e5m2"], None)) == ["mix_tolerance"]


def test_normal_float_defaults_mix_tolerance():
    """float32 默认 mix_tolerance（生态算子开源精度标准）。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float32"], None)) == ["mix_tolerance"]


def test_normal_float_cli_wins():
    """CLI 指定优先于 Spec.tolerance。"""
    assert _tokens(resolve_tolerance({"float32": {"standard": "binary_equal"}}, None, None, ["float32"], "close")) == [
        "close"
    ]


def test_cli_mixed_alias_matches_mix_tolerance():
    """CLI 简写 --compare mixed 与 mix_tolerance 同判据同阈值表。"""
    rs = resolve_tolerance(None, None, None, ["float32"], "mixed")
    assert _tokens(rs) == ["mixed"]
    p = _params(rs)
    assert p["rtol"] == 2**-10  # mix_tolerance float32 表值
    assert p["required_matched_ratio"] == 0.99


def test_normal_float_spec_binary_equal():
    """Spec.tolerance 可将 float16 改为 binary_equal。"""
    assert _tokens(resolve_tolerance({"float16": {"standard": "binary_equal"}}, None, None, ["float16"], None)) == [
        "binary_equal"
    ]


def test_multi_output_mixed():
    """多输出混合 dtype → 各自路由到对应 standard。"""
    out = _tokens(resolve_tolerance(None, None, None, ["int32", "float32", "complex64", "float8_e5m2"], None))
    assert out == ["binary_equal", "mix_tolerance", "isclose", "mix_tolerance"]


# —— threshold 解析（resolve_tolerance 唯一解析点；比对类不查表）——
def _params(rs, idx=0):
    """取第 idx 个 ResolvedStandard 的 params dict。"""
    return rs[idx].params


def test_threshold_resolution():
    """Spec.tolerance threshold override 优先于表默认。"""
    spec = {"float32": {"standard": "stat_rel_err", "threshold": 1e-3}}
    rs = resolve_tolerance(spec, None, None, ["float32"], None)
    assert _params(rs)["threshold"] == 1e-3


# —— mix_tolerance 阈值表解析 ——
def test_mix_tolerance_table_defaults():
    """dtype 命中混合容差表：rtol/atol/required_matched_ratio/max_abs_error_limit 表值。"""
    rs = resolve_tolerance(None, None, None, ["float16"], None)
    p = _params(rs)
    assert p["rtol"] == 2**-9
    assert p["atol"] == 2**-9
    assert p["required_matched_ratio"] == 0.99
    assert p["max_abs_error_limit"] is None  # 未配置 → None 哨兵（mix 内走动态 ULP）
    assert p["max_abs_error_floor"] == 1e-1


def test_mix_tolerance_unknown_dtype_falls_back_float32():
    """表外浮点 dtype（float64）回落 float32 档。"""
    rs = resolve_tolerance(None, None, None, ["float64"], None)
    p = _params(rs)
    assert p["rtol"] == 2**-10
    assert p["atol"] == 2**-16


def test_mix_tolerance_spec_override():
    """Spec.tolerance 可覆盖 rtol/atol/required_matched_ratio/max_abs_error_limit。"""
    spec = {"float32": {"standard": "mix_tolerance", "rtol": 0.5, "required_matched_ratio": 0.9}}
    p = _params(resolve_tolerance(spec, None, None, ["float32"], None))
    assert p["rtol"] == 0.5
    assert p["required_matched_ratio"] == 0.9
    assert p["atol"] == 2**-16  # 未覆盖的仍取表值


# —— threshold 与 mix_tolerance 的冲突护栏（不静默丢弃）——
def test_threshold_only_spec_without_standard_raises():
    """只配 threshold 未配 standard：默认路由 mix_tolerance 会丢弃阈值，必须报错。"""
    import pytest

    with pytest.raises(ValueError, match="threshold"):
        resolve_tolerance({"float16": {"threshold": 0.01}}, None, None, ["float16"], None)


def test_mix_tolerance_with_threshold_raises():
    """显式 standard='mix_tolerance' 却带 threshold：配置错误，报错。"""
    import pytest

    spec = {"float32": {"standard": "mix_tolerance", "threshold": 0.01}}
    with pytest.raises(ValueError, match="threshold"):
        resolve_tolerance(spec, None, None, ["float32"], None)


def test_threshold_only_spec_cli_stat_rel_err_still_works():
    """threshold-only Spec + CLI --compare stat_rel_err：判据命中，threshold 正常生效。"""
    rs = resolve_tolerance({"float16": {"threshold": 0.01}}, None, None, ["float16"], "stat_rel_err")
    assert _params(rs)["threshold"] == 0.01


def test_cli_mix_tolerance_overrides_stat_rel_err_threshold_raises():
    """Spec 声明 stat_rel_err+threshold 但 CLI 强制 mix_tolerance：阈值会失效，报错。"""
    import pytest

    spec = {"float32": {"standard": "stat_rel_err", "threshold": 0.002}}
    with pytest.raises(ValueError, match="threshold"):
        resolve_tolerance(spec, None, None, ["float32"], "mix_tolerance")


def test_mix_tolerance_fp8_table_defaults():
    """float8 命中混合容差阈值表（float8_e4m3fn / float8_e5m2 均在表内）。"""
    rs = resolve_tolerance(None, None, None, ["float8_e4m3fn"], None)
    p = _params(rs)
    assert p["rtol"] == 2**-2
    assert p["atol"] == 2**-4
    assert p["max_abs_error_limit"] is None
    assert p["max_abs_error_floor"] == 1e-0


def test_mix_tolerance_hifloat32_floor():
    """hifloat32 兜底值 1e-1；ULP 部分由 mix 动态计算。"""
    rs = resolve_tolerance(None, None, None, ["hifloat32"], None)
    p = _params(rs)
    assert p["max_abs_error_limit"] is None
    assert p["max_abs_error_floor"] == 1e-1


def test_hifloat8_defaults_requant():
    """hifloat8 不在混合容差阈值表内（表外量化 dtype），保持 requant 默认。"""
    assert _tokens(resolve_tolerance(None, None, None, ["hifloat8"], None)) == ["requant"]


# —— needs_promote_golden（golden 强制升精度的判据路由，各 profiling 流程共用）——
def test_needs_promote_golden_routes():
    """cross_check/mixed/mix_tolerance 升精度；stat_rel_err/binary_equal/isclose 不升。"""
    from ttk.core_modules.common.comparison.resolve import needs_promote_golden

    assert needs_promote_golden(resolve_tolerance(None, None, None, ["float32"], None)) is True  # 默认 mix_tolerance
    assert needs_promote_golden(resolve_tolerance(None, None, None, ["float32"], "mixed")) is True
    assert (
        needs_promote_golden(
            resolve_tolerance({"float32": {"standard": "cross_check", "level": "L1"}}, None, None, ["float32"], None)
        )
        is True
    )
    assert (
        needs_promote_golden(
            resolve_tolerance({"float32": {"standard": "stat_rel_err"}}, None, None, ["float32"], None)
        )
        is False
    )
    assert needs_promote_golden(resolve_tolerance(None, None, None, ["int32"], None)) is False  # binary_equal
    assert needs_promote_golden(resolve_tolerance(None, None, None, ["float32"], "close")) is False  # isclose
    # 多输出任一命中即升精度
    assert needs_promote_golden(resolve_tolerance(None, None, None, ["int32", "float32"], None)) is True


def test_unknown_token_fails_at_resolve():
    """Spec standard 拼写错误在 resolve 期报错（早于 NPU 执行/golden 生成），非比对期。"""
    import pytest

    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    with pytest.raises(ValueError, match="not recognized"):
        resolve_tolerance({"float32": {"standard": "stst_rel_err"}}, None, None, ["float32"], None)


def test_none_dtype_ignores_spec_entry():
    """dtype None 不再取名（此前 dtype_to_name(None) 回落 'float64'，会命中 Spec float64 条目）。"""
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    stds = resolve_tolerance({"float64": {"standard": "stat_rel_err", "threshold": 1.0}}, None, None, [None], None)
    assert stds[0].token == "binary_equal"  # noqa: S105 token 为比对标准名，非口令
    assert "threshold" not in stds[0].params


class TestRoutingMatrix:
    """两轴路由矩阵（dtype 轴 × 来源轴 + quant 升格例外）——可执行的语义文档。"""

    def _tokens(self, dtype, tolerance=None, cli=None):
        from ttk.core_modules.common.comparison.resolve import resolve_tolerance

        return [s.token for s in resolve_tolerance(tolerance, None, None, [dtype], cli)]

    def test_dtype_axis_defaults(self):
        """dtype 轴：无任何声明时各类别走类别默认。"""
        assert self._tokens("int32") == ["binary_equal"]
        assert self._tokens("bool") == ["binary_equal"]
        assert self._tokens("complex64") == ["isclose"]
        assert self._tokens("hifloat8") == ["requant"]
        assert self._tokens("float8_e8m0") == ["bin"]
        assert self._tokens("float4_e2m1") == ["bin"]
        assert self._tokens("float32") == ["mix_tolerance"]

    def test_source_axis_priority(self):
        """来源轴：CLI > Spec.tolerance > 类别默认（浮点族声明全生效）。"""
        assert self._tokens("float32", cli="cosine") == ["cosine"]
        assert self._tokens("float32", {"float32": {"standard": "stat_rel_err"}}) == ["stat_rel_err"]
        assert self._tokens("float32", {"float32": {"standard": "stat_rel_err"}}, cli="cosine") == ["cosine"]
        assert self._tokens("float8_e5m2", cli="requant") == ["requant"]  # 浮点族不限取值

    def test_dtype_axis_filters_source(self):
        """受限类别：声明仅在该类别的可生效集内原样生效（别名归一），其余回落类别默认。"""
        # complex：bin/binary 归一 binary_equal，isclose 生效，其余回落 isclose
        assert self._tokens("complex64", cli="bin") == ["binary_equal"]
        assert self._tokens("complex64", cli="binary") == ["binary_equal"]
        assert self._tokens("complex64", cli="isclose") == ["isclose"]
        assert self._tokens("complex64", cli="mixed") == ["isclose"]
        # hifloat8：bin/binary/requant 生效，其余回落 requant
        assert self._tokens("hifloat8", cli="binary") == ["binary"]
        assert self._tokens("hifloat8", cli="requant") == ["requant"]
        assert self._tokens("hifloat8", cli="close") == ["requant"]
        # 整数 / 打包位：类别强制，声明不生效（现状语义，改进项另议）
        assert self._tokens("int32", {"int32": {"standard": "close"}}) == ["binary_equal"]
        assert self._tokens("float8_e8m0", {"float8_e8m0": {"standard": "stat_rel_err"}}) == ["bin"]

    def test_quant_promotion_breaks_int_forcing(self):
        """升格例外：Spec 声明 quant 突破整数强制；CLI 仅整数适用取值可覆盖。"""
        tol = {"int8": {"standard": "quant"}}
        assert self._tokens("int8", tol) == ["quant"]
        assert self._tokens("int8", tol, cli="bin") == ["bin"]
        assert self._tokens("int8", tol, cli="close") == ["quant"]  # 浮点取值不覆盖量化声明
        # uint8 与 int8 同为量化目标（非对称量化 zero_point != 0，值域 [0, 255]），
        # 升格 / CLI 覆盖语义与 int8 完全一致
        tol_u8 = {"uint8": {"standard": "quant"}}
        assert self._tokens("uint8", tol_u8) == ["quant"]
        assert self._tokens("uint8", tol_u8, cli="bin") == ["bin"]
        assert self._tokens("uint8", tol_u8, cli="close") == ["quant"]
        # 未声明时维持整数类别强制——存量 uint8 算子行为不变
        assert self._tokens("uint8") == ["binary_equal"]


def test_quant_uint8_guardrails_unchanged():
    """uint8 入白名单后护栏不变：仍要求至少一个浮点输入，其余宽整型继续拦截。"""
    import pytest

    # 合法：uint8 输出 + 浮点输入（可附带整型量化参数 zero_point / offset）
    stds = resolve_tolerance({"uint8": {"standard": "quant"}}, None, None, ["uint8"], None, ["float16", "int32"])
    assert _tokens(stds) == ["quant"]

    # 纯整型输入（索引/计数/掩码语义）仍被拦截
    with pytest.raises(ValueError, match="all integer"):
        resolve_tolerance({"uint8": {"standard": "quant"}}, None, None, ["uint8"], None, ["int32"])

    # 白名单外整型（int16/uint16/int32/...）声明 quant 继续报错——±1 容忍会放过真缺陷
    for dt in ("int16", "uint16", "int32", "uint32", "int64", "uint64"):
        with pytest.raises(ValueError, match="quant only applies to"):
            resolve_tolerance({dt: {"standard": "quant"}}, None, None, [dt], None, ["float16"])


def test_requant_spec_ptol_passthrough():
    """requant 的 Spec ptol 透传（与 quant 对称；此前无分支被静默丢弃）。"""
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    stds = resolve_tolerance({"hifloat8": {"standard": "requant", "ptol": 0.05}}, None, None, ["hifloat8"], None)
    assert stds[0].token == "requant"  # noqa: S105 token 为比对标准名，非口令
    assert stds[0].params["ptol"] == 0.05

    # 不配置时无 ptol 键 → 判据侧按 dtype 分档默认（fp8 家族 0.001）
    stds = resolve_tolerance({"hifloat8": {"standard": "requant"}}, None, None, ["hifloat8"], None)
    assert "ptol" not in stds[0].params
