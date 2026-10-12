#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from dataclasses import dataclass, field
from typing import Optional

from ....utilities.dtypes import dtype_to_name, is_float_dtype
from .registry import _TOKEN_MODULES, ComparisonRegister


@dataclass
class ResolvedStandard:
    token: str
    params: dict = field(default_factory=dict)


# 2.1 stat_rel_err 阈值表（resolve_tolerance 是唯一解析点——比对类不查表）
THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
    "float8_e4m3fn": 2**-3,
    "float8_e5m2": 2**-2,
}
DEFAULT_THRESHOLD = 2**-13

# 生态算子开源精度标准（混合容差）阈值表（resolve 唯一解析点）。
# max_abs_error_floor 为绝对误差兜底值（表 2.2 的 "1e-X" 部分）；"or 32*ULP" 部分由
# mix_tolerance 按最大误差点值域动态计算（对齐 cann/opbase mixed_tolerance_standard），不在表内固化。
MIX_TOLERANCE = {
    "float16": {"rtol": 2**-9, "atol": 2**-9, "max_abs_error_floor": 1e-1},
    "bfloat16": {"rtol": 2**-6, "atol": 2**-6, "max_abs_error_floor": 1e-0},
    "float32": {"rtol": 2**-10, "atol": 2**-16, "max_abs_error_floor": 1e-2},
    "hifloat32": {"rtol": 2**-9, "atol": 2**-10, "max_abs_error_floor": 1e-1},
    "float8_e4m3fn": {"rtol": 2**-2, "atol": 2**-4, "max_abs_error_floor": 1e-0},
    "float8_e5m2": {"rtol": 2**-1, "atol": 2**-3, "max_abs_error_floor": 1e-1},
}
MIX_REQUIRED_MATCHED_RATIO = 0.99
DEFAULT_MIX_TOLERANCE = MIX_TOLERANCE["float32"]  # 表外浮点 dtype（float64 等）回落 float32 档

PROMOTE_GOLDEN_TOKENS = ("cross_check", "mixed", "mix_tolerance")

# cross_check level 预设（level 矩阵）
LEVEL_PRESETS = {
    "L0": {"mare_ratio": 10.0, "mere_ratio": 2.0, "rmse_ratio": 2.0},
    "L1": {"mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5},
    "L2": {"mare_ratio": 2.0, "mere_ratio": 1.2, "rmse_ratio": 1.2},
}

# small-value thresholds（key 必须跟 dtype_to_name 输出一致；cross_check 只支持 fp16/bf16/fp32）
SMALL_VALUE = {
    "float16": {"small_value": 2**-11, "small_value_atol": 2**-16},
    "bfloat16": {"small_value": 2**-8, "small_value_atol": 2**-16},
    "float32": {"small_value": 2**-14, "small_value_atol": 2**-30},
}

_REQUANT_DTYPES = {"hifloat8"}
_BIN_DTYPES = {"float4_e2m1", "float4_e1m2", "float8_e8m0"}


def _is_int_or_bool(s: str) -> bool:
    return s == "bool" or s.startswith(("int", "uint"))


def _is_complex(s: str) -> bool:
    return "complex" in s


# 量化输出的目标 dtype：标准限定 int4 / int8。
# 不含 int32/int64——那些通常承载索引、计数、掩码等精确整数语义，差 1 就是错。
_QUANT_OUT_DTYPES = {"int4", "int8"}

# --compare 中对【整型输出】有意义的取值：可覆盖 Spec 的 quant 声明。
# 其余取值均为浮点判据，对量化整型无意义，不参与覆盖。
_INT_APPLICABLE_TOKENS = {"bin", "binary", "binary_equal", "requant", "quant"}


def _spec_standard(tolerance, dtype_str):
    """取 Spec.tolerance 为该 dtype 显式声明的标准名；未声明返回 None。"""
    if not tolerance or dtype_str not in tolerance:
        return None
    std = (tolerance[dtype_str] or {}).get("standard")
    return std.lower() if isinstance(std, str) else None


def _check_quant_applicable(dtype_str, input_dtypes):
    """声明 quant 时校验前提，不匹配直接报错——**不静默纠正**。

    【为何要校验】quant 的判据是绝对误差 <= 1，只对"浮点输入 + int4/int8 输出"的
    量化场景成立。若用在索引/计数类输出上，±1 的容忍会放过真缺陷，而且悄无声息。
    【为何报错而非降级】误用是配置错误，应当让作者当场看见并改正；静默换成别的判据
    会让 Spec 声明与实际判定不一致，是更隐蔽的坑。
    【为何要求至少一个浮点输入】量化必然存在 float→int 的转换，被量化的数据必为
    浮点。但量化算子天然带整型量化参数（如 zero_points / antiquant offset），要求
    "全部输入浮点"会误伤这类合法算子。改为"至少一个浮点输入"——既保留对纯整型算子
    （索引/计数/掩码，无任何浮点输入）的拦截，又不拒绝带整型量化参数的量化算子。
    """
    if dtype_str not in _QUANT_OUT_DTYPES:
        raise ValueError(
            f"Spec.tolerance declares standard='quant' for [{dtype_str}], but quant only "
            f"applies to {sorted(_QUANT_OUT_DTYPES)} outputs (absolute error <= 1 is a "
            f"quantization semantic; index/count outputs are wrong when off by 1)."
        )
    if input_dtypes is not None:
        ins = [dtype_to_name(d) for d in input_dtypes if d is not None]
        if ins and not any(is_float_dtype(i) for i in ins):
            raise ValueError(
                f"Spec.tolerance declares standard='quant' for [{dtype_str}], but input "
                f"dtypes {ins} are all integer. quant requires float inputs with int4/int8 "
                f"outputs; pure integer ops (index/count/mask) should use binary_equal."
            )


def _choice_from_sources(tolerance: Optional[dict], dtype_str: str, compare_method: Optional[str]) -> Optional[str]:
    """来源轴：CLI > Spec.tolerance.standard，均未声明返回 None（不掺默认——默认属 dtype 轴）。"""
    if compare_method:  # a) CLI 显式指定
        return compare_method.lower()
    if tolerance and dtype_str in tolerance:  # b) Spec 配了该 dtype
        std = tolerance[dtype_str].get("standard")
        if std:
            return std.lower()
    return None


def _dtype_policy(dtype_str: str):
    """dtype 轴：类别 → (默认判据, 声明可生效的取值映射)。

    映射键 = CLI/Spec 声明的取值，值 = 实际路由产出的判据名（complex 的
    bin/binary 归一为 binary_equal；hifloat8 原样）。空 dict = 类别强制默认，
    声明不生效（静默忽略——现状语义；"声明冲突应报错"为待决策改进项）。
    None = 浮点族：声明全部原样生效。"""
    if _is_int_or_bool(dtype_str):
        return "binary_equal", {}
    if _is_complex(dtype_str):
        return "isclose", {"bin": "binary_equal", "binary": "binary_equal", "isclose": "isclose"}
    if dtype_str in _REQUANT_DTYPES:
        return "requant", {"bin": "bin", "binary": "binary", "requant": "requant"}
    if dtype_str in _BIN_DTYPES:
        return "bin", {}
    return "mix_tolerance", None  # 浮点族默认：生态算子开源精度标准（混合容差）


def _route(dtype_str, spec_std, tolerance, compare_method, input_dtypes) -> str:
    """判据路由：dtype 轴 × 来源轴，外加一个声明升格例外。

    升格例外：Spec 显式声明 quant——量化语义声明突破整数类别强制（算子作者
    最清楚输出是量化值），经 _check_quant_applicable 校验前提后生效；CLI 仅
    【整数适用】取值（bin/binary/binary_equal/requant/quant）可覆盖，其余浮点
    判据对量化整型无意义、不连带覆盖。"""
    if spec_std == "quant":
        _check_quant_applicable(dtype_str, input_dtypes)
        cli = (compare_method or "").lower()
        return cli if cli in _INT_APPLICABLE_TOKENS else "quant"
    default, allowed = _dtype_policy(dtype_str)
    choice = _choice_from_sources(tolerance, dtype_str, compare_method)
    if allowed is None:
        return choice if choice is not None else default
    return allowed.get(choice, default)


def _stat_rel_err_params(extra: dict, dtype_str: str) -> dict:
    """threshold：Spec 覆盖 > THRESHOLDS 表（表外 dtype 回落 DEFAULT_THRESHOLD）。"""
    th = extra.pop("threshold", None) or THRESHOLDS.get(dtype_str, DEFAULT_THRESHOLD)
    return {"threshold": th}


def _mix_tolerance_params(extra: dict, dtype_str: str) -> dict:
    """rtol/atol/required_matched_ratio/max_abs_error_limit：Spec 覆盖 > MIX_TOLERANCE 表。"""
    # threshold 是 stat_rel_err 专属参数，对 mix_tolerance 无效——不静默丢弃。
    # 命中典型场景：Spec 只配 {"threshold": x} 未配 standard，默认路由从 stat_rel_err
    # 切到 mix_tolerance 后阈值失效（宽松阈值误报 / 严格阈值漏报），必须当场报错。
    if extra.get("threshold") is not None:
        raise ValueError(
            f"Spec.tolerance[{dtype_str!r}] sets the stat_rel_err-only parameter 'threshold', "
            f"but the standard resolved for this output is mix_tolerance, where threshold "
            f"has no effect. For statistical relative error, declare standard='stat_rel_err' "
            f"explicitly (or CLI --compare stat_rel_err); for mixed tolerance remove "
            f"threshold and use rtol/atol/required_matched_ratio/max_abs_error_limit."
        )
    row = MIX_TOLERANCE.get(dtype_str, DEFAULT_MIX_TOLERANCE)
    # !292 max_abs_error_limit：显式配置 → 原值（mix 内完全替代动态式）；
    # 未配置 → None（动态式：max(floor, 32*ULP(g_low))，见 mix_tolerance 判据）
    return {
        "rtol": extra.get("rtol", row["rtol"]),
        "atol": extra.get("atol", row["atol"]),
        "required_matched_ratio": extra.get("required_matched_ratio", MIX_REQUIRED_MATCHED_RATIO),
        "max_abs_error_limit": extra.get("max_abs_error_limit"),
        "max_abs_error_floor": row["max_abs_error_floor"],
    }


def _ptol_params(extra: dict, dtype_str: str) -> dict:
    """quant / requant 共用：Spec 可配 ptol（允许超差元素占比）。

    quant 默认 0（绝对误差<=1 是绝对标准，超出即缺陷）；requant 默认按 dtype
    分档（fp8 家族 0.001，其余 0）——默认值由判据侧 ComparisonBase 的
    _default_ptol 钩子兜底（tolerance 终值化见 registry._finalize_tolerances）。"""
    return {k: v for k, v in extra.items() if k == "ptol"}


def _cross_check_params(extra: dict, dtype_str: str) -> dict:
    """level（三比值预设，Spec 可逐项覆盖）+ small_value 两件：Spec 覆盖 > 表。"""
    if dtype_str not in SMALL_VALUE:
        raise ValueError(f"[{dtype_str}] cross_check unsupported dtype; supported: {sorted(SMALL_VALUE)}")
    level = extra.get("level")
    if level is not None:
        if level not in LEVEL_PRESETS:
            raise ValueError(f"[{dtype_str}] cross_check unknown level: {level!r}; expected {sorted(LEVEL_PRESETS)}")
        limits = dict(LEVEL_PRESETS[level])
    else:
        level = "L1"
        limits = dict(LEVEL_PRESETS[level])
    for k in ("mare_ratio", "mere_ratio", "rmse_ratio"):
        if k in extra:
            limits[k] = extra[k]
    sv_default = SMALL_VALUE[dtype_str]
    return {
        "level": level,
        **limits,
        "small_value": extra.get("small_value", sv_default["small_value"]),
        "small_value_atol": extra.get("small_value_atol", sv_default["small_value_atol"]),
    }


# 判据 → 参数解析器。未登记的判据（bin/isclose/cosine）无表驱动参数——
# 它们读 legacy CSV 容差，由 resolve_tolerance 调用方注入。
_PARAM_RESOLVERS = {
    "stat_rel_err": _stat_rel_err_params,
    "mixed": _mix_tolerance_params,  # mixed 是 --compare 的 CLI 简写
    "mix_tolerance": _mix_tolerance_params,
    "quant": _ptol_params,
    "requant": _ptol_params,
    "cross_check": _cross_check_params,
}


def _resolve_params(standard, tolerance, dtype_str) -> dict:
    cfg = (tolerance or {}).get(dtype_str, {})
    extra = {k: v for k, v in cfg.items() if k != "standard"}
    resolver = _PARAM_RESOLVERS.get(standard)
    return resolver(extra, dtype_str) if resolver is not None else {}


def _check_token_exists(token: str):
    """判据名存在性早校验：Spec/CLI 拼写错误在 resolve 期报错（执行 NPU/生成
    golden 之前），而非比对期才爆。_TOKEN_MODULES 是静态表不触发判据模块加载；
    插件经 register_comparison 注册的 token 在 SpecResolver 加载后已入 registry。"""
    if token not in _TOKEN_MODULES and token not in ComparisonRegister.registry:
        raise ValueError(
            f"Comparison standard [{token}] is not recognized (Spec/CLI 拼写错误？). "
            f"Registered: {sorted(set(_TOKEN_MODULES) | set(ComparisonRegister.registry))}"
        )


def resolve_tolerance(
    tolerance, precision_tolerances, absolute_precision, output_dtypes, compare_method, input_dtypes=None
):
    """input_dtypes 可选：仅供 quant 的护栏校验"输入是否至少含一个浮点"，不参与判据选择。
    不传时跳过该校验，Spec 声明的 quant 依然生效——判据由算子作者的声明决定，
    拿不到输入 dtype 只是少了一道校验，不应反过来否定声明。
    未声明 quant 的存量算子不受影响（整数仍走 binary_equal）。"""
    standards = []
    for idx, dtype in enumerate(output_dtypes):
        # dtype None（输出 dtype 未知）不取名、不查 Spec——直接按位比对
        # （dtype_to_name(None) 会回落 'float64'，查 Spec 会意外命中其 float64 条目）。
        if dtype is None:
            token = "binary_equal"  # noqa: S105
            params = {}
        else:
            s = dtype_to_name(dtype)
            token = _route(s, _spec_standard(tolerance, s), tolerance, compare_method, input_dtypes)
            _check_token_exists(token)
            params = _resolve_params(token, tolerance, s)
        # legacy 注入（per-output 标量，命名空间隔离）
        params["legacy"] = {
            "rtol": precision_tolerances[idx][0] if precision_tolerances and idx < len(precision_tolerances) else None,
            "ptol": precision_tolerances[idx][1] if precision_tolerances and idx < len(precision_tolerances) else None,
            "atol": absolute_precision[idx]
            if isinstance(absolute_precision, (tuple, list)) and idx < len(absolute_precision)
            else (absolute_precision if not isinstance(absolute_precision, (tuple, list)) else None),
        }
        standards.append(ResolvedStandard(token, params))
    return standards


def needs_promote_golden(standards) -> bool:
    """任一输出判据属于 PROMOTE_GOLDEN_TOKENS 时返回 True（golden 须升精度生成）。"""
    return any(s.token in PROMOTE_GOLDEN_TOKENS for s in standards)
