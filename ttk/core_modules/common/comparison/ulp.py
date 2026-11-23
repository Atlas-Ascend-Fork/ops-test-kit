#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""cross_check 三方 ULP 误差明细：按输出 dtype 网格计算两两整数 ULP 距离。

口径：以【声明的输出 dtype】为刻度（ruler），三方各自 RNE 折叠到该档
（Promote 抬档的 golden / 异 dtype 的 XPU 同折；NPU 输出恒等），位模式经
单调整调键求整数距离 |Δkey|——即两值间隔的可表示值个数（经典 ULP 距离，
fold_to_output_dtype 同款折叠先例）。ULP 是观测指标，不参与 pass/fail 判定。

消费面（两个，均纯 Python 字面量进出，CSV eval 往返约束）：
- worker 侧：cross_check.compare_impl 调 scan_ulp_triplet，产物挂
  metrics["ulp"] 过 pipe 回传（与 mare/mere/rmse 同通道）；
- 主进程侧：write_row 钩子经 pop_ulp_payloads 摘出（result.csv 的
  precision_metrics 单元格不残留 ulp），iter_detail_rows 渲染写
  compare_detail.csv——宽表一行 = 一个 (case, mode, output)：三对
  max_ulp 各作一列，主表 precision_metrics 的 cross_check 参数并入同行。

载荷契约（检视意见 P3：worker 只计算/传回宽表实际消费的字段）：
  {"dtype", "golden_is_third_party", "pairs": {对名: {"max_ulp": int|None}}}
逐点差异诊断（首差位/位模式）不在交付面，需要时再按需启用并补输出通道。

分块协议：全部中间数组 O(CHUNK_SIZE)，max 为块间 running max。
特殊值语义：finite 有限对才有 ULP；NaN/±Inf 一律不参与 max，
无 finite 对时 max_ulp=None（渲染 N/A）。±0.0 与 -0.0 相距 1 ULP（位级语义）。
"""

from typing import NamedTuple

import numpy as np

from ....utilities.dtypes import DTYPE_MAP, dtype_to_name, resolve_numpy_dtypes
from .registry import CHUNK_SIZE

__all__ = [
    "COMPARE_DETAIL_HEADER",
    "iter_detail_rows",
    "pop_ulp_payloads",
    "scan_ulp_triplet",
    "ulp_monotone_key",
]


# ruler dtype → (uint 视图 dtype, 符号位掩码, 全 1 掩码, 减法承载 dtype)。
# 减法升位承载：uint 视图两端键差可达全幅（如 0xFFFF-0x0），原生 uint
# 减法回绕成小值（re_quantize 的同款教训），int32/int64 无此风险。
_ULP_RULERS = {
    "float16": (np.uint16, np.uint16(0x8000), np.uint16(0xFFFF), np.int32),
    "bfloat16": (np.uint16, np.uint16(0x8000), np.uint16(0xFFFF), np.int32),
    "float32": (np.uint32, np.uint32(0x80000000), np.uint32(0xFFFFFFFF), np.int64),
}

# compare_detail.csv 表头（唯一权威定义，instance 落盘侧直接引用）。
# 宽表：一行 = 一个 (case, mode, output)——三对 max_ulp 各作一列；主表
# precision_metrics 的 cross_check 参数（判据标识/配置/比值结果）并入
# 同行，比对明细与判定结论一表可读。golden_is_third_party 标注 golden
# 实际来源（--golden-mode XPU 时 golden≡third_party，检视意见 P2：
# 该场景无 CPU 参与，npu_cpu/xpu_cpu 两列渲染 N/A 而非误导性数值）。
COMPARE_DETAIL_HEADER = (
    "testcase_name",
    "run_mode",
    "output_idx",
    "output_dtype",
    "golden_is_third_party",
    "npu_cpu_max_ulp",
    "npu_xpu_max_ulp",
    "xpu_cpu_max_ulp",
    "standard",
    "level",
    "pass",
    "mare",
    "mere",
    "rmse",
    "small_err_cnt_target",
    "small_err_cnt_third",
    "small_err_ratio",
    "mare_limit",
    "mere_limit",
    "rmse_limit",
    "small_value",
    "small_value_atol",
    "reason",
)

# 行序固定（透视友好）；左/右角色由对名决定：npu_cpu→(NPU, CPU golden)、
# npu_xpu→(NPU, XPU third_party)、xpu_cpu→(XPU, CPU golden)。
_PAIR_ORDER = ("npu_cpu", "npu_xpu", "xpu_cpu")


class _Ruler(NamedTuple):
    """ruler 派生量打包：scan 解包一次、全链单对象传递（兼满足复杂度基线）。"""

    np_dtype: object  # numpy 承载 dtype（折叠目标；bf16 依赖 ml_dtypes）
    view_dtype: object  # 位视图 dtype（uint16/uint32）
    sub_dtype: object  # 距离减法承载 dtype（int32/int64，防 uint 回绕）


def ulp_monotone_key(bits: np.ndarray, sign_mask, ones_mask) -> np.ndarray:
    """IEEE754 位模式 → 全序单调键：负数区翻转（值随键增），正数区抬高符号位。

    fp16 例：-inf(0xFC00)→0x03FF … -0.0(0x8000)→0x7FFF < +0.0(0x0000)→0x8000
    … subnormal … +inf(0x7C00)→0xFC00。|Δkey| 即两值间隔的可表示值个数。
    where 的两分支对全量元素都求值：被丢弃分支的 uint 回绕无副作用。
    """
    neg = bits >= sign_mask  # 符号位置 1 = 负数区
    return np.where(neg, ones_mask - bits, bits + sign_mask)


def _canonical_dtype(dtype) -> str:
    """dtype 声明（str/别名/dtype 对象）→ 规范名（'fp16'→'float16'）。"""
    name = dtype_to_name(dtype)
    return DTYPE_MAP.get(name, name)


def _ruler_np_dtype(dtype_name: str):
    """ruler 的 numpy 承载 dtype（bf16 依赖 ml_dtypes）；无承载返回 None。"""
    try:
        return np.dtype(resolve_numpy_dtypes(dtype_name))
    except (TypeError, RuntimeError, ValueError):
        return None


class UlpPairScan:
    """单对（如 npu_cpu）的分块 ULP 统计：块间 running max。

    仅统计宽表实际消费的 max_ulp（检视意见 P3：首差位逐点明细不在交付面，
    不计算不传回；需要逐点诊断时再按需恢复采集与输出通道）。
    """

    __slots__ = ("max_ulp",)

    def __init__(self):
        self.max_ulp = -1  # 尚无 finite 对；收敛时 None 表示无 finite 对

    def update(self, left_key: np.ndarray, right_key: np.ndarray, finite: np.ndarray, sub_dtype):
        """单块更新：left/right 为本块单调键，finite 为双方皆有限的掩码。"""
        pos = np.flatnonzero(finite)
        if pos.size == 0:
            return
        dist = np.abs(left_key[pos].astype(sub_dtype) - right_key[pos].astype(sub_dtype))
        self.max_ulp = max(self.max_ulp, int(dist.max()))


def scan_ulp_triplet(t, g, b, output_dtype, golden_is_third_party: bool = False) -> dict:
    """三方两两 ULP 扫描入口（cross_check.compare_impl 调用）。

    t/g/b：已 flatten 的三方 ndarray（元素数已由 _precheck 校验一致）；
    output_dtype：声明的输出 dtype（刻度依据，别名/dtype 对象均可）；
    golden_is_third_party：golden 与 third_party 是否同一份数据
    （--golden-mode XPU，调用方以 np.shares_memory 判定）——据此宽表
    对无 CPU 参与的两列渲染 N/A。
    返回纯 Python 字面量 dict（挂 metrics["ulp"]），结构见模块 docstring。
    """
    dtype_name = _canonical_dtype(output_dtype)
    ruler_table = _ULP_RULERS.get(dtype_name)
    np_dtype = _ruler_np_dtype(dtype_name) if ruler_table is not None else None
    if ruler_table is None or np_dtype is None or np_dtype.itemsize != np.dtype(ruler_table[0]).itemsize:
        # 不支持的 dtype（或无 numpy 承载/承载宽度异常）：显式占位，不静默——
        # 消费方可判别"跑了三方但 ULP 不可算"，而非"没跑三方"
        return {"dtype": dtype_name, "golden_is_third_party": bool(golden_is_third_party), "pairs": {}}
    view_dtype, sign_mask, ones_mask, sub_dtype = ruler_table
    ruler = _Ruler(np_dtype, view_dtype, sub_dtype)
    sources = {"t": t, "g": g, "b": b}
    states = {name: UlpPairScan() for name in _PAIR_ORDER}

    with np.errstate(over="ignore", invalid="ignore"):
        for s in range(0, t.size, CHUNK_SIZE):
            keys, fins = {}, {}
            for role, arr in sources.items():
                blk = arr[s : s + CHUNK_SIZE]
                if blk.dtype != ruler.np_dtype:
                    blk = blk.astype(ruler.np_dtype)  # RNE 折叠（golden Promote / 异 dtype XPU）
                if not blk.flags.c_contiguous:
                    blk = np.ascontiguousarray(blk)
                fins[role] = np.isfinite(blk)
                keys[role] = ulp_monotone_key(blk.view(ruler.view_dtype), sign_mask, ones_mask)
            fin_t, fin_g, fin_b = fins["t"], fins["g"], fins["b"]
            states["npu_cpu"].update(keys["t"], keys["g"], fin_t & fin_g, ruler.sub_dtype)
            states["npu_xpu"].update(keys["t"], keys["b"], fin_t & fin_b, ruler.sub_dtype)
            states["xpu_cpu"].update(keys["b"], keys["g"], fin_b & fin_g, ruler.sub_dtype)
    return {
        "dtype": dtype_name,
        "golden_is_third_party": bool(golden_is_third_party),
        "pairs": {
            name: {"max_ulp": None if states[name].max_ulp < 0 else states[name].max_ulp} for name in _PAIR_ORDER
        },
    }


def pop_ulp_payloads(metrics) -> dict:
    """从 metrics 嵌套中原地摘出【本判据的】"ulp" 子树（leaf pop）。

    摘出条件（检视意见 P2：自定义比对插件允许在 metrics 里返回任意键，
    含名为 "ulp" 的标量——实测会被误摘丢失，故从严判定）：
    1. 叶 metrics 的 standard == "cross_check"（仅本判据挂 ulp 载荷）；
    2. 其 "ulp" 为本模块载荷结构（dict 且含 "pairs" 键）。
    其余（standard="custom"、mix_tolerance、标量 ulp、占位行）一律不动。

    返回 {(mode, out_idx): (ulp_payload, leaf_metrics)}——leaf 为摘出后的
    逐输出 metrics（宽表按其 cross_check 参数列合并）。嵌套两形态（compare()
    返回契约）：平铺 {out_idx: leaf}（aclnn）/ {mode: {out_idx: leaf}}（kernel
    dyn/cst/bin、geir、e2e）；leaf 判据：dict 含 "standard" 键。摘出后
    precision_metrics 各叶不再含 "ulp"（两产物互斥）。
    """
    payloads = {}
    if not isinstance(metrics, dict):
        return payloads

    def _walk(node, mode):
        for key, value in node.items():
            if not isinstance(value, dict):
                continue
            if "standard" in value:
                ulp = value.get("ulp")
                if value.get("standard") == "cross_check" and isinstance(ulp, dict) and "pairs" in ulp:
                    payloads[(mode, key)] = (value.pop("ulp"), value)
            else:
                _walk(value, key)

    _walk(metrics, "")
    return payloads


def iter_detail_rows(payload, metrics, testcase_name, run_mode, output_idx):
    """单输出 → compare_detail.csv 宽表单行：三对 max_ulp 作列 + cross_check 参数并入。

    - ULP 列：npu_cpu/npu_xpu/xpu_cpu 各一列；无 finite 对 → 'N/A'；
      dtype 不支持（pairs 空）→ 三列均 'N/A'；
    - golden_is_third_party=True（--golden-mode XPU：golden 即 third_party 同一
      份数据）→ npu_cpu/xpu_cpu 两列渲染 'N/A'——本无 CPU 参与，npu_xpu 列
      即 target vs golden 的真实度量，xpu_cpu 自比恒 0 无信息量（检视意见 P2：
      不产出误导性的"CPU 零误差"）；
    - 参数列：主表 precision_metrics 的全部参数平铺——standard/level/pass、
      config（三比值 limit + small_value 两件）、result（mare/mere/rmse 实测
      比值 + 小值误差计数）；reason 仅失败时有值。metrics 形态由
      cross_check._build_metrics 约束，.get 防御缺键。
    """
    if not isinstance(payload, dict):
        return
    pairs = payload.get("pairs") or {}
    golden_is_tp = bool(payload.get("golden_is_third_party", False))

    def _cell(name):
        max_ulp = (pairs.get(name) or {}).get("max_ulp")
        return "N/A" if max_ulp is None else max_ulp

    ulp_cells = [
        "N/A" if golden_is_tp else _cell("npu_cpu"),
        _cell("npu_xpu"),
        "N/A" if golden_is_tp else _cell("xpu_cpu"),
    ]

    m = metrics if isinstance(metrics, dict) else {}
    cfg = m.get("config") or {}
    res = m.get("result") or {}
    yield (
        testcase_name,
        run_mode,
        output_idx,
        str(payload.get("dtype", "")),
        golden_is_tp,
        *ulp_cells,
        m.get("standard", ""),
        m.get("level", ""),
        m.get("pass", ""),
        res.get("mare", ""),
        res.get("mere", ""),
        res.get("rmse", ""),
        res.get("small_err_cnt_target", ""),
        res.get("small_err_cnt_third", ""),
        res.get("small_err_ratio", ""),
        cfg.get("mare", ""),
        cfg.get("mere", ""),
        cfg.get("rmse", ""),
        cfg.get("small_value", ""),
        cfg.get("small_value_atol", ""),
        m.get("reason", ""),
    )
