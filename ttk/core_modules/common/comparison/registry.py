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
comparison method register
"""

# Standard Packages
import gc
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from ....utilities import get


@dataclass
class EachCompareResult:
    precision: Union[str, float, int]
    diff_index: Optional[np.ndarray] = None
    is_pass: bool = False
    log: str = ""
    standard: str = ""
    metrics: dict = field(default_factory=dict)
    error_info: Optional[str] = None  # custom compare 接口失败时填；compare() 打 ERROR 日志
    # diff_index 的排序语义（如 "worst-first by error"）；None = 索引升序。
    # _log_diff_output 摘要行带此标签，回显块自解释（判据间口径本就不同）
    diff_order: Optional[str] = None


class _FirstK:
    """分块前 K 收集器（与 _TopK 对偶）：索引升序进队、只保留前 K 个。

    diff_index 仅被 _log_diff_output 消费（打印前 MAX_DIFF_OUTPUT 行），故
    索引升序的判据只需前 K 个失败位，无需全量索引数组。"""

    __slots__ = ("k", "_items")

    def __init__(self, k: int):
        self.k = k
        self._items = []

    def add(self, idx: np.ndarray):
        """idx：本块的失败位（全局索引，任意顺序——按入队先后截前 K）。"""
        if idx.size and len(self._items) < self.k:
            self._items.extend(idx[: self.k - len(self._items)].tolist())

    def result(self):
        """前 K 个失败位（升序数组）；无候选返回 None。"""
        return np.asarray(self._items, dtype=np.int64) if self._items else None


def _legacy_seq(legacy: dict, key: str) -> list:
    """legacy 容差（CSV 列，per-output 标量或序列）统一转列表，供 get() 按输出位取值。"""
    value = (legacy or {}).get(key, None)
    return value if isinstance(value, (tuple, list)) else [value]


def _ptol_from_options(options: dict, default: float) -> float:
    """从判据参数读 ptol（允许超差元素占比）；缺省/非法值回落 default。"""
    try:
        return float((options or {}).get("ptol", default) or default)
    except (TypeError, ValueError):
        return default


def _to_numpy(arr):
    """compare() 入口把 torch 统一转 numpy（numpy 直通）。
    复用 utilities.dtypes.torch_to_numpy_tensor（处理 bf16/fp8/complex32/普通）。"""
    if not hasattr(arr, "numel"):
        return arr
    from ....utilities.dtypes import torch_to_numpy_tensor

    return torch_to_numpy_tensor(arr)


# 分块扫描大小：判据 FAIL 路径的全量临时数组 + 全量 int64 索引 + argsort
# 约占数十字节/元素，大数组（数百 M 元素）会把 worker 推向 OOM。各数值判据
# 统一按本块大小分块——中间数组 O(CHUNK_SIZE) 不随元素数增长。
# 恒为 2^20：被所有 dtype itemsize（均为 2 的幂）整除，字节级分块不切断元素。
CHUNK_SIZE = 1_048_576


class _TopK:
    """分块 top-K 收集器：key 降序（worst-first），并列按全局索引升序。

    各块先局部 top-K（argpartition），全局 top-K 必在诸块 top-K 的并集内；
    块按全局升序进队 → 拼接后整体 stable argsort(-key) 与全量
    argsort(-key, kind='stable') 的排序一致（除非并列值在块边界被
    argpartition 任意取舍——并列位展示顺序可能与全量实现不同，判定不受影响）。"""

    __slots__ = ("k", "_keys", "_idxs")

    def __init__(self, k: int):
        self.k = k
        self._keys, self._idxs = [], []

    def add(self, key: np.ndarray, idx: np.ndarray):
        """key/idx：本块的 (排序键, 全局索引)，idx 需升序。"""
        if key.size == 0:
            return
        if key.size > self.k:
            # stable argsort 截前 k：并列键按索引升序取舍——块间拼接后全局
            # stable 排序与全量 argsort(-key, kind='stable') 完全一致（含并列，
            # 如 mix 的 NaN-mismatch 全映射为 inf 键）。比 argpartition 多一个
            # log 因子，换取与全量实现的逐位等价。
            order = np.argsort(-key, kind="stable")[: self.k]
            key, idx = key[order], idx[order]
        self._keys.append(key)
        self._idxs.append(idx)

    def result(self):
        """全局 top-K 索引（worst-first）；无候选返回 None。"""
        if not self._keys:
            return None
        key = np.concatenate(self._keys)
        idx = np.concatenate(self._idxs)
        return idx[np.argsort(-key, kind="stable")[: self.k]]


class ComparisonBase(metaclass=ABCMeta):
    # Max diff entries printed by _log_diff_output (worst-first). One place to tune.
    MAX_DIFF_OUTPUT = 100

    # 容差消费面与默认值：None = 该判据不消费此容差（legacy 列有值也静默忽略，
    # 与旧 __post_init__ 按需解析一致）；标量 = 兜底默认。默认值依赖 dtype 分档
    # 或运行期条件的判据（isclose/requant）覆写 _default_rtol/_default_ptol 钩子，
    # 类属性退化为消费声明（值可为分档表，由钩子解读）。
    _DEFAULT_RTOL = None
    _DEFAULT_ATOL = None
    _DEFAULT_PTOL = None

    def __init__(
        self,
        output: Union[np.ndarray, np.ndarray],
        golden: Union[np.ndarray, np.ndarray],
        output_idx: int,
        output_dtype: str,
        options: dict = None,
        third_party=None,
    ):
        if not options:
            options = {}
        self.tol_options = options
        self.output = output.reshape([-1])
        self.golden = golden.reshape([-1])
        # third_party 跟 output/golden 同款 flatten；None 位置跳过 reshape（cross_check → GOLDEN_FAILURE）
        self.third_party = third_party.reshape([-1]) if third_party is not None else None
        self._pad_int4_if_need()
        self.output_idx = output_idx
        self.output_dtype = output_dtype
        self.__post_init__()
        self.standard = getattr(type(self), "STANDARD_NAME", "")

    def __post_init__(self):  # noqa: B027
        pass

    @abstractmethod
    def compare_impl(self) -> EachCompareResult:
        pass

    def _precheck(self) -> Optional[EachCompareResult]:
        """都空→PASS；一方空→FAIL；元素数不一致→FAIL；其余→None（进 compare_impl）。"""
        out_empty = self.output.size == 0
        gold_empty = self.golden.size == 0
        if out_empty and gold_empty:
            return EachCompareResult(
                1, is_pass=True, standard=self.standard, metrics={"standard": self.standard, "pass": True}
            )
        if out_empty or gold_empty:
            return EachCompareResult(
                0, is_pass=False, standard=self.standard, metrics={"standard": self.standard, "pass": False}
            )
        if self.output.size != self.golden.size:
            precision = f"{self.output.size} vs {self.golden.size}"
            log = f"Output {self.output_idx} size is different with golden size: {precision}\n"
            return EachCompareResult(precision, None, False, log)
        return None

    def _finalize_tolerances(self):
        """rtol/atol/ptol 解析为终值成员变量（compare_impl 直接消费）。

        来源优先级：legacy CSV 列（per-output，get 按输出位取值）
        > Spec.tolerance 顶层（仅 ptol——quant/requant 通路经 _ptol_params 注入）
        > 判据默认钩子（标量默认读类属性；dtype 分档/运行期条件由子类覆写）。
        ptol=0 须与"未配置"区分（0 是合法的零容忍），顶层解析不走 or 短路。
        """
        legacy = self.tol_options.get("legacy", {})
        for name, declared in (
            ("rtol", self._DEFAULT_RTOL),
            ("atol", self._DEFAULT_ATOL),
            ("ptol", self._DEFAULT_PTOL),
        ):
            if declared is None:
                continue  # 判据不消费此容差（legacy 值静默忽略——消费面由判据声明）
            value = get(_legacy_seq(legacy, name), self.output_idx)
            if value is None and name == "ptol":
                raw = self.tol_options.get("ptol")
                if raw is not None:
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        value = None
            if value is None:
                value = getattr(self, f"_default_{name}")(self.output.dtype)
            setattr(self, name, value)

    def _default_rtol(self, dtype):
        return self._DEFAULT_RTOL

    def _default_atol(self, dtype):
        return self._DEFAULT_ATOL

    def _default_ptol(self, dtype):
        return self._DEFAULT_PTOL

    def compare(self):
        # 入口统一 numpy（torch 输入转 numpy；比对类内部纯 numpy）
        self.output = _to_numpy(self.output)
        self.golden = _to_numpy(self.golden)
        self.third_party = _to_numpy(self.third_party) if self.third_party is not None else None
        # 入口前置短路（唯一点）：都空→PASS、一方空或元素数不一致→FAIL；
        # compare_impl 只处理非空同数
        compare_result = self._precheck()
        if compare_result is None:
            # 容差终值化在 _to_numpy 之后：分档默认（fp8 家族）依据的是 torch→numpy
            # 转换后的 dtype；compare_impl 只见终值（per-output 取位/默认兜底在此单点）
            self._finalize_tolerances()
            compare_result = self.compare_impl()
        if compare_result.error_info:  # 失败原因→ERROR 日志（空不打；不进 metrics）
            logging.error("Output %d %s", self.output_idx, compare_result.error_info)
        if isinstance(compare_result.precision, str):
            precision = compare_result.precision
        else:
            precision = f"{compare_result.precision * 100}%"
        compare_result.log += f"Output {self.output_idx} Precision is {precision}\n"
        compare_result.log += self._log_diff_output(compare_result.diff_index, compare_result.diff_order)
        gc.collect()
        return precision, compare_result.log, compare_result.is_pass, compare_result.metrics

    def _pad_int4_if_need(self):
        # only numpy dtype has `name` attribute. Op api output will always be torch.Tensor.
        from ....utilities.dtypes import is_4bit_dtype

        # output will always be 2*n
        if (
            hasattr(self.output.dtype, "name")
            and is_4bit_dtype(self.output.dtype)
            and self.output.size >= 2
            and self.golden.size == self.output.size - 1
        ):
            self.golden = np.pad(self.golden, (0, 1), mode="constant")

    @staticmethod
    def _fmt_log_value(value, spec=":<14.18f") -> str:
        """复数和整数走 str 保留原值，避免 float() 丢失虚部或整数精度，其余按 spec 定宽。"""
        if isinstance(value, (complex, np.complexfloating)):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return str(value)
        return format(float(value), spec)

    def _log_diff_output(self, diff_index, diff_order=None) -> str:
        log = ""
        if diff_index is not None:
            diff_index_size = diff_index.size
            golden_size = self.golden.size
            log += (
                f"Output {self.output_idx} Compare Difference length {diff_index_size} "
                f"({diff_order or 'index ascending'})\n"
            )
            for idx in range(min(diff_index_size, self.MAX_DIFF_OUTPUT)):
                real_index = diff_index[idx]
                golden_v = self.golden[real_index] if golden_size > 0 else 0
                actual_v = self.output[real_index]
                if golden_size <= 0:
                    log += (
                        f"Index: {idx:03d} RealIndex: {real_index:06d} Expected: 0 "
                        f"Actual: {self._fmt_log_value(actual_v)}, Diff: inf\n"
                    )
                elif any(s in str(self.golden.dtype) for s in ("float8", "float4", "hifloat8")):
                    # out-of-range number will be nan/inf in
                    # `- (golden_v - actual_v) / golden_v`. convert to float.
                    tmp_array = np.array([float(golden_v), float(actual_v)], dtype=np.float32)
                    log += (
                        f"Index: {idx:03d} RealIndex: {real_index:06d} Expected: {self._fmt_log_value(golden_v)} "
                        f"Actual: {self._fmt_log_value(actual_v)} "
                        f"Diff: {self._fmt_log_value(-(tmp_array[0] - tmp_array[1]) / tmp_array[0], ':<14.06f')}\n"
                    )
                elif "bool" not in str(self.golden.dtype):
                    # (golden_v - actual_v) is different with (actual_v - golden_v) in BFP16 vs FP32-Golden
                    log += (
                        f"Index: {idx:03d} RealIndex: {real_index:06d} Expected: {self._fmt_log_value(golden_v)} "
                        f"Actual: {self._fmt_log_value(actual_v)} "
                        f"Diff: {self._fmt_log_value(-(golden_v - actual_v) / golden_v, ':<14.06f')}\n"
                    )
                else:
                    gold_str, actual_str = str(golden_v), str(actual_v)
                    if "bool" in str(self.output.dtype):
                        # 非规范 bool 字节（如 0x02）值层仍渲染 True，不带字节值会呈现
                        # "Expected: True Actual: True Diff: FAIL" 的自相矛盾日志
                        gold_str = f"{gold_str}(0x{self.golden.view(np.uint8)[real_index]:02x})"
                        actual_str = f"{actual_str}(0x{self.output.view(np.uint8)[real_index]:02x})"
                    log += (
                        f"Index: {idx:03d} RealIndex: {real_index:06d} "
                        f"Expected: {gold_str} Actual: {actual_str} Diff: FAIL\n"
                    )
        return log


class ComparisonRegister:
    registry: Dict[str, type] = {}  # token(lower) -> 比对类；只此一份，无 metadata dict


# token → 判据模块（声明式，加判据=加一行；懒加载——首次查询才 import，触发
# @register_comparison 注册。跨_check 场景只 import cross_check 及其依赖；
# 判据模块顶层可自由 import 重依赖（torch 等）而不拖累轻量路径）
_TOKEN_MODULES = {
    "binary_equal": ".binary_equal",
    "bin": ".binary_equal",
    "binary": ".binary_equal",  # CLI --compare / Spec standard 取值（CLI choices 含 binary）
    "mix_tolerance": ".mix_tolerance",
    "mixed": ".mix_tolerance",  # CLI 别名（3675093）
    "stat_rel_err": ".stat_rel_err",
    "close": ".is_close",
    "isclose": ".is_close",
    "cosine": ".cosine_similarity",
    "cross_check": ".cross_check",
    "quant": ".quant",
    "requant": ".re_quantize",
}


def get_comparison(token: str) -> type:
    """token → 比对类。首次查询 import 对应模块（触发 @register_comparison），
    之后 registry 缓存。未知 token 抛 ValueError（列出已注册 token）。"""
    key = token.lower()
    cls = ComparisonRegister.registry.get(key)
    if cls is not None:
        return cls
    module_name = _TOKEN_MODULES.get(key)
    if module_name is not None:
        import importlib

        module = importlib.import_module(module_name, __package__)
        cls = ComparisonRegister.registry.get(key)
        if cls is not None:
            return cls
        raise ValueError(f"comparison module {module.__name__} loaded but token [{token}] not registered")
    raise ValueError(
        f"Comparison standard [{token}] is not recognized. Registered: {sorted(ComparisonRegister.registry)}"
    )


# 失败 reason 模板（PASS 无 reason）；带 {占位符} 的由各比对类 .format() 填实值
FAIL_REASONS = {
    "nan_inf_mismatch": "NaN/Inf mismatch",
    "threshold_exceeded": "threshold exceeded: {metrics}",
    "bitwise_mismatch": "bitwise mismatch",
    "cross_dtype_uncomparable": "cross-dtype not comparable",
    "tolerance_exceeded": "exceeded atol/rtol",
    "similarity_below_threshold": "cosine similarity below threshold",
    "precision_exceeded": "exceeded ptol",
    "third_party_count_mismatch": "third_party count != outputs",
    "third_party_unavailable": "third_party unavailable",
    "ratio_exceeded": "ratio exceeded: {exceeded}",
    "small_value_exceeded": "small value ErrorCount ratio exceeded: {ratio}",
}


def register_comparison(methods):
    if not isinstance(methods, (list, tuple, str)):
        raise TypeError(f"Register class for comparison must receive a list / tuple / str, not {methods}")
    if isinstance(methods, str):
        methods = [methods]

    def decorator(cls):
        for method in methods:
            key = method.lower()
            if key in ComparisonRegister.registry:
                logging.warning(f"comparison class of [{key}] has already been registered!")
            ComparisonRegister.registry[key] = cls
        return cls

    return decorator
