#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""比对执行引擎（compare_with_hooks 为四通路唯一执行入口）。

三段结构：
- validate_outputs：compare 前的专职分类——逐输出 VALID / 占位（三段式语义见
  PLACEHOLDER_PASS）/ 整体异常（空 outputs、third_parties 数量不符）；
- compare：纯数值比对——占位透传 + VALID 子集按 standards 逐输出分发判据；
- TestSpec 钩子：pre_compare / custom compare（算子自实现比对），与数值比对
  共用一份实现，避免各通路复制漂移。
"""

# Standard Packages
import gc
import inspect
import logging
import reprlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np

from ....utilities import get
from ....utilities.container import deep_flatten, nest_by_distribution

# Third-party Packages
from .registry import FAIL_REASONS, get_comparison

__all__ = [
    "PLACEHOLDER_PASS",
    "OutputVerdict",
    "validate_outputs",
    "compare",
    "CompareHooks",
    "compare_with_hooks",
    "CompareContext",
    "invoke_pre_compare",
    "invoke_custom_compare",
]


# ---------------------------------------------------------------------------
# 输出校验（占位分类）
# ---------------------------------------------------------------------------
# fake-pass 集合：模式开关/能力不适用占位 + SUPPRESSED（golden 显式豁免、
# 执行侧模式未跑）。出现在精度列时不判 FAIL（行结论由通路判定——Kernel perf
# 判定共用本表）。golden 生成失败类与算子未找到/输入缺失类不在此表——判
# FAIL：真值缺失与环境错误必须暴露，静默通过会掩盖回归。
# golden 侧 "SUPPRESSED"（golden_mode=Disable）不查本表，_classify_one 单独
# 判 SKIP。
PLACEHOLDER_PASS = frozenset(
    {
        "SUPPRESSED",
        "DYN_OFF",
        "CST_OFF",
        "BIN_OFF",
        "STC_OFF",  # 历史拼写变体（存量结果 CSV 兼容）
        "DYN_UNSUPPORTED",
        "CST_UNSUPPORTED",
        "BIN_UNSUPPORTED",
        "STC_UNSUPPORTED",
    }
)

# 整体异常 status 值
STATUS_EMPTY = "EMPTY"
STATUS_THIRD_PARTY_COUNT_MISMATCH = "THIRD_PARTY_COUNT_MISMATCH"


@dataclass
class OutputVerdict:
    """输出校验结论。

    needs_compare：any 输出为 VALID 时 True（只喂 VALID 路进数值比对）。
    per_output：逐输出结论——VALID 或 (precision, pass_) 占位元组，
    聚合时拼回结果（混合占位场景：占位路透传、有效路比对）。
    log_lines：占位位置留痕行（VALID → None）——占位必须在比对日志中可见：
    SUPPRESSED/占位必须在比对日志中可见，否则该位置凭空消失。
    status：整体异常（EMPTY / THIRD_PARTY_COUNT_MISMATCH → COMPARE_FAILURE）。
    """

    needs_compare: bool = False
    per_output: list = field(default_factory=list)
    status: Optional[str] = None
    log_lines: list = field(default_factory=list)


def _classify_one(output, golden) -> tuple:
    """单输出分类：(is_valid, entry, log_suffix)——entry 为 VALID 哨兵或 (precision, pass_)。

    pass_ 三态：True=放行（模式开关/显式豁免）、False=FAIL、None=SKIP。
    """
    if isinstance(golden, str) and golden == "SUPPRESSED":
        # golden 侧 SUPPRESSED 只由 golden_mode=Disable 产出，与 golden=None
        # 的单输出豁免语义相反（PASS），须区分；先于 output 哨兵判定——Disable
        # 下即使该模式未跑（output 为 *_OFF），整轮也没做过比对，仍判 SKIP
        return False, (golden, None), " (golden disabled)"
    if isinstance(output, str):
        return False, (output, output in PLACEHOLDER_PASS), " (kernel output sentinel)"
    if isinstance(golden, str):
        return False, (golden, golden in PLACEHOLDER_PASS), " (golden sentinel)"
    if golden is None:
        return False, ("SUPPRESSED", True), " (golden is None)"
    if output is None:
        return False, ("NO_OUTPUT", False), " (no kernel output data)"
    return True, "VALID", None


def validate_outputs(outputs, goldens, third_parties=None) -> "OutputVerdict":
    """输出校验：逐输出分类 + 聚合。

    分类：VALID（双 ndarray）/ golden 占位（"SUPPRESSED" → 透传+SKIP，golden
    Disable；str ∈ PLACEHOLDER_PASS → 透传+PASS；str ∉ 表 → 透传+FAIL——
    UNSUPPORTED/GOLDEN_TIMEOUT/GOLDEN_FAILURE 即此路）/ golden None →
    SUPPRESSED+PASS / output None → NO_OUTPUT+FAIL；整体：空 outputs →
    EMPTY；third_parties 数 < outputs → THIRD_PARTY_COUNT_MISMATCH（cross_check
    前置数据不足）。
    """
    verdict = OutputVerdict()
    if not outputs:
        verdict.status = STATUS_EMPTY
        return verdict

    if third_parties and not isinstance(outputs[0], str) and len(third_parties) < len(outputs):
        verdict.status = STATUS_THIRD_PARTY_COUNT_MISMATCH
        return verdict

    for output, golden in zip(outputs, goldens):
        is_valid, entry, log_suffix = _classify_one(output, golden)
        verdict.needs_compare = verdict.needs_compare or is_valid
        verdict.per_output.append(entry)
        if is_valid:
            verdict.log_lines.append(None)
        else:
            verdict.log_lines.append(f"Output {len(verdict.log_lines)} Precision is {entry[0]}{log_suffix}\n")
    return verdict


# ---------------------------------------------------------------------------
# 数值比对（compare_with_hooks 的回落层；自身占位安全——占位透传 + VALID 子集分发）
# ---------------------------------------------------------------------------
def _fold_pass(passes: list) -> Optional[bool]:
    """三态聚合：任一 False → FAIL；有 None（未比对位）→ SKIP；全 True → PASS。"""
    if any(p is False for p in passes):
        return False
    if any(p is None for p in passes):
        return None
    return True


def compare(
    outputs: Union[Tuple[np.ndarray], List[np.ndarray]],
    goldens: Union[Tuple[np.ndarray], List[np.ndarray]],
    output_dtypes: tuple,
    *,
    standards,
    third_parties=None,
    _verdict=None,
) -> Tuple[str, str, Optional[bool], dict]:
    """逐输出比对（占位安全：str/None 占位透传，仅 VALID 走数值判据）。

    standards: resolve_tolerance 产出（与 outputs 同长）。
    返回 (precision_str, log, is_pass, metrics_dict)——metrics 键为原始输出序号；
    is_pass 三态：True=PASS / False=FAIL / None=SKIP（golden Disable，未比对）。
    _verdict: 内部参数——compare_with_hooks 已算过的分类结论，免二次扫描；
    钩子路径传入的是变换后数据，必须留空重算。
    """
    verdict = _verdict if _verdict is not None else validate_outputs(outputs, goldens, third_parties)
    if verdict.status == STATUS_EMPTY:
        return ("UNKNOWN", "Output or Golden data is empty, compare result UNKNOWN\n", False, {})
    if verdict.status == STATUS_THIRD_PARTY_COUNT_MISMATCH:
        return (
            "COMPARE_FAILURE",
            f"third_parties({len(third_parties)}) < outputs({len(outputs)})",
            False,
            {"reason": FAIL_REASONS["third_party_count_mismatch"]},
        )
    if not verdict.needs_compare:
        precisions = [entry[0] for entry in verdict.per_output]
        passes = [entry[1] for entry in verdict.per_output]
        log = "".join(line for line in verdict.log_lines if line)
        return ",".join(precisions), log, _fold_pass(passes), {}

    total_precision, total_pass, total_log, total_metrics = [], [], "", {}
    for idx, (output, golden) in enumerate(zip(outputs, goldens)):
        if verdict.per_output[idx] != "VALID":
            total_log += verdict.log_lines[idx] or ""
            continue
        if idx >= len(standards):
            # 前置契约（standards 与 outputs 同长）被破坏时给出可定位信息，
            # 而非裸 list index out of range（如 GEIR shape-unknown 元数据槽
            # 并入后未重解析 standards 的失配）。
            raise IndexError(
                f"standards({len(standards)}) mismatch with outputs({len(outputs)}): "
                f"no tolerance standard for output[{idx}] "
                f"(resolve_tolerance out of sync with output slots)"
            )
        token = standards[idx].token
        comparison = get_comparison(token)
        c = comparison(
            output,
            golden,
            idx,
            get(output_dtypes, idx),
            standards[idx].params,
            third_party=get(third_parties, idx) if third_parties else None,
        )

        precision, log, is_pass, metrics = c.compare()
        total_precision.append(precision)
        total_pass.append(is_pass)
        total_log += log
        total_metrics[idx] = metrics
        gc.collect()

    # 原位拼回：占位路透传 (precision, pass)，VALID 路取子比对真实结果
    precisions, passes, sub_idx = [], [], 0
    for entry in verdict.per_output:
        if entry == "VALID":
            precisions.append(total_precision[sub_idx])
            passes.append(total_pass[sub_idx])
            sub_idx += 1
        else:
            precisions.append(entry[0])
            passes.append(entry[1])
    return ",".join(precisions), total_log, _fold_pass(passes), total_metrics


# ---------------------------------------------------------------------------
# TestSpec 钩子（pre_compare / custom compare）与统一入口
# ---------------------------------------------------------------------------
def _copy_goldens(goldens):
    """比对钩子可能就地改写 golden，先复制以免污染后续使用（如 dump）。"""
    copied = []
    for golden in goldens:
        if isinstance(golden, np.ndarray):
            copied.append(golden.copy())
        elif hasattr(golden, "clone"):
            copied.append(golden.clone())
        else:
            copied.append(golden)
    return copied


@dataclass
class CompareHooks:
    """比对钩子与数据打包（compare_with_hooks 的 5 参契约载体）。"""

    output_dtypes: tuple = ()
    third_parties: Optional[Any] = None
    pre_compare: Optional[Any] = None
    custom_compare: Optional[Any] = None


def compare_with_hooks(testcase, outputs, goldens, standards, hooks: CompareHooks):
    """带 TestSpec 钩子（pre_compare / compare）的比对统一入口（四通路共用）。

    流程：
    1) validate_outputs 分类——无钩子/输出占位 → 委托 compare()（占位路透传 +
        VALID 子集走标准比对，分类结论复用不重算）；
    2) 声明钩子且输出均为真实数组 → 钩子路径：列表浅拷贝隔离 + 复制 golden
        防改写 → pre_compare → custom compare（非 None 采信）→ 回落标准
        compare（对变换后数据重新分类）。golden 的 str 哨兵由
        invoke_pre_compare/invoke_custom_compare 内部 _can_customize 拦截；
        None（抑制位，如 return_softmax_lse=False 的 LSE）交由插件钩子
        自行处理，不禁用钩子（对齐重构前 comparison/custom.py 门控语义）。

    【形状契约】钩子拿到即真实形状——由各通路数据源头保证（kernel 在
    __outputs_to_numpy_arrays 解码处按声明 shape 还原，geir 在 _parse_stdout
    还原，aclnn/e2e 输出天生真实形状）；本函数不做形状兜底（按 golden 形状
    还原输出会在声明排布 ≠ golden 时给钩子错误形状）。

    【为何共用】算子自实现 compare 是刚需而非锦上添花（如 NonZeroWithValue 的
    静态 max-size buffer 须只比有效前缀 [0:N]）——钩子逻辑各通路共用，避免复制漂移。
    """
    verdict = validate_outputs(outputs, goldens, hooks.third_parties)
    hooks_enabled = testcase is not None and (hooks.pre_compare is not None or hooks.custom_compare is not None)
    # 重构前语义：只要输出是真实数组即可进钩子；golden 的 str 哨兵在
    # invoke_pre_compare/invoke_custom_compare 内部拦截，None 抑制位交由插件处理
    has_runtime_output = bool(outputs) and not any(isinstance(output, (str, type(None))) for output in outputs)
    if hooks_enabled and has_runtime_output:
        mode_outputs = list(outputs)  # 浅拷贝隔离：invoke_pre_compare 原地改写不外溢到调用方列表
        mode_goldens = _copy_goldens(goldens)
        invoke_pre_compare(testcase, mode_outputs, mode_goldens, hooks.pre_compare)
        custom_result = invoke_custom_compare(testcase, mode_outputs, mode_goldens, hooks.custom_compare)
        if custom_result is not None:
            precision, logging_data, passed, custom_metrics = custom_result
            return precision, logging_data, passed, custom_metrics
        return compare(
            mode_outputs,
            mode_goldens,
            hooks.output_dtypes,
            standards=standards,
            third_parties=hooks.third_parties,
        )
    return compare(
        outputs,
        goldens,
        hooks.output_dtypes,
        standards=standards,
        third_parties=hooks.third_parties,
        _verdict=verdict,
    )


@dataclass(frozen=True)
class CompareContext:
    """Replay-safe input context available to an explicitly opted-in compare hook."""

    api_name: Optional[str]
    testcase_name: Optional[str]
    input_tensors: Any
    input_scalars: Any
    attributes: Mapping[str, Any]
    csv_fields: Mapping[str, Any]


def _can_customize(outputs, goldens):
    # str 哨兵（SUPPRESSED/UNSUPPORTED/GOLDEN_FAILURE 等）不进钩子；None golden
    # （抑制位置，如 shape 大小为 0 的输出）交由插件钩子自行处理，不禁用钩子
    return bool(outputs) and bool(goldens) and not any(isinstance(golden, str) for golden in goldens)


def _fold_outputs(testcase, outputs, goldens):
    output_dist = getattr(testcase, "output_dist", None)
    if output_dist is None:
        output_dist = getattr(testcase, "output_distribution", None)
    if output_dist:
        return nest_by_distribution(outputs, output_dist), nest_by_distribution(goldens, output_dist)
    return list(outputs), list(goldens)


def invoke_pre_compare(testcase, outputs, goldens, func):
    """Apply a TestSpec pre_compare function to flat output and golden lists."""
    if func is None or not _can_customize(outputs, goldens):
        return

    nested_outputs, nested_goldens = _fold_outputs(testcase, outputs, goldens)
    transformed = func(*nested_outputs, *nested_goldens)
    if transformed is None:
        return

    output_count = len(nested_outputs)
    expected_count = output_count + len(nested_goldens)
    if not isinstance(transformed, (list, tuple)) or len(transformed) != expected_count:
        actual_count = len(transformed) if hasattr(transformed, "__len__") else "?"
        raise ValueError(
            f"[{testcase.testcase_name}] pre_compare returned len={actual_count}, "
            f"expected {expected_count} (npu={output_count} + golden={len(nested_goldens)})"
        )

    flat_outputs = deep_flatten(transformed[:output_count])
    flat_goldens = deep_flatten(transformed[output_count:])
    if len(flat_outputs) != len(outputs) or len(flat_goldens) != len(goldens):
        raise ValueError(
            f"[{testcase.testcase_name}] pre_compare unfolded len mismatch: "
            f"npu={len(flat_outputs)}/{len(outputs)}, "
            f"golden={len(flat_goldens)}/{len(goldens)} (check tensor-list nesting)"
        )
    outputs[:] = flat_outputs
    goldens[:] = flat_goldens


def _compare_kwargs(testcase):
    from ttk.core_modules.common.testcase.batch import batch_relation_kwargs

    kwargs = batch_relation_kwargs(testcase)
    batch_consistency_id = getattr(testcase, "batch_consistency_id", None)
    if kwargs and batch_consistency_id is not None:
        kwargs["batch_consistency_id"] = batch_consistency_id
    return kwargs


def _read_only_mapping(value):
    return MappingProxyType(dict(value if value is not None else {}))


def _compare_context(testcase):
    api_name = getattr(testcase, "api_name", None)
    if api_name is None:
        api_name = getattr(testcase, "op_name", None)
    input_tensors = getattr(testcase, "tensors", None)
    if input_tensors is None:
        input_tensors = getattr(testcase, "input_arrays", None)
    return CompareContext(
        api_name=api_name,
        testcase_name=getattr(testcase, "testcase_name", None),
        input_tensors=input_tensors,
        input_scalars=getattr(testcase, "scalars", ()),
        attributes=_read_only_mapping(getattr(testcase, "attributes", None)),
        csv_fields=_read_only_mapping(getattr(testcase, "original_dict", None)),
    )


def invoke_custom_compare(testcase, outputs, goldens, func):
    """Run a TestSpec compare function and normalize its public result contract."""
    if func is None or not _can_customize(outputs, goldens):
        return None

    nested_outputs, nested_goldens = _fold_outputs(testcase, outputs, goldens)
    kwargs = _compare_kwargs(testcase)
    signature = inspect.signature(func)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not accepts_kwargs:
        kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}

    context_param = signature.parameters.get("compare_context")
    if context_param is not None:
        if context_param.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("compare_context must be a keyword or keyword-only parameter")
        kwargs["compare_context"] = _compare_context(testcase)

    result = func(*nested_outputs, *nested_goldens, **kwargs)
    return _normalize_custom_result(testcase, result)


def _format_failure_detail(value):
    """Bound nested plugin diagnostics before sending them to logs or the terminal."""
    if isinstance(value, str):
        text = value[:512].replace("\n", "\\n")
        return text + (f"... (total chars={len(value)})" if len(value) > 512 else "")
    formatter = reprlib.Repr()
    formatter.maxlevel = 3
    formatter.maxlist = formatter.maxtuple = formatter.maxdict = formatter.maxset = 20
    formatter.maxstring = formatter.maxother = 512
    text = formatter.repr(value)
    if len(text) > 2048:
        text = text[:2048] + "... (truncated)"
    try:
        count = len(value)
    except TypeError:
        count = None
    if count is not None and count > 20:
        text += f" (total={count})"
    return text


def _csv_literal(value):
    """numpy 标量/数组 → Python 字面量（precision_metrics 列 eval 往返约束）。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _csv_literal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_csv_literal(v) for v in value]
    return value


# 结构化条目的容器界，与 _format_failure_detail 的 reprlib maxlist 同值（前 20）
_CUSTOM_LIST_CAP = 20


_CUSTOM_RESERVED_KEYS = frozenset({"standard", "pass", "reason", "diff_indices", "diff_indices_total"})


def _structured_custom_entry(item, reason):
    """失败 custom 结果 → 结构化条目（键与标准比对 metrics 同构：输出序号 → dict）。

    标量字段原样（保字面量）；容器类与日志 detail 行同界（前 20 + total），
    防止 precision_metrics 列被无界插件数据撑爆。保留键（standard/pass/reason/
    diff_indices*）不被插件 metrics 覆盖——standard=="custom" 是回显判别标记。"""
    entry = {"standard": "custom", "pass": False, "reason": reason}
    diff_indices = _csv_literal(item.get("diff_indices"))
    if isinstance(diff_indices, list):
        entry["diff_indices"] = diff_indices[:_CUSTOM_LIST_CAP]
        if len(diff_indices) > _CUSTOM_LIST_CAP:
            entry["diff_indices_total"] = len(diff_indices)
    elif diff_indices is not None:
        entry["diff_indices"] = diff_indices
    plugin_metrics = item.get("metrics")
    if isinstance(plugin_metrics, dict):
        for key, raw_val in plugin_metrics.items():
            if key in _CUSTOM_RESERVED_KEYS:
                continue
            literal = _csv_literal(raw_val)
            entry[key] = (
                literal if isinstance(literal, (int, float, bool, str, type(None))) else _format_failure_detail(literal)
            )
    return entry


def _normalize_custom_result(testcase, result):
    """规整 Spec.compare 返回值：dict/列表/嵌套 → (precision 拼接, error_info 日志, pass, metrics)。

    metrics 键为输出序号（与标准比对 compare() 的 total_metrics 同构），失败输出
    条目含 standard="custom"/pass/reason/diff_indices 及插件标量指标——结构化供
    CSV precision_metrics 与父进程回显消费，文本进日志。
    """
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"[{testcase.testcase_name}] compare must return dict or list[dict]")
    if not result:
        raise ValueError(f"[{testcase.testcase_name}] compare returned empty list")

    items = deep_flatten(result) if any(isinstance(item, (list, tuple)) for item in result) else result
    precisions = []
    passes = []
    log_lines = []
    custom_metrics = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"[{testcase.testcase_name}] compare output[{index}] is not a dict")
        missing_keys = [key for key in ("pass", "precision") if key not in item]
        if missing_keys:
            formatted_keys = ", ".join(repr(key) for key in missing_keys)
            raise ValueError(
                f"[{testcase.testcase_name}] compare output[{index}] missing required key(s): {formatted_keys}"
            )
        precision = item["precision"]
        precisions.append(f"{precision}%" if isinstance(precision, (int, float)) else str(precision))
        passes.append(bool(item["pass"]))
        reason = item.get("error_info")
        if not item["pass"] and not reason:
            reason = "Custom comparison returned pass=False without error_info"
        if reason:
            detail = f"Output {index}: {_format_failure_detail(reason)} (precision={precision})"
            for key in ("diff_indices", "metrics"):
                if key in item:
                    detail += f"; {key}={_format_failure_detail(item[key])}"
            log_lines.append(detail)
            if not item["pass"]:
                logging.error("[%s] %s", testcase.testcase_name, detail)
        if not item["pass"]:
            custom_metrics[index] = _structured_custom_entry(item, _format_failure_detail(reason))

    log_data = "\n".join(log_lines) + ("\n" if log_lines else "")
    return ",".join(precisions), log_data, all(passes), custom_metrics
