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
Comparison utilities for dynamic shape: 三模式精度比对 + perf fake-pass 判定.
"""

__all__ = ["comparing", "handle_profiling_result"]

# Standard Packages
import logging
from typing import Tuple, Union

import numpy

from ttk.switches import get_global_storage

# Third-Party Packages
from ...utilities import get, is_4bit_dtype, resolve_numpy_dtypes, shape_numel, unpack_4bits
from ..common.comparison import CompareHooks, compare_with_hooks
from ..common.comparison.comparison import PLACEHOLDER_PASS
from .result import ComparisonResult
from .testcase import TestcaseKernel


def comparing(
    dyn_kernel_name: str,
    cst_kernel_name: str,
    bin_kernel_name: str,
    dyn_outputs: Tuple[Union[str, numpy.ndarray]],
    cst_outputs: Tuple[Union[str, numpy.ndarray]],
    bin_outputs: Tuple[Union[str, numpy.ndarray]],
    goldens: Tuple[Union[str, numpy.ndarray]],
    output_dtypes: tuple,
    *,
    standards,
    third_parties=None,
    testcase=None,
    pre_compare=None,
    custom_compare=None,
) -> ComparisonResult:
    # outputs may hold sentinels (DYN_OFF/None) of different length than dtypes;
    # __outputs_to_numpy_arrays + compare() skip sentinels — no length assert.
    output_shapes = getattr(testcase, "flat_output_shapes", None) if testcase is not None else None
    __outputs_to_numpy_arrays(dyn_outputs, output_dtypes, output_shapes, goldens)
    __outputs_to_numpy_arrays(cst_outputs, output_dtypes, output_shapes, goldens)
    __outputs_to_numpy_arrays(bin_outputs, output_dtypes, output_shapes, goldens)
    try:
        logging_data = "\n"
        _std_tokens = sorted({str(s.token) for s in standards}) if standards else []
        _std = ",".join(_std_tokens) if _std_tokens else "unknown"
        logging_data += f"Comparing {dyn_kernel_name} with {_std}\n"
        dyn_precision, _logging_data, d_passed, dyn_m = _compare_mode(
            testcase, dyn_outputs, goldens, output_dtypes, standards, third_parties, pre_compare, custom_compare
        )
        logging_data += _logging_data
        logging_data += f"Comparing {cst_kernel_name} with {_std}\n"
        cst_precision, _logging_data, c_passed, cst_m = _compare_mode(
            testcase, cst_outputs, goldens, output_dtypes, standards, third_parties, pre_compare, custom_compare
        )
        logging_data += _logging_data
        logging_data += f"Comparing {bin_kernel_name} with {_std}\n"
        bin_precision, _logging_data, b_passed, bin_m = _compare_mode(
            testcase, bin_outputs, goldens, output_dtypes, standards, third_parties, pre_compare, custom_compare
        )
        logging_data += _logging_data
        logging.debugc(logging_data)

        # 三态聚合：任一 FAIL → FAIL；有 SKIP（golden Disable）无 FAIL → SKIP
        verdicts = [d_passed, c_passed, b_passed]
        if any(v is False for v in verdicts):
            passed = "FAIL"
        elif any(v is None for v in verdicts):
            passed = "SKIP"
        else:
            passed = "PASS"
        metrics = {"dyn": dyn_m, "cst": cst_m, "bin": bin_m}
    except Exception:
        logging.exception("Comparison failed")
        return ComparisonResult("COMPARE_FAILURE")
    return ComparisonResult(None).set(dyn_precision, cst_precision, bin_precision, passed, metrics)


def _compare_mode(testcase, outputs, goldens, output_dtypes, standards, third_parties, pre_compare, custom_compare):
    # 统一比对入口（钩子逻辑供各通路共用——GEIR 原先没有这套逻辑，自实现 compare
    # 在该通路静默失效）。此处保留同名薄封装，kernel 通路调用点与行为不变。
    return compare_with_hooks(
        testcase,
        outputs,
        goldens,
        standards,
        CompareHooks(
            output_dtypes=output_dtypes,
            third_parties=third_parties,
            pre_compare=pre_compare,
            custom_compare=custom_compare,
        ),
    )


def __outputs_to_numpy_arrays(outputs, output_dtypes, output_shapes=None, goldens=None):
    output_dtypes = resolve_numpy_dtypes(output_dtypes)
    for idx, output in enumerate(outputs):
        if isinstance(output, (str, type(None))):
            continue
        output_dtype = get(output_dtypes, idx)
        # Probably ctypes char array
        if "complex32" in str(output_dtype):
            outputs[idx] = numpy.frombuffer(output, "float16")
        elif is_4bit_dtype(output_dtype):
            output_uint8 = numpy.frombuffer(output, "uint8")
            outputs[idx] = unpack_4bits(output_uint8, output_dtype)
        elif str(output_dtype) == "uint1":
            outputs[idx] = numpy.frombuffer(output, "uint8")
        else:
            outputs[idx] = numpy.frombuffer(output, output_dtype)
        __restore_output_shape(
            outputs, idx, get(output_shapes, idx) if output_shapes else None, get(goldens, idx) if goldens else None
        )


def __restore_output_shape(outputs, idx, declared_shape, golden):
    """数据源头形状还原（flat 输入 → 源头嵌套）。

    frombuffer/unpack 产出恒为 1-D；钩子作者（pre_compare / Spec.compare）按真实
    张量书写，标准判据入口自行拍扁不受影响。优先按输出【自身声明】的 shape 还原
    （声明排布 ≠ golden 时钩子拿到输出真形而非 golden 形状——形状真不一致时应
    诚实暴露）；声明缺失（动态 shape 未知）回退 golden 形状（该场景 golden 是
    形状真值）；均不匹配保持扁平。
    """
    arr = outputs[idx]
    for shape in (declared_shape, getattr(golden, "shape", None)):
        if shape is not None and shape_numel(shape) == arr.size:
            outputs[idx] = arr.reshape(shape)
            return


def handle_profiling_result(context: TestcaseKernel):
    """
    Perf fake-pass 判定：三模式 cycle 占位集合（PLACEHOLDER_PASS 单表，
    与比对侧精度列的占位判定同源）按 PASS 放行，数值 ≤ 0 / 非数值 FAIL。
    """

    # noinspection PyBroadException
    def _get_cycle(result):
        if isinstance(result.cycle, str):
            if result.cycle in PLACEHOLDER_PASS:
                return "PASS"
            cs = result.cycle.split(",")
            if all(s.strip() == "OK" for s in cs):  # in case  --task-prof=false
                return "PASS"
            # NPUSim (ASCEND_CAMODEL) simulation reports no single-valued cycle
            # ("_process_model_cycles" hardcodes "UNKNOWN"); treat that as "not
            # measured" rather than a profiling failure. Real devices keep their
            # original behaviour (is_model() is False there).
            if get_global_storage().mode.is_model() and result.cycle == "UNKNOWN":
                return "PASS"
        elif get_global_storage().mode.is_model() and isinstance(result.cycle, (list, tuple)):
            if all(isinstance(s, int) and s > 0 for s in result.cycle):
                return "PASS"

        _passed, _cycle_f = "EXCEPTION", "RTS_PROF_INVALID"
        try:
            _cycle_f = float(result.cycle)
            if _cycle_f <= 0:
                _cycle_f = "RTS_PROF_INVALID"
            else:
                _passed = "PASS"
        except Exception:
            _cycle_f = "RTS_PROF_INVALID"
        if _passed == "PASS":
            result.cycle = _cycle_f
        return _passed

    dyn_pass = _get_cycle(context.dyn_prof_result)
    cst_pass = _get_cycle(context.cst_prof_result)
    bin_pass = _get_cycle(context.bin_prof_result)

    passed = all(s == "PASS" for s in (dyn_pass, cst_pass, bin_pass))
    return "PASS" if passed else "FAIL"
