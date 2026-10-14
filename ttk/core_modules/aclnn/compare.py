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
Comparison for op api
"""

__all__ = ["Comparator"]


# Standard Packages
import logging

import numpy

from ttk.switches import get_global_storage

from ...utilities import get, is_4bit_dtype, numpy_to_torch_tensor, resolve_numpy_dtypes, unpack_4bits

# Third-party Packages
from ..common.comparison import CompareHooks, compare_with_hooks
from ..common.comparison.resolve import resolve_tolerance
from ..common.plugin import SpecResolver
from ..common.testcase.tensor import make_storage_view
from .result import ApiComparisonResult
from .testcase import TestcaseAclnn


class Comparator:
    def __init__(self, context: TestcaseAclnn, standards=None, third_parties=None):
        self._ctx = context
        self._standards = standards
        self._third_parties = third_parties
        plugin_path = getattr(get_global_storage(), "plugin_path", None)
        self._tolerance = SpecResolver(context.api_name, "aclnn", plugin_path).resolve_tolerance()
        self._pre_compare = SpecResolver(context.api_name, "aclnn", plugin_path).resolve_pre_compare()
        self._custom_compare = SpecResolver(context.api_name, "aclnn", plugin_path).resolve_compare()

    @staticmethod
    def _decode_output_bytes(raw, dtype):
        """Decode raw bytes to numpy array based on dtype."""
        if "complex32" in str(dtype):
            return numpy.frombuffer(raw, "float16")
        if is_4bit_dtype(dtype):
            return unpack_4bits(numpy.frombuffer(raw, "uint8"), dtype)
        return numpy.frombuffer(raw, dtype)

    def _build_torch_view(self, np_array, idx, dtype, outputs):
        import torch

        ctx = self._ctx
        t_storage = numpy_to_torch_tensor(np_array, is_complex32="complex32" in str(dtype))
        t_storage = t_storage.reshape(ctx.flat_output_storage_shapes[idx])
        v_shape = ctx.flat_output_view_shapes[idx]
        v_stride = ctx.flat_output_view_strides[idx]
        v_offset = ctx.flat_output_view_offsets[idx]
        try:
            if t_storage.dim() == 0 and v_shape == ():
                outputs[idx] = t_storage
            else:
                outputs[idx] = torch.as_strided(t_storage, v_shape, v_stride, v_offset)
            ret_v_shape = tuple(ctx.prof_result.output_view_shapes[idx])
            if ret_v_shape != tuple(v_shape):
                outputs[idx] = outputs[idx].resize_(ret_v_shape)
        except RuntimeError:
            logging.error(
                f"torch.as_strided failed. storage_shape={t_storage.shape} "
                f"view_shape={v_shape}, view_stride={v_stride}, view_offset={v_offset}"
            )
            raise

    def _build_numpy_view(self, np_array, idx, outputs):
        ctx = self._ctx
        np_storage = np_array.reshape(ctx.flat_output_storage_shapes[idx])
        v_shape = ctx.flat_output_view_shapes[idx]
        v_stride = ctx.flat_output_view_strides[idx]
        v_offset = ctx.flat_output_view_offsets[idx]
        try:
            outputs[idx] = make_storage_view(np_storage, v_shape, v_stride, v_offset)
            ret_v_shape = tuple(ctx.prof_result.output_view_shapes[idx])
            if ret_v_shape != tuple(v_shape):
                outputs[idx] = outputs[idx].reshape(ret_v_shape)
        except Exception:
            logging.error(
                f"numpy.as_strided failed. storage_shape={np_storage.shape} "
                f"view_shape={v_shape}, view_stride={v_stride}, view_offset={v_offset}"
            )
            raise

    def _output_bytes_to_tensors(self):
        outputs = self._ctx.prof_result.output_bytes
        dtypes = resolve_numpy_dtypes(self._ctx.flat_output_dtypes)
        use_torch = self._ctx.is_torch_dtype_support()
        for idx, o in enumerate(outputs):
            if isinstance(o, str) or o is None:
                continue
            if (
                not isinstance(o, (bytes, bytearray))
                and not hasattr(o, "raw")
                and not (hasattr(o, "__class__") and "c_char" in o.__class__.__name__)
            ):
                continue
            dtype = get(dtypes, idx)
            np_array = self._decode_output_bytes(o, dtype)
            if use_torch:
                self._build_torch_view(np_array, idx, dtype, outputs)
            else:
                self._build_numpy_view(np_array, idx, outputs)

    def compare(self, third_parties=None):
        if self._standards is not None:
            standards = self._standards
        else:
            compare_method = get_global_storage().compare_method
            output_dtypes = resolve_numpy_dtypes(self._ctx.flat_output_dtypes)
            standards = resolve_tolerance(
                self._tolerance,
                self._ctx.flat_precision_tolerances,
                self._ctx.flat_absolute_precision,
                output_dtypes,
                compare_method,
            )
        effective_third_parties = self._third_parties if third_parties is None else third_parties
        self._output_bytes_to_tensors()
        try:
            logging_data = "\n"
            logging_data += f"Comparing {self._ctx.testcase_name} with golden\n"
            outputs = self._ctx.prof_result.output_bytes
            goldens = self._ctx.golden_tensors
            hooks = CompareHooks(
                output_dtypes=resolve_numpy_dtypes(self._ctx.flat_output_dtypes),
                third_parties=effective_third_parties,
                pre_compare=self._pre_compare,
                custom_compare=self._custom_compare,
            )
            precision, _logging_data, passed, metrics = compare_with_hooks(
                self._ctx, outputs, goldens, standards, hooks
            )
            logging_data += _logging_data
            logging.debugc(logging_data)
            # None=SKIP：golden Disable，未比对
            passed = "SKIP" if passed is None else "PASS" if passed else "FAIL"
        except Exception:
            logging.exception("Comparison failed")
            return ApiComparisonResult("COMPARE_FAILURE")
        return ApiComparisonResult(None).set(precision, passed, metrics)
