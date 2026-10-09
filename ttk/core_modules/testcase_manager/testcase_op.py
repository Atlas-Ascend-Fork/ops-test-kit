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
OP Testcase Structure
"""

__all__ = ["TestcaseOp"]


# Standard Packages
import ast
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple, Union

import numpy

# Third-Party Packages
from ...utilities import (
    BaseCompilationResult,
    BinaryCompilationResult,
    ConstCompilationResult,
    DynamicCompilationResult,
    DynamicOpTilingResult,
    eliminate_scalar_shapes,
    get,
    get_dtype_width,
    get_global_storage,
    input_apply_as_list,
    parse_dtype,
    shape_product,
)
from ...utilities.container_utils import pickup_by_names
from ..infershape.infershape import shape_inference
from .field_types import FIELD_TYPES
from .testcase_base import TestcaseBase

if TYPE_CHECKING:
    import ttk


class TestcaseOp(TestcaseBase):
    """
    Structure for Op Profiling
    """

    __slots__ = (
        # === testcase valid configurations === #
        "op_name",
        # static input tensors shape & format & dtype
        "input_shapes",
        "input_formats",
        "input_dtypes",
        # static output tensors shape & format & dtype
        "output_shapes",
        # static input tensors original shape & format
        "input_ori_shapes",
        "input_ori_formats",
        # static output tensors original shape & format
        "output_ori_shapes",
        "output_ori_formats",
        "output_formats",
        "output_dtypes",
        # special
        "output_inplace_indexes",
        "output_shape_unknown_indexes",
        # number of metadata outputs appended to the declared output fields by
        # append_output_metadata (read back by golden regeneration — GEIR)
        "_output_meta_count",
        # specify dump file prefix
        "dump_file_prefix",
        # manual configurations
        "manual_input_binaries",
        "manual_golden_binaries",
        "manual_xpu_binaries",
        # operator attributes
        "attributes",
        "attributes1",
        "attributes2",
        "attributes3",
        "attributes4",
        "attributes5",
        "attributes6",
        "attributes7",
        "attributes8",
        "attributes9",
        # === Runtime parameters below === #
        # Compilation result
        "dyn_compile_result",
        "cst_compile_result",
        "bin_compile_result",
        # Profiling result
        "dyn_prof_result",
        "cst_prof_result",
        "bin_prof_result",
        # impl.dynamic.xxx implement function parameters
        "dyn_func_params",
        "input_arrays",
        "original_input_arrays",
        # 免上传：{内容指纹: 生成配方}，供三方腿免传输入（未开启时为 None）
        "input_recipes",
        "golden_arrays",
        "output_arrays",
        "dyn_workspace_arrays",
        "cst_workspace_arrays",
        "bin_workspace_arrays",
        "dyn_tuple_tiling_data",
        "dyn_str_tiling_data",
        "dyn_tiling_data_bytes",
        "bin_tuple_tiling_data",
        "bin_str_tiling_data",
        "bin_tiling_data_bytes",
        "compile_done",
        "compile_done",
        "torch_func",
        "tf_func",
        "kb_pid",
        "core_type",
        # XPU third-party
        "xpu_results",
        "golden_mode_override",
        # private
        "_computed_list_distribution",
        "_input_distribution",
        "_output_distribution",
        "_flat_input_shapes",
        "_flat_input_dtypes",
        "_flat_input_formats",
        "_flat_input_ori_shapes",
        "_flat_input_ori_formats",
        "_flat_output_shapes",
        "_flat_output_dtypes",
        "_flat_output_formats",
        "_flat_output_ori_shapes",
        "_flat_output_ori_formats",
        "_flat_input_data_ranges",
        "_flat_precision_tolerances",
        "_flat_absolute_precision",
        "_spec_tensors_cache",
        "_spec_attrs_cache",
        "_flat_input_arrays",
        "_flat_manual_input_binaries",
        "_flat_manual_golden_binaries",
        "_flat_manual_xpu_binaries",
        "_dyn_clear_atomic",
        "_cst_clear_atomic",
        "_bin_clear_atomic",
        "_dyn_tensor_dict",
        "_bin_tensor_dict",
        "_dyn_input_ranges_cache",
        "_dyn_output_ranges_cache",
        "xpu_results",
    )
    identity_headers: Dict[str, tuple] = {
        **TestcaseBase.identity_headers,
        "op_name": (FIELD_TYPES.STRING, None),  # Required
    }
    non_platform_static_property_headers: Dict[str, tuple] = {
        "input_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
        "input_ori_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, ("input_shapes",)),
        "output_ori_shapes": (FIELD_TYPES.SHAPELIKE_STC_EX_NESTED, ("output_shapes",), None),
        "input_ori_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, ("input_formats",), ("ND",)),
        "output_ori_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, ("output_formats",), ("ND",)),
        "attributes": (FIELD_TYPES.DICT, None, {}),
        **{f"attributes{i}": (FIELD_TYPES.DICT, None, {}) for i in range(1, 10)},
    }
    static_property_headers: Dict[str, tuple] = {
        "input_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
        "output_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None),
        "output_shapes": (FIELD_TYPES.SHAPELIKE_STC_EX_NESTED, None, None),
        "input_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ("ND",)),
        "output_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, ("output_ori_formats",), ("ND",)),
    }
    special_property_headers: Dict[str, tuple] = {
        **TestcaseBase.special_property_headers,
        "output_inplace_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "output_shape_unknown_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
    }
    property_headers: Dict[str, tuple] = {
        **non_platform_static_property_headers,
        **static_property_headers,
        **special_property_headers,
    }
    option_headers: Dict[str, tuple] = {
        **TestcaseBase.option_headers,
        # Manually controlled property
        "dump_file_prefix": (FIELD_TYPES.STRING, None, None),
        "manual_input_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
        "manual_golden_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
        "manual_xpu_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
    }
    complete_headers: Dict[str, tuple] = {**identity_headers, **property_headers, **option_headers}
    force_clear_atomic_ops = ("ones_like",)

    def __init__(self):
        super().__init__()
        self.op_name: Optional[str] = None
        self._init_stc_fields()
        self._init_manual_and_runtime_fields()
        self._init_compile_and_tiling_fields()
        self._init_flat_and_property_fields()

    @property
    def cst_clear_atomic(self):
        if self._cst_clear_atomic is not None:
            return self._cst_clear_atomic
        if (
            self.cst_compile_result
            and self.cst_compile_result.kernel_json_info
            and self.cst_compile_result.kernel_json_info.clear_atomic
        ):
            return True
        return self.op_name in self.force_clear_atomic_ops

    @property
    def dyn_clear_atomic(self):
        if self._dyn_clear_atomic is not None:
            return self._dyn_clear_atomic
        if (
            self.dyn_compile_result
            and self.dyn_compile_result.kernel_json_info
            and self.dyn_compile_result.kernel_json_info.clear_atomic
        ):
            return True
        return self.op_name in self.force_clear_atomic_ops

    @property
    def bin_clear_atomic(self):
        if self._bin_clear_atomic is not None:
            return self._bin_clear_atomic
        if (
            self.bin_compile_result
            and self.bin_compile_result.kernel_json_info
            and self.bin_compile_result.kernel_json_info.clear_atomic
        ):
            return True
        return self.op_name in self.force_clear_atomic_ops

    @property
    def const_input_indexes(self):
        """Nested indexes of const inputs, derived from op_info valueDepend."""
        from ..operator.op_info_keeper import OpInfoKeeper

        op_info = OpInfoKeeper().info_of(self.op_name)
        if not op_info:
            return ()
        return tuple(
            idx for idx, ipt in enumerate(op_info["inputs"]) if ipt.get("valueDepend") in ("optional", "required")
        )

    @property
    def spec_tensors(self) -> dict:
        return getattr(self, "_spec_tensors_cache", {})

    @property
    def spec_attrs(self) -> dict:
        return getattr(self, "_spec_attrs_cache", {})

    def _cache_split_attributes(self):
        from ..operator.op_info_keeper import OpInfoKeeper

        op_info = OpInfoKeeper().info_of(self.op_name)
        input_names = [ipt["name"] for ipt in op_info["inputs"]] if op_info else []
        self._spec_tensors_cache = pickup_by_names(self.attributes or {}, input_names)
        self._spec_attrs_cache = {k: v for k, v in (self.attributes or {}).items() if k not in self._spec_tensors_cache}

    @property
    def input_bytes(self):
        bytes_lst: list = []
        for idx, _ in enumerate(self.flat_input_shapes):
            if self.flat_input_shapes[idx] is None:
                continue
            try:
                bytes_lst.append(
                    shape_product(self.flat_input_shapes[idx]) * get_dtype_width(get(self.flat_input_dtypes, idx))
                )
            except Exception:
                bytes_lst.append(0)
        return sum(bytes_lst)

    @property
    def output_bytes(self):
        bytes_lst: list = []
        for idx, _ in enumerate(self.flat_output_shapes):
            if self.flat_output_shapes[idx] is None:
                continue
            try:
                bytes_lst.append(
                    shape_product(self.flat_output_shapes[idx]) * get_dtype_width(get(self.flat_output_dtypes, idx))
                )
            except Exception:
                bytes_lst.append(0)
        return sum(bytes_lst)

    @property
    def dyn_tensor_dict(self):
        # return nested tensor dict matching TensorList distribution.
        if self._dyn_tensor_dict is not None:
            return self._dyn_tensor_dict
        inputs = self._construct_op_xput_tensor_dict(
            self._eliminate_scalar_shapes_nested(self.dyn_inputs),
            self.dyn_input_dtypes,
            self.dyn_input_formats,
            self.dyn_input_ranges,  # nested
            self._eliminate_scalar_shapes_nested(self.dyn_ori_inputs),
            self.dyn_input_ori_formats,
        )
        outputs = self._construct_op_xput_tensor_dict(
            self.dyn_outputs,
            self.output_dtypes,
            self.output_formats,
            self.dyn_output_ranges,  # nested
            self.dyn_ori_outputs,
            self.output_ori_formats,
        )
        self._dyn_tensor_dict = (inputs, outputs)
        return self._dyn_tensor_dict

    @property
    def bin_tensor_dict(self):
        # return nested tensor dict with shapes replaced for binary matching.
        if self._bin_tensor_dict is not None:
            return self._bin_tensor_dict
        dyn_inputs = self._eliminate_scalar_shapes_nested(self.dyn_inputs)
        dyn_ranges = self.dyn_input_ranges
        dyn_ori_inputs = self._eliminate_scalar_shapes_nested(self.dyn_ori_inputs)
        const_indexes = self.const_input_indexes or ()
        bin_shapes, bin_ranges = [], []
        for pos, shape in enumerate(dyn_inputs):
            if shape is None:
                bin_shapes.append(None)
                bin_ranges.append(None)
            elif pos in const_indexes:
                bin_shapes.append(shape)
                bin_ranges.append(dyn_ranges[pos] if dyn_ranges else None)
            elif isinstance(shape, tuple) and shape and isinstance(shape[0], tuple):
                # TensorList: replace each sub-tensor shape
                bin_shapes.append(tuple((-2,) for _ in shape))
                bin_ranges.append(tuple(((1, None),) for _ in shape))
            else:
                bin_shapes.append((-2,))
                bin_ranges.append(((1, None),))
        inputs = self._construct_op_xput_tensor_dict(
            tuple(bin_shapes),
            self.dyn_input_dtypes,
            self.dyn_input_formats,
            tuple(bin_ranges),
            dyn_ori_inputs,
            self.dyn_input_ori_formats,
        )
        dyn_outputs = self.dyn_outputs
        dyn_ori_outputs = self.dyn_ori_outputs
        bin_out_shapes, bin_out_ranges = [], []
        for _, shape in enumerate(dyn_outputs):
            if shape is None:
                bin_out_shapes.append(None)
                bin_out_ranges.append(None)
            elif isinstance(shape, tuple) and shape and isinstance(shape[0], tuple):
                # TensorList output
                bin_out_shapes.append(tuple((-2,) for _ in shape))
                bin_out_ranges.append(tuple(((1, None),) for _ in shape))
            else:
                bin_out_shapes.append((-2,))
                bin_out_ranges.append(((1, None),))
        outputs = self._construct_op_xput_tensor_dict(
            tuple(bin_out_shapes),
            self.output_dtypes,
            self.output_formats,
            tuple(bin_out_ranges),
            dyn_ori_outputs,
            self.output_ori_formats,
        )
        self._bin_tensor_dict = (inputs, outputs)
        return self._bin_tensor_dict

    @cst_clear_atomic.setter
    def cst_clear_atomic(self, value):
        self._cst_clear_atomic = value

    @dyn_clear_atomic.setter
    def dyn_clear_atomic(self, value):
        self._dyn_clear_atomic = value

    @bin_clear_atomic.setter
    def bin_clear_atomic(self, value):
        self._bin_clear_atomic = value

    @staticmethod
    def _dynamicize(shapes):
        return tuple(tuple(-1 if d > 0 else d for d in s) if s else s for s in shapes)

    @staticmethod
    def _dynamicize_nested(nested_shapes, distribution):
        """Dynamicize nested shapes, preserving TensorList structure."""
        if not nested_shapes:
            return nested_shapes
        result = []
        for i, shape in enumerate(nested_shapes):
            if shape is None:
                result.append(None)
            elif distribution and i < len(distribution) and distribution[i] > 0:
                # TensorList position: shape is tuple of shapes
                result.append(tuple(tuple(-1 if d > 0 else d for d in s) if s else s for s in shape))
            else:
                result.append(tuple(-1 if d > 0 else d for d in shape) if shape else shape)
        return tuple(result)

    @staticmethod
    def _recursively_parse_dtypes(field):
        if not field:
            return field
        result = []
        for item in field:
            if isinstance(item, (tuple, list)):
                result.append(tuple(parse_dtype(d) for d in item))
            else:
                result.append(parse_dtype(item))
        return tuple(result)

    @property
    def dyn_inputs(self):
        return self._dynamicize_nested(self.input_shapes, self.input_distribution)

    @property
    def dyn_input_dtypes(self):
        return self.input_dtypes

    @property
    def dyn_input_formats(self):
        return self.input_formats

    @property
    def dyn_outputs(self):
        return self._dynamicize_nested(self.output_shapes, self.output_distribution)

    @property
    def dyn_ori_inputs(self):
        return self._dynamicize_nested(self.input_ori_shapes, self.input_distribution)

    @property
    def dyn_input_ori_formats(self):
        return self.input_ori_formats

    @property
    def dyn_ori_outputs(self):
        return self._dynamicize_nested(self.output_ori_shapes, self.output_distribution)

    # ========== flat_dyn_* backward-compatible aliases ==========

    @property
    def flat_dyn_inputs(self):
        if not self.input_distribution:
            return self.dyn_inputs
        return self._flatten_by_distribution(self.dyn_inputs, self.input_distribution)

    @property
    def flat_input_arrays(self):
        if self._flat_input_arrays is not None:
            return self._flat_input_arrays
        if not self.input_arrays:
            return ()
        from ...utilities.container_utils import deep_flatten

        self._flat_input_arrays = tuple(deep_flatten(self.input_arrays))
        return self._flat_input_arrays

    @property
    def flat_dyn_input_dtypes(self):
        return self.flat_input_dtypes

    @property
    def flat_dyn_input_formats(self):
        return self.flat_input_formats

    @property
    def flat_dyn_input_ori_formats(self):
        return self.flat_input_ori_formats

    @property
    def flat_dyn_ori_inputs(self):
        if not self.input_distribution:
            return self.dyn_ori_inputs
        return self._flatten_by_distribution(self.dyn_ori_inputs, self.input_distribution)

    @property
    def flat_dyn_outputs(self):
        if not self.output_distribution:
            return self.dyn_outputs
        return self._flatten_by_distribution(self.dyn_outputs, self.output_distribution)

    @property
    def flat_dyn_ori_outputs(self):
        if not self.output_distribution:
            return self.dyn_ori_outputs
        return self._flatten_by_distribution(self.dyn_ori_outputs, self.output_distribution)

    @property
    def tensor_list_distribution(self):
        return self._computed_list_distribution or ()

    @property
    def dyn_input_ranges(self):
        if self._dyn_input_ranges_cache is None:
            flat_ranges = shape_inference(self.flat_dyn_inputs, (), "RANGE")
            self._dyn_input_ranges_cache = self._renest_ranges(flat_ranges, self.input_distribution)
        return self._dyn_input_ranges_cache

    @property
    def dyn_output_ranges(self):
        if self._dyn_output_ranges_cache is None:
            flat_ranges = shape_inference(self.flat_dyn_outputs, (), "RANGE")
            self._dyn_output_ranges_cache = self._renest_ranges(flat_ranges, self.output_distribution)
        return self._dyn_output_ranges_cache

    @property
    def flat_dyn_input_ranges(self):
        if not self.input_distribution:
            return self.dyn_input_ranges
        return self._flatten_ranges_by_distribution(self.dyn_input_ranges, self.input_distribution)

    @property
    def flat_dyn_output_ranges(self):
        if not self.output_distribution:
            return self.dyn_output_ranges
        return self._flatten_ranges_by_distribution(self.dyn_output_ranges, self.output_distribution)

    @staticmethod
    def supported_rerun_title() -> tuple:
        return "dyn_perf_us", "cst_perf_us", "bin_perf_us", "perf_status", "precision_status"

    @staticmethod
    def hash_cases_to_groups(testcases: Iterable["TestcaseOp"]) -> Dict[int, set]:
        testcase_group_dict = {}
        if not testcases:
            return testcase_group_dict
        for testcase in testcases:
            hash_index = testcase.get_compilation_hash()
            testcase_group = testcase_group_dict.setdefault(hash_index, set())
            if testcase not in testcase_group:
                testcase_group.add(testcase)
            else:
                logging.warning(f"Duplicate testcase: {testcase.testcase_name}")
        return testcase_group_dict

    def invalidate_flat_cache(self, *fields):
        """Invalidate flat cache(s) by field name(s).

        Example: invalidate_flat_cache("input_dtypes", "output_dtypes")
        sets self._flat_input_dtypes = None and self._flat_output_dtypes = None.
        """
        for field in fields:
            attr = f"_flat_{field}"
            if hasattr(self, attr):
                setattr(self, attr, None)

    def get_compilation_hash(self, is_binary: bool = False) -> int:
        """
        Get compilation related param hash
        :return: Optional[int] Hash
        """
        must_factors = ("op_name",)
        common_factors = (
            "dyn_input_dtypes",
            "output_dtypes",
            "dyn_input_formats",
            "dyn_input_ori_formats",
            "output_formats",
            "output_ori_formats",
            "tensor_list_distribution",
            "attributes",
        )
        binary_factors = (
            "dyn_inputs",
            "dyn_outputs",
            "dyn_input_ranges",
            "dyn_output_ranges",
            "dyn_ori_inputs",
            "dyn_ori_outputs",
        )
        factors = must_factors + common_factors + binary_factors
        if is_binary:
            factors = tuple(ele for ele in factors if ele not in binary_factors)
        compilation_params = [getattr(self, ele) for ele in factors]
        if is_binary:
            # add back dyn_inputs for binary
            compilation_params.append(str([-2 if x is not None else None for x in self.flat_dyn_inputs]))
        return hash(str(compilation_params))

    def ready_for_profile(self) -> bool:
        """Ready for profile"""
        return self.compile_done == 3

    def validate(self):
        super().validate()
        self.input_shapes = self.input_shapes or ()
        self._check_op_name()
        self._check_input_count()
        self._check_attributes()
        # Step 1: input distribution → flat_input_shapes available
        self._compute_input_distribution()
        self._parse_input_dtypes()
        self._parse_output_shapes()
        self._check_output_count()
        self._parse_output_ori_shapes()
        self._parse_output_dtypes()
        # Step 2: output distribution (output_shapes now resolved)
        self._compute_output_distribution()
        # Step 3: normalize all compressed fields
        self._normalize_compressed_fields()
        self._check_manual_binaries()
        self._check_manual_output_binaries()
        self._check_manual_xpu_binaries()
        self._check_manual_xpu_dependencies()
        self._stc_shape_size_check()
        self._set_case_core_type()
        self._auto_set_inplace_indexes()
        self._cache_split_attributes()

    def str_tiling_key(self, is_binary: bool = False):
        tiling_key = self.bin_compile_result.tiling_key if is_binary else self.dyn_compile_result.tiling_key
        return ("0x" + hex(tiling_key)[2:].zfill(16)) if isinstance(tiling_key, int) else tiling_key

    def compile_failed(self):
        return "SUCC" not in [getattr(self, f"{t}_compile_result").compile_result for t in ("dyn", "cst", "bin")]

    def compile_dynamic_op_success(self):
        return "SUCC" in [getattr(self, f"{t}_compile_result").compile_result for t in ("dyn", "cst", "bin")]

    def apply_compile_result(self, compile_result: BaseCompilationResult):
        if (
            isinstance(compile_result, (DynamicCompilationResult, BinaryCompilationResult, ConstCompilationResult))
            and self.dyn_func_params is None
        ):
            self.dyn_func_params = compile_result.func_params
        if isinstance(compile_result, DynamicCompilationResult):
            # Dyn & Bin
            if isinstance(compile_result, BinaryCompilationResult):
                self.bin_compile_result = compile_result
            else:
                self.dyn_compile_result = compile_result
            if compile_result.tiling_result is None:
                compile_result.tiling_result = DynamicOpTilingResult()
                compile_result.tiling_result.all_set(compile_result.compile_result)
        elif isinstance(compile_result, ConstCompilationResult):
            self.cst_compile_result = compile_result
        self.compile_done += 1

    def get_dyn_func_param_name(self, param_idx: int, dyn_func_params: Optional[Union[tuple, list]] = None):
        if dyn_func_params is None:
            dyn_func_params = self.dyn_func_params
        list_dist = self.tensor_list_distribution or []
        tmp = numpy.array(list_dist[:param_idx]) - 1
        return dyn_func_params[param_idx - sum(tmp[tmp > 0])]

    """ static private methods """

    @staticmethod
    def _construct_op_xput_tensor_dict(shapes, dtypes, formats, ranges, ori_shapes, ori_formats) -> tuple:
        """Build tensor dicts preserving TensorList nesting.

        shapes/dtypes/formats/ranges are nested: each element is either a single
        value (non-TensorList) or a tuple of values (TensorList).
        ori_shapes/ori_formats are flat (from external APIs).
        Returns nested structure matching shapes.
        """
        tensors = []
        for idx, shape in enumerate(shapes):
            if shape is None:
                tensors.append(None)
            elif isinstance(shape, tuple) and shape and isinstance(shape[0], tuple):
                # TensorList position: shape is tuple of shapes
                group = []
                for sub_idx, sub_shape in enumerate(shape):
                    if sub_shape is None:
                        group.append(None)
                    else:
                        dtype_val = dtypes[idx]
                        fmt_val = formats[idx]
                        range_val = ranges[idx]
                        ori_shape_val = get(ori_shapes, idx)
                        ori_fmt_val = ori_formats[idx]
                        if isinstance(dtype_val, (tuple, list)):
                            dtype_val = dtype_val[sub_idx]
                        if isinstance(fmt_val, (tuple, list)):
                            fmt_val = fmt_val[sub_idx]
                        if isinstance(range_val, (tuple, list)):
                            range_val = range_val[sub_idx]
                        if (
                            isinstance(ori_shape_val, (tuple, list))
                            and ori_shape_val
                            and isinstance(ori_shape_val[0], (tuple, list))
                        ):
                            ori_shape_val = ori_shape_val[sub_idx]
                        if isinstance(ori_fmt_val, (tuple, list)):
                            ori_fmt_val = ori_fmt_val[sub_idx]
                        group.append(
                            {
                                "shape": sub_shape,
                                "ori_shape": ori_shape_val,
                                "range": range_val,
                                "dtype": dtype_val,
                                "format": fmt_val,
                                "ori_format": ori_fmt_val,
                            }
                        )
                tensors.append(tuple(group))
            else:
                tensors.append(
                    {
                        "shape": shape,
                        "ori_shape": get(ori_shapes, idx),
                        "range": get(ranges, idx),
                        "dtype": get(dtypes, idx),
                        "format": get(formats, idx),
                        "ori_format": get(ori_formats, idx),
                    }
                )
        return tuple(tensors)

    @staticmethod
    def _eliminate_scalar_shapes_nested(shapes):
        """Remove scalar dimensions from nested shapes.

        Like eliminate_scalar_shapes but handles TensorList nesting where
        an element may be a tuple of shapes instead of a single shape.
        """
        if not shapes:
            return shapes
        result = []
        for shape in shapes:
            if shape is None:
                result.append(None)
            elif isinstance(shape, tuple) and shape and isinstance(shape[0], tuple):
                # TensorList position: shape is tuple of shapes
                result.append(tuple(eliminate_scalar_shapes((s,))[0] if s is not None else None for s in shape))
            else:
                # Single shape: wrap in tuple to use eliminate_scalar_shapes
                result.append(eliminate_scalar_shapes((shape,))[0])
        return tuple(result)

    @staticmethod
    def _do_shape_inference(inputs: tuple, outputs: str, args: dict) -> tuple:
        """Shape inference"""
        if None in inputs:
            raise ValueError("Automatic inference doesn't support None input")
        if outputs in ("ELEWISE",):
            return shape_inference(inputs, (1, None), outputs)
        if outputs in ("REDUCE",):
            if "axes" in args:
                axes = args["axes"]
            elif "axis" in args:
                axes = args["axis"]
            else:
                axes = None
            return shape_inference(inputs, (axes, 1, None), outputs)
        if "ELEWISE" in outputs:
            try:
                args = ast.literal_eval(outputs[7:])
            except Exception as e:
                raise ValueError(f"Unable to parse shape inference args from {outputs}") from e
            else:
                return shape_inference(inputs, args, "ELEWISE")
        elif "REDUCE" in outputs:
            try:
                _args = ast.literal_eval(outputs[6:])
                if "axes" in args:
                    args = (args["axes"], *_args[1:])
                elif "axis" in args:
                    args = (args["axis"], *_args[1:])
                else:
                    args = _args
            except Exception as e:
                raise ValueError(f"Unable to parse shape inference args from {outputs}") from e
            else:
                return shape_inference(inputs, args, "REDUCE")
        else:
            raise ValueError(f"Invalid shape inference value {outputs}")

    @staticmethod
    def _expand_indices(ipt_size: int, list_distribution: Union[list, tuple], inplace_indices: list):
        flatten_ipts = list(range(ipt_size))
        structure_ipts = input_apply_as_list(flatten_ipts, list_distribution)
        # flatten
        result = []
        for item in inplace_indices:
            if isinstance(item, int):
                ipt = structure_ipts[item]
                if isinstance(ipt, (list, tuple)):
                    result.extend(ipt)
                else:
                    result.append(ipt)
            else:  # None
                result.append(item)
        return result

    """ private methods """

    def _compute_input_distribution(self):
        """Compute input distribution from input_shapes nesting."""
        from ...utilities.container_utils import infer_list_distribution_from_nesting

        if self._is_nested_shapes(self.input_shapes):
            self._input_distribution = infer_list_distribution_from_nesting(self.input_shapes)
        elif self.input_shapes:
            self._input_distribution = (0,) * len(self.input_shapes)
        else:
            self._input_distribution = ()

    def _compute_output_distribution(self):
        """Compute output distribution from output_shapes (already resolved)."""
        from ...utilities.container_utils import infer_list_distribution_from_nesting

        if isinstance(self.output_shapes, tuple) and self._is_nested_shapes(self.output_shapes):
            self._output_distribution = infer_list_distribution_from_nesting(self.output_shapes)
        elif isinstance(self.output_shapes, tuple):
            self._output_distribution = (0,) * len(self.output_shapes)
        else:
            self._output_distribution = ()
        self._computed_list_distribution = (self._input_distribution or ()) + (self._output_distribution or ())

    def _flatten_nested_fields(self):
        """Compute both input and output distributions."""
        self._compute_input_distribution()
        self._compute_output_distribution()

    def _normalize_compressed_fields(self):
        """Expand compressed fields to match distribution nesting exactly."""
        input_dist = self.input_distribution
        output_dist = self.output_distribution
        if input_dist:
            for f in ("input_dtypes", "input_formats", "input_ori_formats"):
                self._normalize_scalar_field_by_dist(f, input_dist)
            for f in ("input_ori_shapes", "input_data_ranges"):
                self._normalize_range_field_by_dist(f, input_dist)
        if output_dist:
            for f in ("output_dtypes", "output_formats", "output_ori_formats", "absolute_precision"):
                self._normalize_scalar_field_by_dist(f, output_dist)
            for f in ("output_ori_shapes", "precision_tolerances"):
                self._normalize_range_field_by_dist(f, output_dist)

    def _check_manual_binaries(self):
        """Normalize, validate and reshape manual_input_binaries.

        On failure: sets is_valid=False + fail_reason, does not raise.
        """
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            self._normalize_manual_binaries_impl()
            self._validate_manual_binaries_impl()
            self._reshape_manual_binaries_impl()
        except Exception:
            self.is_valid = False
            self.fail_reason = "MANUAL_BINARIES_INVALID"
            logging.exception("manual_input_binaries validation failed")

    def _normalize_manual_binaries_impl(self):
        binaries = self.manual_input_binaries
        if not binaries:
            return
        if isinstance(binaries, str):
            self.manual_input_binaries = (binaries,)
            return
        if not isinstance(binaries, (tuple, list)):
            raise ValueError(f"Invalid manual_input_binaries: {binaries}. Expected str, tuple or list.")
        self.manual_input_binaries = self._normalize_binaries_recursive(binaries)

    def _normalize_binaries_recursive(self, container):
        result = []
        for item in container:
            if isinstance(item, (tuple, list)):
                result.append(self._normalize_binaries_recursive(item))
            elif isinstance(item, str) and item == "None":
                result.append(None)
            else:
                result.append(item)
        return tuple(result)

    def _validate_manual_binaries_impl(self):
        binaries = self.manual_input_binaries
        if not binaries:
            return

        has_tensor_list = any(d > 0 for d in self.input_distribution)
        is_nested = any(isinstance(b, (tuple, list)) for b in binaries)
        n_params = len(self.input_distribution)

        if is_nested:
            if not has_tensor_list:
                raise ValueError(
                    f"manual_input_binaries must be flat when no TensorList in inputs, got nested: {binaries}"
                )
            if len(binaries) != n_params:
                raise ValueError(
                    f"manual_input_binaries top-level count {len(binaries)} != input params count {n_params}"
                )
            offset = 0
            for i, dist_val in enumerate(self.input_distribution):
                count = max(dist_val, 1)
                group_inputs = self.flat_input_shapes[offset : offset + count]
                offset += count
                group = binaries[i]
                if dist_val > 0:
                    if not isinstance(group, (tuple, list)):
                        raise ValueError(
                            f"manual_input_binaries param {i}: "
                            f"expected tuple/list for TensorList, got {type(group).__name__}"
                        )
                    self._check_none_alignment(group, group_inputs, f"group {i}")
                else:
                    if isinstance(group, (tuple, list)):
                        raise ValueError(
                            f"manual_input_binaries param {i}: "
                            f"expected str/None for non-TensorList, got {type(group).__name__}"
                        )
                    self._check_none_alignment((group,), group_inputs, f"param {i}")
        else:
            self._check_none_alignment(binaries, self.flat_input_shapes, "flat")

    @staticmethod
    def _check_none_alignment(binaries, inputs, context_label, field_name="manual_input_binaries"):
        for i, inp in enumerate(inputs):
            binary = binaries[i] if i < len(binaries) else None
            if inp is not None and binary is None:
                raise ValueError(f"{field_name} {context_label}[{i}]: input is non-None but binary is missing/None")
            if inp is None and binary is not None:
                raise ValueError(f"{field_name} {context_label}[{i}]: input is None but binary is {binary!r}")
        if len(binaries) > len(inputs):
            raise ValueError(f"{field_name} {context_label}: {len(binaries)} entries > {len(inputs)} inputs")

    def _reshape_manual_binaries_impl(self):
        binaries = self.manual_input_binaries
        if not binaries:
            return
        flat_inputs = self.flat_input_shapes
        dist = self.input_distribution
        flat_bins = self._flatten_binaries(binaries, dist)
        if len(flat_bins) < len(flat_inputs):
            flat_bins = flat_bins + (None,) * (len(flat_inputs) - len(flat_bins))
        result = []
        offset = 0
        for d in dist:
            if d > 0:
                result.append(tuple(flat_bins[offset : offset + d]))
            else:
                result.append(flat_bins[offset])
            offset += max(d, 1)
        self.manual_input_binaries = tuple(result)
        self.manual_input_binaries = tuple(result)

    @staticmethod
    def _flatten_binaries(binaries, dist):
        """Flatten binaries (handle both flat and nested input)."""
        if not any(isinstance(b, (tuple, list)) for b in binaries):
            return tuple(binaries)
        result = []
        for i, d in enumerate(dist):
            if d > 0 and isinstance(binaries[i], (tuple, list)):
                result.extend(binaries[i])
            else:
                result.append(binaries[i])
        return tuple(result)

    # ---------- Output manual binaries ----------

    def _check_manual_output_binaries(self):
        self._check_manual_reference_binaries("manual_golden_binaries", "MANUAL_OUTPUT_BINARIES_INVALID")

    def _check_manual_xpu_binaries(self):
        self._check_manual_reference_binaries("manual_xpu_binaries", "MANUAL_XPU_BINARIES_INVALID")

    def _check_manual_xpu_dependencies(self):
        if not self.is_valid or not self.manual_xpu_binaries:
            return
        has_inputs = any(shape is not None for shape in (self.flat_input_shapes or ()))
        if has_inputs and not self.manual_input_binaries:
            self.is_valid = False
            self.fail_reason = "MANUAL_XPU_INPUT_REQUIRED"
            logging.error("manual_xpu_binaries requires manual_input_binaries")
            return
        if not self.manual_golden_binaries:
            self.is_valid = False
            self.fail_reason = "MANUAL_XPU_GOLDEN_REQUIRED"
            logging.error("manual_xpu_binaries requires manual_golden_binaries")

    def _check_manual_reference_binaries(self, field_name, failure_reason):
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            self._normalize_manual_reference_binaries_impl(field_name)
            self._validate_manual_reference_binaries_impl(field_name)
            self._reshape_manual_reference_binaries_impl(field_name)
        except Exception:
            self.is_valid = False
            self.fail_reason = failure_reason
            logging.exception("%s validation failed", field_name)

    def _normalize_manual_reference_binaries_impl(self, field_name):
        # binaries
        binaries = getattr(self, field_name)
        if binaries:
            if isinstance(binaries, str):
                setattr(self, field_name, (binaries,))
            elif isinstance(binaries, (tuple, list)):
                setattr(self, field_name, self._normalize_binaries_recursive(binaries))
            else:
                raise ValueError(f"Invalid {field_name}: {binaries}")

    def _validate_manual_reference_binaries_impl(self, field_name):
        dist = self.output_distribution or ()
        flat_outputs = self.flat_output_shapes

        # validate binaries
        binaries = getattr(self, field_name)
        if binaries:
            has_tensor_list = any(d > 0 for d in dist)
            is_nested = any(isinstance(b, (tuple, list)) for b in binaries)
            n_params = len(dist)

            if is_nested:
                if not has_tensor_list:
                    raise ValueError(f"{field_name} must be flat when no TensorList in outputs, got nested: {binaries}")
                if len(binaries) != n_params:
                    raise ValueError(f"{field_name} top-level count {len(binaries)} != output params count {n_params}")
                offset = 0
                for i, dist_val in enumerate(dist):
                    count = max(dist_val, 1)
                    group_outputs = flat_outputs[offset : offset + count]
                    offset += count
                    group = binaries[i]
                    if dist_val > 0:
                        if not isinstance(group, (tuple, list)):
                            raise ValueError(
                                f"{field_name} param {i}: "
                                f"expected tuple/list for TensorList, got {type(group).__name__}"
                            )
                        self._check_none_alignment(group, group_outputs, f"output group {i}", field_name)
                    else:
                        if isinstance(group, (tuple, list)):
                            raise ValueError(
                                f"{field_name} param {i}: "
                                f"expected str/None for non-TensorList, got {type(group).__name__}"
                            )
                        self._check_none_alignment((group,), group_outputs, f"output param {i}", field_name)
            else:
                self._check_none_alignment(binaries, flat_outputs, "output flat", field_name)

    def _reshape_manual_reference_binaries_impl(self, field_name):
        binaries = getattr(self, field_name)
        if not binaries:
            return
        dist = self.output_distribution or ()
        flat_outputs = self.flat_output_shapes
        flat_bins = self._flatten_binaries(binaries, dist)
        if len(flat_bins) < len(flat_outputs):
            flat_bins = flat_bins + (None,) * (len(flat_outputs) - len(flat_bins))
        result = []
        offset = 0
        for d in dist:
            if d > 0:
                result.append(tuple(flat_bins[offset : offset + d]))
            else:
                result.append(flat_bins[offset])
            offset += max(d, 1)
        setattr(self, field_name, tuple(result))

    def _normalize_manual_output_binaries_impl(self):
        self._normalize_manual_reference_binaries_impl("manual_golden_binaries")

    def _validate_manual_output_binaries_impl(self):
        self._validate_manual_reference_binaries_impl("manual_golden_binaries")

    def _reshape_manual_output_binaries_impl(self):
        self._reshape_manual_reference_binaries_impl("manual_golden_binaries")

    @property
    def input_distribution(self):
        if self._input_distribution is None:
            self._flatten_nested_fields()
        return self._input_distribution or ()

    @property
    def output_distribution(self):
        if self._output_distribution is None:
            self._flatten_nested_fields()
        return self._output_distribution or ()

    # ========== Flat properties ==========

    @property
    def flat_input_shapes(self):
        if self._flat_input_shapes is not None:
            return self._flat_input_shapes
        if not self.input_shapes or not self.input_distribution:
            self._flat_input_shapes = self.input_shapes
        else:
            from ...utilities.container_utils import flatten_nested_sequence

            self._flat_input_shapes = flatten_nested_sequence(self.input_shapes)
        return self._flat_input_shapes

    @property
    def flat_manual_input_binaries(self):
        if self._flat_manual_input_binaries is not None:
            return self._flat_manual_input_binaries
        if not self.manual_input_binaries or not self.input_distribution:
            self._flat_manual_input_binaries = self.manual_input_binaries
        else:
            from ...utilities.container_utils import deep_flatten

            self._flat_manual_input_binaries = deep_flatten(self.manual_input_binaries)
        return self._flat_manual_input_binaries

    @property
    def flat_manual_golden_binaries(self):
        if self._flat_manual_golden_binaries is not None:
            return self._flat_manual_golden_binaries
        if not self.manual_golden_binaries or not self.output_distribution:
            self._flat_manual_golden_binaries = self.manual_golden_binaries
        else:
            from ...utilities.container_utils import deep_flatten

            self._flat_manual_golden_binaries = deep_flatten(self.manual_golden_binaries)
        return self._flat_manual_golden_binaries

    @property
    def flat_manual_xpu_binaries(self):
        if self._flat_manual_xpu_binaries is not None:
            return self._flat_manual_xpu_binaries
        if not self.manual_xpu_binaries or not self.output_distribution:
            self._flat_manual_xpu_binaries = self.manual_xpu_binaries
        else:
            from ...utilities.container_utils import deep_flatten

            self._flat_manual_xpu_binaries = deep_flatten(self.manual_xpu_binaries)
        return self._flat_manual_xpu_binaries

    @property
    def flat_input_dtypes(self):
        if self._flat_input_dtypes is not None:
            return self._flat_input_dtypes
        if not self.input_dtypes:
            return self.input_dtypes
        dist = self.input_distribution
        if dist:
            self._flat_input_dtypes = self._flatten_by_distribution(self.input_dtypes, dist)
        else:
            self._flat_input_dtypes = self.input_dtypes
        return self._flat_input_dtypes

    @property
    def flat_input_formats(self):
        if self._flat_input_formats is not None:
            return self._flat_input_formats
        if not self.input_formats:
            return self.input_formats
        dist = self.input_distribution
        if dist:
            self._flat_input_formats = self._flatten_by_distribution(self.input_formats, dist)
        else:
            self._flat_input_formats = self.input_formats
        return self._flat_input_formats

    @property
    def flat_input_ori_shapes(self):
        if self._flat_input_ori_shapes is not None:
            return self._flat_input_ori_shapes
        if not self.input_ori_shapes:
            return self.input_ori_shapes
        dist = self.input_distribution
        if dist:
            self._flat_input_ori_shapes = self._flatten_by_distribution(self.input_ori_shapes, dist)
        else:
            self._flat_input_ori_shapes = self.input_ori_shapes
        return self._flat_input_ori_shapes

    @property
    def flat_input_ori_formats(self):
        if self._flat_input_ori_formats is not None:
            return self._flat_input_ori_formats
        if not self.input_ori_formats:
            return self.input_ori_formats
        dist = self.input_distribution
        if dist:
            self._flat_input_ori_formats = self._flatten_by_distribution(self.input_ori_formats, dist)
        else:
            self._flat_input_ori_formats = self.input_ori_formats
        return self._flat_input_ori_formats

    @property
    def flat_output_shapes(self):
        if self._flat_output_shapes is not None:
            return self._flat_output_shapes
        if not self.output_shapes or not self.output_distribution:
            self._flat_output_shapes = self.output_shapes
        else:
            from ...utilities.container_utils import flatten_nested_sequence

            self._flat_output_shapes = flatten_nested_sequence(self.output_shapes)
        return self._flat_output_shapes

    @property
    def flat_output_dtypes(self):
        if self._flat_output_dtypes is not None:
            return self._flat_output_dtypes
        if not self.output_dtypes:
            return self.output_dtypes
        dist = self.output_distribution
        if dist:
            self._flat_output_dtypes = self._flatten_by_distribution(self.output_dtypes, dist)
        else:
            self._flat_output_dtypes = self.output_dtypes
        return self._flat_output_dtypes

    def append_output_metadata(self, dtype: str, fmt: str, shape: tuple):
        """Append extra output metadata beyond declared outputs (e.g. shape-unknown tensor).

        Appends dtype, format, shape (with ori_shape=shape, ori_format=fmt) and
        extends output_distribution so all flat caches recompute naturally.
        """
        self.output_dtypes = tuple(list(self.output_dtypes) + [dtype])
        self.output_formats = tuple(list(self.output_formats) + [fmt])
        self.output_ori_formats = tuple(list(self.output_ori_formats) + [fmt])
        self.output_shapes = tuple(list(self.output_shapes) + [shape])
        self.output_ori_shapes = tuple(list(self.output_ori_shapes) + [shape])
        if self.precision_tolerances is not None:
            self.precision_tolerances = tuple(list(self.precision_tolerances) + [(0, 0)])
        if isinstance(self.absolute_precision, (tuple, list)):
            self.absolute_precision = tuple(list(self.absolute_precision) + [0.0])
        # Extend distribution to cover the extra element
        self._output_distribution = tuple(list(self.output_distribution) + [0])
        self._computed_list_distribution = (self._input_distribution or ()) + self._output_distribution
        # Invalidate all flat output caches — they recompute via updated distribution
        self.invalidate_flat_cache(
            "output_shapes",
            "output_dtypes",
            "output_formats",
            "output_ori_shapes",
            "output_ori_formats",
            "precision_tolerances",
            "absolute_precision",
        )
        self._output_meta_count += 1

    @property
    def flat_output_formats(self):
        if self._flat_output_formats is not None:
            return self._flat_output_formats
        if not self.output_formats:
            return self.output_formats
        dist = self.output_distribution
        if dist:
            self._flat_output_formats = self._flatten_by_distribution(self.output_formats, dist)
        else:
            self._flat_output_formats = self.output_formats
        return self._flat_output_formats

    @property
    def flat_output_ori_shapes(self):
        if self._flat_output_ori_shapes is not None:
            return self._flat_output_ori_shapes
        if not self.output_ori_shapes:
            return self.output_ori_shapes
        dist = self.output_distribution
        if dist:
            self._flat_output_ori_shapes = self._flatten_by_distribution(self.output_ori_shapes, dist)
        else:
            self._flat_output_ori_shapes = self.output_ori_shapes
        return self._flat_output_ori_shapes

    @property
    def flat_output_ori_formats(self):
        if self._flat_output_ori_formats is not None:
            return self._flat_output_ori_formats
        if not self.output_ori_formats:
            return self.output_ori_formats
        dist = self.output_distribution
        if dist:
            self._flat_output_ori_formats = self._flatten_by_distribution(self.output_ori_formats, dist)
        else:
            self._flat_output_ori_formats = self.output_ori_formats
        return self._flat_output_ori_formats

    @property
    def flat_input_data_ranges(self):
        if self._flat_input_data_ranges is not None:
            return self._flat_input_data_ranges
        if not self.input_data_ranges:
            return self.input_data_ranges
        dist = self.input_distribution
        if dist:
            self._flat_input_data_ranges = self._flatten_by_distribution(self.input_data_ranges, dist)
        else:
            self._flat_input_data_ranges = self.input_data_ranges
        return self._flat_input_data_ranges

    @property
    def flat_precision_tolerances(self):
        if self._flat_precision_tolerances is not None:
            return self._flat_precision_tolerances
        if not self.precision_tolerances:
            return self.precision_tolerances
        dist = self.output_distribution
        if dist:
            self._flat_precision_tolerances = self._flatten_by_distribution(self.precision_tolerances, dist)
        else:
            self._flat_precision_tolerances = self.precision_tolerances
        return self._flat_precision_tolerances

    @property
    def flat_absolute_precision(self):
        if self._flat_absolute_precision is not None:
            return self._flat_absolute_precision
        if not self.absolute_precision:
            return self.absolute_precision
        dist = self.output_distribution
        if dist and isinstance(self.absolute_precision, (tuple, list)):
            self._flat_absolute_precision = self._flatten_by_distribution(self.absolute_precision, dist)
        else:
            self._flat_absolute_precision = self.absolute_precision
        return self._flat_absolute_precision

    @staticmethod
    def _is_nested_shapes(shapes):
        if not shapes or not isinstance(shapes, tuple):
            return False
        for element in shapes:
            if (
                element is not None
                and isinstance(element, tuple)
                and len(element) > 0
                and isinstance(element[0], tuple)
            ):
                return True
        return False

    # ========== Normalize helpers ==========

    @staticmethod
    def _flatten_by_distribution(values, distribution):
        """Flatten a distribution-aligned field to one-value-per-tensor.

        Respects TensorList boundaries: a tuple entry at dist>0 position
        is either already expanded (len==num) or compressed (len==1).
        """
        result = []
        for vi, num in enumerate(distribution):
            val = values[vi]
            if num > 0:
                if isinstance(val, (tuple, list)):
                    if len(val) == num:
                        result.extend(val)
                    elif len(val) == 1:
                        result.extend([val[0]] * num)
                    else:
                        result.extend(val)
                else:
                    result.extend([val] * num)
            else:
                result.append(val)
        return tuple(result)

    @staticmethod
    def _renest_ranges(flat_ranges, distribution):
        """Re-nest flat ranges back to nested format matching distribution.

        Each flat element is a range tuple (min, max) or None — never split.
        """
        if not flat_ranges or not distribution:
            return flat_ranges
        result = []
        vi = 0
        for num in distribution:
            if num > 0:
                group = tuple(flat_ranges[vi + j] for j in range(num))
                result.append(group)
                vi += num
            else:
                result.append(flat_ranges[vi])
                vi += 1
        return tuple(result)

    @staticmethod
    def _flatten_ranges_by_distribution(nested_ranges, distribution):
        """Flatten nested ranges back to flat, one range per tensor."""
        if not nested_ranges or not distribution:
            return nested_ranges
        result = []
        for i, num in enumerate(distribution):
            val = nested_ranges[i]
            if num > 0 and isinstance(val, (tuple, list)):
                result.extend(val)
            else:
                result.append(val)
        return tuple(result)

    def _init_stc_fields(self):
        """Initialize STC (shape/dtype/format) and attributes fields."""
        # STC
        self.input_shapes: Optional[tuple] = None
        self.input_dtypes: Optional[tuple] = None
        self.input_formats: Optional[tuple] = None
        self.output_shapes: Optional[tuple] = None
        self.input_ori_shapes: Optional[tuple] = None
        self.input_ori_formats: Optional[tuple] = None
        self.output_dtypes = None
        self.output_formats = None
        self.output_ori_shapes = None
        self.output_ori_formats = None
        # Others
        self.attributes: Optional[dict] = None
        self.attributes1: Optional[dict] = None
        self.attributes2: Optional[dict] = None
        self.attributes3: Optional[dict] = None
        self.attributes4: Optional[dict] = None
        self.attributes5: Optional[dict] = None
        self.attributes6: Optional[dict] = None
        self.attributes7: Optional[dict] = None
        self.attributes8: Optional[dict] = None
        self.attributes9: Optional[dict] = None
        self.output_inplace_indexes = None  # flat input indexes after tensor_list expansion
        self.output_shape_unknown_indexes: Optional[tuple] = None
        self._output_meta_count: int = 0

    def _init_manual_and_runtime_fields(self):
        """Initialize manual-controlled parameters and runtime arrays."""
        # Manual controlled parameters
        self.dump_file_prefix = None
        self.manual_input_binaries: Optional[Tuple[str, ...]] = None
        self.manual_golden_binaries = None
        self.manual_xpu_binaries = None
        # End of testcase valid configurations
        self.input_arrays = None
        self.original_input_arrays = None
        self.input_recipes = None
        self.golden_arrays = None
        self.output_arrays = None
        self.dyn_workspace_arrays: Tuple[numpy.ndarray, ...] = ()
        self.cst_workspace_arrays: Tuple[numpy.ndarray, ...] = ()
        self.bin_workspace_arrays: Tuple[numpy.ndarray, ...] = ()

    def _init_compile_and_tiling_fields(self):
        """Initialize DYN runtime, compilation results, tiling and prof outputs."""
        # DYN_RUNTIME
        self.dyn_func_params = None
        # compilation results
        self.dyn_compile_result: Optional[DynamicCompilationResult] = None
        self.cst_compile_result: Optional[ConstCompilationResult] = None
        self.bin_compile_result: Optional[BinaryCompilationResult] = None
        # Tiling
        self.dyn_tuple_tiling_data = None
        self.dyn_str_tiling_data = None
        self.dyn_tiling_data_bytes = None
        self.bin_tuple_tiling_data = None
        self.bin_str_tiling_data = None
        self.bin_tiling_data_bytes = None
        # Outputs
        self.dyn_prof_result: Optional[ttk.core_modules.npu.RTSProfilingResult] = None
        self.cst_prof_result: Optional[ttk.core_modules.npu.RTSProfilingResult] = None
        self.bin_prof_result: Optional[ttk.core_modules.npu.RTSProfilingResult] = None

    def _init_flat_and_property_fields(self):
        """Initialize test runtime, flat-tensor caches and property overrides."""
        # Test Runtime Attributes
        self.compile_done: Optional[int] = 0
        self.torch_func: Optional[Callable] = None
        self.tf_func: Optional[Callable] = None
        self.kb_pid: Optional[int] = None
        self.core_type = "AiCore"
        self._computed_list_distribution = None
        self._input_distribution = None
        self._output_distribution = None
        self._flat_input_shapes = None
        self._flat_input_dtypes = None
        self._flat_input_formats = None
        self._flat_input_ori_shapes = None
        self._flat_input_ori_formats = None
        self._flat_output_shapes = None
        self._flat_output_dtypes = None
        self._flat_output_formats = None
        self._flat_output_ori_shapes = None
        self._flat_output_ori_formats = None
        self._flat_input_data_ranges = None
        self._flat_precision_tolerances = None
        self._flat_absolute_precision = None
        self._flat_input_arrays = None
        self._flat_manual_input_binaries = None
        self._flat_manual_golden_binaries = None
        self._flat_manual_xpu_binaries = None
        # property override
        self._dyn_clear_atomic = None
        self._cst_clear_atomic = None
        self._bin_clear_atomic = None
        self._dyn_tensor_dict: Optional[Tuple[Tuple[dict], Tuple[dict]]] = None
        self._bin_tensor_dict: Optional[Tuple[Tuple[dict], Tuple[dict]]] = None
        self._dyn_input_ranges_cache: Optional[tuple] = None
        self._dyn_output_ranges_cache: Optional[tuple] = None
        self.xpu_results: dict = {}

    # ========== actual_input_data_ranges override ==========

    @TestcaseBase.actual_input_data_ranges.setter
    def actual_input_data_ranges(self, value):
        if value is not None and self.input_distribution:
            self._actual_input_data_ranges = value
            self._normalize_range_field_by_dist("_actual_input_data_ranges", self.input_distribution, (None, None))
        else:
            self._actual_input_data_ranges = value

    def _shape_size_exceed(self):
        """Check shape product size"""
        shapes = eliminate_scalar_shapes(self.flat_input_shapes)
        shapes_out = eliminate_scalar_shapes(self.flat_output_shapes)
        total_shape_product_value = 0
        for idx, shape in enumerate(shapes):
            if shape is not None:
                shape_size = shape_product(shape)
                total_shape_product_value += shape_size * get_dtype_width(get(self.flat_input_dtypes, idx))

        for idx, shape in enumerate(shapes_out):
            if shape is not None:
                shape_size = shape_product(shape)
                total_shape_product_value += shape_size * get_dtype_width(get(self.flat_output_dtypes, idx))

        return total_shape_product_value > get_global_storage().DAVINCI_HBM_SIZE_LIMIT * 1024 * 1024 * 1024

    def _check_op_name(self):
        if not self.op_name:
            self.is_valid = False
            self.fail_reason = "OP_NAME_MISSING"
            logging.exception("op_name must be specified.")
            return
        from ..operator.op_info_keeper import OpInfoKeeper

        if not OpInfoKeeper().info_of(self.op_name):
            self.is_valid = False
            self.fail_reason = "SOC_NOT_SUPPORT"
            logging.warning(f"Operator [{self.op_name}] is not supported on current SOC.")

    def _check_input_count(self):
        if not self.is_valid:
            return
        from ..operator.op_info_keeper import OpInfoKeeper

        op_info = OpInfoKeeper().info_of(self.op_name)
        if not op_info:
            return
        expected_in = len(op_info["inputs"])
        actual_in = len(self.input_shapes) if self.input_shapes else 0
        if actual_in != expected_in:
            self.is_valid = False
            self.fail_reason = "INPUT_COUNT_MISMATCH"
            logging.error(
                f"Testcase [{self.testcase_name}] input count mismatch: "
                f"got {actual_in}, expected {expected_in} from op_info."
            )

    def _check_output_count(self):
        if not self.is_valid:
            return
        from ..operator.op_info_keeper import OpInfoKeeper

        op_info = OpInfoKeeper().info_of(self.op_name)
        if not op_info or not op_info.get("outputs"):
            return
        expected_out = len(op_info["outputs"])
        actual_out = len(self.output_shapes) if self.output_shapes else 0
        if actual_out != expected_out:
            self.is_valid = False
            self.fail_reason = "OUTPUT_COUNT_MISMATCH"
            logging.error(
                f"Testcase [{self.testcase_name}] output count mismatch: "
                f"got {actual_out}, expected {expected_out} from op_info."
            )

    def _check_attributes(self):
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            if get_global_storage().op_impl_mode is not None:
                # command line has higher priority
                self.attributes["impl_mode"] = get_global_storage().op_impl_mode
        except Exception:
            self.is_valid = False
            self.fail_reason = "OTHER_PARAMS_INVALID"
            logging.exception("Attributes parsing failed")

    def _parse_input_dtypes(self):
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            self.input_dtypes = self._recursively_parse_dtypes(self.input_dtypes)
        except Exception:
            self.is_valid = False
            self.fail_reason = "STC_INPUT_DTYPES_INVALID"
            logging.exception(f"Static input dtypes parse failed: {self.input_dtypes}")

    def _parse_output_shapes(self):
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            if isinstance(self.output_shapes, str):
                self.output_shapes = self._do_shape_inference(
                    self.flat_input_shapes, self.output_shapes, self.attributes
                )
            elif self.output_shapes is None:
                self.is_valid = False
                self.fail_reason = "STC_OUTPUT_NOT_SPECIFIED"
                logging.error(f"Static output shape not specified for {self.testcase_name}")
        except Exception:
            self.is_valid = False
            self.fail_reason = "STC_OUTPUT_INFERENCE_FAILED"
            logging.exception("Static output shape inference failed")

    def _parse_output_ori_shapes(self):
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            if isinstance(self.output_ori_shapes, str):
                self.output_ori_shapes = self._do_shape_inference(
                    self.flat_input_ori_shapes, self.output_ori_shapes, self.attributes
                )
            elif self.output_ori_shapes is None:
                self.is_valid = False
                self.fail_reason = "STC_ORI_OUTPUT_NOT_SPECIFIED"
                logging.error(f"Static original output shape not specified for {self.testcase_name}")
        except Exception:
            self.is_valid = False
            self.fail_reason = "STC_ORI_OUTPUT_INFERENCE_FAILED"
            logging.exception("Static original output shape inference failed")

    def _parse_output_dtypes(self):
        if not self.is_valid:
            return
        # noinspection PyBroadException
        try:
            self.output_dtypes = self._recursively_parse_dtypes(self.output_dtypes)
        except Exception:
            self.is_valid = False
            self.fail_reason = "OUTPUT_DTYPES_INVALID"
            logging.exception(f"Output dtypes parse failed: {self.output_dtypes}")

    def _stc_shape_size_check(self):
        if not self.is_valid:
            return
        if self._shape_size_exceed():
            logging.error(
                f"Testcase {self.testcase_name} is invalid because shape is out of bound: "
                f"limit={get_global_storage().DAVINCI_HBM_SIZE_LIMIT}GB, inputs: {self.input_shapes}, "
                f"dtype: {self.input_dtypes}, "
                f"outputs: {self.output_shapes}, "
                f"dtype: {self.output_dtypes}"
            )
            self.is_valid = False
            self.fail_reason = "STC_SHAPE_OUT_OF_BOUND"

    def _set_case_core_type(self):
        from ..operator.op_info_keeper import OpInfoKeeper

        op_info = OpInfoKeeper().info_of(self.op_name)
        if op_info is None:
            self.core_type = "AiCore"
        else:
            self.core_type = op_info["coreType.value"]

    def _auto_set_inplace_indexes(self):
        from ..operator.op_info_keeper import OpInfoKeeper

        op_info = OpInfoKeeper().info_of(self.op_name)
        if op_info and not self.output_inplace_indexes:
            inputs = [ipt["name"] for ipt in op_info["inputs"]]
            outputs = [opt["name"] for opt in op_info["outputs"]]
            inplace_indices = [inputs.index(o) if o in inputs else None for o in outputs]
            if all(i is None for i in inplace_indices):
                return
            for opt_idx, ipt_idx in enumerate(inplace_indices):
                if ipt_idx is None:
                    continue
                ipt_type = op_info["inputs"][ipt_idx].get("paramType")
                opt_type = op_info["outputs"][opt_idx].get("paramType")
                # mem_set_v2
                if (ipt_type == "dynamic" or opt_type == "dynamic") and ipt_type != opt_type:
                    logging.error(
                        f"Testcase {self.testcase_name} "
                        f"{opt_idx}th output inplace {ipt_idx}th input. "
                        f"The `paramType` mismatch: should be both `dynamic`. "
                        f"But got input={ipt_type}, output={opt_type}"
                    )
                    self.is_valid = False
                    self.fail_reason = "OUTPUT_INPLACE_INDEXES_ERROR"
                    break
            # expand for dynamic.
            flatten_inplace_indices = self._expand_indices(len(inputs), self.tensor_list_distribution, inplace_indices)
            self.output_inplace_indexes = tuple(flatten_inplace_indices)
