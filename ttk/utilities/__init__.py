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
ttk.utilities 懒加载门面：模块级 __getattr__ 按名转发，
import ttk.utilities 零副作用。

新增公开名的流程：在所属子模块声明（有 __all__ 加 __all__），并同步本表
（scripts/check_conventions.py 对照公开面时按 __all__ 校验）。
"""

import importlib

VERSION = "3.0.0"
FAQ = ""

# 名 → 子模块映射（懒加载门面，__getattr__ 按名转发）。
# 注意：DumpConfig/DumpLevel 与全局开关族（SWITCHES/MODE 等）不经本门面导出
# ——前者直连 ttk.core_modules.common.dump，
# 后者升格为顶层模块 ttk/switches.py（运行时状态域，与静态 config/ 平级）。
_EXPORTS = {
    "ACL_DTYPE_TO_ID": ".dtypes",
    "ACL_ID_TO_DTYPE": ".dtypes",
    "BLOCK_SIZE": ".format",
    "DTYPE_MAP": ".dtypes",
    "DTYPE_PROMOTE_MAP": ".dtypes",
    "DTYPE_WIDTH_MAP": ".dtypes",
    "FORMAT_CODES": ".format",
    "MX_ELEMENT_DTYPES": ".mx_quant",
    "PAD_C0_MAPPING": ".format",
    "NZ_FORMATS": ".format",
    "ONNX_DTYPE_MAP": ".dtypes",
    "ONNX_ID_TO_DTYPE": ".dtypes",
    "RandomData": ".data",
    "Singleton": ".singleton",
    "acl_to_torch_dtype": ".dtypes",
    "align": ".math",
    "align_factor": ".format",
    "map_dtypes": ".dtypes",
    "camel_to_snake": ".string",
    "ceil_div": ".math",
    "cgroup_mem_current": ".memory",
    "cgroup_mem_limit": ".memory",
    "cgroup_memory_snapshot": ".memory",
    "limit_cann_compile_jobs": ".memory",
    "cpu_count": ".proc",
    "deep_flatten": ".container",
    "delete_files": ".file_utils",
    "resolve_c0": ".format",
    "dtype_to_name": ".dtypes",
    "read_csv_cols": ".file_utils",
    "extract_plog_errors": ".plog",
    "from_nc1hwc0": ".format",
    "constant_array": ".data",
    "flatten_groups": ".container",
    "FORMAT_TRANSFORMS": ".format",
    "from_fractal_z_3d": ".format",
    "transpose_axes": ".format",
    "get": ".container",
    "get_dtype_overflow_boundary": ".dtypes",
    "get_dtype_range": ".dtypes",
    "get_dtype_width": ".dtypes",
    "file_md5": ".file_utils",
    "get_loaded_so_path": ".proc",
    "infer_list_distribution_from_nesting": ".container",
    "is_4bit_dtype": ".dtypes",
    "is_float_dtype": ".dtypes",
    "is_main_process": ".proc",
    "is_nchw_like": ".format",
    "is_ndhwc_like": ".format",
    "is_negative_zero": ".math",
    "is_onnx_native_dtype": ".dtypes",
    "is_positive_zero": ".math",
    "is_shape": ".shape",
    "is_tf_native_dtype": ".dtypes",
    "is_torch_native_dtype": ".dtypes",
    "is_transformable": ".format",
    "kernel_debug_compile_enabled": ".proc",
    "lcm": ".math",
    "load_array": ".file_utils",
    "longest_match": ".string",
    "msdebug_runtime_injection_enabled": ".proc",
    "mx_quantize": ".mx_quant",
    "nd_shape_to_5hd": ".format",
    "nd_shape_to_nz": ".format",
    "nd_to_fractal_nz": ".format",
    "nd_to_fractal_z": ".format",
    "nest_by_distribution": ".container",
    "normalize_dtype": ".dtypes",
    "normalize_from_tf_dtype": ".dtypes",
    "normalize_to_tf_dtype": ".dtypes",
    "numpy_as_strided_safe": ".views",
    "numpy_to_torch_tensor": ".dtypes",
    "from_fractal_nz": ".format",
    "pack_4bits": ".dtypes",
    "process_kernel_string": ".string",
    "promote_scalar_shapes": ".shape",
    "read_file": ".file_utils",
    "render_frameless_table": ".string",
    "render_table": ".string",
    "resolve_numpy_dtype": ".dtypes",
    "resolve_numpy_dtypes": ".dtypes",
    "set_process_name": ".proc",
    "set_thread_name": ".proc",
    "shape_numel": ".shape",
    "shape_span": ".shape",
    "shape_stride": ".shape",
    "from_ndc1hwc0": ".format",
    "signal_registered": ".proc",
    "extract_zip": ".file_utils",
    "split_distribution": ".container",
    "str_to_onnx_dtype": ".dtypes",
    "str_to_tf_dtype": ".dtypes",
    "str_to_torch_dtype": ".dtypes",
    "to_nc1hwc0": ".format",
    "to_ndc1hwc0": ".format",
    "to_fractal_z": ".format",
    "to_fractal_z_3d": ".format",
    "to_fractal_z_c04": ".format",
    "torch_to_numpy_tensor": ".dtypes",
    "transform": ".format",
    "unpack_4bits": ".dtypes",
}

_SUBMODULES = (
    ".classes",
    ".container",
    ".data",
    ".dtypes",
    ".file_utils",
    ".format",
    ".func_dispatch",
    ".math",
    ".memory",
    ".mx_quant",
    ".plog",
    ".proc",
    ".singleton",
    ".string",
    ".table_reader",
    ".views",
    ".shape",
)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__package__}.{name[1:]}")
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__package__}{module}")
    return getattr(mod, name)


def __dir__():
    return sorted(_EXPORTS) + [n[1:] for n in _SUBMODULES] + ["VERSION", "FAQ"]
