#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Interfaces to access gen_simplifiedKey_func
Functions above are registered in libregister.so from liboptiling.so/libopmaster_rt2.0.so
"""

__all__ = ["customize_gen_simplified_key"]


# Standard Packages
import ctypes
import json
import logging
import math
import os
from typing import Union

# Third-Party Packages
from ...common.device import get_ascend_scene_info, get_op_impl_paths


def _load_custom_vendor_tiling(opp_impl_path):
    """Load custom/vendor tiling SO: flat path, only liboptiling.so + TbeLoadSoAndSaveToRegistry.

    Follows official CANN op_tiling.py: custom/vendor SO is at
    {impl}/ai_core/tbe/op_tiling/liboptiling.so (no lib/{os}/{arch} subdirectory).
    OSError is silently ignored (SO may not exist).
    """
    so_path = os.path.join(opp_impl_path, "ai_core", "tbe", "op_tiling", "liboptiling.so")
    if not os.path.isfile(so_path):
        return
    try:
        dll = ctypes.CDLL(so_path)
        dll.TbeLoadSoAndSaveToRegistry(str(so_path).encode("utf_8"))
    except OSError:
        pass


def _load_builtin_tiling(opp_impl_path):
    """Load builtin tiling SOs with full if-else chain.

    Follows official CANN op_tiling.py:
      1. liboptiling.so (1.0 registration, CDLL only)
      2. libopmaster_rt.so → libopmaster_rt2.0.so → libcust_opmaster_rt2.0.so → op_host/*.so
         (2.0 registration via TbeLoadSoAndSaveToRegistry, mutually exclusive)
    """
    scene_os, scene_arch = get_ascend_scene_info()

    op_tiling_so_dir = os.path.join(opp_impl_path, "ai_core", "tbe", "op_tiling")

    if scene_os:
        op_tiling_so_dir = os.path.join(op_tiling_so_dir, "lib")
    op_tiling_so_dir = os.path.join(op_tiling_so_dir, scene_os, scene_arch)

    op_tiling_so = os.path.join(op_tiling_so_dir, "liboptiling.so")
    master_rt_so = os.path.join(op_tiling_so_dir, "libopmaster_rt.so")
    master_rt2_so = os.path.join(op_tiling_so_dir, "libopmaster_rt2.0.so")
    cust_master_rt2_so = os.path.join(op_tiling_so_dir, "libcust_opmaster_rt2.0.so")
    op_host_dir = os.path.join(opp_impl_path, "ai_core", "tbe", "op_host", "lib", scene_os, scene_arch)

    # 1. builtin optiling 1.0 registration
    if os.path.isfile(op_tiling_so):
        ctypes.CDLL(op_tiling_so)

    # 2. builtin optiling 2.0 registration (mutually exclusive)
    if os.path.isfile(master_rt_so):
        dll = ctypes.CDLL(master_rt_so)
        dll.TbeLoadSoAndSaveToRegistry(str(master_rt_so).encode("utf_8"))
    elif os.path.isfile(master_rt2_so):
        dll = ctypes.CDLL(master_rt2_so)
        dll.TbeLoadSoAndSaveToRegistry(str(master_rt2_so).encode("utf_8"))
    elif os.path.isfile(cust_master_rt2_so):
        dll = ctypes.CDLL(cust_master_rt2_so)
        dll.TbeLoadSoAndSaveToRegistry(str(cust_master_rt2_so).encode("utf_8"))
    elif os.path.isdir(op_host_dir):
        non_legacy = []
        legacy = []
        for root, _, files in os.walk(op_host_dir, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                if os.path.isfile(file_path) and file_path.endswith(".so"):
                    if "legacy" in f.lower():
                        legacy.append(file_path)
                    else:
                        non_legacy.append(file_path)
        for fp in non_legacy:
            dll = ctypes.CDLL(fp)
            dll.TbeLoadSoAndSaveToRegistry(str(fp).encode("utf_8"))
        for fp in legacy:
            dll = ctypes.CDLL(fp)
            dll.TbeLoadSoAndSaveToRegistry(str(fp).encode("utf_8"))


def load_op_registries():
    # Load in priority order: custom → vendors → built-in (first-registered-wins)
    for source in ("custom", "vendor", "builtin"):
        try:
            for impl_path in get_op_impl_paths(source):
                if os.path.isdir(impl_path):
                    logging.info(f"Loading {source} tiling registries from: {impl_path}")
                    if source in ("custom", "vendor"):
                        _load_custom_vendor_tiling(impl_path)
                    else:
                        _load_builtin_tiling(impl_path)
        except BaseException as e:
            logging.critical(f"TilingInterface init failed for {source}: {e}")
            raise


_registries_loaded = False
_registry_accessor = None
_handle_cache = {}


def _ensure_loaded():
    global _registries_loaded, _registry_accessor
    if _registries_loaded:
        return
    load_op_registries()
    from ..compile.cext_loader import load_cext

    _registry_accessor = load_cext("libttk_op_registry_accessor.so", "op_registry_accessor")
    if not hasattr(_registry_accessor, "FindGenSimplifiedKeyFuncs"):
        raise RuntimeError("Interface [FindGenSimplifiedKeyFuncs] is not found.")
    if not hasattr(_registry_accessor, "InvokeGenSimplifiedKey"):
        raise RuntimeError("Interface [InvokeGenSimplifiedKey] is not found.")
    _registry_accessor.FindGenSimplifiedKeyFuncs.restype = ctypes.c_int
    _registry_accessor.FindGenSimplifiedKeyFuncs.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    _registry_accessor.InvokeGenSimplifiedKey.restype = ctypes.c_int
    _registry_accessor.InvokeGenSimplifiedKey.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    _registries_loaded = True


def _find_funcs(op_type: str):
    """Look up gen_simplifiedkey funcs for op_type, cached per op_type."""
    cached = _handle_cache.get(op_type)
    if cached is not None:
        return cached
    _ensure_loaded()
    handle = ctypes.c_void_p()
    ret = _registry_accessor.FindGenSimplifiedKeyFuncs(op_type.encode("utf_8"), ctypes.byref(handle))
    if ret != 0:
        _handle_cache[op_type] = False
        return False
    _handle_cache[op_type] = handle
    return handle


def customize_gen_simplified_key(simplified_key, op_type, inputs, outputs, attrs=None) -> str:
    """invoke customize simplified key generator.
    simplified_key = f"{op_type}/d={deterministic},p={impl_mode}/xxxxx"
    """
    handle = _find_funcs(op_type)
    if not handle:
        raise RuntimeError(f"Customized simplified key generator function is not found for op [{op_type}].")

    deterministic = int(simplified_key.split("/")[1].split(",")[0].split("=")[1])
    extra_params = {"op_name": op_type, "deterministic": deterministic}
    _inputs_pre_process(inputs)
    _attrs_pre_process(attrs)
    op_type_c = op_type.encode("utf_8")
    inputs_c = json.dumps(inputs).encode("utf_8")
    outputs_c = json.dumps(outputs).encode("utf_8")
    extra_params_c = json.dumps(extra_params).encode("utf_8")
    attrs_c = None if not attrs else json.dumps(attrs).encode("utf_8")

    res_buf = ctypes.create_string_buffer(simplified_key.encode("utf_8"), 256)

    invoke_fn = _registry_accessor.InvokeGenSimplifiedKey
    ret = invoke_fn(handle, op_type_c, inputs_c, outputs_c, attrs_c, extra_params_c, res_buf)
    if ret != 0:
        msg = _parse_c_return_code(ret)
        raise RuntimeError(
            f"invoke customized simplified key generator for op [{op_type}] failed: {msg}. "
            f"Please check `plog` for more details."
        )
    return res_buf.value.decode("utf-8")


def _parse_c_return_code(ret):
    CODE_MAP = {
        1: "Customized simplified key generator function is not found",
        2: "Parse input/output/attr/extra_info failed",
        3: "Invoke customized simplified key generator function failed",
    }
    return CODE_MAP.get(ret, f"Unknown return code: [{ret}]")


def _inputs_pre_process(inputs: Union[list, tuple]):
    if not isinstance(inputs, (list, tuple)):
        return
    for ipt in inputs:
        if not isinstance(ipt, dict):
            continue
        const_value = ipt.get("const_value")
        if not isinstance(const_value, (list, tuple)):
            continue
        const_value_list = list(const_value)
        const_value_null_desc = _gen_null_desc(const_value_list)
        if const_value_null_desc is not None:
            ipt["const_value"] = const_value_list
            ipt["const_value_null_desc"] = const_value_null_desc


def _attrs_pre_process(attrs):
    if not isinstance(attrs, (list, tuple)):
        return
    for single_attr in attrs:
        if not isinstance(single_attr, dict):
            continue
        attr_dtype = single_attr.get("dtype")
        if attr_dtype not in (
            "float",
            "float32",
            "float64",
            "double",
            "list_float",
            "list_float32",
            "list_float64",
            "list_double",
        ):
            continue
        attr_value = single_attr.get("value")
        if attr_value is None:
            continue
        is_single_element = False
        if not isinstance(attr_value, (list, tuple)):
            is_single_element = True
            attr_value = [attr_value]
        attr_value_list = list(attr_value)
        attr_null_desc = _gen_null_desc(attr_value_list)
        if attr_null_desc is not None:
            if is_single_element:
                single_attr["value_null_desc"] = attr_null_desc[0]
                single_attr["value"] = attr_value_list[0]
            else:
                single_attr["value_null_desc"] = attr_null_desc
                single_attr["value"] = attr_value_list


def _gen_null_desc(value_list):
    if not isinstance(value_list, list):
        return None
    value_null_desc = []
    is_exist_null = False
    for idx, value in enumerate(value_list):
        if not isinstance(value, float):
            continue
        if value == float("inf"):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("inf")
        elif value == float("-inf"):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("-inf")
        elif math.isnan(value):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("nan")
        else:
            value_null_desc.append(None)

    return value_null_desc if is_exist_null else None
