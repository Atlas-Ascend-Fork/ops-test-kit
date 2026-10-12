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
dtype utils
"""

import importlib
import logging
import re
from typing import Dict, Union

import numpy

# Canonical TTK dtype name -> actual attribute name on torch/torch_npu
# (torch spells float8_e8m0 as float8_e8m0fnu).
_TORCH_DTYPE_ALIAS = {
    "float8_e8m0": "float8_e8m0fnu",
}

# Numpy 4-bit dtypes (unpacked, 1 byte/element) -> torch packed dtype names.
# Torch stores two values per byte (x2 suffix); numpy stores one.
_NUMPY_TO_TORCH_4BIT_DTYPE = {
    "float4_e2m1": "float4_e2m1fn_x2",
    "float4_e1m2": "float4_e1m2fn_x2",
    # HIF4 uses FP4 E1M2 logical storage.  It may not have a torch.dtype
    # counterpart, in which case the packed uint8 tensor is intentional.
    "hifloat4": "float4_e1m2fn_x2",
    "int4": "int4",
}

# Numpy float8 dtypes -> torch dtype attribute names.
_NUMPY_TO_TORCH_FLOAT8_DTYPE = {
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
    "float8_e8m0": "float8_e8m0fnu",
}

# Shared dtype promotion map for golden "Promote" mode.
#
# Used by BOTH flows:
#   - KERNEL (output_generation.__promote_dtype) — numpy arrays
#   - ACLNN   (golden_generation._promote_dtype)  — torch tensors
#
# Promoting to a wider dtype yields the "true value" of the op computed at
# higher precision, against which the low-precision NPU output is compared.
# Kept here (not in either flow module) to avoid a cross-flow import dependency
# and duplication.
DTYPE_PROMOTE_MAP: Dict[str, str] = {
    "float16": "float32",
    "bfloat16": "float32",
    "float32": "float64",
    "complex32": "complex64",
    "complex64": "complex128",
    "float8_e5m2": "float32",
    "float8_e4m3fn": "float32",
    "float8_e8m0": "float32",
    "hifloat8": "float32",
}

DTYPE_WIDTH_MAP = {
    "complex32": 4,
    "complex64": 8,
    "complex128": 16,
    "double": 8,
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "uint64": 8,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
    "uint1": 1,
    "bool": 1,
    "float4_e2m1": 0.5,
    "float4_e1m2": 0.5,
    "float8_e8m0": 1,
    "float8_e5m2": 1,
    "float8_e4m3fn": 1,
    "hifloat8": 1,
}

DTYPE_MAP = {
    "c32": "complex32",
    "c64": "complex64",
    "complex32": "complex32",
    "complex64": "complex64",
    "c128": "complex128",
    "complex128": "complex128",
    "f64": "float64",
    "fp64": "float64",
    "float64": "float64",
    "double": "double",
    "f32": "float32",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
    "f16": "float16",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfp16": "bfloat16",
    "bfloat16": "bfloat16",
    "s64": "int64",
    "int64": "int64",
    "u64": "uint64",
    "uint64": "uint64",
    "s32": "int32",
    "int32": "int32",
    "u32": "uint32",
    "uint32": "uint32",
    "s16": "int16",
    "int16": "int16",
    "u16": "uint16",
    "uint16": "uint16",
    "s8": "int8",
    "int8": "int8",
    "u8": "uint8",
    "uint8": "uint8",
    "s4": "int4",
    "int4": "int4",
    "u1": "uint1",
    "uint1": "uint1",
    "s1": "uint1",
    "int1": "uint1",
    "bool": "bool",
    "f4_e2m1": "float4_e2m1",
    "fp4_e2m1": "float4_e2m1",
    "float4_e2m1": "float4_e2m1",
    "float4_e2m1fn": "float4_e2m1",
    "fp4_e2m1fn": "float4_e2m1",
    "f4_e2m1fn": "float4_e2m1",
    "f4_e1m2": "float4_e1m2",
    "fp4_e1m2": "float4_e1m2",
    "float4_e1m2": "float4_e1m2",
    "float4_e1m2fn": "float4_e1m2",
    "fp4_e1m2fn": "float4_e1m2",
    "f4_e1m2fn": "float4_e1m2",
    "f8_e8m0": "float8_e8m0",
    "fp8_e8m0": "float8_e8m0",
    "float8_e8m0": "float8_e8m0",
    "f8_e5m2": "float8_e5m2",
    "fp8_e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "f8_e4m3fn": "float8_e4m3fn",
    "fp8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "hif8": "hifloat8",
    "hifp8": "hifloat8",
    "hifloat8": "hifloat8",
    "hifloat4": "hifloat4",
    None: None,
}

# ACL (Ascend Computing Language) dtype enum id -> canonical dtype name
ACL_DTYPE_TO_ID = {
    "float32": 0,
    "float16": 1,
    "int8": 2,
    "int32": 3,
    "uint8": 4,
    "int16": 6,
    "uint16": 7,
    "uint32": 8,
    "int64": 9,
    "uint64": 10,
    "double": 11,
    "float64": 11,
    "bool": 12,
    "complex64": 16,
    "complex128": 17,
    "qint8": 18,
    "qint16": 19,
    "qint32": 20,
    "quint8": 21,
    "quint16": 22,
    "resource": 23,
    "dual": 25,
    "variant": 26,
    "bf16": 27,
    "bfloat16": 27,
    "int4": 29,
    "uint1": 30,
    "int2": 31,
    "uint2": 32,
    "complex32": 33,
    "hifloat8": 34,
    "float8_e5m2": 35,
    "float8_e4m3fn": 36,
    "float8_e8m0": 37,
    "float8_e8m0fnu": 37,
    "float4_e2m1": 40,
    "float4_e1m2": 41,
    "hifloat4": 42,
}

ACL_ID_TO_DTYPE = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    18: "qint8",
    19: "qint16",
    20: "qint32",
    21: "quint8",
    22: "quint16",
    23: "resource",
    25: "dual",
    26: "variant",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    31: "int2",
    32: "uint2",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}

# Custom numpy dtypes (not provided by numpy itself) -> provider package:
# (module_name, attr_name, min_version); min_version None means no version check.
# hifloat4 uses FP4 E1M2 as its logical storage dtype.
_NUMPY_DTYPE_SOURCES: Dict[str, tuple] = {
    "bfloat16": ("ml_dtypes", "bfloat16", None),
    "int4": ("ml_dtypes", "int4", None),
    "float8_e5m2": ("ml_dtypes", "float8_e5m2", None),
    "float8_e4m3fn": ("ml_dtypes", "float8_e4m3fn", None),
    "float8_e8m0": ("en_dtypes", "float8_e8m0", "0.0.4"),
    "float4_e2m1": ("en_dtypes", "float4_e2m1", "0.0.4"),
    "float4_e1m2": ("en_dtypes", "float4_e1m2", "0.0.4"),
    "hifloat8": ("en_dtypes", "hifloat8", "0.0.4"),
    "hifloat4": ("en_dtypes", "float4_e1m2", "0.0.4"),
}


def normalize_dtype(dtype_str: str):
    """Normalize a dtype alias ('fp16', 'bf16', ...) to its canonical name ('float16', ...)."""
    return DTYPE_MAP[dtype_str]


def dtype_to_name(dtype) -> str:
    """Robustly extract a dtype name string ('float16', 'bfloat16', ...).

    Accepts numpy dtypes, torch dtypes, ml_dtypes class objects/dtypes and
    plain names.  numpy.dtype(...).name is authoritative when parseable;
    direct str().split('.')[-1] on a class object yields "bfloat16'>" (the
    trailing "'>" makes name-keyed lookups silently miss), so it is only the
    fallback.
    """
    try:
        return numpy.dtype(dtype).name
    except Exception:
        return str(dtype).split(".")[-1].rstrip("'>\" ")


def is_float_dtype(dtype_name) -> bool:
    """True for floating-point dtypes: float*/bfloat*/hifloat* families, double, half.

    Accepts dtype names (canonical or alias like 'fp16'/'bf16') and dtype
    objects; complex, integer and bool types are not floating-point.
    """
    if dtype_name is None:
        return False
    name = dtype_to_name(dtype_name)
    canonical = DTYPE_MAP.get(name, name)
    return canonical.startswith(("float", "bfloat", "hifloat")) or canonical in ("double", "half")


def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = numpy.dtype(dt)
    numpy_info = numpy.iinfo(numpy_dtype) if numpy_dtype.kind in "iu" else numpy.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def is_4bit_dtype(dtype) -> bool:
    """
    True if dtype is a packed 4-bit type (int4 / float4_e2m1 / float4_e1m2 / hifloat4).

    Accepts str, numpy.dtype, torch.dtype, or any object whose str() contains
    the dtype name.
    """
    s = str(dtype)
    return "int4" in s or "float4" in s


def get_dtype_width(dt: str) -> Union[int, float]:
    """Dtype string to byte width mapping."""
    if dt is None:
        return 0
    str_dtype = dt if isinstance(dt, str) else getattr(dt, "name", "")
    if is_4bit_dtype(str_dtype):
        return 0.5
    try:
        return numpy.dtype(dt).itemsize
    except TypeError:
        pass
    try:
        return DTYPE_WIDTH_MAP[dt]
    except KeyError:
        raise RuntimeError(f"Unknown dtype [{dt}]") from None


def numpy_int4():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import int4

        return int4
    except ModuleNotFoundError:
        raise RuntimeError(
            "ml_dtypes is needed to support int4 dtype!!! Please install with `pip3 install ml-dtypes`"
        ) from None


def numpy_float4_e2m1():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float4_e2m1

        _ensure_en_dtypes_version("0.0.4")
        return float4_e2m1
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float4_e2m1 dtype!!! Please install with `pip3 install en-dtypes`"
        ) from None


def _ensure_en_dtypes_version(min_version: str):
    import en_dtypes

    cur_ver = list(map(int, en_dtypes.__version__.split("."))) + [0, 0]
    min_ver = list(map(int, min_version.split("."))) + [0, 0]
    if (cur_ver[0], cur_ver[1], cur_ver[2]) < (min_ver[0], min_ver[1], min_ver[2]):
        raise RuntimeError(f"Please upgrade en-dtypes to at least {min_version}")


def resolve_numpy_dtype(dtype_name: str):
    """Resolve a custom dtype name (bfloat16/int4/fp8/fp4/hifloat) to its numpy dtype object.

    These dtypes are provided by third-party packages (ml_dtypes/en_dtypes).
    Raises RuntimeError with install/upgrade instructions when the package is
    missing or too old; bfloat16 falls back to tensorflow when ml_dtypes is
    absent.
    """
    spec = _NUMPY_DTYPE_SOURCES.get(dtype_name)
    if spec is None:
        raise RuntimeError(f"Unsupported custom numpy dtype [{dtype_name}]")
    module_name, attr_name, min_version = spec
    try:
        module = importlib.import_module(module_name)
        np_dtype = getattr(module, attr_name)
    except ModuleNotFoundError:
        if dtype_name == "bfloat16":
            try:
                import tensorflow

                return tensorflow.bfloat16.as_numpy_dtype
            except ModuleNotFoundError:
                raise RuntimeError(
                    "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                    "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`"
                ) from None
        raise RuntimeError(
            f"{module_name} is needed to support {dtype_name} dtype!!! "
            f"Please install with `pip3 install {module_name.replace('_', '-')}`"
        ) from None
    except AttributeError:
        raise RuntimeError(
            f"Please upgrade {module_name} to at least {min_version} to support {dtype_name} dtype!!! "
            f"Command is `pip3 install --upgrade {module_name.replace('_', '-')}`"
        ) from None
    if min_version is not None:
        _ensure_en_dtypes_version(min_version)
    return np_dtype


def map_dtypes(container, fn):
    """Recursively apply fn to each dtype in a nested tuple/list structure.

    Nesting depth is preserved; sequence levels are returned as tuples.
    """
    if not container:
        return container

    def _convert(item):
        if isinstance(item, (tuple, list)):
            return tuple(_convert(e) for e in item)
        return fn(item)

    return _convert(container)


def resolve_numpy_dtypes(container):
    """Convert custom dtype names to numpy dtype objects, supporting nested tuples/lists.

    TF quantized dtypes (quint8/qint8/...) are aliased to their numpy storage
    dtypes (uint8/int8/...); the declared dtype string stays on the testcase
    for tf conversion (!297).
    """

    def _resolve(name):
        if not isinstance(name, str):
            return name
        if name in TF_QUANTIZED_DTYPE_NUMPY_STORAGE:
            return TF_QUANTIZED_DTYPE_NUMPY_STORAGE[name]
        if name in _NUMPY_DTYPE_SOURCES:
            return resolve_numpy_dtype(name)
        return name

    return map_dtypes(container, _resolve)


def pack_4bits(src: numpy.ndarray):
    """
    Pack int4/float4 numpy array (each int4 number stored in one byte actually)
    to be continuous bytes stored in uint8
    """
    if not is_4bit_dtype(src.dtype):
        raise RuntimeError("Dtype of source tensor only support int4/float4")
    pack_size = 2
    shift = numpy.array([0, 4], dtype=numpy.uint8)
    array = src
    if src.size % pack_size != 0:
        array = numpy.pad(src.flatten(), (0, pack_size - src.size % pack_size), mode="constant")
    reshaped = array.reshape([-1, 2])
    # bitwise_and is for arm
    out = numpy.sum(numpy.bitwise_and(reshaped.view(numpy.uint8), 0b00001111) << shift, axis=1, dtype=numpy.uint8)
    return out


def unpack_4bits(src: numpy.ndarray, dst_dtype):
    """
    Unpack uint8 numpy array to int4 array
    """
    if isinstance(dst_dtype, str):
        dst_dtype = resolve_numpy_dtype(dst_dtype)
    shift = numpy.array([0, 4], dtype=numpy.uint8)
    return numpy.bitwise_and(src.reshape([-1, 1]) >> shift, 0b00001111).view(dst_dtype).reshape([-1])


# TF quantized dtypes (tf.quint8/qint8/...) have no numpy counterpart: numpy
# storages hold the plain integer dtype below, and the quantized semantics are
# restored at tf.convert_to_tensor time via dtype=tf.<name> (declared dtype
# stays "quint8"/... in the testcase metadata).
TF_QUANTIZED_DTYPE_NUMPY_STORAGE = {
    "qint8": "int8",
    "qint16": "int16",
    "qint32": "int32",
    "quint8": "uint8",
    "quint16": "uint16",
}


def tf_quantized_dtype(declared_dtype):
    """Return the tf quantized dtype object (tf.quint8/...) for a declared dtype string, or None."""
    name = str(declared_dtype) if declared_dtype is not None else ""
    if name in TF_QUANTIZED_DTYPE_NUMPY_STORAGE:
        import tensorflow as tf

        return getattr(tf, name)
    return None


def normalize_to_tf_dtype(np_array: numpy.ndarray):
    import tensorflow as tf

    if np_array.dtype.name == "bfloat16" and np_array.dtype.type != tf.bfloat16.as_numpy_dtype.dtype.type:
        return np_array.view(tf.bfloat16.as_numpy_dtype)
    return np_array


def normalize_from_tf_dtype(tf_tensor):
    """Inverse of normalize_to_tf_dtype: re-view TF's bfloat16 as TTK's numpy bfloat16 if they differ."""
    if tf_tensor.dtype.name == "bfloat16":
        ttk_bf16 = resolve_numpy_dtype("bfloat16")
        if ttk_bf16.dtype.type != tf_tensor.dtype.type:
            return tf_tensor.view(ttk_bf16)
    return tf_tensor


def acl_to_torch_dtype(container):
    """Convert a sequence of ACL dtype enum ids to torch dtype objects."""
    import torch

    names = [ACL_ID_TO_DTYPE[acl_id] for acl_id in container]
    return tuple(getattr(torch, name) if isinstance(name, str) else name for name in names)


def numpy_to_torch_tensor(np_array: numpy.ndarray, is_complex32: bool = False):
    """Convert a numpy array to a torch tensor, keeping non-native dtypes (bf16/fp8/fp4/complex32)."""
    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        return _numpy_bfloat16_to_torch(np_array)
    if is_4bit_dtype(np_dtype):
        return _numpy_4bit_to_torch(np_array)
    if "float8" in np_dtype:
        return _numpy_float8_to_torch(np_array)
    if is_complex32:
        return _numpy_complex32_to_torch(np_array)
    import torch

    return torch.from_numpy(np_array)


def _numpy_bfloat16_to_torch(np_array: numpy.ndarray):
    import torch

    return torch.from_numpy(np_array.view(dtype=numpy.int16)).view(torch.bfloat16)


def _numpy_4bit_to_torch(np_array: numpy.ndarray):
    import torch

    np_dtype = np_array.dtype.name
    if np_dtype == "int4":
        raise RuntimeError("int4 numpy arrays cannot be converted to torch.Tensor (no functional torch.int4)")
    torch_dtype_name = _NUMPY_TO_TORCH_4BIT_DTYPE.get(np_dtype)
    torch_dtype = None if torch_dtype_name is None else getattr(torch, torch_dtype_name, None)
    if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
        torch_dtype = None
    if np_array.ndim == 0 or np_array.shape[-1] % 2:
        raise RuntimeError(f"4-bit tensor must have an even last dimension, got shape {tuple(np_array.shape)}")
    packed_shape = (*np_array.shape[:-1], np_array.shape[-1] // 2)
    packed = pack_4bits(numpy.ascontiguousarray(np_array)).reshape(packed_shape)
    if torch_dtype is None:
        # torch_npu exposes some 4-bit identifiers as ACL enum integers,
        # not torch.dtype objects.  Keep their packed storage as uint8;
        # the operator's dtype attribute supplies the device semantics.
        logging.warning(
            "Torch dtype %s is unavailable for %s; using packed uint8 storage with shape %s. "
            "The operator must supply the device dtype semantics.",
            torch_dtype_name,
            np_dtype,
            packed_shape,
        )
        return torch.from_numpy(packed)
    return torch.from_numpy(packed).view(torch_dtype)


def _numpy_float8_to_torch(np_array: numpy.ndarray):
    import torch

    torch_dtype_name = _NUMPY_TO_TORCH_FLOAT8_DTYPE.get(np_array.dtype.name)
    if torch_dtype_name is None:
        raise RuntimeError(f"Dtype [{np_array.dtype.name}] is not supported to convert to torch.Tensor yet.")
    if not hasattr(torch, torch_dtype_name):
        raise RuntimeError(
            f"Current pytorch version [{torch.__version__}] is too old. {torch_dtype_name} is not supported."
        )
    return torch.from_numpy(np_array.view(dtype=numpy.uint8)).view(getattr(torch, torch_dtype_name))


def _numpy_complex32_to_torch(np_array: numpy.ndarray):
    import torch

    if np_array.dtype.name != "float16":
        raise RuntimeError(f"Can only transfer numpy.float16 to torch.complex32 rather than {np_array.dtype.name}")
    if not hasattr(torch, "complex32"):
        raise RuntimeError(
            f"Current pytorch version [{torch.__version__}] is too old. Please update to at least v1.13.1"
        )
    return torch.from_numpy(np_array).view(torch.complex32)


def torch_to_numpy_tensor(torch_tensor) -> numpy.ndarray:
    """Convert a torch tensor to a numpy array, keeping non-native dtypes (bf16/fp8/fp4/complex32)."""
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    dtype_str = str(torch_tensor.dtype)
    if torch_tensor.dtype == torch.bfloat16:
        return _torch_bfloat16_to_numpy(torch_tensor)
    if "complex32" in dtype_str:
        return torch_tensor.view(torch.float16).numpy()
    if "float8" in dtype_str:
        return _torch_float8_to_numpy(torch_tensor)
    if is_4bit_dtype(dtype_str):
        return _torch_4bit_to_numpy(torch_tensor)
    return torch_tensor.numpy()


def _torch_bfloat16_to_numpy(torch_tensor):
    import torch

    np_int16 = torch_tensor.view(torch.int16).numpy()
    return np_int16.view(dtype=resolve_numpy_dtype("bfloat16"))


def _torch_float8_to_numpy(torch_tensor):
    import torch

    suffix = str(torch_tensor.dtype).rsplit(".", maxsplit=1)[-1].replace("fnu", "")
    np_uint8 = torch_tensor.view(torch.uint8).numpy()
    return np_uint8.view(resolve_numpy_dtype(suffix))


def _torch_4bit_to_numpy(torch_tensor):
    import torch

    np_uint8 = torch_tensor.view(torch.uint8).numpy()
    torch_dtype_name = str(torch_tensor.dtype).rsplit(".", maxsplit=1)[-1]
    for np_name, torch_name in _NUMPY_TO_TORCH_4BIT_DTYPE.items():
        if torch_name == torch_dtype_name:
            unpacked = unpack_4bits(np_uint8, resolve_numpy_dtype(np_name))
            if np_uint8.ndim == 0:
                return unpacked
            return unpacked.reshape(*np_uint8.shape[:-1], np_uint8.shape[-1] * 2)
    raise RuntimeError(f"Unsupported torch 4-bit dtype [{torch_dtype_name}]")


# MX quantization element dtypes (the target formats mx_quantize supports).
MX_ELEMENT_DTYPES = ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2")


def _mx_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, block_size: int):
    fp_array = numpy.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = numpy.pad(fp_array, pad, "constant")
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple):
    # Undo tile reshaping
    fp_array = fp_array.reshape(padded_shape)
    # Undo padding
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = numpy.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(numpy.log2(max_norm))
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True)
    res = numpy.floor(numpy.log2(fp_abs_max.astype(numpy.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))) - ele_emax
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_nv(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True).astype(numpy.float32)
    s_fp32 = fp_abs_max / max_norm
    binary_ints = numpy.array(s_fp32.view(numpy.uint32))
    exponent_mask = numpy.uint32(0x7F800000)
    mantissa_mask = numpy.uint32(0x007FFFFF)
    # Extract the exponent bits and convert to int16
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(numpy.int16)
    # Extract the mantissa bits
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    # A mantissa >= 2**22 with zero exponent means the fp32 value rounds up to
    # the next power of two when ties are broken away from zero.
    condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
    exponents_int16 = numpy.where((condition_1 | condition_2), exponents_int16 + 1, exponents_int16)
    res = (exponents_int16 - 127).astype(numpy.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
    """
    For example:
    fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, -0.0, 0.0, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
    - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
    - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
    - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0,   0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
    - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    """
    if round_mode in ("rint", "even"):  # tie to even(c language rint)
        fp_array = numpy.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
        sign = numpy.signbit(fp_array)
        rounded_abs = numpy.floor(numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
        fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity(c language floor)
        fp_array = numpy.floor(fp_array)
    elif round_mode == "ceil":  # round to positive infinity(c language ceil)
        fp_array = numpy.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero(c language truncation)
        fp_array = numpy.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(
    fp_array: numpy.ndarray, share_exp: numpy.ndarray, mx_ele_dtype: str, round_mode: str
):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r"e(\d+)m(\d+)", mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2**share_exp)
    private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))).astype(
        fp_array.dtype, copy=False
    )
    # The minimum representable exponent
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2**private_exp) * (2**mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2**mantissa_bits) * (2**private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    max_norm = get_dtype_range(mx_dtype)[1]
    numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _pad_scale_to_even(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
    """Pad `axis` to an even length.

    The pad value is 2**-127 (the smallest representable E8M0 scale), NOT
    zero: E8M0 is a pure exponent format and cannot encode 0, so a zero pad
    would corrupt the scale stream.
    """
    if tensor.shape[axis] % 2 == 0:
        return tensor
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)
    return numpy.pad(tensor, pad_width, mode="constant", constant_values=2**-127)


def _interleave_pairs(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
    """Make `axis`-adjacent elements contiguous in a trailing pair dimension.

    (..., n, ...) with even n along `axis` becomes (..., n//2, ..., 2).
    """
    if tensor.shape[axis] % 2:
        raise ValueError(f"Axis length ({tensor.shape[axis]}) must be even")
    shape = list(tensor.shape)
    new_shape = shape[:axis] + [shape[axis] // 2, 2] + shape[axis + 1 :]
    transpose_order = list(range(axis + 1)) + list(range(axis + 2, len(new_shape))) + [axis + 1]
    return tensor.reshape(new_shape).transpose(transpose_order)


def _mx_pack_scales(scale_array: numpy.ndarray, axis: int) -> numpy.ndarray:
    """Pack float32 scales into the NPU MX scale layout: pairs along `axis` as a trailing dim of 2.

    Cube units require an even number of scales, so odd tails are padded first.
    For non-last axes, axis-adjacent scales are interleaved so that each pair
    becomes memory-contiguous before the final (..., n//2, 2) reshape.
    """
    scale_array = _pad_scale_to_even(scale_array, axis=axis)
    result_shape = list(scale_array.shape)
    result_shape[axis] //= 2
    result_shape.append(2)
    if axis != len(scale_array.shape) - 1:
        scale_array = _interleave_pairs(scale_array, axis=axis)
    return scale_array.reshape(result_shape).astype(resolve_numpy_dtype("float8_e8m0"), copy=False)


def mx_quantize(
    fp_array: numpy.ndarray,
    mx_ele_dtype: str = "float4_e2m1",
    axis: int = -1,
    block_size: int = 32,
    round_mode: str = "rint",
    scale_alg: int = 0,
) -> tuple:
    """
    quantize BFP16/FP16/FP32 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16/FP32
    :parameter mx_ele_dtype: dtype of element in MX dtype.
               support float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :parameter scale_alg: The calculation method for scale.Support MxFP8(OCP , count 0) or MxFP8(nvidia-cuBLAS , count 1)
    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """
    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in MX_ELEMENT_DTYPES:
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
    # get mx scale exponents
    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    else:
        share_exp = _mx_calculate_share_exp_nv(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis + 1)

    ele_dtype_np = resolve_numpy_dtype(mx_ele_dtype)
    # share_exp is always float32
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = _mx_pack_scales(scale_array, axis=axis)

    return scale_array, ele_array


def is_torch_native_dtype(dtype_name) -> bool:
    """Check if a dtype name is natively supported by the current torch version.

    Uses ``hasattr(torch, dtype_name)`` to dynamically detect support,
    avoiding hardcoded blacklists that become stale as torch evolves.

    Note: ``torch.int4`` exists but is non-functional on CPU (cannot create
    valued tensors, no numpy interop), so it is explicitly excluded.

    ``float8_e8m0`` is an alias for ``torch.float8_e8m0fnu``; the canonical
    torch attribute name differs, so the alias is resolved before checking.
    """
    import torch

    name = str(dtype_name)
    if name == "int4":
        return False
    return hasattr(torch, _TORCH_DTYPE_ALIAS.get(name, name))


def _import_module_safe(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def str_to_torch_dtype(dtype_str: str):
    """Convert a dtype string to a torch/torch_npu dtype object.

    Accepts bare names ('float16') and qualified names ('torch.float16',
    'torch_npu.hifloat8').  A bare name resolves against torch first, then
    torch_npu.  Returns None when the name is not recognized (including when
    the required module is not installed); non-str input is returned unchanged.
    """
    if not isinstance(dtype_str, str):
        return dtype_str
    parts = dtype_str.split(".")
    if len(parts) > 2:
        return None
    canonical = parts[-1]
    if len(parts) == 2 and parts[0] not in ("torch", "torch_npu"):
        return None
    if len(parts) == 1 and is_torch_native_dtype(canonical):
        module = _import_module_safe("torch")
    else:
        module = _import_module_safe(parts[0] if len(parts) == 2 else "torch_npu")
    if module is None:
        return None
    return getattr(module, _TORCH_DTYPE_ALIAS.get(canonical, canonical), None)


_TF_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
    "double": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
    "complex64": "complex64",
    "complex128": "complex128",
    "qint8": "qint8",
    "qint16": "qint16",
    "qint32": "qint32",
    "quint8": "quint8",
    "quint16": "quint16",
}


def str_to_tf_dtype(dtype_str: str):
    """Convert dtype string to tf.dtype object.

    Args:
        dtype_str: string like 'float16', 'int8', 'fp16', 'bf16', etc.

    Returns:
        tf.dtype object, or None if not recognized.
    """
    if not isinstance(dtype_str, str):
        return dtype_str
    import tensorflow as tf

    canonical = DTYPE_MAP.get(dtype_str, dtype_str)
    tf_name = _TF_DTYPE_MAP.get(canonical)
    if tf_name is None:
        return None
    return getattr(tf, tf_name, None)


def is_tf_native_dtype(dtype_name) -> bool:
    """Check if a dtype name is natively supported by TensorFlow."""
    canonical = DTYPE_MAP.get(str(dtype_name), str(dtype_name))
    return canonical in _TF_DTYPE_MAP
