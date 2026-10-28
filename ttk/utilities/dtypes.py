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
TTK dtype 归一与各生态桥接（纯数据/纯转换层，不触任何动态库）。

分区导航：
- canonical 表：DTYPE_MAP（别名 → 规范名）、DTYPE_WIDTH_MAP、DTYPE_PROMOTE_MAP、
  _DTYPE_RANGE_OVERRIDES（get_dtype_range 的自定义 dtype 显式范围表）；
- ACL 桥：ACL_DTYPE_TO_ID / ACL_ID_TO_DTYPE（canonical 名 ↔ ACL dtype 枚举 id）；
- ONNX 桥：ONNX_DTYPE_MAP / str_to_onnx_dtype / is_onnx_native_dtype
  （TensorProto 裸 int，不 import onnx）；
- torch 桥：numpy ↔ torch 张量互转（非原生 dtype 以视图保持）、str_to_torch_dtype；
- tf 桥：str_to_tf_dtype / is_tf_native_dtype、TF 量化 dtype 的 numpy 存储别名；
- numpy 扩展 dtype：ml_dtypes/en_dtypes 提供的 bf16/int4/fp8/fp4/hifloat 解析；
- 4bit 打包：pack_4bits / unpack_4bits。

MX 量化算法在 utilities/mx_quant.py，本模块尾部以 PEP 562 惰性转发保持历史
import 路径（饿式反向 import 在「先 import mx_quant」的入口序下会成环）。
"""

import importlib
import logging
from typing import Dict, Tuple, Union

import numpy

# Canonical TTK dtype name -> actual attribute name on torch/torch_npu
# (torch spells float8_e8m0 as float8_e8m0fnu).
_TORCH_DTYPE_ALIAS = {
    "float8_e8m0": "float8_e8m0fnu",
}

# torch/torch_npu attribute spelling -> canonical TTK dtype name (reverse of
# _TORCH_DTYPE_ALIAS); folds e.g. dtype_to_name(torch.float8_e8m0fnu) back to
# the canonical name for table lookups.
_TORCH_DTYPE_TO_CANONICAL = {v: k for k, v in _TORCH_DTYPE_ALIAS.items()}

# Numpy 4-bit dtypes (unpacked, 1 byte/element) -> torch packed dtype names.
# Torch stores two values per byte (x2 suffix); numpy stores one.
_NUMPY_TO_TORCH_4BIT_DTYPE = {
    "float4_e2m1": "float4_e2m1fn_x2",
    "float4_e1m2": "float4_e1m2fn_x2",
    # HIF4 uses FP4 E1M2 logical storage.  It may not have a torch.dtype
    # counterpart, in which case the packed uint8 tensor is intentional.
    "hifloat4": "float4_e1m2fn_x2",
    # int4 不登记 view 目标：torch.int4 是 1 字节/元素的非功能 dtype，pack
    # 产物（2 元素/字节）经 view 会错长——走 packed uint8 降级分支，dtype
    # 语义由算子侧属性（ACL 枚举，如 x_dtype）承载
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
    "hifloat4": 0.5,
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

# ACL (Ascend Computing Language) canonical dtype name -> ACL dtype enum id
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


# Explicit range table (canonical dtype name -> (min, max)) for formats that
# have no numpy counterpart or carry non-numpy semantics; keys are canonical
# names only, so lookups never hit by substring. complex32 is not listed here:
# each of its components is a float16, so it resolves through numpy.finfo.
_DTYPE_RANGE_OVERRIDES = {
    "bfloat16": (-float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")),
    "uint4": (0, 15),
    "int4": (-8, 7),
    "bool": (0, 1),
    "float4_e2m1": (-float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")),
    "float4_e1m2": (-float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")),
    "float8_e8m0": (float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")),
    "float8_e5m2": (-float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")),
    "float8_e4m3fn": (-float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")),
    "hifloat8": (-float.fromhex("0x1.p15"), float.fromhex("0x1.p15")),
}


def get_dtype_range(dt):
    """Dtype value range as (min, max).

    Accepts canonical names, aliases ('fp16'/'bf16'/...), numpy/torch/
    ml_dtypes dtype objects: the name is normalized to its canonical form
    first (dtype_to_name + DTYPE_MAP, torch spellings folded via
    _TORCH_DTYPE_TO_CANONICAL), then looked up in _DTYPE_RANGE_OVERRIDES;
    complex32 resolves per float16 component; everything else goes through
    numpy.iinfo/finfo. Unknown dtypes surface numpy's own TypeError.
    """
    name = dtype_to_name(dt)
    canonical = DTYPE_MAP.get(_TORCH_DTYPE_TO_CANONICAL.get(name, name), name)
    if canonical == "complex32":
        canonical = "float16"
    override = _DTYPE_RANGE_OVERRIDES.get(canonical)
    if override is not None:
        return override
    numpy_dtype = numpy.dtype(canonical)
    numpy_info = numpy.iinfo(numpy_dtype) if numpy_dtype.kind in "iu" else numpy.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


# 特殊格式顶 binade 虚拟步长：RNE 溢出边界 = max_finite + 步长/2。
# IEEE 满尾数格式由 frexp 公式 (max + 2^(emax+1))/2 覆盖（两式相等）；下列格式
# 顶 binade 结构特殊，frexp 不适用，逐格式登记（Ascend950 实测）：
#   float8_e4m3fn: 顶 binade [2^8,2^9) 尾数槽被 NaN 编码占用（无 inf），
#                  上一格是 480=NaN 槽而非 2^9 → 步长 2^8/8=32，边界 464，
#                  tie 舍到偶尾数 110 → 448（有限值，与 IEEE tie→inf 相反）
#   hifloat8:      顶 binade [2^15,2^16) 尾数 1 位，inf 槽位于 1.5×2^15 虚位
#                  （0x6F）→ 步长 2^15/2=16384，边界 40960，tie 归 inf
_SPECIAL_TOP_BINADE_STEP = {"float8_e4m3fn": 32.0, "hifloat8": 16384.0}


def get_dtype_overflow_boundary(dt, rounding: str = "rne") -> Tuple[float, float]:
    """按 IEEE 754 舍入模式返回浮点 dtype 的溢出截断点 (lo, hi)。

    约定：value <= lo → -inf；value >= hi → +inf（端点含）；
    某侧返回 ±inf 表示该模式下永不向该侧溢出（超界值饱和到该侧 max_finite）。
    例外：float8_e4m3fn 无 inf 编码且 tie 归有限值——消费方须用严格比较
    （> hi / < lo）并把带外值替换为 NaN 而非 inf。

    舍入模式（rounding 参数，大小写不敏感）：
        rne  roundToNearestTiesToEven   就近舍入·偶数优先（IEEE 二进制默认）
        rna  roundToNearestTiesToAway   就近舍入·远离零
        rtp  roundTowardPositive        向 +∞
        rtn  roundTowardNegative        向 -∞
        rtz  roundTowardZero            向零截断

    各模式截断点：
        rne/rna: ±(max_finite + 2^(emax+1)) / 2   —— tie 归入 inf 侧
                 （等价 max_finite + 顶 binade 步长/2，顶 binade 尾数全满时两式
                 相等；e4m3fn/hifloat8 顶 binade 结构特殊，按
                 _SPECIAL_TOP_BINADE_STEP 登记步长）
        rtp:     (-inf, nextafter(max_finite, +inf)) —— 正侧超 max 即 +inf，负侧饱和
        rtn:     (nextafter(-max_finite, -inf), +inf)
        rtz:     (-inf, +inf)                      —— 永不溢出（双向饱和）

    适用于含 inf 编码的格式（float16/bfloat16/float32/float64/float8_e5m2/
    float8_e4m3fn/hifloat8）。float8_e8m0 无 inf 且双侧饱和，无溢出边界概念，
    不适用——消费方直接用 get_dtype_range 做饱和。
    """
    mode = str(rounding).lower()
    if mode not in ("rne", "rna", "rtp", "rtn", "rtz"):
        raise ValueError(f"unsupported rounding mode: {rounding!r}; expected rne/rna/rtp/rtn/rtz")
    lo, hi = get_dtype_range(dt)
    if mode in ("rne", "rna"):
        step = next((v for k, v in _SPECIAL_TOP_BINADE_STEP.items() if k in str(dt)), None)
        boundary = hi + step / 2.0 if step is not None else (hi + 2.0 ** numpy.frexp(hi)[1]) / 2.0
        return -boundary, boundary
    if mode == "rtp":
        return float("-inf"), float(numpy.nextafter(hi, numpy.inf))
    if mode == "rtn":
        return float(numpy.nextafter(lo, -numpy.inf)), float("inf")
    return float("-inf"), float("inf")


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
            torch_dtype_name or "<unmapped>",
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
    if torch_tensor.is_conj():
        torch_tensor = torch_tensor.resolve_conj()
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


# ============ ONNX (TensorProto) dtype bridge ============
# onnx TensorProto.Enum values as bare ints — do NOT import onnx here: dtypes
# is a low-level module that must stay importable in environments without
# onnx installed. Values follow the official onnx proto numbering; entries
# past 16 do not exist in every onnx release (e.g. 1.13.x stops at BFLOAT16),
# they are forward registrations of the proto enum, not local capability.
#
# Not mapped (str_to_onnx_dtype -> 0 / is_onnx_native_dtype -> False); callers
# rejecting these dtypes should name both sides — the TTK name and the missing
# TensorProto enum:
#   TTK 侧无 TensorProto 对应：
#   uint1/int1          proto 无 UINT1；
#   hifloat8/hifloat4   华为私有格式，无 onnx 枚举；
#   float4_e1m2         proto 的 4bit 浮点仅定义 E2M1；
#   complex32           proto 最小复数是 complex64；
#   TensorProto 侧无 TTK canonical 名：
#   STRING(8)           非张量数据；
#   FLOAT8E4M3FNUZ(18)/FLOAT8E5M2FNUZ(20)。
ONNX_DTYPE_MAP = {  # canonical dtype name -> TensorProto.Enum
    "float32": 1,
    "uint8": 2,
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "bool": 9,
    "float16": 10,
    "float64": 11,
    "double": 11,
    "uint32": 12,
    "uint64": 13,
    "complex64": 14,
    "complex128": 15,
    "bfloat16": 16,
    "float8_e4m3fn": 17,
    "float8_e5m2": 19,
    "int4": 22,
    "float4_e2m1": 23,
    "float8_e8m0": 24,
    "uint2": 25,
    "int2": 26,
}
ONNX_ID_TO_DTYPE = {v: k for k, v in ONNX_DTYPE_MAP.items() if k != "double"}  # id -> canonical


def str_to_onnx_dtype(dtype_str) -> int:
    """Convert a dtype name or object to the onnx TensorProto enum id (bare int).

    Aliases are normalized through DTYPE_MAP first (torch spellings such as
    'float8_e8m0fnu' folded via _TORCH_DTYPE_TO_CANONICAL); dtype objects
    (numpy/torch/ml_dtypes) go through dtype_to_name. Unknown or unmapped
    dtypes return 0 (TensorProto.UNDEFINED).
    """
    if dtype_str is None:
        return 0
    name = dtype_str if isinstance(dtype_str, str) else dtype_to_name(dtype_str)
    canonical = DTYPE_MAP.get(_TORCH_DTYPE_TO_CANONICAL.get(name, name), name)
    return ONNX_DTYPE_MAP.get(canonical, 0)


def is_onnx_native_dtype(dtype_name) -> bool:
    """Check if a dtype name is natively supported by onnx TensorProto.

    与 str_to_onnx_dtype 同口径归一（torch 属性拼写经 _TORCH_DTYPE_TO_CANONICAL
    折叠），两谓词对同一入参结论一致。
    """
    name = str(dtype_name)
    canonical = DTYPE_MAP.get(_TORCH_DTYPE_TO_CANONICAL.get(name, name), name)
    return canonical in ONNX_DTYPE_MAP


# ============ MX quantization (moved to utilities/mx_quant.py) ============
# Compatibility shim for historical import paths
# (`from ttk.utilities.dtypes import mx_quantize` etc.): the algorithm lives in
# mx_quant.py and is re-exported lazily via PEP 562 module __getattr__. An
# eager tail re-import would form a half-initialized cycle when mx_quant
# itself is imported first (mx_quant imports dtypes at its top), so the
# forwarding must stay lazy.
_MX_QUANT_FORWARDS = ("MX_ELEMENT_DTYPES", "mx_quantize")


def __getattr__(name):
    if name in _MX_QUANT_FORWARDS:
        module = importlib.import_module(".mx_quant", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
