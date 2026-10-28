#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
MX (microscaling) quantization: BFP16/FP16/FP32 -> MX dtypes.

Pure-algorithm module; the only ttk dependency is dtypes' get_dtype_range /
resolve_numpy_dtype. Consumers import via the ttk.utilities facade or the
lazy forwarding shim kept at the tail of dtypes.py — both paths resolve to
the same objects.
"""

import re

import numpy

from .dtypes import get_dtype_range, resolve_numpy_dtype

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
