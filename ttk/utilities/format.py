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
NPU format utilities: format code dict, format predicates and ND <-> private
format (FRACTAL_Z / NZ / NC1HWC0 ...) data transformation.
"""

__all__ = [
    "FORMAT_CODES",
    "NZ_FORMATS",
    "is_nchw_like",
    "is_ndhwc_like",
    "BLOCK_SIZE",
    "PAD_C0_MAPPING",
    "align_factor",
    "resolve_c0",
    "transpose_axes",
    "is_transformable",
    "transform",
    "nd_shape_to_5hd",
    "nd_shape_to_nz",
    "FORMAT_TRANSFORMS",
    "from_nc1hwc0",
    "from_ndc1hwc0",
    "from_fractal_nz",
    "to_fractal_z",
    "to_fractal_z_c04",
    "to_fractal_z_3d",
    "from_fractal_z_3d",
    "to_nc1hwc0",
    "to_ndc1hwc0",
    "nd_to_fractal_nz",
    "nd_to_fractal_z",
]

# Standard Packages
import inspect
from typing import Tuple, Union

# Third-Party Packages
import numpy

from .math import align, ceil_div, lcm

# NZ-family layouts needing npu_format_cast after H2D (5HD/FRACTAL_Z ... excluded).
NZ_FORMATS = ["FRACTAL_NZ", "NZ"]


# NPU format name -> format code.
FORMAT_CODES = {
    "NCHW": 0,
    "NHWC": 1,
    "ND": 2,
    "NC1HWC0": 3,
    "FRACTAL_Z": 4,
    "NC1C0HWPAD": 5,
    "NHWC1C0": 6,
    "FSR_NCHW": 7,
    "FRACTAL_DECONV": 8,
    "C1HWNC0": 9,
    "FRACTAL_DECONV_TRANSPOSE": 10,
    "FRACTAL_DECONV_SP_STRIDE_TRANS": 11,
    "NC1HWC0_C04": 12,
    "FRACTAL_Z_C04": 13,
    "CHWN": 14,
    "FRACTAL_DECONV_SP_STRIDE8_TRANS": 15,
    "HWCN": 16,
    "NC1KHKWHWC0": 17,
    "BN_WEIGHT": 18,
    "FILTER_HWCK": 19,
    "HASHTABLE_LOOKUP_LOOKUPS": 20,
    "HASHTABLE_LOOKUP_KEYS": 21,
    "HASHTABLE_LOOKUP_VALUE": 22,
    "HASHTABLE_LOOKUP_OUTPUT": 23,
    "HASHTABLE_LOOKUP_HITS": 24,
    "C1HWNCoC0": 25,
    "MD": 26,
    "NDHWC": 27,
    "FRACTAL_ZZ": 28,
    "FRACTAL_NZ": 29,
    "NZ": 29,
    "NCDHW": 30,
    "DHWCN": 31,
    "NDC1HWC0": 32,
    "FRACTAL_Z_3D": 33,
    "CN": 34,
    "NC": 35,
    "DHWNC": 36,
    "FRACTAL_Z_3D_TRANSPOSE": 37,
    "FRACTAL_ZN_LSTM": 38,
    "FRACTAL_Z_G": 39,
    "RESERVED": 40,
    "ALL": 41,
    "NULL": 42,
    "ND_RNN_BIAS": 43,
    "FRACTAL_ZN_RNN": 44,
    "NYUV": 45,
    "NYUV_A": 46,
    "NCL": 47,
}


def is_nchw_like(shape, format_: str) -> bool:
    """
    check (shape rank 4) and format is NCHW-like or not
    """
    if len(shape) != 4:
        return False
    return len(format_) == 4 and all(c in format_ for c in "NCHW")


def is_ndhwc_like(shape, format_: str) -> bool:
    """
    check (shape rank 5) and format is NDHWC-like or not
    """
    if len(shape) != 5:
        return False
    return len(format_) == 5 and all(c in format_ for c in "NDCHW")


BLOCK_SIZE = 16  # AiCore cube block size.

# dtype -> C0 alignment factor (elements).
PAD_C0_MAPPING = {
    "int64": 4,
    "uint64": 4,
    "float32": 16,
    "bfloat16": 16,
    "int32": 16,
    "uint32": 16,
    "float16": 16,
    "int16": 16,
    "uint16": 16,
    "int8": 32,
    "uint8": 32,
    "bool": 32,
    "uint1": 256,
}


def align_factor(dtype: str = "float16"):
    """C0 alignment factor for a dtype (default 16)."""
    return PAD_C0_MAPPING.get(dtype, 16)


def transpose_axes(offset, base):
    """Transpose axes: leading dims 0..offset-1 kept, then base shifted by offset."""
    return list(range(offset)) + [x + offset for x in base]


def _calculate_group(cin, cout, groups, c0):
    """Grouped-conv split params: per-group cin/cout, magnification factor, padded counts."""
    cin_ori, cout_ori = cin // groups, cout // groups
    mag_factor0 = lcm(cin_ori, c0) // cin_ori
    mag_factor1 = lcm(cout_ori, BLOCK_SIZE) // cout_ori
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, BLOCK_SIZE)

    group_dict = {
        "real_g": ceil_div(groups, mag_factor),
        "mag_factor": mag_factor,
        "cin_g": cin_g,
        "cin1_g": cin_g // c0,
        "cout_g": cout_g,
        "cout1_g": cout_g // BLOCK_SIZE,
        "groups": groups,
        "cin_ori": cin_ori,
        "cout_ori": cout_ori,
    }
    return group_dict


def _get_ndchw_dim_and_transpose_axis(data: numpy.ndarray, data_format: str, is_ndchw: bool = False):
    """Extract (N(D)CHW dims, transpose axes) from data in an arbitrary format."""
    data_shape = data.shape
    n, d, c, h, w = 1, 1, 1, 1, 1
    transpose_axis = []
    if "N" in data_format:
        transpose_axis.append(data_format.index("N"))
        n = data_shape[transpose_axis[-1]]
    if "C" in data_format:
        transpose_axis.append(data_format.index("C"))
        c = data_shape[transpose_axis[-1]]
    else:
        data = data.reshape(data.shape + (1,))
        transpose_axis.append(len(data.shape) - 1)
    if is_ndchw and "D" in data_format:
        transpose_axis.append(data_format.index("D"))
        d = data_shape[transpose_axis[-1]]
    if "H" in data_format:
        transpose_axis.append(data_format.index("H"))
        h = data_shape[transpose_axis[-1]]
    if "W" in data_format:
        transpose_axis.append(data_format.index("W"))
        w = data_shape[transpose_axis[-1]]
    if is_ndchw:
        return (n, d, c, h, w), transpose_axis
    return (n, c, h, w), transpose_axis


def resolve_c0(dtype, target_shape: Union[list, tuple] = None) -> int:
    """C0 = target shape tail if set, else dtype alignment factor."""
    dtype = getattr(dtype, "name", dtype)
    if target_shape and target_shape[-1] > 0:
        return target_shape[-1]
    return align_factor(dtype)


def nd_shape_to_5hd(
    nd_shape, nd_format: str = "NCHW", dtype: str = "float16", fhd_shape: Union[list, tuple] = None
) -> Tuple:
    """NCHW-like shape -> (N, C1, H, W, C0) with C1 = ceil(C / C0)."""
    if not is_nchw_like(nd_shape, nd_format):
        raise RuntimeError(f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")
    c0 = resolve_c0(dtype, fhd_shape)
    n, c = nd_shape[nd_format.index("N")], nd_shape[nd_format.index("C")]
    h, w = nd_shape[nd_format.index("H")], nd_shape[nd_format.index("W")]
    c1 = ceil_div(c, c0)
    return n, c1, h, w, c0


def nd_shape_to_nz(nd_shape: Union[list, tuple], dtype: str = "float16", nz_shape: Union[list, tuple] = None) -> Tuple:
    """Shape with tail (M, N) -> (..., N1, M1, M0=16, N0) NZ blocks."""
    m, n = nd_shape[-2:]
    m0 = 16
    n0 = resolve_c0(dtype, nz_shape)
    m1 = ceil_div(m, m0)
    n1 = ceil_div(n, n0)
    return tuple(nd_shape[:-2]) + (n1, m1, m0, n0)


def from_nc1hwc0(data, target_shape, target_format: str = "NCHW"):
    """NC1HWC0 (C0-padded) -> NCHW-like, tail C padding stripped."""
    if not is_nchw_like(target_shape, target_format):
        raise RuntimeError(f"shape: {target_shape} of format {target_format} is not NCHW-like.")
    if data.size == 0 and 0 in target_shape:
        # 空张量（零元素、逻辑 shape 含零维）无块可拼、无数据需转置
        return numpy.empty(target_shape, dtype=data.dtype)
    fhd_shape = data.shape
    pad = 1 + (target_shape[target_format.index("C")] - 1) % fhd_shape[-1]
    main_block = data[:, : fhd_shape[1] - 1, :, :, :]  # main block
    tail_block = data[:, fhd_shape[1] - 1, :, :, :pad]  # tail block
    # NC1HWC0 -> NHWC1C0
    main_block = main_block.transpose((0, 2, 3, 1, 4))
    # NHWC1C0 -> NHWC
    main_block = main_block.reshape(main_block.shape[:3] + (-1,))
    # concatenate
    nhwc = numpy.concatenate((main_block, tail_block), axis=-1)
    if target_format != "NHWC":
        return nhwc.transpose(
            (
                "NHWC".index(target_format[0]),
                "NHWC".index(target_format[1]),
                "NHWC".index(target_format[2]),
                "NHWC".index(target_format[3]),
            )
        )
    return nhwc


def from_ndc1hwc0(data, target_shape, target_format: str = "NDCHW"):
    """NDC1HWC0 (C0-padded) -> NDCHW-like, tail C padding stripped."""
    if not is_ndhwc_like(target_shape, target_format):
        raise RuntimeError(f"shape: {target_shape} of format {target_format} is not NDCHW-like.")
    if data.size == 0 and 0 in target_shape:
        # 空张量（零元素、逻辑 shape 含零维）无块可拼、无数据需转置
        return numpy.empty(target_shape, dtype=data.dtype)
    shd_shape = data.shape
    pad = 1 + (target_shape[target_format.index("C")] - 1) % shd_shape[-1]
    main_block = data[:, :, : shd_shape[2] - 1, :, :, :]
    tail_block = data[:, :, shd_shape[2] - 1, :, :, :pad]
    # NDC1HWC0 -> NDHWC1C0
    main_block = main_block.transpose((0, 1, 3, 4, 2, 5))
    # NDHWC1C0 -> NDHWC
    main_block = main_block.reshape(main_block.shape[:4] + (-1,))
    # concatenate
    ndhwc = numpy.concatenate((main_block, tail_block), axis=-1)
    if target_format != "NDHWC":
        return ndhwc.transpose(
            (
                "NDHWC".index(target_format[0]),
                "NDHWC".index(target_format[1]),
                "NDHWC".index(target_format[2]),
                "NDHWC".index(target_format[3]),
                "NDHWC".index(target_format[4]),
            )
        )
    return ndhwc


def from_fractal_nz(data, target_shape):
    """
    FRACTAL_NZ -> ND, tail M/N padding stripped:
    (A0, A1, A2, ..., An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    (A, M1, M0, N1, N0) -> (A, [0, M1-2], M0, [0, N1-2], N0) == (A, (M1-1), M0, (N1-1), N0)
                                                             -> (A, (M1-1) *M0, (N1-1) *N0)
                        -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0) == (A, 1, pad_m, (N1-1), N0)
                                                                    -> (A, pad_m, (N1-1) *N0)
                        -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1]) == (A, (M1-1), M0, 1, pad_n)
                                                                    -> (A, (M1-1) *M0, pad_n)
                        -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1]) == (A, 1, pad_m, 1, pad_n)
                                                                           -> (A, pad_m, pad_n)
    M = (M1-1) *M0 + pad_m
    N = (N1-1) *N0 + pad_n
    (A, (M1-1) *M0, (N1-1) *N0) + (A, pad_m, (N1-1) *N0) -> (A, (M1-1) *M0 + pad_m, (N1-1) *N0) -> (A, M, (N1-1) *N0)
    (A, (M1-1) *M0, pad_n) + (A, pad_m, pad_n) -> (A, (M1-1) *M0 + pad_m, pad_n) -> (A, M, pad_n)
    (A, M, (N1-1) *N0) + (A, M, pad_n) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, N)
    (A, M, N) -> (A0, A1, A2, ..., An, M, N)
    """
    if data.size == 0 and 0 in target_shape:
        # 空张量（零元素、逻辑 shape 含零维）无块可拼、无数据需转置
        return numpy.empty(target_shape, dtype=data.dtype)
    if len(data.shape) == 4:
        data = numpy.reshape(data, (1,) + data.shape)
    nd_shape = (1,) + tuple(target_shape)
    data_shape = data.shape
    m, n = nd_shape[-2:]
    N1, M1 = data_shape[-4:-2]
    M0, N0 = data_shape[-2:]
    pad_m = 1 + (m - 1) % M0
    pad_n = 1 + (n - 1) % N0
    # (A0, A1, A2, ... , An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    data = numpy.reshape(data, (numpy.prod(data_shape[:-4]),) + data_shape[-4:]).transpose((0, 2, 3, 1, 4))
    main_block = data[:, : M1 - 1, :, : N1 - 1, :]  # main block
    part_1 = data[:, M1 - 1, :pad_m, : N1 - 1, :]  # part 1
    part_2 = data[:, : M1 - 1, :, N1 - 1, :pad_n]  # part 2
    tail_block = data[:, M1 - 1, :pad_m, N1 - 1, :pad_n]  # tail_block
    # Reshape
    A = data.shape[0]
    main_block = numpy.reshape(main_block, (A, (M1 - 1) * M0, (N1 - 1) * N0))
    part_1 = numpy.reshape(part_1, (A, pad_m, (N1 - 1) * N0))
    part_2 = numpy.reshape(part_2, (A, (M1 - 1) * M0, pad_n))
    tail_block = numpy.reshape(tail_block, (A, pad_m, pad_n))
    # Concatenate
    main_concat_part1 = numpy.concatenate((main_block, part_1), axis=1)  # (A, M, (N1-1) *N0)
    part_2_concat_tail = numpy.concatenate((part_2, tail_block), axis=1)  # (A, M, pad_n)
    nd = numpy.concatenate((main_concat_part1, part_2_concat_tail), axis=-1)
    # Reshape
    nd = numpy.reshape(nd, data_shape[:-4] + (m, n))
    nd = numpy.reshape(nd, target_shape)

    return nd


def to_fractal_z(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    """NCHW-like conv weight (N=cout, C=cin) -> FRACTAL_Z, with optional groups."""
    data_shape = data.shape
    if not is_nchw_like(data_shape, ori_format):
        raise RuntimeError(f"shape: {data_shape} of format {ori_format} is not NCHW-like.")
    # data_format:NCHW or NHWC
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    if groups is None:
        groups = 1
    c_in = c * groups
    c_out = n
    c0 = resolve_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(c_in, c_out, groups, c0)
    G = group_dict["real_g"]
    ci_ori = group_dict["cin_ori"]
    co_ori = group_dict["cout_ori"]
    cin1_g = group_dict["cin1_g"]
    cou1_g = group_dict["cout1_g"]
    E = group_dict["mag_factor"]
    # Initialization
    out = numpy.zeros([G * cou1_g * BLOCK_SIZE, cin1_g * c0, h, w], dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"), ori_format.index("H"), ori_format.index("W")])
    for m in range(groups):
        for k in range(co_ori):
            for ci in range(ci_ori):
                i = m // E
                j = m % E
                out[i * E * co_ori + j * co_ori + k, j * ci_ori + ci, :, :] = data[
                    i * E * co_ori + j * co_ori + k, ci, :, :
                ]
    # nchw->FRACTAL_Z
    out = out.reshape((G, cou1_g * BLOCK_SIZE, cin1_g, c0, h, w)).transpose(0, 2, 4, 5, 1, 3)
    out = out.reshape(G * cin1_g * h * w, cou1_g, BLOCK_SIZE, c0)
    return out


def to_fractal_z_c04(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    """NCHW-like conv weight (C<=4) -> FRACTAL_Z_C04, GE format_transfer_fz_c04 semantics.

    Pad C to 4 and N to 16-multiple, flatten (H, W, C) to z = h*w*4 + c, pad z to
    c0-multiple, arrange as (ceil(4*h*w/c0), ceil(N/16), 16, c0). GE rejects C > 4
    and groups > 1 for this format.
    """
    data_shape = data.shape
    if not is_nchw_like(data_shape, ori_format):
        raise RuntimeError(f"shape: {data_shape} of format {ori_format} is not NCHW-like.")
    if groups is not None and groups not in (0, 1):
        raise RuntimeError(f"FRACTAL_Z_C04 supports groups=1 only, got groups={groups}.")
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    if not 0 < c <= 4:
        raise RuntimeError(f"FRACTAL_Z_C04 requires 0 < C <= 4, got C={c}.")
    c0 = resolve_c0(data.dtype.name, target_shape)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"), ori_format.index("H"), ori_format.index("W")])
    n1 = ceil_div(n, BLOCK_SIZE)
    padded = numpy.zeros((n1 * BLOCK_SIZE, 4, h, w), dtype=data.dtype)
    padded[:n, :c] = data
    mat = padded.transpose(0, 2, 3, 1).reshape(n1 * BLOCK_SIZE, h * w * 4)
    z1 = ceil_div(h * w * 4, c0)
    buf = numpy.zeros((n1 * BLOCK_SIZE, z1 * c0), dtype=data.dtype)
    buf[:, : h * w * 4] = mat
    return buf.reshape(n1, BLOCK_SIZE, z1, c0).transpose(2, 0, 1, 3)


def to_fractal_z_3d(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    """NCDHW/NDHWC 5D conv weight -> FRACTAL_Z_3D, with optional groups."""
    data_shape = data.shape
    # data_format: 'NCDHW' or 'NDHWC'
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    d = data_shape[ori_format.index("D")]
    if groups is None:
        groups = 1
    fmap_c = c * groups
    out_c = n
    c0 = resolve_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(fmap_c, out_c, groups, c0)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    mag_factor = group_dict["mag_factor"]
    cout1_g = group_dict["cout1_g"]
    weight_group = numpy.zeros((real_g, d, cin1_g, h, w, cout_g, c0), dtype=data.dtype)
    data = data.transpose(
        [
            ori_format.index("N"),
            ori_format.index("C"),
            ori_format.index("D"),
            ori_format.index("H"),
            ori_format.index("W"),
        ]
    )
    for g in range(groups):
        for ci in range(c):
            for co in range(n // groups):
                e = g % mag_factor
                dst_cin = e * c + ci
                dst_cout = e * (n // groups) + co
                src_cout = g * (n // groups) + co
                weight_group[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0] = data[
                    src_cout, ci, :, :, :
                ]
    weight_group = weight_group.reshape([real_g * d * cin1_g * h * w, cout1_g, BLOCK_SIZE, c0])
    return weight_group


def from_fractal_z_3d(
    data: numpy.ndarray, target_shape: Union[list, tuple] = None, target_format: str = "NCDHW", groups=None
):
    """FRACTAL_Z_3D -> NCDHW-like 5D weight, inverse of to_fractal_z_3d."""
    if groups is None:
        groups = 1
    n = target_shape[target_format.index("N")]
    c_in = target_shape[target_format.index("C")]
    d = target_shape[target_format.index("D")]
    h = target_shape[target_format.index("H")]
    w = target_shape[target_format.index("W")]
    c0 = data.shape[-1]
    n0 = BLOCK_SIZE
    cin_ori = c_in
    cout_ori = n // groups

    group_dict = _calculate_group(c_in * groups, n, groups, c0)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    mag_factor = group_dict["mag_factor"]
    cout1_g = group_dict["cout1_g"]

    data = numpy.ascontiguousarray(data)
    data_reshaped = data.reshape((real_g, d, cin1_g, h, w, cout1_g, n0, c0))

    result = numpy.zeros((n, c_in, d, h, w), dtype=data.dtype)
    for g in range(groups):
        for ci in range(cin_ori):
            for co in range(cout_ori):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                result[src_cout, ci, :, :, :] = data_reshaped[
                    g // mag_factor, :, dst_cin // c0, :, :, dst_cout // n0, dst_cout % n0, dst_cin % c0
                ]

    if target_format == "NDHWC":
        return result.transpose(0, 2, 3, 4, 1)
    if target_format == "DHWCN":
        return result.transpose(2, 3, 4, 1, 0)
    return result


def to_nc1hwc0(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None):
    """NCHW-like (rank<=4) -> NC1HWC0 with C0 zero padding."""
    ori_shape = data.shape
    if len(ori_shape) > 4:
        raise RuntimeError(
            "Please check original format and original shape: NC1HWC0 transformer doesn't support"
            f"{len(ori_shape)}D shape"
        )
    c0 = resolve_c0(data.dtype.name, target_shape)
    nchw, transpose_axis = _get_ndchw_dim_and_transpose_axis(data, ori_format)
    n, c, h, w = nchw
    c1 = ceil_div(c, c0)
    data = data.transpose(transpose_axis)
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = numpy.zeros((n, num_2_padding_in_cin, h, w), dtype=data.dtype)
    data = numpy.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, h, w)).transpose(0, 1, 3, 4, 2)
    return data


def to_ndc1hwc0(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None):
    """NDCHW-like (rank<=5) -> NDC1HWC0 with C0 zero padding."""
    ori_shape = data.shape
    if len(ori_shape) > 5:
        raise RuntimeError(
            "Please check original format and original shape: NDC1HWC0 transformer doesn't support"
            f"{len(ori_shape)}D shape"
        )
    c0 = resolve_c0(data.dtype.name, target_shape)
    ndchw, transpose_axis = _get_ndchw_dim_and_transpose_axis(data, ori_format, is_ndchw=True)
    n, d, c, h, w = ndchw
    c1 = ceil_div(c, c0)
    data = data.transpose(transpose_axis)
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = numpy.zeros((n, num_2_padding_in_cin, d, h, w), dtype=data.dtype)
    data = numpy.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, d, h, w)).transpose(0, 3, 1, 4, 5, 2)
    return data


def nd_to_fractal_nz(data: numpy.ndarray, target_shape: Union[list, tuple] = None):
    """Generic ND matrix (tail M, N) -> FRACTAL_NZ with zero padding."""
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    m0, n0 = 16, resolve_c0(data.dtype.name, target_shape)
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = numpy.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant")
    array_trans = transpose_axes(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def nd_to_fractal_z(data: numpy.ndarray, target_shape: Union[list, tuple] = None):
    """Generic ND matrix (tail M, N) -> FRACTAL_Z with zero padding."""
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    m0, n0 = resolve_c0(data.dtype.name, target_shape), 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = numpy.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant")
    array_trans = transpose_axes(len(data.shape) - 2, [0, 2, 3, 1])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def is_transformable(ori_format, target_format):
    """True if transform() supports ori_format -> target_format."""
    return ori_format in FORMAT_TRANSFORMS and target_format in FORMAT_TRANSFORMS[ori_format]


def transform(data, ori_format, target_format, target_shape: Union[list, tuple] = None, groups=None):
    """Dispatch ND <-> private-format transforms via FORMAT_TRANSFORMS; None if unsupported."""
    if not is_transformable(ori_format, target_format):
        return None
    transform_func = FORMAT_TRANSFORMS[ori_format][target_format]
    params = inspect.signature(transform_func).parameters
    kwargs = {}
    if "ori_format" in params:
        kwargs["ori_format"] = ori_format
    if "target_format" in params:
        kwargs["target_format"] = target_format
    if "target_shape" in params:
        kwargs["target_shape"] = target_shape
    if "groups" in params:
        kwargs["groups"] = groups
    return transform_func(data, **kwargs)


# ori_format -> {target_format: transform func}.
FORMAT_TRANSFORMS = {
    "NHWC": {"NC1HWC0": to_nc1hwc0, "FRACTAL_Z": to_fractal_z, "FRACTAL_Z_C04": to_fractal_z_c04},
    "NCHW": {"NC1HWC0": to_nc1hwc0, "FRACTAL_Z": to_fractal_z, "FRACTAL_Z_C04": to_fractal_z_c04},
    "HWCN": {"NC1HWC0": to_nc1hwc0, "FRACTAL_Z": to_fractal_z, "FRACTAL_Z_C04": to_fractal_z_c04},
    "NDHWC": {"NDC1HWC0": to_ndc1hwc0, "FRACTAL_Z_3D": to_fractal_z_3d},
    "NCDHW": {"NDC1HWC0": to_ndc1hwc0, "FRACTAL_Z_3D": to_fractal_z_3d},
    "DHWCN": {"NDC1HWC0": to_ndc1hwc0, "FRACTAL_Z_3D": to_fractal_z_3d},
    "ND": {"FRACTAL_NZ": nd_to_fractal_nz, "FRACTAL_Z": nd_to_fractal_z, "FRACTAL_ZN_RNN": nd_to_fractal_z},
    # revert
    "NC1HWC0": {
        "NHWC": from_nc1hwc0,
        "NCHW": from_nc1hwc0,
        "HWCN": from_nc1hwc0,
    },
    "NDC1HWC0": {
        "NDHWC": from_ndc1hwc0,
        "NCDHW": from_ndc1hwc0,
        "DHWCN": from_ndc1hwc0,
    },
    "FRACTAL_NZ": {
        "ND": from_fractal_nz,
    },
    "FRACTAL_Z_3D": {
        "NCDHW": from_fractal_z_3d,
        "NDHWC": from_fractal_z_3d,
        "DHWCN": from_fractal_z_3d,
    },
}
