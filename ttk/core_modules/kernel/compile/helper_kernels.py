#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
helper_kernels
"""

import numpy
import tbe.common.platform as tbe_platform
from tbe import tik


def clear_ub(ipt: dict, full_soc_version: str, core_type: str, kernel_name: str = "clear_ub"):
    """
    clear UB data
    """
    tbe_platform.set_current_compile_soc_info(full_soc_version, core_type)
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    dtype = ipt["dtype"]
    dtype_bytes = numpy.dtype(dtype).itemsize
    input_gm = tik_instance.Tensor(dtype, (32 // dtype_bytes,), name="input_gm", scope=tik.scope_gm)
    src_ub = tik_instance.Tensor(dtype, (ub_size // dtype_bytes,), name="src_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(dst=src_ub, src=input_gm, sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
    clean_val_scalar = tik_instance.Scalar(dtype=dtype, name="clean_val_scalar", init_value=src_ub[0])

    with tik_instance.for_range(0, core_num, block_num=core_num):
        _clear_ub_each_core(tik_instance, src_ub, clean_val_scalar, dtype_bytes)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_gm,), outputs=())
    return tik_instance


def _clear_ub_each_core(tik_instance: tik.Tik, ub_to_clear: tik.Tensor, clear_value, dtype_bytes):
    block_elements = 32 // dtype_bytes
    mask_max = block_elements * 8
    repeat = ub_to_clear.size // mask_max
    repeat_tail = ub_to_clear.size % mask_max
    offset = 0
    while repeat > 255:
        tik_instance.vec_dup(
            mask=mask_max, dst=ub_to_clear[offset], scalar=clear_value, repeat_times=255, dst_rep_stride=8
        )
        repeat = repeat - 255
        offset = offset + mask_max * 255
    if repeat > 0:
        tik_instance.vec_dup(
            mask=mask_max, dst=ub_to_clear[offset], scalar=clear_value, repeat_times=repeat, dst_rep_stride=8
        )
        offset = offset + mask_max * repeat
    if repeat_tail > 0:
        tik_instance.vec_dup(
            mask=repeat_tail, dst=ub_to_clear[offset], scalar=clear_value, repeat_times=1, dst_rep_stride=8
        )


def clear_l1(ipt: dict, full_soc_version: str, kernel_name: str = "clear_l1"):
    """
    clear L1 data
    """
    tbe_platform.set_current_compile_soc_info(full_soc_version, "AiCore")
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")

    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    dtype = ipt["dtype"]
    dtype_bytes = numpy.dtype(dtype).itemsize
    # input will be always fixed as (128, 1024) bytes
    input_bytes = 128 * 1024
    input_gm = tik_instance.Tensor(dtype, (input_bytes // dtype_bytes,), name="input_gm", scope=tik.scope_gm)
    dst_l1 = tik_instance.Tensor(dtype, (l1_size // dtype_bytes,), name="dst_l1", scope=tik.scope_cbuf)

    with tik_instance.for_range(0, core_num, block_num=core_num):
        repeat = l1_size // input_bytes
        for idx in range(repeat):
            tik_instance.data_move(
                dst=dst_l1[idx * input_bytes // dtype_bytes],
                src=input_gm,
                sid=0,
                nburst=1,
                burst=input_bytes // 32,
                src_stride=0,
                dst_stride=0,
            )
        repeat_tail = l1_size % input_bytes
        tail_burst = repeat_tail // 32
        if repeat_tail > 0 and tail_burst > 0:
            tik_instance.data_move(
                dst=dst_l1[repeat * input_bytes // dtype_bytes],
                src=input_gm,
                sid=0,
                nburst=1,
                burst=tail_burst,
                src_stride=0,
                dst_stride=0,
            )

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_gm,), outputs=())
    return tik_instance


def clear_l0(ipt: dict, full_soc_version: str, kernel_name: str = "clear_l0"):
    """
    clear L0A/L0B/L0C data via two-step matmul (GM->L1->L0A/L0B->L0C).
    matmul internally uses mmad instruction to load L1->L0A/L0B and write L0C.
    Step1 fills L0A fully and writes L0C; Step2 fills L0B fully and overwrites L0C.
    On Ascend950, matmul uses f16f16f16 (L0C as float16) with api_check_support
    patched because tik_api_map lacks the 950 entry.
    On Ascend910B, matmul uses f16f16f32 (L0C as float32).
    """
    import tbe.common.platform.platform_info as _pi

    _orig_check = _pi.api_check_support

    def _patched_check(name, dtype_str):
        if name == "tik.matmul" and dtype_str == "f16f16f16":
            return True
        if name == "tik.load2dv2" and dtype_str == "float16":
            return True
        return _orig_check(name, dtype_str)

    _pi.api_check_support = _patched_check
    import tbe.tik.api.cube.matmul as _matmul_mod

    _matmul_mod.api_check_support = _patched_check
    import tbe.tik.tik_lib.tik_mmad_convert_api.tik_mmad_convert_operation as _load2d_mod

    _load2d_mod.api_check_support = _patched_check

    tbe_platform.set_current_compile_soc_info(full_soc_version, "AiCore")
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")

    l0a_size = tbe_platform.get_soc_spec(tbe_platform.L0A_SIZE)
    l0b_size = tbe_platform.get_soc_spec(tbe_platform.L0B_SIZE)
    dtype = "float16"
    dtype_bytes = numpy.dtype(dtype).itemsize

    short_soc = full_soc_version
    is_950 = "950" in short_soc or "Ascend950" in short_soc
    l0c_dtype = dtype if is_950 else "float32"

    matrix_k = 16
    matrix_m = l0a_size // (matrix_k * dtype_bytes)
    matrix_n = l0b_size // (matrix_k * dtype_bytes)

    input_bytes = 128 * 1024
    input_gm = tik_instance.Tensor(dtype, (input_bytes // dtype_bytes,), name="input_gm", scope=tik.scope_gm)

    src_a1 = tik_instance.Tensor(dtype, (matrix_m, matrix_k), name="src_a1", scope=tik.scope_cbuf)
    src_b1 = tik_instance.Tensor(dtype, (matrix_k, matrix_k), name="src_b1", scope=tik.scope_cbuf)
    src_a2 = tik_instance.Tensor(dtype, (matrix_k, matrix_k), name="src_a2", scope=tik.scope_cbuf)
    src_b2 = tik_instance.Tensor(dtype, (matrix_k, matrix_n), name="src_b2", scope=tik.scope_cbuf)
    dst_l0c = tik_instance.Tensor(l0c_dtype, (matrix_m, matrix_k), name="dst_l0c", scope=tik.scope_cc)

    a1_burst = matrix_m * matrix_k * dtype_bytes // 32
    b2_burst = matrix_k * matrix_n * dtype_bytes // 32
    small_burst = matrix_k * matrix_k * dtype_bytes // 32

    with tik_instance.for_range(0, core_num, block_num=core_num):
        tik_instance.data_move(dst=src_a1, src=input_gm, sid=0, nburst=1, burst=a1_burst, src_stride=0, dst_stride=0)
        tik_instance.data_move(dst=src_b1, src=input_gm, sid=0, nburst=1, burst=small_burst, src_stride=0, dst_stride=0)
        tik_instance.matmul(dst=dst_l0c, a=src_a1, b=src_b1, m=matrix_m, k=matrix_k, n=matrix_k, init_l1out=True)

        tik_instance.data_move(dst=src_a2, src=input_gm, sid=0, nburst=1, burst=small_burst, src_stride=0, dst_stride=0)
        tik_instance.data_move(dst=src_b2, src=input_gm, sid=0, nburst=1, burst=b2_burst, src_stride=0, dst_stride=0)
        tik_instance.matmul(dst=dst_l0c, a=src_a2, b=src_b2, m=matrix_k, k=matrix_k, n=matrix_n, init_l1out=True)

    try:
        tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_gm,), outputs=())
    finally:
        _pi.api_check_support = _orig_check
        _matmul_mod.api_check_support = _orig_check
        _load2d_mod.api_check_support = _orig_check

    return tik_instance


def warmup(full_soc_version: str, kernel_name: str = "warmup"):
    """
    a kernel without any action, but warmup all the cores, like: smmu/biu/tlb,
    to avoid cycles which is not belong to the kernel going to test.
    """
    tbe_platform.set_current_compile_soc_info(full_soc_version, "AiCore")
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")

    with tik_instance.for_range(0, core_num, block_num=core_num):
        tik_instance.tik_return()
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(), outputs=())
    return tik_instance
