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
output generation method for Universal testcases
"""

# Standard Packages
import contextlib
import gc
import logging
from collections.abc import Callable
from typing import List, Optional, Union

import numpy

# Third-party Packages
from ttk.switches import get_global_storage

from ...utilities import (
    DTYPE_PROMOTE_MAP,
    ceil_div,
    deep_flatten,
    get,
    get_dtype_range,
    load_array,
    nest_by_distribution,
    resolve_numpy_dtype,
    resolve_numpy_dtypes,
    shape_numel,
)
from ..common.invoke import (
    Invoke,
    accepts_var_keyword,
    golden_flatten,
    golden_mode,
    instantiate_and_call,
    sig_params,
)
from ..common.invoke import _framework_of as framework_of
from ..common.plugin import SpecResolver
from .info_keeper import OpInfoKeeper
from .testcase import TestcaseKernel

KERNEL_GOLDEN: dict = {
    "floor_div": numpy.floor_divide,
    "neg": numpy.negative,
    "acos": numpy.arccos,
    "acosh": numpy.arccosh,
    "asin": numpy.arcsin,
    "asinh": numpy.arcsinh,
    "atan": numpy.arctan,
    "atan2": numpy.arctan2,
    "atanh": numpy.arctanh,
    "assign_sub": numpy.subtract,
    "assign_add": numpy.add,
    "mod": numpy.fmod,
    "is_finite": numpy.isfinite,
    "is_nan": numpy.isnan,
    "is_inf": numpy.isinf,
    "is_pos_inf": numpy.isposinf,
    "is_neg_inf": numpy.isneginf,
    "is_close": numpy.isclose,
}


def __promote_dtype(context: TestcaseKernel):
    need_promote = [d in DTYPE_PROMOTE_MAP for d in context.flat_input_dtypes]
    if not any(need_promote):
        yield
        return

    # Backup
    bak_input_arrays = context.input_arrays
    bak_ori_input_arrays = context.original_input_arrays
    bak_input_dtypes = context.input_dtypes
    bak_output_dtypes = context.output_dtypes

    # --- Promote inputs: flat loop → astype → nested ---
    dist = context.input_distribution
    flat_arrays = list(deep_flatten(context.input_arrays))
    flat_ori_arrays = list(deep_flatten(context.original_input_arrays)) if context.original_input_arrays else None
    flat_dtypes = context.flat_input_dtypes
    new_flat_dtypes = list(flat_dtypes)

    for idx, array in enumerate(flat_arrays):
        if get(need_promote, idx) and array is not None:
            original_dtype = get(flat_dtypes, idx)
            promoted = DTYPE_PROMOTE_MAP[original_dtype]
            if original_dtype == "complex32":
                real, imag = numpy.split(array, 2, axis=-1)
                flat_arrays[idx] = (real + imag * 1j).reshape(array.shape[:-1])
            else:
                flat_arrays[idx] = array.astype(promoted)
            new_flat_dtypes[idx] = promoted
            # ori_arrays promote in sync
            if flat_ori_arrays is not None and idx < len(flat_ori_arrays) and flat_ori_arrays[idx] is not None:
                flat_ori_arrays[idx] = flat_ori_arrays[idx].astype(promoted)

    context.input_arrays = tuple(nest_by_distribution(flat_arrays, dist))
    if flat_ori_arrays is not None:
        context.original_input_arrays = tuple(nest_by_distribution(flat_ori_arrays, dist))
    context.input_dtypes = tuple(nest_by_distribution(new_flat_dtypes, dist))
    context.invalidate_flat_cache("input_arrays", "input_dtypes")

    # --- Promote output dtypes: flat → promote → nested ---
    # Skip _BIN_DTYPES (float4/float8_e8m0): they use binary_equal comparison
    # which requires exact dtype match. Promoting them to float32 would cause
    # cross-dtype rejection in BinaryComparison._cross_dtype_compare.
    from ..common.comparison.resolve import _BIN_DTYPES

    flat_out_dtypes = context.flat_output_dtypes
    new_flat_out_dtypes = [
        DTYPE_PROMOTE_MAP[d] if d in DTYPE_PROMOTE_MAP and d not in _BIN_DTYPES else d for d in flat_out_dtypes
    ]
    context.output_dtypes = tuple(nest_by_distribution(new_flat_out_dtypes, context.output_distribution))
    context.invalidate_flat_cache("output_dtypes")

    try:
        yield
    finally:
        # Restore. MUST be in `finally`: golden computation raises on invalid
        # testcases (e.g. an out-of-range attr default), and without this the
        # promoted dtypes stay on the context and leak into everything built
        # afterwards -- notably the GEIR graph, which then declares DT_DOUBLE
        # for a float32 testcase and gets rejected by the operator.
        context.input_arrays = bak_input_arrays
        context.original_input_arrays = bak_ori_input_arrays
        context.input_dtypes = bak_input_dtypes
        context.output_dtypes = bak_output_dtypes
        context.invalidate_flat_cache("input_arrays", "input_dtypes", "output_dtypes")

        del flat_arrays, flat_ori_arrays
        gc.collect()


@contextlib.contextmanager
def __golden_mode(mode: str, context: TestcaseKernel):
    if mode != "Promote":
        yield
    else:
        yield from __promote_dtype(context)


def __call_numpy_api(context: TestcaseKernel, golden_func, golden_parameters, output_dtypes: list):
    """KERNEL numpy API golden computation."""
    kwargs = __correct_kwargs_for_golden(context, golden_parameters)
    # auto convert to fp32 and invoke numpy function for bfloat16.
    # otherwise numpy function will consider data as float16 to compute which is wrong.
    # NOTE: this bf16→fp32 branch is ORTHOGONAL to __golden_mode/Promote — numpy
    # cannot compute bfloat16 natively, so we lift to fp32 even when golden_mode
    # != Promote. The Promote-context wrap is applied once at the outer
    # __invoke_golden dispatch (covers custom/class/torch/tf too).
    if any("bfloat16" in str(arr.dtype) for arr in context.flat_input_arrays):
        input_arrays = [
            arr if "bfloat16" not in str(arr.dtype) else arr.astype("float32") for arr in context.flat_input_arrays
        ]
        input_arrays = nest_by_distribution(input_arrays, context.input_distribution)
        results = golden_func(*input_arrays, **kwargs)
        results = golden_flatten(results, deep=True)
        results = [
            arr.astype(get(output_dtypes, i), copy=False)
            if (isinstance(arr, numpy.ndarray) and "bfloat16" in str(get(output_dtypes, i)))
            else arr
            for i, arr in enumerate(results)
        ]
        del input_arrays
    else:
        results = golden_func(*context.input_arrays, **kwargs)
    return results


def __call_custom_golden_func(context: TestcaseKernel, golden_func, golden_parameters):
    """
    customized & decouple function like:
    def xx_xx(input0, input1, *, attr0, attr1, **kargs):
        return output0, output1
    """
    kwargs = __collect_dynamic_golden_kwargs(context)
    if golden_parameters and not accepts_var_keyword(golden_parameters):
        kwargs = {k: v for k, v in kwargs.items() if k in golden_parameters}
    results = golden_func(*context.input_arrays, **kwargs)
    return results


def _torch_attrs_by_schema(func, attrs: dict) -> dict:
    """从 torch.ops.aten.* 的 OpOverload schema 取 attr 名过滤 attrs。
    schema 在 OpOverload(.default)上,不在 packet 上;非 aten/无 schema → {}。"""
    ovl = getattr(func, "default", func)
    sch = getattr(ovl, "_schema", None)
    if sch is None:
        return {}
    out = {}
    for a in sch.arguments:
        if str(a.type) == "Tensor":  # 输入(已 positional 喂),跳过
            continue
        if a.name in attrs:
            out[a.name] = attrs[a.name]
    return out


def __call_torch_api(context: TestcaseKernel, func):
    from ...utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

    t = [numpy_to_torch_tensor(a) if a is not None else None for a in context.flat_input_arrays]
    t = nest_by_distribution(t, context.input_distribution)
    kwargs = _torch_attrs_by_schema(func, context.attributes)
    results = golden_flatten(func(*t, **kwargs), deep=True)
    return [torch_to_numpy_tensor(r) if hasattr(r, "numpy") else r for r in results]


def __call_tf_api(context: TestcaseKernel, func):
    from ...utilities import normalize_from_tf_dtype, normalize_to_tf_dtype

    tf_in = [normalize_to_tf_dtype(a) if a is not None else None for a in context.flat_input_arrays]
    tf_in = nest_by_distribution(tf_in, context.input_distribution)
    results = golden_flatten(func(*tf_in), deep=True)  # tf 无 schema 等价,attrs 暂不传
    out = []
    for r in results:
        n = r.numpy() if hasattr(r, "numpy") else r
        out.append(normalize_from_tf_dtype(n))
    return out


def __invoke_class(context: TestcaseKernel, cls):
    op_info = OpInfoKeeper().info_of(context.op_name)
    inputs_named = {}
    if op_info:
        for inp, arr in zip(op_info["inputs"], context.flat_input_arrays):
            inputs_named[inp["name"]] = arr
    pool = {**inputs_named, **(context.attributes or {})}  # attributes 正常为 dict,or {} 防 None
    return instantiate_and_call(cls, pool)


def __invoke_golden(context: TestcaseKernel, golden_func, output_dtypes):
    """通路执行闭包（str 已由 Invoke.golden 解析）：Promote 包裹 + class/numpy/torch/tf/custom 分发。

    Single Promote-context wrap around the ENTIRE dispatch (class / numpy /
    torch / tf / custom) — previously only the numpy builtin + class paths
    were guarded, so under golden_mode=Promote the custom / torch / tf goldens
    ran on un-promoted bfloat16/float16 inputs (inaccurate "true value").
    """
    with __golden_mode(golden_mode(context, get_global_storage()), context):
        if isinstance(golden_func, type):
            return __invoke_class(context, golden_func)
        fw = framework_of(golden_func)
        if fw == "numpy":
            params = sig_params(golden_func)
            return __call_numpy_api(context, golden_func, params, output_dtypes)
        if fw == "torch":
            return __call_torch_api(context, golden_func)
        if fw == "tf":
            return __call_tf_api(context, golden_func)
        return __call_custom_golden_func(context, golden_func, sig_params(golden_func))


def __builtin_golden_fallback(op_name):
    """kernel 回退链：KERNEL_GOLDEN 映射 → numpy 同名属性（SpecResolver fallback 注入）。"""
    golden_func = KERNEL_GOLDEN.get(op_name)
    if golden_func is not None:
        return golden_func
    if op_name in numpy.__dir__() and isinstance(getattr(numpy, op_name), Callable):
        return getattr(numpy, op_name)
    return None


def __generate_golden(context: TestcaseKernel, output_dtypes: list) -> list:
    switches = get_global_storage()
    golden_func = SpecResolver(context.op_name, "kernel", switches.plugin_path).resolve_golden(
        lambda: __builtin_golden_fallback(context.op_name)
    )
    if golden_func is None:
        logging.warning(f"Golden function for operator {context.op_name} is not provided!")
    return golden_flatten(
        Invoke.golden(context, golden_func, switches, lambda func: __invoke_golden(context, func, output_dtypes)),
        deep=True,
    )


def __load_golden_from_file(fp: str, dtype: str, shape: Optional[Union[list, tuple]]):
    if shape is None:
        return None
    return load_array(fp, dtype, shape)


def gen_output(context: TestcaseKernel, stored_goldens=None):
    # Enable tensorflow numpy bfloat16 support
    output_dtypes = resolve_numpy_dtypes(context.flat_output_dtypes)
    # Remember: golden_arrays is always flatten.
    golden_arrays = __select_golden_source(context, output_dtypes, stored_goldens)
    if len(golden_arrays) > len(context.flat_output_shapes):
        golden_arrays = golden_arrays[: len(context.flat_output_shapes)]
    __normalize_goldens(golden_arrays, output_dtypes)
    output_arrays = __alloc_output_buffers(context, golden_arrays, output_dtypes)
    n_shape_slots = context.shape_unknown_slot_count
    __append_out_shape_unknown_golden(context, golden_arrays, output_arrays, extend_fields=not n_shape_slots)
    context.golden_arrays = golden_arrays
    context.output_arrays = tuple(output_arrays)


def __select_golden_source(context: TestcaseKernel, output_dtypes: list, stored_goldens) -> list:
    """golden 来源四选一：manual-data 复跑装载 > Disable 占位 > 手工 bin > 现算。"""
    switches = get_global_storage()
    if stored_goldens is not None:
        logging.info("Using manual-data replay golden data")
        return list(stored_goldens)
    if golden_mode(context, switches) == "Disable":
        return ["SUPPRESSED"]
    if context.manual_golden_binaries:
        logging.info("Using manually configured output data")
        flat_binaries = context.flat_manual_golden_binaries
        flat_shapes = context.flat_output_shapes
        return [
            __load_golden_from_file(get(flat_binaries, i), get(output_dtypes, i), get(flat_shapes, i))
            for i in range(len(flat_shapes))
        ]
    return __generate_golden(context, output_dtypes)


def __alloc_output_buffers(context: TestcaseKernel, golden_arrays: list, output_dtypes: list) -> list:
    """被测 kernel 输出缓冲构造：dtype 打包调整（complex32/uint1）→ golden shape
    采纳（不一致警告后替换）→ inplace 位走索引、其余 numpy.ones 实配。

    附带副作用：golden_arrays[idx] 就地替换为 nan_to_num 后的值（out_dtype
    打包结果在此算出，拆开会重复推导）。
    """
    output_arrays = []
    # Regenerating golden on a testcase whose output fields were already
    # extended (GEIR regenerates per mode: const/dynamic) must walk only the
    # declared outputs — shape slots carry no independent golden and would
    # trip __output_shape_needs_golden.
    n_shape_slots = context.shape_unknown_slot_count
    declared_shapes = context.flat_output_shapes[: len(context.flat_output_shapes) - n_shape_slots]
    for idx, output_shape in enumerate(declared_shapes):
        # alloc_shape is the shape the output buffer is actually allocated
        # with: the declared shape adjusted for dtype packing (complex32/uint1),
        # then replaced by the golden shape. Kept separate from the loop
        # variable since it is rebound below.
        alloc_shape = output_shape
        out_dtype = get(output_dtypes, idx)
        if "complex32" in str(out_dtype):
            out_dtype = "float16"
            alloc_shape = list(output_shape) + [2]
        elif str(out_dtype) == "uint1":
            out_dtype = "uint8"
            alloc_shape = [ceil_div(shape_numel(output_shape), 8)]
        golden_shape = None
        if idx < len(golden_arrays) and isinstance(golden_arrays[idx], numpy.ndarray):
            golden_shape = golden_arrays[idx].shape
            golden_arrays[idx] = __nan_to_num(golden_arrays[idx], context.op_name, out_dtype)
        # Enable inplace
        if idx < len(context.output_inplace_indexes) and context.output_inplace_indexes[idx] is not None:
            output_arrays.append(context.output_inplace_indexes[idx])
        else:
            if golden_shape is not None:
                if alloc_shape and tuple(alloc_shape) != tuple(golden_shape):
                    logging.warning(
                        f"Golden shape {golden_shape} not match with testcase shape {alloc_shape}, replacing..."
                    )
                alloc_shape = golden_shape
            else:
                __output_shape_needs_golden(context, idx)
            output_arrays.append(numpy.ones(alloc_shape, dtype=out_dtype))
    return output_arrays


def __normalize_goldens(golden_arrays: List[Union[str, numpy.ndarray]], output_dtypes: list):
    # golden is rebound after view/concat conversions and written back to
    # golden_arrays — iterate by index so the rebinding stays explicit.
    for idx in range(len(golden_arrays)):
        golden = golden_arrays[idx]
        if not isinstance(golden, numpy.ndarray):
            continue
        npu_dtype = get(output_dtypes, idx)
        if "complex32" in str(npu_dtype):
            npu_dtype = "float16"
        if str(npu_dtype) == "uint1":
            npu_dtype = "uint8"
        npu_dtype = numpy.dtype(npu_dtype).name
        if golden.dtype.name == "bfloat16":
            ttk_bf16 = resolve_numpy_dtype("bfloat16")
            if golden.dtype.type != ttk_bf16.dtype.type:
                golden = golden.view(ttk_bf16)
                golden_arrays[idx] = golden
        if golden.dtype.name == npu_dtype or npu_dtype not in DTYPE_PROMOTE_MAP:
            continue
        if "complex32" in str(get(output_dtypes, idx)):
            # complex32 is a bit complicated:
            # real/imag part needs to be extracted when dtype not equal (to float16)
            # it will be complex64 or complex128
            real, imag = golden.real.reshape(golden.real.shape + (1,)), golden.imag.reshape(golden.imag.shape + (1,))
            golden = numpy.concatenate((real, imag), axis=-1)
            golden_arrays[idx] = golden
        dr = get_dtype_range(npu_dtype)
        if get_global_storage().overflow_mode == 0:
            numpy.clip(golden, a_min=dr[0], a_max=dr[1], out=golden)
        else:
            golden[golden < dr[0]] = -numpy.inf
            golden[golden > dr[1]] = numpy.inf


# 饱和白名单（仅 overflow_mode=0 休眠通路消费；生产 overflow_mode 恒 1，见
# switches.py）：饱和语义下硬件真正会饱和、golden 需清洗 NaN/Inf 的 SoC。
# 值 = 不饱和（恒产出 INF/NAN）的 dtype 例外——观测事实只有一条：新 SoC 的
# 饱和开关对 fp32 不生效；未观测的 dtype 维持原判定（饱和）。
# 未收录 SoC 一律恒 INF/NAN——后续新芯片即此行为，无需登记；仅确认某芯片
# 有饱和能力时才补登。
_SATURATE_ALL = ()
_FP32_IEEE = ("float32",)
SATURATING_SOC = {
    # 老 SoC 封闭集（PLATFORM_BEFORE_DAVID，310P 除外——其全 dtype 恒 INF/NAN）
    "Ascend910": _SATURATE_ALL,
    "Ascend310": _SATURATE_ALL,
    "Ascend310B": _SATURATE_ALL,
    "Ascend610": _SATURATE_ALL,
    "Ascend610Lite": _SATURATE_ALL,
    "BS9SX1A": _SATURATE_ALL,
    "Hi3796CV300CS": _SATURATE_ALL,
    "Hi3796CV300ES": _SATURATE_ALL,
    "OPTG": _SATURATE_ALL,
    "SD3403": _SATURATE_ALL,
    "TsnsC": _SATURATE_ALL,
    "TsnsE": _SATURATE_ALL,
    # 新 SoC：仅 fp32 例外（开关不生效，恒 IEEE）
    "Ascend910B": _FP32_IEEE,
    "Ascend910_93": _FP32_IEEE,
    "Ascend031": _FP32_IEEE,
    "MC62CM12A": _FP32_IEEE,
    "Ascend950": _FP32_IEEE,
}


def __soc_saturates(short_soc: str, dtype_name: str) -> bool:
    """饱和语义下 (SoC, dtype) 硬件是否饱和；未收录 SoC 恒 INF/NAN（新芯片默认）。"""
    exceptions = SATURATING_SOC.get(short_soc)
    return exceptions is not None and dtype_name not in exceptions


def __nan_to_num(golden_array: numpy.ndarray, op: str, npu_dtype):
    DMA_COPY_OPS = (
        "as_strided",
        "batch_to_space",
        "batch_to_space_nd",
        "broadcast_to",
        "concat",
        "depth_to_space",
        "diag_v2",
        "diag_flat",
        "flatten",
        "gather",
        "gather_nd",
        "gather_v2",
        "inv",
        "im2col",
        "masked_scatter",
        "mirror_pad",
        "moe_init_routing",
        "moe_init_routing_v2",
        "pack",
        "reverse_v2",
        "resize_nearest_neighbor_v2",
        "scatter",
        "scatter_elements",
        "scatter_nd",
        "scatter_update",
        "scatter_nd_update",
        "sign",
        "sign_bits_pack",
        "sign_bits_unpack",
        "slice",
        "space_to_batch",
        "space_to_batch_nd",
        "space_to_depth",
        "split",
        "split_v",
        "strided_slice",
        "strided_slice_v3",
        "transpose",
        "trans_data",
        "tril",
        "triu",
        "view_copy",
        "unpack",
    )
    if get_global_storage().overflow_mode != 0:  # 1: INF/NAN mode
        return golden_array
    if "float" not in str(npu_dtype):
        return golden_array
    if op in DMA_COPY_OPS or (op.endswith("_d") and op[:-2] in DMA_COPY_OPS):
        return golden_array
    npu_dtype = numpy.dtype(npu_dtype).name
    if not __soc_saturates(get_global_storage().short_soc_version, npu_dtype):
        # 恒产出 INF/NAN：310P 全 dtype / 新 SoC fp32 / 未收录芯片（新芯片默认）
        return golden_array
    if npu_dtype == "bfloat16":
        dr = get_dtype_range("bfloat16")
        tmp = numpy.array([0, dr[1], dr[0], numpy.inf, -numpy.inf], dtype=resolve_numpy_dtype("bfloat16"))
        nan = numpy.isnan(golden_array)
        if numpy.any(nan):
            golden_array = numpy.where(nan, tmp[0], golden_array)
        golden_array[golden_array == tmp[3]] = tmp[1]
        golden_array[golden_array == tmp[4]] = tmp[2]
    elif numpy.any(numpy.isnan(golden_array)) or numpy.any(numpy.isinf(golden_array)):
        numpy.nan_to_num(golden_array, copy=False)
    return golden_array


def __correct_kwargs_for_golden(context: TestcaseKernel, golden_parameters: list):
    attributes = context.attributes.copy()
    # Remove input names — already passed positionally via *context.input_arrays
    op_info = OpInfoKeeper().info_of(context.op_name)
    if op_info:
        for inp in op_info["inputs"]:
            attributes.pop(inp["name"], None)
    for key in tuple(attributes.keys()):
        if key not in golden_parameters:
            del attributes[key]
    return attributes


def __collect_dynamic_golden_kwargs(context: TestcaseKernel):
    switches = get_global_storage()
    kwargs = context.attributes.copy()
    # delete internal attributes
    keys = list(kwargs.keys())
    for k in keys:
        if str(k)[0] in ("!", "#", "@"):
            del kwargs[k]
    # delete const inputs in attributes
    op_info = OpInfoKeeper().info_of(context.op_name)
    if op_info:
        for ipt in op_info["inputs"]:
            kwargs.pop(ipt["name"], None)
    # add some additional information.
    kwargs.update(
        {
            "full_soc_version": switches.dev_plat,
            "short_soc_version": switches.short_soc_version,
            "testcase_name": context.testcase_name,
            "input_ori_shapes": context.input_ori_shapes,
            "output_ori_shapes": context.output_ori_shapes,
            "input_formats": context.input_formats,
            "output_formats": context.output_formats,
            "input_ori_formats": context.input_ori_formats,
            "output_ori_formats": context.output_ori_formats,
            "input_dtypes": context.input_dtypes,
            "output_dtypes": context.output_dtypes,
            # 【为何要做】golden 收到的输入精度是框架定的：三方(cross_check)场景会走
            # golden_mode=Promote，按 DTYPE_PROMOTE_MAP 把各输入抬一档(fp16/bf16→fp32、
            # fp32→fp64)，好让标杆比被测更准；非该场景则用算子声明的原精度。但这个决定
            # 此前不下发，golden 无从知道自己拿到的是"已抬过的"还是"原生的"——两种情形下
            # 收到 fp32 完全无法区分，作者只能猜，猜错就会写出把框架抬上去的精度又砍回来
            # 的代码(如硬编码 astype(float32))，或反过来在原生场景下算出算子不会产出的
            # 中间量。
            # 【实现逻辑】mode 在 __invoke_golden 里已解析好(逐用例 override 优先于全局
            # --golden-mode)，这里原样透出，不新增任何判定。
            # 【实现效果】golden 可按 kwargs['golden_mode'] 明确分支：Promote 下透传不做
            # 任何 cast，其余情形自行处理低精度类型。未使用该键的存量 golden 不受影响——
            # kwargs 按 golden 签名裁剪，带 **kwargs 的自动收到、没带的自动丢弃。
            "golden_mode": golden_mode(context, switches),
        }
    )
    return kwargs


class GoldenShapeRequiredError(RuntimeError):
    """shape-unknown 输出缺 golden：输出 buffer 分配依赖 golden 现算 shape。"""


def __output_shape_needs_golden(context: TestcaseKernel, output_idx: int):
    if context.output_shape_unknown_indexes:
        raise GoldenShapeRequiredError(
            f"Golden must be supplied for {output_idx}th (count from 0) "
            f"output of {context.op_name} "
            f"since its shape depends on the value of input tensors."
        )


def __append_out_shape_unknown_golden(
    context: TestcaseKernel, golden_arrays: list, output_arrays: list, extend_fields: bool = True
):
    """append output shape tensor to golden & output_arrays

    Args:
        extend_fields: also grow the testcase output fields via
            append_output_metadata. Callers regenerating golden on an
            already-extended testcase (GEIR per-mode regeneration) pass False —
            the metadata golden itself is still appended (fresh list every
            call), only the persistent field growth must not repeat.
    """
    if not context.output_shape_unknown_indexes:
        return
    output_shape_golden = []
    for idx in sorted(context.output_shape_unknown_indexes):
        golden_shape = golden_arrays[idx].shape
        shape_len = len(golden_shape)
        if shape_len > 8:
            raise RuntimeError(f"At most 8 axes are supported for un-inferable output shape. But got {golden_shape}")
        # `1` is the initial value for output.
        tmp_array = numpy.ones([9], dtype=numpy.uint64)
        # set the effective number count.
        tmp_array[0] = shape_len
        tmp_array[1 : shape_len + 1] = golden_shape
        output_shape_golden.extend(list(tmp_array))
    output_shape_golden_array = numpy.array(output_shape_golden, dtype="uint64")
    # mark it as uint64 encoding. set uint64 low32 bit highest bit as 1.
    # SE says only the first one needs to be set.
    output_shape_golden_array.view(numpy.uint8)[3] = 128
    output_arrays.append(numpy.ones(output_shape_golden_array.shape, dtype="uint64"))
    golden_arrays.append(output_shape_golden_array)
    if extend_fields:
        context.append_output_metadata("uint64", "ND", output_shape_golden_array.shape)
