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
import inspect
import logging
from collections.abc import Callable
from typing import List, Optional, Sequence, Union

import numpy

# Third-party Packages
from ....utilities import (
    DTYPE_PROMOTE_MAP,
    bind_by_name,
    ceil_div,
    framework_of,
    get,
    get_dtype_range,
    get_global_storage,
    input_apply_as_list,
    load_numpy_data,
    numpy_bfloat16,
    resolve_callable_str,
    resolve_custom_numpy_dtypes,
    shape_product,
)
from ...operator.op_info_keeper import OpInfoKeeper
from ...plugin_loader import get_plugin_function
from ...testcase_manager import TestcaseOp

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


def __promote_dtype(context: TestcaseOp):
    from ....utilities.container_utils import deep_flatten

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

    context.input_arrays = tuple(input_apply_as_list(flat_arrays, dist))
    if flat_ori_arrays is not None:
        context.original_input_arrays = tuple(input_apply_as_list(flat_ori_arrays, dist))
    context.input_dtypes = tuple(input_apply_as_list(new_flat_dtypes, dist))
    context.invalidate_flat_cache("input_arrays", "input_dtypes")

    # --- Promote output dtypes: flat → promote → nested ---
    # Skip _BIN_DTYPES (float4/float8_e8m0): they use binary_equal comparison
    # which requires exact dtype match. Promoting them to float32 would cause
    # cross-dtype rejection in BinaryComparison._cross_dtype_compare.
    from ...comparison.resolve import _BIN_DTYPES

    flat_out_dtypes = context.flat_output_dtypes
    new_flat_out_dtypes = [
        DTYPE_PROMOTE_MAP[d] if d in DTYPE_PROMOTE_MAP and d not in _BIN_DTYPES else d for d in flat_out_dtypes
    ]
    context.output_dtypes = tuple(input_apply_as_list(new_flat_out_dtypes, context.output_distribution))
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
def __golden_mode(mode: str, context: TestcaseOp):
    if mode != "Promote":
        yield
    else:
        yield from __promote_dtype(context)


def __call_numpy_api(context: TestcaseOp, golden_func, golden_parameters, output_dtypes: list):
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
        input_arrays = input_apply_as_list(input_arrays, context.input_distribution)
        results = golden_func(*input_arrays, **kwargs)
        results = __golden_flatten(results)
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


def __call_custom_golden_func(context: TestcaseOp, golden_func, golden_parameters):
    """
    customized & decouple function like:
    def xx_xx(input0, input1, *, attr0, attr1, **kargs):
        return output0, output1
    """
    kwargs = __collect_dynamic_golden_kwargs(context)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in golden_parameters.values())
    if golden_parameters and not has_var_kw:
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


def __call_torch_api(context: TestcaseOp, func):
    from ....utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

    t = [numpy_to_torch_tensor(a) if a is not None else None for a in context.flat_input_arrays]
    t = input_apply_as_list(t, context.input_distribution)
    kwargs = _torch_attrs_by_schema(func, context.attributes)
    results = __golden_flatten(func(*t, **kwargs))
    return [torch_to_numpy_tensor(r) if hasattr(r, "numpy") else r for r in results]


def __call_tf_api(context: TestcaseOp, func):
    from ....utilities import normalize_to_tf_dtype, tf_dtype_revert

    tf_in = [normalize_to_tf_dtype(a) if a is not None else None for a in context.flat_input_arrays]
    tf_in = input_apply_as_list(tf_in, context.input_distribution)
    results = __golden_flatten(func(*tf_in))  # tf 无 schema 等价,attrs 暂不传
    out = []
    for r in results:
        n = r.numpy() if hasattr(r, "numpy") else r
        out.append(tf_dtype_revert(n))
    return out


def __invoke_class(context: TestcaseOp, cls, output_dtypes):
    op_info = OpInfoKeeper().info_of(context.op_name)
    inputs_named = {}
    if op_info:
        for inp, arr in zip(op_info["inputs"], context.flat_input_arrays):
            inputs_named[inp["name"]] = arr
    pool = {**inputs_named, **(context.attributes or {})}  # attributes 正常为 dict,or {} 防 None
    if cls.__init__ is object.__init__:
        inst = cls()
    else:
        ia, ik = bind_by_name(cls.__init__, pool)
        inst = cls(*ia, **ik)
    ca, ck = bind_by_name(inst.__call__, pool)
    results = inst(*ca, **ck)
    return results


def __invoke_golden(context: TestcaseOp, golden_func, output_dtypes):
    if isinstance(golden_func, str):
        golden_func = resolve_callable_str(golden_func)
    # Single Promote-context wrap around the ENTIRE dispatch (class / numpy /
    # torch / tf / custom) — previously only the numpy builtin + class paths
    # were guarded, so under golden_mode=Promote the custom / torch / tf goldens
    # ran on un-promoted bfloat16/float16 inputs (inaccurate "true value").
    mode = getattr(context, "golden_mode_override", None) or get_global_storage().golden_mode
    with __golden_mode(mode, context):
        if isinstance(golden_func, type):
            return __invoke_class(context, golden_func, output_dtypes)
        fw = framework_of(golden_func)
        if fw == "numpy":
            params = __sig_params(golden_func)
            return __call_numpy_api(context, golden_func, params, output_dtypes)
        if fw == "torch":
            return __call_torch_api(context, golden_func)
        if fw == "tf":
            return __call_tf_api(context, golden_func)
        return __call_custom_golden_func(context, golden_func, __sig_params(golden_func))


def __sig_params(func):
    try:
        return inspect.signature(func).parameters
    except Exception:
        return {}  # 与正常值(mappingproxy,dict-like)同类型:异常返回 {} 非 [],in/.values()/.items() 统一可用


def __generate_golden(context: TestcaseOp, output_dtypes: list) -> list:
    switches = get_global_storage()
    golden_func = get_plugin_function(context.op_name, "golden", "kernel", switches.plugin_path)
    if golden_func is None:
        golden_func = KERNEL_GOLDEN.get(context.op_name)
    if (
        golden_func is None
        and context.op_name in numpy.__dir__()
        and isinstance(getattr(numpy, context.op_name), Callable)
    ):
        golden_func = getattr(numpy, context.op_name)
    if golden_func:
        # noinspection PyBroadException
        try:
            golden_results = __invoke_golden(context, golden_func, output_dtypes)
        except TimeoutError:
            logging.exception("Golden generation timeout")
            golden_results = ["GOLDEN_TIMEOUT"]
        except Exception as e:
            logging.exception(f"Golden generation failure: {e}")
            golden_results = ["GOLDEN_FAILURE"]
        finally:
            gc.collect()
        golden_arrays = __golden_flatten(golden_results)
    else:
        logging.warning(f"Golden function for operator {context.op_name} is not provided!")
        golden_arrays = ["UNSUPPORTED"]
    return golden_arrays


def __load_golden_from_file(fp: str, dtype: str, shape: Optional[Union[list, tuple]]):
    if shape is None:
        return None
    return load_numpy_data(fp, dtype, shape)


def __gen_output(context: TestcaseOp, stored_goldens=None):
    # Enable tensorflow numpy bfloat16 support
    output_dtypes = resolve_custom_numpy_dtypes(context.flat_output_dtypes)
    switches = get_global_storage()
    output_arrays = []

    if stored_goldens is not None:
        logging.info("Using prepared Kernel golden data")
        golden_arrays = list(stored_goldens)
    elif switches.golden_mode == "Disable":
        golden_arrays = ["SUPPRESSED"]
    elif context.manual_golden_binaries:
        logging.info("Using manually configured output data")
        flat_binaries = context.flat_manual_golden_binaries
        flat_shapes = context.flat_output_shapes
        golden_arrays = [
            __load_golden_from_file(get(flat_binaries, i), get(output_dtypes, i), get(flat_shapes, i))
            for i in range(len(flat_shapes))
        ]
    else:
        golden_arrays = __generate_golden(context, output_dtypes)

    # Remember: golden_arrays is always flatten.

    if len(golden_arrays) > len(context.flat_output_shapes):
        golden_arrays = golden_arrays[: len(context.flat_output_shapes)]
    __normalize_goldens(golden_arrays, output_dtypes)
    # Regenerating golden on a testcase whose output fields were already
    # extended (GEIR regenerates per mode: const/dynamic) must walk only the
    # declared outputs — appended metadata entries carry no independent
    # golden and would trip __output_shape_needs_golden.
    n_meta = getattr(context, "_output_meta_count", 0)
    declared_shapes = context.flat_output_shapes[: len(context.flat_output_shapes) - n_meta]
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
            alloc_shape = [ceil_div(shape_product(output_shape), 8)]
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
    __append_out_shape_unknown_golden(context, golden_arrays, output_arrays, extend_fields=not n_meta)
    context.golden_arrays = golden_arrays
    context.output_arrays = tuple(output_arrays)


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
            ttk_bf16 = numpy_bfloat16()
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
    if get_global_storage().short_soc_version in ("Ascend310P",):  # result of this soc acts like INF/NAN mode.
        return golden_array
    npu_dtype = numpy.dtype(npu_dtype).name
    if npu_dtype in ("float32",) and get_global_storage().short_soc_version in (
        "Ascend910B",
        "Ascend910_93",
        "Ascend031",
        "MC62CM12A",
        "Ascend950",
    ):
        # always INF/NAN, switch not work on such dtypes + soc.
        return golden_array
    if npu_dtype == "bfloat16":
        dr = get_dtype_range("bfloat16")
        tmp = numpy.array([0, dr[1], dr[0], numpy.inf, -numpy.inf], dtype=numpy_bfloat16())
        nan = numpy.isnan(golden_array)
        if numpy.any(nan):
            golden_array = numpy.where(nan, tmp[0], golden_array)
        golden_array[golden_array == tmp[3]] = tmp[1]
        golden_array[golden_array == tmp[4]] = tmp[2]
    elif numpy.any(numpy.isnan(golden_array)) or numpy.any(numpy.isinf(golden_array)):
        numpy.nan_to_num(golden_array, copy=False)
    return golden_array


def __correct_kwargs_for_golden(context: TestcaseOp, golden_parameters: list):
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


def __collect_dynamic_golden_kwargs(context: TestcaseOp):
    switches = get_global_storage()
    kwargs = context.attributes.copy()
    # delete internal attributes
    keys = list(kwargs.keys())
    for k in keys:
        if str(k)[0] in ("!", "#", "@"):
            del kwargs[k]
    # delete const inputs in attributes
    op_info = OpInfoKeeper().info_of(context.op_name)
    inputs = [ipt["name"] for ipt in op_info["inputs"]]
    keys = list(kwargs.keys())
    for ipt in inputs:
        if ipt in keys:
            del kwargs[ipt]
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
            "golden_mode": (getattr(context, "golden_mode_override", None) or switches.golden_mode),
        }
    )
    return kwargs


def __golden_flatten(golden_results):
    from ....utilities.container_utils import deep_flatten

    if not isinstance(golden_results, Sequence):
        golden_results = [golden_results]
    return [numpy.array((x,)) if isinstance(x, numpy.generic) else x for x in deep_flatten(golden_results)]


def __output_shape_needs_golden(context: TestcaseOp, output_idx: int):
    if context.output_shape_unknown_indexes:
        raise RuntimeError(
            f"Golden must be supplied for {output_idx}th (count from 0) "
            f"output of {context.op_name} "
            f"since its shape depends on the value of input tensors."
        )


def __append_out_shape_unknown_golden(
    context: TestcaseOp, golden_arrays: list, output_arrays: list, extend_fields: bool = True
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
