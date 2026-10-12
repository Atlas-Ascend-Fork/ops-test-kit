#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
C++ source generator for GEIR tests.

Op-level template: one CPP per operator (cached), per-case data (shapes,
dtypes, formats, attrs) written as JSON config and read at runtime.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from .proto_loader import ProtoLoader

_DTYPE_TO_GE_ENUM = {
    "float32": "DT_FLOAT",
    "float16": "DT_FLOAT16",
    "bfloat16": "DT_BF16",
    "int32": "DT_INT32",
    "int16": "DT_INT16",
    "int8": "DT_INT8",
    "int4": "DT_INT4",
    "uint8": "DT_UINT8",
    "uint16": "DT_UINT16",
    "uint32": "DT_UINT32",
    "int64": "DT_INT64",
    "uint64": "DT_UINT64",
    "float64": "DT_DOUBLE",
    "double": "DT_DOUBLE",
    "bool": "DT_BOOL",
    "uint1": "DT_UINT1",
    "float8_e4m3fn": "DT_FLOAT8_E4M3FN",
    "float8_e5m2": "DT_FLOAT8_E5M2",
    "float8_e8m0": "DT_FLOAT8_E8M0",
    "hifloat8": "DT_HIFLOAT8",
    "float4_e2m1": "DT_FLOAT4_E2M1",
    "float4_e1m2": "DT_FLOAT4_E1M2",
    "hifloat4": "DT_HIFLOAT4",
    "complex32": "DT_COMPLEX32",
    "complex64": "DT_COMPLEX64",
    "complex128": "DT_COMPLEX128",
}

_DTYPE_SIZE_BY_ENUM = {
    "DT_FLOAT": 4,
    "DT_FLOAT16": 2,
    "DT_BF16": 2,
    "DT_INT32": 4,
    "DT_INT16": 2,
    "DT_INT8": 1,
    "DT_UINT8": 1,
    "DT_UINT16": 2,
    "DT_UINT32": 4,
    "DT_INT64": 8,
    "DT_UINT64": 8,
    "DT_DOUBLE": 8,
    "DT_BOOL": 1,
    "DT_FLOAT8_E4M3FN": 1,
    "DT_FLOAT8_E5M2": 1,
    "DT_FLOAT8_E8M0": 1,
    "DT_HIFLOAT8": 1,
    "DT_UINT1": 1,
    "DT_COMPLEX32": 4,
    "DT_COMPLEX64": 8,
    "DT_COMPLEX128": 16,
}

_PACKED_FLOAT4_ENUMS = ("DT_FLOAT4_E2M1", "DT_FLOAT4_E1M2", "DT_HIFLOAT4", "DT_INT4")

_FORMAT_TO_GE_ENUM = {
    "ND": "FORMAT_ND",
    "NC1HWC0": "FORMAT_NC1HWC0",
    "FRACTAL_Z": "FORMAT_FRACTAL_Z",
    "FRACTAL_NZ": "FORMAT_FRACTAL_NZ",
    "NCHW": "FORMAT_NCHW",
    "NHWC": "FORMAT_NHWC",
    "HWCN": "FORMAT_HWCN",
    "C1HWNCoC0": "FORMAT_C1HWNCoC0",
    "NDHWC": "FORMAT_NDHWC",
    "NCDHW": "FORMAT_NCDHW",
    "NDC1HWC0": "FORMAT_NDC1HWC0",
}

_PROTO_ATTR_TYPE_TO_CPP = {
    "Bool": "bool",
    "Int": "int64_t",
    "Int64": "int64_t",
    "UInt64": "int64_t",
    "Float": "float",
    "Double": "double",
    "String": "string",
    "ListInt": "vector_int64",
    "ListInt64": "vector_int64",
    "ListFloat": "vector_float",
    "ListDouble": "vector_double",
    "ListBool": "vector_bool",
    "ListString": "vector_string",
    "ListType": "vector_int64",
    "Type": "ge_dtype",
    "DataType": "ge_dtype",
    "Format": "ge_format",
}

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _render_template(name: str, **ctx) -> str:
    # noqa: S701 — the template renders C++ source; autoescape would HTML-escape
    # angle brackets (`vector<int64_t>`, `#include <...>`) and corrupt codegen.
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)  # noqa: S701
    tpl = env.get_template(name)
    return tpl.render(**ctx)


def _resolve_dtype(dtype_str: str) -> str:
    return _DTYPE_TO_GE_ENUM.get(dtype_str, "DT_FLOAT")


def _resolve_format(fmt_str: str) -> str:
    return _FORMAT_TO_GE_ENUM.get(fmt_str, "FORMAT_ND")


def _nested_or_broadcast(field, idx, default, count):
    if isinstance(field, (list, tuple)) and idx < len(field) and isinstance(field[idx], (list, tuple)):
        return field[idx]
    return [default] * count


def _attr_value_to_json(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_attr_value_to_json(x) for x in v]
    return str(v)


def _core_limit_options(core_limit) -> dict:
    """--core-limit (N|A,V) → GE Option 条目（ge.aicoreNum / ge.vectorcoreNum）。

    空侧出空串条目——C++ 模板按空串跳过（与 compile_dynamic_mode 同协议），
    保持 build_options 键集稳定。
    """
    ai_limit = vec_limit = ""
    if core_limit and isinstance(core_limit, tuple):
        ai_limit = str(core_limit[0]) if core_limit[0] is not None else ""
        vec_limit = str(core_limit[1]) if core_limit[1] is not None else ""
    return {"ge.aicoreNum": ai_limit, "ge.vectorcoreNum": vec_limit}


# Inplace 输出继承输入 desc 的键集（shape/dtype/format 三元组协议）。
_DESC_KEYS = ("desc_shape", "dtype", "format", "ori_format", "ori_shape")


def _desc_of(entry):
    return {k: entry[k] for k in _DESC_KEYS if k in entry}


def _dtype_at(dtypes, i):
    return dtypes[i] if isinstance(dtypes, (list, tuple)) and i < len(dtypes) else str(dtypes[0])


def _fmt_at(formats, i):
    if not formats:
        return "ND"
    return formats[i] if isinstance(formats, (list, tuple)) and i < len(formats) else "ND"


def _resolve_desc_shape(data_shape, dyn_shapes, i, is_dynamic):
    if is_dynamic and dyn_shapes and i < len(dyn_shapes) and dyn_shapes[i] is not None:
        return list(dyn_shapes[i])
    if is_dynamic:
        return [-1 for _ in data_shape]
    return data_shape


def _build_dynamic_input_group(
    i,
    name,
    input_shapes,
    input_dtypes,
    dtype_str,
    fmt_str,
    ori_fmt_str,
    input_ori_shapes,
    dyn_input_shapes,
    is_dynamic,
    data_idx,
):
    """DYNAMIC_INPUT(TensorList)端口:input_shapes[i] 是嵌套的 shape 列表,
    逐元素展开为 elements,模板按变参口(create_dynamic_input_X)接线。"""
    elems = []
    elt_dtypes = (
        input_dtypes[i]
        if (
            isinstance(input_dtypes, (list, tuple))
            and i < len(input_dtypes)
            and isinstance(input_dtypes[i], (list, tuple))
        )
        else [dtype_str] * len(input_shapes[i])
    )
    elt_ori_shapes = _nested_or_broadcast(input_ori_shapes, i, None, len(input_shapes[i]))
    for j, eshape in enumerate(input_shapes[i]):
        edt = elt_dtypes[j] if j < len(elt_dtypes) else elt_dtypes[-1]
        eori_shape = elt_ori_shapes[j] if j < len(elt_ori_shapes) else None
        edata_shape = list(eshape)
        # 与普通输入对齐:dynamic 模式下 desc_shape 用 dyn_input_shapes
        # 对应元素(支持 -1/-2),缺省回退 -1;const 模式用实际 shape。
        edesc_shape = edata_shape
        if is_dynamic:
            edesc_shape = [-1 for _ in edata_shape]
            if dyn_input_shapes and i < len(dyn_input_shapes) and dyn_input_shapes[i] is not None:
                dyn_el = dyn_input_shapes[i]
                if isinstance(dyn_el, (list, tuple)) and j < len(dyn_el) and isinstance(dyn_el[j], (list, tuple)):
                    edesc_shape = list(dyn_el[j])
        elems.append(
            {
                "data_idx": data_idx + j,
                "data_shape": edata_shape,
                "desc_shape": edesc_shape,
                "dtype": _resolve_dtype(edt),
                "format": _resolve_format(fmt_str),
                "ori_format": _resolve_format(ori_fmt_str),
                # 与普通输入对齐:dynamic 模式下 ori_shape 用动态 shape(-1/-2)
                "ori_shape": (
                    edesc_shape if is_dynamic else (list(eori_shape) if eori_shape is not None else edata_shape)
                ),
            }
        )
    return {
        "name": name,
        "is_const": False,
        "dynamic": True,
        "count": len(elems),
        "elements": elems,
    }


def _build_inputs_json(
    input_names,
    input_shapes,
    input_dtypes,
    input_formats,
    input_ori_formats,
    input_ori_shapes,
    dyn_input_shapes,
    attr_keys,
    dynamic_input_names,
    is_dynamic,
):
    inputs_json: List[Optional[Dict[str, Any]]] = []
    data_idx = 0
    for i, name in enumerate(input_names):
        if i >= len(input_shapes):
            inputs_json.append(None)
            continue
        if input_shapes[i] is None:
            inputs_json.append(None)
            continue
        dtype_str = _dtype_at(input_dtypes, i)
        fmt_str = _fmt_at(input_formats, i)
        ori_fmt_str = _fmt_at(input_ori_formats, i)
        if (
            name in dynamic_input_names
            and isinstance(input_shapes[i], (list, tuple))
            and input_shapes[i]
            and isinstance(input_shapes[i][0], (list, tuple))
        ):
            entry = _build_dynamic_input_group(
                i,
                name,
                input_shapes,
                input_dtypes,
                dtype_str,
                fmt_str,
                ori_fmt_str,
                input_ori_shapes,
                dyn_input_shapes,
                is_dynamic,
                data_idx,
            )
            inputs_json.append(entry)
            data_idx += entry["count"]
            continue
        data_shape = list(input_shapes[i])
        desc_shape = _resolve_desc_shape(data_shape, dyn_input_shapes, i, is_dynamic)
        inputs_json.append(
            {
                "name": name,
                "is_const": (name in attr_keys) or bool(os.environ.get("GEIR_CONST_FEED")),
                "data_idx": data_idx,
                "data_shape": data_shape,
                "desc_shape": desc_shape,
                "dtype": _resolve_dtype(dtype_str),
                "format": _resolve_format(fmt_str),
                "ori_format": _resolve_format(ori_fmt_str),
                # dynamic 模式下 ori_shape 也用动态 shape(-1/-2)，与真实动态场景一致：
                # 算子 infershape 的 GetInputShape 读的是 origin shape，编译期应看到未知维度。
                "ori_shape": (
                    desc_shape
                    if is_dynamic
                    else (
                        list(input_ori_shapes[i])
                        if isinstance(input_ori_shapes, (list, tuple))
                        and i < len(input_ori_shapes)
                        and input_ori_shapes[i] is not None
                        else list(input_shapes[i])
                    )
                ),
            }
        )
        data_idx += 1
    return inputs_json


def _build_input_desc_maps(inputs_json):
    """flat data_idx → desc 映射 + DYNAMIC_INPUT 组 elements 表（inplace 继承用）。"""
    input_desc_map = {}
    # flat data_idx -> 所属 DYNAMIC_INPUT(TensorList)组的 elements 列表，
    # 供 inplace 的 DYNAMIC_OUTPUT 逐元素继承输入 desc 使用。
    dynamic_input_groups = {}
    for inp in inputs_json:
        if inp is None:
            continue
        if inp.get("dynamic"):
            elements = inp.get("elements", [])
            for el in elements:
                di = el.get("data_idx")
                if di is not None:
                    input_desc_map[di] = _desc_of(el)
                    dynamic_input_groups[di] = elements
        else:
            di = inp.get("data_idx")
            if di is not None:
                input_desc_map[di] = _desc_of(inp)
    return input_desc_map, dynamic_input_groups


def _build_dynamic_output_group(
    i,
    name,
    output_shapes,
    output_dtypes,
    output_formats,
    output_ori_formats,
    output_ori_shapes,
    dtype_str,
    fmt_str,
    ori_fmt_str,
    is_dynamic,
    inplace_idx,
    dynamic_input_groups,
):
    """DYNAMIC_OUTPUT(TensorList)端口：逐元素 desc。dtype 落 CSV 声明的
    output_dtypes（嵌套/广播解析，与输入组同款）——update_dynamic_output_desc
    对部分算子（如 Split）不接受 DT_UNDEFINED（E39999），且变参口无
    InferDataType 推导路径可言，声明值即权威。"""
    count = len(output_shapes[i])
    elt_dtypes = (
        output_dtypes[i]
        if (
            isinstance(output_dtypes, (list, tuple))
            and i < len(output_dtypes)
            and isinstance(output_dtypes[i], (list, tuple))
        )
        else None
    )
    elt_fmts = _nested_or_broadcast(output_formats, i, fmt_str, count)
    elt_ori_fmts = _nested_or_broadcast(output_ori_formats, i, ori_fmt_str, count)
    elt_ori_shapes = _nested_or_broadcast(output_ori_shapes, i, None, count)
    elems = []
    for j, eshape in enumerate(output_shapes[i]):
        edt = (elt_dtypes[j] if j < len(elt_dtypes) else elt_dtypes[-1]) if elt_dtypes else dtype_str
        efmt = elt_fmts[j] if j < len(elt_fmts) else elt_fmts[-1]
        eori_fmt = elt_ori_fmts[j] if j < len(elt_ori_fmts) else elt_ori_fmts[-1]
        eori_shape = elt_ori_shapes[j] if j < len(elt_ori_shapes) else None
        out_data_shape = list(eshape)
        out_desc_shape = [-1 for _ in out_data_shape] if is_dynamic else out_data_shape
        elems.append(
            {
                "data_shape": out_data_shape,
                "desc_shape": out_desc_shape,
                "dtype": _resolve_dtype(edt),
                "format": _resolve_format(efmt),
                "ori_format": _resolve_format(eori_fmt),
                "ori_shape": (
                    out_desc_shape if is_dynamic else (list(eori_shape) if eori_shape is not None else out_data_shape)
                ),
            }
        )
    # inplace 的 DYNAMIC_OUTPUT(TensorList)：逐元素继承对应输入组
    # 元素的 desc（dtype 已知，不参与 InferDataType 推导触发）。
    # 模板按 elements 逐个 update_dynamic_output_desc，若沿用
    # DT_UNDEFINED 会触发 GE Unsupported_Operator(EZ3002)。
    if inplace_idx is not None and inplace_idx in dynamic_input_groups:
        src_elements = dynamic_input_groups[inplace_idx]
        for j, el in enumerate(elems):
            if j < len(src_elements):
                el.update(_desc_of(src_elements[j]))
    return {
        "name": name,
        "dynamic": True,
        "count": len(elems),
        "elements": elems,
    }


def _build_outputs_json(
    out_names,
    output_shapes,
    output_dtypes,
    output_formats,
    output_ori_formats,
    output_ori_shapes,
    inplace_indexes,
    dynamic_output_names,
    is_dynamic,
    input_desc_map,
    dynamic_input_groups,
):
    outputs_json: List[Optional[Dict[str, Any]]] = []
    for i, name in enumerate(out_names):
        if i >= len(output_shapes):
            outputs_json.append(None)
            continue
        if output_shapes[i] is None:
            outputs_json.append(None)
            continue
        fmt_str = _fmt_at(output_formats, i)
        ori_fmt_str = _fmt_at(output_ori_formats, i)
        dtype_str = _dtype_at(output_dtypes, i)
        inplace_idx = inplace_indexes[i] if i < len(inplace_indexes) else None
        if (
            name in dynamic_output_names
            and isinstance(output_shapes[i], (list, tuple))
            and output_shapes[i]
            and isinstance(output_shapes[i][0], (list, tuple))
        ):
            entry = _build_dynamic_output_group(
                i,
                name,
                output_shapes,
                output_dtypes,
                output_formats,
                output_ori_formats,
                output_ori_shapes,
                dtype_str,
                fmt_str,
                ori_fmt_str,
                is_dynamic,
                inplace_idx,
                dynamic_input_groups,
            )
        else:
            out_data_shape = list(output_shapes[i])
            out_desc_shape = [-1 for _ in out_data_shape] if is_dynamic else out_data_shape
            entry = {
                "name": name,
                "data_shape": out_data_shape,
                "desc_shape": out_desc_shape,
                "dtype": "DT_UNDEFINED",
                "format": _resolve_format(fmt_str),
                "ori_format": _resolve_format(ori_fmt_str),
                "ori_shape": (
                    out_desc_shape
                    if is_dynamic
                    else (
                        list(output_ori_shapes[i])
                        if isinstance(output_ori_shapes, (list, tuple))
                        and i < len(output_ori_shapes)
                        and output_ori_shapes[i] is not None
                        else list(output_shapes[i])
                    )
                ),
            }
        if inplace_idx is not None:
            entry["inplace_input_idx"] = inplace_idx
            entry.update(input_desc_map.get(inplace_idx, {}))
        outputs_json.append(entry)

    # ---- inplace outputs without proto OUTPUT declaration ----
    # When proto has no OUTPUT (out_names empty), the for loop above doesn't
    # run. Inplace outputs from CSV's output_inplace_indexes still need to be
    # added to outputs_json with desc inherited from the corresponding input.
    if not out_names:
        for idx in inplace_indexes:
            if idx is not None:
                entry = {"inplace_input_idx": idx}
                entry.update(input_desc_map.get(idx, {}))
                outputs_json.append(entry)
    return outputs_json


def _build_attrs_json(attrs, proto_info, input_names):
    # attrs（exclude input names and special prefixes）
    input_name_set = set(input_names)
    attr_proto_types = dict(proto_info.attrs)
    attrs_json: Dict[str, Any] = {}
    for k, v in attrs.items():
        if k in input_name_set or str(k)[0] in ("!", "#", "@"):
            continue
        proto_type = attr_proto_types.get(k)
        if proto_type in ("Type", "DataType") and isinstance(v, str):
            attrs_json[k] = _resolve_dtype(v)
        elif proto_type == "Format" and isinstance(v, str):
            attrs_json[k] = _resolve_format(v)
        else:
            attrs_json[k] = _attr_value_to_json(v)
    return attrs_json


def _build_ge_build_options(switches, is_dynamic, is_binary):
    jit_compile = "0" if is_binary else "1"
    compile_dynamic_mode = "1" if (is_dynamic and not is_binary) else ""
    ge_deterministic_level = ""
    if getattr(switches, "deterministic_level", 0) == 1:
        ge_deterministic_level = "1"
    return {
        "ge.jit_compile": jit_compile,
        "ge.compile_dynamic_mode": compile_dynamic_mode,
        "ge.deterministicLevel": ge_deterministic_level,
        "ge.deterministic": "1" if ge_deterministic_level == "1" else "0",
        # --core-limit 经 GE Option 下发（ai→ge.aicoreNum，vec→ge.vectorcoreNum；
        # 空串条目由 C++ 模板侧跳过——与 compile_dynamic_mode 同协议）
        **_core_limit_options(getattr(switches, "core_limit", None)),
    }


def _shape_unknown_indexes(testcase):
    """output shape-unknown indexes (kernel-route protocol).

    CSV's output_shape_unknown_indexes marks data-dependent outputs whose
    effective shape is only known after compute. The kernel route appends
    a uint64[9*k] shape-metadata golden (output_generation.py
    __append_out_shape_unknown_golden); GEIR mirrors it on the device side
    by emitting one extra output chunk built from the actual GE-computed
    output shape, so Spec.compare receives the same 2-outputs/2-goldens
    contract on both routes.
    """
    return sorted(
        {int(idx) for idx in (getattr(testcase, "output_shape_unknown_indexes", None) or ()) if idx is not None}
    )


class GeirGraphBuilder:
    """Generate op-level C++ source and per-case JSON config for GEIR tests."""

    def __init__(self, switches):
        self._switches = switches
        self._proto_loader = ProtoLoader()
        self._work_dir: Optional[str] = None
        self._op_dir: Optional[str] = None
        self._last_proto_file: Optional[str] = None

    @property
    def work_dir(self) -> Optional[str]:
        return self._work_dir

    @property
    def op_dir(self) -> Optional[str]:
        return self._op_dir

    @property
    def last_proto_file(self) -> Optional[str]:
        return self._last_proto_file

    def _compute_dirs(self, mode: str, work_dir: str = None):
        is_dynamic = mode.startswith("dynamic")
        is_binary = "binary" in mode
        base_dir = "dynamic" if is_dynamic else "const"
        sub_dir = os.path.join(base_dir, "binary") if is_binary else base_dir
        root = getattr(self._switches, "root_path", os.getcwd())
        self._work_dir = work_dir or os.path.join(root, "geir", sub_dir)
        self._op_dir = os.path.join(root, "geir", "ops")
        os.makedirs(self._work_dir, exist_ok=True)
        os.makedirs(self._op_dir, exist_ok=True)

    def _build_attr_entries(self, proto_info) -> List[Dict[str, str]]:
        input_name_set = set(proto_info.inputs)
        attr_entries = []
        for attr_name, proto_type in proto_info.attrs:
            if attr_name in input_name_set:
                continue
            if attr_name[0] in ("!", "#", "@"):
                continue
            cpp_type = _PROTO_ATTR_TYPE_TO_CPP.get(proto_type, "")
            if not cpp_type:
                raise RuntimeError(
                    f"Unsupported proto attr type '{proto_type}' for attr '{attr_name}' "
                    f"on operator '{proto_info.op_class}'"
                )
            attr_entries.append(
                {
                    "name": attr_name,
                    "proto_type": proto_type,
                    "cpp_type": cpp_type,
                }
            )
        return attr_entries

    def generate_op_source(self, op_name: str, mode="const", work_dir: str = None) -> Optional[str]:
        """Generate op-level C++ source (one per operator, cached). Returns file path."""
        self._compute_dirs(mode, work_dir)

        proto_info = self._proto_loader.get_op_info(op_name)
        if proto_info is None:
            logging.error(
                "No op IR (REG_OP) found for operator '%s' in the op_proto headers; "
                "GEIR graph cannot be built for it. See the preceding warning for searched paths.",
                op_name,
            )
            return None
        self._last_proto_file = proto_info.proto_file

        attr_entries = self._build_attr_entries(proto_info)

        from .compiler import detect_cann_features

        features = detect_cann_features()
        ge_enums = features["ge_enums"] if features else None
        # DT_UNDEFINED 供输出 desc 使用：dtype 置 DT_UNDEFINED 触发 GE 调算子
        # InferDataType 回调，StringToDataType 需可解析该枚举。
        dtype_map = [("DT_UNDEFINED", "DT_UNDEFINED")] + [
            (v, v) for v in _DTYPE_TO_GE_ENUM.values() if ge_enums is None or v in ge_enums
        ]
        dtype_size_cases = [(e, s) for e, s in _DTYPE_SIZE_BY_ENUM.items() if ge_enums is None or e in ge_enums]
        packed_float4_enums = [e for e in _PACKED_FLOAT4_ENUMS if ge_enums is None or e in ge_enums]

        source = _render_template(
            "geir_op_template.cpp.j2",
            proto_file=proto_info.proto_file,
            op_class=proto_info.op_class,
            input_names=proto_info.inputs[:],
            output_names=proto_info.outputs[:],
            dynamic_input_names=(proto_info.dynamic_inputs or []),
            dynamic_output_names=(proto_info.dynamic_outputs or []),
            attr_entries=attr_entries,
            dtype_map=dtype_map,
            format_map=[(v, v) for v in _FORMAT_TO_GE_ENUM.values()],
            dtype_size_cases=dtype_size_cases,
            packed_float4_enums=packed_float4_enums,
            ge_prof_ok=(features["ge_prof_ok"] if features else True),
        )

        source_path = os.path.join(self._op_dir, f"{op_name}.cpp")
        if os.path.isfile(source_path):
            try:
                with open(source_path, encoding="utf-8") as f:
                    if f.read() == source:
                        logging.info("GEIR op source unchanged: %s", source_path)
                        return source_path
            except OSError:
                pass
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source)
        logging.info("Generated GEIR op source: %s", source_path)
        return source_path

    def write_case_config(self, testcase, mode="const", work_dir: str = None) -> Optional[str]:
        """Write per-case JSON config. Returns file path."""
        if work_dir:
            self._work_dir = work_dir
        if not self._work_dir:
            self._compute_dirs(mode)

        op_name = testcase.op_name
        input_shapes = testcase.input_shapes
        dyn_input_shapes = getattr(testcase, "dyn_input_shapes", None)
        input_dtypes = testcase.input_dtypes
        input_formats = getattr(testcase, "input_formats", None) or ()
        input_ori_formats = getattr(testcase, "input_ori_formats", None) or ()
        input_ori_shapes = getattr(testcase, "input_ori_shapes", None) or ()
        output_ori_shapes = getattr(testcase, "output_ori_shapes", None) or ()
        output_shapes = testcase.output_shapes
        output_dtypes = testcase.output_dtypes
        output_formats = getattr(testcase, "output_formats", None) or ()
        output_ori_formats = getattr(testcase, "output_ori_formats", None) or ()
        attrs = testcase.attributes or {}

        is_dynamic = mode.startswith("dynamic")
        is_binary = "binary" in mode
        # 输出 desc 的 dtype 一律置 DT_UNDEFINED，GE 编译期必须调用算子的
        # InferDataType 回调推导，否则 dtype 建图时已知、回调永不触发。
        # 推导出的 dtype 由 GE 自行落位：回调缺失或执行失败会导致图编译/执行失败，
        # 即完成功能验证；inplace 输出仍继承对应输入的 desc（dtype 已知），不参与
        # 推导触发。

        proto_info = self._proto_loader.get_op_info(op_name)
        if proto_info is None:
            logging.error(
                "No op IR (REG_OP) found for operator '%s' in the op_proto headers; "
                "GEIR graph cannot be built for it. See the preceding warning for searched paths.",
                op_name,
            )
            return None
        self._last_proto_file = proto_info.proto_file

        input_names = proto_info.inputs[:]
        out_names = proto_info.outputs[:]
        attr_keys = set(attrs.keys())

        # ---- inputs ----
        dynamic_input_names = set(getattr(proto_info, "dynamic_inputs", None) or [])
        inputs_json = _build_inputs_json(
            input_names,
            input_shapes,
            input_dtypes,
            input_formats,
            input_ori_formats,
            input_ori_shapes,
            dyn_input_shapes,
            attr_keys,
            dynamic_input_names,
            is_dynamic,
        )

        # ---- outputs ----
        # Inplace outputs inherit desc (shape/dtype/format) from the
        # corresponding input data node. Build a flat lookup: data_idx → desc.
        dynamic_output_names = set(getattr(proto_info, "dynamic_outputs", None) or [])
        inplace_indexes = getattr(testcase, "output_inplace_indexes", None) or ()
        input_desc_map, dynamic_input_groups = _build_input_desc_maps(inputs_json)
        outputs_json = _build_outputs_json(
            out_names,
            output_shapes,
            output_dtypes,
            output_formats,
            output_ori_formats,
            output_ori_shapes,
            inplace_indexes,
            dynamic_output_names,
            is_dynamic,
            input_desc_map,
            dynamic_input_groups,
        )

        # ---- attrs (exclude input names and special prefixes) ----
        attrs_json = _build_attrs_json(attrs, proto_info, input_names)

        # ---- build options ----
        build_options = _build_ge_build_options(self._switches, is_dynamic, is_binary)

        # ---- output shape-unknown indexes (kernel-route protocol) ----
        shape_unknown_indexes = _shape_unknown_indexes(testcase)

        config = {
            "inputs": inputs_json,
            "outputs": outputs_json,
            "attrs": attrs_json,
            "build_options": build_options,
            "shape_unknown_indexes": shape_unknown_indexes,
        }

        config_path = os.path.join(self._work_dir, f"{testcase.testcase_name}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logging.info("Generated GEIR case config: %s", config_path)
        return config_path
