#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Standard Packages
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import pathlib
from typing import Callable, List, Mapping, Optional, Tuple, Union

from ....utilities import (
    ACL_DTYPE_TO_ID,
    FORMAT_CODES,
    is_nchw_like,
    is_ndhwc_like,
)
from ..info_keeper import OpInfoKeeper
from ..operator.op_interface import OperatorInterface
from ..params import adapt_pickup_by_names
from ..testcase import TestcaseKernel
from .result import DynamicCompilationResult, KernelJsonInfo

# Third-Party Packages


# cache
op_generalize_func: dict = {}
# dtype mode
DTYPE_MODE_NORMAL = "normal"
DTYPE_MODE_BIT = "bit"
DTYPE_MODE_BOOL = "bool"
# bit mode equivalent set
DTYPE_MODE_64_SET = ("int64", "uint64", "float64", "double", "complex64")
DTYPE_MODE_32_SET = ("int32", "uint32", "float32", "float", "complex32")
DTYPE_MODE_16_SET = ("int16", "uint16", "float16", "bfloat16")
DTYPE_MODE_8_SET = ("int8", "uint8", "bool", "float8_e8m0", "hifloat8", "float8_e5m2", "float8_e4m3fn")
DTYPE_MODE_6_SET = ("float6_e3m2", "float6_e2m3")
DTYPE_MODE_4_SET = ("float4_e2m1", "float4_e1m2", "hifloat4")
# format mode
FORMAT_MODE_NORMAL = "normal"
FORMAT_MODE_AGNOSTIC = "agnostic"
FORMAT_MODE_ND_AGNOSTIC = "nd_agnostic"
FORMAT_MODE_STATIC_ND_AGNOSTIC = "static_nd_agnostic"
# support simplifiedKey mode id
SUPPORT_SIMPLIFIEDKEY_ID_SET = (0, 1, 2)


def binary_kernel_match(interface: OperatorInterface, testcase: TestcaseKernel, compile_options: dict = None):
    op_name = testcase.op_name  # snake_name
    op_type = OpInfoKeeper().op_type_of(op_name)  # CamelName
    if not op_type:
        raise RuntimeError(f"[{op_name}] is not configured in [aic-*-ops-info.json]")

    matched_bin_info = None
    if support_simplified_key_mode(op_type, testcase):
        simplified_key = generate_simplified_key(interface, testcase, compile_options)
        matched_bin_info = kernel_match_by_simplified_key(simplified_key, op_name)
    if not matched_bin_info:
        bin_cfg: dict = OpInfoKeeper().binary_static_key_config_of(op_name)
        if not bin_cfg:
            return None
        stc_generalize_info, dynamic_generalize_info = generalize_op(interface, testcase, compile_options)
        optional_input_indices = get_optional_input_indices(op_name)
        matched_bin_info = kernel_match_by_static_key(
            testcase, bin_cfg, stc_generalize_info, dynamic_generalize_info, optional_input_indices
        )
    if not matched_bin_info:
        return None
    if "binInfo" not in matched_bin_info:  # simplified key mode
        if "jsonPath" not in matched_bin_info:
            raise RuntimeError(f"Invalid binary_info_config.json for op [{op_type}]. no jsonPath in it.")
        return matched_bin_info["jsonPath"]
    # static key mode
    if "jsonFilePath" not in matched_bin_info["binInfo"]:
        raise RuntimeError(f"Invalid binary info json of op [{op_type}]. no jsonFilePath in it.")
    return matched_bin_info["binInfo"]["jsonFilePath"]


def parse_matched_bin_info(
    interface: OperatorInterface,
    testcase: TestcaseKernel,
    matched_bin_info: Union[str, pathlib.Path],
    result: DynamicCompilationResult,
) -> None:
    if not os.path.exists(matched_bin_info):
        raise RuntimeError(f"Binary kernel info json file not exist: {matched_bin_info}")
    with open(matched_bin_info, encoding="UTF-8") as f:
        info: dict = json.load(f)
    result.kernel_name = info.get("binFileName")
    compile_info: dict = info.get("compileInfo")
    tiling_op_type: str = OpInfoKeeper().op_type_of(testcase.op_name)
    # it will never be None
    op_func = interface.get_dyn_operator(testcase)
    dyn_func_params = interface.get_op_func_params(op_func, testcase.op_name)
    kernel_dir = os.path.dirname(matched_bin_info)
    kernel_json_info = KernelJsonInfo.from_dict(info)
    # set compilation result
    result.standard_set(
        compile_info,
        tiling_op_type,
        "SUCC",
        "BINARY_MATCH",
        dyn_func_params,
        kernel_json_info,
        result.kernel_name,
        kernel_dir,
    )


def support_simplified_key_mode(op_type: str, testcase: TestcaseKernel):
    binary_info = OpInfoKeeper().binary_info_config_of(testcase.op_name)
    if not binary_info:
        logging.info(
            f"OpType [{op_type}] is not configured in binary_info_config.json. SimplifiedKey mode is not supported."
        )
        return False
    simplified_key_mode = binary_info.get("simplifiedKeyMode", -1)
    if simplified_key_mode not in SUPPORT_SIMPLIFIEDKEY_ID_SET:
        logging.warning(f"SimplifiedKeyMode [{simplified_key_mode}] is not supported for {op_type}")
        return False
    op_info = OpInfoKeeper().info_of(testcase.op_name)
    op_info_input_args_len = len(op_info.get("inputs", ()))
    op_info_output_args_len = len(op_info.get("outputs", ()))
    config_input_args_len = len(binary_info.get("params", {}).get("inputs", ()))
    config_output_args_len = len(binary_info.get("params", {}).get("outputs", ()))
    if op_info_input_args_len != config_input_args_len or op_info_output_args_len != config_output_args_len:
        logging.warning(
            f"Inputs or outputs count mismatch which in binary_info_config.json for [{op_type}]"
            f", not support simplifiedKeyMode"
        )
        return False
    return True


def gather_dynamic_tensors(interface: OperatorInterface, testcase: TestcaseKernel):
    return interface.prepare_operator_parameters(testcase, mode="dyn")


def dtype_normalize(dtype_mode, dtype):
    # DtypeNormalize
    if dtype_mode == DTYPE_MODE_NORMAL:
        return dtype
    if dtype_mode == DTYPE_MODE_BIT:
        if dtype in DTYPE_MODE_64_SET:
            return "int64"
        if dtype in DTYPE_MODE_32_SET:
            return "int32"
        if dtype in DTYPE_MODE_16_SET:
            return "int16"
        if dtype in DTYPE_MODE_8_SET:
            return "int8"
        if dtype in DTYPE_MODE_6_SET:
            return "float6_e3m2"
        if dtype in DTYPE_MODE_4_SET:
            return "float4_e2m1"
    elif dtype_mode == DTYPE_MODE_BOOL:
        if dtype == "bool":
            return "int8"
        return None
    return None


def format_normalize(format_mode, tensor_format, shape):
    # FormatNormalize
    if format_mode == FORMAT_MODE_NORMAL:
        return tensor_format
    if format_mode == FORMAT_MODE_AGNOSTIC:
        return "ND"
    if format_mode in (FORMAT_MODE_ND_AGNOSTIC, FORMAT_MODE_STATIC_ND_AGNOSTIC):
        return "ND" if is_ndhwc_like(shape, tensor_format) or is_nchw_like(shape, tensor_format) else tensor_format
    return tensor_format


def get_impl_mode_int(impl_mode: str) -> int:
    IMPL_MODE_DICT: dict = {
        "high_performance": 1,
        "high_precision": 2,
        "super_performance": 3,
        "support_out_of_bound_index": 4,
        "enable_float_32_execution": 5,
        "enable_hi_float_32_execution": 6,
        "keep_fp16": 7,
    }
    return IMPL_MODE_DICT.get(impl_mode, 0)


def generalize_attr_for_simple_mode(
    interface: OperatorInterface, testcase: TestcaseKernel, op_info: dict
) -> Optional[str]:
    # GenerateAttrs
    attr_in_op_info = op_info.get("attr")
    if not attr_in_op_info:
        return None
    generalized_attr = []
    xput_num = len(op_info.get("inputs", ())) + len(op_info.get("outputs", ()))
    # fix axes, axis
    op_func = interface.get_dyn_operator(testcase)
    dyn_func_params = interface.get_op_func_parameter_dict(op_func, testcase.op_name)
    op_kwargs = adapt_pickup_by_names(testcase.spec_attrs, tuple(dyn_func_params))
    idx = 0
    for k, v in dyn_func_params.items():
        idx = idx + 1
        if idx <= xput_num:
            continue
        if idx - xput_num > len(attr_in_op_info) or k == "kernel_name":
            break
        attr_cfg = attr_in_op_info[idx - xput_num - 1]
        attr_value = (
            op_kwargs.get(k)
            if k in op_kwargs
            else v.default
            if v.default is not inspect.Parameter.empty
            else attr_cfg["defaultValue"]
        )
        attr_type = attr_cfg.get("type")
        if attr_type == "str":
            generalized_attr.append(attr_value)
        elif attr_type == "bool":
            generalized_attr.append("1" if attr_value else "0")
        else:
            generalized_attr.append("")
    return ",".join(generalized_attr)


def generate_simplified_key(interface: OperatorInterface, testcase: TestcaseKernel, compile_options: dict = None):
    # GenerateSimpleKeyStr
    def _construct_dtype_format(xput_tensor: Union[List[dict], dict], _dtype_mode, _format_mode, explain: list):
        tensor = xput_tensor[0] if isinstance(xput_tensor, (list, tuple)) else xput_tensor
        dtype, fmt, shape = tensor["dtype"], tensor["format"], tensor["shape"]
        dtype = dtype_normalize(_dtype_mode, dtype)
        fmt = format_normalize(_format_mode, fmt, shape)
        explain.append(f"{dtype},{fmt}")
        # do not use get. we wanna get a RuntimeError when not supported.
        return f"{ACL_DTYPE_TO_ID[dtype]},{FORMAT_CODES[fmt]}"

    # simplifiedKey: op_type/deterministic,impl_mode/required input dtype,format/output dtype,format
    op_type = OpInfoKeeper().op_type_of(testcase.op_name)
    deterministic = 1 if (compile_options or {}).get("enable_deterministic_mode", "0") == "1" else 0
    impl_mode_str = testcase.attributes.get("impl_mode", "")
    impl_mode = get_impl_mode_int(impl_mode_str)

    op_info = OpInfoKeeper().info_of(testcase.op_name)
    binary_info = OpInfoKeeper().binary_info_config_of(testcase.op_name)
    dyn_inputs, dyn_outputs = gather_dynamic_tensors(interface, testcase)
    case_xputs: dict = {"inputs": dyn_inputs, "outputs": dyn_outputs}
    binary_info_params_value = binary_info.get("params", None)
    simplified_key_mode = binary_info["simplifiedKeyMode"]
    simplified_key = f"{op_type}/d={deterministic},p={impl_mode}/"
    key_explain = f"{op_type}/determination={deterministic},impl_mode={impl_mode_str}/"
    if simplified_key_mode == 2:
        from ..operator.registries import customize_gen_simplified_key

        inputs, outputs, attrs = interface.prepare_tiling_params(testcase)
        simplified_key = customize_gen_simplified_key(simplified_key, op_type, inputs, outputs, attrs)
        logging.info(f"customized simplified_key is {simplified_key}")
        return simplified_key
    optional_input_mode = binary_info.get("optionalInputMode", "no_placeholder")
    dynamic_param_mode = binary_info.get("dynamicParamMode", "")
    xput_key = []
    xput_key_exp = []
    for str_xpt in ("inputs", "outputs"):
        for idx, xpt_x in enumerate(op_info.get(str_xpt, ())):
            if not case_xputs[str_xpt][idx]:
                if simplified_key_mode == 1:
                    xput_key.append(",")
                    xput_key_exp.append(",")
                continue
            if xpt_x.get("paramType") == "optional" and (
                simplified_key_mode != 1 or optional_input_mode != "gen_placeholder"
            ):
                continue
            binary_info_xpt = binary_info_params_value[str_xpt][idx]
            binary_info_xpt = binary_info_xpt[0] if isinstance(binary_info_xpt, list) else binary_info_xpt
            dtype_mode = binary_info_xpt.get("dtypeMode", DTYPE_MODE_NORMAL)
            format_mode = binary_info_xpt.get("formatMode", FORMAT_MODE_NORMAL)
            xput_key.append(_construct_dtype_format(case_xputs[str_xpt][idx], dtype_mode, format_mode, xput_key_exp))
            if (
                xpt_x.get("paramType") == "dynamic"
                and simplified_key_mode == 1
                and dynamic_param_mode != "folded_with_desc"
            ):
                xput_key_exp[-1] += ",tensor_list_count"
                xput_key[-1] += f",{len(case_xputs[str_xpt][idx])}"
    if simplified_key_mode == 1:
        attr_key = generalize_attr_for_simple_mode(interface, testcase, op_info)
        if attr_key is not None:
            xput_key_exp.append("attributes")
            xput_key.append(attr_key)
    simplified_key += f"{'/'.join(xput_key)}"
    key_explain += f"{'/'.join(xput_key_exp)}"
    logging.info(f"simplified_key is {simplified_key} Explain:{key_explain}")
    return simplified_key


def kernel_match_by_simplified_key(simplified_key: str, op_name: str):
    binary_info = OpInfoKeeper().binary_info_config_of(op_name)
    bin_list = binary_info.get("binaryList", ())
    matched_bin_list = [b for b in bin_list if "simplifiedKey" in b and simplified_key in b["simplifiedKey"]]
    logging.debug(f"Simplified key of {simplified_key} matched bin count: {len(matched_bin_list)}")
    matched_bin = matched_bin_list[0] if matched_bin_list else None
    return matched_bin


def generalize_op(interface: OperatorInterface, testcase: TestcaseKernel, compile_options: dict = None):
    # GeneralizeOps
    dyn_inputs, dyn_outputs = gather_dynamic_tensors(interface, testcase)
    op_info = OpInfoKeeper().info_of(testcase.op_name)

    # it will never be None
    op_func = interface.get_dyn_operator(testcase)
    dyn_func_params = interface.get_op_func_parameter_dict(op_func, testcase.op_name)
    static_json, dynamic_json = generalize_with_default_rule(
        testcase, op_info, dyn_func_params, dyn_inputs, dyn_outputs, compile_options=compile_options
    )

    generalize_func = get_generalize_func_registered(interface, testcase)
    if generalize_func:
        try:
            registered_json = generalize_with_register_func(
                testcase, dyn_func_params, dyn_inputs, dyn_outputs, generalize_func
            )
            logging.debug(f"generalized json from registered function: {registered_json}")
        except BaseException as e:
            raise RuntimeError(f"Invoke registered generalize function failed: {e}") from None
        else:
            parse_registered_generalize_result(registered_json, static_json, dynamic_json)
    logging.debug(f"The last static json: {static_json}")
    logging.debug(f"The last dynamic json: {dynamic_json}")
    return static_json, dynamic_json


def get_optional_input_indices(op_name: str) -> list:
    op_info = OpInfoKeeper().info_of(op_name)
    inpts = op_info.get("inputs", ())
    return [idx for idx, inpt in enumerate(inpts) if inpt.get("paramType") == "optional"]


def get_generalize_func_registered(interface: OperatorInterface, testcase: TestcaseKernel) -> Optional[Callable]:
    if testcase.op_name in op_generalize_func:
        return op_generalize_func[testcase.op_name]
    op_func = interface.get_dyn_operator(testcase)
    if not op_func:
        generalize_func = None
    else:
        op_type = OpInfoKeeper().op_type_of(testcase.op_name)
        generalize_func = None if not op_type else interface.get_op_generalize_func(op_type)
    op_generalize_func[testcase.op_name] = generalize_func
    return generalize_func


def generalize_with_default_rule(
    testcase: TestcaseKernel,
    op_info: dict,
    dyn_func_params: Mapping[str, inspect.Parameter],
    dyn_inputs: Tuple[dict],
    dyn_outputs: Tuple[dict],
    compile_options: dict = None,
):
    op_pattern = op_info.get("op.pattern")
    support_dynamic_rank = str(op_info.get("dynamicRankSupport.flag", False) or False)
    if support_dynamic_rank.lower() != "true":
        # print warning log in case some operator forget to configure it.
        logging.warning(
            f"dynamicRankSupport of operator [{testcase.op_name}] is not configured "
            f"or configured to False !!! Binary kernel match may fail."
        )
    inputs_json = [
        generalize_xput(dyn_inputs[idx], input_x, op_pattern, support_dynamic_rank)
        for idx, input_x in enumerate(op_info.get("inputs", ()))
    ]
    static_inputs_json, dynamic_inputs_json = zip(*inputs_json)
    outputs_json = [
        generalize_xput(dyn_outputs[idx], output_x, op_pattern, support_dynamic_rank)
        for idx, output_x in enumerate(op_info.get("outputs", ()))
    ]
    static_outputs_json, dynamic_outputs_json = zip(*outputs_json) if outputs_json else (None, None)
    static_attr_json, dynamic_attr_json = generalize_attr(testcase, op_info, dyn_func_params)
    if static_outputs_json:
        static_json = {"inputs": static_inputs_json, "outputs": static_outputs_json}
        dynamic_json = {"inputs": dynamic_inputs_json, "outputs": dynamic_outputs_json}
    else:
        static_json = {"inputs": static_inputs_json}
        dynamic_json = {"inputs": dynamic_inputs_json}
    if static_attr_json:
        static_json.update({"attrs": static_attr_json})
    if dynamic_attr_json:
        dynamic_json.update({"attrs": dynamic_attr_json})
    deterministic_opt = (compile_options or {}).get("enable_deterministic_mode")
    if deterministic_opt:
        dynamic_json.update({"deterministic": deterministic_opt.lower()})
    logging.debug(f"Static json: {static_json}")
    logging.debug(f"Dynamic json: {dynamic_json}")
    return static_json, dynamic_json


def generalize_with_register_func(
    testcase: TestcaseKernel,
    dyn_func_params: Mapping[str, inspect.Parameter],
    dyn_inputs: Tuple[dict],
    dyn_outputs: Tuple[dict],
    generalize_func: Callable,
):
    op_info = OpInfoKeeper().info_of(testcase.op_name)
    attr_in_op_info = op_info.get("attr")
    # fix axes, axis
    dyn_func_param_names = tuple(dyn_func_params)
    kwargs = adapt_pickup_by_names(testcase.spec_attrs, dyn_func_param_names)
    kwargs.update({"generalize_config": {"mode": "all_shape", "single_op": "true"}})
    # pick attrs from kwargs to make BinaryMatchBase happy ...
    attrs: list = []
    for idx, name in enumerate(dyn_func_param_names[len(dyn_inputs) + len(dyn_outputs) :]):
        if name == "kernel_name":
            break
        if name in kwargs:
            attrs.append(kwargs[name])
            del kwargs[name]
        else:
            attr_parm_in_func = dyn_func_params[name]
            if attr_parm_in_func.default is not inspect.Parameter.empty:
                attrs.append(attr_parm_in_func.default)
            else:  # get from op_info.
                attrs.append(attr_in_op_info[idx]["defaultValue"])

    # call registered function
    return generalize_func(*dyn_inputs, *dyn_outputs, *attrs, **kwargs)


def generalize_xput(xput_tensor: Union[dict, List[dict]], op_info_xput: dict, op_pattern: str, support_dynamic_rank):
    # GeneralizeInOrOutputs
    if not xput_tensor:
        return None, None
    value_depend, param_type = op_info_xput.get("valueDepend"), op_info_xput.get("paramType")
    if value_depend == "optional" or (value_depend == "ignore" and param_type in ("optional", "required")):
        return generalize_xput_without_value_depend(xput_tensor, op_pattern, support_dynamic_rank)
    return feed_xput_directory(xput_tensor)


def generalize_xput_without_value_depend(xput_tensor: Union[List[dict], dict], op_pattern: str, support_dynamic_rank):
    def _construct_tensor_info(tensor):
        static_json = {
            "dtype": tensor["dtype"],
            "format": "ND" if op_pattern == "formatAgnostic" else tensor["format"],
            "shape": [-2] if support_dynamic_rank.lower() == "true" else [-1] * len(tensor["shape"]),
        }
        dynamic_json = {
            "ori_format": tensor["ori_format"],
            "ori_shape": tensor["ori_shape"],
            "ori_range": [[ele, ele] if ele > 0 else [1, None] for ele in tensor["ori_shape"]],
            "range": tensor["range"],
        }
        dynamic_json.update(static_json)
        return static_json, dynamic_json

    if isinstance(xput_tensor, (list, tuple)):
        tensor_info = [_construct_tensor_info(t) for t in xput_tensor]
        return zip(*tensor_info)
    return _construct_tensor_info(xput_tensor)


def feed_xput_directory(xput_tensor: Union[List[dict], dict]):
    def _construct_tensor_info(tensor):
        static_json = {"dtype": tensor["dtype"], "format": tensor["format"], "shape": tensor["shape"]}
        dynamic_json = {"ori_format": tensor["ori_format"], "ori_shape": tensor["ori_shape"]}
        dynamic_json.update(static_json)
        return static_json, dynamic_json

    if isinstance(xput_tensor, (list, tuple)):
        tensor_info = [_construct_tensor_info(t) for t in xput_tensor]
        return zip(*tensor_info)
    return _construct_tensor_info(xput_tensor)


def generalize_attr(testcase: TestcaseKernel, op_info: dict, dyn_func_params: Mapping[str, inspect.Parameter]):
    # GenerateNormalizeFusionAttrTmpJson
    attr_in_op_info = op_info.get("attr")
    if not attr_in_op_info:
        return None, None
    xput_num = len(op_info.get("inputs", ())) + len(op_info.get("outputs", ()))
    # fix axes, axis
    op_kwargs = adapt_pickup_by_names(testcase.spec_attrs, tuple(dyn_func_params))
    all_attr_info_dyn = []
    all_attr_info_stc = []
    idx = 0
    for k, v in dyn_func_params.items():
        idx = idx + 1
        if idx <= xput_num:
            continue
        if idx - xput_num > len(attr_in_op_info) or k == "kernel_name":
            break
        attr_info = {}
        attr_cfg = attr_in_op_info[idx - xput_num - 1]
        attr_type = attr_cfg.get("type")
        attr_info["dtype"] = attr_type.lower().replace("list", "list_").replace("str", "string")
        attr_info["value"] = (
            op_kwargs.get(k)
            if k in op_kwargs
            else v.default
            if v.default is not inspect.Parameter.empty
            else attr_cfg["defaultValue"]
        )
        if attr_cfg.get("value") == "all" and attr_type not in ("str", "bool"):
            attr_info["value"] = None
        all_attr_info_dyn.append(copy.deepcopy(attr_info))
        if attr_type in ("float", "listFloat"):
            attr_info["value"] = None
        all_attr_info_stc.append(attr_info)
    return all_attr_info_stc, all_attr_info_dyn


def parse_registered_generalize_result(registered_json: list, static_json: dict, dynamic_json: dict):
    def _replace_generalize_info(src, dst):
        def _replace_with_key(key):
            if key in src:
                dst[key] = src[key]

        if "ori_format" in src and "ori_format" in dst:
            dst["ori_format"] = src["ori_format"]
        _replace_with_key("shape")
        _replace_with_key("dtype")
        _replace_with_key("format")

    def _replace_xput_generalize_info(registered_xput_generalize, default_generalize, index):
        if registered_xput_generalize:
            if isinstance(registered_xput_generalize, (list, tuple)):
                if not isinstance(default_generalize, (list, tuple)) or len(registered_xput_generalize) != len(
                    default_generalize
                ):
                    raise RuntimeError(
                        f"Size mismatch for generalized input/output with index {index}!"
                        f"Generalized is {registered_xput_generalize}, default is {default_generalize}"
                    )
                for i, gt in enumerate(registered_xput_generalize):
                    _replace_generalize_info(gt, default_generalize[i])
            else:
                _replace_generalize_info(registered_xput_generalize, default_generalize)

    def _replace_attr_generalize_info(is_static, registered_attr_generalize, default_generalize):
        if "value" not in default_generalize:
            return
        default_generalize["value"] = registered_attr_generalize
        if is_static and "dtype" in default_generalize and default_generalize["dtype"] in ("float", "list_float"):
            default_generalize["value"] = None

    def _replace_info(x_json: dict, reg_json: Union[list, tuple], is_static: bool):
        dft_general_inputs = x_json.get("inputs", ())
        dft_general_outputs = x_json.get("outputs", ())
        dft_general_attrs = x_json.get("attrs", ())
        for idx, generalized_tensor in enumerate(reg_json):
            if idx < len(dft_general_inputs):
                # inputs
                _replace_xput_generalize_info(generalized_tensor, dft_general_inputs[idx], idx)
            elif idx < len(dft_general_inputs) + len(dft_general_outputs):
                # outputs
                _replace_xput_generalize_info(
                    generalized_tensor, dft_general_outputs[idx - len(dft_general_inputs)], idx
                )
            elif idx < len(dft_general_inputs) + len(dft_general_outputs) + len(dft_general_attrs):
                # attrs
                if generalized_tensor is not None and not isinstance(
                    generalized_tensor, (str, list, tuple, float, int, bool)
                ):
                    raise RuntimeError(f"Generalized attribute value invalid: {generalized_tensor} of index {idx}.")
                _replace_attr_generalize_info(
                    is_static,
                    generalized_tensor,
                    dft_general_attrs[idx - len(dft_general_inputs) - len(dft_general_outputs)],
                )
            else:
                pass

    if not registered_json:
        return
    if not isinstance(registered_json, (list, tuple)):
        raise RuntimeError(f"Invalid return value from registered generalize function: {registered_json}")
    for r in registered_json:
        if "result" in r:
            reason = r.get("reason", "Unknown")
            raise RuntimeError(f"Generalize failed, result: {r['result']}, reason: {reason}")
        if not isinstance(r, (list, tuple)):
            raise RuntimeError(f"Invalid element in generalized result: {r}")
        _replace_info(static_json, r, True)
        _replace_info(dynamic_json, r, False)


def kernel_match_by_static_key(
    testcase: TestcaseKernel,
    bin_cfg: dict,
    stc_generalize_info: dict,
    dyn_generalize_info: dict,
    optional_input_indices: list = None,
):
    # BinaryMatchWithStaticKeyAndDynInfo
    generate_build_options(testcase, stc_generalize_info)
    static_key = json_sha256(stc_generalize_info)
    matched_bin_list = get_matched_bin_list(static_key, bin_cfg)
    if not matched_bin_list:
        # miss match full inputs, try match again with optional input placeholder
        matched_bin_list = kernel_match_without_optional_inputs(bin_cfg, stc_generalize_info, optional_input_indices)
    matched_bin = None
    if matched_bin_list:
        matched_bin = match_op_params(matched_bin_list, dyn_generalize_info, optional_input_indices)
    return matched_bin


def generate_build_options(testcase: TestcaseKernel, stc_generalize_info: dict):
    # GenBuildOptions
    opt_json = {}
    # SocInfo_.l2Mode is skipped !!
    # OpInfo->GetOpImplMode()
    if "impl_mode" in testcase.attributes:
        opt_json.update({"implMode": testcase.attributes["impl_mode"]})
    # SocInfo_.opStatusCheck
    opt_json.update({"status_check": "true"})
    stc_generalize_info["buildOptions"] = opt_json


def json_sha256(stc_generalize_info: dict) -> str:
    str_to_hash = json.dumps(stc_generalize_info, separators=(",", ":"), sort_keys=True)
    logging.debug(f"Json to sha256 is {str_to_hash}")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str_to_hash.encode("utf-8"))
    return sha256_hash.hexdigest()


def kernel_match_without_optional_inputs(
    bin_cfg: dict, stc_generalize_info: dict, optional_input_indices: list = None
) -> Optional[list]:
    if not optional_input_indices or "inputs" not in stc_generalize_info:
        return None
    stc_generalize_info["inputs"] = [
        None if idx in optional_input_indices else inpt for idx, inpt in enumerate(stc_generalize_info["inputs"])
    ]
    static_key = json_sha256(stc_generalize_info)
    matched_bin_list = get_matched_bin_list(static_key, bin_cfg)
    if not matched_bin_list:
        return None
    filtered_bin_list = [mbl for mbl in matched_bin_list if mbl.get("optionalInputMode", "") == "gen_placeholder"]
    return filtered_bin_list


def get_matched_bin_list(static_key: str, bin_cfg: dict):
    bin_list = bin_cfg.get("binList", ())
    matched_bin_list = [
        b for b in bin_list if "staticKey" in b and isinstance(b["staticKey"], str) and static_key in b["staticKey"]
    ]
    logging.debug(f"Static key {static_key} matched bin count: {len(matched_bin_list)}")
    return matched_bin_list


def match_op_params(
    matched_bin_list: list, dyn_generalize_info: dict, optional_input_indices: list = None
) -> Optional[dict]:
    for b in matched_bin_list:
        if not match_deterministic(b, dyn_generalize_info):
            logging.debug("deterministic match failed.")
            continue
        if not match_xputs(b.get("inputs"), dyn_generalize_info.get("inputs"), optional_input_indices):
            logging.debug("inputs match failed.")
            continue
        if not match_xputs(b.get("outputs"), dyn_generalize_info.get("outputs")):
            logging.debug("outputs match failed.")
            continue
        if not match_attrs(b, dyn_generalize_info):
            logging.debug("attrs match failed.")
            continue
        return b
    return None


def match_deterministic(bin_cfg: dict, dyn_generalize_info: dict) -> bool:
    dyn_deterministic = dyn_generalize_info.get("deterministic", "false")
    bin_deterministic = bin_cfg.get("deterministic", "false")
    return bin_deterministic in ("ignore", dyn_deterministic) or dyn_deterministic == "ignore"


def match_xputs(
    bin_xputs: Optional[list], dyn_generalize_xputs: Optional[list], optional_input_indices: Optional[list] = None
) -> bool:
    if bin_xputs is None and dyn_generalize_xputs is None:
        return True
    if bin_xputs is None or dyn_generalize_xputs is None:
        return False
    if not isinstance(bin_xputs, (tuple, list)) or not isinstance(dyn_generalize_xputs, (tuple, list)):
        return False
    if len(bin_xputs) != len(dyn_generalize_xputs):
        return False
    for idx, dgx in enumerate(dyn_generalize_xputs):
        if dgx is None and optional_input_indices and idx in optional_input_indices:
            continue
        if dgx is not None and bin_xputs[idx] is None:
            return False
        if not match_range("range", bin_xputs[idx], dgx):  # ori_range is skipped
            return False
        if not match_const_value(bin_xputs[idx]):
            return False
        if not match_no_range_params(bin_xputs[idx], dgx):
            return False
        if optional_input_indices and idx in optional_input_indices and not match_optional_input(bin_xputs[idx], dgx):
            return False
    return True


def match_range(key, bin_xput: Optional[dict], dyn_generalize_xput: Optional[dict]) -> bool:
    def _replace(ele):
        return None if isinstance(ele, int) and ele < 1 else ele

    if bin_xput is None or key not in bin_xput:
        return True
    if dyn_generalize_xput is None or key not in dyn_generalize_xput:
        return False
    if len(bin_xput[key]) != len(dyn_generalize_xput[key]):
        return False
    bin_range = [[_replace(ele[0]), _replace(ele[1])] for ele in bin_xput[key]]
    dyn_range = [[_replace(ele[0]), _replace(ele[1])] for ele in dyn_generalize_xput[key]]
    debug_log = f"xput range mismatch. case range is {dyn_range}, binary range is {bin_range}"
    for idx, dr in enumerate(dyn_range):
        if bin_range[idx][0] is not None and (dr[0] is None or dr[0] < bin_range[idx][0]):
            logging.debug(debug_log)
            return False
        if bin_range[idx][1] is not None and (dr[1] is None or dr[1] > bin_range[idx][1]):
            return False
    return True


def match_const_value(bin_xput: Optional[dict]) -> bool:
    def _get(key):
        if bin_xput is None or key not in bin_xput:
            return None
        # like {"const_value": {"const_value": X}}
        return None if key not in bin_xput[key] else bin_xput[key]

    cv = _get("const_value")
    cvr = _get("const_value_range")
    return cv is None and cvr is None


def match_no_range_params(bin_xput: Optional[dict], dyn_generalize_xput: Optional[dict]) -> bool:
    if not match_param_in_json("ori_format", bin_xput, dyn_generalize_xput, None, ""):
        return False
    # addr_type, split_index, is_first_layer, slice_offset, valid_shape, total_shape, L1_*** are all skipped.
    return match_param_in_json(
        "ori_shape",
        bin_xput,
        dyn_generalize_xput,
        lambda b, d: all(x == -1 or x == d[i] for i, x in enumerate(b)),
        (),
        (-2,),
    )


def match_param_in_json(
    key, bin_xput: Optional[dict], dyn_generalize_xput: Optional[dict], custom_compare: Optional[Callable], *dft
) -> bool:
    if bin_xput is None or key not in bin_xput:
        return True
    if any(list(bin_xput[key]) == list(d) if isinstance(d, (list, tuple)) else bin_xput[key] == d for d in dft):
        return True
    if not dyn_generalize_xput or key not in dyn_generalize_xput:
        return False
    if any(
        list(dyn_generalize_xput[key]) == list(d) if isinstance(d, (list, tuple)) else dyn_generalize_xput[key] == d
        for d in dft
    ):
        return False
    bv, dv = bin_xput[key], dyn_generalize_xput[key]
    if isinstance(bv, (list, tuple)) and isinstance(dv, (list, tuple)):
        if len(bv) != len(dv):
            return False
        if list(bv) == list(dv):
            return True
        return bool(custom_compare and custom_compare(bv, dv))
    return bv == dv


def match_optional_input(bin_xput: Optional[dict], dyn_generalize_xput: Optional[dict]) -> bool:
    if not match_param_in_json("dtype", bin_xput, dyn_generalize_xput, None, ""):
        return False
    if not match_param_in_json("format", bin_xput, dyn_generalize_xput, None, ""):
        return False
    return match_param_in_json("shape", bin_xput, dyn_generalize_xput, None, ())


def match_attrs(bin_cfg: dict, dyn_generalize_info: dict) -> bool:
    bin_attrs = bin_cfg.get("attrs")
    dyn_attrs = dyn_generalize_info.get("attrs")
    if bin_attrs is None and dyn_attrs is None:
        return True
    if bin_attrs is None or dyn_attrs is None:
        return False
    if not isinstance(bin_attrs, (tuple, list)) or not isinstance(dyn_attrs, (tuple, list)):
        return False
    if len(bin_attrs) != len(dyn_attrs):
        return False
    for idx, ba in enumerate(bin_attrs):
        b_dtype = ba.get("dtype", None)
        d_dtype = dyn_attrs[idx].get("dtype", None)
        if not b_dtype or not d_dtype:
            return False
        if b_dtype != d_dtype:
            return False
        if d_dtype not in ("float", "list_float"):
            continue
        b_val = ba.get("value", None)
        if b_val is None:
            continue
        d_val = dyn_attrs[idx].get("value", None)
        if d_val is None:
            return False
        if not compare_attr_val(b_val, d_val, b_dtype):
            return False
    return True


def compare_attr_val(b_val, d_val, dtype: str) -> bool:
    if isinstance(b_val, (list, tuple)):
        if len(b_val) != len(d_val):
            return False
        return all(compare_attr_val(bv, d_val[idx], dtype) for idx, bv in enumerate(b_val))
    atomic_dtype = dtype.split("_")[-1]
    if dtype == "data_type" or atomic_dtype == "int":
        return b_val in (-1, d_val)
    if atomic_dtype == "float":
        return math.isclose(b_val, d_val)
    if atomic_dtype in ("bool", "string"):
        return b_val == d_val
    return False
