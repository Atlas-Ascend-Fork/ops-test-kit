#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
onnx 建模：纯函数组装单算子 ModelProto。

不读 testcase——eager_call（args/kwargs 反解）与 golden（testcase 平铺字段）
两侧喂同一函数，保证被测与 golden 同构。

规则：
  1. opset 取 `onnx.defs.onnx_opset_version()`（环境最高；onnx 包版本
     约束由 param_extractor 校验）；ir_version 用 onnx 包默认
     （IR 8 × opset 18 atc 通过）；
  2. 输入名固定 `input_0..N`（与 atc `--input_shape` 键一致），实际存在的
     输入按序编号；**尾部 Optional 输入直接省略**（node.inputs 短列表——`""`
     空串占位被 onnx parser 拒绝 E19999）；非尾部 Optional 输入
      省略不支持（省略位后仍有实际输入时，输入对位无法表达 → 显式报错）；
     Variadic 输入恒居尾（ai.onnx ≤18 注册表全量核实，无反例），非尾部
     Variadic 的元数分配有歧义 → 显式报错；
  3. 属性经 schema 类型归一后 `make_attribute` 直写（值类型词对齐
     coerce_value 产物与 CSV 原值两条入口，保证两侧同构）；未知属性 /
     缺失必选属性 / 支持集外的属性类型（TENSOR/GRAPH 载荷——CSV 路径
     由 param_extractor 解析期前置拒绝，公开面直调在此兜底）在建模期
     显式报错（CSV 属性拼写错误不静默落 extra_attrs）；
  4. 输出 ValueInfo 只给名字（elem_type=0、shape=None），靠 atc shape
     inference；**Optional 输出省略**（尾部短列表，
     atc 拒转声明了 Optional 输出的模型[MaxPool.Indices / Dropout.mask
     探针实证]，ReferenceEvaluator 对 BatchNorm/Dropout 亦不产出；全 Optional
     输出算子[GRU/RNN/LSTM]显式报错）；Variadic 输出的元数按 ONNX 惯例解析：
     `num_outputs` INT 属性（Split@18）优先（在场但非正整数 → 按算子/属性点名
     显式报错），其次尾段 1-D 整型 Optional sizes 输入（如 Split 的
     `split`，其长度即输出数），均无 → 显式报错（Split 是 ai.onnx ≤18 唯一
     带 Variadic 输出且无 GRAPH 属性的算子；机制按属性/输入
     形态泛化，不按算子名分支）；
  5. 不做 `infer_shapes` 预推理（atc 自做，避免双份推理不一致）；**不调
     `onnx.checker.check_model`**（name-only 输出过不了 checker 校验）；
  6. 进程级 LRU：`(op_type, opset, input_specs, attrs)` → 序列化字节
     （eager/golden 同键复用；条目为 bytes，返回值按需反序列化，缓存不可
     被调用方变异）。单线程消费（e2e 顺序执行），不加锁。

`build_input_specs`（输入侧反解）：ParamPlan.build_args
产物 args（含 None 占位）→ (input_specs, host_inputs)，Variadic 双产物形态
兼容（CSV 嵌套 TensorList 单列表位 / flat 逐张量多位）；无 Variadic 吸收的
超发 args 显式报错（CSV 路径经 match_overload 前置拦截，公开面直调兜底）；
`build_model_bytes` 为字节形态公开面（om_compiler 落盘入口），与 proto 形态
共用同一条 LRU 路径。
"""

from collections import OrderedDict
from typing import Optional

from ttk.utilities.dtypes import dtype_to_name, str_to_onnx_dtype

__all__ = ["build_model_proto", "build_model_bytes", "build_input_specs"]

# 进程级模型缓存：key → SerializeToString 字节。
# 单算子模型小体量（KB 级），128 条上限仅防长跑多用例无界增长。
_MODEL_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_MODEL_CACHE_MAX = 128

# 支持集与 param_extractor._ATTR_TYPE_WORDS 对齐（CSV 路径解析期已前置
# 拒绝；此处兜底 build_model_* 公开面直调）。
_ATTR_SCALAR_KINDS = {"INT": int, "FLOAT": float, "STRING": str}
_ATTR_LIST_KINDS = {"INTS": int, "FLOATS": float, "STRINGS": str}


def _freeze(value):
    """LRU 键组件归一：list/dict → 可哈希 tuple（递归）。"""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def _attr_error_prefix(op_type: str) -> str:
    return f"onnx.{op_type}"


def _normalize_attrs(op_type: str, schema, attrs: dict) -> dict:
    """CSV/kwargs 属性 → schema 类型归一值（int/float/str/列表）。

    未知属性 / 缺失必选属性 / 支持集外的属性类型（TENSOR/GRAPH 载荷——
    CSV 路径由 param_extractor 解析期前置拒绝，此处兜底公开面直调）显式
    报错；INTS/FLOATS/STRINGS 接受标量或序列（对齐 coerce_value 的标量→
    单元序列宽容）。
    """
    prefix = _attr_error_prefix(op_type)
    unknown = sorted(set(attrs) - set(schema.attributes))
    if unknown:
        raise ValueError(
            f"{prefix}: unknown attribute(s) {unknown}; declared attributes are {sorted(schema.attributes)}"
        )
    missing = sorted(name for name, attr in schema.attributes.items() if attr.required and name not in attrs)
    if missing:
        raise ValueError(f"{prefix}: required attribute(s) {missing} missing from testcase attributes")

    normalized = {}
    for name, value in attrs.items():
        kind = str(schema.attributes[name].type).rsplit(".", 1)[-1]
        if kind not in _ATTR_SCALAR_KINDS and kind not in _ATTR_LIST_KINDS:
            # 支持集外的 kind（TENSOR/GRAPH(S)/SPARSE_TENSOR/TYPE_PROTO 等）
            # 无标量转换路径；裸 KeyError 不带定位信息（直调 If/Loop/Constant
            # 类算子时触发）
            raise ValueError(
                f"{prefix}: attribute '{name}' has unsupported schema type {kind}; "
                f"only {sorted({**_ATTR_SCALAR_KINDS, **_ATTR_LIST_KINDS})} are buildable "
                f"(TENSOR/GRAPH-carrying ops like If/Loop/Constant are out of scope)"
            )
        try:
            if kind in _ATTR_SCALAR_KINDS:
                normalized[name] = _ATTR_SCALAR_KINDS[kind](value)
            else:  # INTS / FLOATS / STRINGS
                items = list(value) if isinstance(value, (list, tuple)) else [value]
                normalized[name] = [_ATTR_LIST_KINDS[kind](v) for v in items]
        except (ValueError, TypeError) as e:
            # 转换失败按算子/属性点名：裸 invalid literal 不带定位信息
            raise ValueError(f"{prefix}: attribute '{name}' expects {kind}, cannot convert {value!r}") from e
    return normalized


def _assign_node_inputs(op_type: str, schema_inputs, input_specs) -> tuple:
    """schema 输入序 × 实际输入 specs → (node 输入名序列, Optional 消费的 (shape, dtype) 列表)。

    位置贪心：Single 消费 1 个；Optional 有则消费、无则省略（省略仅当其后
    无 Single——否则后续 Single 断供报错，即非尾部省略不支持）；Variadic
    （恒居尾）吸收全部剩余，0 个显式报错（minArity 不落 ParamInfo 的建模期
    对齐）。剩余未消费 → 输入数超 schema 容量。
    """
    prefix = _attr_error_prefix(op_type)
    names = [spec[0] for spec in input_specs]
    specs = [(tuple(spec[1]), spec[2]) for spec in input_specs]
    node_inputs = []
    optional_specs = []
    idx = 0
    for pos, formal in enumerate(schema_inputs):
        option = str(formal.option).rsplit(".", 1)[-1]
        if option == "Variadic":
            if pos != len(schema_inputs) - 1:
                raise ValueError(
                    f"{prefix}: Variadic input '{formal.name}' is not in trailing position; "
                    f"arity assignment for non-trailing Variadic inputs is unsupported"
                )
            if idx >= len(names):
                raise ValueError(
                    f"{prefix}: Variadic input '{formal.name}' requires at least 1 tensor "
                    f"(minArity is not exposed by the onnx schema registry; all ai.onnx "
                    f"Variadic inputs declare minArity >= 1)"
                )
            node_inputs.extend(names[idx:])
            idx = len(names)
        elif option == "Single":
            if idx >= len(names):
                raise ValueError(
                    f"{prefix}: required input '{formal.name}' not provided — only trailing "
                    f"Optional inputs may be omitted (non-trailing Optional omission is "
                    f"unsupported: '' placeholder is rejected by the onnx parser)"
                )
            node_inputs.append(names[idx])
            idx += 1
        elif idx < len(names):  # Optional：有则消费、无则省略（尾部短列表）
            node_inputs.append(names[idx])
            optional_specs.append(specs[idx])
            idx += 1
    if idx != len(names):
        raise ValueError(
            f"{prefix}: {len(names)} inputs provided but schema consumes only {idx} "
            f"(input params: {[f.name for f in schema_inputs]})"
        )
    return node_inputs, optional_specs


def _variadic_output_count(op_type: str, normalized_attrs: dict, optional_specs: list) -> int:
    """Variadic 输出元数解析（建模期静态可知源，见模块 docstring 规则 4）。

    ``num_outputs`` 在场但非正整数时按算子/属性点名显式报错（不静默
    回退到「无法确定」的误导信息——用户已提供该属性，回退信息会指向错误
    的修复方向）；sizes 输入候选须同时满足 1-D 与整型 dtype（Split 类逐输出
    定长输入的 schema 形态——float 1-D 输入不是元数来源）。
    """
    prefix = _attr_error_prefix(op_type)
    count = normalized_attrs.get("num_outputs")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count <= 0):
        raise ValueError(
            f"{prefix}: attribute 'num_outputs' must be a positive int to pin the Variadic output count, got {count!r}"
        )
    if count is not None:
        return count
    for shape, dtype in reversed(optional_specs):
        if len(shape) == 1 and str(dtype).startswith(("int", "uint")):
            return int(shape[0])
    raise ValueError(
        f"{prefix}: cannot determine Variadic output count — provide the 'num_outputs' "
        f"attribute or a 1-D integer sizes input (e.g. Split's 'split'); data-dependent "
        f"output arity is unsupported"
    )


def _output_names(op_type: str, schema_outputs, normalized_attrs: dict, optional_specs: list) -> list:
    """输出名序列 `output_0..N`（Single + Variadic 展开；Optional 输出省略）。

    编号对「实际产出」紧凑连续：Variadic 自当前 len(names) 续编而非
    range(count) 重编，混合 Single+Variadic 形态不重名（当前注册表无此
    形态，机制泛化见模块 docstring 规则 4）。

    Optional 输出（MaxPool.Indices / Dropout.mask / BatchNormalization.running_*）
    省略为尾部短列表语义——实测（CANN 9.2.0）atc 对声明了 Optional 输出的模型直接
    拒转（MaxPool 双输出 EZ9999「The output of Indices is not support」/ Dropout
    双输出 E19999「Cannot find output tensor desc」），BatchNorm/Dropout 的
    ReferenceEvaluator 亦不产出该位——「非 Optional + Variadic 推断」是
    atc/reference 双侧可编译的最大公约面。
    省略仅限尾部（其后无非 Optional 输出）：ONNX 输出位无空串占位语义，跳过
    非尾部 Optional 会使后续输出错位绑定 schema 位；全 Optional 输出算子
    （GRU/RNN/LSTM）省略后零输出，建模期显式报错。
    """
    prefix = _attr_error_prefix(op_type)
    options = [str(formal.option).rsplit(".", 1)[-1] for formal in schema_outputs]
    for pos, option in enumerate(options):
        if option == "Optional" and any(o != "Optional" for o in options[pos + 1 :]):
            raise ValueError(
                f"{prefix}: Optional output '{schema_outputs[pos].name}' is followed by non-Optional "
                f"outputs — omitting it would shift later outputs onto wrong schema positions "
                f"(ONNX output slots have no '' placeholder semantics); unsupported"
            )
    names = []
    for option in options:
        if option == "Optional":
            continue
        if option == "Variadic":
            # 续编号而非 range(count) 重编：混合 Single+Variadic 输出时前导
            # Single 已占 output_0，重编会重名（当前注册表无此形态——ai.onnx
            # ≤26 全量扫描仅纯 Variadic；机制泛化对齐模块 docstring 规则 4）
            count = _variadic_output_count(op_type, normalized_attrs, optional_specs)
            base = len(names)  # 基准先落：genexp 惰性求值 len(names) 会随 extend 逐元素追加而漂移
            names.extend(f"output_{base + i}" for i in range(count))
        else:
            names.append(f"output_{len(names)}")
    if not names:
        raise ValueError(
            f"{prefix}: all outputs are Optional (omitted per atc limitation) — cannot build "
            f"a zero-output model; this op is unsupported (compare via plugin golden "
            f"is not possible either: the device leg cannot produce the omitted outputs)"
        )
    return names


def _build(op_type: str, input_specs, attrs: dict, opset: int):
    import onnx
    import onnx.defs
    from onnx import helper

    try:
        schema = onnx.defs.get_schema(op_type)
    except onnx.defs.SchemaError as e:
        raise ValueError(f"onnx.{op_type}: no schema registered in the environment registry ({e})") from e

    input_value_infos = []
    for name, shape, dtype in input_specs:
        elem_type = str_to_onnx_dtype(dtype)
        if elem_type == 0:
            raise ValueError(
                f"onnx.{op_type}: input '{name}' dtype {dtype!r} has no onnx TensorProto enum "
                f"(TTK canonical {dtype!r} is not in ttk.utilities.dtypes.ONNX_DTYPE_MAP; "
                f"unmapped by design: uint1/int1, hifloat8/hifloat4, float4_e1m2, complex32, string)"
            )
        dims = tuple(int(d) for d in shape)
        if any(d < 0 for d in dims):
            raise ValueError(
                f"onnx.{op_type}: input '{name}' has dynamic dim(s) {dims}; "
                f"only static input shapes are supported (atc dynamic shape is a non-goal)"
            )
        input_value_infos.append(helper.make_tensor_value_info(name, elem_type, dims))

    node_inputs, optional_specs = _assign_node_inputs(op_type, schema.inputs, input_specs)
    normalized_attrs = _normalize_attrs(op_type, schema, attrs)
    output_names = _output_names(op_type, schema.outputs, normalized_attrs, optional_specs)

    node = helper.make_node(
        op_type,
        node_inputs,
        output_names,
        name=f"{op_type.lower()}_0",
        **normalized_attrs,
    )
    # 输出 ValueInfo 只给名字：elem_type=0、shape=None（atc shape inference）。
    graph = helper.make_graph(
        [node],
        f"ttk_{op_type.lower()}",
        input_value_infos,
        [helper.make_tensor_value_info(name, 0, None) for name in output_names],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def build_model_bytes(op_type: str, input_specs: list, attrs: Optional[dict] = None) -> bytes:
    """组装单算子 ModelProto 并序列化（进程级 LRU，缓存条目即返回字节）。

    om_compiler 的落盘入口；参数契约与 ``build_model_proto`` 一致，
    两条公开面共用同一条缓存路径（同键命中直接返回字节，不再反序列化）。

    Raises:
        ValueError: 同 ``build_model_proto``。
    """
    import onnx.defs

    attrs = attrs or {}
    opset = onnx.defs.onnx_opset_version()
    key = (op_type, opset, _freeze(list(input_specs)), _freeze(attrs))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        _MODEL_CACHE.move_to_end(key)
        return cached
    data = _build(op_type, list(input_specs), attrs, opset).SerializeToString()
    _MODEL_CACHE[key] = data
    while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)
    return data


def build_model_proto(op_type: str, input_specs: list, attrs: Optional[dict] = None):
    """组装单算子 ModelProto（纯函数 + 进程级 LRU）。

    Args:
        op_type: ONNX 算子类型（如 ``Add``，不带 onnx. 前缀与 opset 版本）。
        input_specs: ``[(name, shape_tuple, canonical_dtype_str), ...]`` —— 仅
            "实际存在"的输入，按 schema 输入序（None 占位槽由调用方剔除，
            args → specs 反解见 build_input_specs）。
        attrs: ``{attr_name: python_value}`` —— 仅 CSV 声明的属性（缺省属性
            走算子默认，不写入节点）。

    Returns:
        onnx.ModelProto（每次调用由序列化字节反序列化，调用方可安全变异，
        缓存不可被返回值污染）。

    Raises:
        ValueError: dtype 无 TensorProto 映射 / 动态输入 dim / 输入对位失败
            （缺必选输入、非尾部 Variadic、Variadic 0 输入、输入超量）/ 未知
            或缺失必选属性 / 属性类型支持集外（TENSOR/GRAPH 载荷，公开面
            直调兜底）/ Variadic 输出元数不可解析 / schema 不存在。
    """
    import onnx

    return onnx.load_model_from_string(build_model_bytes(op_type, input_specs, attrs))


def build_input_specs(param_info, args) -> tuple:
    """ParamPlan.build_args 产物 args（含 None 占位）→ (input_specs, host_inputs)。

    eager_call 的输入侧反解：args 按参数模型对位——
    每个 Single/Optional 输入占一位（省略位为 None），Variadic 输入吸收其余
    全部位，兼容两种产物形态：CSV 嵌套 TensorList（单列表位，逐项展开）与
    flat 逐张量多位（build_args 对 Variadic 参数的另一种展开产物）。onnx 域
    属性一律 kw-only，args 中只有输入张量位，无标量混入。

    省略的 Optional 输入必须居尾（其后无在场输入）——atc 拒绝空串占位
    （E19999），非尾部省略不支持（显式报错）。输入名按实际
    在场序编号 input_0..N（与 atc --input_shape 键一致）；dtype 取自实际数组
    （Promote 时 golden 侧拿到的是抬精度后的数组，模型随数组精度走）。
    无 Variadic 吸收时的超发 args 显式报错——静默丢弃会把丢参/错位掩盖
    成正常构建（CSV 路径经 match_overload 前置拦截，此处兜底公开面直调）。

    Args:
        param_info: ``extract_onnx_params`` 产出的 APIParamInfo。
        args: ``ParamPlan.build_args`` 的 args 位（kwargs 即 attrs，不经本函数）。

    Returns:
        (input_specs, host_inputs)：specs 直接喂 ``build_model_proto`` /
        ``build_model_bytes``，host_inputs 为与 specs 对位的宿主数组列表。

    Raises:
        ValueError: 无 Variadic 参数时 args 超出已声明张量参数数（多余项
            会被静默丢弃 → 显式拒绝）；非尾部 Optional 输入省略。
    """
    input_params = [p for p in param_info.params if p.is_tensor_like]
    slots = []
    idx = 0
    for param in input_params:
        if param.is_var_positional:
            rest = args[idx:]
            idx = len(args)
            if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
                slots.extend((param, item) for item in rest[0])
            else:
                slots.extend((param, value) for value in rest)
        else:
            slots.append((param, args[idx] if idx < len(args) else None))
            idx += 1

    if idx < len(args):
        # Variadic 分支已将 idx 推至 len(args)（吸收全部），此处仅无 Variadic
        # 时的超发可达；尾缺是合法的 Optional 省略语义，不反向检查
        api = getattr(param_info, "api_name", None) or "onnx api"
        raise ValueError(
            f"{api}: {len(args) - idx} extra positional arg(s) beyond the {idx} declared "
            f"tensor inputs ({[p.name for p in input_params]}) — extra args would be "
            f"silently dropped; pass only the tensors declared by the param plan"
        )

    last_provided = max((pos for pos, (_p, v) in enumerate(slots) if v is not None), default=-1)
    for pos, (param, value) in enumerate(slots):
        if value is None and pos < last_provided:
            raise ValueError(
                f"ONNX_OPTIONAL_INPUT_UNSUPPORTED: input '{param.name}' (slot {pos}) is omitted "
                f"while later inputs are present — non-trailing Optional omission is unsupported "
                f"(the onnx parser rejects '' placeholders); "
                f"fix the testcase's tensor_view_shapes declaration"
            )

    input_specs = []
    host_inputs = []
    for _param, arr in slots:
        if arr is None:
            continue
        name = f"input_{len(input_specs)}"
        input_specs.append((name, tuple(int(d) for d in arr.shape), dtype_to_name(arr.dtype)))
        host_inputs.append(arr)
    return input_specs, host_inputs
