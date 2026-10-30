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
onnx 参数解析：onnx.defs schema（环境注册表，单源权威）→ APIParamInfo。

无多级回退链：schema 注册表即权威元数据（对比 torch 的六级链）。
opset 一律取环境注册表最新——get_schema 不带版本参数；
onnx 包版本约束（`onnx_opset_version() ≤ ATC_MAX_OPSET`）在此
校验，超限时明确报错并提示 onnx 版本约束，替代 atc 裸 E13010。

参数映射：
  - 输入 Single    → type="Tensor"
  - 输入 Variadic  → type="tuple of Tensors" + is_var_positional
                     （CSV 嵌套 TensorList 语义直接复用）
  - 输入 Optional  → type="Tensor" + is_optional——不用 "Optional[Tensor]" 记法，
                     与 torch 侧 `_normalize_aten_schema_type` 的 Optional[...]
                     剥壳约定一致（ParamInfo.is_tensor_like 按精确类型词匹配，
                     可选性由 is_optional 承载）
  - 属性           → 一律 kw-only（build_args 落 kwargs，eager_call 反解
                     零歧义）；required=False 带 default → is_optional + default
                     （CSV 缺省时走算子默认，见 model_builder 属性直写规则）
  - 属性类型       → INT/FLOAT/STRING → int/float/str；INTS/FLOATS/STRINGS →
                     tuple of ints/floats/strs；其余（TENSOR(S)/SPARSE_TENSOR(S)/
                     GRAPH(S)/TYPE_PROTO(S)）解析期拒绝 ONNX_ATTR_TYPE_UNSUPPORTED
                     ——onnx AttrType 无 BOOL（ai.onnx 注册表无此类型属性）
  - is_mutable 恒 False（ONNX 函数式 IR 无原地写）；overloads 恒单条（无
    Python 语义重载，opset 历史版本不展开——环境最新即权威）
  - _return_counts=[len(schema.outputs)]（输出参数个数，无截断
    路径，多输出对账走 output_tensor_indexes + 执行侧输出数）

已知边界：Variadic 的 minArity 不落 ParamInfo（onnx
1.13 的 FormalParameter 未导出该字段）——0 输入用例过校验后在建模期报错
（model_builder 的错误信息按算子/参数点名）。
"""

import logging
from typing import Optional

from ..api_model import APIParamInfo, ParamInfo

__all__ = ["extract_onnx_params", "parse_onnx_api_name", "ATC_MAX_OPSET"]

# atc onnx 算子插件注册的 opset 上限（随 CANN 版本维护；实测：
# cann-9.2.0 按 ai.onnx::<opset>::<OpType> 精确匹配、标准算子注册至 18 段）。
# 环境须满足 onnx.defs.onnx_opset_version() ≤ 该值（当前对应 onnx 1.13.x）。
ATC_MAX_OPSET = 18

_ONNX_API_PREFIX = "onnx."

# OpSchema.AttrType → ParamInfo 类型词（coerce_value 既有词汇）。
# 支持集外的属性类型（TENSOR(S)/SPARSE_TENSOR(S)/GRAPH(S)/TYPE_PROTO(S)）
# 无法从 CSV 标量属性构造 → 解析期拒绝（ONNX_ATTR_TYPE_UNSUPPORTED）。
_ATTR_TYPE_WORDS = {
    "INT": "int",
    "FLOAT": "float",
    "STRING": "str",
    "INTS": "tuple of ints",
    "FLOATS": "tuple of floats",
    "STRINGS": "tuple of strs",
}


def _attr_python_default(attr_def) -> Optional[object]:
    """schema 属性 default_value（AttributeProto）→ Python 值。

    按 AttributeProto 自身的 type tag 解码：onnx schema 仅在显式编码默认值时
    填充该 proto，tag=UNDEFINED 即「无默认」→ None（ai.onnx ≤18 可建
    类型属性共 81 个不带 proto 默认，如 GRU.clip/hidden_size、Split.num_outputs；
    按 schema 声明类型读空 proto 会伪造 0/0.0/""/()，经 build_args 补位写入
    节点后篡改算子语义）。仅处理 _ATTR_TYPE_WORDS 覆盖的类型（覆盖集外已
    在上游整体拒绝）。
    """
    from onnx import AttributeProto

    value = attr_def.default_value
    if value.type == AttributeProto.UNDEFINED:
        return None
    if value.type == AttributeProto.INT:
        return int(value.i)
    if value.type == AttributeProto.FLOAT:
        return float(value.f)
    if value.type == AttributeProto.STRING:
        return value.s.decode("utf-8", errors="replace")
    if value.type == AttributeProto.INTS:
        return tuple(int(v) for v in value.ints)
    if value.type == AttributeProto.FLOATS:
        return tuple(float(v) for v in value.floats)
    if value.type == AttributeProto.STRINGS:
        return tuple(v.decode("utf-8", errors="replace") for v in value.strings)
    return None


def parse_onnx_api_name(api_name: str):
    """api_name → (op_type, opset|None)。

    - ``onnx.<OpType>``：未声明 opset → None（建模用环境注册表最新）
    - ``onnx.<opset>.<OpType>``：数字段即显式 opset（atc 按
      ai.onnx::<opset>::<OpType> 精确注册算子，高版本 onnx 包可借此向下建模）
    - 非数字段与更长的 dotted 名：域限定形态，报 ONNX_DOMAIN_UNSUPPORTED

    返回 None 表示非 onnx 域（调用方静默跳过）；空 OpType（onnx.）同 None，
    带点的空段（onnx.13.）按域限定报错。
    """
    if not isinstance(api_name, str) or not api_name.startswith(_ONNX_API_PREFIX):
        return None
    rest = api_name[len(_ONNX_API_PREFIX) :]
    if not rest:
        return None
    if "." not in rest:
        return rest, None
    qualifier, op_type = rest.split(".", 1)
    if not op_type or "." in op_type or not qualifier or not qualifier.isdigit():
        raise ValueError(
            f"ONNX_DOMAIN_UNSUPPORTED: {api_name!r} carries a domain qualifier; "
            f"only the default ai.onnx domain (api_name = onnx.<OpType> or "
            f"onnx.<opset>.<OpType>) is supported"
        )
    return op_type, int(qualifier)


def _check_onnx_opset_version() -> None:
    """onnx 包版本约束校验（裸名建模用 env 最高 opset）：超 atc 插件上限明确报错。"""
    import onnx.defs

    opset = onnx.defs.onnx_opset_version()
    if opset > ATC_MAX_OPSET:
        raise ValueError(
            f"ONNX_OPSET_VERSION_UNSUPPORTED: onnx.defs.onnx_opset_version()={opset} "
            f"exceeds the atc plugin ceiling ({ATC_MAX_OPSET}); install an onnx package "
            f"whose opset version is <= {ATC_MAX_OPSET} (e.g. onnx 1.13.x for CANN 9.2.x), "
            f"or pin the opset explicitly in the api_name (onnx.{ATC_MAX_OPSET}.<OpType>)"
        )


def extract_onnx_params(api_name: str) -> Optional[APIParamInfo]:
    """onnx.defs schema（环境注册表最新）→ APIParamInfo。

    Args:
        api_name: ``onnx.<OpType>``（建模 opset = 环境注册表最新）或
            ``onnx.<opset>.<OpType>``（显式 opset，见 parse_onnx_api_name）。

    Returns:
        APIParamInfo（单 overload）；schema 不存在 → None（上游收敛 API_PARSE_FAIL）。

    Raises:
        ValueError: domain 不支持 / 属性类型不支持 / opset 超约束（显式段超
            ATC_MAX_OPSET 或裸名环境 opset 超限），消息带 ONNX_*_UNSUPPORTED
            语义前缀（log 透传后落 API_PARSE_FAIL）。
    """
    parsed = parse_onnx_api_name(api_name)
    if parsed is None:
        return None
    op_type, explicit_opset = parsed

    import onnx.defs

    if explicit_opset is None:
        # 裸名：建模 opset = env 最高，超 atc 插件上限即整通路不支持
        _check_onnx_opset_version()
    elif not 1 <= explicit_opset <= ATC_MAX_OPSET:
        raise ValueError(
            f"ONNX_OPSET_VERSION_UNSUPPORTED: {api_name!r} pins opset {explicit_opset}, "
            f"outside the atc plugin ceiling [1, {ATC_MAX_OPSET}]"
        )

    try:
        schema = onnx.defs.get_schema(op_type)
    except onnx.defs.SchemaError as e:
        logging.warning(f"Cannot resolve onnx schema for {api_name}: {e}")
        return None

    unsupported = [
        f"{name}({str(attr.type).rsplit('.', 1)[-1]})"
        for name, attr in schema.attributes.items()
        if str(attr.type).rsplit(".", 1)[-1] not in _ATTR_TYPE_WORDS
    ]
    if unsupported:
        raise ValueError(
            f"ONNX_ATTR_TYPE_UNSUPPORTED: {api_name} declares attribute(s) "
            f"{', '.join(unsupported)}; only {sorted(_ATTR_TYPE_WORDS)} are buildable "
            f"from CSV scalar attributes"
        )

    params = []
    for formal in schema.inputs:
        option = str(formal.option).rsplit(".", 1)[-1]
        if option == "Variadic":
            params.append(
                ParamInfo(
                    name=formal.name,
                    type="tuple of Tensors",
                    is_var_positional=True,
                )
            )
        else:
            # Single / Optional：类型词同源（Optional[...] 剥壳，见模块 docstring），
            # 可选性由 is_optional 承载（CSV None 槽 / build_args None 占位语义）。
            params.append(
                ParamInfo(
                    name=formal.name,
                    type="Tensor",
                    is_optional=option == "Optional",
                )
            )
    for name, attr in schema.attributes.items():
        kind = str(attr.type).rsplit(".", 1)[-1]
        has_default = not attr.required
        params.append(
            ParamInfo(
                name=name,
                type=_ATTR_TYPE_WORDS[kind],
                default=_attr_python_default(attr) if has_default else None,
                is_optional=has_default,
                is_keyword_only=True,
            )
        )

    info = APIParamInfo(
        api_name=api_name,
        params=params,
        source=f"onnx.schema({op_type}:{schema.since_version})",
        overloads=[params],
        _return_counts=[len(schema.outputs)],
    )
    logging.debug(f"Parsed {api_name}: {len(params)} params from onnx.schema({op_type}:{schema.since_version})")
    return info
