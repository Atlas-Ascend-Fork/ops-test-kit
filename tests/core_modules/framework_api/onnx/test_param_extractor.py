# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""param_extractor 单测（纯 CPU，依赖 onnx 包；期望值取自环境注册表，不锚定版本）。

覆盖：Add/Concat/Split 参数模型（单输入 / Variadic / Optional 输入 / 属性
kw-only）、未知算子拒绝路径、opset 门禁（版本无关触发）、ParamPlan 装配契约
（输出槽语义 + Variadic 嵌套）。

跨环境：onnx 缺席 / opset 超限经 conftest.onnx_registry 处理（映射逻辑与
opset 无关，门禁 bypass；门禁行为由 test_opset_version_gate 独立钉住）；
Split 的 num_outputs 语义经 split_with_num_outputs 能力探针（注册表实际
声明判定，旧 onnx 缺该属性时 skip）。
"""

import pytest

from ttk.core_modules.framework_api.api_model import APIParamInfo
from ttk.core_modules.framework_api.api_param import build_positional_args
from ttk.core_modules.framework_api.onnx.param_extractor import extract_onnx_params

pytestmark = pytest.mark.usefixtures("onnx_registry")


def _by_name(info):
    return {p.name: p for p in info.params}


# ── 基础算子参数模型 ─────────────────


def test_add_two_single_inputs():
    import onnx.defs

    info = extract_onnx_params("onnx.Add")
    assert isinstance(info, APIParamInfo)
    # source 的 since_version 取环境注册表（不锚定具体数值）
    assert info.source == f"onnx.schema(Add:{onnx.defs.get_schema('Add').since_version})"
    params = _by_name(info)
    assert set(params) == {"A", "B"}
    for p in params.values():
        assert p.type == "Tensor"
        assert p.is_tensor
        assert not p.is_tensor_list
        assert not p.is_optional
        assert not p.is_keyword_only
        assert not p.is_var_positional
        assert not p.is_mutable
    assert info.tensor_distribution == (0, 0)


def test_concat_variadic_input_plus_required_attr():
    import onnx.defs

    info = extract_onnx_params("onnx.Concat")
    assert info.source == f"onnx.schema(Concat:{onnx.defs.get_schema('Concat').since_version})"
    params = _by_name(info)
    inputs = params["inputs"]
    assert inputs.type == "tuple of Tensors"
    assert inputs.is_tensor_list
    assert inputs.is_var_positional
    assert not inputs.is_optional
    axis = params["axis"]
    assert axis.type == "int"
    assert axis.is_keyword_only
    assert not axis.is_optional  # 必选属性：CSV 必须声明
    assert axis.default is None
    assert info.tensor_distribution == (-1,)


def test_split_optional_input_and_variadic_output(split_with_num_outputs):
    import onnx.defs

    info = extract_onnx_params("onnx.Split")
    assert info.source == f"onnx.schema(Split:{onnx.defs.get_schema('Split').since_version})"
    params = _by_name(info)
    assert params["input"].type == "Tensor"
    assert not params["input"].is_optional
    # Optional 输入：类型词剥壳为 Tensor、可选性由 is_optional 承载（torch 侧
    # _normalize_aten_schema_type 的 Optional[...] 剥壳同款约定）
    split = params["split"]
    assert split.type == "Tensor"
    assert split.is_tensor
    assert split.is_optional
    axis, num_outputs = params["axis"], params["num_outputs"]
    for attr in (axis, num_outputs):
        assert attr.is_keyword_only
        assert attr.is_optional
    assert axis.type == "int"
    assert axis.default == 0
    assert num_outputs.type == "int"
    # num_outputs 无 schema 编码默认（proto tag=UNDEFINED）→ None，不伪造 0
    assert num_outputs.default is None
    # _return_counts=[len(schema.outputs)]=[1]（单 Variadic 输出参数），
    # 无截断路径；多输出对账走 output_tensor_indexes + 执行侧输出数
    assert info._return_counts == [1]
    assert len(info.overloads) == 1
    assert info.overloads[0].return_count == 1


# ── 拒绝路径 ──────────────────────────────────────────────────────────────


def test_unknown_op_returns_none():
    assert extract_onnx_params("onnx.NoSuchOpXYZ") is None  # 收敛 API_PARSE_FAIL


def test_opset_version_gate(monkeypatch):
    """opset 门禁：环境 opset 超 ATC_MAX_OPSET 时解析期拒绝。

    版本无关触发——把上限压到「环境 opset - 1」，任意 onnx 版本可跑，不
    依赖具体数值（上限本身的维护语义见 param_extractor.ATC_MAX_OPSET）。
    """
    import onnx.defs

    from ttk.core_modules.framework_api.onnx import param_extractor

    monkeypatch.setattr(param_extractor, "ATC_MAX_OPSET", onnx.defs.onnx_opset_version() - 1)
    with pytest.raises(ValueError, match="ONNX_OPSET_VERSION_UNSUPPORTED"):
        param_extractor.extract_onnx_params("onnx.Add")


# ── ParamPlan 装配契约（eager_call 反解的输入侧锚点）───────────────────


def test_param_plan_assembly(split_with_num_outputs):
    """eager_call 反解的输入侧锚点：输出槽剔除 + Optional None 占位
    （build_positional_args / build_input_specs 同一套反解）。"""
    import numpy as np

    info = extract_onnx_params("onnx.Split")
    nested = [np.zeros((6, 4), dtype=np.float32), None, None, None]
    args, kwargs, _ = build_positional_args("onnx.Split", nested, {"axis": 0, "num_outputs": 3}, [1, 2, 3], None, info)
    assert len(args) == 2
    assert args[0].shape == (6, 4)
    assert args[1] is None
    assert kwargs == {"axis": 0, "num_outputs": 3}

    # Variadic 槽以嵌套列表整体占一个 args 位（torch.cat 同构）；eager_call
    # 反解时按序展开为 input_0..N。
    info = extract_onnx_params("onnx.Concat")
    nested = [[np.zeros((2, 3), dtype=np.int32), np.zeros((2, 4), dtype=np.int32)]]
    args, kwargs, _ = build_positional_args("onnx.Concat", nested, {"axis": 1}, [], None, info)
    assert len(args) == 1
    assert isinstance(args[0], list)
    assert len(args[0]) == 2
    assert kwargs == {"axis": 1}
