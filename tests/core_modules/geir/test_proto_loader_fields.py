# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ProtoLoader 增量字段：optional_inputs / required_attrs（图模式校验 R2/REQUIRED_ATTR 依赖）。

既有字段形状不动——inputs 仍按 _ANY_INPUT_PATTERN 混合捕获声明序、attrs 仍为
ATTR+REQUIRED_ATTR 拼接；新字段只做纯增量提取。夹具覆盖 DropOutV3 型
"OPTIONAL_INPUT 夹在必选中间"与 DYNAMIC_INPUT 混排，防止按位置假设的解析错位。
"""

from ttk.core_modules.geir.proto_loader import ProtoLoader

_IR = """
REG_OP(MixedPorts)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .DYNAMIC_INPUT(extra, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .DYNAMIC_OUTPUT(aux, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(num_split, Int)
    .ATTR(alpha, Float, 1.0)
    .REQUIRED_ATTR(eps, Float)
    .OP_END_FACTORY_REG(MixedPorts)
"""


def _parse(tmp_path):
    inc = tmp_path / "opp" / "built-in" / "op_graph" / "inc"
    inc.mkdir(parents=True)
    (inc / "ops_proto_graphmode.h").write_text(_IR, encoding="utf-8")
    return ProtoLoader(ascend_path=str(tmp_path)).get_op_info("MixedPorts")


def test_optional_inputs_and_required_attrs_extracted(tmp_path):
    info = _parse(tmp_path)
    assert info is not None, "IR 未解析成功"
    # 新增量字段
    assert info.optional_inputs == ["bias"]
    assert ("num_split", "Int") in info.required_attrs
    assert ("eps", "Float") in info.required_attrs
    assert ("alpha", "Float") not in [a[0] for a in info.required_attrs] or (
        "alpha",
        "Float",
    ) not in info.required_attrs
    # attrs 仍为 ATTR + REQUIRED_ATTR 拼接（存量消费者形状不变）
    assert ("alpha", "Float") in info.attrs
    assert ("num_split", "Int") in info.attrs
    assert ("eps", "Float") in info.attrs


def test_existing_fields_unchanged_shape(tmp_path):
    info = _parse(tmp_path)
    # inputs 混合捕获声明序（INPUT/OPTIONAL/DYNAMIC 一把抓），顺序保持声明位置
    assert info.inputs == ["x1", "bias", "x2", "extra"]
    assert info.dynamic_inputs == ["extra"]
    assert info.outputs == ["y", "aux"]
    assert info.dynamic_outputs == ["aux"]


def test_required_attrs_empty_for_attr_only_op(tmp_path):
    inc = tmp_path / "opp" / "built-in" / "op_graph" / "inc"
    inc.mkdir(parents=True)
    (inc / "ops_proto_plain.h").write_text(
        "REG_OP(PlainOp)\n"
        "    .INPUT(x, TensorType({DT_FLOAT16}))\n"
        "    .OUTPUT(y, TensorType({DT_FLOAT16}))\n"
        "    .ATTR(alpha, Float, 1.0)\n"
        "    .OP_END_FACTORY_REG(PlainOp)\n",
        encoding="utf-8",
    )
    info = ProtoLoader(ascend_path=str(tmp_path)).get_op_info("PlainOp")
    assert info is not None
    assert info.optional_inputs == []
    assert info.required_attrs == []
    assert info.attrs == [("alpha", "Float")]
