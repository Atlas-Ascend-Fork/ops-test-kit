# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""IR 解析：REG_OP 与左括号之间允许空白。

`REG_OP (Op)` 是合法的函数式宏调用，C++ 编译与 GE 注册均与 `REG_OP(Op)` 等价，
canndev op_proto/inc 中确有这样写的算子（FfnWorkerBatching）。解析器不认这种写法时，
表现为 "GEIR op source generation failed"，会被误判成算子缺陷。
"""

import pytest

from ttk.core_modules.geir.proto_loader import ProtoLoader

_IR = """
/* 无关的前导注释 */
REG_OP{reg_sp}(SpacedOp)
    .INPUT(x, TensorType({{DT_FLOAT16}}))
    .OPTIONAL_INPUT(bias, TensorType({{DT_FLOAT16}}))
    .OUTPUT(y, TensorType({{DT_FLOAT16}}))
    .REQUIRED_ATTR(expert_num, Int)
    .ATTR(alpha, Float, 1.0)
    .OP_END_FACTORY_REG{end_sp}(SpacedOp)
"""


def _parse(tmp_path, reg_sp="", end_sp=""):
    """走公开接口解析：把 IR 落成一个 proto 头，再用 ascend_path 指向它构造 loader。

    不直接调内部解析函数（G.CLS.11 禁止类外访问受保护成员），也就顺带覆盖了
    "扫描 proto 目录 -> 匹配算子 -> 解析" 这条真实链路。ProtoLoader 是 Singleton，
    但键里含构造参数，每个用例的 tmp_path 唯一，因此实例互不复用。
    """
    inc = tmp_path / "opp" / "built-in" / "op_graph" / "inc"
    inc.mkdir(parents=True)
    (inc / "ops_proto_experiment.h").write_text(_IR.format(reg_sp=reg_sp, end_sp=end_sp), encoding="utf-8")
    return ProtoLoader(ascend_path=str(tmp_path)).get_op_info("SpacedOp")


@pytest.mark.parametrize(
    ("reg_sp", "end_sp"),
    [
        ("", ""),  # 常规写法
        (" ", ""),  # FfnWorkerBatching 的实际写法：仅 REG_OP 后有空格
        (" ", " "),  # 两处都有空格
        ("\t", "\t"),  # 制表符
    ],
)
def test_reg_op_accepts_whitespace_before_paren(tmp_path, reg_sp, end_sp):
    info = _parse(tmp_path, reg_sp, end_sp)
    assert info is not None, "IR 未解析成功"
    assert info.op_class == "SpacedOp"
    assert info.inputs == ["x", "bias"]
    assert info.outputs == ["y"]
    assert ("expert_num", "Int") in info.attrs
