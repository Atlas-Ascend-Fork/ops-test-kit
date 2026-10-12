#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GEIR Data/Const 重复注册（CANN 新版 proto 头自带 Data/Const 导致编译报错）。

新版 CANN 包把 REG_OP(Data)/REG_OP(Const) 的注册打进 proto 头（如
ops_proto_math.h），且用 OPS_PROTO_DEF_DATA/OPS_PROTO_DEF_CONST 防重定义宏
守卫。GEIR 模板再无条件自带一份时，REG_OP 宏展开为类定义，同一编译单元内
ge::op::Data/Const 重定义（redefinition of 'class ge::op::Data'）。老版
CANN 包的 proto 头不带这两个注册，模板又必须补上，否则建图用的
op::Data/op::Const 类型不存在。修复：模板的补注册块套用与 CANN 头同款的
OPS_PROTO_DEF_* 宏。模板先 include proto 头再落补注册块，预处理器自动
二选一：头已注册（宏已定义）则跳过，头未注册（宏未定义）则补上并定义宏。
"""

import re
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.geir.graph_builder import GeirGraphBuilder

_GXX = shutil.which("g++")

_FALLBACK_RE = re.compile(r"namespace ge \{.*?\}  // namespace ge", re.DOTALL)

# 最新 CANN 包 ops_proto_math.h 中 Data/Const 的注册布局（同款防重定义宏）。
_GUARDED_HEADER = """
#ifndef OPS_PROTO_DEF_CONST
#define OPS_PROTO_DEF_CONST
REG_OP(Const).OUTPUT(y, TensorType::ALL()).ATTR(value, Tensor, Tensor()).OP_END_FACTORY_REG(Const)
#endif

#ifndef OPS_PROTO_DEF_DATA
#define OPS_PROTO_DEF_DATA
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
#endif
"""

# 只走预处理（g++ -E），用桩宏还原 REG_OP 展开为类定义的关键形态，
# 无需 CANN 头即可验证防重定义宏的二选一行为。
_STUB_MACROS = """
#define REG_OP(x) namespace op { class x {
#define INPUT(a, b)
#define OUTPUT(a, b)
#define ATTR(a, b, c)
#define OP_END_FACTORY_REG(x) }; }
"""


def _render_op_source(tmp_path):
    switches = MagicMock()
    switches.root_path = str(tmp_path)
    with patch("ttk.core_modules.geir.graph_builder.ProtoLoader") as proto_loader:
        proto_loader.return_value.get_op_info.return_value = MagicMock(
            op_class="ProbeOp",
            inputs=["x"],
            outputs=["y"],
            attrs=[],
            dynamic_inputs=[],
            dynamic_outputs=[],
            proto_file="ops_proto_experiment.h",
        )
        builder = GeirGraphBuilder(switches)
        source_path = builder.generate_op_source("ProbeOp")
    with open(source_path, encoding="utf-8") as f:
        return f.read()


def _preprocess_fallback(tmp_path, header_text):
    fallback = _FALLBACK_RE.search(_render_op_source(tmp_path)).group(0)
    tu = tmp_path / "tu.cpp"
    tu.write_text(_STUB_MACROS + header_text + fallback, encoding="utf-8")
    result = subprocess.run([_GXX, "-E", "-P", str(tu)], capture_output=True, text=True, check=True)
    return result.stdout


def test_op_source_wraps_fallback_in_proto_def_guards(tmp_path):
    source = _render_op_source(tmp_path)
    fallback = _FALLBACK_RE.search(source).group(0)
    assert "#ifndef OPS_PROTO_DEF_DATA" in fallback
    assert "#define OPS_PROTO_DEF_DATA" in fallback
    assert "#ifndef OPS_PROTO_DEF_CONST" in fallback
    assert "#define OPS_PROTO_DEF_CONST" in fallback
    assert "REG_OP(Data)" in fallback
    assert "REG_OP(Const)" in fallback


@pytest.mark.skipif(_GXX is None, reason="g++ not available")
def test_guard_dedupes_when_proto_registers_data_const(tmp_path):
    """用户场景：proto 头（ops_proto_math.h 布局）已带宏守卫的 Data/Const 注册，
    补注册块必须被预处理掉，Data/Const 各只剩一份类定义。"""
    out = _preprocess_fallback(tmp_path, _GUARDED_HEADER)
    assert out.count("class Data {") == 1
    assert out.count("class Const {") == 1


@pytest.mark.skipif(_GXX is None, reason="g++ not available")
def test_fallback_survives_when_proto_lacks_data_const(tmp_path):
    """老版 CANN 包场景：proto 头不带 Data/Const，宏未定义，补注册必须生效。"""
    out = _preprocess_fallback(tmp_path, "")
    assert out.count("class Data {") == 1
    assert out.count("class Const {") == 1
