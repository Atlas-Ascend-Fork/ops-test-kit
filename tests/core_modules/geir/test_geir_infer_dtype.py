#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for GEIR output dtype inference via DT_UNDEFINED.

看护修复：GEIR 生成图直接携带 output_dtypes，算子 InferDataType 回调永远不被
调用。修复后输出 desc 的 dtype 一律置 DT_UNDEFINED，GE 编译期必须走
InferDataType 推导；推导出的 dtype 由 GE 自行落位，回调缺失或执行失败会导致
图编译/执行失败，即为功能验证（不与 output_dtypes 单独比对）。
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.geir.graph_builder import GeirGraphBuilder


def _config_path(tmp_path, mode):
    """与 GeirGraphBuilder._compute_dirs 的目录约定保持一致，不触碰保护成员。"""
    base_dir = "dynamic" if mode.startswith("dynamic") else "const"
    sub_dir = tmp_path / "geir" / base_dir / "binary" if "binary" in mode else tmp_path / "geir" / base_dir
    return sub_dir / f"case_{mode}.json"


def _write_config(
    tmp_path,
    mode,
    *,
    input_shapes=((8, 8, 8),),
    input_dtypes=("float32",),
    output_shapes=((8, 8, 8),),
    output_dtypes=("float32",),
    output_inplace_indexes=(),
    dynamic_inputs=None,
    dynamic_outputs=None,
    attributes=None,
):
    tc = MagicMock()
    tc.op_name = "Add"
    tc.testcase_name = f"case_{mode}"
    tc.input_shapes = input_shapes
    tc.dyn_input_shapes = None
    tc.input_dtypes = input_dtypes
    tc.input_formats = ()
    tc.input_ori_formats = ()
    tc.input_ori_shapes = ()
    tc.output_shapes = output_shapes
    tc.output_dtypes = output_dtypes
    tc.output_formats = ()
    tc.output_ori_formats = ()
    tc.output_ori_shapes = ()
    tc.output_inplace_indexes = output_inplace_indexes
    tc.output_shape_unknown_indexes = ()
    tc.attributes = {} if attributes is None else attributes

    switches = MagicMock()
    switches.root_path = str(tmp_path)

    with patch("ttk.core_modules.geir.graph_builder.ProtoLoader") as proto_loader:
        proto_loader.return_value.get_op_info.return_value = MagicMock(
            inputs=["x"],
            outputs=["y"],
            attrs=[],
            dynamic_inputs=dynamic_inputs or [],
            dynamic_outputs=dynamic_outputs or [],
            proto_file="dummy",
        )
        builder = GeirGraphBuilder(switches)
        builder.write_case_config(tc, mode=mode)

    with open(_config_path(tmp_path, mode), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("mode", ["const", "dynamic", "const_binary"], ids=["const", "dynamic", "const-binary"])
def test_output_dtype_is_undefined(tmp_path, mode):
    """核心看护：输出 desc 的 dtype 必须是 DT_UNDEFINED（全模式一致），触发 InferDataType 推导。"""
    cfg = _write_config(tmp_path, mode, output_dtypes=("float16",))
    assert cfg["outputs"][0]["dtype"] == "DT_UNDEFINED"
    # 输入不受影响
    assert cfg["inputs"][0]["dtype"] == "DT_FLOAT"


def test_dynamic_output_elements_use_declared_dtype(tmp_path):
    """DYNAMIC_OUTPUT(TensorList) 逐元素落 CSV 声明的 dtype（嵌套逐元素解析）。

    update_dynamic_output_desc 对部分算子（如 Split）不接受 DT_UNDEFINED
    （E39999），且变参口无 InferDataType 推导路径，声明值即权威。"""
    cfg = _write_config(
        tmp_path,
        "const",
        output_shapes=(((8, 8), (4, 4)),),
        output_dtypes=(("float32", "int32"),),
        dynamic_outputs=["y"],
    )
    assert [el["dtype"] for el in cfg["outputs"][0]["elements"]] == ["DT_FLOAT", "DT_INT32"]


def test_dynamic_output_elements_dtype_broadcast(tmp_path):
    """DYNAMIC_OUTPUT 顶层广播 dtype（('float16',) 非嵌套）填充全部元素。"""
    cfg = _write_config(
        tmp_path,
        "const",
        output_shapes=(((8, 8), (4, 4)),),
        output_dtypes=("float16",),
        dynamic_outputs=["y"],
    )
    assert [el["dtype"] for el in cfg["outputs"][0]["elements"]] == ["DT_FLOAT16", "DT_FLOAT16"]


def test_inplace_output_keeps_input_dtype(tmp_path):
    """inplace 输出继承输入 desc（dtype 已知），不参与推导触发。"""
    cfg = _write_config(tmp_path, "const", output_inplace_indexes=(0,))
    assert cfg["outputs"][0]["dtype"] == "DT_FLOAT"
    assert cfg["outputs"][0]["inplace_input_idx"] == 0


def test_inplace_dynamic_output_elements_inherit_input_desc(tmp_path):
    """inplace 的 DYNAMIC_OUTPUT(TensorList) 逐元素继承输入组元素 desc。

    模板按 elements 逐个 update_dynamic_output_desc，若元素仍为 DT_UNDEFINED
    会触发 GE Unsupported_Operator(EZ3002)。
    """
    cfg = _write_config(
        tmp_path,
        "const",
        input_shapes=(((8, 8), (4, 4)),),
        input_dtypes=(("float32", "int32"),),
        output_shapes=(((8, 8), (4, 4)),),
        output_dtypes=(("float32", "int32"),),
        output_inplace_indexes=(0,),
        dynamic_inputs=["x"],
        dynamic_outputs=["y"],
    )
    out = cfg["outputs"][0]
    assert out["inplace_input_idx"] == 0
    assert [el["dtype"] for el in out["elements"]] == ["DT_FLOAT", "DT_INT32"]
    assert [el["desc_shape"] for el in out["elements"]] == [[8, 8], [4, 4]]


def test_inf_attr_serialization_roundtrip(tmp_path):
    """非有限浮点属性经 json.dump 写出 Infinity（Python 默认 allow_nan），
    需能被读回为 inf（C++ 侧解析由模板 parseNumber 的 Infinity/NaN 分支支持）。"""
    cfg = _write_config(tmp_path, "const", attributes={"value": float("inf")})
    assert cfg["attrs"]["value"] == float("inf")
    raw = (_config_path(tmp_path, "const")).read_text(encoding="utf-8")
    assert "Infinity" in raw
