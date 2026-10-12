#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for GEIR case config ori_shape generation.

看护修复：dynamic 模式下 ori_shape 必须使用动态 shape(-1/-2)，
使算子编译期 infershape 的 GetInputShape（读 origin shape）能看到未知维度，
与真实动态图场景一致；const 模式 ori_shape 保持 input_ori_shapes/input_shapes 回退逻辑。
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
    dyn_input_shapes=None,
    input_ori_shapes=None,
    output_ori_shapes=None,
    input_shapes=((8, 8, 8),),
    input_dtypes=("float32",),
    output_shapes=((8, 8, 8),),
    output_dtypes=("float32",),
    dynamic_inputs=None,
    dynamic_outputs=None,
):
    tc = MagicMock()
    tc.op_name = "Add"
    tc.testcase_name = f"case_{mode}"
    tc.input_shapes = input_shapes
    tc.dyn_input_shapes = dyn_input_shapes
    tc.input_dtypes = input_dtypes
    tc.input_formats = ()
    tc.input_ori_formats = ()
    tc.input_ori_shapes = input_ori_shapes
    tc.output_shapes = output_shapes
    tc.output_dtypes = output_dtypes
    tc.output_formats = ()
    tc.output_ori_formats = ()
    tc.output_ori_shapes = output_ori_shapes
    tc.attributes = {}

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


@pytest.mark.parametrize("mode", ["const", "const_binary"], ids=["const", "const-binary"])
def test_const_mode_ori_shape_positive(tmp_path, mode):
    cfg = _write_config(tmp_path, mode)
    assert cfg["inputs"][0]["desc_shape"] == [8, 8, 8]
    assert cfg["inputs"][0]["ori_shape"] == [8, 8, 8]
    assert cfg["outputs"][0]["desc_shape"] == [8, 8, 8]
    assert cfg["outputs"][0]["ori_shape"] == [8, 8, 8]


def test_const_mode_ori_shape_prefers_input_ori_shapes(tmp_path):
    cfg = _write_config(tmp_path, "const", input_ori_shapes=((4, 16, 16),), output_ori_shapes=((4, 16, 16),))
    assert cfg["inputs"][0]["desc_shape"] == [8, 8, 8]
    assert cfg["inputs"][0]["ori_shape"] == [4, 16, 16]
    assert cfg["outputs"][0]["ori_shape"] == [4, 16, 16]


@pytest.mark.parametrize("mode", ["dynamic", "dynamic_binary"], ids=["dynamic", "dynamic-binary"])
def test_dynamic_mode_ori_shape_is_dynamic(tmp_path, mode):
    """核心看护：dynamic 模式（含 binary 变体）ori_shape 必须为 -1，编译期 infershape 才能看到未知维度。"""
    cfg = _write_config(tmp_path, mode)
    assert cfg["inputs"][0]["desc_shape"] == [-1, -1, -1]
    assert cfg["inputs"][0]["ori_shape"] == [-1, -1, -1]
    assert cfg["outputs"][0]["desc_shape"] == [-1, -1, -1]
    assert cfg["outputs"][0]["ori_shape"] == [-1, -1, -1]


def test_dynamic_mode_ori_shape_follows_dyn_input_shapes(tmp_path):
    """dyn_input_shapes 列（含 -2 unknown rank）应同时透传到 desc_shape 与 ori_shape。"""
    cfg = _write_config(tmp_path, "dynamic", dyn_input_shapes=((-2,),))
    assert cfg["inputs"][0]["desc_shape"] == [-2]
    assert cfg["inputs"][0]["ori_shape"] == [-2]


def test_dynamic_mode_ori_shape_ignores_positive_ori_shapes(tmp_path):
    """dynamic 模式下即使 CSV 给了正值 input_ori_shapes，ori_shape 也必须用动态 shape。"""
    cfg = _write_config(tmp_path, "dynamic", input_ori_shapes=((4, 16, 16),))
    assert cfg["inputs"][0]["ori_shape"] == [-1, -1, -1]


# ---------- DYNAMIC_INPUT(TensorList) ----------

_TL_SHAPES = (((8, 8, 8), (4, 4, 4)),)
_TL_DTYPES = (("float32", "float32"),)


def _tl_elements(cfg):
    return cfg["inputs"][0]["elements"]


def test_tensorlist_const_mode_shapes_positive(tmp_path):
    cfg = _write_config(tmp_path, "const", input_shapes=_TL_SHAPES, input_dtypes=_TL_DTYPES, dynamic_inputs=["x"])
    shapes = [(el["data_shape"], el["desc_shape"], el["ori_shape"]) for el in _tl_elements(cfg)]
    assert shapes == [([8, 8, 8], [8, 8, 8], [8, 8, 8]), ([4, 4, 4], [4, 4, 4], [4, 4, 4])]


def test_tensorlist_dynamic_mode_shapes_dynamic(tmp_path):
    """TensorList 的 desc_shape/ori_shape 在 dynamic 模式必须为 -1（修复前为正值）。"""
    cfg = _write_config(tmp_path, "dynamic", input_shapes=_TL_SHAPES, input_dtypes=_TL_DTYPES, dynamic_inputs=["x"])
    for el in _tl_elements(cfg):
        assert el["desc_shape"] == [-1] * len(el["data_shape"])
        assert el["ori_shape"] == [-1] * len(el["data_shape"])


def test_tensorlist_dynamic_mode_follows_dyn_input_shapes(tmp_path):
    """TensorList 应逐元素读取 dyn_input_shapes 列（含 -2）。"""
    cfg = _write_config(
        tmp_path,
        "dynamic",
        input_shapes=_TL_SHAPES,
        input_dtypes=_TL_DTYPES,
        dyn_input_shapes=(((-1, -1, -1), (-2,)),),
        dynamic_inputs=["x"],
    )
    descs = [el["desc_shape"] for el in _tl_elements(cfg)]
    oris = [el["ori_shape"] for el in _tl_elements(cfg)]
    assert descs == [[-1, -1, -1], [-2]]
    assert oris == [[-1, -1, -1], [-2]]


def test_tensorlist_const_mode_ori_shape_prefers_nested_ori_shapes(tmp_path):
    cfg = _write_config(
        tmp_path,
        "const",
        input_shapes=_TL_SHAPES,
        input_dtypes=_TL_DTYPES,
        input_ori_shapes=(((2, 4, 4), (1, 2, 2)),),
        dynamic_inputs=["x"],
    )
    oris = [el["ori_shape"] for el in _tl_elements(cfg)]
    assert oris == [[2, 4, 4], [1, 2, 2]]
