# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""model_builder 单测（纯 CPU，依赖 onnx 包；期望值随环境注册表/包默认走，
不锚定 onnx/opset/IR 版本号）。

覆盖：Add proto 结构+数值、Split 尾部 Optional 输入省略、bf16 MatMul 数值
（ReferenceEvaluator）、错误路径（缺必选输入/dtype 无映射/动态 dim/属性
类型支持集外）、build_input_specs 反解（None 占位剔除 / Variadic 嵌套展开）。Concat 的建模+
数值经 golden 层同构参照覆盖（test_golden.py，共用 build_model_proto）。

跨环境：onnx 缺席整文件 skip（importorskip）；opset 超限仅影响经
extract_onnx_params 的用例（conftest.onnx_registry 门禁 bypass）；Split 的
num_outputs 语义经 split_with_num_outputs 能力探针（旧注册表缺席时 skip）；
bf16 参考评估经 reference_evaluator_bf16_matmul 能力探针（onnx 1.17~1.20
上游 dtype 比较缺陷误抛 Output type mismatch 时 skip）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.framework_api.onnx.model_builder import _output_names, build_input_specs, build_model_proto
from ttk.core_modules.framework_api.onnx.param_extractor import extract_onnx_params

pytest.importorskip("onnx")

import onnx.defs  # noqa: E402
from onnx import TensorProto  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

OPSET = onnx.defs.onnx_opset_version()


def _node(model):
    assert len(model.graph.node) == 1
    return model.graph.node[0]


def _attrs(node):
    return {a.name: a for a in node.attribute}


# ── 结构 + 数值端到端 ─────────────────────────────────────────────────────


def test_add_structure_and_eval():
    """Add：proto 结构断言 + ReferenceEvaluator 数值对拍 numpy 手算。"""
    m = build_model_proto("Add", [("input_0", (2, 3), "float32"), ("input_1", (2, 3), "float32")], {})
    assert m.ir_version == onnx.IR_VERSION  # 包默认（随环境 onnx 版本走）
    assert len(m.opset_import) == 1
    assert m.opset_import[0].domain == ""
    assert m.opset_import[0].version == OPSET
    node = _node(m)
    assert node.op_type == "Add"
    assert list(node.input) == ["input_0", "input_1"]
    assert list(node.output) == ["output_0"]
    assert not node.attribute
    assert [vi.name for vi in m.graph.input] == ["input_0", "input_1"]
    for vi, shape in zip(m.graph.input, [(2, 3), (2, 3)]):
        assert vi.type.tensor_type.elem_type == TensorProto.FLOAT
        assert [d.dim_value for d in vi.type.tensor_type.shape.dim] == list(shape)
    # 输出 ValueInfo 只给名字：elem_type=0、无 shape
    out = m.graph.output[0]
    assert out.name == "output_0"
    assert out.type.tensor_type.elem_type == 0
    assert not out.type.tensor_type.HasField("shape")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = (a + 10).astype(np.float32)
    (result,) = ReferenceEvaluator(m).run(None, {"input_0": a, "input_1": b})
    np.testing.assert_array_equal(result, a + b)


def test_split_trailing_optional_omitted(split_with_num_outputs):
    """尾部 Optional 输入直接省略（node.inputs 短列表）——`""` 占位被拒。"""
    m = build_model_proto("Split", [("input_0", (6, 4), "float32")], {"axis": 0, "num_outputs": 3})
    node = _node(m)
    assert list(node.input) == ["input_0"]  # 短列表，无空串占位
    assert list(node.output) == ["output_0", "output_1", "output_2"]  # num_outputs=3
    attr = _attrs(node)
    assert attr["axis"].i == 0
    assert attr["num_outputs"].i == 3
    assert [vi.name for vi in m.graph.output] == ["output_0", "output_1", "output_2"]


def test_explicit_opset_pins_opset_and_ir_version():
    """显式 opset（api_name 带版本段）：opset_import 按声明值、ir_version 恒钉
    IR 8（新版 onnx 包默认 ir 更高、atc 适配风险）；裸名路径不受影响。"""
    specs = [("input_0", (2, 3), "float32"), ("input_1", (2, 3), "float32")]
    m = build_model_proto("Add", specs, {}, opset=13)
    assert m.opset_import[0].version == 13
    assert m.ir_version == 8
    # 同参数字节与裸名（env opset）模型不一致——opset 参与 LRU 键与 atc 输入
    bare = build_model_proto("Add", specs, {})
    assert bare.opset_import[0].version == OPSET
    assert build_model_proto("Add", specs, {}, opset=13).opset_import[0].version == 13  # LRU 不回退


def test_output_names_mixed_single_and_variadic():
    """Variadic 输出续编号（非 range 重编）：混合 Single+Variadic 形态不重名
    （当前注册表无此形态——ai.onnx ≤26 全量扫描仅 Split/If/Loop/Scan/
    SequenceMap 且均为纯 Variadic，故用假 formal 钉住机制泛化；纯 Variadic
    的 Split 命名由 test_split_trailing_optional_omitted 覆盖）。"""
    formals = [
        SimpleNamespace(name="X", option="Single"),
        SimpleNamespace(name="outputs", option="Variadic"),
    ]
    assert _output_names("FakeOp", formals, {"num_outputs": 2}, []) == ["output_0", "output_1", "output_2"]


def test_eval_matmul_bfloat16_ml_dtypes(reference_evaluator_bf16_matmul):
    """bf16 数值端到端（ml_dtypes 缺席或探针命中上游 dtype 误报时跳过）：
    elem_type=16 直通 ReferenceEvaluator。"""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m = build_model_proto("MatMul", [("input_0", (4, 8), "bfloat16"), ("input_1", (8, 2), "bfloat16")], {})
    a = np.random.RandomState(0).uniform(-1, 1, (4, 8)).astype(ml_dtypes.bfloat16)
    b = np.random.RandomState(1).uniform(-1, 1, (8, 2)).astype(ml_dtypes.bfloat16)
    (out,) = ReferenceEvaluator(m).run(None, {"input_0": a, "input_1": b})
    assert out.dtype == ml_dtypes.bfloat16
    np.testing.assert_allclose(
        out.astype(np.float32), a.astype(np.float32) @ b.astype(np.float32), rtol=2e-2, atol=2e-2
    )


# ── 错误路径 ──────────────────────────────────────────────────────────────


def test_build_errors():
    """缺必选输入 / dtype 无映射 / 动态 dim 三类拒绝路径。"""
    with pytest.raises(ValueError, match=r"required input 'B' not provided.*trailing Optional"):
        build_model_proto("Add", [("input_0", (2, 3), "float32")], {})
    with pytest.raises(ValueError, match=r"ONNX_DTYPE|dtype.*hifloat8.*TensorProto") as excinfo:
        build_model_proto("Add", [("input_0", (2, 3), "float32"), ("input_1", (2, 3), "hifloat8")], {})
    assert "ONNX_DTYPE_MAP" in str(excinfo.value)  # 错误信息指向两侧映射名
    with pytest.raises(ValueError, match=r"dynamic dim.*static input shapes"):
        build_model_proto("Add", [("input_0", (-1, 3), "float32"), ("input_1", (-1, 3), "float32")], {})


def test_unsupported_attr_type():
    """支持集外属性类型（TENSOR/GRAPH 载荷）公开面兜底：按算子/属性点名
    显式报错而非裸 KeyError（CSV 路径由 param_extractor 解析期前置拒绝）。"""
    with pytest.raises(ValueError, match=r"onnx.Constant: attribute 'value' has unsupported schema type TENSOR"):
        build_model_proto("Constant", [], {"value": object()})
    with pytest.raises(ValueError, match=r"onnx.If: attribute 'then_branch' has unsupported schema type GRAPH"):
        build_model_proto("If", [("input_0", (1,), "bool")], {"then_branch": object(), "else_branch": object()})


# ── build_input_specs（args 反解）──────────────────────


def test_build_input_specs_none_placeholder_and_nested_list(onnx_registry, split_with_num_outputs):
    """Split 省略 split 输入：None 占位槽剔除，仅在场输入编号 input_0..N；
    Variadic 嵌套形态（CSV TensorList）：单列表位展开为逐项。"""
    info = extract_onnx_params("onnx.Split")
    arr = np.zeros((6, 4), np.float32)
    specs, host = build_input_specs(info, [arr, None])
    assert specs == [("input_0", (6, 4), "float32")]
    assert host == [arr]

    info = extract_onnx_params("onnx.Concat")
    t1, t2 = np.zeros((2, 3), np.int32), np.zeros((2, 4), np.int32)
    specs, host = build_input_specs(info, [[t1, t2]])
    assert specs == [("input_0", (2, 3), "int32"), ("input_1", (2, 4), "int32")]
    assert host == [t1, t2]


def test_build_input_specs_extra_args_rejected(onnx_registry):
    """无 Variadic 时超发 args 显式报错——静默丢弃会把丢参/错位掩盖成正常
    构建且无任何线索（CSV 路径经 match_overload 前置拦截，公开面直调兜底）。"""
    info = extract_onnx_params("onnx.Add")
    a, b, c = (np.zeros((2, 3), np.float32) for _ in range(3))
    with pytest.raises(ValueError, match=r"onnx.Add: 1 extra positional arg\(s\) beyond the 2 declared"):
        build_input_specs(info, [a, b, c])


def test_build_input_specs_variadic_absorbs_all_args(onnx_registry):
    """Variadic 输入吸收全部 args（flat 多位形态）：超发检查不得误伤。"""
    info = extract_onnx_params("onnx.Concat")
    t1, t2, t3 = (np.zeros((2, 3), np.int32) for _ in range(3))
    specs, host = build_input_specs(info, [t1, t2, t3])
    assert specs == [(f"input_{i}", (2, 3), "int32") for i in range(3)]
    assert host == [t1, t2, t3]
