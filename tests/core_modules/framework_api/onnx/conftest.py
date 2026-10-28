# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""onnx 通路单测的跨环境门禁（换机器可移植性，不锚定环境版本号）。

三类门禁，语义不同：
- ``onnx_registry``：onnx 包可用性 + opset 策略门禁的处理。环境 opset 超
  ``param_extractor.ATC_MAX_OPSET``（atc 插件注册上限，随 CANN 版本维护）时
  生产语义是解析期拒绝——但 schema→ParamInfo 映射逻辑与 opset 无关
  （注册表内 since_version 不随包版本漂移），此处抬高上限继续测映射本身；
  门禁行为由 test_param_extractor.test_opset_version_gate（任意 onnx 版本
  可跑、不依赖具体数值）单独钉住。真 atc 的物理限制不可 bypass——
  test_om_compiler / smoke 走 skip 而非本 fixture。
- ``split_with_num_outputs``：Split 的 num_outputs 属性能力探针（该属性为
  较新 Split schema 引入，旧 onnx 注册表无此声明）——按注册表实际声明判定
  而非版本号，缺席时依赖该语义的用例 skip。
- ``reference_evaluator_bf16_matmul``：ReferenceEvaluator 的 bf16 评估能力
  探针。onnx 上游缺陷（1.17 引入、1.21 移除该转换后自愈）：``OpRunBinary.run``
  输出侧把 ml_dtypes 结果 view 成 onnx 自定义 dtype（``convert_from_ml_dtypes``），
  ``OpRunBinaryNum.run`` 再与原 ml_dtypes 输入比较恒不等 → 对 bf16 等一切
  ml_dtypes 二元数值算子误抛 ``RuntimeTypeError("Output type mismatch")``。
  按实际行为判定而非版本号，命中该已知报错时依赖 bf16 参考评估的用例 skip。
"""

import pytest


@pytest.fixture
def onnx_registry(monkeypatch):
    """onnx 包可用 + opset 门禁 bypass（映射逻辑测试环境）。

    Returns:
        onnx.defs 模块（环境注册表，可直取 onnx_opset_version / get_schema）。
    """
    onnx_defs = pytest.importorskip("onnx.defs")

    from ttk.core_modules.framework_api.onnx import param_extractor

    env_opset = onnx_defs.onnx_opset_version()
    if env_opset > param_extractor.ATC_MAX_OPSET:
        monkeypatch.setattr(param_extractor, "ATC_MAX_OPSET", env_opset)
    return onnx_defs


@pytest.fixture
def split_with_num_outputs():
    """Split 的 num_outputs 属性能力探针（注册表实际声明判定，无版本号）。"""
    import onnx.defs

    if "num_outputs" not in onnx.defs.get_schema("Split").attributes:
        pytest.skip(
            "installed Split schema does not declare the 'num_outputs' attribute "
            "(older registry); tests requiring num_outputs semantics are skipped"
        )


@pytest.fixture
def reference_evaluator_bf16_matmul():
    """ReferenceEvaluator 的 bf16 MatMul 能力探针（实际行为判定，无版本号）。

    命中 onnx 上游 dtype 比较缺陷时 skip；其余异常照常上抛——真正的建模
    回归（如 elem_type 误映射）会以别的错误形态出现，不应被探针吞掉。
    """
    ml_dtypes = pytest.importorskip("ml_dtypes")
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import RuntimeTypeError

    from ttk.core_modules.framework_api.onnx.model_builder import build_model_proto

    model = build_model_proto("MatMul", [("input_0", (1, 1), "bfloat16"), ("input_1", (1, 1), "bfloat16")], {})
    feeds = {
        "input_0": np.zeros((1, 1), ml_dtypes.bfloat16),
        "input_1": np.zeros((1, 1), ml_dtypes.bfloat16),
    }
    try:
        ReferenceEvaluator(model).run(None, feeds)
    except RuntimeTypeError as exc:
        if "Output type mismatch" not in str(exc):
            raise
        pytest.skip(
            f"onnx.reference ReferenceEvaluator mis-compares bf16 outputs in this "
            f"environment (onnx {onnx.__version__}; upstream dtype-mismatch bug "
            f"introduced in onnx 1.17 and fixed in 1.21); tests requiring bf16 "
            f"reference evaluation are skipped — onnx >=1.21 or <=1.16 restores them"
        )
