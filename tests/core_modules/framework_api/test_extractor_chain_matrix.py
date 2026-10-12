# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""解析链编排矩阵回归：28 API 胜者分布固化（extract_api_params 对外行为）。

原语级单测（tests/core_modules/framework_api/test_param_extractor_primitives.py）测不到编排层回归
——源调用行丢失类损伤（如 _strip_inplace_if_separate_exists 事故）原语全绿而
编排静默坏。本文件按「胜者 source + 主参数表名字序列」钉死家族路由
（A schema / B npu / C 通用）与组合裁决矩阵（D/T/A 八分支）的输出。

钉死断言仅在基线环境生效（BASELINE_TORCH / BASELINE_TORCH_NPU 完整版本精确
匹配，+cpu 等本地后缀忽略）；torch/torch_npu 版本演化导致的签名漂移（重载数/
参数表变化）会击穿钉死表——属预期行为。非基线环境自动降级为编排层结构不变
量（解析链非 None + source/参数表非空，仍可抓获链路断裂类损伤），并 skip 钉死
断言。历史上因跨环境签名漂移下掉的条目：torch.mm（TypeError 多重载分支，由
cat/clamp/where/ones/randint/cumsum/gather 代表）、torch_npu.npu_mm_all_reduce_base
（npu._schemas 家族，由 npu_conv2d/npu_scatter 代表）。

按当期版本重新冻结基线：
    python3 -c "import ttk.core_modules.framework_api.torch.param_extractor as pe; \
        [print(api, pe.extract_api_params(api).source, [p.name for p in pe.extract_api_params(api).params]) \
         for api in [...]]"
随后更新 CHAIN_MATRIX 与 BASELINE_TORCH / BASELINE_TORCH_NPU。
"""

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import torch_npu  # noqa: F401

    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch 不可导入")

# 基线实测冻结版本（忽略 +cpu 等本地构建后缀）
BASELINE_TORCH = "2.9.0"
BASELINE_TORCH_NPU = "2.9.0.post7"


def _base_version(version: str) -> str:
    return version.split("+")[0]


def _row_on_baseline(api_name: str) -> bool:
    if _base_version(torch.__version__) != BASELINE_TORCH:
        return False
    if api_name.startswith("torch_npu.") and HAS_TORCH_NPU:
        return _base_version(torch_npu.__version__) == BASELINE_TORCH_NPU
    return True


# (api_name, 预期胜者 source, 预期主参数表名字序列)
CHAIN_MATRIX = [
    ("torch.add", "merge(docstring(first-line+Args), TypeError(single-signature))", ["input", "other", "alpha", "out"]),
    ("torch.cat", "TypeError(multi-overload, 2 signatures)", ["tensors", "dim", "out"]),
    ("torch.stack", "docstring(first-line+Args)", ["tensors", "dim", "out"]),
    ("torch.split", "docstring(Args-only)", ["tensor", "split_size_or_sections", "dim"]),
    ("torch.topk", "docstring(first-line+Args)", ["input", "k", "dim", "largest", "sorted", "out"]),
    ("torch.clamp", "TypeError(multi-overload, 2 signatures)", ["input", "min", "max", "out"]),
    ("torch.where", "TypeError(multi-overload, 5 signatures)", ["condition", "input", "other", "out"]),
    (
        "torch.ones",
        "TypeError(multi-overload, 2 signatures)",
        ["size", "names", "dtype", "layout", "device", "pin_memory", "requires_grad"],
    ),
    (
        "torch.randint",
        "TypeError(multi-overload, 4 signatures)",
        ["high", "size", "generator", "out", "dtype", "layout", "device", "pin_memory", "requires_grad"],
    ),
    (
        "torch.embedding",
        "alias(torch.nn.functional.embedding, docstring(Args-only))",
        ["input", "weight", "padding_idx", "max_norm"],
    ),
    ("torch.nn.functional.relu", "docstring(first-line)", ["input", "inplace"]),
    ("torch.nn.functional.linear", "docstring(first-line)", ["input", "weight", "bias"]),
    (
        "torch.nn.functional.conv2d",
        "TypeError(multi-overload, 2 signatures)",
        ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
    ),
    (
        "torch.nn.functional.cross_entropy",
        "docstring(Args-only)",
        ["input", "target", "weight", "size_average", "ignore_index", "reduce", "reduction", "label_smoothing"],
    ),
    ("torch.nn.functional.softmax", "docstring(Args-only)", ["input", "dim", "dtype"]),
    ("torch.nn.functional.layer_norm", "__annotations__", ["input", "normalized_shape", "weight", "bias", "eps"]),
    ("torch.nn.functional.pad", "docstring(first-line+Args)", ["input", "pad", "mode", "value"]),
    ("torch.Tensor.relu_", "alias(torch.nn.functional.relu_, docstring(first-line))", ["input"]),
    ("torch.Tensor.add", "docstring(first-line)", ["other", "alpha"]),
    ("torch.Tensor.view", "docstring(first-line+Args)", ["shape"]),
    ("torch.Tensor.permute", "docstring(first-line)", ["dims"]),
    ("torch.ops.aten.add.Tensor", "aten._schemas", ["self", "other", "alpha"]),
    ("torch.ops.aten.mm.default", "aten._schemas", ["self", "mat2"]),
    (
        "torch_npu.npu_conv2d",
        "npu._schemas(2 overloads)",
        ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
    ),
    ("torch_npu.npu_scatter", "npu._schemas", ["input", "indices", "updates", "dim"]),
    ("torch.t", "docstring(first-line+Args)", ["input"]),
    ("torch.cumsum", "TypeError(multi-overload, 2 signatures)", ["input", "dim", "dtype", "out"]),
    ("torch.gather", "TypeError(multi-overload, 2 signatures)", ["input", "dim", "index", "sparse_grad", "out"]),
]


@pytest.mark.parametrize(("api_name", "expected_source", "expected_params"), CHAIN_MATRIX)
def test_chain_winner_and_param_table(api_name, expected_source, expected_params):
    """胜者 source 与主参数表均须与冻结基线一致（编排层回归网）。"""
    if api_name.startswith("torch_npu.") and not HAS_TORCH_NPU:
        pytest.skip("torch_npu 不可导入（无 CANN 环境）")
    from ttk.core_modules.framework_api.torch.param_extractor import extract_api_params

    info = extract_api_params(api_name)
    assert info is not None, f"{api_name}: 解析链返回 None"
    if not _row_on_baseline(api_name):
        # 非基线环境：仅校验编排层结构不变量（链路断裂类损伤仍可抓获），
        # 胜者/参数表钉死按当期版本重新冻结（见模块 docstring）。
        assert info.source, f"{api_name}: 编排链结构异常 source={info.source!r}"
        assert info.params, f"{api_name}: 编排链结构异常 params={info.params!r}"
        npu_ver = _base_version(torch_npu.__version__) if HAS_TORCH_NPU else "absent"
        pytest.skip(
            f"非基线环境 torch={_base_version(torch.__version__)} torch_npu={npu_ver}"
            f"（钉死基线 torch={BASELINE_TORCH} / torch_npu={BASELINE_TORCH_NPU}）"
        )
    assert info.source == expected_source, f"{api_name}: 胜者漂移 {info.source!r} != {expected_source!r}"
    actual = [p.name for p in info.params]
    assert actual == expected_params, f"{api_name}: 参数表漂移 {actual} != {expected_params}"
