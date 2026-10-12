# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""hccl 手工签名表防腐化断言。

- torch_npu 系不登记：npu._schemas（aten 权威源）可完整解析——本断言
  钉死该前提，防止 torch_npu 回退（无 _schemas 暴露）时静默丢失签名能力；
- 表内禁止再出现 torch_npu.* 项（新项须先证明解析链不可达并更新文档）；
- 保留项（mindspeed 系）若在当前环境可自动解析，参数名集合必须与表一致
  ——表是快照，权威源演化后表不跟即 FAIL。
"""

import pytest

from ttk.core_modules.framework_api.torch.hccl_apis import hccl_apis

try:
    import torch_npu  # noqa: F401

    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

# 不登记的 torch_npu 系项及其权威解析基线（torch_npu 2.9.0 实测：source=npu._schemas）
_RETIRED_TORCH_NPU_APIS = {
    "torch_npu.npu_all_gather_base_mm": 2,
    "torch_npu.npu_all_gather_quant_mm": 2,
    "torch_npu.npu_all_to_all_matmul": 2,
    "torch_npu.npu_all_to_all_quant_matmul": 2,
    "torch_npu.npu_alltoallv_gmm": 2,
    "torch_npu.npu_alltoallv_quant_gmm": 4,
    "torch_npu.npu_gmm_alltoallv": 2,
    "torch_npu.npu_matmul_all_to_all": 2,
    "torch_npu.npu_mm_all_reduce_base": 2,
    "torch_npu.npu_mm_reduce_scatter_base": 2,
    "torch_npu.npu_quant_gmm_alltoallv": 4,
    "torch_npu.npu_quant_matmul_all_to_all": 2,
}


def test_table_contains_no_torch_npu_entries():
    """torch_npu 系签名由 npu._schemas 权威解析，表内不得回流。"""
    offenders = [k for k in hccl_apis if k.startswith("torch_npu.")]
    assert not offenders, (
        f"torch_npu 项回流手工表: {offenders}——npu._schemas 已可解析；"
        "若个别算子确不可达，须更新本测试与 hccl_apis.py 文档说明"
    )


class TestLazyTableLoad:
    """手工表惰性装载：keeper.get 整链（→ adapter → get_api_params）首查触发，
    无 eager 前置。mindspeed 未安装亦可验证——manual 命中短路在六源解析链
    （需 import mindspeed）之前。全量保真断言：overloads/return_count 不因
    装载路径丢失。
    """

    def setup_method(self):
        from ttk.core_modules.framework_api.torch import param_extractor as pe

        pe._MANUAL_OVERRIDES.clear()
        pe._HCCL_TABLE_READY = False

    def teardown_method(self):
        from ttk.core_modules.framework_api.torch import param_extractor as pe

        pe._MANUAL_OVERRIDES.clear()
        pe._HCCL_TABLE_READY = False

    def test_keeper_full_chain_loads_table_lazily(self):
        from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper

        keeper = FrameworkApiInfoKeeper()
        keeper.clear_cache()
        try:
            info = keeper.get("mindspeed.npu_mm_all_reduce_add_rms_norm")
            assert info is not None
            assert info.source == "manual-hccl"
            assert info.overloads[0].return_count == 2  # 全量保真（平参登记会丢）
            assert any(p.name == "hcom" for p in info.params)
        finally:
            keeper.clear_cache()


@pytest.mark.skipif(not HAS_TORCH_NPU, reason="torch_npu 不可导入（无 CANN 环境）")
@pytest.mark.parametrize(("api_name", "required_tensors"), sorted(_RETIRED_TORCH_NPU_APIS.items()))
def test_retired_torch_npu_apis_parse_from_schemas(api_name, required_tensors):
    """解析前提钉死：解析链（不经手工表）可完整解析，注入位在签名内。"""
    from ttk.core_modules.framework_api.torch.param_extractor import extract_api_params

    info = extract_api_params(api_name)
    assert info is not None, f"{api_name}: 解析链返回 None（npu._schemas 不可达？）"
    assert "_schemas" in info.source, f"{api_name}: source={info.source}（预期权威 schema 源）"
    actual_req = sum(1 for p in info.params if p.is_tensor_like and not p.is_optional)
    assert actual_req == required_tensors, f"{api_name}: 必选张量 {actual_req} != {required_tensors}"
    injected = {p.name for p in info.params if p.name in ("hcom", "world_size")}
    assert injected, f"{api_name}: 签名内未见注入位（hcom/world_size）"


def test_table_entries_not_stale_against_auto_parse():
    """防腐化：表项在可自动解析的环境下必须与权威结果一致，否则即快照腐化。

    解析不可达（如 mindspeed 未安装）时跳过——表项存在即合理。
    """
    from ttk.core_modules.framework_api.torch.param_extractor import extract_api_params

    checked = 0
    for api_name, info in hccl_apis.items():
        auto = extract_api_params(api_name)
        if auto is None:
            continue
        checked += 1
        manual_names = {p.name for p in info.params}
        auto_names = {p.name for p in auto.params}
        assert manual_names == auto_names, (
            f"{api_name}: 手工表与自动解析脱节——"
            f"仅表有(疑似过时): {sorted(manual_names - auto_names)}, "
            f"仅自动有(表缺失): {sorted(auto_names - manual_names)}"
        )
    if not checked:
        pytest.skip("表内所有 API 在当前环境均不可自动解析（包缺失），无对账对象")
