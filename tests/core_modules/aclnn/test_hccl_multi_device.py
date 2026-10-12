# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""hccl_multi_device 多卡执行框架纯度与 plugin 契约 pin。

golden 侧三路分发——plugin golden → hccl_dispatcher 框架入口（无 plugin 则
SKIP、有则委派），hccl_cascade 级联 golden 留仓。执行侧按 SE 方向解耦：
框架只认 device_ids（机器清单），通信拓扑由 CSV attributes 声明，算子特定
rank 输入重生成经 input hook 的 testcase 上下文注入。本文件 pin 三件事：
- hccl_multi_device 执行路径零算子名（原内置量化重生成/权重共享/算子名清单已退场）；
- 双通信域门控 = attributes 声明（epWorldSize/tpWorldSize），非算子名匹配；
- 无 plugin 时多卡 golden 走 hccl_dispatcher 兜底（无 golden 的算子 →
  SKIP(NO_GOLDEN)），有 plugin 则按 (thread_contexts, device_ids,
  all_precision) 契约分发。
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ttk.core_modules.aclnn import hccl_multi_device
from ttk.core_modules.aclnn.input_generate import InputGenerator
from ttk.core_modules.aclnn.testcase import TestcaseAclnn

_RETIRED_OP_MARKERS = (
    "BatchMatMulReduceScatter",
    "AlltoAllAllGather",
    "GroupedMatMul",
    "AlltoAllvGrouped",
    "MatmulReduceScatter",
    "AllGatherMatmul",
    "MatmulAllReduce",
    "MatmulAlltoAll",
    "AlltoAllMatmul",
    "QuantMatmulAllReduce",
    "QuantAllReduce",
    "QuantReduceScatter",
    "QuantMatmulAlltoAll",
    "MoeDistribute",
)


def test_hccl_execution_path_carries_no_operator_names():
    """hccl_multi_device 执行路径零算子名：原算子清单与算子特定输入逻辑全部外迁 plugin。"""
    source = Path(hccl_multi_device.__file__).read_text(encoding="utf-8")
    leaked = [marker for marker in _RETIRED_OP_MARKERS if marker in source]
    assert not leaked, f"operator names leaked into the hccl_multi_device execution path: {leaked}"


def test_thread_so_resolution_is_generic():
    """so 解析下沉 info_keeper 通用能力：hccl_multi_device 线程体不再内嵌候选探测。"""
    source = Path(hccl_multi_device.__file__).read_text(encoding="utf-8")
    assert "libopapi_transformer" not in source
    assert "get_ascend_lib64_path" not in source
    assert "ensure_api_so_path" in source


def test_profile_multi_device_lifecycle_owned_by_scope():
    """资源生命周期由 HcclResourceScope 拥有：setup 失败即回退，成功 retain；
    旧的线程自毁 comm 残骸（post_exec_barrier/comm_destroyed_by_thread）不得回潮。"""
    source = Path(hccl_multi_device.__file__).read_text(encoding="utf-8")
    assert "with HcclResourceScope(" in source
    assert "post_exec_barrier" not in source
    assert "comm_destroyed_by_thread" not in source


def test_thread_failure_paths_abort_pre_exec_barrier():
    """线程失败必须 abort barrier：兄弟 rank 立即 BrokenBarrierError，
    不再干等 1800s 超时（输入生成失败早退路径 + 兜底异常路径都要 abort）。"""
    source = Path(hccl_multi_device.__file__).read_text(encoding="utf-8")
    assert source.count("_abort_barrier(pre_exec_barrier)") >= 2


def test_per_rank_input_generation_is_serialized():
    """numpy.random 是进程级全局状态：rank 线程的 seed+生成必须持锁串行，
    否则 GIL 切换会让 rank 间种子流交叉污染，--seed 多卡不可复现。"""
    source = Path(hccl_multi_device.__file__).read_text(encoding="utf-8")
    assert "_INPUT_GEN_LOCK" in source
    assert "with _INPUT_GEN_LOCK:" in source


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        pytest.param(None, False, id="no_attributes"),
        pytest.param({}, False, id="empty_attributes"),
        pytest.param({"group": "hcom_0"}, False, id="plain_group_only"),
        pytest.param({"epWorldSize": 2}, False, id="ep_only_is_single_domain"),
        pytest.param({"tpWorldSize": 2}, False, id="tp_only_is_single_domain"),
        pytest.param({"epWorldSize": 2, "tpWorldSize": 2}, True, id="both_declared"),
    ],
)
def test_dual_comm_requested_is_attribute_driven(attributes, expected):
    """双通信域门控读 attributes 声明（双域=两者都声明），与算子名无关；
    单域声明（grouped GMM 只声明 epWorldSize）回退世界域，不进子域 setup。"""
    assert hccl_multi_device._dual_comm_requested(attributes) is expected


def _rank_ctx(failed=False):
    return SimpleNamespace(
        api_name="aclnnTestComm",
        prof_result=SimpleNamespace(failed=lambda: failed),
        attributes={},
        golden_tensors=None,
    )


def _multi_device_context(ranks):
    return SimpleNamespace(
        api_name="aclnnTestComm",
        _multi_device_thread_contexts={dev: _rank_ctx() for dev in ranks},
        _multi_device_hccl_handles=None,
    )


def _switches(**kwargs):
    defaults = {"plugin_path": None, "proc_timeout": 0}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _NoGoldenResolver:
    def __init__(self, name, pathway, plugin_path):
        pass

    def resolve_golden(self, fallback=None):
        return None


class _PluginGoldenResolver:
    def __init__(self, name, pathway, plugin_path):
        pass

    def resolve_golden(self, fallback=None):
        return _plugin_golden


def _plugin_golden(thread_contexts, device_ids, all_precision):
    for did in device_ids:
        all_precision.append(f"rank{did}:PASS")


def test_compare_multi_device_without_plugin_falls_back_to_dispatcher(monkeypatch):
    """无 plugin：走 hccl_dispatcher 兜底——未迁移算子逐 rank SKIP(NO_GOLDEN)。"""
    monkeypatch.setattr(hccl_multi_device, "get_global_storage", _switches)
    monkeypatch.setattr(hccl_multi_device, "SpecResolver", _NoGoldenResolver)
    context = _multi_device_context((0, 1))

    result = hccl_multi_device.compare_multi_device(context, [0, 1])

    assert result.precision == "rank0:SKIP(NO_GOLDEN),rank1:SKIP(NO_GOLDEN)"
    assert result.passed == "PASS"
    for tc in context._multi_device_thread_contexts.values():
        assert tc.golden_tensors == []


def test_compare_multi_device_dispatches_plugin_golden(monkeypatch):
    """有 plugin：按 (thread_contexts, device_ids, all_precision) 契约分发，
    proc_timeout 注入各 rank attributes。"""
    monkeypatch.setattr(hccl_multi_device, "get_global_storage", lambda: _switches(proc_timeout=120))
    monkeypatch.setattr(hccl_multi_device, "SpecResolver", _PluginGoldenResolver)
    context = _multi_device_context((0, 1))

    result = hccl_multi_device.compare_multi_device(context, [0, 1])

    assert result.precision == "rank0:PASS,rank1:PASS"
    assert result.passed == "PASS"
    for tc in context._multi_device_thread_contexts.values():
        assert tc.attributes["proc_timeout"] == 120


def test_compare_multi_device_rank_failure_short_circuits(monkeypatch):
    """任一 rank 执行失败 → MULTI_DEVICE_EXECUTION_FAILURE，不进 golden。"""
    monkeypatch.setattr(hccl_multi_device, "get_global_storage", _switches)
    monkeypatch.setattr(hccl_multi_device, "SpecResolver", _PluginGoldenResolver)
    context = _multi_device_context((0, 1))
    context._multi_device_thread_contexts[1] = _rank_ctx(failed=True)

    result = hccl_multi_device.compare_multi_device(context, [0, 1])

    assert result.precision == "MULTI_DEVICE_EXECUTION_FAILURE"
    assert result.passed == "FAIL"


class _StubPlan:
    def build_args(self, tensors, scalars, attributes):
        extra = {"someAttr": 1, "testcase": "csv_attribute_value"}
        return (list(tensors) + list(scalars), extra)


def _hook_case():
    case = TestcaseAclnn()
    case.testcase_name = "hook_contract"
    case.api_name = "aclnnTestComm"
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.scalar_dtypes = ()
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.attributes = {"group": "hcom_0"}
    case.tensors = ("t0", "t1")
    case.scalars = ()
    case._param_plan_cache = _StubPlan()
    return case


def test_custom_input_hook_receives_testcase_context(monkeypatch):
    """input hook 契约：kwargs 携 testcase 上下文（框架保留键，不被 CSV
    attributes 同名键覆盖）——多卡 rank 输入重生成的插件触达口。"""
    monkeypatch.setattr(
        "ttk.core_modules.aclnn.input_generate.get_global_storage",
        lambda: SimpleNamespace(short_soc_version="Ascend910B3"),
    )
    case = _hook_case()
    captured = {}

    def hook(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    InputGenerator(case)._call_custom_input(hook)

    assert captured["args"] == ("t0", "t1")
    assert captured["kwargs"]["someAttr"] == 1
    assert captured["kwargs"]["testcase"] is case


def test_thread_device_clone_owned_by_factory():
    """线程设备克隆收口 AclInterface.clone_for_thread（防手工 __dict__ 手术回归）。

    原手工克隆漏重置 _acl_tensor_lists/_acl_scalar_lists/_acl_tensor_to_device_mem，
    且 _acl_device_memories 为不存在的属性名（错名死赋值）。
    """
    source = Path(hccl_multi_device.__file__).read_text(encoding="utf-8")
    assert "main_device.clone_for_thread(dev_id)" in source
    assert "AclInterface.__new__" not in source
    assert "_acl_device_memories" not in source
