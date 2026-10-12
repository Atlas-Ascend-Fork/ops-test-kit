# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Focused contract tests for npu_preprocess-materialized E2E tensor slots."""

from types import SimpleNamespace

import pytest

from ttk.core_modules.aclnn.api_param import AclnnParamPlan
from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.invoke import (
    Invoke,
    apply_aclnn_npu_preprocess_result,
    apply_npu_preprocess_result,
)
from ttk.core_modules.common.testcase.field_types import FIELD_TYPES
from ttk.core_modules.common.testcase.input_data import generate_np_storages
from ttk.core_modules.framework_api.api_model import ParamInfo
from ttk.core_modules.framework_api.api_param import ParamPlan
from ttk.core_modules.framework_api.testcase import TestcaseE2e


class FakeTensor:
    def __init__(self, shape, device="npu:0"):
        self.shape = shape
        self.device = device


def make_case(shapes=((-1,),), dtypes=("int32",), **fields):
    case = TestcaseE2e()
    case.testcase_name = "dynamic_metadata"
    case.api_name = fields.pop("api_name", "torch.fake")
    case.tensor_view_shapes = shapes
    case.tensor_dtypes = dtypes
    case.input_data_ranges = ((None, None),) * len(shapes)
    case.is_valid = True
    for name, value in fields.items():
        setattr(case, name, value)
    return case


def make_plan(params):
    return ParamPlan("torch.fake", params, 0, (), {})


def test_dynamic_shape_header_uses_dynamic_parser():
    assert TestcaseE2e.tensor_property_headers["tensor_view_shapes"][0] is FIELD_TYPES.SHAPELIKE_DYN_NESTED
    parser = TestcaseE2e.get_header_func("tensor_view_shapes")
    assert parser("((-1,), (4, 128))") == ((-1,), (4, 128))


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"shapes": ((-2,),)}, "DYN_SHAPE_INVALID"),
        ({"shapes": (((-1,), (2,)),), "dtypes": (("int32", "int32"),)}, "DYN_SHAPE_IN_TENSORLIST"),
        ({"output_tensor_indexes": (0,)}, "DYN_SHAPE_ON_OUTPUT_SLOT"),
        ({"inplace_input_indexes": (0,)}, "DYN_SHAPE_ON_INPLACE_SLOT"),
        ({"tensor_storage_shapes": ((8,),)}, "DYN_SHAPE_WITH_VIEW_OVERRIDE"),
        ({"dtypes": ("int4",)}, "DYN_SHAPE_DTYPE_UNSUPPORTED"),
    ],
)
def test_dynamic_slot_validation_matrix(fields, reason):
    fields = dict(fields)
    case = make_case(fields.pop("shapes", ((-1,),)), fields.pop("dtypes", ("int32",)), **fields)
    case._check_dyn_shape_slots()
    assert not case.is_valid
    assert case.fail_reason == reason


def test_only_dynamic_slot_dtype_is_checked():
    case = make_case(
        ((4,), (-1,)),
        ("int4", "int32"),
        tensor_storage_shapes=((4,), ()),
        tensor_view_offsets=(1, 0),
    )
    case._check_dyn_shape_slots()
    assert case.is_valid
    assert case.dyn_input_slot_indexes == (1,)


@pytest.mark.parametrize("kind", ["e2e", "aclnn"])
@pytest.mark.parametrize("compressed", [False, True])
def test_dynamic_slot_respects_compressed_storage_broadcast(kind, compressed):
    if kind == "e2e":
        case = make_case(((4,), (-1,)), ("float32", "int32"))
        case.tensor_storage_shapes = ((4,),) if compressed else ((4,), None)
    else:
        case = make_aclnn_case()
        case.tensor_storage_shapes = ((2,),) if compressed else ((2,), None, (2,))

    case._normalize_compressed_fields()
    case._check_dyn_shape_slots()

    if compressed:
        assert not case.is_valid
        assert case.fail_reason == "DYN_SHAPE_WITH_VIEW_OVERRIDE"
    else:
        assert case.is_valid


def test_dynamic_storage_is_deferred():
    case = make_case(((2,), (-1,)), ("float32", "int32"))
    switches = SimpleNamespace(input_distribution="uniform", random_seed=None)
    storages, _ = generate_np_storages(case, switches)
    case.np_storages = storages
    assert case.np_storages[0].shape == (2,)
    assert case.np_storages[1] is None


@pytest.mark.parametrize(
    ("golden_mode", "golden_api", "manual_has_goldens", "has_plugin", "expected"),
    [
        ("Disable", None, False, False, True),
        ("Enable", "Disable", False, False, True),
        ("Enable", None, True, False, True),
        ("Enable", None, False, True, True),
        ("Enable", None, False, False, False),
        ("Enable", "torch.other", False, True, False),
    ],
)
def test_dynamic_slot_requires_usable_golden_source(
    monkeypatch, golden_mode, golden_api, manual_has_goldens, has_plugin, expected
):
    from ttk.core_modules.framework_api import compare as profiling

    case = make_case()
    case.golden_api = golden_api
    switches = SimpleNamespace(golden_mode=golden_mode, plugin_path=None)
    manual_case = SimpleNamespace(has_goldens=manual_has_goldens) if manual_has_goldens else None
    monkeypatch.setattr(profiling, "get_plugin_function", lambda *_args: object() if has_plugin else None)

    assert profiling.has_dynamic_golden_source(case, switches, manual_case) is expected


def test_manual_replay_treats_dynamic_slot_as_none_marker():
    from ttk.core_modules.common.dump.manual_data import _input_specs

    case = make_case(((-1,), (4,)), ("int32", "float32"))
    assert _input_specs(case, "e2e") == [None, ("float32", (4,))]


def test_apply_result_maps_past_tensor_list_without_cpu_backfill():
    case = make_case((((2,), (2,)), (-1,)), (("float32", "float32"), "int32"))
    params = [ParamInfo(name="values", type="List[Tensor]"), ParamInfo(name="metadata", type="Tensor")]
    plan = make_plan(params)
    values = [FakeTensor((2,)), FakeTensor((2,))]
    args, kwargs, _ = plan.build_args([values, None])
    generated = FakeTensor((96,))

    patched = apply_npu_preprocess_result(case, plan, args, kwargs, {"metadata": generated})

    assert args == [values, generated]
    assert patched == [(2, "metadata", generated)]
    assert case.tensor_view_shapes[-1] == (-1,)
    assert case.np_storages is None


def test_apply_result_supports_keyword_only_tensor():
    case = make_case(((4,), (-1,)), ("float32", "int32"))
    params = [
        ParamInfo(name="query", type="Tensor"),
        ParamInfo(name="metadata", type="Tensor", is_keyword_only=True),
    ]
    plan = make_plan(params)
    query = FakeTensor((4,))
    args, kwargs, _ = plan.build_args([query, None])
    generated = FakeTensor((64,))

    apply_npu_preprocess_result(case, plan, args, kwargs, {"metadata": generated})

    assert args == [query]
    assert kwargs["metadata"] is generated


def test_tensor_param_binding_uses_normal_build_rules_for_outputs_and_keywords():
    params = [
        ParamInfo(name="query", type="Tensor"),
        ParamInfo(name="metadata", type="Tensor", is_keyword_only=True),
        ParamInfo(name="out", type="Tensor", is_optional=True),
    ]
    plan = ParamPlan("torch.fake", params, 0, (2,), {})
    query, output = FakeTensor((4,)), FakeTensor((2,))
    bindings = plan.tensor_param_bindings([query, None, output])
    assert bindings["query"][0][0] == 0
    assert bindings["metadata"][0][0] == 1
    assert bindings["out"][0][0] == 2


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        ({}, "not materialized"),
        ({"query": FakeTensor((4,))}, "not declared"),
        ({"metadata": FakeTensor((8,), "cpu")}, "expected npu:0"),
    ],
)
def test_apply_result_rejects_contract_violations(returned, message):
    case = make_case(((4,), (-1,)), ("float32", "int32"))
    plan = make_plan([ParamInfo(name="query", type="Tensor"), ParamInfo(name="metadata", type="Tensor")])
    args, kwargs, _ = plan.build_args([FakeTensor((4,)), None])
    with pytest.raises(RuntimeError, match=message):
        apply_npu_preprocess_result(case, plan, args, kwargs, returned)


def test_invoke_accepts_dict_and_rejects_non_tensor_values():
    """invoke 是纯执行器：dict 形态校验后原样返回（落地由通路 absorb 负责）。"""
    case = make_case()
    switches = SimpleNamespace(short_soc_version="Ascend950", plugin_path=None)
    generated = FakeTensor((32,))

    result = Invoke.npu_preprocess(
        case,
        lambda metadata, **kwargs: {"metadata": generated},
        switches,
        device_args=([None], {}),
    )
    assert result == {"metadata": generated}

    with pytest.raises(RuntimeError, match="not a tensor"):
        Invoke.npu_preprocess(
            case,
            lambda metadata, **kwargs: {"metadata": (32,)},
            switches,
            device_args=([None], {}),
        )


def test_absorb_rejects_dict_without_dynamic_slot_declaration():
    """无 -1 声明却落地物化 dict → apply 逐名声明校验拒绝。"""
    case = make_case(((8,),), ("float32",))  # 无 -1 槽
    plan = make_plan([ParamInfo(name="x", type="Tensor")])
    args, kwargs = [FakeTensor((8,))], {}

    with pytest.raises(RuntimeError, match="not declared as a supported -1 slot"):
        case.absorb_preprocess({"metadata": FakeTensor((32,))}, plan, args, kwargs)


def make_aclnn_case(metadata_shape=(-1,), metadata_dtype="int32"):
    import numpy as np
    import torch

    case = TestcaseAclnn()
    case.testcase_name = "aclnn_dynamic_metadata"
    case.api_name = "aclnnFake"
    case.tensor_view_shapes = ((2,), metadata_shape, (2,))
    case.tensor_dtypes = ("float32", metadata_dtype, "float32")
    case.tensor_formats = ("ND", "ND", "ND")
    case.tensor_storage_shapes = ()
    case.tensor_view_strides = ()
    case.tensor_view_offsets = ()
    case.scalar_dtypes = ()
    case.output_tensor_indexes = (2,)
    case.output_inplace_indexes = ()
    case.tensors = [torch.ones(2), None, torch.ones(2)]
    case.np_storages = [np.ones(2, dtype=np.float32), None, np.ones(2, dtype=np.float32)]
    case.is_valid = True
    return case


def make_aclnn_plan():
    plan = object.__new__(AclnnParamPlan)
    plan.api_name = "aclnnFake"
    plan.param_layout = [
        (AclnnParamPlan.TENSOR, "query", "aclTensor*", None),
        (AclnnParamPlan.TENSOR, "metadataOptional", "aclTensor*", None),
        (AclnnParamPlan.TENSOR, "output", "aclTensor*", None),
    ]
    plan.tensor_count = 3
    plan.scalar_count = 0
    return plan


def test_aclnn_dynamic_header_and_storage_are_deferred():
    from ttk.core_modules.aclnn.input_generate import InputGenerator

    assert TestcaseAclnn.tensor_property_headers["tensor_view_shapes"][0] is FIELD_TYPES.SHAPELIKE_DYN_NESTED
    case = make_aclnn_case()
    generator = InputGenerator(case)
    generator._switch = SimpleNamespace(input_distribution="uniform", random_seed=None)
    generator._realtime_random_tensors()
    assert case.np_storages[0].shape == (2,)
    assert case.np_storages[1] is None


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        (make_aclnn_case((-2,)), "DYN_SHAPE_INVALID"),
        (make_aclnn_case(), "DYN_SHAPE_ON_OUTPUT_SLOT"),
        (make_aclnn_case(metadata_dtype="int4"), "DYN_SHAPE_DTYPE_UNSUPPORTED"),
    ],
)
def test_aclnn_dynamic_slot_validation(case, reason):
    if reason == "DYN_SHAPE_ON_OUTPUT_SLOT":
        case.output_tensor_indexes = (1, 2)
    case._check_dyn_shape_slots()
    assert not case.is_valid
    assert case.fail_reason == reason


def test_aclnn_manual_replay_treats_dynamic_slot_as_none_marker():
    from ttk.core_modules.common.dump.manual_data import _input_specs

    assert _input_specs(make_aclnn_case(), "aclnn")[1] is None


def test_apply_aclnn_result_materializes_host_input_without_changing_csv_shape():
    import torch

    case = make_aclnn_case()
    generated = torch.arange(12, dtype=torch.int32)

    patched = apply_aclnn_npu_preprocess_result(case, make_aclnn_plan(), {"metadataOptional": generated})

    assert patched[0][:2] == (1, "metadataOptional")
    assert case.flat_tensors[1] is None
    assert case.np_storages[1] is None
    assert tuple(case._aclnn_dynamic_tensors[1].shape) == (12,)
    assert case.tensor_view_shapes[1] == (-1,)


def test_aclnn_phase1_uses_materialized_shape_for_dynamic_slot():
    import torch

    from ttk.core_modules.aclnn.execute import Phase1ParamBuilder

    class FakeDevice:
        def __init__(self):
            self.storage_shapes = []

        def create_acl_tensor(self, _tensor, _format, storage_shape, **_kwargs):
            self.storage_shapes.append(tuple(storage_shape))
            return len(self.storage_shapes)

    case = make_aclnn_case()
    apply_aclnn_npu_preprocess_result(
        case,
        make_aclnn_plan(),
        {"metadataOptional": torch.arange(12, dtype=torch.int32)},
    )
    device = FakeDevice()

    Phase1ParamBuilder(case, device)._create_acl_tensor()

    assert device.storage_shapes[1] == (12,)


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        ({}, "not materialized"),
        ({"query": None}, "not declared"),
        ({"metadataOptional": None}, "not a torch.Tensor"),
    ],
)
def test_apply_aclnn_result_rejects_contract_violations(returned, message):
    if returned.get("query") is None and "query" in returned:
        import torch

        returned = {"query": torch.ones(1)}
    with pytest.raises(RuntimeError, match=message):
        apply_aclnn_npu_preprocess_result(make_aclnn_case(), make_aclnn_plan(), returned)


def test_apply_aclnn_result_uses_actual_shape_and_rejects_dtype_mismatch():
    import torch

    partial = make_aclnn_case((4, -1))
    apply_aclnn_npu_preprocess_result(
        partial,
        make_aclnn_plan(),
        {"metadataOptional": torch.ones((3, 8), dtype=torch.int32)},
    )
    assert tuple(partial._aclnn_dynamic_tensors[1].shape) == (3, 8)

    with pytest.raises(RuntimeError, match="dtype"):
        apply_aclnn_npu_preprocess_result(
            make_aclnn_case(),
            make_aclnn_plan(),
            {"metadataOptional": torch.ones(8, dtype=torch.float32)},
        )


def test_tf_graph_rejects_preprocess_tensor_materialization(monkeypatch, caplog):
    from ttk.core_modules.framework_api.tf import graph_runner as tf_graph_execution

    case = make_case(((4,),), ("float32",), api_name="tf.fake")
    backend = SimpleNamespace(is_npu=lambda: True, device_scope=lambda _dev_id: None)
    generated = FakeTensor((4,))
    monkeypatch.setattr(tf_graph_execution, "prepare_device_args", lambda *_args: ([generated], {}))
    monkeypatch.setattr(
        tf_graph_execution,
        "Invoke",
        SimpleNamespace(npu_preprocess=lambda *_args, **_kwargs: {"input": generated}),
    )

    result = tf_graph_execution.execute_tf_graph(
        case,
        backend,
        0,
        SimpleNamespace(plugin_path=None),
        object(),
        lambda value: value,
        False,
        False,
        [generated],
        False,
    )

    assert result == ([], None, None)
    assert "TF graph mode does not support npu_preprocess tensor materialization" in caplog.text


# ── absorb 协议（npu_preprocess 结果落地）─────────────────────────────


def test_absorb_none_without_declaration_is_noop():
    """无 -1 声明 × None 返回 → 基类默认 no-op（纯副作用）。"""
    case = make_case(((8,),), ("float32",))
    assert case.absorb_preprocess(None, None) is None


def test_absorb_none_with_declaration_requires_slot_refresh():
    """有 -1 声明 × None 返回 × 槽位未反刷 → 契约失败（提示双通道）。"""
    case = make_case()  # ((-1,),)
    with pytest.raises(RuntimeError, match="not materialized.*dict return or slot-table write"):
        case.absorb_preprocess(None, None)


def test_absorb_none_with_slot_refresh_passes_aclnn_implicit_channel():
    """有 -1 声明 × None 返回 × 槽位已反刷 → 隐式通道核验放行。"""
    case = make_case()
    materialized = FakeTensor((32,))
    case.set_flat_tensor(0, materialized)
    assert case.absorb_preprocess(None, None) is None  # E2E：批 2 前无补丁，核验放行


def test_absorb_e2e_dict_patches_args():
    """E2E 显式通道：dict → apply 回填 args，返回原 dict。"""
    case = make_case()
    case._param_plan_cache = make_plan([ParamInfo(name="metadata", type="Tensor")])
    plan = case._param_plan_cache
    args, kwargs = [None], {}
    generated = FakeTensor((32,))

    result = case.absorb_preprocess({"metadata": generated}, plan, args, kwargs)

    assert result == {"metadata": generated}
    assert args == [generated]


def test_absorb_aclnn_dict_writes_dynamic_tensors():
    """ACLNN 显式通道：dict → 主机副本写 _aclnn_dynamic_tensors，返回 True。"""
    import torch

    case = make_aclnn_case()
    plan = make_aclnn_plan()
    generated = torch.zeros(32, dtype=torch.int32)

    result = case.absorb_preprocess({"metadataOptional": generated}, plan)

    assert result is True
    assert 1 in case._aclnn_dynamic_tensors
    host = case._aclnn_dynamic_tensors[1]
    assert host.dtype == torch.int32
    assert host.device.type == "cpu"


def test_absorb_base_rejects_dict_on_unsupported_pathway():
    """未实现 absorb 覆写的通路子类：基类默认对 dict 拒绝（NotImplementedError）。"""
    from ttk.core_modules.common.testcase.api_base import ApiTestcaseBase

    class _Bare(ApiTestcaseBase):
        pass

    bare = _Bare.__new__(_Bare)
    with pytest.raises(NotImplementedError):
        bare.absorb_preprocess({"x": FakeTensor((2,))}, None)


def test_failure_prefix_matches_invoke_constant():
    """api_base 侧前缀字面量与 Invoke.NPU_PREPROCESS_FAILURE 同值（防漂移）。"""
    from ttk.core_modules.common.invoke import Invoke
    from ttk.core_modules.common.testcase.api_base import _NPU_PREPROCESS_FAILURE_PREFIX

    assert _NPU_PREPROCESS_FAILURE_PREFIX == Invoke.NPU_PREPROCESS_FAILURE
