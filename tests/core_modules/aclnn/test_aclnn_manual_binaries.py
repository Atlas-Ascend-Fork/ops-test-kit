#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn manual_tensor_binaries / manual_golden_binaries 文件装载测试。

覆盖：
- generate_np_storages overrides：声明槽位从文件装载、未声明槽位保持常规生成
  （随机输入/纯输出常量）、动态槽声明报错
- flat_manual_tensor_binaries：嵌套声明随 TensorList 分布展开
- _use_manual_input：精确对齐——非动态输入槽缺路径报错、动态槽必须 None、
  纯输出槽可缺省、overrides 构造、声明数超槽报错
- _load_golden_from_file：路径与输出槽一一对齐；未声明槽位回退自动生成
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.aclnn import golden as aclnn_golden
from ttk.core_modules.aclnn import input_generate as aclnn_input_generate
from ttk.core_modules.aclnn import worker as aclnn_worker
from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.testcase.input_data import generate_np_storages


def _stub_case(view_shapes, dtypes, pure_output_indexes=()):
    """generate_np_storages 所需的最小 testcase 桩（batch 字段缺省 → 无 batch 关系）。"""
    flat_storage_shapes = tuple(view_shapes)

    def flat_storage_shape(idx):
        return flat_storage_shapes[idx]

    return SimpleNamespace(
        api_name="aclnnTest",
        flat_tensor_view_shapes=tuple(view_shapes),
        flat_tensor_dtypes=tuple(dtypes),
        flat_input_data_ranges=(),
        flat_storage_shape=flat_storage_shape,
        pure_output_indexes=tuple(pure_output_indexes),
    )


def _stub_switches(seed=None):
    return SimpleNamespace(input_distribution="uniform", random_seed=seed)


def _flat_index_of(dist):
    """桩用 flat_index_of（与 ApiTestcaseBase.flat_index_of 同语义）。"""

    def flat_index_of(nested_idx):
        return sum(max(d, 1) for d in dist[:nested_idx]) if dist else nested_idx

    return flat_index_of


def test_generate_np_storages_overrides_load_declared_slots(tmp_path):
    """声明槽位从文件装载（dtype/shape 按 CSV 声明），未声明槽保持常规生成。"""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    fp = tmp_path / "input_0.npy"
    np.save(fp, arr)

    case = _stub_case(((2, 3), (4,), (3,)), ("float32", "int32", "float32"), pure_output_indexes=(2,))
    storages, ranges = generate_np_storages(case, _stub_switches(), overrides={0: str(fp)})

    np.testing.assert_array_equal(storages[0], arr)
    assert ranges[0] == (None, None)
    # 未声明输入槽：随机生成（非文件内容）
    assert storages[1] is not None
    assert storages[1].shape == (4,)
    assert ranges[1] != (None, None)
    # 纯输出槽：常量 1 初始化（常规语义不变）
    assert storages[2].dtype == np.float32
    np.testing.assert_array_equal(storages[2], np.ones(3, np.float32))


def test_generate_np_storages_override_dynamic_slot_raises(tmp_path):
    """动态/未知 shape 槽位声明路径 → RuntimeError（无 shape 无从装载）。"""
    fp = tmp_path / "dyn.bin"
    fp.write_bytes(b"\x00" * 8)
    case = _stub_case((None,), ("float32",))
    with pytest.raises(RuntimeError, match="dynamic/unknown-shape slot 0"):
        generate_np_storages(case, _stub_switches(), overrides={0: str(fp)})


def test_flat_manual_tensor_binaries_flattens_nesting():
    """嵌套声明（随 TensorList 分布）展开为 flat 槽序；空声明为 ()。"""
    case = TestcaseAclnn()
    case.manual_tensor_binaries = (("a.bin", "b.bin"), None, "c.bin")
    assert case.flat_manual_tensor_binaries == ("a.bin", "b.bin", None, "c.bin")

    case.manual_tensor_binaries = ("a.bin", "b.bin")
    assert case.flat_manual_tensor_binaries == ("a.bin", "b.bin")

    case.manual_tensor_binaries = None
    assert case.flat_manual_tensor_binaries == ()


def test_use_manual_input_builds_overrides(monkeypatch):
    """overrides 按声明槽构造；纯输出槽可缺省（不进 overrides、保持常量初始化）。"""
    captured = {}

    def fake_generate(case, switches, zero_init_apis=(), overrides=None):
        captured["overrides"] = overrides
        captured["zero_init_apis"] = zero_init_apis
        return ["s0", "s1", "s2"], [(None, None)] * 3

    monkeypatch.setattr(aclnn_input_generate, "generate_np_storages", fake_generate)
    case = SimpleNamespace(
        flat_manual_tensor_binaries=("f0.bin", "f1.bin", None),
        flat_tensor_view_shapes=((2,), (2,), (2,)),
        pure_output_indexes=(2,),
    )
    aclnn_input_generate.InputGenerator(case)._use_manual_input()

    assert captured["overrides"] == {0: "f0.bin", 1: "f1.bin"}
    assert case.np_storages == ["s0", "s1", "s2"]


def test_use_manual_input_dynamic_slot_none_allowed(monkeypatch):
    """动态槽（shape 未知/-1）必须 None——显式 None 合法，不进 overrides。"""
    captured = {}

    def fake_generate(case, switches, zero_init_apis=(), overrides=None):
        captured["overrides"] = overrides
        return ["s0", "s1"], [(None, None)] * 2

    monkeypatch.setattr(aclnn_input_generate, "generate_np_storages", fake_generate)
    case = SimpleNamespace(
        flat_manual_tensor_binaries=("f0.bin", None),
        flat_tensor_view_shapes=((2,), None),
        pure_output_indexes=(),
    )
    aclnn_input_generate.InputGenerator(case)._use_manual_input()

    assert captured["overrides"] == {0: "f0.bin"}


def test_use_manual_input_missing_input_slot_raises(monkeypatch):
    """非动态输入槽缺路径 → RuntimeError（防止"少给"静默回退随机数）。"""
    case = SimpleNamespace(
        flat_manual_tensor_binaries=("f0.bin", None, "f2.bin"),
        flat_tensor_view_shapes=((2,), (2,), (2,)),
        pure_output_indexes=(),
    )
    with pytest.raises(RuntimeError, match="missing paths for non-dynamic input slots \\[1\\]"):
        aclnn_input_generate.InputGenerator(case)._use_manual_input()


def test_use_manual_input_rejects_overflow_declaration(monkeypatch):
    """声明路径数超过 flat 张量槽数 → RuntimeError。"""
    case = SimpleNamespace(
        flat_manual_tensor_binaries=("a.bin", "b.bin", "c.bin"),
        flat_tensor_view_shapes=((2,), (2,)),
        pure_output_indexes=(),
    )
    with pytest.raises(RuntimeError, match="more than flat tensor slots"):
        aclnn_input_generate.InputGenerator(case)._use_manual_input()


def test_load_golden_from_file_aligns_output_slots(tmp_path):
    """golden 路径与 output_tensor_indexes 槽序对齐、全覆盖装载。"""
    g0 = np.full((3, 5), 7.0, dtype=np.float32)
    g1 = np.full((4,), 9.0, dtype=np.float32)
    fp0 = tmp_path / "golden_0.npy"
    fp1 = tmp_path / "golden_1.npy"
    np.save(fp0, g0)
    np.save(fp1, g1)

    case = SimpleNamespace(
        manual_golden_binaries=(str(fp0), str(fp1)),
        output_tensor_indexes=(0, 2),
        tensor_list_dist=(0, 0, 0),
        flat_index_of=_flat_index_of((0, 0, 0)),
        flat_output_dtypes=("float32", "float32", "float32"),
        flat_output_view_shapes=((3, 5), (2,), (4,)),
    )
    goldens = aclnn_golden.GoldenGenerator(case)._load_golden_from_file()

    assert len(goldens) == 2
    np.testing.assert_array_equal(goldens[0], g0)
    np.testing.assert_array_equal(goldens[1], g1)


def test_load_golden_from_file_missing_slot_falls_back_to_auto(tmp_path, monkeypatch):
    """golden 未声明槽位回退自动生成（从已装载输入推导），比对不缺席。"""
    g0 = np.full((3, 5), 7.0, dtype=np.float32)
    fp0 = tmp_path / "golden_0.npy"
    np.save(fp0, g0)
    auto = [np.full((3, 5), 1.0, np.float32), np.full((4,), 2.0, np.float32)]
    monkeypatch.setattr(aclnn_golden.GoldenGenerator, "_generate_golden", lambda self: auto)

    case = SimpleNamespace(
        manual_golden_binaries=(str(fp0),),
        output_tensor_indexes=(0, 2),
        tensor_list_dist=(0, 0, 0),
        flat_index_of=_flat_index_of((0, 0, 0)),
        flat_output_dtypes=("float32", "float32", "float32"),
        flat_output_view_shapes=((3, 5), (2,), (4,)),
    )
    goldens = aclnn_golden.GoldenGenerator(case)._load_golden_from_file()

    np.testing.assert_array_equal(goldens[0], g0)
    np.testing.assert_array_equal(goldens[1], auto[1])


def test_load_golden_from_file_tensorlist_slot_uses_flat_prefix(tmp_path):
    """TensorList 输出槽：dtype/shape 取该槽首个 flat 输出的声明。"""
    g = np.arange(8, dtype=np.float32).reshape(4, 2)
    fp = tmp_path / "golden_list.npy"
    np.save(fp, g)

    case = SimpleNamespace(
        manual_golden_binaries=(str(fp),),
        output_tensor_indexes=(1,),
        tensor_list_dist=(0, 2),
        flat_index_of=_flat_index_of((0, 2)),
        flat_output_dtypes=("float32", "float32", "float32"),
        flat_output_view_shapes=((4, 2), (4, 2), (4, 2)),
    )
    goldens = aclnn_golden.GoldenGenerator(case)._load_golden_from_file()

    np.testing.assert_array_equal(goldens[0], g)


# ══════════════════════════════════════════════════════════════════════════
# manual_xpu_binaries 三方短路（CSV/case 目录 → 跳过远端 dispatch）
# ══════════════════════════════════════════════════════════════════════════
def test_manual_xpu_header_registered():
    assert "manual_xpu_binaries" in TestcaseAclnn.complete_headers


def _mx_case(path):
    # 装载器 duck-typing：flat 三件套即可（不依赖完整 TestcaseAclnn 解析链）
    return SimpleNamespace(
        testcase_name="aclnn_manual_xpu",
        api_name="aclnnAdd",
        flat_manual_xpu_binaries=(str(path),) if path else None,
        flat_output_view_shapes=((8,),),
        flat_output_dtypes=("float32",),
        xpu_attrs={},
    )


def _mx_slot_case(paths):
    # manual-data case 目录槽位形态（load_manual_third_parties 的第二来源）
    return SimpleNamespace(has_third_parties=True, third_party_paths=lambda: list(paths))


def _mx_switches():
    return SimpleNamespace(dump_config=SimpleNamespace(dump_xpu=False), xpu_perf=False, golden_mode="Enable")


def _mx_process_ctx(recorder):
    return SimpleNamespace(notify_status=lambda stage, *_: recorder.append(stage))


def test_manual_xpu_short_circuits_remote_collect(tmp_path, monkeypatch):
    path = tmp_path / "gpu.bin"
    expected = np.arange(8, dtype=np.float32)
    expected.tofile(path)

    recorder = []

    def _no_remote(**kwargs):
        raise AssertionError("remote collect must not run when manual_xpu_binaries declared")

    monkeypatch.setattr("ttk.core_modules.common.xpu.collect_third_party", _no_remote)

    xpu = aclnn_worker._dispatch_xpu(_mx_case(path), _mx_switches(), _mx_process_ctx(recorder), True, False)
    assert not xpu.pending  # manual 源 → ready
    third_parties, xpu_results, provider = xpu.collect()

    np.testing.assert_array_equal(third_parties[0], expected)
    assert xpu_results == {"manual_binary": {"status": "PASS", "api": "manual_xpu_binaries"}}
    assert provider == "manual_binary"
    assert recorder == ["OnLoadManualXpu"]


@pytest.mark.parametrize("via_slots", [False, True], ids=["csv-path", "case-dir-slots"])
def test_golden_xpu_mode_also_short_circuits(tmp_path, monkeypatch, via_slots):
    """golden-mode XPU + manual（CSV 路径 / case 目录槽位）：golden 位真值源同样来自本地装载。"""
    path = tmp_path / "gpu.bin"
    np.arange(8, dtype=np.float32).tofile(path)

    recorder = []

    def _no_remote(**kwargs):
        raise AssertionError("remote collect must not run when manual_xpu_binaries declared")

    monkeypatch.setattr("ttk.core_modules.common.xpu.collect_third_party", _no_remote)

    golden_switches = SimpleNamespace(dump_config=SimpleNamespace(dump_xpu=False), xpu_perf=False, golden_mode="XPU")
    manual_case = _mx_slot_case((str(path),)) if via_slots else None
    xpu = aclnn_worker._dispatch_xpu(
        _mx_case(None if via_slots else path), golden_switches, _mx_process_ctx(recorder), False, True, manual_case
    )
    third_parties, xpu_results, _provider = xpu.collect()

    assert third_parties is not None
    np.testing.assert_array_equal(third_parties[0], np.arange(8, dtype=np.float32))
    assert xpu_results["manual_binary"]["status"] == "PASS"
    assert recorder == ["OnLoadManualXpu"]


@pytest.mark.parametrize("miss_third_party", [False, True])
def test_undeclared_goes_remote(tmp_path, monkeypatch, caplog, miss_third_party):
    """同步路径：未声明 manual → 远端采集；cross_check 无三方输出时告警。"""
    import logging

    case = _mx_case(tmp_path / "unused.bin")
    case.flat_manual_xpu_binaries = None
    recorder = []
    sentinel_third = [np.arange(8, dtype=np.float32)]

    def _remote(**kwargs):
        recorder.append("remote")
        if miss_third_party:
            return None, None, None
        return "remote", sentinel_third, {"remote": {"status": "PASS"}}

    monkeypatch.setattr("ttk.core_modules.common.xpu.collect_third_party", _remote)
    # 免完整 TestcaseAclnn 解析链：远端分支的输入提取器全部打桩
    monkeypatch.setattr(aclnn_worker, "_aclnn_xpu_inputs", lambda ctx: [])
    monkeypatch.setattr(aclnn_worker, "_aclnn_xpu_input_names", lambda ctx: [])
    monkeypatch.setattr(aclnn_worker, "_aclnn_param_order", lambda ctx: None)
    monkeypatch.setattr(aclnn_worker, "_aclnn_xpu_input_dtypes", lambda ctx: None)
    monkeypatch.setattr("ttk.core_modules.common.xpu.xpu_overlap_enabled", lambda: False)  # 本用例钉同步路径

    caplog.set_level(logging.WARNING)
    xpu = aclnn_worker._dispatch_xpu(case, _mx_switches(), _mx_process_ctx(recorder), True, False)
    third_parties, xpu_results, provider = xpu.collect()

    assert recorder == ["OnXpuProfiling", "remote"]
    if miss_third_party:
        assert third_parties is None
        assert any("no third_party output" in record.message for record in caplog.records)
    else:
        assert third_parties is sentinel_third
        assert provider == "remote"
