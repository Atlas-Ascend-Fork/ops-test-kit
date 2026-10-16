#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
XPU 三方能力专项测试：collect 门面（remote-only）/ ExecutionSpec / 模式位。
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.common.xpu import (
    MANUAL_XPU_PROVIDER,
    ManualXpuLoadError,
    XpuDispatch,
    dump_manual_xpu_archive,
    load_manual_third_parties,
    manual_xpu_results,
    overlap_deadline_s,
    snapshot_xpu_inputs,
    xpu_join_timeout,
    xpu_overlap_enabled,
)

torch = pytest.importorskip("torch")


def _switches(**kw):
    from argparse import Namespace

    kw.setdefault("backend", "npu")
    return Namespace(**kw)


class TestExecutionSpecFields:
    def test_data_fields_default_none(self):
        from ttk.remote import ExecutionSpec

        spec = ExecutionSpec(provider="torch", api="torch.add")
        assert spec.inputs is None
        assert spec.attrs is None
        assert spec.testcase_name is None

    def test_data_fields_carry_payload(self):
        from ttk.remote import ExecutionSpec

        spec = ExecutionSpec(provider="torch", api="torch.add", inputs=[1], attrs={"a": 1})
        assert spec.inputs == [1]
        assert spec.attrs == {"a": 1}


class TestCollectFacadeRouting:
    def test_remote_delegates_to_remote_client(self, monkeypatch):
        import ttk.core_modules.common.xpu as xpu_mod
        import ttk.remote.client as client_mod

        def fake_remote_collect(**kw):
            calls.update(kw)
            return ("p", None, {})

        calls = {}
        monkeypatch.setattr(client_mod, "collect_third_party", fake_remote_collect)
        out = xpu_mod.collect_third_party(
            op_name="Add",
            inputs=[],
            input_names=[],
            op_type=None,
            attributes={},
            testcase_name="t",
            switches=_switches(),
            need_data=True,
        )
        assert out == ("p", None, {})
        assert calls["op_name"] == "Add"


class TestXpuMode:
    """xpu_mode：worker 三方采集编排的按位模式查询（DATA|PERF）。"""

    @staticmethod
    def _sw(**kw):
        from argparse import Namespace

        return Namespace(dump_config=Namespace(dump_xpu=kw.pop("dump_xpu", False)), xpu_perf=kw.pop("xpu_perf", False))

    def test_mode_zero_when_nothing_requested(self):
        from ttk.core_modules.common.xpu import XPU_DATA, XPU_PERF, xpu_mode

        assert xpu_mode(self._sw(), need_data=False) == 0
        assert (XPU_DATA, XPU_PERF) == (1, 2)  # 位值即公开契约

    def test_need_data_sets_data_bit(self):
        from ttk.core_modules.common.xpu import XPU_DATA, xpu_mode

        assert xpu_mode(self._sw(), need_data=True) == XPU_DATA

    def test_dump_xpu_sets_data_bit_without_cross_check(self):
        """--dump xpu 单独即触发 DATA 采集（不依赖 cross_check 判据）。"""
        from ttk.core_modules.common.xpu import XPU_DATA, xpu_mode

        assert xpu_mode(self._sw(dump_xpu=True), need_data=False) == XPU_DATA

    def test_xpu_perf_sets_perf_bit_and_combines(self):
        from ttk.core_modules.common.xpu import XPU_DATA, XPU_PERF, xpu_mode

        assert xpu_mode(self._sw(xpu_perf=True), need_data=False) == XPU_PERF
        assert xpu_mode(self._sw(xpu_perf=True, dump_xpu=True), need_data=False) == (XPU_DATA | XPU_PERF)
        assert xpu_mode(self._sw(xpu_perf=True), need_data=True) == (XPU_DATA | XPU_PERF)


class TestBindParams:
    """bind_params 按名绑定契约（自 ttk.core_modules.common.xpu 单份归位）。"""

    @staticmethod
    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            pytest.param({"reduction": "none"}, (2.0, 0.25, "none"), id="first_two_omitted"),
            pytest.param({"gamma": 2.0, "reduction": "none"}, (2.0, 0.25, "none"), id="middle_omitted"),
            pytest.param({"alpha": -1.6e38, "reduction": "none"}, (2.0, -1.6e38, "none"), id="head_omitted"),
            pytest.param({"gamma": 3.689, "alpha": 0.7, "reduction": "none"}, (3.689, 0.7, "none"), id="all_given"),
        ],
    )
    def test_omitted_earlier_optional_keeps_slot_alignment(attrs, expected):
        """省略前序可选属性时, 后面的值按名落位不得左移错位(类形式三方绑定场景)。"""
        from ttk.core_modules.common.xpu import bind_params

        class Impl:
            def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
                self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

        pool = dict(attrs)
        pool.update({"pred": "P", "target": "T", "weight": None})
        ia, ik = bind_params(Impl.__init__, pool, warn_leftover=False)
        inst = Impl(*ia, **ik)
        assert (inst.gamma, inst.alpha, inst.reduction) == expected

    @staticmethod
    def test_pre_star_default_holds_slot_for_varargs_and_kwonly():
        """缺省的前置可选参数占位后, *args 收集剩余、keyword-only 仍走 Python 默认。"""
        from ttk.core_modules.common.xpu import bind_params

        def f(a=1, b=2, *rest, key=None):
            return (a, b, rest, key)

        ia, ik = bind_params(f, {"b": 9, "zzz": "leftover", "key": "K"})
        assert f(*ia, **ik) == (1, 9, ("leftover",), "K")

        def g(x, *, axis=-1):
            return (x, axis)

        ia, ik = bind_params(g, {"x": 1})
        assert g(*ia, **ik) == (1, -1)


# ══════════════════════════════════════════════════════════════════════════
# XPU/NPU 执行重叠（XpuDispatch 句柄 / 快照契约 / deadline 补偿 / 总闸）
# + manual_xpu 本地装载（loader / 归档）——common/xpu 域
# ══════════════════════════════════════════════════════════════════════════
def _overlap_switches(proc_timeout=None):
    return SimpleNamespace(proc_timeout=proc_timeout)


def _manual_xpu_case(paths, shapes=((8,),), dtypes=("float32",), name="manual_xpu_case"):
    return SimpleNamespace(
        testcase_name=name,
        flat_manual_xpu_binaries=paths,
        flat_output_shapes=shapes,
        flat_output_dtypes=dtypes,
    )


class TestXpuDispatchHandle:
    """句柄语义：ready 即取幂等 / background 透传 / 异常重放 / discard。"""

    def test_ready_collect_is_idempotent(self):
        handle = XpuDispatch.ready(("third", "results", "provider"))
        assert not handle.pending
        assert handle.collect() == ("third", "results", "provider")
        assert handle.collect() == ("third", "results", "provider")  # 幂等

    def test_background_passes_value_and_pending_flag(self):
        handle = XpuDispatch.background(lambda: (1, 2, 3))
        assert handle.pending
        assert handle.collect() == (1, 2, 3)
        assert not handle.pending  # 收割后不再是 pending
        assert handle.collect() == (1, 2, 3)  # 幂等

    def test_background_runs_before_collect(self):
        """重叠本质：fn 在 collect 之前已在执行（NPU 执行期并行）。"""
        order = []
        handle = XpuDispatch.background(lambda: order.append("xpu") or "done")
        for _ in range(200):
            if order:
                break
            time.sleep(0.01)
        assert order == ["xpu"]  # collect 前已执行
        assert handle.collect() == "done"

    def test_background_reraises_unexpected_exception(self):
        """线程内异常在 collect 处重放。"""

        def _bug():
            raise KeyError("unexpected")

        handle = XpuDispatch.background(_bug)
        with pytest.raises(KeyError, match="unexpected"):
            handle.collect()

    def test_discard_abandons_and_guards_collect(self):
        started = threading.Event()

        def _slow():
            started.set()
            time.sleep(30)

        handle = XpuDispatch.background(_slow)
        started.wait(5)
        handle.discard(timeout=0.1)  # 不等 30s
        with pytest.raises(RuntimeError, match="discarded"):
            handle.collect()

    def test_background_join_timeout_bounds_collect(self):
        """join_timeout 生效：线程滞留时 collect 抛 TimeoutError，不无限等待。"""
        started = threading.Event()

        def _stuck():
            started.set()
            time.sleep(30)

        handle = XpuDispatch.background(_stuck, join_timeout=0.1)
        started.wait(5)
        with pytest.raises(TimeoutError, match="join timeout"):
            handle.collect()


class TestSnapshotInputs:
    """快照契约：numpy 叶子 copy、嵌套递归、与原数组解耦。"""

    def test_deep_snapshot_decouples_from_mutation(self):
        arr = np.ones(4)
        snapshot = snapshot_xpu_inputs([arr, (arr, None), "attr"])
        arr[:] = 99
        assert snapshot[0][0] == 1
        assert snapshot[1][0][0] == 1
        assert snapshot[2] == "attr"


class TestOverlapDeadline:
    """deadline 补偿：dispatch_deadline + NPU 预算（锚点 --proc-timeout = 用例
    全程上界，语义即 NPU 段上界——长用例预算同步放大，不新增超时失败）。"""

    @pytest.mark.parametrize(
        ("proc_timeout", "expected"),
        [(None, 300.0 + 600.0), (60, 300.0 + 60.0), (3600, 300.0 + 3600.0)],
    )
    def test_deadline_budget(self, proc_timeout, expected):
        assert overlap_deadline_s(_switches(proc_timeout=proc_timeout)) == expected


class TestOverlapGate:
    """remote config 总闸（ttk.conf.yaml remote 段 xpu_overlap，缺省 true）。"""

    def test_default_enabled(self):
        assert xpu_overlap_enabled() is True

    def test_config_false_disables(self, monkeypatch):
        import ttk.config.loader as loader_mod

        monkeypatch.setattr(loader_mod, "get_remote_config", lambda: SimpleNamespace(xpu_overlap=False))
        assert xpu_overlap_enabled() is False


class TestKernelDispatchOverlap:
    """kernel 分流：cross_check + 重叠开启 → background（远端在 collect 前执行）。"""

    def test_background_dispatch_overlaps_and_collects(self, monkeypatch):
        from ttk.core_modules.kernel import worker as kernel_worker

        order = []
        case = SimpleNamespace(
            op_name="Add",
            testcase_name="overlap_case",
            flat_manual_xpu_binaries=None,
        )

        def _fake_do_xpu_profiling(context, mode, deadline_s=None, inputs=None):
            order.append("xpu-start")
            context.xpu_results = {"remote": {"status": "PASS", "api": "x", "outputs": [[np.arange(4.0)]]}}
            time.sleep(0.05)
            order.append("xpu-done")
            return "remote"

        monkeypatch.setattr(kernel_worker, "_do_xpu_profiling", _fake_do_xpu_profiling)
        monkeypatch.setattr(kernel_worker, "_xpu_inputs", lambda ctx: [np.ones(4)])
        monkeypatch.setattr(
            kernel_worker, "_extract_third_party", lambda results, priority: results[priority]["outputs"]
        )
        monkeypatch.setattr(
            kernel_worker, "process_context", lambda: SimpleNamespace(notify_status=lambda *a, **k: None)
        )
        monkeypatch.setattr(
            kernel_worker,
            "get_global_storage",
            lambda: SimpleNamespace(proc_timeout=10, xpu_perf=False, dump_config=SimpleNamespace(dump_xpu=False)),
        )

        xpu = kernel_worker._dispatch_xpu(case, need_3party=True, golden_is_xpu=False, manual_case=None)

        assert xpu.pending  # background 形态
        # 收割上界随 deadline 预算联动（proc_timeout=10 → budget=310）
        assert xpu._join_timeout == xpu_join_timeout(310.0)
        order.append("npu")  # 模拟 NPU 执行段
        for _ in range(200):
            if "xpu-start" in order:
                break
            time.sleep(0.01)
        third_parties, xpu_results, provider = xpu.collect()

        assert order.index("xpu-start") < order.index("npu")  # dispatch 先于 NPU
        assert provider == "remote"
        assert xpu_results["remote"]["status"] == "PASS"
        np.testing.assert_array_equal(third_parties[0], np.arange(4.0))

    def test_golden_xpu_stays_synchronous(self, monkeypatch):
        """golden-XPU 恒同步（设备前收割契约）——ready 句柄；golden 自身即 DATA 判据
        （need_3party=False、未配 xpu_perf/dump_xpu 时仍须派发，mode 含 DATA 位）。"""
        from ttk.core_modules.kernel import worker as kernel_worker
        from ttk.remote import DATA

        case = SimpleNamespace(op_name="Add", testcase_name="gx_case", flat_manual_xpu_binaries=None)
        recorder = []

        def _fake_do_xpu_profiling(context, mode, deadline_s=None, inputs=None):
            recorder.append(mode)
            context.xpu_results = {"r": {"status": "PASS", "api": "x", "outputs": [[np.arange(4.0)]]}}
            return "r"

        monkeypatch.setattr(kernel_worker, "_do_xpu_profiling", _fake_do_xpu_profiling)
        monkeypatch.setattr(kernel_worker, "_xpu_inputs", lambda ctx: [])
        monkeypatch.setattr(
            kernel_worker, "_extract_third_party", lambda results, priority: results[priority]["outputs"]
        )
        monkeypatch.setattr(
            kernel_worker, "process_context", lambda: SimpleNamespace(notify_status=lambda *a, **k: None)
        )
        monkeypatch.setattr(
            kernel_worker,
            "get_global_storage",
            lambda: SimpleNamespace(xpu_perf=False, dump_config=SimpleNamespace(dump_xpu=False)),
        )

        xpu = kernel_worker._dispatch_xpu(case, need_3party=False, golden_is_xpu=True, manual_case=None)

        assert not xpu.pending  # ready——dispatch 点即完成
        assert recorder == [DATA]  # 同步已执行，且 DATA 位来自 need_data（含 golden_is_xpu）
        third_parties, _results, _provider = xpu.collect()
        np.testing.assert_array_equal(third_parties[0], np.arange(4.0))


class TestLoadManualThirdParties:
    """装载器：未声明→None；槽位对齐；缺位补 None；shape 校验。"""

    def test_undeclared_returns_none(self):
        assert load_manual_third_parties(_manual_xpu_case(None)) is None
        assert load_manual_third_parties(_manual_xpu_case(())) is None

    def test_loads_by_declared_shape_and_dtype(self, tmp_path):
        path = tmp_path / "gpu.bin"
        expected = np.arange(8, dtype=np.float32)
        expected.tofile(path)

        arrays = load_manual_third_parties(_manual_xpu_case((str(path),)))

        assert len(arrays) == 1
        np.testing.assert_array_equal(arrays[0], expected)
        assert arrays[0].dtype == np.float32

    def test_fewer_paths_pad_none_for_uncovered_slots(self, tmp_path):
        path = tmp_path / "gpu.bin"
        np.arange(8, dtype=np.float32).tofile(path)

        arrays = load_manual_third_parties(
            _manual_xpu_case((str(path),), shapes=((8,), (8,)), dtypes=("float32", "float32"))
        )

        assert len(arrays) == 2
        assert arrays[1] is None

    def test_dynamic_shape_slot_pads_none(self, tmp_path):
        path = tmp_path / "gpu.bin"
        np.arange(8, dtype=np.float32).tofile(path)

        arrays = load_manual_third_parties(_manual_xpu_case((str(path), None), shapes=((8,), None)))

        assert arrays[1] is None
        np.testing.assert_array_equal(arrays[0], np.arange(8, dtype=np.float32))

    def test_shape_mismatch_raises(self, tmp_path):
        path = tmp_path / "gpu.bin"
        np.arange(7, dtype=np.float32).tofile(path)

        with pytest.raises(ManualXpuLoadError, match="does not match"):
            load_manual_third_parties(_manual_xpu_case((str(path),)))

    def test_complex32_loads_as_physical_storage_shape(self, tmp_path):
        """complex32 逻辑 (2,3) 声明 → 装载为物理 (2,3,2) fp16（实/虚尾维），不误报 shape 不符。

        远端三方 complex32 输出同为 float16 物理视图——两侧口径一致。
        """
        path = tmp_path / "gpu.bin"
        np.arange(12, dtype=np.float16).tofile(path)

        arrays = load_manual_third_parties(_manual_xpu_case((str(path),), shapes=((2, 3),), dtypes=("complex32",)))

        assert arrays[0].shape == (2, 3, 2)
        assert arrays[0].dtype == np.float16

    def test_wrong_element_count_still_rejected_for_special_dtypes(self, tmp_path):
        path = tmp_path / "gpu.bin"
        np.arange(7, dtype=np.float16).tofile(path)

        with pytest.raises(ManualXpuLoadError, match="does not match"):
            load_manual_third_parties(_manual_xpu_case((str(path),), shapes=((2, 3),), dtypes=("complex32",)))

    def test_kernel_shape_accessor_preferred_over_view_shapes(self, tmp_path):
        """kernel 语义 flat_output_shapes 优先；缺失时回退 aclnn/e2e 的 flat_output_view_shapes。"""
        path = tmp_path / "gpu.bin"
        np.arange(8, dtype=np.float32).tofile(path)

        case = SimpleNamespace(
            testcase_name="t",
            flat_manual_xpu_binaries=(str(path),),
            flat_output_shapes=None,  # kernel 槽存在但为 None（无声明输出）→ 回退 view
            flat_output_view_shapes=((8,),),
            flat_output_dtypes=("float32",),
        )
        arrays = load_manual_third_parties(case)
        np.testing.assert_array_equal(arrays[0], np.arange(8, dtype=np.float32))


class TestDumpManualXpuArchive:
    """归档增强：--dump xpu 开启时 manual 产物按回放协议落盘（与远端产物同命名）。"""

    def test_results_entry_shape_matches_remote_contract(self):
        assert manual_xpu_results() == {MANUAL_XPU_PROVIDER: {"status": "PASS", "api": "manual_xpu_binaries"}}

    @staticmethod
    def _switches(dump_xpu):
        import ttk.switches as switches_mod

        return SimpleNamespace(dump_config=SimpleNamespace(dump_xpu=dump_xpu)), switches_mod

    def test_dump_disabled_writes_nothing(self, tmp_path, monkeypatch):
        sw, switches_mod = self._switches(False)
        monkeypatch.setattr(switches_mod, "get_global_storage", lambda: sw)
        monkeypatch.setenv("NPU_DUMP_PATH", str(tmp_path))

        dump_manual_xpu_archive(_manual_xpu_case(None), [np.arange(8, dtype=np.float32)])

        assert list(tmp_path.iterdir()) == []

    def test_dump_enabled_writes_replay_contract_filenames(self, tmp_path, monkeypatch):
        sw, switches_mod = self._switches(True)
        monkeypatch.setattr(switches_mod, "get_global_storage", lambda: sw)
        monkeypatch.setenv("NPU_DUMP_PATH", str(tmp_path))

        arrays = [np.arange(8, dtype=np.float32), None]
        dump_manual_xpu_archive(_manual_xpu_case(None), arrays)

        files = sorted(p.name for p in tmp_path.iterdir())
        assert files == ["manual_xpu_case_xpu_golden_0.bin"]  # None 槽跳过；命名即回放契约
