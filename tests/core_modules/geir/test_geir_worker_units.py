# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GEIR worker/execute 单元测试。

覆盖：dl=1 确定性状态判定（_det_status）、逐模式结果合并（_merge_mode_result
/ _absorb_fields 的 cst_/dyn_ 槽映射与吸收语义）。
"""

import logging
import types
from unittest import mock

import numpy as np
import pytest

import ttk.core_modules.geir.execute as geir_execute
import ttk.core_modules.geir.worker as geir_worker
from ttk.core_modules.common.testcase.tensor import Tensor
from ttk.core_modules.geir.graph_builder import _core_limit_options
from ttk.core_modules.geir.result import GeirReturnStructure
from ttk.core_modules.geir.worker import _absorb_fields, _merge_mode_result


class TestCoreLimitOptions:
    """--core-limit → GE Option（ge.aicoreNum / ge.vectorcoreNum）。"""

    def test_none_gives_empty_entries(self):
        opts = _core_limit_options(None)
        assert opts == {"ge.aicoreNum": "", "ge.vectorcoreNum": ""}

    def test_ai_only(self):
        opts = _core_limit_options((2, None))
        assert opts == {"ge.aicoreNum": "2", "ge.vectorcoreNum": ""}

    def test_ai_and_vector(self):
        opts = _core_limit_options((2, 32))
        assert opts == {"ge.aicoreNum": "2", "ge.vectorcoreNum": "32"}


class TestKeeperSnapshotInjection:
    """prepare_workers 建池前注入 OpInfoKeeper 快照（主进程一次，worker 直接用）。"""

    def test_prepare_workers_injects_snapshot(self):
        from ttk.core_modules.common.process import SimpleCommandProcess
        from ttk.core_modules.geir import GeirInstance

        ins = GeirInstance()
        captured = {}
        with mock.patch(
            "ttk.core_modules.kernel.info_keeper.snapshot_keeper", return_value={"ops_info": {"Add": {}}}
        ), mock.patch.object(type(ins).__mro__[1], "prepare_workers", return_value=None) as super_pw:
            ins.prepare_workers()
            captured["super_called"] = super_pw.called
        assert captured["super_called"] is True
        assert SimpleCommandProcess._bootstrap_keeper_state == {
            "ttk.core_modules.kernel.info_keeper": {"ops_info": {"Add": {}}}
        }
        SimpleCommandProcess._bootstrap_keeper_state = None  # 清理全局态


class TestBatchConsistency:
    """dl=3 batch 一致性——BatchSummary 回传 + cleanup 组装比对。"""

    @staticmethod
    def _summary(tc_name, bcid, hashes):
        from ttk.core_modules.common.comparison.batch_consistency import BatchSummary

        return BatchSummary(testcase_name=tc_name, batch_consistency_id=bcid, sample_hashes=hashes)

    @staticmethod
    def _case(bcid, batch_axis, dl=3, name="tc"):
        return types.SimpleNamespace(
            testcase_name=name,
            op_name="Add",
            batch_consistency_id=bcid,
            batch_axis=batch_axis,
            batch_slice_info=(0, 1),  # 元数据在位即可（切片沿 axis 0 全样本）
            _dl=dl,
        )

    def test_pack_result_carries_batch_summary_slot(self):
        from ttk.core_modules.geir.execute import GraphExecResult

        tc = types.SimpleNamespace(batch_consistency_id="s1")
        result = geir_worker._pack_result("const", GraphExecResult(), ("1.0", "log", True, {}), None, tc)
        assert result.batch_summary is None  # 槽位默认 None（非 CSV 列）

    def test_assemble_summary_slices_output_zero(self):
        from ttk.core_modules.common.comparison.batch_consistency import assemble_batch_summary

        tc = self._case("s1|axis0", 0)
        outputs = [np.arange(12, dtype=np.float32).reshape(4, 3)]
        summary = assemble_batch_summary(outputs, tc, tc.batch_consistency_id)
        assert summary is not None
        assert len(summary.sample_hashes) == 4  # axis=0 切 4 样本

    def test_instance_collects_and_compares(self, caplog):
        from ttk.core_modules.geir import GeirInstance

        ins = GeirInstance()
        ins.case_result_title = GeirReturnStructure.get_titles()

        # 两个同 id 用例：样本哈希一致 → PASS
        # （收集已上提 InstanceBase._collect_batch_summary——主循环在
        # handle_result 前统一调用）
        for name, hashes in (("tc_a", ["h1", "h2"]), ("tc_b", ["h1", "h2"])):
            result = GeirReturnStructure()
            result.batch_summary = self._summary(name, "s1", hashes)
            task = types.SimpleNamespace(testcase=types.SimpleNamespace(testcase_name=name))
            ins._collect_batch_summary(task, result)
            ins.handle_result(task, result)

        with caplog.at_level(logging.ERROR):
            ins.cleanup()
        assert ins._batch_summaries
        assert not [r for r in caplog.records if "FAIL" in r.getMessage()]

        # 改成不一致 → FAIL 日志
        ins2 = GeirInstance()
        ins2.case_result_title = GeirReturnStructure.get_titles()
        for name, hashes in (("tc_a", ["h1", "h2"]), ("tc_b", ["h1", "XX"])):
            result = GeirReturnStructure()
            result.batch_summary = self._summary(name, "s1", hashes)
            task = types.SimpleNamespace(testcase=types.SimpleNamespace(testcase_name=name))
            ins2._collect_batch_summary(task, result)
            ins2.handle_result(task, result)
        with mock.patch.object(type(ins2), "close_subprocesses", return_value=None), mock.patch.object(
            type(ins2), "teardown", return_value=None
        ), caplog.at_level(logging.ERROR):
            ins2.cleanup()
        assert any("batch consistency FAIL" in r.getMessage() for r in caplog.records)


def _result(**kwargs) -> GeirReturnStructure:
    result = GeirReturnStructure()
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


class TestDetStatus:
    """dl=1 轮间一致性判定（execute._det_status）。"""

    def test_zero_rounds_returns_none(self):
        # run_time=0 → md5_list 空 → None（非 PASS），与原实现逐分支一致
        assert geir_execute._det_status([], True, "tc") is None

    def test_not_deterministic_returns_none(self):
        assert geir_execute._det_status(["a"], False, "tc") is None

    def test_single_round_pass(self):
        assert geir_execute._det_status(["a"], True, "tc") == "PASS"

    def test_consistent_pass(self):
        assert geir_execute._det_status(["a", "a", "a"], True, "tc") == "PASS"

    def test_mismatch_fail(self):
        assert geir_execute._det_status(["a", "b"], True, "tc") == "FAIL"


class TestMergeModeResult:
    """模式展开结果合并（worker._merge_mode_result / _absorb_fields）。"""

    def test_const_mode_maps_cst_slot(self):
        mode_result = _result(precision="99%")
        merged = _merge_mode_result(None, mode_result, "const")
        assert merged is mode_result
        assert mode_result.cst_precision == "99%"
        assert mode_result.dyn_precision == ""

    def test_dynamic_binary_mode_maps_dyn_slot(self):
        mode_result = _result(precision="98%")
        merged = _merge_mode_result(None, mode_result, "dynamic_binary")
        assert merged is mode_result
        assert mode_result.dyn_precision == "98%"

    def test_absorb_overwrites_only_populated_fields(self):
        base = _merge_mode_result(None, _result(precision="FAIL", passed=False), "const")
        later = _result(precision="97%", cst_precision="97%", cst_perf_us=5.0)
        later.precision_metrics = {"cst": {"a": 1}}
        later.passed = True
        later.precision_status = "PASS"
        _absorb_fields(base, later)
        assert base.cst_precision == "97%"
        assert base.cst_perf_us == 5.0
        assert base.precision_metrics == {"cst": {"a": 1}}
        assert base.passed is True
        assert base.precision_status == "PASS"
        assert base.precision == "97%"
        assert base.dyn_precision == ""  # 未触及字段不覆盖

    def test_absorb_keeps_first_pass_semantics(self):
        base = _result(precision="99%", passed=True, precision_status="PASS")
        later = _result(precision="50%", passed=False)
        _absorb_fields(base, later)
        # 已通过的累计结果不被未通过模式覆盖
        assert base.passed is True
        assert base.precision == "99%"
        assert base.precision_status == "PASS"


class TestDetRoundHash:
    """dl=1 逐轮规范化哈希（双消费点之二：pre_compare 双半边，hash 只用 outputs 半边）。"""

    def test_no_hook_hashes_raw_outputs(self):
        outputs = [np.array([3.0, 1.0, 2.0])]
        goldens = [np.array([9.9])]
        tc = types.SimpleNamespace(testcase_name="tc")
        assert geir_execute._det_round_hash(tc, outputs, goldens, None) == geir_execute._md5_of(outputs)

    def test_hook_normalizes_outputs_half_only(self):
        tc = types.SimpleNamespace(testcase_name="tc")
        outputs = [np.array([3.0, 1.0, 2.0])]
        goldens = [np.array([9.9])]

        # 规范化钩子：输出排序（模拟索引序等非确定表示）；golden 半边翻转（应被弃用）
        def _sort_out_and_flip_golden(output, golden):
            return [np.sort(output)], [golden[::-1]]

        h1 = geir_execute._det_round_hash(tc, outputs, goldens, _sort_out_and_flip_golden)
        h2 = geir_execute._det_round_hash(tc, outputs, goldens, _sort_out_and_flip_golden)
        assert h1 == h2  # 同输入经规范化后轮间稳定
        assert h1 != geir_execute._md5_of(outputs)  # 规范化生效（raw 序不同）
        assert h1 == geir_execute._md5_of([np.sort(outputs[0])])  # hash = 规范化后的 outputs 半边
        assert list(outputs[0]) == [3.0, 1.0, 2.0]  # 浅拷贝隔离：原 outputs 保持 raw（比对侧自行规范化）


class TestReconcileInputs:
    """输入段后对账：结构 fail-fast + 字节 advisory + 基础设施容错。"""

    @staticmethod
    def _case(slots, input_arrays):
        return types.SimpleNamespace(
            testcase_name="tc", op_name="Add", input_tensor_slots=slots, input_arrays=input_arrays
        )

    def test_match_passes_silently(self, caplog):
        slots = (Tensor(shape=(2, 3), dtype="float32"),)
        arrays = [np.zeros((2, 3), dtype=np.float32)]
        with caplog.at_level(logging.WARNING):
            geir_worker._reconcile_inputs(self._case(slots, arrays))
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_none_misplacement_fails_fast(self):
        slots = (None, Tensor(shape=(2,), dtype="float32"))
        arrays = [np.zeros(2, dtype=np.float32), None]  # None 错位（optional 槽位拿到数据）
        with pytest.raises(RuntimeError, match="structure reconciliation failed"):
            geir_worker._reconcile_inputs(self._case(slots, arrays))

    def test_tensorlist_nesting_mismatch_fails_fast(self):
        slots = ((Tensor(shape=(2,), dtype="float32"), Tensor(shape=(2,), dtype="float32")),)
        arrays = [np.zeros(2, dtype=np.float32)]  # 应为嵌套序列
        with pytest.raises(RuntimeError, match="structure reconciliation failed"):
            geir_worker._reconcile_inputs(self._case(slots, arrays))

    def test_byte_drift_warns_without_raising(self, caplog):
        slots = (Tensor(shape=(2, 3), dtype="float32"),)
        arrays = [np.zeros((2, 4), dtype=np.float32)]  # shape 漂移（结构层面仍匹配）
        with caplog.at_level(logging.WARNING):
            geir_worker._reconcile_inputs(self._case(slots, arrays))
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warns
        assert "byte drift" in warns[0].getMessage()

    def test_dynamic_dim_decl_skipped_in_byte_check(self, caplog):
        slots = (Tensor(shape=(-1, 3), dtype="float32"),)
        arrays = [np.zeros((5, 3), dtype=np.float32)]  # 非具体维声明 → 字节对账跳过
        with caplog.at_level(logging.WARNING):
            geir_worker._reconcile_inputs(self._case(slots, arrays))
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_empty_slots_noop(self):
        geir_worker._reconcile_inputs(self._case((), []))

    def test_materialization_failure_skips_reconcile(self):
        class _Broken:
            testcase_name = "tc"
            op_name = "Add"
            input_arrays = []

            @property
            def input_tensor_slots(self):
                raise RuntimeError("boom")

        geir_worker._reconcile_inputs(_Broken())  # 不抛——对账基础设施不得破坏通路
