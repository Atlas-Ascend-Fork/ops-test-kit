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

覆盖：dl=1 确定性状态判定（_det_status）、execute_graph 轮数编排（--run 直达
serve 轮数，与 kernel/aclnn 同口径）、_collect_round_perf 时间戳聚合（最早目
录剔除/捕获不足保护/差集隔离历史目录）、逐模式结果合并（_merge_mode_result
/ _absorb_fields 的 cst_/dyn_ 槽映射与吸收语义）。
"""

import logging
import types
from contextlib import ExitStack
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
        from ttk.core_modules.common.comparison.batch_consistency import BatchSummary, SampleHash

        samples = [SampleHash(seeds=(seed,), lengths=(1,), digest=digest) for seed, digest in hashes]
        return BatchSummary(testcase_name=tc_name, batch_consistency_id=bcid, sample_hashes=samples)

    @staticmethod
    def _case(bcid, batch_axis, dl=3, name="tc"):
        return types.SimpleNamespace(
            testcase_name=name,
            op_name="Add",
            batch_consistency_id=bcid,
            batch_axis=batch_axis,
            batch_slice_info=([[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]],),
            batch_seed=([[7, 8, 9, 10]],),
            _dl=dl,
        )

    def test_pack_result_carries_batch_summary_slot(self):
        from ttk.core_modules.geir.execute import GraphExecResult

        tc = types.SimpleNamespace(batch_consistency_id="s1")
        result = geir_worker._pack_result("const", GraphExecResult(), ("1.0", "log", True, {}), None, tc)
        assert result.batch_summary is None  # 槽位默认 None（非 CSV 列）

    def test_assemble_summary_slices_output_zero(self):
        from ttk.core_modules.common.comparison.batch_consistency import assemble_batch_summary

        tc = self._case("s1|axis0", ((0,),))
        outputs = [np.arange(12, dtype=np.float32).reshape(4, 3)]
        summary = assemble_batch_summary(outputs, tc, tc.batch_consistency_id)
        assert summary is not None
        assert len(summary.sample_hashes) == 4  # 4 个声明样本各 1 行

    def test_instance_collects_and_compares(self, caplog):
        from ttk.core_modules.geir import GeirInstance

        ins = GeirInstance()
        ins.case_result_title = GeirReturnStructure.get_titles()

        # 两个用例声明同 seed 样本：片段哈希一致 → PASS
        # （收集已上提 InstanceBase._collect_batch_summary——主循环在
        # handle_result 前统一调用）
        for name, hashes in (("tc_a", [(7, "h1"), (8, "h2")]), ("tc_b", [(8, "h2"), (7, "h1")])):
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
        for name, hashes in (("tc_a", [(7, "h1"), (8, "h2")]), ("tc_b", [(8, "h2"), (7, "XX")])):
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


class TestExecuteGraphRunCount:
    """execute_graph 轮数编排：--run 即执行轮数（与 kernel/aclnn 同口径），perf 走聚合。"""

    @staticmethod
    def _switches(run_time, warmup=False, dl=0):
        return types.SimpleNamespace(run_time=run_time, deterministic_level=dl, warmup=warmup)

    @staticmethod
    def _case():
        arr = np.ones(2, dtype=np.float32)
        return types.SimpleNamespace(
            testcase_name="tc", input_arrays=[arr], golden_arrays=[arr.copy()], original_input_arrays=None
        )

    @staticmethod
    def _plan(tmp_path, prof=True):
        return geir_execute.GraphExecPlan(
            binary="bin",
            config_path="cfg",
            build_dir=str(tmp_path),
            prof_path=str(tmp_path / "prof") if prof else "",
            input_prefix=str(tmp_path / "in"),
        )

    def _execute(self, tmp_path, switches, perf_value=None, prof=True):
        """mock serve 单轮/输入落盘/轮间清态/整批 perf 聚合，仅保留轮编排逻辑。"""
        plan = self._plan(tmp_path, prof=prof)
        testcase = self._case()
        state = {"rounds": 0, "rounds_seq": [], "collect": None}

        def _fake_run_once_serve(tc, dev_id, sw, pl):
            state["rounds"] += 1
            state["rounds_seq"].append((pl.round_idx, pl.run_total))
            return [np.ones(2, dtype=np.float32)]

        def _fake_collect(pl, before, case_name):
            state["collect"] = (pl.run_total, case_name)
            return perf_value

        with ExitStack() as stack:
            write = stack.enter_context(mock.patch.object(geir_execute, "_write_all_input_bins"))
            stack.enter_context(mock.patch.object(geir_execute, "clear_device_state"))
            stack.enter_context(mock.patch.object(geir_execute, "_cleanup_bins"))
            stack.enter_context(mock.patch.object(geir_execute, "_run_once_serve", side_effect=_fake_run_once_serve))
            stack.enter_context(mock.patch.object(geir_execute, "_collect_round_perf", side_effect=_fake_collect))
            result = geir_execute.execute_graph(testcase, 0, switches, plan)
        return result, state["rounds"], state["rounds_seq"], write, state["collect"]

    @pytest.mark.parametrize(
        ("run_count", "warmup", "expected_seq"),
        [
            (1, False, [(0, 1)]),
            (2, False, [(0, 2), (1, 2)]),
            (3, False, [(0, 3), (1, 3), (2, 3)]),
            (1, True, [(-1, 1), (0, 1)]),
            (3, True, [(-1, 3), (0, 3), (1, 3), (2, 3)]),
        ],
    )
    def test_round_schedule_and_input_rewrite(self, tmp_path, run_count, warmup, expected_seq):
        """轮次序列 + 输入重写：预写 1 次、除首条目外逐轮重写，合计 = 执行轮数。"""
        result, rounds, rounds_seq, write, collect = self._execute(tmp_path, self._switches(run_count, warmup=warmup))
        assert rounds_seq == expected_seq
        assert rounds == len(expected_seq)
        assert write.call_count == rounds
        assert collect == (run_count, "tc")

    def test_perf_passthrough(self, tmp_path):
        """聚合返回值直通 device_perf_us：有值透传、None 保持、prof 关为 None。"""
        result, *_ = self._execute(tmp_path, self._switches(2), perf_value=25.0)
        assert result.device_perf_us == 25.0
        result, *_ = self._execute(tmp_path, self._switches(2), perf_value=None)
        assert result.device_perf_us is None
        result, *_ = self._execute(tmp_path, self._switches(2), prof=False)
        assert result.device_perf_us is None

    def test_dl1_md5_samples_exclude_warmup(self, tmp_path):
        """dl=1：MD5 样本数 = --run N（预热轮不进样本），全一致 → PASS。"""
        det_hash = mock.Mock(side_effect=lambda *args: "md5")
        with mock.patch.object(geir_execute, "_det_round_hash", det_hash):
            result, rounds, _, _, _ = self._execute(tmp_path, self._switches(2, dl=1, warmup=True))
        assert rounds == 3  # 预热轮 + 2 运行轮
        assert det_hash.call_count == 2
        assert result.det_status == "PASS"


class TestCollectRoundProf:
    """_collect_round_perf：单会话聚合——本用例新目录全量入样，历史目录差集隔离。"""

    @staticmethod
    def _plan(prof_root):
        return geir_execute.GraphExecPlan(
            binary="bin", config_path="cfg", build_dir=".", prof_path=str(prof_root), input_prefix="prefix"
        )

    @staticmethod
    def _mk_prof(root, seq):
        path = root / f"PROF_{seq:06d}_20261027052300000_0000000TESTCASE"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _collect(self, tmp_path, new_count=0, old_count=0, value=None, run_count=3):
        """创建 old_count 个历史目录 + new_count 个本轮目录；返回 (聚合值, 解析入参列表, 窗口轮数)。"""
        root = tmp_path / "prof"
        root.mkdir()
        for i in range(old_count):
            self._mk_prof(root, i + 1)
        before = geir_execute._snapshot_prof_entries(str(root))
        for i in range(new_count):
            self._mk_prof(root, old_count + i + 1)
        parsed = []
        rounds_seen = []

        def _fake_parse(dirs, rounds):
            parsed.append(list(dirs))
            rounds_seen.append(rounds)
            return value

        plan = self._plan(root)
        plan.run_total = run_count
        with mock.patch.object(geir_execute, "_parse_msprof_task_duration", side_effect=_fake_parse):
            result = geir_execute._collect_round_perf(plan, before, "tc")
        return result, parsed, rounds_seen

    def test_single_session_dir_parsed(self, tmp_path):
        """单会话目录：整目录入样，窗口轮数 = run-1（kernel start_step=1 口径）。"""
        result, parsed, rounds = self._collect(tmp_path, new_count=1, value=12.5)
        assert result == 12.5
        assert len(parsed) == 1
        assert len(parsed[0]) == 1
        assert rounds == [2]

    def test_run1_window_is_single_round(self, tmp_path):
        """run=1：唯一一轮入窗（窗口轮数 = 1）。"""
        result, _, rounds = self._collect(tmp_path, new_count=1, value=10.0, run_count=1)
        assert result == 10.0
        assert rounds == [1]

    def test_multi_dirs_all_included_with_warning(self, tmp_path, caplog):
        """异常多目录（残留/重复 init）：全量入样防丢并告警。"""
        with caplog.at_level(logging.WARNING):
            result, parsed, _ = self._collect(tmp_path, new_count=2, value=20.0)
        assert result == 20.0
        assert len(parsed[0]) == 2
        assert any("session dirs captured" in r.getMessage() for r in caplog.records)

    def test_old_dirs_excluded_by_diff(self, tmp_path):
        """历史 PROF 不入差集：仅本轮新建目录进解析（旧目录混入会使入参超 1）。"""
        result, parsed, _ = self._collect(tmp_path, new_count=1, old_count=2, value=15.0)
        assert result == 15.0
        assert len(parsed) == 1
        assert len(parsed[0]) == 1

    def test_no_new_dirs_returns_none(self, tmp_path):
        """无新目录（prof 会话未开启/首轮不采集）：聚合返回 None。"""
        result, parsed, rounds = self._collect(tmp_path, new_count=0)
        assert result is None
        assert parsed == []
        assert rounds == []

    def test_parse_none_returns_none(self, tmp_path, caplog):
        """目录在但无 task 数据：返回 None 并告警。"""
        with caplog.at_level(logging.WARNING):
            result, _, _ = self._collect(tmp_path, new_count=1, value=None)
        assert result is None
        assert any("no task duration" in r.getMessage() for r in caplog.records)

    def test_empty_prof_path_returns_none(self, tmp_path):
        """task-prof 关闭（prof_path 空）时不采集。"""
        plan = geir_execute.GraphExecPlan(
            binary="bin", config_path="cfg", build_dir=".", prof_path="", input_prefix="prefix"
        )
        assert geir_execute._collect_round_perf(plan, frozenset(), "tc") is None


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
