# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""fusion_result.json 采集（解析 + serve 快照 diff）、结果列、按二进制签名调度排序。"""

import json
from types import SimpleNamespace

import pytest

from ttk.core_modules.geir.execute import _GeirServe
from ttk.core_modules.geir.fusion_parse import (
    FUSION_STAT_FILENAME,
    format_pass_stats,
    parse_fusion_sessions,
)
from ttk.core_modules.geir.result import GeirReturnStructure
from ttk.core_modules.geir.testcase import GeirTestcase


def _write_sessions(path, sessions):
    """按 fusion_statistic_writer.cc 的 schema 落盘：会话层 + 分组层 + 复数键 + 字符串计数。"""
    path.write_text(
        json.dumps(
            {
                session: {
                    "graph_fusion": {
                        name: {"match_times": str(hit), "effect_times": str(eff)} for name, (hit, eff) in passes.items()
                    }
                }
                for session, passes in sessions.items()
            }
        ),
        encoding="utf-8",
    )


class TestFusionParse:
    def test_missing_file_returns_none(self, tmp_path):
        assert parse_fusion_sessions(str(tmp_path / "nope.json")) is None
        assert parse_fusion_sessions("") is None

    def test_bad_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        assert parse_fusion_sessions(str(p)) is None

    def test_non_dict_root_returns_none(self, tmp_path):
        p = tmp_path / FUSION_STAT_FILENAME
        p.write_text(json.dumps(["P1", "P2"]), encoding="utf-8")
        assert parse_fusion_sessions(str(p)) is None

    def test_real_schema_parsed_per_session(self, tmp_path):
        p = tmp_path / FUSION_STAT_FILENAME
        _write_sessions(
            p,
            {
                "session_and_graph_id_0_0": {"PassA": (1, 1)},
                "session_and_graph_id_1_0": {"PassA": (2, 0), "PassB": (1, 1)},
            },
        )
        sessions = parse_fusion_sessions(str(p))
        assert set(sessions) == {"session_and_graph_id_0_0", "session_and_graph_id_1_0"}
        assert sessions["session_and_graph_id_0_0"] == {"PassA": (1, 1)}
        assert sessions["session_and_graph_id_1_0"] == {"PassA": (2, 0), "PassB": (1, 1)}

    def test_int_encoded_counts_accepted(self, tmp_path):
        p = tmp_path / FUSION_STAT_FILENAME
        p.write_text(
            json.dumps({"s0": {"graph_fusion": {"PassA": {"match_times": 3, "effect_times": 2}}}}),
            encoding="utf-8",
        )
        assert parse_fusion_sessions(str(p))["s0"] == {"PassA": (3, 2)}

    def test_ub_fusion_group_ignored(self, tmp_path):
        """ub_fusion（UB/buffer 融合）属另一条管线，不进结果列。"""
        p = tmp_path / FUSION_STAT_FILENAME
        p.write_text(
            json.dumps(
                {
                    "s0": {
                        "graph_fusion": {"GraphPass": {"match_times": "1", "effect_times": "1"}},
                        "ub_fusion": {"UbPass": {"match_times": "9", "effect_times": "9"}},
                    }
                }
            ),
            encoding="utf-8",
        )
        assert parse_fusion_sessions(str(p))["s0"] == {"GraphPass": (1, 1)}

    def test_format_multiple_passes_ordered(self):
        assert format_pass_stats({"PassA": (1, 1), "PassB": (2, 0)}) == "PassA:hit=1,effect=1;PassB:hit=2,effect=0"

    def test_format_missing_fields_stay_none(self):
        assert format_pass_stats({"P": (None, 1)}) == "P:hit=None,effect=1"

    def test_format_empty_returns_none(self):
        assert format_pass_stats({}) is None

    def test_stat_filename_convention(self):
        assert FUSION_STAT_FILENAME == "fusion_result.json"


class TestServeFusionStats:
    """collect_fusion_stats：快照态绑定 serve 实例生命周期。"""

    @staticmethod
    def _fake_serve(tmp_path, content):
        serve = SimpleNamespace(
            serve_dir=str(tmp_path),
            alive=lambda: True,
        )
        serve._seen_fusion_sessions = set()
        serve.collect_fusion_stats = lambda: _GeirServe.collect_fusion_stats(serve)
        if content is not None:
            (tmp_path / FUSION_STAT_FILENAME).write_text(json.dumps(content), encoding="utf-8")
        return serve

    def test_returns_new_sessions_only(self, tmp_path, monkeypatch):
        serve = self._fake_serve(
            tmp_path,
            {"s0": {"graph_fusion": {"PassA": {"match_times": "1", "effect_times": "1"}}}},
        )
        assert serve.collect_fusion_stats() == "PassA:hit=1,effect=1"
        assert serve.collect_fusion_stats() is None  # 无新增会话
        (tmp_path / FUSION_STAT_FILENAME).write_text(
            json.dumps(
                {
                    "s0": {"graph_fusion": {"PassA": {"match_times": "1", "effect_times": "1"}}},
                    "s1": {"graph_fusion": {"PassB": {"match_times": "2", "effect_times": "1"}}},
                }
            ),
            encoding="utf-8",
        )
        assert serve.collect_fusion_stats() == "PassB:hit=2,effect=1"

    def test_multiple_new_sessions_takes_last(self, tmp_path, monkeypatch):
        """同用例多轮（多新会话）各轮图相同，取末轮会话计数，不跨会话累加。"""
        serve = self._fake_serve(
            tmp_path,
            {
                "s0": {"graph_fusion": {"PassA": {"match_times": "1", "effect_times": "1"}}},
                "s1": {"graph_fusion": {"PassA": {"match_times": "1", "effect_times": "1"}}},
                "s2": {"graph_fusion": {"PassA": {"match_times": "2", "effect_times": "1"}}},
            },
        )
        assert serve.collect_fusion_stats() == "PassA:hit=2,effect=1"

    def test_missing_file_returns_none(self, tmp_path):
        serve = self._fake_serve(tmp_path, None)
        assert serve.collect_fusion_stats() is None

    def test_empty_file_returns_none(self, tmp_path):
        serve = self._fake_serve(tmp_path, None)
        (tmp_path / FUSION_STAT_FILENAME).write_text("{}", encoding="utf-8")
        assert serve.collect_fusion_stats() is None

    def test_dead_serve_returns_none(self, tmp_path):
        serve = self._fake_serve(tmp_path, None)
        serve.alive = lambda: False
        assert serve.collect_fusion_stats() is None


class TestResultColumns:
    def test_titles_include_fusion_column(self):
        titles = GeirReturnStructure.get_titles()
        assert "fusion_pass_details" in titles
        assert "fusion_passes" not in titles

    def test_pick_data_na_when_no_mode_ran(self):
        r = GeirReturnStructure()
        data = r.pick_data(GeirReturnStructure.get_titles())
        assert data[GeirReturnStructure.get_titles().index("fusion_pass_details")] == "NA"

    def test_pick_data_assembles_per_mode_segments(self):
        """单列按模式组装：cst:xxx,dyn:xxx；两模式融合触发可不同，未执行模式不出段。"""
        r = GeirReturnStructure()
        # -c -d 双模式：静态无命中，动态命中
        r.cst_fusion_raw = "NA"
        r.dyn_fusion_raw = "MulAddNPass:hit=1,effect=1"  # noqa: S105  # pass 计数列值，非口令
        data = r.pick_data(GeirReturnStructure.get_titles())
        assert data[GeirReturnStructure.get_titles().index("fusion_pass_details")] == (
            "cst:NA,dyn:MulAddNPass:hit=1,effect=1"
        )
        # 仅 -c：动态未执行不出段
        r2 = GeirReturnStructure()
        r2.cst_fusion_raw = "MulAddNPass:hit=1,effect=1"  # noqa: S105
        data2 = r2.pick_data(GeirReturnStructure.get_titles())
        assert data2[GeirReturnStructure.get_titles().index("fusion_pass_details")] == "cst:MulAddNPass:hit=1,effect=1"

    def test_supported_rerun_title_excludes_fusion(self):
        """fusion_pass_details 取值为计数串或 NA，不作为 rerun 目标列。"""
        assert "fusion_pass_details" not in GeirTestcase.supported_rerun_title()
        assert "fusion_passes" not in GeirTestcase.supported_rerun_title()


class TestBinarySignatureSort:
    def test_signature_key(self):
        from ttk.core_modules.geir import GeirInstance

        graph_tc = SimpleNamespace(op_name="anchor", _resolved_graph={"ops": ("mul", "add_n")})
        op_tc = SimpleNamespace(op_name="add", _resolved_graph=None)
        assert GeirInstance._binary_signature_key(graph_tc) == "graph__3_mul5_add_n"
        # 长图签名截断 + 摘要：含 .cpp 不超 255 字节文件名上限，且组合间不碰撞
        from ttk.core_modules.geir.graph_builder import graph_signature

        long_sig = graph_signature(("relu",) * 42)
        assert len(long_sig) + len(".cpp") <= 255
        assert long_sig != graph_signature(("relu",) * 41 + ("add_n",))
        assert GeirInstance._binary_signature_key(op_tc) == "add"

    def test_build_tasks_groups_by_signature(self):
        """无圈选序时按签名稳定排序（同组相邻）；invalid 用例照旧计数跳过。"""
        from unittest import mock

        from ttk.core_modules.geir import GeirInstance

        tcs = [
            SimpleNamespace(
                testcase_name=f"c{i}",
                is_valid=True,
                op_name="mul",
                _resolved_graph={"ops": ("mul", "add_n")} if i % 2 else None,
            )
            for i in range(4)
        ]
        tcs.append(SimpleNamespace(testcase_name="bad", is_valid=False, op_name="x", _resolved_graph=None))
        ins = GeirInstance()
        ins.switches = SimpleNamespace()
        ins.skipped_cases = 0
        with mock.patch.object(GeirInstance, "result_titles", return_value=()), mock.patch(
            "ttk.core_modules.common.process.SimpleCommandProcess.device_grants",
            return_value=({}, {}),
        ):
            ins._selection_order_key = None
            tasks = ins.build_tasks(list(tcs))
        # 4 valid（1 个 invalid 跳过）；排序后 graph 签名组与 op 组各自相邻
        names = [t.testcase.testcase_name for t in tasks]
        assert len(tasks) == 4
        assert ins.skipped_cases == 1
        keys = [GeirInstance._binary_signature_key(t.testcase) for t in tasks]
        assert keys == sorted(keys), f"not grouped: {list(zip(names, keys))}"
        # 稳定性：同签名内保持传入相对序（c0/c2 op 组、c1/c3 graph 组）
        assert [n for n in names if n in ("c0", "c2")] == ["c0", "c2"]
        assert [n for n in names if n in ("c1", "c3")] == ["c1", "c3"]


class TestFailedCaseNoLeak:
    """失败用例的融合会话在 finally 推进快照，不泄漏给下一个复用 serve 的用例。"""

    def test_failure_advances_snapshot(self, tmp_path, monkeypatch):
        import ttk.core_modules.geir.execute as geir_execute

        serve = SimpleNamespace(serve_dir=str(tmp_path), alive=lambda: True)
        serve._seen_fusion_sessions = set()
        serve.collect_fusion_stats = lambda: _GeirServe.collect_fusion_stats(serve)
        (tmp_path / FUSION_STAT_FILENAME).write_text(
            json.dumps({"s0": {"graph_fusion": {"BadPass": {"match_times": "1", "effect_times": "1"}}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(geir_execute, "_SERVE_INST", serve)
        monkeypatch.setattr(
            geir_execute, "_run_once_serve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        plan = geir_execute.GraphExecPlan(
            binary="b", config_path="c", build_dir=str(tmp_path), prof_path="", input_prefix=str(tmp_path / "in")
        )
        tc = SimpleNamespace(
            input_arrays=[],
            original_input_arrays=None,
            flat_output_dtypes=[],
            flat_output_shapes=[],
            golden_arrays=[],
            testcase_name="t",
        )
        switches = SimpleNamespace(
            run_time=1, warmup=False, task_profiling=False, deterministic_level=0, proc_timeout=10
        )
        monkeypatch.setattr(geir_execute, "_write_all_input_bins", lambda *a, **k: None)
        monkeypatch.setattr(geir_execute, "_cleanup_bins", lambda *a, **k: None)
        monkeypatch.setattr(geir_execute, "clear_device_state", lambda *a, **k: None)
        monkeypatch.setattr(geir_execute, "_snapshot_prof_entries", lambda *a, **k: frozenset())
        with pytest.raises(RuntimeError, match="boom"):
            geir_execute.execute_graph(tc, 0, switches, plan)
        assert serve._seen_fusion_sessions == {"s0"}
        # 下一用例复用 serve：残留会话不再出现
        assert serve.collect_fusion_stats() is None
