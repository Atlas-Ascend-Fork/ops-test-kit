# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------
"""utilities.memory：cgroup 限额解析 + CANN 编译并行度收敛。"""

import ttk.utilities.memory as mem

GB = 1024**3


def test_cgroup_limit_v2(monkeypatch):
    monkeypatch.setattr(mem, "_read_int", lambda p: 32 * GB if p == mem._CGROUP_V2_MAX else None)
    assert mem.cgroup_mem_limit() == 32 * GB
    assert mem.cgroup_mem_current() is None


def test_cgroup_limit_v1_fallback(monkeypatch):
    def read(path):
        if path == mem._CGROUP_V1_LIMIT:
            return 16 * GB
        if path == mem._CGROUP_V1_CURRENT:
            return 4 * GB
        return None

    monkeypatch.setattr(mem, "_read_int", read)
    assert mem.cgroup_mem_limit() == 16 * GB
    assert mem.cgroup_mem_current() == 4 * GB


def test_cgroup_unlimited_is_none(monkeypatch):
    # v2 "max" → 非整数 → None；v1 巨值 → None
    monkeypatch.setattr(
        mem,
        "_read_int",
        lambda p: 9223372036854771712 if p == mem._CGROUP_V1_LIMIT else None,
    )
    assert mem.cgroup_mem_limit() is None


def test_no_cgroup_is_none(monkeypatch):
    monkeypatch.setattr(mem, "_read_int", lambda p: None)
    assert mem.cgroup_memory_snapshot() == (None, None)


def test_limit_jobs_throttles_small_cgroup(monkeypatch):
    """32G 限额 → 预算(限额-当前-预留) / (8 并发 × 4G) → 串行(1)。"""
    monkeypatch.delenv("TILINGKEY_PARALLEL_JOB", raising=False)
    monkeypatch.delenv("TILINGKEY_PAR_COMPILE", raising=False)
    monkeypatch.setattr(mem, "cgroup_memory_snapshot", lambda: (32 * GB, 8 * GB))
    jobs = mem.limit_cann_compile_jobs()
    assert jobs == 1
    import os

    assert os.environ["TILINGKEY_PARALLEL_JOB"] == "1"


def test_limit_jobs_noop_without_cgroup(monkeypatch):
    monkeypatch.delenv("TILINGKEY_PARALLEL_JOB", raising=False)
    monkeypatch.setattr(mem, "cgroup_memory_snapshot", lambda: (None, None))
    assert mem.limit_cann_compile_jobs() is None
    import os

    assert "TILINGKEY_PARALLEL_JOB" not in os.environ


def test_limit_jobs_respects_user_override(monkeypatch):
    monkeypatch.setenv("TILINGKEY_PARALLEL_JOB", "4")
    monkeypatch.setattr(mem, "cgroup_memory_snapshot", lambda: (32 * GB, 0))
    assert mem.limit_cann_compile_jobs() is None
    import os

    assert os.environ["TILINGKEY_PARALLEL_JOB"] == "4"


def test_limit_jobs_large_cgroup_keeps_cann_default(monkeypatch):
    """预算充足（推算 ≥ 8）→ 不注入，保持 CANN 默认。"""
    monkeypatch.delenv("TILINGKEY_PARALLEL_JOB", raising=False)
    monkeypatch.setattr(mem, "cgroup_memory_snapshot", lambda: (512 * GB, 8 * GB))
    assert mem.limit_cann_compile_jobs() is None
    import os

    assert "TILINGKEY_PARALLEL_JOB" not in os.environ
