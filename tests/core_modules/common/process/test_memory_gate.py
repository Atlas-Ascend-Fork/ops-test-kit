# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""gate_host_memory：宽松版 host 内存闸门（整段删除后按重设计策略恢复）。

旧闸门（waiting_for_memory）：available ≤ 50% 且 ≤ 128GB → 无限死等。
新语义：仅临界低位拦截（min(8GB, total*10%)）、有界等待、超时放行、
override 文件立即放行、--no-memory-check 跳过。
"""

from types import SimpleNamespace

import pytest

import ttk.core_modules.common.process.memory_gate as mg
from ttk.core_modules.common.process.memory_gate import gate_host_memory, host_memory_floor

GB = 1024**3


@pytest.fixture(autouse=True)
def _no_cgroup(monkeypatch):
    """默认关掉 cgroup 感知——现有用例只 mock 宿主快照，保持封闭。"""
    monkeypatch.setattr(mg, "cgroup_mem_limit", lambda: None)
    monkeypatch.setattr(mg, "cgroup_mem_current", lambda: None)


def _vm(total, available):
    return SimpleNamespace(total=total, available=available)


class _Ctx:
    def __init__(self):
        self.statuses = []

    def notify_status(self, status):
        self.statuses.append(status)


def test_floor_scales_down_on_small_hosts():
    """大机器取 8GB 绝对下限；小机器按 10% 比例缩。"""
    assert host_memory_floor(512 * GB) == 8 * GB
    assert host_memory_floor(64 * GB) == 6.4 * GB
    assert host_memory_floor(8 * GB) == 0.8 * GB


def test_gate_passes_when_headroom_sufficient(monkeypatch):
    """available ≥ 水位 → 立即返回，不上报等待状态。"""
    monkeypatch.setattr("psutil.virtual_memory", lambda: _vm(512 * GB, 100 * GB))
    ctx = _Ctx()
    gate_host_memory(False, ctx)
    assert ctx.statuses == []


def test_gate_no_check_skips_entirely(monkeypatch):
    """--no-memory-check → 连 psutil 都不碰。"""

    def boom():
        raise AssertionError("psutil must not be touched")

    monkeypatch.setattr("psutil.virtual_memory", boom)
    gate_host_memory(True, None)


def test_gate_waits_until_memory_recovers(monkeypatch):
    """低于水位 → 等待；内存回升 → 放行（上报 OnWaitingForMemory）。"""
    readings = [_vm(512 * GB, 2 * GB), _vm(512 * GB, 2 * GB), _vm(512 * GB, 20 * GB)]

    def read():
        return readings.pop(0) if readings else _vm(512 * GB, 20 * GB)

    monkeypatch.setattr("psutil.virtual_memory", read)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    ctx = _Ctx()
    gate_host_memory(False, ctx, timeout_s=60)
    assert ctx.statuses == ["OnWaitingForMemory"]


def test_gate_times_out_and_proceeds(monkeypatch, caplog):
    """持续低于水位 → 有界等待后放行并告警（不再死等）。"""
    monkeypatch.setattr("psutil.virtual_memory", lambda: _vm(512 * GB, 1 * GB))
    # 时间快进：单调钟直接跳过 deadline
    clock = iter([0, 10, 20, 10_000, 10_001, 10_002])
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        gate_host_memory(False, None, timeout_s=60)
    assert any("proceeding anyway" in r.message for r in caplog.records)


def test_gate_override_file_releases(monkeypatch, tmp_path, caplog):
    """/tmp/TTK_FORCE_MEMORY_OVERRIDE（旧运维放行口）→ 立即放行（断言告警，区别于超时静默放行）。"""
    monkeypatch.setattr("psutil.virtual_memory", lambda: _vm(512 * GB, 1 * GB))
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(mg, "OVERRIDE_FILE", str(tmp_path / "override"))
    (tmp_path / "override").write_text("")
    with caplog.at_level("WARNING"):
        gate_host_memory(False, None, timeout_s=60)
    assert any("override detected" in r.message for r in caplog.records)


def test_gate_trips_on_cgroup_headroom(monkeypatch):
    """限额容器：宿主 available 充裕但 cgroup 余量低于水位 → 拦截等待。"""
    monkeypatch.setattr("psutil.virtual_memory", lambda: _vm(754 * GB, 700 * GB))
    monkeypatch.setattr(mg, "cgroup_mem_limit", lambda: 32 * GB)
    headrooms = [2 * GB, 2 * GB, 30 * GB]

    def read_current():
        # 余量序列 2G（低于水位 3.2G）→ 2G → 30G（恢复）
        return 32 * GB - headrooms.pop(0)

    monkeypatch.setattr(mg, "cgroup_mem_current", read_current)
    monkeypatch.setattr("time.sleep", lambda s: None)
    ctx = _Ctx()
    gate_host_memory(False, ctx, timeout_s=60)
    assert ctx.statuses == ["OnWaitingForMemory"]


def test_gate_cgroup_floor_scales_by_limit(monkeypatch):
    """水位按 min(限额, 宿主) 计算：32G 限额 → 水位 3.2G 而非 8G。"""
    monkeypatch.setattr("psutil.virtual_memory", lambda: _vm(754 * GB, 700 * GB))
    monkeypatch.setattr(mg, "cgroup_mem_limit", lambda: 32 * GB)
    monkeypatch.setattr(mg, "cgroup_mem_current", lambda: 32 * GB - 4 * GB)  # 余量 4G > 3.2G
    ctx = _Ctx()
    gate_host_memory(False, ctx)
    assert ctx.statuses == []
