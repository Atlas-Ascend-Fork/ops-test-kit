# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_handle_run 设备锁集成测试：验证非 dry_run 时调 _assign_device、dry_run 跳过分配、CPU 模式无 KeyError。"""

import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ttk.remote.server import xpu_server
from ttk.remote.server.xpu_server import XpuRequestHandler

FAKE_IN = "/tmp/fake"


def _setup_handler(h, device_ids, gpu_locks, dry_run, use_device=False, hardware="gpu", profile=None):
    """Configure a minimal XpuRequestHandler instance for _handle_run tests.

    The real _handle_run reads tenant_id via self._get_header; we stub it to
    always yield a valid tenant so the request reaches the dispatch body.

    hardware 形参保留——heartbeat (:287) 读 self.hardware。profile 形参用于
    _handle_run :464 kwargs（executor 消费）。is None 守护保留 {} cpu 兜底
    （非 or，防 {} 被 "gpu" str 覆盖导致 profile["torch_lib"] TypeError）。
    """
    h.dry_run = dry_run
    h.use_device = use_device
    h.device_ids = device_ids
    h.hardware = hardware
    h.profile = profile if profile is not None else {"torch_lib": "cuda"}
    h._device_rr_counter = 0
    h._device_rr_lock = threading.Lock()
    xpu_server._device_locks = gpu_locks
    h.data_gate = None
    h.tmp_root = "/tmp"
    h.sync_base_dir = "/tmp"
    h.run_deadline_s = 30
    h.sandbox = "none"
    h.device_unhealthy_detect_interval_s = 60.0

    def _get_header(k, d=""):
        return "test_tenant" if k == "X-Tenant-ID" else d

    h._get_header = _get_header
    h._send_run_ok = MagicMock()
    h._send_json = MagicMock()
    h.rfile = MagicMock()
    h.headers = MagicMock()


def _patch_run_env(monkeypatch):
    """Stub body receive, subprocess, and rmtree so _handle_run walks the path
    without touching the filesystem (except the getsize on FAKE_IN)."""
    monkeypatch.setattr(
        "ttk.remote.server.xpu_server._run_in_subprocess",
        lambda kw, deadline: {
            "ok": True,
            "http_status": 200,
            "output_path": None,
            "output_count": 0,
            "shapes": [],
            "dtypes": [],
            "perf": None,
            "api": None,
        },
    )
    monkeypatch.setattr("ttk.remote.server.xpu_server._receive_body_to_file", lambda handler, dir=None: FAKE_IN)
    monkeypatch.setattr("ttk.remote.server.xpu_server.shutil.rmtree", lambda *a, **kw: None)


def test_handle_run_uses_assign_device(monkeypatch):
    """_handle_run 非 dry_run 时调 _assign_device（而非 _device_locks / device_ids[0]）。"""
    _patch_run_env(monkeypatch)
    with open(FAKE_IN, "wb"):
        pass
    try:
        h = XpuRequestHandler.__new__(XpuRequestHandler)
        _setup_handler(h, [0, 1], {0: threading.Lock(), 1: threading.Lock()}, dry_run=False, use_device=True)

        captured = {}

        def fake_assign():
            captured["called"] = True
            # mirror real _assign_device contract: acquire the per-device lock
            # so the caller's finally can release it.
            assert xpu_server._device_locks[0].acquire(blocking=False) is True
            return 0

        h._assign_device = fake_assign

        h._handle_run()

        assert captured.get("called") is True
        # device 0 的锁被 release（finally）——能再次 acquire 证明已 release
        assert xpu_server._device_locks[0].acquire(blocking=False) is True
        xpu_server._device_locks[0].release()
    finally:
        if os.path.exists(FAKE_IN):
            os.remove(FAKE_IN)


# ══════════════════════════════════════════════════════════════════════════
# /v1/run 请求侧守护：tenant_id 校验 / TenantManager 清理边界 / gate 释放
# ══════════════════════════════════════════════════════════════════════════
class TestTenantIdValidation:
    """tenant_id 入口校验：拒绝 `.`/`..`（经 join 逃出 sync base）、路径分隔、超长。"""

    @pytest.mark.parametrize(
        ("tid", "expected"),
        [
            (".", False),
            ("..", False),
            ("a/b", False),
            ("x" * 65, False),
            ("abc", True),
            ("a.b_c-1", True),
        ],
    )
    def test_tenant_id_validation(self, tid, expected):
        assert xpu_server.XpuRequestHandler._valid_tenant_id(tid) is expected


class TestTenantManagerSafeRmtree:
    """TenantManager 清理边界：越界路径拒绝删除，界内照常。"""

    @pytest.mark.parametrize("escape_kind", ["sibling", "parent"])
    def test_outside_base_refused(self, tmp_path, escape_kind):
        target = tmp_path / (".." if escape_kind == "parent" else "outside")
        target.mkdir() if escape_kind == "sibling" else None
        keep = target / "keep.txt"
        keep.write_text("x")
        mgr = xpu_server.TenantManager(str(tmp_path / "base"))
        mgr._tenants["evil"] = {"last_heartbeat": 0, "path": str(target)}

        mgr.cleanup("evil")
        assert keep.exists()  # 未被删除

    def test_inside_base_removed(self, tmp_path):
        base = tmp_path / "base"
        tenant = base / "t1"
        tenant.mkdir(parents=True)
        (tenant / "data.bin").write_bytes(b"1")
        mgr = xpu_server.TenantManager(str(base))
        mgr._tenants["t1"] = {"last_heartbeat": 0, "path": str(tenant)}

        assert mgr.cleanup("t1") is True
        assert not tenant.exists()


class TestMkdtempGateLeak:
    """请求目录创建失败时 data_gate 必须释放。"""

    def test_gate_released_when_mkdtemp_fails(self, tmp_path, monkeypatch):
        handler = SimpleNamespace(
            data_gate=threading.Semaphore(1),
            gate_wait_s=0.1,
            tmp_root=str(tmp_path),
        )

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(xpu_server.tempfile, "mkdtemp", _boom)
        monkeypatch.setattr(
            xpu_server.XpuRequestHandler,
            "_dispatch_run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("unreachable")),
        )
        sent = []
        monkeypatch.setattr(xpu_server.XpuRequestHandler, "_send_run_result", lambda *a, **k: sent.append(a))

        with pytest.raises(OSError, match="disk full"):
            xpu_server.XpuRequestHandler._run_under_gate(handler, "t1", "api", {})

        # 关键断言：gate 已释放（可再次立即获取）
        assert handler.data_gate.acquire(timeout=0.01) is True
