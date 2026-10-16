# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""executor.execute_request — direct in-process unit tests (DATA path).

本模块对 ``ttk.remote.server.executor`` 的核心路径做单测，覆盖：
- DATA 模式 API / Spec 执行（含错误路径与安全脱敏）
- 子进程隔离执行 + 真实 server E2E（含 backpressure）
"""

import http.client
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import pytest

# torch_npu._C + tensorflow C extension 在同一进程 import 时符号冲突 → segfault。
# 仅当两者共存时跳过；XPU server 环境（有 torch/tf 无 torch_npu）正常通过。


def _npz(path, **arrs):
    np.savez_compressed(str(path), **{f"a{i}": v for i, v in enumerate(arrs.values())})
    return str(path)


def _kw(**over):
    base = {
        "tenant_sync_dir": "",
        "exec_type": "api",
        "provider": "numpy",
        "api": None,
        "spec_module": None,
        "spec_class": None,
        "mode": 1,
        "input_schema": [],
        "attrs": {},
        "tmp_in_path": None,
        "input_count": 0,
        "device_id": 0,
        "use_device": False,
        "output_dir": None,
    }
    base.update(over)
    return base


class TestApiData:
    """DATA 模式 API 直调路径。"""

    def test_numpy_add(self, tmp_path):
        """numpy.add 经 execute_request 正常返回 200 + 单输出。"""
        from ttk.remote import DATA
        from ttk.remote.server import executor

        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]), x2=np.array([3.0, 4.0]))
        env = executor.execute_request(
            **_kw(
                api="numpy.add",
                mode=DATA,
                input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
                tmp_in_path=inp,
                input_count=2,
                output_dir=str(tmp_path),
            )
        )
        assert env["ok"]
        assert env["http_status"] == 200
        assert env["output_count"] == 1
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"], np.array([4.0, 6.0]))

    def test_api_missing_module_is_500(self, tmp_path):
        """api 指向不存在的模块 -> 500，且 missing=None（非 424 syncable）。"""
        from ttk.remote import DATA
        from ttk.remote.server import executor

        env = executor.execute_request(**_kw(api="definitely_not_a_module_xyz.fn", mode=DATA, output_dir=str(tmp_path)))
        assert env["ok"] is False
        assert env["http_status"] == 500
        assert env["missing"] is None  # not a syncable 424


class TestSpecData:
    """DATA 模式 Spec 路径（third_party 实现加载 + 输入注入 + 错误脱敏）。"""

    def _write(self, path, body):
        path.write_text(body)

    @pytest.mark.parametrize(
        ("mode", "body", "x1", "x2", "expected"),
        [
            pytest.param(
                "b",
                "class AddImpl:\n"
                "    def __init__(self, **kw): pass\n"
                "    def __call__(self, x1, x2): return [x1 + x2]\n"
                "class AddTestSpec:\n"
                "    third_party = {'numpy': AddImpl}\n",
                np.array([1.0]),
                np.array([5.0]),
                np.array([6.0]),
                id="mode_b_inputs_in_call",
            ),
            pytest.param(
                "a",
                "class AddImpl:\n"
                "    def __init__(self, x1, x2, **kw):\n"
                "        self.x1, self.x2 = x1, x2\n"
                "    def __call__(self):\n"
                "        return [self.x1 + self.x2]\n"
                "class AddTestSpec:\n"
                "    third_party = {'numpy': AddImpl}\n",
                np.array([2.0]),
                np.array([10.0]),
                np.array([12.0]),
                id="mode_a_inputs_in_init",
            ),
        ],
    )
    def test_spec_input_injection(self, tmp_path, mode, body, x1, x2, expected):
        """mode A（输入注入 ``__init__``）/ mode B（输入注入 ``__call__``）均正常返回。"""
        from ttk.remote import DATA
        from ttk.remote.server import executor

        fname = f"add{mode}.py"
        self._write(tmp_path / fname, body)
        inp = _npz(tmp_path / "in.npz", x1=x1, x2=x2)
        env = executor.execute_request(
            **_kw(
                exec_type="spec",
                spec_module=f"add{mode}",
                spec_class="AddTestSpec",
                mode=DATA,
                provider="numpy",
                input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
                tmp_in_path=inp,
                input_count=2,
                tenant_sync_dir=str(tmp_path),
                output_dir=str(tmp_path),
            )
        )
        assert env["ok"]
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"], expected)

    @pytest.mark.parametrize(
        "scenario",
        [
            pytest.param("missing_module_424", id="missing_spec_module_is_424"),
            pytest.param("unknown_param_400", id="unknown_param_is_400"),
        ],
    )
    def test_spec_error_paths(self, tmp_path, scenario):
        """Spec 错误路径：缺模块 -> 424（syncable，missing=模块名）；未知参数 -> 400。"""
        from ttk.remote import DATA
        from ttk.remote.server import executor

        if scenario == "missing_module_424":
            env = executor.execute_request(
                **_kw(
                    exec_type="spec",
                    spec_module="no_such_spec",
                    spec_class="X",
                    mode=DATA,
                    provider="numpy",
                    tenant_sync_dir=str(tmp_path),
                    output_dir=str(tmp_path),
                )
            )
            assert env["ok"] is False
            assert env["http_status"] == 424
            assert env["missing"] == "no_such_spec"
        else:  # unknown_param_400
            self._write(
                tmp_path / "bad.py",
                "class BadImpl:\n"
                "    def __call__(self, x1, bogus): return [x1]\n"
                "class BadTestSpec:\n"
                "    third_party = {'numpy': BadImpl}\n",
            )
            inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]))
            env = executor.execute_request(
                **_kw(
                    exec_type="spec",
                    spec_module="bad",
                    spec_class="BadTestSpec",
                    mode=DATA,
                    provider="numpy",
                    input_schema=[{"name": "x1", "index": 0}],
                    tmp_in_path=inp,
                    input_count=1,
                    tenant_sync_dir=str(tmp_path),
                    output_dir=str(tmp_path),
                )
            )
            assert env["ok"] is False
            assert env["http_status"] == 400
            assert "bogus" in env["error"]

    def test_op_failure_error_is_sanitized_no_traceback(self, tmp_path):
        """Security (OWASP Improper Error Handling): a failing op's client error
        is TYPE+MESSAGE only — no traceback, no server FS path leaks over the
        wire / into xpu-metrics. Full traceback stays server-side
        (logging.exception). _client_error is the single control point."""
        from ttk.remote import DATA
        from ttk.remote.server import executor

        self._write(
            tmp_path / "boom.py",
            "class BoomImpl:\n"
            "    def __call__(self, x1): raise RuntimeError('op exploded')\n"
            "class BoomSpec:\n"
            "    third_party = {'numpy': BoomImpl}\n",
        )
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]))
        env = executor.execute_request(
            **_kw(
                exec_type="spec",
                spec_module="boom",
                spec_class="BoomSpec",
                mode=DATA,
                provider="numpy",
                input_schema=[{"name": "x1", "index": 0}],
                tmp_in_path=inp,
                input_count=1,
                tenant_sync_dir=str(tmp_path),
                output_dir=str(tmp_path),
            )
        )
        assert env["ok"] is False
        assert env["http_status"] == 500
        assert env["error"] == "RuntimeError: op exploded"
        assert "Traceback" not in env["error"]
        assert ".py" not in env["error"]  # no server/spec file path leaked


class TestPerfPath:
    """PERF timing: CPU NA-fallback + simulated CUDA timing (no GPU needed)."""

    def test_data_perf_cpu_marks_na(self, tmp_path):
        """DATA|PERF 在 CPU 上 device_us / peak_memory_mb 均为 NA。"""
        from ttk.remote import DATA, PERF
        from ttk.remote.server import executor

        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]), x2=np.array([3.0, 4.0]))
        env = executor.execute_request(
            **_kw(
                api="numpy.add",
                mode=DATA | PERF,
                input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
                tmp_in_path=inp,
                input_count=2,
                output_dir=str(tmp_path),
            )
        )
        assert env["ok"]
        assert env["output_count"] == 1
        perf = env["perf"]
        assert perf["device_us"] == "NA"
        assert perf["peak_memory_mb"] == "NA"


class TestViaSubprocess:
    """_run_in_subprocess: execute_request in a real fresh child process."""

    def test_numpy_add_runs_in_child_process(self, tmp_path):
        """子进程内正常执行 numpy.add，输出与父进程一致。"""
        from ttk.remote import DATA
        from ttk.remote.server.xpu_server import _run_in_subprocess

        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]), x2=np.array([3.0, 4.0]))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        env = _run_in_subprocess(
            _kw(
                api="numpy.add",
                mode=DATA,
                input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
                tmp_in_path=inp,
                input_count=2,
                output_dir=str(out_dir),
            ),
            deadline=60,
        )
        assert env["ok"]
        assert env["output_count"] == 1
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"], np.array([4.0, 6.0]))

    def test_deadline_is_relative_seconds_not_absolute(self, tmp_path):
        """_run_in_subprocess 的 deadline 为相对秒数：超时子进程被 kill 并回 500。"""
        from ttk.remote import DATA
        from ttk.remote.server.xpu_server import _run_in_subprocess

        (tmp_path / "slowfix.py").write_text(
            "import time\n"
            "class SlowImpl:\n"
            "    def __call__(self, x1): time.sleep(60)\n"
            "class SlowTestSpec:\n"
            "    third_party = {'numpy': SlowImpl}\n"
        )
        start = time.monotonic()
        env = _run_in_subprocess(
            _kw(
                exec_type="spec",
                spec_module="slowfix",
                spec_class="SlowTestSpec",
                mode=DATA,
                provider="numpy",
                input_schema=[{"name": "x1", "index": 0}],
                tmp_in_path=_npz(tmp_path / "in.npz", x1=np.array([1.0])),
                input_count=1,
                tenant_sync_dir=str(tmp_path),
                output_dir=str(tmp_path),
            ),
            deadline=2,
        )
        elapsed = time.monotonic() - start

        assert env["ok"] is False  # 超时 500，而非等到 60s 或恒 1s
        assert "timed out after 2s" in env.get("error", "")
        assert elapsed < 30  # 相对秒生效（60s 任务 2s 被 kill）

    def test_child_crash_returns_500(self, tmp_path):
        """子进程 ``os._exit(139)`` -> 500，错误信息含退出码 139。"""
        from ttk.remote import DATA
        from ttk.remote.server.xpu_server import _run_in_subprocess

        (tmp_path / "crashfix.py").write_text(
            "import os\n"
            "class CrashImpl:\n"
            "    def __call__(self, x1): os._exit(139)\n"
            "class CrashTestSpec:\n"
            "    third_party = {'numpy': CrashImpl}\n"
        )
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]))
        env = _run_in_subprocess(
            _kw(
                exec_type="spec",
                spec_module="crashfix",
                spec_class="CrashTestSpec",
                mode=DATA,
                provider="numpy",
                input_schema=[{"name": "x1", "index": 0}],
                tmp_in_path=inp,
                input_count=1,
                tenant_sync_dir=str(tmp_path),
                output_dir=str(tmp_path),
            ),
            deadline=60,
        )
        assert env["ok"] is False
        assert env["http_status"] == 500
        assert "139" in env["error"]


def _post_run(port, body, headers, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/run", body=body, headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def _sync_module(port, tenant, rel_path, content):
    import base64
    import hashlib

    body = json.dumps(
        {
            "files": {
                rel_path: {
                    "content": base64.b64encode(content.encode()).decode(),
                    "hash": hashlib.sha256(content.encode()).hexdigest(),
                }
            }
        }
    )
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/sync", body=body, headers={"Content-Type": "application/json", "X-Tenant-ID": tenant})
    r = c.getresponse()
    r.read()
    c.close()
    return r.status


class TestServerRunViaSubprocess:
    """End-to-end: real non-dry-run server runs /v1/run in a child process."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory):
        sync = tmp_path_factory.mktemp("sync")
        tmp = tmp_path_factory.mktemp("tmp")

        # Create a config file for the server
        import yaml

        config_file = tmp_path_factory.mktemp("config") / "xpu_server.yaml"
        config_data = {
            "storage": {
                "sync_dir": str(sync),
                "tmp_dir": str(tmp),
            },
            "server": {
                "run_deadline_s": 60,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server.xpu_server",
                "--port",
                "19150",
                "--devices",
                "cpu",
                "--config",
                str(config_file),
            ],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            try:
                c = http.client.HTTPConnection("127.0.0.1", 19150, timeout=1)
                c.request("GET", "/v1/heartbeat")
                r = c.getresponse()
                c.close()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, OSError):
                pass
            time.sleep(0.5)
        yield proc
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_run_numpy_add(self, server):
        from ttk.remote import DATA

        a0 = np.array([1.0, 2.0])
        a1 = np.array([3.0, 4.0])
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=a0, a1=a1)
        c = http.client.HTTPConnection("127.0.0.1", 19150, timeout=60)
        c.request(
            "POST",
            "/v1/run",
            body=buf.getvalue(),
            headers={
                "X-Execution-Type": "api",
                "X-Provider": "numpy",
                "X-API": "numpy.add",
                "X-Mode": str(DATA),
                "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}, {"name": "x2", "index": 1}]),
                "X-Input-Count": "2",
                "X-Attrs": "{}",
                "X-Tenant-ID": "e2e1",
                "Content-Type": "application/octet-stream",
            },
        )
        r = c.getresponse()
        data = r.read()
        c.close()
        assert r.status == 200
        np.testing.assert_array_equal(np.load(io.BytesIO(data))["a0"], a0 + a1)

    def test_crash_then_server_keeps_serving(self, server):
        from ttk.remote import DATA

        _sync_module(
            19150,
            "cr",
            "crashfix.py",
            "import os\n"
            "class CrashImpl:\n"
            "    def __call__(self, x1): os._exit(139)\n"
            "class CrashTestSpec:\n"
            "    third_party = {'numpy': CrashImpl}\n",
        )
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([1.0]))
        h = {
            "X-Execution-Type": "spec",
            "X-Provider": "numpy",
            "X-Spec-Module": "crashfix",
            "X-Spec-Class": "CrashTestSpec",
            "X-Mode": str(DATA),
            "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}]),
            "X-Input-Count": "1",
            "X-Attrs": "{}",
            "X-Tenant-ID": "cr",
            "Content-Type": "application/octet-stream",
        }
        s1, _ = _post_run(19150, buf.getvalue(), h)
        assert s1 == 500  # child crashed

        addbuf = io.BytesIO()
        np.savez_compressed(addbuf, a0=np.array([1.0]), a1=np.array([2.0]))
        s2, b2 = _post_run(
            19150,
            addbuf.getvalue(),
            {
                "X-Execution-Type": "api",
                "X-Provider": "numpy",
                "X-API": "numpy.add",
                "X-Mode": str(DATA),
                "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}, {"name": "x2", "index": 1}]),
                "X-Input-Count": "2",
                "X-Attrs": "{}",
                "X-Tenant-ID": "cr2",
                "Content-Type": "application/octet-stream",
            },
        )
        assert s2 == 200  # server survived, fresh child
        np.testing.assert_array_equal(np.load(io.BytesIO(b2))["a0"], np.array([3.0]))

    def test_424_then_sync_then_success(self, server):
        from ttk.remote import DATA

        add_src = (
            "class AddImpl:\n"
            "    def __call__(self, x1, x2): return [x1 + x2]\n"
            "class AddTestSpec:\n"
            "    third_party = {'numpy': AddImpl}\n"
        )
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([2.0]), a1=np.array([3.0]))
        h = {
            "X-Execution-Type": "spec",
            "X-Provider": "numpy",
            "X-Spec-Module": "addfix",
            "X-Spec-Class": "AddTestSpec",
            "X-Mode": str(DATA),
            "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}, {"name": "x2", "index": 1}]),
            "X-Input-Count": "2",
            "X-Attrs": "{}",
            "X-Tenant-ID": "resync",
            "Content-Type": "application/octet-stream",
        }
        s1, b1 = _post_run(19150, buf.getvalue(), h)
        assert s1 == 424
        assert json.loads(b1).get("missing") in ("addfix", "addfix.py")

        assert _sync_module(19150, "resync", "addfix.py", add_src) == 200
        s3, b3 = _post_run(19150, buf.getvalue(), h)
        assert s3 == 200  # fresh child reads the synced file
        np.testing.assert_array_equal(np.load(io.BytesIO(b3))["a0"], np.array([5.0]))

    def test_cross_tenant_isolation(self, server):
        # Two tenants sync the SAME module name with DIFFERENT impls; concurrent
        # requests must each get their own impl (fresh sys.modules per child).
        from ttk.remote import DATA

        _sync_module(
            19150,
            "ta",
            "sharedfix.py",
            "class AddImpl:\n"
            "    def __call__(self, x1, x2): return [x1 + x2]\n"
            "class AddTestSpec:\n"
            "    third_party = {'numpy': AddImpl}\n",
        )
        _sync_module(
            19150,
            "tb",
            "sharedfix.py",
            "class AddImpl:\n"
            "    def __call__(self, x1, x2): return [x1 * x2]\n"
            "class AddTestSpec:\n"
            "    third_party = {'numpy': AddImpl}\n",
        )
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([2.0]), a1=np.array([3.0]))

        def hdr(t):
            return {
                "X-Execution-Type": "spec",
                "X-Provider": "numpy",
                "X-Spec-Module": "sharedfix",
                "X-Spec-Class": "AddTestSpec",
                "X-Mode": str(DATA),
                "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}, {"name": "x2", "index": 1}]),
                "X-Input-Count": "2",
                "X-Attrs": "{}",
                "X-Tenant-ID": t,
                "Content-Type": "application/octet-stream",
            }

        results = []
        errors = []

        def fire(t):
            try:
                results.append((t, _post_run(19150, buf.getvalue(), hdr(t))))
            except Exception:
                errors.append((t, traceback.format_exc()))

        threads = []
        for _ in range(8):
            threads.append(threading.Thread(target=fire, args=("ta",)))
            threads.append(threading.Thread(target=fire, args=("tb",)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=70)

        assert not errors, f"fire raised: {errors}"
        assert len(results) == 16, f"results={len(results)} errors={errors}"
        for tenant, (status, data) in results:
            assert status == 200
            out = np.load(io.BytesIO(data))["a0"]
            expected = np.array([5.0]) if tenant == "ta" else np.array([6.0])
            np.testing.assert_array_equal(out, expected)

    def test_perf_mode_returns_x_perf_header(self, server):
        from ttk.remote import DATA, PERF

        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([1.0]), a1=np.array([2.0]))
        c = http.client.HTTPConnection("127.0.0.1", 19150, timeout=60)
        c.request(
            "POST",
            "/v1/run",
            body=buf.getvalue(),
            headers={
                "X-Execution-Type": "api",
                "X-Provider": "numpy",
                "X-API": "numpy.add",
                "X-Mode": str(DATA | PERF),
                "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}, {"name": "x2", "index": 1}]),
                "X-Input-Count": "2",
                "X-Attrs": "{}",
                "X-Tenant-ID": "perf1",
                "Content-Type": "application/octet-stream",
            },
        )
        r = c.getresponse()
        perf_h = r.getheader("X-Perf")
        r.read()
        c.close()
        assert r.status == 200
        assert perf_h is not None
        json.loads(perf_h)


class TestBackpressure:
    """MAX_CONCURRENT gate: overflow -> 503; control plane bypasses it."""

    @pytest.fixture(scope="class")
    def gate_server(self, tmp_path_factory):
        sync = tmp_path_factory.mktemp("sync_g")
        tmp = tmp_path_factory.mktemp("tmp_g")

        # Create a config file for the server
        import yaml

        config_file = tmp_path_factory.mktemp("config") / "xpu_server.yaml"
        config_data = {
            "storage": {
                "sync_dir": str(sync),
                "tmp_dir": str(tmp),
            },
            "server": {
                "max_concurrent": 1,
                "gate_wait_s": 0.3,
                "run_deadline_s": 60,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server.xpu_server",
                "--port",
                "19151",
                "--devices",
                "cpu",
                "--config",
                str(config_file),
            ],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            try:
                c = http.client.HTTPConnection("127.0.0.1", 19151, timeout=1)
                c.request("GET", "/v1/heartbeat")
                r = c.getresponse()
                c.close()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, OSError):
                pass
            time.sleep(0.5)
        yield proc
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_gate_full_returns_503_and_control_plane_ok(self, gate_server):
        from ttk.remote import DATA

        slow = (
            "import time\n"
            "class SlowImpl:\n"
            "    def __call__(self, x1):\n"
            "        time.sleep(2.0)\n"
            "        return [x1]\n"
            "class SlowTestSpec:\n"
            "    third_party = {'numpy': SlowImpl}\n"
        )
        _sync_module(19151, "g", "slowfix.py", slow)
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([1.0]))
        h = {
            "X-Execution-Type": "spec",
            "X-Provider": "numpy",
            "X-Spec-Module": "slowfix",
            "X-Spec-Class": "SlowTestSpec",
            "X-Mode": str(DATA),
            "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}]),
            "X-Input-Count": "1",
            "X-Attrs": "{}",
            "X-Tenant-ID": "g",
            "Content-Type": "application/octet-stream",
        }

        holder = {}

        def fire():
            holder["s"] = _post_run(19151, buf.getvalue(), h)[0]

        t = threading.Thread(target=fire)
        t.start()
        time.sleep(0.8)  # let the holder acquire the 1 slot
        s2, _ = _post_run(19151, buf.getvalue(), h)
        assert s2 == 503  # gate full, 0.3s wait elapses

        c = http.client.HTTPConnection("127.0.0.1", 19151, timeout=2)  # control plane
        c.request("GET", "/v1/heartbeat")
        r = c.getresponse()
        r.read()
        c.close()
        assert r.status == 200

        t.join(timeout=15)
        assert holder.get("s") == 200
