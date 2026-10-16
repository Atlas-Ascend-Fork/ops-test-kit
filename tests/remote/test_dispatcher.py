# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for remote dispatcher: serialization, dispatch, backoff, round-trip."""

import io
import json
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def xpu_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server", "--port", "19092", "--dry-run"],
        # 服务端输出丢弃, 不能用 PIPE: 没人读时管道缓冲区(64 KB)写满会让服务端
        # **永久阻塞**(框架预载一次就能吐 100 KB+), 表现为请求超时而非报错。
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            import http.client

            conn = http.client.HTTPConnection("127.0.0.1", 19092, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            conn.getresponse().read()
            conn.close()
            break
        except (ConnectionRefusedError, OSError):
            continue
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestDispatchApiMode:
    def test_dispatch_api_string(self, xpu_server):
        from ttk.remote.dispatcher import dispatch_to_remote

        inputs = [np.random.randn(4, 8).astype(np.float32)]
        outputs = dispatch_to_remote(
            op_name="softmax_v2",
            inputs=inputs,
            provider="torch",
            attrs={"axis": -1},
            endpoint_host="127.0.0.1",
            endpoint_port=19092,
            tenant_id="disp_test_001",
        )
        assert isinstance(outputs, list)
        assert len(outputs) >= 1
        assert isinstance(outputs[0], np.ndarray)


class TestSerializationRoundtrip:
    def test_numpy_roundtrip(self):
        from ttk.remote.dispatcher import _load_npz_outputs, _serialize_to_file

        original = [
            np.random.randn(4, 8).astype(np.float32),
            np.random.randn(3, 5).astype(np.int64),
        ]
        tmp = _serialize_to_file(original)
        try:
            schema = [{"index": 0, "dtype": "float32"}, {"index": 1, "dtype": "int64"}]
            outputs = _load_npz_outputs(tmp, schema)
            assert len(outputs) == 2
            np.testing.assert_array_equal(outputs[0], original[0])
            np.testing.assert_array_equal(outputs[1], original[1])
        finally:
            import os

            os.unlink(tmp)


class TestInputSchema:
    def test_build_schema_tensor_list(self):
        # 守卫：list slot 发 indices + 真实 dtype（卡 index/dtype 错配）
        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema

        a, b, c = np.array([1.0]), np.array([2.0]), np.array([3.0])
        schema = _build_input_schema(inputs=[[a, b], c], input_names=["x", "y"])
        assert schema == [
            {"name": "x", "indices": [0, 1], "dtype": "float64", "format": None, "logical_dtype": None},
            {"name": "y", "index": 2, "dtype": "float64", "format": None, "logical_dtype": None},
        ]

    def test_build_schema_logical_dtypes(self):
        # logical_dtype：逐 slot 嵌入 CSV 声明值（complex32 等）；TensorList 的
        # 嵌套 dtype 取首叶子；越界/缺失 → None
        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema

        a = np.zeros((2, 3, 2), dtype=np.float16)  # complex32 物理存储布局
        b = np.zeros((2, 3), dtype=np.float32)
        c = np.zeros(4, dtype=np.float16)
        schema = _build_input_schema(
            inputs=[a, b, c],
            input_names=["x", "y", "z"],
            input_dtypes=["complex32", "float32", "float16"],
        )
        assert schema[0]["dtype"] == "float16"  # 物理 dtype 不变
        assert schema[0]["logical_dtype"] == "complex32"
        assert schema[1]["logical_dtype"] == "float32"
        assert schema[2]["logical_dtype"] == "float16"

    def test_build_schema_logical_dtype_tensor_list_first_leaf(self):
        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema

        a, b = np.zeros(2, dtype=np.float16), np.zeros(2, dtype=np.float16)
        schema = _build_input_schema(
            inputs=[[a, b]],
            input_names=["x"],
            input_dtypes=[("complex32", "complex32")],  # TensorList 嵌套声明
        )
        assert schema[0]["logical_dtype"] == "complex32"
        # 无声明 → None；None slot / 尾部补齐 name 同样为 None
        schema2 = _build_input_schema(inputs=[None, a], input_names=["p", "q", "r"])
        assert schema2[0]["logical_dtype"] is None
        assert schema2[1]["logical_dtype"] is None
        assert schema2[2]["logical_dtype"] is None


class TestSerialize:
    def test_serialize_flattens_nested_and_skips_none(self):
        # npz 叶子数守卫（卡 merge corruption —— 原 bug 把 [a,b] 堆成 (2,1)）
        import os

        import numpy as np

        from ttk.remote.dispatcher import _serialize_to_file

        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        tmp = _serialize_to_file([[a, b], c, [d, e], None])
        try:
            npz = np.load(tmp)
            assert len(npz.files) == 5  # a0..a4,不是 3
            for i in range(5):
                assert npz[f"a{i}"].shape == (1,)  # 每个 (1,),不是 (2,1)
        finally:
            os.unlink(tmp)


class TestSchemaLeafCount:
    def test_count_equals_npz_and_schema(self):
        # 两路不变量—— _schema_leaf_count(schema) == npz 叶子数
        import os

        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema, _schema_leaf_count, _serialize_to_file

        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        inputs = [[a, b], c, [d, e], None]
        names = ["p0", "p1", "p2", "p3"]
        schema = _build_input_schema(inputs, names)
        npz_path = _serialize_to_file(inputs)
        try:
            npz_n = len(np.load(npz_path).files)
            assert _schema_leaf_count(schema) == npz_n == 5
        finally:
            os.unlink(npz_path)


class TestErrorHandling:
    def test_connection_refused(self):
        from ttk.remote.dispatcher import RemoteConnectionError, dispatch_to_remote

        with pytest.raises(RemoteConnectionError):
            dispatch_to_remote(
                op_name="test",
                inputs=[np.array([1.0])],
                endpoint_host="127.0.0.1",
                endpoint_port=19999,  # nobody listening
                tenant_id="err_test",
                timeout=2,
            )


class TestSyncSemaphore:
    def test_acquire_true_syncs_and_sets_ok(self, tmp_path):
        from ttk.remote import dispatcher

        class FakeCtx:
            def __init__(self):
                self.set_calls = []

            def acquire_semaphore(self, name):
                return True

            def set_semaphore(self, name, value):
                self.set_calls.append((name, value))

            def get_semaphore(self, name):
                raise AssertionError("should not poll when acquired")

        fake = FakeCtx()
        with patch("ttk.remote.dispatcher._do_http_sync", return_value=True) as do_sync, patch(
            "ttk.core_modules.common.process.process_context", return_value=fake
        ):
            ok = dispatcher._sync_missing_dependency("m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is True
        assert do_sync.called
        assert fake.set_calls == [("xpu_sync_127.0.0.1:19090:m", "ok")]

    def test_off_main_thread_syncs_directly_without_semaphore(self, tmp_path):
        """XPU 重叠的后台线程不做信号量 RPC（命令管道双读者互吞消息）——直接上传。"""
        import threading

        from ttk.remote import dispatcher

        class FakeCtx:
            def acquire_semaphore(self, name):
                raise AssertionError("background thread must not use semaphore RPC")

            def set_semaphore(self, name, value):
                raise AssertionError("background thread must not use semaphore RPC")

            def get_semaphore(self, name):
                raise AssertionError("background thread must not use semaphore RPC")

        result = {}
        with patch("ttk.remote.dispatcher._do_http_sync", return_value=True) as do_sync, patch(
            "ttk.core_modules.common.process.process_context", return_value=FakeCtx()
        ):
            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "ok", dispatcher._sync_missing_dependency("m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
                )
            )
            worker.start()
            worker.join(10)

        assert not worker.is_alive()
        assert result["ok"] is True
        assert do_sync.called


class TestClientHelpers:
    def test_backoff_delay_caps_and_jitters(self):
        from ttk.remote.dispatcher import _backoff_delay

        assert _backoff_delay(0, 0.5, 10.0, 0.25, lambda a, b: b) == 0.5 * 1.25
        assert _backoff_delay(5, 0.5, 10.0, 0.25, lambda a, b: b) == 10.0 * 1.25
        assert _backoff_delay(0, 0.5, 10.0, 0.25, lambda a, b: a) == 0.5 * 0.75


class _FakeResp:
    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self._h = headers or {}
        self._body = body

    def getheader(self, name, default=None):
        return self._h.get(name, default)

    def read(self):
        return self._body


class _ScriptedConn:
    script = []
    sent_modes = []

    def __init__(self, host, port, timeout=None):
        self.timeout = timeout

    def putrequest(self, *a, **k):
        pass

    def putheader(self, k, v):
        if k == "X-Mode":
            _ScriptedConn.sent_modes.append(v)

    def endheaders(self):
        pass

    def send(self, chunk):
        pass

    def getresponse(self):
        return _ScriptedConn.script.pop(0) if _ScriptedConn.script else _FakeResp(500)

    def close(self):
        pass


@pytest.fixture
def scripted(monkeypatch):
    _ScriptedConn.script = []
    _ScriptedConn.sent_modes = []
    monkeypatch.setattr("ttk.remote.dispatcher.http.client.HTTPConnection", _ScriptedConn)
    monkeypatch.setattr("ttk.remote.dispatcher.time.sleep", lambda *a, **k: None)
    return _ScriptedConn


def _npz_body(*arrs):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(arrs)})
    return buf.getvalue()


class TestDispatchBackoff:
    def test_503_then_200(self, scripted):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import dispatch_to_remote

        scripted.script = [
            _FakeResp(503),
            _FakeResp(503),
            _FakeResp(
                200,
                {"X-Output-Count": "1", "X-Output-Schema": json.dumps([{"index": 0, "dtype": "float64"}])},
                _npz_body(np.array([2.0])),
            ),
        ]
        out = dispatch_to_remote(
            op_name="add", inputs=[np.array([1.0])], input_names=["x"], mode=DATA, endpoint_port=9, tenant_id="t"
        )
        assert len(out) == 1

    def test_500_no_retry(self, scripted):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import RemoteExecutionError, dispatch_to_remote

        scripted.script = [_FakeResp(500, body=b"boom")]
        with pytest.raises(RemoteExecutionError):
            dispatch_to_remote(
                op_name="add", inputs=[np.array([1.0])], input_names=["x"], mode=DATA, endpoint_port=9, tenant_id="t"
            )
        assert len(scripted.script) == 0  # only one attempt; no retry


class TestRoundTrip:
    def test_nested_round_trips_through_match_params_v1(self):
        # 端到端 schema→serialize→server match_params_v1 结构对等
        # 未修代码上必红（p0 是 merged (2,1) 数组,不是 list）
        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema, _serialize_to_file
        from ttk.remote.server.executor import match_params_v1

        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        inputs = [[a, b], c, [d, e], None]
        names = ["p0", "p1", "p2", "p3"]
        schema = _build_input_schema(inputs, names)
        npz_path = _serialize_to_file(inputs)
        try:
            npz = np.load(npz_path)
            flat = [npz[k] for k in npz.files]
            named = match_params_v1(schema, flat)  # 真实 server 函数
            # p0/p2 是 list（不是 merged 数组）,p3 是 None
            assert isinstance(named["p0"], list)
            assert len(named["p0"]) == 2
            np.testing.assert_array_equal(named["p0"][0], a)
            np.testing.assert_array_equal(named["p0"][1], b)
            assert isinstance(named["p2"], list)
            assert len(named["p2"]) == 2
            assert named["p3"] is None
        finally:
            import os

            os.unlink(npz_path)


class TestParseClientMode:
    """X-Mode 客户端解析：data/data_perf/perf 全档映射，未知回 DATA。"""

    def test_mode_mapping(self):
        from ttk.remote import DATA, PERF
        from ttk.remote.dispatcher import _parse_client_mode

        assert _parse_client_mode("data") == DATA
        assert _parse_client_mode("data_perf") == DATA | PERF
        assert _parse_client_mode("perf") == PERF
        assert _parse_client_mode(None) == DATA
        assert _parse_client_mode(3) == 3  # int 直传


class TestPoolReuseTimeout:
    """连接池复用：复用连接的 socket timeout 对齐本次请求。"""

    def test_reused_connection_timeout_refreshed(self, monkeypatch):
        import ttk.remote.dispatcher as dispatcher_mod

        class _FakeSock:
            def __init__(self):
                self.timeouts = []

            def settimeout(self, value):
                self.timeouts.append(value)

        class _FakeConn:
            def __init__(self):
                self.sock = _FakeSock()

            def close(self):
                pass

        pool_conn = _FakeConn()
        monkeypatch.setattr(dispatcher_mod, "_CONN_POOL", {})
        monkeypatch.setattr(dispatcher_mod, "_create_connection", lambda h, p, timeout: pool_conn)

        conn1, reused1 = dispatcher_mod._acquire_connection("h", 1, timeout=300)
        assert reused1 is False

        dispatcher_mod._settle_connection("h", 1, conn1, reusable=True)
        conn2, reused2 = dispatcher_mod._acquire_connection("h", 1, timeout=10)
        assert reused2 is True
        assert pool_conn.sock.timeouts[-1] == 10  # 复用时被刷新为本请求 timeout


class TestProbeUsesTlsConfig:
    """恢复探活经 _create_connection（与 dispatch 同一 TLS 配置链路）。"""

    def test_probe_goes_through_tls_factory(self, monkeypatch):
        from types import SimpleNamespace

        from ttk.remote import client as client_mod
        from ttk.remote import dispatcher as dispatcher_mod

        created = []

        class _FakeResp:
            status = 200

        class _FakeConn:
            def request(self, method, path):
                created.append((method, path))

            def getresponse(self):
                return _FakeResp()

            def close(self):
                pass

        monkeypatch.setattr(dispatcher_mod, "_create_connection", lambda h, p, timeout: _FakeConn())
        ep = SimpleNamespace(host="h", port=1)
        ev = SimpleNamespace(_endpoints=[ep])

        assert client_mod._probe_endpoints_alive(ev) is True
        assert created == [("GET", "/v1/heartbeat")]
