# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for /v1/sync endpoint."""

import base64
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

SYNC_TMP_DIR = os.path.join(tempfile.gettempdir(), "ttk_sync_test")


@pytest.fixture(scope="module")
def xpu_server():
    # Ensure a clean writable sync directory
    if os.path.exists(SYNC_TMP_DIR):
        shutil.rmtree(SYNC_TMP_DIR, ignore_errors=True)
    os.makedirs(SYNC_TMP_DIR, exist_ok=True)

    # Create a config file for the server
    import yaml

    tmp_dir = tempfile.mkdtemp(prefix="ttk_sync_tmp_test_")
    config_file = os.path.join(tempfile.gettempdir(), "xpu_server_sync_test.yaml")
    config_data = {
        "storage": {
            "sync_dir": SYNC_TMP_DIR,
            "tmp_dir": tmp_dir,
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server", "--port", "19091", "--dry-run", "--config", config_file],
        # 服务端输出丢弃, 不能用 PIPE: 没人读时管道缓冲区(64 KB)写满会让服务端
        # **永久阻塞**(框架预载一次就能吐 100 KB+), 表现为请求超时而非报错。
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    # Wait for server ready
    for _ in range(20):
        time.sleep(0.5)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 19091, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            break
        except (ConnectionRefusedError, OSError):
            continue
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    shutil.rmtree(SYNC_TMP_DIR, ignore_errors=True)


@pytest.fixture
def http_conn():
    conn = http.client.HTTPConnection("127.0.0.1", 19091, timeout=5)
    yield conn
    conn.close()


class TestSyncEndpoint:
    def test_sync_single_file(self, xpu_server, http_conn):
        content = base64.b64encode(b"print('hello')").decode()
        h = hashlib.sha256(b"print('hello')").hexdigest()
        body = json.dumps({"files": {"nn/demo.py": {"content": content, "hash": f"sha256:{h}"}}})
        http_conn.request(
            "POST", "/v1/sync", body=body, headers={"Content-Type": "application/json", "X-Tenant-ID": "sync_test_001"}
        )
        resp = http_conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["synced"] == 1
