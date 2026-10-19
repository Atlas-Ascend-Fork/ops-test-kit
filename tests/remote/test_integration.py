# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Full-pipeline integration tests using xpu_server --dry-run.

Validates end-to-end flow: config -> heartbeat -> dispatch -> verify -> cleanup.
Uses port 19094 to avoid conflicts with other test suites.
"""

import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

# Isolated sync directory so this suite does not clash with test_sync.py
_INTEGRATION_SYNC_DIR = os.path.join(tempfile.gettempdir(), "ttk_sync_integration")


@pytest.fixture(scope="module")
def xpu_server():
    # Clean sync directory
    if os.path.exists(_INTEGRATION_SYNC_DIR):
        shutil.rmtree(_INTEGRATION_SYNC_DIR, ignore_errors=True)
    os.makedirs(_INTEGRATION_SYNC_DIR, exist_ok=True)

    # Create a config file for the server
    import yaml

    tmp_dir = tempfile.mkdtemp(prefix="ttk_tmp_integration_")
    config_file = os.path.join(tempfile.gettempdir(), "xpu_server_integration.yaml")
    config_data = {
        "storage": {
            "sync_dir": _INTEGRATION_SYNC_DIR,
            "tmp_dir": tmp_dir,
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server", "--port", "19094", "--dry-run", "--config", config_file],
        # 服务端输出丢弃, 不能用 PIPE: 没人读时管道缓冲区(64 KB)写满会让服务端
        # **永久阻塞**(框架预载一次就能吐 100 KB+), 表现为请求超时而非报错。
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    # Wait for server readiness
    for _ in range(20):
        time.sleep(0.5)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 19094, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
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
    shutil.rmtree(_INTEGRATION_SYNC_DIR, ignore_errors=True)


class TestFullPipeline:
    """End-to-end: config -> heartbeat -> dispatch -> verify -> cleanup."""

    def test_api_mode_full_flow(self, xpu_server):
        """API mode: config -> heartbeat -> /run -> verify -> cleanup."""
        from ttk.remote.config import RemoteConfig
        from ttk.remote.dispatcher import dispatch_to_remote

        # 1. Config
        RemoteConfig.from_dict({"endpoints": [{"host": "127.0.0.1", "port": 19094}]})
        tenant_id = "integration_test_001"

        # 2. Heartbeat
        conn = http.client.HTTPConnection("127.0.0.1", 19094, timeout=5)
        conn.request("GET", f"/v1/heartbeat?tenant_id={tenant_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()

        # 3. Dispatch
        inputs = [np.random.randn(8, 16).astype(np.float32)]
        outputs = dispatch_to_remote(
            op_name="softmax_v2",
            inputs=inputs,
            provider="torch",
            attrs={"axis": -1},
            endpoint_host="127.0.0.1",
            endpoint_port=19094,
            tenant_id=tenant_id,
        )

        # 4. Verify
        assert len(outputs) >= 1
        assert isinstance(outputs[0], np.ndarray)

        # 5. Cleanup
        conn.request("DELETE", f"/v1/tenant/{tenant_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["cleaned"] is True
        conn.close()
