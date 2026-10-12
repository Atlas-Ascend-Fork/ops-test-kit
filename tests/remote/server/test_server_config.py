# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Test that server config reads all fields from YAML, not env vars."""

import os
import tempfile

import yaml


def test_server_config_reads_all_fields(monkeypatch):
    """Server config should read all fields from YAML, not env vars."""
    # Ensure no env vars are set
    monkeypatch.delenv("TTK_XPU_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("TTK_XPU_GATE_WAIT_S", raising=False)
    monkeypatch.delenv("TTK_XPU_RUN_DEADLINE_S", raising=False)

    # Create a temporary YAML file with non-default values
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "server": {
                    "bind": "0.0.0.0",  # noqa: S104 测试数据：断言 YAML 配置读回
                    "port": 9999,
                    "max_concurrent": 20,
                    "run_deadline_s": 600,
                },
                "execution": {
                    "gate_wait_s": 5.0,
                },
                "storage": {
                    "sync_dir": "/custom/sync",
                    "tmp_dir": "/custom/tmp",
                },
            },
            f,
        )
        config_path = f.name

    try:
        from ttk.remote.server.config import load_server_config

        config = load_server_config(config_path)

        # Verify all fields are read from YAML
        assert config["bind"] == "0.0.0.0"  # noqa: S104 测试数据
        assert config["port"] == 9999
        assert config["max_concurrent"] == 20
        assert config["run_deadline_s"] == 600
        assert config["gate_wait_s"] == 5.0
        assert config["sync_dir"] == "/custom/sync"
        assert config["tmp_dir"] == "/custom/tmp"
    finally:
        os.unlink(config_path)
