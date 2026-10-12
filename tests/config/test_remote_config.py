# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""RemoteConfig dataclass 字段测试：backoff/retry/TLS 三组字段解析 + 默认值 + 全字段组合。"""

import pytest

from ttk.config.loader import RemoteConfig


def _cfg(**overrides):
    """构造 RemoteConfig，base 含 1 个 endpoint，overrides 覆盖额外字段。"""
    base = {"endpoints": [{"host": "127.0.0.1", "port": 9090}]}
    base.update(overrides)
    return RemoteConfig.from_dict(base)


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # backoff 三字段
        (
            {"backoff_base_s": 0.5, "backoff_max_s": 60.0, "backoff_jitter": 0.1},
            {"backoff_base_s": 0.5, "backoff_max_s": 60.0, "backoff_jitter": 0.1},
        ),
        # retry 三字段
        (
            {"max_503_retries": 15, "max_conn_retries": 8, "dispatch_deadline_s": 600},
            {"max_503_retries": 15, "max_conn_retries": 8, "dispatch_deadline_s": 600},
        ),
        # TLS 三字段
        (
            {"tls_ca": "/path/to/ca.pem", "tls_cert": "/path/to/cert.pem", "tls_key": "/path/to/key.pem"},
            {"tls_ca": "/path/to/ca.pem", "tls_cert": "/path/to/cert.pem", "tls_key": "/path/to/key.pem"},
        ),
    ],
    ids=["backoff", "retry", "tls"],
)
def test_remote_config_field_groups(fields, expected):
    """backoff / retry / TLS 三组字段从 yaml dict 正确解析到 RemoteConfig。"""
    config = _cfg(**fields)
    for key, val in expected.items():
        assert getattr(config, key) == val


def test_remote_config_defaults():
    """未指定字段时使用合理默认值（backoff/retry/TLS 全部有默认）。"""
    config = _cfg()
    assert config.backoff_base_s == 0.5
    assert config.backoff_max_s == 10.0
    assert config.backoff_jitter == 0.25
    assert config.max_503_retries == 10
    assert config.max_conn_retries == 5
    assert config.dispatch_deadline_s == 300
    assert config.tls_ca == ""
    assert config.tls_cert == ""
    assert config.tls_key == ""


def test_remote_config_all_fields_together():
    """全字段 + 多 endpoint 组合配置。"""
    config = RemoteConfig.from_dict(
        {
            "endpoints": [
                {"host": "127.0.0.1", "port": 9090},
                {"host": "192.168.1.1", "port": 8080},
            ],
            "backoff_base_s": 1.0,
            "backoff_max_s": 30.0,
            "backoff_jitter": 0.05,
            "max_503_retries": 20,
            "max_conn_retries": 10,
            "dispatch_deadline_s": 900,
            "tls_ca": "/etc/ssl/ca.crt",
            "tls_cert": "/etc/ssl/client.crt",
            "tls_key": "/etc/ssl/client.key",
        }
    )
    assert len(config.endpoints) == 2
    assert config.backoff_base_s == 1.0
    assert config.backoff_max_s == 30.0
    assert config.backoff_jitter == 0.05
    assert config.max_503_retries == 20
    assert config.max_conn_retries == 10
    assert config.dispatch_deadline_s == 900
    assert config.tls_ca == "/etc/ssl/ca.crt"
    assert config.tls_cert == "/etc/ssl/client.crt"
    assert config.tls_key == "/etc/ssl/client.key"
