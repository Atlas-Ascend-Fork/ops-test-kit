# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TLS 测试：build_tls_connection 选择 HTTP/HTTPS、tls_from_config 配置提取与 gate、skip_verify。"""

import http.client
from unittest.mock import MagicMock

from ttk.remote.tls import build_tls_connection

# -- build_tls_connection ---------------------------------------------------


def test_build_https_for_cert_key_and_ca_only(monkeypatch):
    """cert+key → HTTPS（加载 cert+key）；CA-only → HTTPS（不加载 cert）。"""
    fake = MagicMock()
    monkeypatch.setattr("ssl.SSLContext", lambda *a, **k: fake)

    # cert + key + CA
    conn = build_tls_connection("x", 9090, 5, {"ca_cert": "ca", "cert": "c", "key": "k"})
    assert isinstance(conn, http.client.HTTPSConnection)
    fake.load_verify_locations.assert_called_once_with("ca")
    fake.load_cert_chain.assert_called_once_with("c", "k")

    # CA only（单向 TLS）
    fake.reset_mock()
    conn = build_tls_connection("x", 9090, 5, {"ca_cert": "ca"})
    assert isinstance(conn, http.client.HTTPSConnection)
    fake.load_verify_locations.assert_called_once_with("ca")
    fake.load_cert_chain.assert_not_called()
