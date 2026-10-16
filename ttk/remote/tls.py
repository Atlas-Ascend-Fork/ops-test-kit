#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared TLS config parsing + connection builder (dispatcher + heartbeat).

单一事实源：cert/key 成对校验（fail-loud）+ `ca or (cert and key)` HTTPS 门。

Stdlib-only (http.client + ssl) — shared leaf consumed by dispatcher and
instance_base via heartbeat; no ttk.* import to avoid cycles.
"""

import http.client
import ssl


def tls_from_config(config) -> dict:
    """RemoteConfig -> tls dict.

    cert/key 不成对则 raise（fail-loud）；无 TLS 配置返回 {}。
    """
    ca = getattr(config, "tls_ca", "") or ""
    cert = getattr(config, "tls_cert", "") or ""
    key = getattr(config, "tls_key", "") or ""
    skip_verify = bool(getattr(config, "tls_skip_verify", False))
    if (cert and not key) or (key and not cert):
        raise RuntimeError(f"TLS misconfiguration: tls_cert/tls_key must be paired (cert={cert!r}, key={key!r})")
    if not (ca or cert):
        return {}
    return {"ca_cert": ca, "cert": cert, "key": key, "skip_verify": skip_verify}


def build_tls_connection(host, port, timeout, tls):
    """tls dict -> HTTP(S)Connection.

    {} 或 None -> HTTP；ca 或 cert+key 成对 -> HTTPS。
    """
    ca = (tls or {}).get("ca_cert", "") or ""
    cert = (tls or {}).get("cert", "") or ""
    key = (tls or {}).get("key", "") or ""
    skip_verify = bool((tls or {}).get("skip_verify", False))
    if ca or (cert and key):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if skip_verify:
            # 内部 mTLS: 关 client 对 server 的 hostname 校验(server IP 多变); CA 签名验证仍防中间人
            ctx.check_hostname = False
        if ca:
            ctx.load_verify_locations(ca)
        else:
            # 无 tls_ca：加载默认信任根（SSL_CERT_FILE 等），空信任库必然验证失败
            ctx.load_default_certs()
        if cert and key:
            ctx.load_cert_chain(cert, key)
        return http.client.HTTPSConnection(host, port, timeout=timeout, context=ctx)
    return http.client.HTTPConnection(host, port, timeout=timeout)
