#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# mTLS 证书管理 for TTK xpu-server。
#
# 三个操作（CA 建一次；server 证书每个 XPU 一次；client 证书每个租户按需），
# 都由 CA 持有者在 CA 机器上运行（都需要 ca.key）：
#
#   ./gen_tls_certs.sh init-ca <CERTDIR>
#       一次。建 CA（ca.key + ca.crt）。ca.key 是私钥、签一切，务必保密。
#       若 <CERTDIR>/ca.crt 已存在则拒绝（不覆盖既有 CA）。
#
#   ./gen_tls_certs.sh server <CERTDIR> <SAN> [<SAN> ...]
#       每个 XPU 一次。用 CA 签 server.crt/server.key。
#       <SAN> = client 实际连接的目标主机，写成 "IP:1.2.3.4" 或 "DNS:host"，必填
#       （TLS client 拿连接 host 对 SAN；不再硬编码 127.0.0.1）。
#       NPU->PC->XPU 桥接场景：传 NPU 在 remote.endpoints 里填的那个 host（如 PC 的 100 网段 IP）。
#
#   ./gen_tls_certs.sh client <CERTDIR> <tenant-name>
#       每个租户按需。用 CA 签 <tenant-name>.client.crt/.key。
#       把 ca.crt + <tenant>.client.crt + <tenant>.client.key 发给该租户。
#       XPU server 不需要这个证书（它信 CA）—— 不用拷到 XPU、不用重启 server。
#
# ca.crt/ca.key 存在 <CERTDIR>；server/client 从那里读 CA。
#
# 【安全】ca.key（CA 私钥）只能留在 CA 机器上，长期保留（轮换叶子证书要靠它），
#        绝不拷贝到任何 client 或 server。发给 client/server 的只有：ca.crt + 各自的证书/私钥。
set -euo pipefail

DAYS="${TTK_CERT_DAYS:-90}"   # 默认 90 天（CA 与叶子证书同寿命）；要更长用 TTK_CERT_DAYS=N 覆盖

die() { echo "error: $*" >&2; exit 1; }

usage() { sed -n '2,24p' "$0"; exit 1; }

[ $# -ge 1 ] || usage
OP="$1"; shift

case "$OP" in
    init-ca)
        [ $# -ge 1 ] || die "usage: $0 init-ca <CERTDIR>"
        D="$1"
        mkdir -p "$D"
        [ -f "$D/ca.crt" ] && die "$D/ca.crt 已存在 —— 拒绝覆盖既有 CA。换个空目录或先移走它。"
        openssl req -x509 -newkey rsa:4096 -keyout "$D/ca.key" -out "$D/ca.crt" \
            -days "$DAYS" -nodes -subj "/CN=TTK-CA"
        chmod 600 "$D/ca.key"
        echo "CA 已建于 $D/"
        echo "【ca.key 只留本机、勿删勿拷】ca.key 只能留在这台 CA 机器上（轮换叶子证书要靠它），绝不发往任何 client/server。"
        ;;

    server)
        [ $# -ge 2 ] || die "usage: $0 server <CERTDIR> <SAN> [<SAN> ...]   （SAN = IP:x.x.x.x 或 DNS:name，必填）"
        D="$1"; shift
        [ -f "$D/ca.crt" ] && [ -f "$D/ca.key" ] || die "$D 下找不到 CA —— 先跑 '$0 init-ca $D'"
        SAN="$(IFS=,; echo "$*")"                       # 剩余参数拼 subjectAltName
        [ -n "$SAN" ] || die "至少一个 SAN（IP:x.x.x.x 或 DNS:name）"
        printf "subjectAltName=%s\n" "$SAN" > "$D/server.ext"
        openssl req -newkey rsa:2048 -keyout "$D/server.key" -out "$D/server.csr" \
            -nodes -subj "/CN=xpu-server"
        openssl x509 -req -in "$D/server.csr" -CA "$D/ca.crt" -CAkey "$D/ca.key" \
            -CAcreateserial -out "$D/server.crt" -days "$DAYS" -extfile "$D/server.ext"
        rm -f "$D/server.csr" "$D/server.ext"
        chmod 600 "$D/server.key"
        echo "Server 证书在 $D/  （server.crt/server.key，SAN=$SAN）"
        echo "装到 XPU server：ca.crt + server.crt + server.key（ca.key 留 CA 机，不发）。"
        ;;

    client)
        [ $# -ge 2 ] || die "usage: $0 client <CERTDIR> <tenant-name>"
        D="$1"; T="$2"
        [ -f "$D/ca.crt" ] && [ -f "$D/ca.key" ] || die "$D 下找不到 CA —— 先跑 '$0 init-ca $D'"
        openssl req -newkey rsa:2048 -keyout "$D/$T.client.key" -out "$D/$T.client.csr" \
            -nodes -subj "/CN=$T"
        openssl x509 -req -in "$D/$T.client.csr" -CA "$D/ca.crt" -CAkey "$D/ca.key" \
            -CAcreateserial -out "$D/$T.client.crt" -days "$DAYS"
        rm -f "$D/$T.client.csr"
        chmod 600 "$D/$T.client.key"
        echo "租户 '$T' 的 client 证书在 $D/  （$T.client.crt/$T.client.key）"
        echo "发给该租户：ca.crt + $T.client.crt + $T.client.key（ca.key 永不发）。server 无需这些（它信 CA）。"
        ;;

    *)
        usage
        ;;
esac
