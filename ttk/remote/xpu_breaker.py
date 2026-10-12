#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""三方采集连续失败熔断器。

三方采集失败在 collector 里被收成 ``{"status": "FAIL", ...}`` 后跑批照常推进 ——
对**偶发**抖动这是对的。但配置性故障(免传输打到不支持的服务端、端点整段不可
用、服务端缺 spec 依赖)每一例都会失败，降级就成了跑满全程产出空数据：判据带
``--compare close`` 的档甚至照报 PASS，只有事后数 ``device_us`` 才看得出来。

本模块只负责区分这两者：**连续**失败到阈值即熔断，中间成功一例就清零。
用例跑在各自的 worker 进程里，故状态落在 ``<root>/.ttk/xpu_breaker.json``，
用 flock 串行化读改写。
"""

import fcntl
import json
import logging
import os

# 阈值要大于"端点自愈期内可能失败的用例数"。客户端本身有 ENDPOINT_RECOVERY_WAIT_S(默认 180s)
# 的端点自愈等待, 隧道瞬断→重连期间会有一批用例先失败; 阈值取 5 时熔断先掐断,
# 自愈根本来不及生效(实测: 隧道瞬断后 mml/marndq 各只回收 4~5 条, 而端点几秒后就恢复了)。
# 取 30: 既能挡住"配置级失效"(整轮都连不上, 30 例后必然触发), 又不会被瞬断误杀。
DEFAULT_LIMIT = 30
_STATE_NAME = "xpu_breaker.json"


class XpuLegBrokenError(RuntimeError):
    """三方采集连续失败到阈值 —— 这一轮的三方数据不可用, 跑批以非零码退出。"""


def limit() -> int:
    """阈值：环境变量 TTK_XPU_FAIL_LIMIT 覆盖，<=0 表示关闭熔断。"""
    raw = os.environ.get("TTK_XPU_FAIL_LIMIT")
    if raw is None or raw.strip() == "":
        return DEFAULT_LIMIT
    try:
        return int(raw)
    except ValueError:
        logging.warning("TTK_XPU_FAIL_LIMIT=%r is not an integer, using default %d", raw, DEFAULT_LIMIT)
        return DEFAULT_LIMIT


def _state_path(root_path: str) -> str:
    return os.path.join(root_path, ".ttk", _STATE_NAME)


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def reset(root_path: str) -> None:
    """跑批开始时清空状态, 避免沿用上一轮的计数。"""
    path = _state_path(root_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"consec_fail": 0, "total_fail": 0, "total_ok": 0, "tripped": False, "reason": ""}, f)


def record(root_path: str, ok: bool, reason: str = "") -> None:
    """记一例三方采集结果。ok=False 累加连续计数, ok=True 清零。

    只在三方采集**被请求过**的用例上调用(见 client.dispatch_xpu)；没有三方采集的
    跑批不会进这里, 因此不存在"没配三方也熔断"的问题。
    """
    if limit() <= 0:
        return
    path = _state_path(root_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            st = _read(path)
            if ok:
                st["consec_fail"] = 0
                st["total_ok"] = st.get("total_ok", 0) + 1
            else:
                st["consec_fail"] = st.get("consec_fail", 0) + 1
                st["total_fail"] = st.get("total_fail", 0) + 1
                st["reason"] = reason
                if st["consec_fail"] >= limit():
                    st["tripped"] = True
            with open(path, "w", encoding="utf-8") as f:
                json.dump(st, f)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def tripped(root_path: str) -> tuple:
    """返回 (是否已熔断, 最后一条失败原因)。从未记录过则 (False, "")。"""
    if limit() <= 0:
        return False, ""
    st = _read(_state_path(root_path))
    if not st.get("tripped"):
        return False, ""
    return True, (
        f"third-party leg failed {st.get('consec_fail', 0)} case(s) in a row "
        f"(total fail {st.get('total_fail', 0)} / ok {st.get('total_ok', 0)}), "
        f"last reason: {st.get('reason', '')}"
    )
