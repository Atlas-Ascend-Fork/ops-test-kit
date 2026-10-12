# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ProcessContext 消息契约（线上格式 + 同步 RPC 模式）与 process_context() 访问器。"""

import multiprocessing
import threading

from ttk.core_modules.common.process import ProcessContext, process_context
from ttk.core_modules.common.process.simple_command import PROCESS_RPC, PROCESS_STATUS_CODE


def test_process_context_returns_none_in_main_process():
    """访问器只在 worker 内有值——主进程（未跑 intermediate_func）为 None。"""
    assert process_context() is None


def test_report_status_sends_bare_status_enum():
    """状态上报是裸枚举消息（非 tuple），主进程 _update 靠 isinstance 分流。"""
    parent, child = multiprocessing.Pipe()
    ctx = ProcessContext(child)
    ctx.report_status(PROCESS_STATUS_CODE.IDLE)
    assert parent.recv() is PROCESS_STATUS_CODE.IDLE


def test_send_data_wire_format():
    """KV 存储消息格式：(STORE_DATA, (name, value))。"""
    parent, child = multiprocessing.Pipe()
    ctx = ProcessContext(child)
    ctx.send_data("stage", "OnProfiling")
    assert parent.recv() == (PROCESS_RPC.STORE_DATA, ("stage", "OnProfiling"))


def test_get_data_synchronous_roundtrip():
    """同步 RPC 模式：send 请求 → 阻塞 recv 等父侧回写。"""
    parent, child = multiprocessing.Pipe()
    ctx = ProcessContext(child)
    result = {}

    def _call():
        result["v"] = ctx.get_data("k")

    worker = threading.Thread(target=_call)
    worker.start()
    cmd, args = parent.recv()
    assert cmd is PROCESS_RPC.GET_DATA
    assert args == ("k",)
    parent.send(42)
    worker.join(timeout=5)
    assert result["v"] == 42
