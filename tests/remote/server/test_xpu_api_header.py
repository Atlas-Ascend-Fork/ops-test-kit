# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""X-API response header (Task 3): server writes the resolved api back on both
the success branch (_send_run_ok) and the error branch (_send_json).

This prevents the client from guessing whether the server actually resolved the
api it was asked to run — it just reads the X-API header. Covers:
  - success: env has api -> X-API header written
  - backward-compat: env api=None -> no X-API header (old envelopes still work)
  - error: _send_json writes X-API from env
"""

from unittest.mock import MagicMock

from ttk.remote.server.xpu_server import XpuRequestHandler


def _make_handler():
    """A bare XpuRequestHandler with mocked HTTP I/O (no socket / __init__)."""
    handler = XpuRequestHandler.__new__(XpuRequestHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    return handler


def test_send_run_ok_x_api_header():
    """_send_run_ok 在 api 非空时写 X-API 头。"""
    handler = _make_handler()
    env = {"output_path": None, "output_count": 0, "shapes": [], "dtypes": [], "perf": None, "api": "torch.add"}
    handler._send_run_ok(env)
    calls = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
    assert calls.get("X-API") == "torch.add"
