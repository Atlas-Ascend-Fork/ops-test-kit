# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_init_device_locks 单元测试：覆盖 GPU/CPU/非连续 id 初始化及 stale 锁清理。"""

from types import SimpleNamespace

import pytest

from ttk.remote.server import xpu_server
from ttk.remote.server.xpu_server import _init_device_locks


def test_init_device_locks_empty_for_cpu():
    """_init_device_locks(["cpu"]) → _device_locks 为空。"""
    _init_device_locks(["cpu"])
    from ttk.remote.server import xpu_server

    assert xpu_server._device_locks == {}


# ══════════════════════════════════════════════════════════════════════════
# _resolve_devices：自动探测 use_device 跟随角色（cpu 兜底不置真）
# ══════════════════════════════════════════════════════════════════════════
class TestResolveDevicesCpuFallback:
    """自动探测：use_device 跟随探测角色（cpu 兜底为 False）。"""

    @pytest.mark.parametrize(
        ("detected", "expect_use_device"),
        [(("cpu", []), False), (("npu", [0, 1]), True)],
    )
    def test_auto_detect_use_device_follows_role(self, monkeypatch, detected, expect_use_device):
        monkeypatch.setattr(xpu_server, "detect_hardware", lambda cfg: detected)
        role, ids, use_device = xpu_server._resolve_devices(None, {"hardware_config": {}})
        assert use_device is expect_use_device
        if not expect_use_device:
            # cpu 分支不依赖 torch_lib 配置
            assert xpu_server._build_device_opts(SimpleNamespace(use_device=False), "cpu") == {"device_id": "cpu"}
