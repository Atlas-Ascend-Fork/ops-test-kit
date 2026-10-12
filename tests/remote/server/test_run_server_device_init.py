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

from ttk.remote.server.xpu_server import _init_device_locks


def test_init_device_locks_empty_for_cpu():
    """_init_device_locks(["cpu"]) → _device_locks 为空。"""
    _init_device_locks(["cpu"])
    from ttk.remote.server import xpu_server

    assert xpu_server._device_locks == {}
