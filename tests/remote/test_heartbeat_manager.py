# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""HeartbeatManager 测试：start（env/路径/idempotent/进程名/tls）、stop（终止/clear/remove/safe）、
supervise（alive noop/dead respawn/no process noop）。"""

import os


def _fake_brief_loop(**kwargs):
    """模块级短 sleep target（forkserver pickle 要求模块级）。"""
    import time

    time.sleep(5)


# -- start ------------------------------------------------------------------


def test_start_sets_health_path_and_idempotent(tmp_path):
    """start: 设置 TTK_XPU_HEALTH_PATH（含 .ttk + tenant_id）；二次 start 不创建新进程。"""
    from ttk.remote.heartbeat_manager import HeartbeatManager

    mgr = HeartbeatManager(
        heartbeat_target=_fake_brief_loop,
        root_path=str(tmp_path),
        tenant_id="tid123",
        endpoints=[],
    )
    mgr.start()
    try:
        assert ".ttk/xpu_health_tid123.json" in os.environ["TTK_XPU_HEALTH_PATH"]
        proc1 = mgr._process
        mgr.start()  # idempotent
        assert mgr._process is proc1
    finally:
        mgr.stop()
        os.environ.pop("TTK_XPU_HEALTH_PATH", None)
