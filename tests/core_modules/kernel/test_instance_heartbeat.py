# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for KernelInstance heartbeat lifecycle."""

from unittest.mock import MagicMock


class TestInstanceBaseHeartbeat:
    def test_run_starts_heartbeat_before_setup(self):
        """run() must call _start_heartbeat_process BEFORE setup,
        and AFTER _parse_testcases (and after validate_only early-return)."""
        from ttk.core_modules.kernel.instance import KernelInstance

        ins = KernelInstance()
        ins.switches.device_count = 1
        ins.switches.process_per_device = 1
        ins.switches.test_mode = "op"
        ins.switches.validate_only = False

        manager = MagicMock()
        # Wire run()'s collaborators to the same mock so we can assert call order.
        # Each step is a distinct child-mock; manager.mock_calls preserves order.
        ins.prepare_env = manager.prepare_env
        ins.detect_platform = manager.detect_platform
        ins.detect_device_count = manager.detect_device_count
        ins.parse_cases = manager.parse_cases
        ins._start_heartbeat_process = manager._start_heartbeat_process
        ins.setup = manager.setup
        ins.open_output = manager.open_output
        ins.prepare_workers = manager.prepare_workers
        ins.create_tasks = manager.create_tasks
        ins.run_loop = manager.run_loop
        ins.cleanup = manager.cleanup
        # make the main loop exit immediately
        ins.total_case_count = 1
        ins.completed_case_count = 1

        ins.run()

        names = [c[0] for c in manager.mock_calls]
        # _start_heartbeat_process must come before setup
        assert "_start_heartbeat_process" in names
        assert "setup" in names
        assert names.index("_start_heartbeat_process") < names.index("setup")
        # and it must come after parse_cases
        assert names.index("parse_cases") < names.index("_start_heartbeat_process")
        # 模板方法阶段序：探测 → 解析 → setup → open_output →
        # prepare_workers → create_tasks → run_loop → cleanup
        order = [
            "prepare_env",
            "detect_platform",
            "detect_device_count",
            "parse_cases",
            "_start_heartbeat_process",
            "setup",
            "open_output",
            "prepare_workers",
            "create_tasks",
            "run_loop",
            "cleanup",
        ]
        positions = [names.index(n) for n in order]
        assert positions == sorted(positions), names

    def test_close_subprocesses_stops_heartbeat(self):
        from ttk.core_modules.kernel.instance import KernelInstance

        ins = KernelInstance()
        mock_mgr = MagicMock()
        ins.heartbeat_manager = mock_mgr
        ins.process_groups = {}

        ins.close_subprocesses()

        mock_mgr.stop.assert_called_once()
