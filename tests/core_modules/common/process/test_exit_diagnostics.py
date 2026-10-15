# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""worker 崩溃诊断单测：describe_exit 信号解码 + 真 worker 段错误端到端
（faulthandler/SIGTERM 装配探针 + SystemError 消息含信号名与成因提示）。"""

import multiprocessing
import os
import signal
import time

from ttk.core_modules.common.process import SimpleCommandProcess, describe_exit


class TestDescribeExit:
    def test_sigsegv_decoded_with_hint(self):
        s = describe_exit(-11)
        assert "SIGSEGV" in s
        assert "segfault" in s

    def test_sigkill_decoded_with_oom_hint(self):
        s = describe_exit(-9)
        assert "SIGKILL" in s
        assert "OOM" in s

    def test_sigterm_decoded_with_timeout_hint(self):
        s = describe_exit(-15)
        assert "SIGTERM" in s
        assert "proc-timeout" in s

    def test_positive_exit_not_mislabeled_as_signal(self):
        # 正退出码是普通 exit——不得误标为信号（exit 1 ≠ SIGHUP）
        assert describe_exit(1) == "exit code 1"
        assert describe_exit(255) == "exit code 255"

    def test_none_returns_empty(self):
        assert describe_exit(None) == ""


def _probe_then_segfault():
    """先回报崩溃诊断装配状态，再自杀（SIGSEGV）——同一 worker 两段探针。"""
    handler = signal.getsignal(signal.SIGTERM)
    import faulthandler

    return faulthandler.is_enabled(), callable(handler)


def _probe_process_context():
    from ttk.core_modules.common.process import process_context

    return process_context() is not None


def _self_segfault():
    os.kill(os.getpid(), signal.SIGSEGV)


def _raise_with_slow_teardown():
    """抛异常自退出，并在 atexit 清理链里滞留 1.5s——模拟真实 worker 的
    设备清理耗时，供 close() 的 join 行为判定。"""
    import atexit
    import time as _time

    atexit.register(_time.sleep, 1.5)
    raise RuntimeError("teardown probe")


class TestWorkerCrashDiagnostics:
    def test_segfault_worker_reports_signal_detail_and_resurrects(self):
        """真 worker 段错误端到端：SystemError 消息带信号名 + 成因提示；崩溃前
        faulthandler 已启用、SIGTERM 转储 handler 已装配；崩溃后 resurrect()
        原地重建可继续服务（错误恢复路径），且 all_processes 不重复登记。"""
        proc = SimpleCommandProcess(multiprocessing.get_context("forkserver"), name="CrashProbe")
        try:
            proc.send_action(_probe_then_segfault, (), {})
            proc.send_action(_self_segfault, (), {})
            results = []
            deadline = time.time() + 60
            while len(results) < 2 and time.time() < deadline:
                proc.update()
                if proc.is_completed():
                    results.append(proc.get_result())
                else:
                    time.sleep(0.05)
            probe, crash = results
            assert probe == (True, True)  # faulthandler 启用 + SIGTERM handler 装配
            assert isinstance(crash, SystemError)
            assert "SIGSEGV" in str(crash)
            assert "segfault" in str(crash)

            proc.resurrect()
            assert SimpleCommandProcess.all_processes.count(proc) == 1  # 复活不重复登记
            assert not proc.is_dead()
            proc.send_action(_probe_process_context, (), {})
            deadline = time.time() + 60
            while not proc.is_completed() and time.time() < deadline:
                proc.update()
                time.sleep(0.05)
            assert proc.get_result() is True  # 重建后可继续执行 task

            # 异常自退出路径：worker 上报 RuntimeError + DEAD 后走 atexit 清理链；
            # close() 须按实际存活 join 到真实退出，不得被上报的 DEAD 状态短路
            proc.send_action(_raise_with_slow_teardown, (), {})
            deadline = time.time() + 60
            while not proc.is_completed() and time.time() < deadline:
                proc.update()
                time.sleep(0.05)
            assert isinstance(proc.get_result(), RuntimeError)
            started = time.time()
            proc.close()  # join 到真实退出后 parent.close，重复调用须幂等
            assert proc.is_dead()
            waited = time.time() - started
            assert 1.0 <= waited < 15  # 等到了清理链且未触顶宽限
        finally:
            proc.close()
