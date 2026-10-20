#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit tests for the NPUSim simulator backend (no device / simulation needed)."""

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ttk.cli.sim_args import apply_sim_args
from ttk.switches import MODE, SWITCHES


class _DummyRTSProfilingParam:
    """Minimal stand-in for RTSProfilingParam (module-level: picklable)."""

    switch = True
    compile_result = "SUCC"
    is_valid = True
    fail_reason = ""
    block_dim = 1

    def clear_atomic_output_workspace(self):
        pass


def _sim_args(**overrides):
    """构造一份带默认 npusim 字段的 SimpleNamespace，可按需覆盖。"""
    base = {
        "backend": "npusim",
        "sim_soc_version": "Ascend950",
        "sim_output_dir": None,
        "sim_report": False,
        "sim_cores": None,
        "sim_object_file": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestSimArgs:
    """apply_sim_args 对 backend/soc/no_prof 等参数的规范化与校验。"""

    def test_default_backend_is_npu(self):
        sw = SWITCHES()
        assert sw.backend == "npu"

    def test_npusim_backend_normalizes_switches(self):
        sw = SWITCHES()
        apply_sim_args(sw, _sim_args())
        assert sw.backend == "npusim"
        assert sw.mode == MODE.ASCEND_CAMODEL
        # SoC name maps to the platform ini used by get_npu_hw_info.
        assert sw.dev_plat == "Ascend950PR_9589"
        assert sw.short_soc_version is None  # filled later by get_device_platform
        assert sw.warmup is False
        assert sw.task_profiling is False
        assert sw.deterministic_level == 0
        assert sw.process_per_device == 1
        assert sw.sim_output_dir == str(Path(sw.root_path) / "sim_output")

    def test_unknown_soc_passthrough(self):
        sw = SWITCHES()
        apply_sim_args(sw, _sim_args(sim_soc_version="Ascend999"))
        assert sw.dev_plat == "Ascend999"

    def test_npu_backend_leaves_mode_untouched(self):
        sw = SWITCHES()
        apply_sim_args(sw, _sim_args(backend="npu"))
        assert sw.mode == MODE.ASCEND_ONBOARD
        assert sw.dev_plat == "AUTO"

    def test_no_prof_rejected(self):
        sw = SWITCHES()
        with pytest.raises(ValueError, match="--backend npusim"):
            apply_sim_args(sw, _sim_args(no_prof=True))


class TestSwitchPickle:
    """SWITCHES 带 sim 字段可 pickle 往返，非 sim 字段同样保留。"""

    def test_pickle_roundtrip_with_sim_fields(self):
        sw = SWITCHES()
        sw.backend = "npusim"
        sw.sim_soc_version = "Ascend950"
        sw.sim_report = True
        restored = pickle.loads(pickle.dumps(sw))
        assert restored.backend == "npusim"
        assert restored.sim_soc_version == "Ascend950"
        assert restored.sim_report is True
        # Non-sim fields survive too.
        assert restored.deterministic_level == sw.deterministic_level


class TestCaseWriter:
    """case_writer 的安全目录名生成与启用的 kernel 模式集合。"""

    def test_case_dir_uses_safe_name(self):
        from ttk.core_modules.common.simulator import case_writer

        sw = SWITCHES()
        sw.sim_output_dir = "/tmp/ttk_sim_out"
        d = case_writer.case_dir(sw, "aclnnAdd/00 x")
        assert str(d) == "/tmp/ttk_sim_out/aclnnAdd_00_x" or "-" in str(d)

    def test_enabled_kernel_modes(self):
        from ttk.core_modules.common.simulator import case_writer

        sw = SWITCHES()
        # dyn enabled+prof on by default; cst/bin disabled.
        assert case_writer.enabled_kernel_modes(sw) == ("dyn",)
        sw.cst_switches.enabled = True
        assert case_writer.enabled_kernel_modes(sw) == ("dyn", "cst")


class TestWrapperGeneration:
    """kernel/aclnn wrapper 脚本生成：含 KernelExecutor/AclOpExecutor、os._exit、skip_teardown。"""

    @pytest.fixture
    def sw(self, tmp_path):
        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        return sw

    def test_kernel_wrapper_generated(self, sw):
        from ttk.core_modules.common.simulator import case_writer
        from ttk.core_modules.common.simulator.wrapper import write_kernel_wrapper

        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_kernel_wrapper(sw, case_path)
        assert wrapper.is_file()
        assert wrapper.name == "caseA_kernel.py"
        text = wrapper.read_text()
        assert "KernelExecutor" in text
        assert "os._exit(0)" in text
        # camodel teardown (rtCtxDestroy/rtDeviceReset) busy-spins on some cases;
        # the wrapper skips it and lets os._exit reclaim the process.
        assert "skip_teardown=True" in text

    def test_aclnn_wrapper_generated(self, sw):
        from ttk.core_modules.common.simulator import case_writer
        from ttk.core_modules.common.simulator.wrapper import write_aclnn_wrapper

        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_aclnn_wrapper(sw, case_path)
        assert wrapper.is_file()
        assert wrapper.name == "caseA_aclnn.py"
        text = wrapper.read_text()
        assert "AclOpExecutor" in text
        assert "os._exit(0)" in text
        assert "skip_teardown=True" in text


class TestResultLoading:
    """结果加载：kernel 与 aclnn 各自覆盖成功/失败/损坏/缺失+wrapper 错误四态。"""

    @pytest.mark.parametrize(
        "scenario",
        ["ok", "failure", "corrupt", "missing_with_wrapper_error"],
    )
    def test_load_mode_result(self, tmp_path, scenario):
        """_load_mode_result 解析 kernel dyn 结果的四类场景。"""
        from ttk.core_modules.common.simulator.sim_profiling import _load_mode_result
        from ttk.core_modules.kernel.result import KernelExecuteResult

        mdir = tmp_path / "dyn"
        if scenario == "ok":
            mdir.mkdir()
            (mdir / "result.json").write_text(json.dumps({"ok": True, "cycle": "UNKNOWN", "oob": "OK,OK"}))
            (mdir / "output_0.bin").write_bytes(b"\x00\x01")
            (mdir / "output_1.bin").write_bytes(b"\x02\x03")
            res = _load_mode_result(tmp_path, "dyn")
            assert isinstance(res, KernelExecuteResult)
            assert res.output_bytes == [b"\x00\x01", b"\x02\x03"]
            assert res.oob == "OK,OK"
        elif scenario == "failure":
            mdir.mkdir()
            (mdir / "result.json").write_text(json.dumps({"ok": False, "error": "SIM_EXECUTION_FAILED"}))
            res = _load_mode_result(tmp_path, "dyn")
            assert isinstance(res, KernelExecuteResult)
            assert res.cycle == "SIM_EXECUTION_FAILED"
        elif scenario == "corrupt":
            # A truncated result.json surfaces as SIM_RESULT_CORRUPT, not a crash.
            mdir.mkdir()
            (mdir / "result.json").write_text('{"ok": true, "cyc')  # truncated json
            res = _load_mode_result(tmp_path, "dyn")
            assert res.cycle.startswith("SIM_RESULT_CORRUPT")
        else:
            # A wrapper crash must surface its traceback, not a bare MISSING.
            (tmp_path / "dyn").mkdir()
            (tmp_path / "wrapper_error.json").write_text(
                'Traceback (most recent call last):\n  File "wrapper.py", line 12, in main\nRuntimeError: boom'
            )
            res = _load_mode_result(tmp_path, "dyn")
            assert res.cycle.startswith("SIM_RESULT_MISSING")
            assert "RuntimeError: boom" in res.cycle

    @pytest.mark.parametrize(
        "scenario",
        ["ok", "missing", "corrupt", "missing_with_wrapper_error"],
    )
    def test_load_aclnn_result(self, tmp_path, scenario):
        """_load_aclnn_result 解析 aclnn 结果的四类场景。"""
        from ttk.core_modules.aclnn.result import ApiProfilingResult
        from ttk.core_modules.common.simulator.sim_profiling import _load_aclnn_result

        if scenario == "ok":
            (tmp_path / "final_result.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "api_prof": "UNKNOWN",
                        "op_prof": "TOTAL_CYCLE_TODO",
                        "oob": "UNKNOWN",
                        "deterministic_status": None,
                    }
                )
            )
            (tmp_path / "output_0.bin").write_bytes(b"\x00\x01")
            with (tmp_path / "output_view_shapes.pkl").open("wb") as f:
                pickle.dump([("float32", [1, 2])], f)
            res = _load_aclnn_result(tmp_path)
            assert isinstance(res, ApiProfilingResult)
            assert res.success is True
            assert res.output_bytes == [b"\x00\x01"]
            assert res.output_view_shapes == (("float32", [1, 2]),)
            # json null round-trips to None (not the literal string "None").
            assert res.deterministic_status is None
        elif scenario == "missing":
            res = _load_aclnn_result(tmp_path)
            assert isinstance(res, ApiProfilingResult)
            assert res.success is False
            assert "SIM_RESULT_MISSING" in str(res.api_prof)
        elif scenario == "corrupt":
            (tmp_path / "final_result.json").write_text('{"ok": true')  # truncated json
            res = _load_aclnn_result(tmp_path)
            assert isinstance(res, ApiProfilingResult)
            assert res.failed()
            assert "SIM_RESULT_CORRUPT" in str(res.api_prof)
        else:
            (tmp_path / "wrapper_error.json").write_text("RuntimeError: boom")
            res = _load_aclnn_result(tmp_path)
            assert isinstance(res, ApiProfilingResult)
            assert res.failed()
            assert "RuntimeError: boom" in str(res.api_prof)


class TestSkipTeardown:
    """skip_teardown guards the camodel teardown busy-spin (rtCtxDestroy)."""

    def test_kernel_wrapper_skips_teardown(self, tmp_path):
        from ttk.core_modules.common.simulator import case_writer
        from ttk.core_modules.common.simulator.wrapper import write_kernel_wrapper

        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_kernel_wrapper(sw, case_path)
        assert "skip_teardown=True" in wrapper.read_text()

    def test_aclnn_wrapper_skips_teardown(self, tmp_path):
        from ttk.core_modules.common.simulator import case_writer
        from ttk.core_modules.common.simulator.wrapper import write_aclnn_wrapper

        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_aclnn_wrapper(sw, case_path)
        assert "skip_teardown=True" in wrapper.read_text()

    def test_rts_skip_teardown_noop(self):
        """skip_teardown=True: destroy_context/reset skip api_call entirely."""
        from ttk.core_modules.kernel.runtime.rts_interface import RTSInterface

        dev = object.__new__(RTSInterface)  # bypass ctypes.CDLL load in __init__
        dev.skip_teardown = True
        dev.context = None
        dev.context_storage = []
        dev.device_id = 0
        calls = []
        dev.api_call = lambda *a, **k: calls.append(a)
        dev.destroy_context()
        dev.reset()
        assert calls == []

    def test_rts_teardown_still_active_by_default(self):
        """skip_teardown=False keeps the original validation before api_call."""
        from ttk.core_modules.kernel.runtime.rts_interface import RTSInterface

        dev = object.__new__(RTSInterface)
        dev.skip_teardown = False
        dev.context = None
        dev.context_storage = []
        # Attributes __del__->reset() touches when the object is GC'd.
        dev.device_id = None
        dev.kernel_binary_storage = {}
        dev.memory_manager = {}
        dev.size_info_storage = []
        calls = []
        dev.api_call = lambda *a, **k: calls.append(a)
        with pytest.raises(ValueError, match="context"):
            dev.destroy_context()  # context not in storage -> raises, no api_call
        assert calls == []

    def test_aclnn_interface_passes_skip_teardown(self, monkeypatch):
        import ttk.core_modules.aclnn.runtime as acl_mod

        captured = {}

        class FakeRTS:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def reset(self, *a, **k):
                pass

            def destroy_context(self, *a, **k):
                pass

        monkeypatch.setattr(acl_mod, "RTSInterface", FakeRTS)
        monkeypatch.setattr(acl_mod.ctypes, "CDLL", lambda *a, **k: object())
        monkeypatch.setattr(acl_mod.AclInterface, "_on_exit", lambda self: None)
        acl_mod.AclInterface("Ascend950", True, skip_teardown=True)
        assert captured.get("skip_teardown") is True

    def test_aclnn_reset_skipped_when_skip_teardown(self, monkeypatch):
        """skip_teardown=True: AclInterface.reset() must not call
        _acl_reset_device() (aclrtResetDevice can busy-spin on camodel)."""
        import ttk.core_modules.aclnn.runtime as acl_mod

        calls = []

        class FakeRTS:
            def __init__(self, **kwargs):
                pass

            def reset(self, *a, **k):
                calls.append("rts_reset")

        monkeypatch.setattr(acl_mod, "RTSInterface", FakeRTS)
        monkeypatch.setattr(acl_mod.ctypes, "CDLL", lambda *a, **k: object())
        monkeypatch.setattr(acl_mod.AclInterface, "_on_exit", lambda self: None)
        dev = acl_mod.AclInterface("Ascend950", True, skip_teardown=True)
        dev._owns_acl_runtime = True
        dev._acl_reset_device = lambda: calls.append("acl_reset_device")
        dev.reset()
        assert calls == []  # neither aclrtResetDevice nor rts reset is issued

    def test_aclnn_reset_still_active_by_default(self, monkeypatch):
        """skip_teardown=False keeps the original aclrtResetDevice path."""
        import ttk.core_modules.aclnn.runtime as acl_mod

        calls = []

        class FakeRTS:
            def __init__(self, **kwargs):
                pass

            def reset(self, *a, **k):
                calls.append("rts_reset")

        monkeypatch.setattr(acl_mod, "RTSInterface", FakeRTS)
        monkeypatch.setattr(acl_mod.ctypes, "CDLL", lambda *a, **k: object())
        monkeypatch.setattr(acl_mod.AclInterface, "_on_exit", lambda self: None)
        dev = acl_mod.AclInterface("Ascend950", True)
        dev._owns_acl_runtime = True
        dev._acl_reset_device = lambda: calls.append("acl_reset_device")
        dev.reset()
        assert "acl_reset_device" in calls
        assert "rts_reset" in calls


class TestCannsimBackend:
    """record/report run the CANN-bundled cannsim, not the repo source."""

    @pytest.mark.parametrize(
        "scenario",
        ["cannsim_prefix", "npusim_prefix", "picks_newest", "empty_raises"],
    )
    def test_latest_record_dir(self, tmp_path, scenario):
        """_latest_record_dir 匹配 cannsim_/npusim_ 前缀；选最新；空目录报错。"""
        from ttk.core_modules.common.simulator.npusim_runner import _latest_record_dir

        if scenario == "cannsim_prefix":
            (tmp_path / "cannsim_20260810_abc").mkdir()
            assert _latest_record_dir(tmp_path).name == "cannsim_20260810_abc"
        elif scenario == "npusim_prefix":
            (tmp_path / "npusim_20260810_abc").mkdir()
            assert _latest_record_dir(tmp_path).name == "npusim_20260810_abc"
        elif scenario == "picks_newest":
            old = tmp_path / "cannsim_20260810_old"
            new = tmp_path / "npusim_20260810_new"
            old.mkdir()
            new.mkdir()
            os.utime(old, (1_000_000, 1_000_000))
            os.utime(new, (2_000_000, 2_000_000))
            assert _latest_record_dir(tmp_path).name == "npusim_20260810_new"
        else:
            with pytest.raises(RuntimeError, match="cannsim record produced no"):
                _latest_record_dir(tmp_path)

    @pytest.mark.parametrize(
        "scenario",
        ["prefers_toolkit_home", "rejects_non_python", "raises_when_missing"],
    )
    def test_locate_cannsim_executable(self, tmp_path, monkeypatch, scenario):
        """locate_cannsim_executable 优先 toolkit_home 的 python 脚本；非 python/缺失时报错。"""
        from ttk.core_modules.common.simulator.npusim_runner import locate_cannsim_executable

        cand = tmp_path / "ascend" / "bin" / "cannsim"
        if scenario != "raises_when_missing":
            cand.parent.mkdir(parents=True)
            if scenario == "prefers_toolkit_home":
                cand.write_text("#!/usr/bin/env python3\nfrom cannsim.main import main\n")
            else:
                cand.write_bytes(b"\x7fELF\x02\x01")  # an unrelated binary on PATH
            cand.chmod(0o755)
            monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "ascend"))
        else:
            monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))  # no cannsim on PATH
        if scenario == "prefers_toolkit_home":
            assert locate_cannsim_executable() == cand.resolve()
        else:
            with pytest.raises(RuntimeError, match="CANN cannsim not found"):
                locate_cannsim_executable()

    def test_build_sim_env_has_no_repo_path(self, tmp_path, monkeypatch):
        from ttk.core_modules.common.simulator.npusim_runner import build_sim_env

        monkeypatch.delenv("PYTHONPATH", raising=False)
        sw = SWITCHES()
        sw.root_path = str(tmp_path)
        env = build_sim_env(sw)
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert str(tmp_path) in parts  # ttk root present
        assert not any("npu-simulator" in p for p in parts)
        assert env.get("CANNSIM_NO_DELAY") == "1"
        assert env.get("NPUSIM_NO_DELAY") == "1"

    def test_cannsim_cmd_uses_sys_executable_and_script(self, monkeypatch):
        from ttk.core_modules.common.simulator import npusim_runner

        fake_script = Path("/opt/ascend/bin/cannsim")
        monkeypatch.setattr(npusim_runner, "locate_cannsim_executable", lambda: fake_script)
        cmd = npusim_runner._cannsim_cmd()
        assert cmd == [sys.executable, str(fake_script)]

    @staticmethod
    def _write_fake_npusim(tmp_path, pid_file):
        fake = tmp_path / "fake_npusim.py"
        fake.write_text(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(120)\n"
        )
        return fake

    @staticmethod
    def _wait_dead(pid, deadline_s=5.0):
        """True when pid is gone or a reparented zombie (state Z in /proc)."""
        import time

        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            try:
                with open(f"/proc/{pid}/stat") as f:
                    state = f.read().rsplit(")", 1)[1].split()[0]
            except OSError:
                return True
            if state == "Z":
                return True
            time.sleep(0.1)
        return False

    def test_run_record_timeout_kills_descendants(self, tmp_path, monkeypatch):
        """超时后 npusim 与 wrapper 后代整组被杀，不留后台仿真进程。"""
        import subprocess

        from ttk.core_modules.common.simulator import npusim_runner

        pid_file = tmp_path / "child.pid"
        fake = self._write_fake_npusim(tmp_path, pid_file)
        sw = SWITCHES()
        sw.proc_timeout = 1
        monkeypatch.setattr(npusim_runner, "_cannsim_cmd", lambda: [sys.executable, str(fake)])
        monkeypatch.setattr(npusim_runner, "build_sim_env", lambda sw: dict(os.environ))
        wrapper = tmp_path / "caseA_kernel.py"
        wrapper.write_text("")
        with pytest.raises(subprocess.TimeoutExpired):
            npusim_runner.run_record(sw, wrapper, tmp_path / "case")
        assert self._wait_dead(int(pid_file.read_text())), "wrapper descendant survived record timeout"

    def test_interrupt_kills_process_group(self, tmp_path, monkeypatch):
        """KeyboardInterrupt 时同样整组清理后重新抛出。"""
        import subprocess

        from ttk.core_modules.common.simulator import npusim_runner

        pid_file = tmp_path / "child.pid"
        fake = self._write_fake_npusim(tmp_path, pid_file)

        real_communicate = subprocess.Popen.communicate
        calls = {"n": 0}

        def fake_communicate(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # Let the fake npusim spawn its child and record the pid
                # before the interrupt lands, otherwise there is nothing to
                # leak-assert against.
                import time

                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                raise KeyboardInterrupt()
            return real_communicate(self, *args, **kwargs)

        monkeypatch.setattr(subprocess.Popen, "communicate", fake_communicate)
        with pytest.raises(KeyboardInterrupt):
            npusim_runner._run_with_cleanup([sys.executable, str(fake)], dict(os.environ), timeout=30)
        assert self._wait_dead(int(pid_file.read_text())), "wrapper descendant survived KeyboardInterrupt"

    def test_sigterm_reaps_detached_group(self, tmp_path):
        """外部 SIGTERM（如 CI 取消任务）同样整组回收：npusim 后代不残留，进程保持信号退出语义。"""
        import signal
        import subprocess
        import time

        pid_file = tmp_path / "child.pid"
        self_pid_file = tmp_path / "self.pid"
        fake_cmd = (
            "import subprocess, sys, time\n"
            f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open({str(pid_file)!r}, 'w').write(str(c.pid))\n"
            "time.sleep(120)\n"
        )
        script = (
            "import os, sys\n"
            "from ttk.core_modules.common.simulator import npusim_runner\n"
            f"open({str(self_pid_file)!r}, 'w').write(str(os.getpid()))\n"
            f"npusim_runner._run_with_cleanup([sys.executable, '-c', {fake_cmd!r}], dict(os.environ), None)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", script], env=dict(os.environ))
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if pid_file.exists() and self_pid_file.exists():
                    break
                time.sleep(0.02)
            os.kill(int(self_pid_file.read_text()), signal.SIGTERM)
            proc.wait(timeout=10)
            assert proc.returncode == -signal.SIGTERM, "TTK 子进程应保持 SIGTERM 信号退出语义"
            assert self._wait_dead(int(pid_file.read_text())), "npusim descendant survived TTK SIGTERM"
        finally:
            if proc.poll() is None:
                proc.kill()


class TestSimRecordFailure:
    """run_record failures surface as structured fail results, not crashes."""

    def test_run_aclnn_sim_returns_fail_when_record_raises(self, tmp_path, monkeypatch):
        import ttk.core_modules.common.simulator.sim_profiling as sim_mod
        from ttk.core_modules.aclnn.result import ApiProfilingResult

        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        monkeypatch.setattr(sim_mod, "get_global_storage", lambda: sw)

        def _boom(*a, **k):
            raise RuntimeError("cannsim not found")

        monkeypatch.setattr(sim_mod, "run_record", _boom)
        context = SimpleNamespace(is_valid=True, testcase_name="caseA")
        res = sim_mod.run_aclnn_sim(context, 0)
        assert isinstance(res, ApiProfilingResult)
        assert res.failed()
        assert "SIM_RECORD_FAILED" in str(res.api_prof)

    def test_run_kernel_sim_returns_fail_when_record_raises(self, tmp_path, monkeypatch):
        import ttk.core_modules.common.simulator.sim_profiling as sim_mod
        import ttk.core_modules.kernel.info_keeper as oik_pkg
        from ttk.core_modules.kernel.result import KernelExecuteResult

        # OpInfoKeeper lazily loads op-info config which needs a real CANN env.
        class _FakeOpInfoKeeper:
            def op_output_defined(self, op_name):
                return False

        monkeypatch.setattr(oik_pkg, "OpInfoKeeper", _FakeOpInfoKeeper)
        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        monkeypatch.setattr(sim_mod, "get_global_storage", lambda: sw)
        monkeypatch.setattr(sim_mod, "_construct_param", lambda *a, **k: _DummyRTSProfilingParam())
        monkeypatch.setattr(sim_mod, "_validate_param", lambda p: None)

        def _boom(*a, **k):
            raise RuntimeError("record crashed")

        monkeypatch.setattr(sim_mod, "run_record", _boom)
        context = SimpleNamespace(testcase_name="caseA", op_name="Add")
        dyn, cst, bin_ = sim_mod.run_kernel_sim(context)
        assert isinstance(dyn, KernelExecuteResult)
        assert "SIM_RECORD_FAILED" in str(dyn.cycle)
        assert cst.cycle == "SUPPRESSED"
        assert bin_.cycle == "SUPPRESSED"


class TestPerfStatusSimUnknown:
    """NPUSim 报 cycle=UNKNOWN 时 perf_status 不应失败；真实设备 UNKNOWN 仍失败。"""

    @staticmethod
    def _ctx(monkeypatch, mode, dyn_cycle, cst_fail, bin_fail):
        import ttk.core_modules.kernel.compare as compare_mod
        from ttk.core_modules.kernel.result import KernelExecuteResult

        sw = SWITCHES()
        sw.mode = mode
        monkeypatch.setattr(compare_mod, "get_global_storage", lambda: sw)
        return SimpleNamespace(
            dyn_prof_result=KernelExecuteResult(dyn_cycle),
            cst_prof_result=KernelExecuteResult.fail(cst_fail),
            bin_prof_result=KernelExecuteResult.fail(bin_fail),
        )

    @pytest.mark.parametrize(
        ("mode", "dyn_cycle", "cst_fail", "bin_fail", "expected"),
        [
            (MODE.ASCEND_CAMODEL, "UNKNOWN", "SUPPRESSED", "SUPPRESSED", "PASS"),
            (MODE.ASCEND_ONBOARD, "UNKNOWN", "SUPPRESSED", "SUPPRESSED", "FAIL"),
            (MODE.ASCEND_CAMODEL, 100.5, "SUPPRESSED", "SUPPRESSED", "PASS"),
            (MODE.ASCEND_ONBOARD, 100.5, "CST_OFF", "BIN_OFF", "PASS"),
        ],
    )
    def test_handle_profiling_result(self, monkeypatch, mode, dyn_cycle, cst_fail, bin_fail, expected):
        from ttk.core_modules.kernel.compare import handle_profiling_result

        ctx = self._ctx(monkeypatch, mode, dyn_cycle, cst_fail, bin_fail)
        assert handle_profiling_result(ctx) == expected
