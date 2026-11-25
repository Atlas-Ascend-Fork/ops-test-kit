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
        # flush-before-exit order is asserted behaviorally in TestWrapperFlushOrder.

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


class _WrapperSW:
    """switches.pkl stand-in（模块级以便 pickle by reference）。"""

    short_soc_version = "Ascend950"

    class mode:
        @staticmethod
        def is_model():
            return True

    def activate(self):
        pass


def _run_wrapper_script(wrapper_path, case_dir, monkeypatch, *, stub_modules, flush_error=None):
    """加载生成的 wrapper 文件并以替身运行，返回 flush/_exit 的调用顺序记录。

    - 经 importlib 加载 write_*_wrapper 落盘的真实生成物（非内嵌字符串）
    - ctypes.CDLL 替身记录 ``cannsim_helper_flush`` 调用（可注入符号缺失）
    - os._exit 替身记录退出码并以 SystemExit 阻止真实退出
    - stub_modules 注入 wrapper 运行期 import 的替身模块（sys.modules 临时替换）
    """
    import ctypes
    import importlib.util
    import types

    calls = []

    class _FakeCDLL:
        def __init__(self, name=None, mode=None):
            pass

        def __getattr__(self, item):
            if item == "cannsim_helper_flush":

                def _flush():
                    calls.append("flush")
                    if flush_error is not None:
                        raise flush_error

                return _flush
            raise AttributeError(item)

    def _fake_exit(code):
        calls.append(("exit", code))
        raise SystemExit(code)

    injected = {}
    for mod_name, attrs in stub_modules.items():
        mod = types.ModuleType(mod_name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        injected[mod_name] = mod
    saved = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)

    monkeypatch.setattr(ctypes, "CDLL", _FakeCDLL)
    monkeypatch.setattr(os, "_exit", _fake_exit)

    spec = importlib.util.spec_from_file_location("generated_wrapper_test", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_wrapper_test"] = module
    spec.loader.exec_module(module)
    old_argv = sys.argv
    sys.argv = ["wrapper.py", str(case_dir)]
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return calls


class TestWrapperFlushOrder:
    """行为级验证 wrapper 收尾顺序：flush 先于 os._exit，覆盖成功/异常/旧 helper 缺符号。

    字符串位置断言（如 index("cannsim_helper_flush()") < index("os._exit(0)")）会
    匹配到函数定义中的符号名，无法证明运行期顺序——这里直接运行生成脚本。
    """

    # ---- kernel wrapper ----

    @pytest.fixture
    def kernel_case_dir(self, tmp_path):
        case_dir = tmp_path / "kernel_case"
        (case_dir / "dyn").mkdir(parents=True)
        with (case_dir / "switches.pkl").open("wb") as f:
            pickle.dump(_WrapperSW(), f)
        with (case_dir / "dyn" / "param.pkl").open("wb") as f:
            pickle.dump(_DummyRTSProfilingParam(), f)
        return case_dir

    @staticmethod
    def _kernel_stubs(execute):
        class FakeRTSInterface:
            def __init__(self, **kwargs):
                pass

            def set_device(self, device_id):
                pass

        class FakeKernelExecutor:
            def __init__(self, dev, param):
                pass

            def execute(self):
                return execute()

        return {
            "ttk.core_modules.kernel.runtime.rts_interface": {"RTSInterface": FakeRTSInterface},
            "ttk.core_modules.kernel.execute": {"KernelExecutor": FakeKernelExecutor},
        }

    @staticmethod
    def _write_kernel_wrapper(tmp_path):
        from ttk.core_modules.common.simulator.wrapper import write_kernel_wrapper

        sw = SimpleNamespace(sim_output_dir=str(tmp_path))
        return write_kernel_wrapper(sw, tmp_path / "case")

    def test_kernel_wrapper_flushes_then_exits_on_success(self, kernel_case_dir, monkeypatch):
        res = SimpleNamespace(output_bytes=[b"\x01"], cycle="0", oob="OK")
        calls = _run_wrapper_script(
            self._write_kernel_wrapper(kernel_case_dir),
            kernel_case_dir,
            monkeypatch,
            stub_modules=self._kernel_stubs(lambda: res),
        )
        assert calls == ["flush", ("exit", 0)]
        assert json.loads((kernel_case_dir / "final_result.json").read_text()) == {"ok": True}

    def test_kernel_wrapper_flushes_then_exits_on_error(self, kernel_case_dir, monkeypatch):
        def boom():
            raise RuntimeError("kernel boom")

        calls = _run_wrapper_script(
            self._write_kernel_wrapper(kernel_case_dir),
            kernel_case_dir,
            monkeypatch,
            stub_modules=self._kernel_stubs(boom),
        )
        assert calls == ["flush", ("exit", 0)]
        assert json.loads((kernel_case_dir / "dyn" / "result.json").read_text())["ok"] is False
        assert json.loads((kernel_case_dir / "final_result.json").read_text()) == {"ok": False}

    def test_kernel_wrapper_exits_when_helper_lacks_flush_symbol(self, kernel_case_dir, monkeypatch):
        res = SimpleNamespace(output_bytes=[b"\x01"], cycle="0", oob="OK")
        calls = _run_wrapper_script(
            self._write_kernel_wrapper(kernel_case_dir),
            kernel_case_dir,
            monkeypatch,
            stub_modules=self._kernel_stubs(lambda: res),
            flush_error=AttributeError("missing cannsim_helper_flush (old CANN)"),
        )
        assert calls == ["flush", ("exit", 0)]

    # ---- aclnn wrapper ----

    @pytest.fixture
    def aclnn_case_dir(self, tmp_path):
        case_dir = tmp_path / "aclnn_case"
        case_dir.mkdir()
        with (case_dir / "switches.pkl").open("wb") as f:
            pickle.dump(_WrapperSW(), f)
        with (case_dir / "context.pkl").open("wb") as f:
            pickle.dump({"case": "ctx"}, f)
        return case_dir

    @staticmethod
    def _aclnn_stubs(do_result):
        class FakeAclInterface:
            def __init__(self, *args, **kwargs):
                pass

            def set_device(self, device_id):
                pass

        class FakeAclOpExecutor:
            def __init__(self, context, dev):
                pass

            def do(self):
                if isinstance(do_result, Exception):
                    raise do_result
                return do_result

        return {
            "ttk.core_modules.aclnn": {"AclInterface": FakeAclInterface},
            "ttk.core_modules.aclnn.execute": {"AclOpExecutor": FakeAclOpExecutor},
        }

    @staticmethod
    def _write_aclnn_wrapper(tmp_path):
        from ttk.core_modules.common.simulator.wrapper import write_aclnn_wrapper

        sw = SimpleNamespace(sim_output_dir=str(tmp_path))
        return write_aclnn_wrapper(sw, tmp_path / "case")

    @staticmethod
    def _aclnn_result(success=True):
        return SimpleNamespace(
            success=success,
            api_prof="1" if success else "ERR",
            op_prof="1",
            oob="OK",
            deterministic_status=None,
            output_bytes=(b"\x01",) if success else (),
            output_view_shapes=[(8,)] if success else [],
        )

    def test_aclnn_wrapper_flushes_then_exits_on_success(self, aclnn_case_dir, monkeypatch):
        calls = _run_wrapper_script(
            self._write_aclnn_wrapper(aclnn_case_dir),
            aclnn_case_dir,
            monkeypatch,
            stub_modules=self._aclnn_stubs(self._aclnn_result(success=True)),
        )
        assert calls == ["flush", ("exit", 0)]
        payload = json.loads((aclnn_case_dir / "final_result.json").read_text())
        assert payload["ok"] is True

    def test_aclnn_wrapper_flushes_then_exits_on_error(self, aclnn_case_dir, monkeypatch):
        calls = _run_wrapper_script(
            self._write_aclnn_wrapper(aclnn_case_dir),
            aclnn_case_dir,
            monkeypatch,
            stub_modules=self._aclnn_stubs(RuntimeError("aclnn boom")),
        )
        assert calls == ["flush", ("exit", 0)]
        assert (aclnn_case_dir / "wrapper_error.json").is_file()

    def test_aclnn_wrapper_exits_when_helper_lacks_flush_symbol(self, aclnn_case_dir, monkeypatch):
        calls = _run_wrapper_script(
            self._write_aclnn_wrapper(aclnn_case_dir),
            aclnn_case_dir,
            monkeypatch,
            stub_modules=self._aclnn_stubs(self._aclnn_result(success=True)),
            flush_error=AttributeError("missing cannsim_helper_flush (old CANN)"),
        )
        assert calls == ["flush", ("exit", 0)]


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
        [
            "prefers_toolkit_home",
            "prefers_npusim",
            "rejects_non_python",
            "path_bogus_npusim_prefers_cannsim",
            "path_only_real_npusim",
            "raises_when_missing",
        ],
    )
    def test_locate_cannsim_executable(self, tmp_path, monkeypatch, scenario):
        """locate_cannsim_executable 优先 toolkit_home 的 python 脚本；非 python/缺失时报错。

        PATH 上无关的 python ``npusim`` 不得抢占有效的 CANN ``cannsim``
        —— 候选校验须匹配 CANN 发布入口的 import 特征。
        """
        from ttk.core_modules.common.simulator.npusim_runner import locate_cannsim_executable

        cand = tmp_path / "ascend" / "bin" / "cannsim"
        if scenario not in ("raises_when_missing", "path_bogus_npusim_prefers_cannsim", "path_only_real_npusim"):
            cand.parent.mkdir(parents=True)
            if scenario in ("prefers_toolkit_home", "prefers_npusim"):
                cand.write_text("#!/usr/bin/env python3\nfrom cannsim.main import main\n")
            else:
                cand.write_bytes(b"\x7fELF\x02\x01")  # an unrelated binary on PATH
            if scenario == "prefers_npusim":
                npusim = cand.with_name("npusim")
                npusim.write_text("#!/usr/bin/env python3\nfrom npusim.main import main\n")
                npusim.chmod(0o755)
            cand.chmod(0o755)
            monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "ascend"))
        else:
            monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
        if scenario == "path_bogus_npusim_prefers_cannsim":
            bogus = tmp_path / "npusim"
            bogus.write_text("#!/usr/bin/env python3\nprint('unrelated python program')\n")
            bogus.chmod(0o755)
            real = tmp_path / "cannsim"
            real.write_text("#!/usr/bin/env python3\nfrom cannsim.main import main\n")
            real.chmod(0o755)
        elif scenario == "path_only_real_npusim":
            bogus_cannsim = tmp_path / "cannsim"
            bogus_cannsim.write_text("#!/usr/bin/env python3\nprint('unrelated python program')\n")
            bogus_cannsim.chmod(0o755)
            real_npusim = tmp_path / "npusim"
            real_npusim.write_text("#!/usr/bin/env python3\nfrom npusim.main import main\n")
            real_npusim.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        if scenario == "prefers_toolkit_home":
            assert locate_cannsim_executable() == cand.resolve()
        elif scenario == "prefers_npusim":
            assert locate_cannsim_executable() == (tmp_path / "ascend" / "bin" / "npusim").resolve()
        elif scenario == "path_bogus_npusim_prefers_cannsim":
            assert locate_cannsim_executable() == (tmp_path / "cannsim").resolve()
        elif scenario == "path_only_real_npusim":
            assert locate_cannsim_executable() == (tmp_path / "npusim").resolve()
        else:
            with pytest.raises(RuntimeError, match="CANN cannsim not found"):
                locate_cannsim_executable()

    def test_build_sim_env_has_no_repo_path(self, tmp_path, monkeypatch):
        from ttk.core_modules.common.simulator.npusim_runner import build_sim_env

        monkeypatch.delenv("PYTHONPATH", raising=False)
        monkeypatch.delenv("CANNSIM_HELPER_PER_CHIP_DB", raising=False)
        sw = SWITCHES()
        sw.root_path = str(tmp_path)
        env = build_sim_env(sw)
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert str(tmp_path) in parts  # ttk root present
        assert not any("npu-simulator" in p for p in parts)
        assert env.get("CANNSIM_NO_DELAY") == "1"
        assert env.get("NPUSIM_NO_DELAY") == "1"
        # wrapper 的 dlopen 直取 rt* 符号，绕过 helper 的 aclrt* 拦截，
        # per-chip DB 是唯一能落 DB 的采集路径（ktest 同款）。
        assert env.get("CANNSIM_HELPER_PER_CHIP_DB") == "1"

    def test_build_sim_env_per_chip_db_explicit_wins(self, monkeypatch):
        from ttk.core_modules.common.simulator.npusim_runner import build_sim_env

        monkeypatch.setenv("CANNSIM_HELPER_PER_CHIP_DB", "0")
        sw = SWITCHES()
        env = build_sim_env(sw)
        assert env["CANNSIM_HELPER_PER_CHIP_DB"] == "0"

    @pytest.mark.parametrize(
        "scenario",
        [
            "legacy_instr_root",
            "legacy_instr_nested",
            "db_flat_record",
            "db_plain_record",
            "db_archive_root",
            "db_helper_dbs_root",
            "db_helper_dbs_nested",
            "db_chip_dir",
            "mixed_instr_and_chip_db",
            "mixed_instr_and_flat_db",
            "empty_none",
        ],
    )
    def test_report_export_dir(self, tmp_path, scenario):
        """_report_export_dir 兼容 v0.1.0 instr.bin 与 >=0.7 helper db 布局。

        契约 DB 优先于 instr.bin：>=0.7 报告器不认 instr.bin，混合归档中
        instr.bin 一票通过会让已完成采集的用例丢流水图（beta.1 平铺报告器
        不会从 record/ 向下找 chip0/，也读不到目录外的根平铺 DB）。
        """
        from ttk.core_modules.common.simulator.npusim_runner import _report_export_dir

        export = tmp_path / "npusim_20260810_case"
        export.mkdir()
        if scenario == "legacy_instr_root":
            (export / "instr.bin").write_bytes(b"x")
            assert _report_export_dir(export) == export
        elif scenario == "legacy_instr_nested":
            (export / "record").mkdir()
            (export / "record" / "instr.bin").write_bytes(b"x")
            assert _report_export_dir(export) == export / "record"
        elif scenario == "db_flat_record":
            (export / "record").mkdir()
            (export / "record" / "chip0.db").write_bytes(b"x")
            assert _report_export_dir(export) == export
        elif scenario == "db_plain_record":
            (export / "record").mkdir()
            (export / "record" / "launch_0000.db").write_bytes(b"x")
            assert _report_export_dir(export) == export
        elif scenario == "db_archive_root":
            (export / "chip0.db").write_bytes(b"x")
            assert _report_export_dir(export) == export
        elif scenario == "db_helper_dbs_root":
            (export / "cannsim_helper_dbs").mkdir()
            (export / "cannsim_helper_dbs" / "launch_0000.db").write_bytes(b"x")
            assert _report_export_dir(export) == export
        elif scenario == "db_helper_dbs_nested":
            (export / "record" / "cannsim_helper_dbs").mkdir(parents=True)
            (export / "record" / "cannsim_helper_dbs" / "launch_0000.db").write_bytes(b"x")
            assert _report_export_dir(export) == export
        elif scenario == "db_chip_dir":
            # 单 chip 目录：直接传 chip 目录（平铺-only 报告器可发现，
            # 按芯片路由的报告器同样接受直接含 *.db 的目录）
            chip = export / "record" / "chip0"
            chip.mkdir(parents=True)
            (chip / "kernel.db").write_bytes(b"x")
            assert _report_export_dir(export) == chip
        elif scenario == "mixed_instr_and_chip_db":
            # 混合归档中 instr.bin 不得遮住 chip DB —— 传 chip 目录而非 record/
            chip = export / "record" / "chip0"
            chip.mkdir(parents=True)
            (chip / "kernel.db").write_bytes(b"x")
            (export / "record" / "instr.bin").write_bytes(b"x")
            assert _report_export_dir(export) == chip
        elif scenario == "mixed_instr_and_flat_db":
            # 根平铺 DB 与 record/instr.bin 并存：传归档根（报告器从根见 DB）
            (export / "record").mkdir()
            (export / "record" / "instr.bin").write_bytes(b"x")
            (export / "chip0.db").write_bytes(b"x")
            assert _report_export_dir(export) == export
        else:
            (export / "record").mkdir()
            assert _report_export_dir(export) is None

    def test_report_multi_chip_runs_one_report_per_chip(self, tmp_path, monkeypatch):
        """平铺-only 报告器从归档根找不到 chip DB，多芯片逐颗报告。

        每颗 chip 一次 report，输出进独立子目录，第二次调用不覆盖第一次。
        """
        from ttk.core_modules.common.simulator import npusim_runner

        export = tmp_path / "npusim_multi_chip"
        for n in (0, 1):
            chip = export / "record" / f"chip{n}"
            chip.mkdir(parents=True)
            (chip / "kernel.db").write_bytes(b"x")
        cmds = []

        def fake_run(cmd, env, timeout=None):
            cmds.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(npusim_runner, "_cannsim_cmd", lambda: ["cannsim"])
        monkeypatch.setattr(npusim_runner, "_run_with_cleanup", fake_run)
        sw = SimpleNamespace(root_path=str(tmp_path), sim_cores="0", sim_object_file=None, proc_timeout=10)
        out = npusim_runner.run_report(sw, export, tmp_path / "report")
        assert out is not None
        assert len(cmds) == 2
        inputs = [cmd[cmd.index("-e") + 1] for cmd in cmds]
        outputs = [cmd[cmd.index("-o") + 1] for cmd in cmds]
        assert inputs[0] != inputs[1]
        assert Path(inputs[0]).name == "chip0"
        assert Path(inputs[1]).name == "chip1"
        # 独立输出目录：第二次调用不得覆盖第一次
        assert outputs[0] != outputs[1]
        assert Path(outputs[0]).name == "chip0"
        assert Path(outputs[1]).name == "chip1"

    def test_cannsim_cmd_uses_sys_executable_and_script(self, monkeypatch):
        from ttk.core_modules.common.simulator import npusim_runner

        fake_script = Path("/opt/ascend/bin/cannsim")
        monkeypatch.setattr(npusim_runner, "locate_cannsim_executable", lambda: fake_script)
        cmd = npusim_runner._cannsim_cmd()
        assert cmd == [sys.executable, str(fake_script)]

    def test_sim_obj_record_report_command_routing(self, tmp_path, monkeypatch):
        """同一用例分别核对 record 与 report 两条命令。

        --sim-obj 不进 record（>= 0.7 拒绝 -f）；由 report --object-file 消费。
        只测 run_report 断言不到 run_record 的参数，故两条命令都捕获。
        """
        from ttk.core_modules.common.simulator import npusim_runner

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        wrapper = case_dir / "wrapper_kernel.py"
        wrapper.write_text("#!/usr/bin/env python3\n")
        obj = case_dir / "kernel.o"
        obj.write_bytes(b"\x7fELF")
        cmds = []

        def fake_run(cmd, env, timeout=None):
            cmds.append(cmd)
            # run_record 之后 _latest_record_dir 找新建归档；伪造一个让流程走通
            if "record" in cmd and "-s" in cmd:
                (case_dir / "record_out" / "npusim_fake_archive").mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(npusim_runner, "_cannsim_cmd", lambda: ["cannsim"])
        monkeypatch.setattr(npusim_runner, "_run_with_cleanup", fake_run)
        sw = SimpleNamespace(
            root_path=str(tmp_path),
            sim_soc_version="Ascend950",
            sim_cores="0",
            sim_object_file=str(obj),
            proc_timeout=10,
        )
        # ① record：不得携带 -f / sim_object_file
        npusim_runner.run_record(sw, wrapper, case_dir, extra_argv=("-u", str(case_dir)))
        record_cmd = cmds[-1]
        assert record_cmd[:2] == ["cannsim", "record"]
        assert "-f" not in record_cmd
        assert str(obj) not in record_cmd
        # ② report：归档含平铺 DB 时 --object-file 路由（绝对路径）
        export = case_dir / "record_out" / "npusim_fake_archive"
        (export / "record").mkdir(exist_ok=True)
        (export / "record" / "launch_0000.db").write_bytes(b"x")
        out = npusim_runner.run_report(sw, export, case_dir / "report")
        report_cmd = cmds[-1]
        assert out is not None
        assert report_cmd[:3] == ["cannsim", "report", "-e"]
        assert "--object-file" in report_cmd
        assert str(obj.resolve()) in report_cmd
        # ③ sim_object_file 缺省时两条命令都不携带目标文件
        sw.sim_object_file = None
        npusim_runner.run_report(sw, export, case_dir / "report")
        assert "--object-file" not in cmds[-1]

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
