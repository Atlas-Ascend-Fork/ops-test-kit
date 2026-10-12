# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MsProfiler 缺符号防御 + aclprofInit 字节长度 + task_type 分类防御 +
msprof.py 子进程导出（sys.executable/timeout）测试。

CDLL __getattr__ 缺符号抛 AttributeError 而非返回 None，故取符号须
getattr(dll, name, None)——预期路径是 RuntimeError（_api_call 族）或
warning + return（report_op_dfx），不是 AttributeError 逃逸。
"""

import ctypes
import logging
import subprocess
import sys

import pytest

from ttk.core_modules.common.profiler.ms_profiler import _MSPROF_EXPORT_TIMEOUT_S, MsProfiler
from ttk.core_modules.common.profiler.types import MsprofCompactInfo, MsProfOpDfx, MsprofTaskType


class _MissingSymbolDll:
    """Mimics ctypes.CDLL attribute access for a symbol that does not exist."""

    def __getattr__(self, name):
        raise AttributeError(name)


class _FakeApi:
    """Records calls; returns a fixed int (aclError-style) result."""

    def __init__(self, ret=0):
        self.ret = ret
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.ret


class _FakeDll:
    """Mimics ctypes.CDLL over a dict of name -> callable."""

    def __init__(self, apis):
        self._apis = apis

    def __getattr__(self, name):
        try:
            return self._apis[name]
        except KeyError:
            raise AttributeError(name) from None


def _bare_profiler():
    """MsProfiler without running __init__ (no CDLL load / rmtree side effects)."""
    return MsProfiler.__new__(MsProfiler)


def test_api_call_missing_symbol_raises_runtime_error():
    prof = _bare_profiler()
    prof._msprof_dll = _MissingSymbolDll()
    with pytest.raises(RuntimeError, match="aclprofInit.*is not defined"):
        prof._api_call("aclprofInit", None)


def test_api_call_with_ptr_return_missing_symbol_raises_runtime_error():
    prof = _bare_profiler()
    prof._msprof_dll = _MissingSymbolDll()
    with pytest.raises(RuntimeError, match="aclprofCreateConfig.*is not defined"):
        prof._api_call_with_ptr_return("aclprofCreateConfig", None)


def test_report_op_dfx_missing_symbol_warns_and_returns(caplog):
    prof = _bare_profiler()
    prof._is_model = False
    prof._msprof_dll = object()  # non-None: report path is active
    prof._prof_api_dll = _MissingSymbolDll()
    with caplog.at_level(logging.WARNING):
        prof.report_op_dfx(MsProfOpDfx(kernel_name="k", block_dim=1, core_type="AiCore"))
    assert "MsprofReportCompactInfo" in caplog.text


def test_get_hash_id_missing_symbol_raises_runtime_error():
    prof = _bare_profiler()
    prof._prof_api_dll = _MissingSymbolDll()
    with pytest.raises(RuntimeError, match="MsprofGetHashId.*is not defined"):
        prof._get_hash_id("some_kernel")


def test_get_sys_cycle_missing_symbol_raises_runtime_error():
    prof = _bare_profiler()
    prof._prof_api_dll = _MissingSymbolDll()
    with pytest.raises(RuntimeError, match="MsprofSysCycleTime.*is not defined"):
        prof._get_sys_cycle()


# --- A1: aclprofInit buffer length must be the encoded byte count ---


def test_acl_prof_init_passes_encoded_byte_length():
    prof = _bare_profiler()
    api = _FakeApi(0)
    prof._msprof_dll = _FakeDll({"aclprofInit": api})
    prof._result_path = "/tmp/性能/结果目录"  # non-ASCII: len(str) != len(bytes)
    prof._acl_prof_init()
    (c_path, c_size) = api.calls[0]
    assert c_size.value == len(prof._result_path.encode("UTF-8"))
    assert c_size.value != len(prof._result_path)  # guard is meaningful for this path
    assert prof._prof_inited is True


# --- A2: _get_op_task_type classification + malformed-ration guards ---


def test_get_op_task_type_classification():
    f = MsProfiler._get_op_task_type
    aicore = MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
    assert f("AiCore", None) == aicore
    assert f("CubeCore", None) == aicore
    assert f("VectorCore", None) == MsprofTaskType.MSPROF_TASK_TYPE_AIV.value
    assert f("MixAic", (2, 3)) == MsprofTaskType.MSPROF_TASK_TYPE_MIX_AIC.value
    assert f("MixAic", (2, 0)) == aicore  # aiv == 0
    assert f("MixAic", (0, 3)) == MsprofTaskType.MSPROF_TASK_TYPE_AIV.value  # aic == 0


def test_get_op_task_type_guards_malformed_ration(caplog):
    """None / short task_ration must not raise (report_op_dfx runs inside the
    per-repeat kernel launch try) — degrade to AI_CORE with a warning."""
    f = MsProfiler._get_op_task_type
    with caplog.at_level(logging.WARNING):
        assert f("MixAic", None) == MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
        assert f("MixAic", (1,)) == MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
    assert "cannot resolve msprof task type" in caplog.text


# --- B1: msprof.py export subprocess (sys.executable + timeout) ---


def _setup_msprof_stub(tmp_path, monkeypatch):
    opp = tmp_path / "opp"
    opp.mkdir()  # ASCEND_OPP_PATH/../ 解析要求该目录存在
    msprof = tmp_path / "tools" / "profiler" / "profiler_tool" / "analysis" / "msprof" / "msprof.py"
    msprof.parent.mkdir(parents=True)
    msprof.write_text("# stub\n")
    monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
    prof = _bare_profiler()
    prof._result_path = str(tmp_path)
    return prof


class _CompletedOk:
    returncode = 0
    stdout = b""
    stderr = b""


def test_analysis_profile_data_uses_sys_executable_and_timeout(tmp_path, monkeypatch):
    prof = _setup_msprof_stub(tmp_path, monkeypatch)
    cmds = []

    def _fake_run(cmd, **kwargs):
        cmds.append((cmd, kwargs))
        return _CompletedOk()

    monkeypatch.setattr("ttk.core_modules.common.profiler.ms_profiler.subprocess.run", _fake_run)
    prof._analysis_profile_data()
    assert [c[0][3] for c in cmds] == ["summary", "timeline"]
    for cmd, kwargs in cmds:
        assert cmd[0] == sys.executable  # not "python3": venv-safe
        assert kwargs.get("timeout") == _MSPROF_EXPORT_TIMEOUT_S


def test_analysis_profile_data_timeout_logged_not_raised(tmp_path, monkeypatch, caplog):
    prof = _setup_msprof_stub(tmp_path, monkeypatch)

    def _hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr("ttk.core_modules.common.profiler.ms_profiler.subprocess.run", _hang)
    prof._analysis_profile_data()  # must not raise (__exit__ must stay clean)
    assert caplog.text.count("timed out") == 2


# --- 生命周期契约：NONE no-op / step·start_step / report happy path / parse_error ---


def test_ms_profiler_none_type_is_noop(tmp_path):
    """TtkMsProfType.NONE = 不采集：构造不加载 libmsprofiler.so、不 rmtree，
    with/step/report_op_dfx 全 no-op——每次 --task-profiling=false 都走这条路径。"""
    from ttk.core_modules.common.profiler.types import TtkMsProfType

    prof = MsProfiler(0, str(tmp_path), TtkMsProfType.NONE)
    assert prof._msprof_dll is None
    assert prof._data_type_cfg == 0
    with prof:
        prof.step()
        prof.report_op_dfx(MsProfOpDfx(kernel_name="k", block_dim=1, core_type="AiCore"))
    assert not prof._prof_inited  # nothing was ever started


def test_ms_profiler_step_starts_only_at_start_step():
    """step() 只在 _current_step == start_step 时 _start() 一次，不重复启动。"""
    started = _FakeApi(0)
    prof = _bare_profiler()
    prof._msprof_dll = _FakeDll(
        {
            "aclprofInit": _FakeApi(0),
            "aclprofCreateConfig": _FakeApi(0x1234),  # non-NULL ptr
            "aclprofStart": started,
        }
    )
    prof._dvc_id = 0
    prof._result_path = "/tmp/x"
    prof._start_step = 2
    prof._current_step = 0
    prof._prof_inited = False
    prof._prof_started = False
    prof._prof_cfg = None
    prof._data_type_cfg = 7
    prof.step()  # 0 -> 1
    prof.step()  # 1 -> 2 (still not start_step)
    assert started.calls == []
    prof.step()  # 2 == start_step -> _start()
    assert len(started.calls) == 1
    prof.step()  # already past start_step, no double start
    assert len(started.calls) == 1


def test_report_op_dfx_calls_compact_info_api():
    """report_op_dfx happy path：hash→compact 结构→MsprofReportCompactInfo(1, ptr, size)。
    compact 结构是 report_op_dfx 的局部对象，须在调用时刻快照（from_buffer_copy）。"""

    class _CapturingCompactApi:
        def __init__(self):
            self.snapshots = []

        def __call__(self, topology, ptr, size):
            info = ctypes.cast(ptr, ctypes.POINTER(MsprofCompactInfo)).contents
            self.snapshots.append((topology, size, MsprofCompactInfo.from_buffer_copy(info)))
            return 0

    prof = _bare_profiler()
    prof._is_model = False
    prof._msprof_dll = object()  # non-None: report path is active
    compact = _CapturingCompactApi()
    prof._prof_api_dll = _FakeDll(
        {
            "MsprofGetHashId": _FakeApi(42),
            "MsprofSysCycleTime": _FakeApi(1000),
            "MsprofReportCompactInfo": compact,
        }
    )
    prof.report_op_dfx(MsProfOpDfx(kernel_name="my_kernel", block_dim=8, core_type="AiCore"))
    topology, size, snapshot = compact.snapshots[0]
    assert topology == 1
    assert size == ctypes.sizeof(MsprofCompactInfo)
    assert snapshot.data.node_basic_info.op_name == 42  # hashed kernel name
    assert snapshot.data.node_basic_info.block_dim == 8
    assert snapshot.data.node_basic_info.task_type == MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value


def test_parse_error_contract():
    """0 成功不抛；已知错误码 → RuntimeError 带符号名；非法类型 → TypeError。"""
    MsProfiler.parse_error(0, "api", "info")  # success: no raise
    with pytest.raises(RuntimeError, match="ACL_ERROR_INVALID_PARAM"):
        MsProfiler.parse_error(100000, "api", "info")
    with pytest.raises(TypeError, match="Invalid acl_ret type"):
        MsProfiler.parse_error("bad", "api", "info")
