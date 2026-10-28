#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import io
import os
import struct
import types

import numpy as np
import pytest

from ttk.core_modules.geir import execute as geir_execute


def _plan():
    return geir_execute.GraphExecPlan(
        binary="/tmp/op", config_path="/tmp/case.json", build_dir="/tmp", prof_path="", input_prefix="/tmp/input"
    )


def _testcase():
    return types.SimpleNamespace(flat_output_dtypes=("float32",), flat_output_shapes=((2,),), testcase_name="case")


def _serve_with_frame(data):
    serve = object.__new__(geir_execute._GeirServe)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, data)
    os.close(write_fd)
    serve._data_r = read_fd
    serve.proc = types.SimpleNamespace(stdin=io.BytesIO())
    serve._stderr_tail = bytearray()
    return serve


def test_serve_success_frame_roundtrip():
    payload = np.array([1.5, 2.5], dtype=np.float32).tobytes()
    frame = struct.pack("<q", 1) + struct.pack("<q", len(payload)) + payload
    serve = _serve_with_frame(frame)
    outputs = serve.run_case(_testcase(), _plan(), types.SimpleNamespace(proc_timeout=1))
    assert len(outputs) == 1
    np.testing.assert_allclose(outputs[0], np.array([1.5, 2.5], dtype=np.float32))
    os.close(serve._data_r)


def test_serve_failure_frame_message():
    message = b"RunGraph failed"
    frame = struct.pack("<q", -1) + struct.pack("<q", len(message)) + message
    serve = _serve_with_frame(frame)
    with pytest.raises(RuntimeError, match="RunGraph failed"):
        serve.run_case(_testcase(), _plan(), types.SimpleNamespace(proc_timeout=1))
    os.close(serve._data_r)


def test_run_once_serve_success(monkeypatch):
    serve = types.SimpleNamespace(run_case=lambda *_: ["output"])
    monkeypatch.setattr(geir_execute, "_serve_instance", lambda *_: serve)
    monkeypatch.setattr(geir_execute, "_close_geir_serve", lambda: pytest.fail("must not close"))
    outputs = geir_execute._run_once_serve(_testcase(), 0, types.SimpleNamespace(proc_timeout=1), _plan())
    assert outputs == ["output"]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("GEIR serve timeout: incomplete frame"),
        RuntimeError("GEIR serve process exited unexpectedly"),
        BrokenPipeError("broken pipe"),
    ],
)
def test_run_once_serve_discards_broken_instance(error, monkeypatch):
    serve = types.SimpleNamespace(run_case=lambda *_: (_ for _ in ()).throw(error))
    closed = []
    monkeypatch.setattr(geir_execute, "_serve_instance", lambda *_: serve)
    monkeypatch.setattr(geir_execute, "_close_geir_serve", lambda: closed.append(True))
    with pytest.raises(type(error)):
        geir_execute._run_once_serve(_testcase(), 0, types.SimpleNamespace(proc_timeout=1), _plan())
    assert closed == [True]


def test_run_once_serve_keeps_instance_after_case_failure(monkeypatch):
    error = RuntimeError(f"{geir_execute._SERVE_CASE_ERROR}RunGraph failed")
    serve = types.SimpleNamespace(run_case=lambda *_: (_ for _ in ()).throw(error))
    monkeypatch.setattr(geir_execute, "_serve_instance", lambda *_: serve)
    monkeypatch.setattr(geir_execute, "_close_geir_serve", lambda: pytest.fail("case failure must not close"))
    with pytest.raises(RuntimeError, match="RunGraph failed"):
        geir_execute._run_once_serve(_testcase(), 0, types.SimpleNamespace(proc_timeout=1), _plan())


def test_serve_instance_respawns_after_death_and_recompile(monkeypatch):
    created = []

    class FakeServe:
        def __init__(self, binary, build_dir, dev_id):
            self.binary = binary
            self.alive_value = True
            self.closed = False
            created.append(self)

        def alive(self):
            return self.alive_value

        def close(self):
            self.closed = True

    monkeypatch.setattr(geir_execute, "_SERVE_INST", None)
    monkeypatch.setattr(geir_execute, "_SERVE_KEY", None)
    monkeypatch.setattr(geir_execute, "_GeirServe", FakeServe)
    mtime = [1]
    monkeypatch.setattr(geir_execute.os.path, "getmtime", lambda _: mtime[0])

    first = geir_execute._serve_instance(_plan(), 0)
    first.alive_value = False
    second = geir_execute._serve_instance(_plan(), 0)
    assert second is not first
    assert first.closed

    mtime[0] = 2
    third = geir_execute._serve_instance(_plan(), 0)
    assert third is not second
    assert second.closed
    geir_execute._close_geir_serve()
    assert third.closed


def test_close_waits_after_kill():
    class FakeProc:
        def __init__(self):
            self.stdin = types.SimpleNamespace(close=lambda: None)
            self.killed = False
            self.wait_calls = 0

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise RuntimeError("still running")
            return 0

        def kill(self):
            self.killed = True

    serve = object.__new__(geir_execute._GeirServe)
    serve.proc = FakeProc()
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    serve._data_r = read_fd
    serve.close()
    assert serve.proc.killed
    assert serve.proc.wait_calls == 2


def test_pdeathsig_preexec_wiring(monkeypatch):
    seen = {}

    class FakeProc:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()

    def fake_popen(args, **kwargs):
        seen.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(geir_execute.subprocess, "Popen", fake_popen)
    geir_execute._GeirServe("bin", "/tmp", 0)
    assert seen.get("preexec_fn") is geir_execute._set_pdeathsig
    geir_execute._set_pdeathsig()
