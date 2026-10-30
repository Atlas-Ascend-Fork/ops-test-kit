# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""onnx 路由/适配器/backend 单测（纯 CPU + mock 会话流）。

detect_framework 前缀路由与 get_adapter 注册/别名、run_graph 快速失败与
4 元组协议、CPU backend 直通域与执行拒绝、会话缓存复用。
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.common.backend import detect_framework, get_backend
from ttk.core_modules.framework_api.adapter import get_adapter


def test_onnx_routing_and_adapter_registry():
    """前缀路由 + adapter 注册表：onnx.* 单源路由，eager 图列别名为 cst。"""
    assert detect_framework("onnx.Add") == "onnx"
    assert detect_framework("onnx.Split") == "onnx"
    assert detect_framework("onnx.") == "onnx"
    # torch.onnx.* 以 torch. 开头：不落 onnx 分支
    assert detect_framework("torch.onnx.export") == "torch"
    assert detect_framework("tf.raw_ops.AddV2") == "tf"
    assert detect_framework("") == "torch"

    from ttk.core_modules.framework_api.onnx import OnnxAdapter

    adapter = get_adapter("onnx")
    assert isinstance(adapter, OnnxAdapter)
    assert adapter.framework == "onnx"
    assert adapter.eager_graph_alias() == "cst"
    assert get_adapter("torch").eager_graph_alias() is None  # torch/tf 无等价
    assert get_adapter("tf").eager_graph_alias() is None


def test_run_graph_fast_failures():
    adapter = get_adapter("onnx")
    with pytest.raises(ValueError, match="ONNX_ACLGRAPH_UNSUPPORTED"):
        adapter.run_graph(None, None, 0, None, None, None, False, False, None, dynamic=False, is_aclgraph=True)
    with pytest.raises(ValueError, match="ONNX_DYNAMIC_GRAPH_UNSUPPORTED"):
        adapter.run_graph(None, None, 0, None, None, None, False, False, None, dynamic=True)


def test_run_graph_returns_run_graph_phase_protocol(monkeypatch):
    """run_graph 成功路径返回 4 元组 (nps, perf, det, err)。

    worker._run_graph_phase 按 4 元组解包（torch/tf 的 execute_graph 同构）；
    onnx 委托 _execute_eager（3 元组）曾漏补错误位——`--eager=false -c` 单开
    全量 GRAPH_EXEC_FAILURE（not enough values to unpack）。
    """
    from ttk.core_modules.framework_api import execute

    adapter = get_adapter("onnx")
    marker = ([np.zeros(2)], {"wall_us": 1.0}, None)
    forwarded = {}

    def _fake_execute_eager(*args, **kwargs):
        forwarded["args"] = args
        forwarded["kwargs"] = kwargs
        return marker

    monkeypatch.setattr(execute, "_execute_eager", _fake_execute_eager)
    nps, perf, det, err = adapter.run_graph(  # 4 元组解包：3 元组会在此抛 ValueError
        "tc", "backend", 3, "switches", "plan", "resolved", False, False, ["raw"], dynamic=False
    )
    assert (nps, perf, det, err) == (*marker, None)  # err 位恒 None（异常经 _run_graph_phase 收敛）
    assert forwarded["args"][:3] == ("tc", "backend", 3)  # 位置参数原样透传
    assert forwarded["kwargs"] == {"deterministic_level": None}


def test_cpu_backend_domain_and_execution_rejection(onnx_registry):
    """numpy 直通域（各 API 恒等）+ CPU 设备上 eager_call 拒绝。"""
    backend = get_backend(force_cpu=True, framework="onnx")
    assert backend.device_type() == "cpu"
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert backend.from_numpy(arr) is arr
    assert backend.to_device(arr) is arr
    assert backend.to_numpy(arr) is arr
    assert backend.supports_graph_mode() is False
    assert backend.supports_format_cast() is False
    assert backend.needs_numpy_fallback(object()) is False
    assert backend.has_device() is False
    # result_to_numpy：ndarray 直通（不拷贝堆叠）
    assert backend.result_to_numpy((arr, arr))[0] is arr

    adapter = get_adapter("onnx")
    handle, _ = adapter.load_api("onnx.Add")
    with pytest.raises(RuntimeError, match="ONNX_CPU_EXECUTION_UNSUPPORTED"):
        adapter.eager_call(handle, [np.zeros((2, 3), np.float32)] * 2, {}, False)


class _FakeNpuBackend:
    """会话流单测的最小 NPU backend 替身。"""

    _dev_id = 0

    def is_npu(self):
        return True

    def device_name(self, dev_id=0):  # noqa: ARG002
        return "Ascend910_9362"


class _FakeSession:
    """会话流单测的最小 OmSession 替身（计数断言用）。"""

    def __init__(self, host_inputs):
        self.host = list(host_inputs)
        self.rounds = 0
        self.rebound = 0
        self.unloaded = 0

    def matches_inputs(self, host_inputs):
        if len(self.host) != len(host_inputs):
            return False
        if all(a is b for a, b in zip(self.host, host_inputs)):
            return True
        return all(np.array_equal(a, b) for a, b in zip(self.host, host_inputs))

    def rebind_inputs(self, host_inputs):
        self.rebound += 1
        self.host = list(host_inputs)

    def execute_round(self):
        self.rounds += 1
        return (np.zeros(1),)

    def unload(self):
        self.unloaded += 1


@pytest.fixture
def session_flow(monkeypatch, onnx_registry):
    """Add 会话流脚手架：_create_session 换 Fake、_sessions 换独立 OrderedDict。

    经 adapter.load_api → extract_onnx_params（onnx_registry 门禁处理，
    onnx 缺席 / opset 超限环境 skip）。
    """
    adapter = get_adapter("onnx")
    handle, _ = adapter.load_api("onnx.Add")
    handle.attach_backend(_FakeNpuBackend())
    created = []

    def _create(_handle, _input_specs, host_inputs, _kwargs):
        sess = _FakeSession(host_inputs)
        created.append(sess)
        return sess

    monkeypatch.setattr(adapter, "_create_session", _create)
    sessions = OrderedDict()
    monkeypatch.setattr(type(adapter), "_sessions", sessions)
    return adapter, handle, created, sessions


def test_session_created_once_across_rounds(session_flow):
    adapter, handle, created, sessions = session_flow
    a = np.zeros((2, 3), np.float32)
    b = np.ones((2, 3), np.float32)
    adapter.eager_call(handle, [a, b], {}, False)
    adapter.eager_call(handle, [a, b], {}, False)
    adapter.eager_call(handle, [a, b], {}, False)
    assert len(created) == 1  # warmup+计时轮共享同一会话（首轮建，后续复用）
    assert created[0].rounds == 3
    assert len(sessions) == 1


def test_case_dir_name_uses_worker_process_name(monkeypatch, onnx_registry):
    """om 缓存目录按用例名归组（worker 进程名即用例名）；MainProcess / 空名
    回退 op_type（legacy 布局）。"""
    import multiprocessing

    from ttk.core_modules.framework_api.onnx import OnnxAdapter

    handle = SimpleNamespace(op_type="Add")
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(name="onnx_add_f32"))
    assert OnnxAdapter._case_dir_name(handle) == "onnx_add_f32"
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(name="MP onnx_add_f32"))
    assert OnnxAdapter._case_dir_name(handle) == "onnx_add_f32"
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(name="MainProcess"))
    assert OnnxAdapter._case_dir_name(handle) == "Add"


def test_load_handle_carries_explicit_opset(onnx_registry):
    """onnx.<opset>.<OpType> 的显式版本段进 handle.opset；不同 opset 的同一
    算子不得复用同一会话（会话键含 opset——om 不同而静默复用是假 PASS 源）。"""
    adapter = get_adapter("onnx")
    bare, _ = adapter.load_api("onnx.Add")
    h13, _ = adapter.load_api("onnx.13.Add")
    h18, _ = adapter.load_api("onnx.18.Add")
    assert bare.opset is None
    assert h13.opset == 13
    assert h18.opset == 18
    for h in (bare, h13, h18):
        h.attach_backend(_FakeNpuBackend())
    specs = [("input_0", (2, 3), "float32"), ("input_1", (2, 3), "float32")]
    keys = {adapter._session_key(h, specs, {}) for h in (bare, h13, h18)}
    assert len(keys) == 3
