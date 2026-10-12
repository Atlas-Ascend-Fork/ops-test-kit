# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TF backend ↔ 通路层依赖路径回归（断链族终态）。

搬家后 ``common/backend/tf_backend.py`` 的 ``inputs_from_numpy`` /
``wrap_eager_callable`` 相对导入落空（``common.input_generation`` /
``common.tf_stateful`` 不存在），调用时 ModuleNotFoundError。终态架构：
  - L1 张量构造经 ``framework_hooks`` 注册表注入（tf/tensor_binding 注册）；
  - eager 包装迁 ``framework_api/tf/eager_runner``（经 adapter.wrap_eager），
    backend 的 wrap_eager_callable 回归 no-op 契约。
monkeypatch 拦截目标函数，只验证依赖路径可解析（不触发 tensorflow import：
torch_npu 与 TF C 扩展同进程冲突，按仓内惯例 skip 全调用型用例）。
"""

import importlib.util

import pytest

from ttk.core_modules.common.backend.framework_hooks import get_input_factory
from ttk.core_modules.common.backend.tf_backend import TfBackend

_TORCH_NPU_TF_CONFLICT = (
    importlib.util.find_spec("torch_npu") is not None and importlib.util.find_spec("tensorflow") is not None
)
_requires_tf = pytest.mark.skipif(
    _TORCH_NPU_TF_CONFLICT, reason="torch_npu._C + tensorflow C extension conflict → segfault"
)


def test_tf_adapter_registers_input_factory():
    """get_adapter("tf")（即 tf/ 包导入）完成 L1 工厂注册（终态回归）。"""
    from ttk.core_modules.framework_api.adapter import get_adapter

    assert get_adapter("tf").framework == "tf"
    assert get_input_factory("tf") is not None


def test_inputs_from_numpy_via_factory(monkeypatch):
    """TfBackend.inputs_from_numpy 经注册表取 L1 构造，不再硬编码通路路径。"""
    inputs = ["raw0", None]
    seen = {}

    def fake_factory(testcase, raw_inputs):
        seen["called"] = (testcase, raw_inputs)
        return list(raw_inputs)

    monkeypatch.setitem(
        __import__("ttk.core_modules.common.backend.framework_hooks", fromlist=["_INPUT_FACTORIES"])._INPUT_FACTORIES,
        "tf",
        fake_factory,
    )

    out = TfBackend().inputs_from_numpy(object(), inputs)
    assert seen["called"][1] is inputs
    assert out == ["raw0", None]


def test_backend_wrap_eager_callable_is_noop():
    """backend 侧 wrap_eager_callable 回归 no-op 契约（包装迁 tf/eager_runner）。"""

    def resolved(*args, **kwargs):
        return args

    assert TfBackend().wrap_eager_callable(resolved, "tf.add") is resolved


def test_eager_runner_cpu_wrap_import_path(monkeypatch):
    """tf/eager_runner CPU 包装的参数域依赖可解析（无 resource 位原样返回）。"""
    from ttk.core_modules.framework_api.tf import eager_runner, param_extractor

    monkeypatch.setattr(param_extractor, "resource_param_names", lambda _api: frozenset())

    def resolved(*args, **kwargs):
        return args

    wrapped = eager_runner.wrap_eager_callable(resolved, "tf.add", TfBackend())
    assert wrapped is resolved


@_requires_tf
def test_eager_runner_npu_wrap_full_call():
    """tf/eager_runner NPU 包装全调用（含 tensorflow import，按仓内惯例跳过）。"""
    from ttk.core_modules.common.backend.npu_tf_backend import NpuTfBackend
    from ttk.core_modules.framework_api.tf.eager_runner import wrap_eager_callable

    def resolved(x):
        return x

    wrapped = wrap_eager_callable(resolved, "tf.add", NpuTfBackend())
    assert callable(wrapped)
    assert wrapped is not resolved


def test_adapter_registry_rejects_unknown_framework():
    from ttk.core_modules.framework_api.adapter import get_adapter

    with pytest.raises(ValueError, match="Unknown framework"):
        get_adapter("mindspore")


# ── CPU 放置原语（context.device，npu_device patch 免疫）────────────────
# TF C 扩展与 torch_npu 同进程冲突 → 子进程运行（test_param_mutable_flags 同款）。


def _tf_available():
    import subprocess
    import sys

    probe = subprocess.run([sys.executable, "-c", "import tensorflow"], capture_output=True, timeout=300, check=False)
    return probe.returncode == 0


def test_cpu_backend_places_tensors_and_variables_on_cpu():
    """CpuTfBackend 的 from_numpy/device_scope 把张量与 Variable 放到 CPU。"""
    if not _tf_available():
        pytest.skip("tensorflow not importable")

    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import numpy as np
        import tensorflow as tf

        from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend

        backend = CpuTfBackend()
        t = backend.from_numpy(np.ones(2))
        with backend.device_scope(0):
            v = tf.Variable(tf.zeros(2))
        assert "CPU" in t.device, t.device
        assert "CPU" in v.device, v.device
        """
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=300, check=False, text=True)
    assert run.returncode == 0, f"subprocess failed: {run.stderr[-2000:]}"


def test_cpu_placement_immune_to_ops_device_patch():
    """npu_device.as_default 的 ops.device 覆写不影响 CPU 放置（作用域解耦回归）。

    模拟 patch（忽略设备参数、恒路由"默认设备"），验证：
    1. patched ops.device 未被调用（放置不经可污染的公共路径）；
    2. 张量/Variable 仍落 CPU。
    """
    if not _tf_available():
        pytest.skip("tensorflow not importable")

    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import numpy as np
        import tensorflow as tf
        import tensorflow.python.framework.ops as ops
        from tensorflow.python.eager import context as tf_context

        calls = []
        _orig = ops.device

        def patched_ops_device(name):
            calls.append(name)
            return tf_context.device("/job:localhost/replica:0/task:0/device:CPU:0")

        ops.device = patched_ops_device

        from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend

        backend = CpuTfBackend()
        with backend.device_scope(0):
            t = backend.from_numpy(np.ones(2))
            v = tf.Variable(tf.zeros(2))
        assert calls == [], f"placement routed through patchable ops.device: {calls}"
        assert "CPU" in t.device, t.device
        assert "CPU" in v.device, v.device
        """
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=300, check=False, text=True)
    assert run.returncode == 0, f"subprocess failed: {run.stderr[-2000:]}"
