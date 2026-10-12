# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""xpu_profiler 双 pass 契约测试（server 同源消费面）。

- torch_xpu_profiler_pass：invoke 次数 = wait+warmup+runtime、outputs 透传、
  CPU-only 下 device_us=0、未知名 activity → RuntimeError（config error 语义）
- tf_xpu_profiler_pass：invoke 次数 = 2 warmup + runtime、返回形状
- _is_device_kernel：容器 span / CPU 算子 / cuda API 事件三种身份剔除
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_is_device_kernel_filters_host_spans():
    """key_averages 拍平后同一段 device 时间以三种身份出现；只保留真实内核条目。"""
    from ttk.core_modules.common.profiler.torch_profiler import _is_device_kernel

    assert _is_device_kernel(SimpleNamespace(key="void elementwise_kernel<void>(int)"))
    assert not _is_device_kernel(SimpleNamespace(key="ProfilerStep#5"))  # schedule 容器 span
    assert not _is_device_kernel(SimpleNamespace(key="aten::add"))  # CPU 侧算子视角
    assert not _is_device_kernel(SimpleNamespace(key="cudaLaunchKernel"))  # host API 事件


def test_torch_xpu_profiler_pass_smoke_cpu():
    from ttk.remote.server.xpu_profiler import torch_xpu_profiler_pass

    calls = []

    def invoke():
        calls.append(1)
        return "out"

    outputs, device_us = torch_xpu_profiler_pass(invoke, runtime=3, device="cuda", activities_cfg=["CPU"])
    assert outputs == "out"
    assert len(calls) == 5  # wait(1) + warmup(1) + active(3)
    assert device_us == 0.0  # CPU-only: no device kernel events


def test_torch_xpu_profiler_pass_unknown_activity_raises():
    """配置错误（未知 activity）经共享 resolve_activities raise ValueError——
    server config error → 500，非 NA degrade。"""
    from ttk.remote.server.xpu_profiler import torch_xpu_profiler_pass

    with pytest.raises(ValueError, match="unknown ProfilerActivity 'WARP'"):
        torch_xpu_profiler_pass(lambda: None, runtime=1, device="cuda", activities_cfg=["WARP"])


def test_tf_xpu_profiler_pass_smoke_cpu():
    """子进程隔离运行：本仓 pytest 进程会加载 torch_npu，与 TF profiler 同进程
    共存会段错误（test_kernel_xpu_e2e.py 的 TF 检查同样走子进程）。"""
    import subprocess
    import sys

    script = """
import tensorflow as tf
from ttk.remote.server.xpu_profiler import tf_xpu_profiler_pass
calls = []
def invoke():
    calls.append(1)
    return tf.constant(1.0) + 1.0
outputs, device_us = tf_xpu_profiler_pass(invoke, runtime=2)
assert outputs is not None
assert len(calls) == 4, len(calls)  # 2 warmup（丢弃）+ 2 profiled
assert device_us >= 0.0
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
