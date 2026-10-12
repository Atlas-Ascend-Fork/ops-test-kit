# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""graph_execution._run_compiled 的 inplace 恢复与 aclgraph 地址稳定契约。

自 tests/core_modules/common/backend/test_profiler_internals.py 移入——它是
framework_api 的消费者测试（借 DisabledProfiler 做 no-op 桩），不属于
profiler 单元测试。
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("kind", ["inplace", "indexes", "kwargs"])
@pytest.mark.parametrize("aclgraph", [False, True])
def test_compiled_inplace_restores_values_and_aclgraph_addresses(tmp_path, kind, aclgraph, monkeypatch):
    from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend
    from ttk.core_modules.common.profiler.base import DisabledProfiler
    from ttk.core_modules.framework_api.torch import graph_runner as graph_execution

    monkeypatch.setattr(graph_execution, "get_profiler", lambda *args: DisabledProfiler())
    backend = CpuTorchBackend()
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    backup = value.clone()
    pointers = []

    def compiled(*args, **kwargs):
        target = kwargs["value"] if kind == "kwargs" else args[0]
        assert torch.equal(target, backup)
        pointers.append(target.data_ptr())
        return target.add_(1)

    switches = SimpleNamespace(TASK_PROFILING=False, run_time=3, root_path=str(tmp_path), warmup=True)
    result, _, status = graph_execution._run_compiled(
        compiled,
        [] if kind == "kwargs" else [value],
        {"value": value} if kind == "kwargs" else {},
        backend,
        0,
        switches,
        kind == "inplace",
        backup if kind == "inplace" else None,
        "torch.add",
        inplace_backups={0: backup} if kind != "inplace" else None,
        inplace_kwargs_keys={0: "value"} if kind == "kwargs" else None,
        deterministic_level=1,
        is_aclgraph=aclgraph,
    )
    assert status == "PASS"
    assert torch.equal(torch.from_numpy(result[0]), backup + 1)
    # 非 aclgraph：warmup 专属克隆 + run_time-1 个循环克隆 + 末轮原始张量；
    # aclgraph：全程同一张量（循环顶 copy_ 复位）
    assert len(set(pointers)) == (1 if aclgraph else 4)
    assert pointers[-1] == value.data_ptr()
