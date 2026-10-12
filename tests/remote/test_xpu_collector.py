#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""xpu_collector 测试：格式化、端点选择和 XPU 模式分发。"""

from types import SimpleNamespace

import numpy as np
import pytest

from ttk.remote import DATA, ExecutionSpec, xpu_collector


@pytest.mark.parametrize(
    ("raw", "expected_status", "has_perf"),
    [
        (
            {
                "torch": {
                    "status": "PASS",
                    "api": "torch.add",
                    "outputs": [],
                    "perf": {"device_us": 120.0, "peak_memory_mb": 8.5},
                }
            },
            "PASS",
            True,
        ),
        ({"tf": {"status": "FAIL", "api": "tf.raw.ops.Add", "error": "import failed"}}, "FAIL", False),
    ],
    ids=["pass-with-perf", "fail-no-perf"],
)
def test_format_xpu_metrics(raw, expected_status, has_perf):
    """_format_xpu_metrics: PASS 带 perf / FAIL 不带 perf。"""
    from ttk.core_modules.common.result import format_xpu_metrics

    result = format_xpu_metrics(raw)
    for provider in raw:
        assert result[provider]["status"] == expected_status
        if has_perf:
            assert "device_us" in result[provider]
        else:
            assert "device_us" not in result[provider]


def test_dump_xpu_data_mode_runs_all_selected_providers(monkeypatch):
    specs = [ExecutionSpec(provider="torch", api="torch.add"), ExecutionSpec(provider="tf", api="tf.raw_ops.Add")]
    calls = []

    class FakeEndpointView:
        def pick_endpoint(self, provider):
            return SimpleNamespace(host="127.0.0.1", port=1)

    def fake_dispatch(**kwargs):
        calls.append((kwargs["provider"], kwargs["mode"]))
        return SimpleNamespace(outputs=[np.array([1], dtype=np.float32)], perf=None, api=kwargs["api"])

    monkeypatch.setattr("ttk.remote.endpoint_view.EndpointView", FakeEndpointView)
    monkeypatch.setattr(xpu_collector, "dispatch_to_remote", fake_dispatch)

    results = xpu_collector.collect_xpu_results(
        specs,
        inputs=[],
        input_names=[],
        mode=DATA,
        tenant_id="test",
        op_name="add",
        dump_xpu=True,
    )

    assert set(results) == {"torch", "tf"}
    assert set(calls) == {("torch", DATA), ("tf", DATA)}
