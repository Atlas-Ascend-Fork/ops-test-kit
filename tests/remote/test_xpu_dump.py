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

import numpy as np
import pytest

from ttk.remote import DATA, client
from ttk.switches import SWITCHES


@pytest.fixture(autouse=True)
def _clear_plugin_func_cache():
    """monkeypatch spec_manager.get_spec_attr 前后清 get_plugin_function 查询缓存。

    dispatch_xpu 经 SpecResolver→get_plugin_function 查询 third_party，模块级
    _func_cache 会把本文件的 patch 结果残留给后续用例。"""
    from ttk.core_modules.common.plugin.loader import clear_plugin_cache

    clear_plugin_cache()
    yield
    clear_plugin_cache()


def _switches(tmp_path):
    from ttk.core_modules.common.dump import DumpConfig

    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.dump_config = DumpConfig()
    switches.dump_config.dump_xpu = True
    switches.dump_config.file_format = "bin"
    return switches


def test_dump_xpu_outputs_writes_nested_successful_outputs(tmp_path):
    outputs = [np.array([1.5, 2.5], dtype=np.float32), [np.array([3, 4], dtype=np.int32)]]
    results = {
        "torch/cuda": {
            "status": "PASS",
            "outputs": outputs,
        }
    }

    client._dump_xpu_outputs(results, "case_1", _switches(tmp_path))

    np.testing.assert_array_equal(np.fromfile(tmp_path / "case_1_xpu_golden_0.bin", dtype=np.float32), outputs[0])
    np.testing.assert_array_equal(np.fromfile(tmp_path / "case_1_xpu_golden_1.bin", dtype=np.int32), outputs[1][0])


def test_dump_xpu_outputs_splits_providers_into_subdirectories(tmp_path):
    first = np.array([1.5], dtype=np.float32)
    second = np.array([2.5], dtype=np.float32)
    results = {
        "torch": {"status": "PASS", "outputs": [first]},
        "tf": {"status": "PASS", "outputs": [second]},
    }

    client._dump_xpu_outputs(results, "case_1", _switches(tmp_path))

    np.testing.assert_array_equal(np.fromfile(tmp_path / "torch" / "case_1_xpu_golden_0.bin", dtype=np.float32), first)
    np.testing.assert_array_equal(np.fromfile(tmp_path / "tf" / "case_1_xpu_golden_0.bin", dtype=np.float32), second)


def test_dump_tokens_do_not_collide_after_sanitization():
    first = client._safe_dump_token("case/1", "case")
    second = client._safe_dump_token("case?1", "case")
    assert first != second


def test_dump_xpu_outputs_uses_environment_path_and_skips_failures(tmp_path, monkeypatch):
    dump_path = tmp_path / "xpu-dump"
    monkeypatch.setenv("NPU_DUMP_PATH", str(dump_path))
    results = {"torch": {"status": "FAIL", "error": "failed"}}

    client._dump_xpu_outputs(results, "case_1", _switches(tmp_path))

    assert not dump_path.exists()


def test_dispatch_xpu_requests_data_when_only_dump_is_enabled(monkeypatch, tmp_path):
    output = np.array([1, 2, 3], dtype=np.int32)
    results = {"torch": {"status": "PASS", "outputs": [output]}}
    captured = {}

    class FakeEndpointView:
        def resolve_providers(self, spec_providers, cli_providers):
            return ["torch"]

    def fake_collect(_specs, **kwargs):
        captured["mode"] = kwargs["mode"]
        return results

    monkeypatch.setattr("ttk.remote.endpoint_view.EndpointView", FakeEndpointView)
    monkeypatch.setattr("ttk.remote.xpu_collector.collect_xpu_results", fake_collect)
    monkeypatch.setattr(
        "ttk.core_modules.common.plugin.spec_manager.get_spec_attr", lambda *_args, **_kwargs: {"torch": "torch.add"}
    )
    monkeypatch.setattr(
        "ttk.core_modules.common.plugin.spec_manager.get_spec_class_meta", lambda *_args, **_kwargs: None
    )

    client.dispatch_xpu(
        op_name="add",
        inputs=[],
        input_names=[],
        op_type=None,
        attributes={},
        testcase_name="case_1",
        switches=_switches(tmp_path),
        need_data=False,
    )

    assert captured["mode"] == DATA
    np.testing.assert_array_equal(np.fromfile(tmp_path / "case_1_xpu_golden_0.bin", dtype=np.int32), output)
