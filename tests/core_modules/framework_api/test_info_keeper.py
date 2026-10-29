#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.framework_api.framework_api_info_keeper:
cache, register, get.
"""

import logging
from unittest.mock import Mock

import pytest

from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper
from ttk.core_modules.framework_api.torch.param_extractor import (
    _MANUAL_OVERRIDES,
    APIParamInfo,
    ParamInfo,
)


def _keeper():
    return FrameworkApiInfoKeeper()


@pytest.fixture
def fake_adapter(monkeypatch):
    from ttk.core_modules.framework_api import adapter

    fake = Mock()
    fake.extract_params.return_value = None
    monkeypatch.setattr(adapter, "adapter_for_api_name", lambda api_name: fake)
    return fake


class TestFrameworkApiInfoKeeperCache:
    def setup_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_get_caches_result(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.api",
            params=[ParamInfo(name="input", type="Tensor")],
            source="test",
        )
        keeper._cache["test.api"] = info
        result = keeper.get("test.api")
        assert result is info

    def test_register_with_param_list(self):
        keeper = _keeper()
        params = [ParamInfo(name="x", type="Tensor"), ParamInfo(name="dim", type="int")]
        keeper.register("test.registered2", params, source="config")
        result = keeper.get("test.registered2")
        assert result is not None
        assert result.tensor_count == 1
        assert result.source == "config"

    def test_get_torch_api_if_available(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        keeper = _keeper()
        info = keeper.get("torch.add")
        assert info is not None
        assert info.tensor_count >= 2
        cached = keeper.get("torch.add")
        assert cached is info

    @pytest.mark.parametrize("api", ["torch.missing", "tf.raw_ops.Missing"])
    def test_failure_diagnostic_is_cached_once(self, fake_adapter, api):
        fake_adapter.load_api.side_effect = AttributeError("API is not registered")
        keeper = _keeper()
        assert keeper.get_failure_reason(api) is None
        fake_adapter.extract_params.assert_not_called()
        fake_adapter.load_api.assert_not_called()
        for _ in range(3):
            assert keeper.get(api) is None
            assert keeper.get_failure_reason(api) == "AttributeError: API is not registered"
        fake_adapter.extract_params.assert_called_once_with(api)
        fake_adapter.load_api.assert_called_once_with(api)

    def test_silent_failure_has_self_contained_reason(self, fake_adapter):
        keeper = _keeper()
        assert keeper.get("torch.test_api") is None
        assert keeper.get_failure_reason("torch.test_api") == (
            "API resolved, but parameter inspection did not produce a supported signature"
        )
        fake_adapter.load_api.assert_called_once()

    @pytest.mark.parametrize(
        "error", [ModuleNotFoundError("optional framework missing"), RuntimeError("parser failed")]
    )
    def test_parser_exception_is_cached_without_reloading(self, fake_adapter, caplog, error):
        fake_adapter.extract_params.side_effect = error
        keeper = _keeper()
        with caplog.at_level(logging.WARNING):
            assert keeper.get("torch.test_api") is None
            assert keeper.get("torch.test_api") is None
        assert keeper.get_failure_reason("torch.test_api") == f"{type(error).__name__}: {error}"
        fake_adapter.extract_params.assert_called_once()
        fake_adapter.load_api.assert_not_called()
        assert sum(record.exc_info is not None for record in caplog.records) == int(isinstance(error, RuntimeError))

    def test_success_does_not_probe_failure(self, fake_adapter):
        info = APIParamInfo(api_name="torch.test_api", params=[ParamInfo(name="x", type="Tensor")], source="test")
        fake_adapter.extract_params.return_value = info
        keeper = _keeper()
        assert keeper.get("torch.test_api") is info
        assert keeper.get_failure_reason("torch.test_api") is None
        fake_adapter.load_api.assert_not_called()

    def test_clear_cache_invalidates_failure_diagnostic(self, fake_adapter):
        keeper = _keeper()
        keeper.get("torch.test_api")
        keeper.clear_cache()
        assert keeper.get_failure_reason("torch.test_api") is None
        keeper.get("torch.test_api")
        assert fake_adapter.extract_params.call_count == 2
        assert fake_adapter.load_api.call_count == 2

    def test_register_replaces_failed_lookup(self, fake_adapter):
        keeper = _keeper()
        keeper.get("torch.test_api")
        info = APIParamInfo(api_name="torch.test_api", params=[ParamInfo(name="x", type="Tensor")], source="test")
        keeper.register("torch.test_api", info)
        assert keeper.get("torch.test_api") is info
        assert keeper.get_failure_reason("torch.test_api") is None
        fake_adapter.extract_params.assert_called_once()
        fake_adapter.load_api.assert_called_once()
