# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""EndpointView 测试：resolve_providers 交集/优先级、pick_endpoint 轮询。"""

import json
import os

import pytest
import yaml

from ttk.remote.endpoint_view import EndpointView, _parse_provider_filter


def _write_health(path, endpoints_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"endpoints": endpoints_data}, f)


def _make_ev(monkeypatch, tmp_path, health_data, endpoints_list):
    """Wire config + health file, construct a fresh EndpointView (singleton cleared)."""
    from ttk.utilities.singleton import Singleton

    Singleton._instances.clear()
    from ttk.config import loader

    loader._config = None
    config_path = tmp_path / "endpoints.yaml"
    config_path.write_text(yaml.safe_dump({"remote": {"endpoints": endpoints_list}}))
    loader.load_config(str(config_path))
    health_path = str(tmp_path / "h.json")
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", health_path)
    _write_health(health_path, health_data)
    return EndpointView()


# -- resolve_providers ------------------------------------------------------


def test_resolve_intersects_and_preserves_priority(monkeypatch, tmp_path):
    """resolve_providers: 无 spec→sorted union；spec 交集 + 保持 spec 声明顺序（非 sorted）。"""
    ev = _make_ev(
        monkeypatch,
        tmp_path,
        {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
        [{"host": "10.0.0.1", "port": 9090}],
    )
    # 无 spec → sorted union
    assert ev.resolve_providers() == ["tf", "torch"]
    # spec 交集
    assert ev.resolve_providers(spec_providers=["torch"]) == ["torch"]
    # spec 顺序 = 优先级（非 sorted）
    assert ev.resolve_providers(spec_providers=["tf", "torch"]) == ["tf", "torch"]
    assert ev.resolve_providers(spec_providers=["torch", "tf"]) == ["torch", "tf"]
    # cli 交集
    assert ev.resolve_providers(cli_providers=["tf"]) == ["tf"]


# -- _parse_provider_filter -------------------------------------------------


def test_parse_provider_filter_aliases_and_empty():
    """--provider 解析：别名归一（tensorflow→tf、np→numpy）；空/None→None。"""
    assert _parse_provider_filter("tensorflow") == ["tf"]
    assert _parse_provider_filter("np") == ["numpy"]
    assert _parse_provider_filter(" tensorflow , np ") == ["tf", "numpy"]
    assert _parse_provider_filter("torch,tf") == ["torch", "tf"]
    assert _parse_provider_filter("") is None
    assert _parse_provider_filter(None) is None
    assert _parse_provider_filter(" , ") is None


def test_cli_alias_filter_resolves_canonical_provider(monkeypatch, tmp_path):
    """--provider tensorflow（别名）经解析后能命中 detect 的规范名 tf。"""
    ev = _make_ev(
        monkeypatch,
        tmp_path,
        {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
        [{"host": "10.0.0.1", "port": 9090}],
    )
    assert ev.resolve_providers(cli_providers=_parse_provider_filter("tensorflow")) == ["tf"]
    with pytest.raises(RuntimeError):
        ev.resolve_providers(cli_providers=_parse_provider_filter("np"))


# -- pick_endpoint ----------------------------------------------------------


def test_pick_endpoint_round_robin_and_none(monkeypatch, tmp_path):
    """pick_endpoint: 双 endpoint 轮询；dead→None；provider 不存在→None。"""
    # 轮询
    ev = _make_ev(
        monkeypatch,
        tmp_path,
        {
            "10.0.0.1:9090": {"alive": True, "providers": ["torch"]},
            "10.0.0.2:9090": {"alive": True, "providers": ["torch"]},
        },
        [{"host": "10.0.0.1", "port": 9090}, {"host": "10.0.0.2", "port": 9090}],
    )
    assert ev.pick_endpoint("torch") is not ev.pick_endpoint("torch")

    # dead → None
    ev = _make_ev(
        monkeypatch,
        tmp_path,
        {"10.0.0.1:9090": {"alive": False, "providers": ["torch"]}},
        [{"host": "10.0.0.1", "port": 9090}],
    )
    assert ev.pick_endpoint("torch") is None

    # provider 不存在 → None
    ev = _make_ev(
        monkeypatch,
        tmp_path,
        {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]}},
        [{"host": "10.0.0.1", "port": 9090}],
    )
    assert ev.pick_endpoint("tf") is None
