#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""--sig 签名源装载/查询测试：惰性装载、全名命中、尾名兜底（唯一/歧义）、回落六源链。"""

from types import SimpleNamespace

import pytest

from ttk.core_modules.framework_api.api_model import APIParamInfo, ParamInfo
from ttk.core_modules.framework_api.torch import param_extractor


def _info(name, source="signature:test"):
    return APIParamInfo(api_name=name, source=source, params=[ParamInfo(name="x", type="Tensor")])


@pytest.fixture
def _registry_sandbox():
    """惰性装载是模块级状态：快照/恢复 _MANUAL_OVERRIDES 与签名索引，隔离用例副作用。

    置位 _SIGNATURE_OVERRIDES_READY 并清空尾名索引再进入用例——全序套件中
    更早的用例可能已触发过装载（READY=True 残留会让本组用例的惰性装载被跳过）。
    """
    overrides = dict(param_extractor._MANUAL_OVERRIDES)
    tail = {k: list(v) for k, v in param_extractor._SIGNATURE_TAIL_INDEX.items()}
    ready = param_extractor._SIGNATURE_OVERRIDES_READY
    param_extractor._SIGNATURE_OVERRIDES_READY = False
    param_extractor._SIGNATURE_TAIL_INDEX.clear()
    yield
    param_extractor._MANUAL_OVERRIDES.clear()
    param_extractor._MANUAL_OVERRIDES.update(overrides)
    param_extractor._SIGNATURE_TAIL_INDEX.clear()
    param_extractor._SIGNATURE_TAIL_INDEX.update(tail)
    param_extractor._SIGNATURE_OVERRIDES_READY = ready


def _install_sources(monkeypatch, sources, entries_by_source):
    """switches 注入 + parse_signature_source 打桩（免 torch：后端已单测覆盖）。"""
    calls = []

    def fake_parse(source):
        calls.append(source)
        if source not in entries_by_source:
            raise RuntimeError(f"cannot read {source}")
        return dict(entries_by_source[source])

    monkeypatch.setattr("ttk.core_modules.framework_api.torch.signature_parser.parse_signature_source", fake_parse)
    monkeypatch.setattr("ttk.switches.get_global_storage", lambda: SimpleNamespace(signature_sources=tuple(sources)))
    return calls


@pytest.mark.usefixtures("_registry_sandbox")
def test_full_name_hit_short_circuits_extraction(monkeypatch):
    info = _info("torch.ops.ttktest.myadd")
    _install_sources(monkeypatch, ["stub.py"], {"stub.py": {"torch.ops.ttktest.myadd": info}})
    monkeypatch.setattr(
        param_extractor,
        "extract_api_params",
        lambda *_: (_ for _ in ()).throw(AssertionError("signature hit must not fall back to extraction")),
    )

    assert param_extractor.get_api_params("torch.ops.ttktest.myadd") is info


@pytest.mark.usefixtures("_registry_sandbox")
def test_tail_match_unique_hits_with_info_log(monkeypatch, caplog):
    """裸名登记（无 namespace 头）+ 查询全名 → 尾名唯一匹配 + INFO 告警。"""
    info = _info("sparse_flash_mla")
    _install_sources(monkeypatch, ["stub.py"], {"stub.py": {"sparse_flash_mla": info}})

    with caplog.at_level("INFO"):
        result = param_extractor.get_api_params("torch.ops.cann_ops_transformer.sparse_flash_mla")

    assert result is info
    assert "tail match" in caplog.text


@pytest.mark.usefixtures("_registry_sandbox")
def test_tail_match_ambiguous_rejected_falls_back(monkeypatch, caplog):
    _install_sources(
        monkeypatch,
        ["a.py", "b.py"],
        {"a.py": {"ns_a.myadd": _info("ns_a.myadd")}, "b.py": {"ns_b.myadd": _info("ns_b.myadd")}},
    )
    sentinel = _info("extracted", source="extract-chain")
    monkeypatch.setattr(param_extractor, "extract_api_params", lambda _name: sentinel)

    with caplog.at_level("ERROR"):
        result = param_extractor.get_api_params("torch.ops.ns_c.myadd")

    assert result is sentinel
    assert "ambiguous tail match" in caplog.text


@pytest.mark.usefixtures("_registry_sandbox")
def test_broken_source_skipped_and_others_still_load(monkeypatch, caplog):
    info = _info("torch.ops.ttktest.myadd")
    _install_sources(monkeypatch, ["bad.py", "good.py"], {"good.py": {"torch.ops.ttktest.myadd": info}})

    with caplog.at_level("ERROR"):
        assert param_extractor.get_api_params("torch.ops.ttktest.myadd") is info
    assert "failed to load signature source bad.py" in caplog.text


@pytest.mark.usefixtures("_registry_sandbox")
def test_load_is_idempotent(monkeypatch):
    info = _info("torch.ops.ttktest.myadd")
    calls = _install_sources(monkeypatch, ["stub.py"], {"stub.py": {"torch.ops.ttktest.myadd": info}})

    param_extractor.get_api_params("torch.ops.ttktest.myadd")
    param_extractor.get_api_params("torch.ops.ttktest.myadd")

    assert calls == ["stub.py"]


@pytest.mark.usefixtures("_registry_sandbox")
def test_no_sources_keeps_fall_back(monkeypatch):
    calls = _install_sources(monkeypatch, [], {})
    sentinel = _info("extracted", source="extract-chain")
    monkeypatch.setattr(param_extractor, "extract_api_params", lambda _name: sentinel)

    assert param_extractor.get_api_params("torch.ops.ttktest.myadd") is sentinel
    assert calls == []
