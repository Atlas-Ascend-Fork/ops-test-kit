# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""loader 单测：load_config 显式路径缺失 fail-loud（原实现静默回退默认配置）；
deep_merge 合并结果与 base/override 双侧隔离且合并语义不回归。"""

import pytest

from ttk.config import loader


def test_explicit_missing_config_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        loader.load_config(str(tmp_path / "nope.yaml"))


def test_explicit_existing_config_loads(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("remote:\n  endpoints: []\n")
    try:
        cfg = loader.load_config(str(p))
        assert cfg["remote"]["endpoints"] == []
    finally:
        loader.load_config()


def test_deep_merge_result_is_independent():
    base = {"remote": {"endpoints": [{"host": "127.0.0.1"}], "tls": {"ca": "a"}}}
    override = {"framework": {"items": [1]}}
    merged = loader.deep_merge(base, override)
    merged["remote"]["endpoints"].append({"host": "9.9.9.9"})
    merged["remote"]["tls"]["ca"] = "mutated"
    merged["framework"]["items"].append(2)
    assert base == {"remote": {"endpoints": [{"host": "127.0.0.1"}], "tls": {"ca": "a"}}}
    assert override == {"framework": {"items": [1]}}


def test_deep_merge_semantics_unchanged():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2], "extra": {"keep": True}}
    override = {"a": 2, "nested": {"y": 3}, "lst": [9]}
    merged = loader.deep_merge(base, override)
    assert merged == {"a": 2, "nested": {"x": 1, "y": 3}, "lst": [9], "extra": {"keep": True}}
