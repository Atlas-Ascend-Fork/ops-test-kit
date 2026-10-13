# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""load_config 显式路径缺失 fail-loud（原实现静默回退默认配置）。"""

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
