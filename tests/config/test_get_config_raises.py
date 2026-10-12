# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for get_config() raising RuntimeError before load_config()."""

import pytest


def test_get_config_raises_before_load():
    """get_config() 在 load_config() 之前调 → raise RuntimeError（不再 lazy fallback）。"""
    from ttk.config import loader

    saved = loader._config
    loader._config = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            loader.get_config()
    finally:
        loader._config = saved  # 恢复，别污染同 session 其它测试
