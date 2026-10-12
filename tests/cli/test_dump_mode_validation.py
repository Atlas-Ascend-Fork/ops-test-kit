#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# SPDX-License-Identifier: CANN-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""--dump 模式校验回归（upstream 9bfa978 同步适配：_apply_dump_config → DumpConfig.from_cli）。"""

import pytest

from ttk.core_modules.common.dump import DumpConfig


def test_unknown_dump_mode_is_rejected():
    with pytest.raises(ValueError, match=r"unknown token 'typo'"):
        DumpConfig.from_cli("in, typo")


def test_dump_modes_are_trimmed_before_application():
    cfg = DumpConfig.from_cli(" in, xpu ")
    assert cfg.dump_input is True
    assert cfg.dump_xpu is True
    assert cfg.dump_output is False
    assert cfg.dump_golden is False


def test_dump_full_enables_all():
    cfg = DumpConfig.from_cli("full")
    assert cfg.dump_input is True
    assert cfg.dump_output is True
    assert cfg.dump_golden is True
    assert cfg.dump_xpu is True
