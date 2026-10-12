#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""kernel worker XPU 辅助单元：_xpu_inputs 数据选择与 _xpu_mode 位运算 gate。

自 tests/cli/test_xpu_perf_collect.py 拆出——本文件测 kernel.worker 内部辅助，
CLI 参数/开关/校验用例留在 tests/cli/test_xpu_perf_collect.py。"""

from types import SimpleNamespace

from ttk.core_modules.common.dump import DumpConfig
from ttk.switches import SWITCHES

# -- _xpu_inputs 数据选择 ----------------------------------------------------


def test_xpu_inputs_prefers_original_shape_arrays():
    """XPU 优先喂 logical ori-shape 数组，而非 NPU run-format 数组（如 NC1HWC0）。"""
    from ttk.core_modules.kernel.worker import _xpu_inputs

    ctx = SimpleNamespace(input_arrays=("NPU_FMT",), original_input_arrays=("ORI",))
    assert _xpu_inputs(ctx) == ("ORI",)


def test_xpu_inputs_falls_back_to_input_arrays():
    """original_input_arrays 为 None 时 fallback 到 input_arrays（manual input / golden off）。"""
    from ttk.core_modules.kernel.worker import _xpu_inputs

    ctx = SimpleNamespace(input_arrays=("NPU_FMT",), original_input_arrays=None)
    assert _xpu_inputs(ctx) == ("NPU_FMT",)


# -- _xpu_mode 位运算 --------------------------------------------------------


def test_xpu_mode_bitwise_or():
    """_xpu_mode = xpu_perf(PERF) | need_data(DATA)；全 False 返回 0。"""
    from ttk.core_modules.kernel import worker as prof
    from ttk.remote import DATA, PERF

    sw_off = SWITCHES()
    sw_off.xpu_perf = False
    assert prof._xpu_mode(sw_off, need_data=False) == 0
    assert prof._xpu_mode(sw_off, need_data=True) == DATA

    sw_on = SWITCHES()
    sw_on.xpu_perf = True
    assert prof._xpu_mode(sw_on, need_data=False) == PERF
    assert prof._xpu_mode(sw_on, need_data=True) == (DATA | PERF)

    sw_dump = SWITCHES()
    sw_dump.dump_config = DumpConfig()
    sw_dump.dump_config.dump_xpu = True
    assert prof._xpu_mode(sw_dump, need_data=False) == DATA

    sw_both = SWITCHES()
    sw_both.xpu_perf = True
    sw_both.dump_config = DumpConfig()
    sw_both.dump_config.dump_xpu = True
    assert prof._xpu_mode(sw_both, need_data=False) == (DATA | PERF)
