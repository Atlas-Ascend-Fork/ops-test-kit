# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""server config 测试：detect_hardware 硬件检测。"""

import os

# -- detect_hardware --------------------------------------------------------


def test_detect_gpu_and_mlu(monkeypatch):
    """detect_hardware: 识别 gpu（nvidia 前缀）/ mlu（cambricon 前缀），排除控制设备。"""
    from ttk.remote.server.config import detect_hardware

    cfg = {"gpu": {"dev_prefix": "nvidia"}, "mlu": {"dev_prefix": "cambricon"}}

    # gpu（排除 nvidiactl 控制设备）
    monkeypatch.setattr(os, "listdir", lambda p: ["nvidia0", "nvidia5", "nvidiactl", "tty"])
    assert detect_hardware(cfg) == ("gpu", [0, 5])

    # mlu
    monkeypatch.setattr(os, "listdir", lambda p: ["cambricon0", "tty"])
    assert detect_hardware(cfg) == ("mlu", [0])


def test_detect_segment_order_and_cpu_fallback(monkeypatch):
    """detect_hardware: 段序保留（gpu 先于 mlu）；空/不可读 /dev → cpu fallback。"""
    from ttk.remote.server.config import detect_hardware

    cfg = {"gpu": {"dev_prefix": "nvidia"}, "mlu": {"dev_prefix": "cambricon"}}

    # 段序：gpu + mlu 都命中，gpu 先
    monkeypatch.setattr(os, "listdir", lambda p: ["nvidia0", "cambricon0"])
    assert detect_hardware(cfg)[0] == "gpu"

    # 空 → cpu
    monkeypatch.setattr(os, "listdir", lambda p: [])
    assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}}) == ("cpu", ["cpu"])

    # 不可读 → cpu
    def boom(p):
        raise OSError("denied")

    monkeypatch.setattr(os, "listdir", boom)
    assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}}) == ("cpu", ["cpu"])
