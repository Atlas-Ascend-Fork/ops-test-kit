#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""DumpConfig：worker 落盘配置（bool 位 + from_cli 解析）。

full = in/out/golden/xpu 四类全开；无远端 XPU 配置时 xpu 位在 CLI 层自动
关闭——有数据才落盘，缺配置不报错。
"""

__all__ = ["DumpConfig"]

# Standard Packages
from dataclasses import dataclass
from typing import Optional


@dataclass
class DumpConfig:
    """worker 落盘配置。构造一次全只读（--dump 解析见 from_cli）。"""

    dump_input: bool = False
    dump_output: bool = False
    dump_golden: bool = False
    dump_xpu: bool = False  # XPU 三方输出落盘（full 含；无远端配置时 CLI 层自动关闭）
    file_format: str = "bin"  # bin / npy / pt / print（formatter 注册键）
    dump_on_fail: bool = False

    @classmethod
    def from_cli(cls, value: Optional[str]) -> "DumpConfig":
        """--dump 参数解析（"in,out,golden,xpu,full" 逗号组合；full = 四类
        全开含 xpu；未知 token 忽略）。"""
        cfg = cls()
        if value is None:
            return cfg
        if value.strip() == "full":
            cfg.dump_input = cfg.dump_output = cfg.dump_golden = cfg.dump_xpu = True
            return cfg
        for token in (t.strip() for t in value.lower().split(",")):
            if not token:
                continue
            if token == "in":  # noqa: S105 判据 token，非凭据
                cfg.dump_input = True
            elif token == "out":  # noqa: S105 判据 token，非凭据
                cfg.dump_output = True
            elif token == "golden":  # noqa: S105 判据 token，非凭据
                cfg.dump_golden = True
            elif token == "xpu":  # noqa: S105 判据 token，非凭据
                cfg.dump_xpu = True
            elif token == "full":  # noqa: S105 判据 token，非凭据
                cfg.dump_input = cfg.dump_output = cfg.dump_golden = cfg.dump_xpu = True
            else:
                raise ValueError(f"--dump: unknown token {token!r} (choose from full, in, out, golden, xpu)")
        return cfg
