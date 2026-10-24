#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/dump：worker 阶段产物的落盘器。

- config.py：DumpConfig（bool 位 + from_cli）
- formatter.py：DumpFormatter 注册制（bin/npy/pt/print 四实现）
- dumper.py：DataDumper 统一调度
- manual_data.py：手动数据模式（store 写/读 + CLI 目录集）
"""

__all__ = [
    # 公开面
    "DataDumper",
    "DumpConfig",
    "register_format",
    "get_formatter",
    "ManualDataStore",
    "ManualDataCase",
    "resolve_manual_case",
    "load_csv_manual_case",
    "snapshot_manual_values",
    "case_directory_name",
    "ManualDataError",
    # store 族编排入口（worker 编排层消费）：dump 方向落盘 + load 方向装载
    "manual_data_dump_store",
    "dump_manual_dataset",
    "load_manual_data_case",
    "load_replay_goldens",
    "manual_data_store",
    "resolve_manual_data_directories",
]

from .config import DumpConfig
from .dumper import DataDumper
from .formatter import DumpFormatter, get_formatter, register_format
from .manual_data import (
    ManualDataCase,
    ManualDataError,
    ManualDataStore,
    case_directory_name,
    dump_manual_dataset,
    load_manual_data_case,
    load_replay_goldens,
    manual_data_dump_store,
    manual_data_store,
    resolve_manual_data_directories,
    snapshot_manual_values,
)

# 来源裁决入口：CLI 目录集命中 → 加载已存用例；未指定/全空 → None（正常生成）。
# CSV manual_input_binaries/manual_golden_binaries 列的装载在通路 testcase 侧
# （实际优先级：CLI 目录集 > CSV 列 > 正常生成），统一入口为
# load_csv_manual_case（尚未实现）。


def resolve_manual_case(testcase, case_type: str, switches, notify_before_load=None, require_goldens: bool = True):
    """来源裁决：CLI 目录集命中 → 加载；未指定/全空 → None（正常生成）。worker 统一入口。"""
    return load_manual_data_case(
        testcase, case_type, switches, notify_before_load=notify_before_load, require_goldens=require_goldens
    )


def load_csv_manual_case(testcase, case_type: str):
    """CSV manual_input_binaries / manual_golden_binaries 列 → ManualDataCase。

    当前该列装载在通路 testcase 侧，本入口尚未实现。"""
    raise NotImplementedError("CSV manual-data loading is not unified here yet")
