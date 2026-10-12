#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""csv 文本表格解析。"""

__all__ = ["read_csv_rows"]

import csv


def read_csv_rows(fileobj) -> tuple:
    """文本流 → (header, rows)。单元格 strip、空行丢弃、首个非空行为表头。
    吃 fileobj 而非 path：zip 成员流与普通文件共用同一解析。"""
    rows = []
    for raw in csv.reader(fileobj):
        row = [column.strip() for column in raw]
        if row:
            rows.append(row)
    if not rows:
        raise ValueError("Empty table: no rows found")
    return rows[0], rows[1:]
