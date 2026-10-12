#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/testcase/reader：输入格式适配层——path → Table。

分层：reader 管"path → 表"，Factory 管"(header, rows) → 用例"，
parse_cases 管编排（files → iter_tables → 逐表 Factory → 聚合去重筛选）。
新增格式 = 新增 reader 文件 + iter_tables 分发一行。
"""

__all__ = ["Table", "iter_tables", "read_table", "read_csv_rows", "resolved_sheet"]

import os
from typing import Iterator, NamedTuple, Optional

from .csv_reader import read_csv_rows
from .xlsx_reader import read_xlsx_rows, resolved_sheet
from .zip_reader import iter_csv_members


class Table(NamedTuple):
    """表来源（输入文件名或 zip 成员名）+ header + rows。"""

    name: str
    header: list
    rows: list


def iter_tables(path: str, sheet: Optional[str] = None) -> Iterator[Table]:
    """输入路径 → 表格迭代器（用例解析的唯一入口）。
    后缀分发：.xlsx/.xlsm → xlsx_reader 单表；.zip → zip_reader 逐 .csv 成员
    （N 张表）；其余（.csv/未知后缀）→ csv_reader 单表。"""
    if path.lower().endswith((".xlsx", ".xlsm")):
        header, rows = read_xlsx_rows(path, sheet)
        yield Table(name=path, header=header, rows=rows)
    elif path.lower().endswith(".zip"):
        for member_name, fileobj in iter_csv_members(path):
            header, rows = read_csv_rows(fileobj)
            yield Table(name=member_name, header=header, rows=rows)
    else:
        # utf-8-sig 兼容带 BOM 的 CSV（Excel 导出），无 BOM 时行为不变
        with open(path, encoding="utf-8-sig", newline="") as f:
            header, rows = read_csv_rows(f)
        yield Table(name=path, header=header, rows=rows)


def read_table(path: str, sheet: Optional[str] = None) -> tuple:
    """单表便捷入口 = next(iter_tables(path, sheet))。"""
    table = next(iter_tables(path, sheet))
    return table.header, table.rows
