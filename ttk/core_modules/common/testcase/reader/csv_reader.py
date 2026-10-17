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

__all__ = ["read_csv_rows", "decode_csv_bytes"]

import csv
import logging


def decode_csv_bytes(data: bytes) -> str:
    """用例文本解码：utf-8-sig 优先（兼容 Excel BOM），非 UTF-8 回退 GB18030。

    GBK/GB2312 是 GB18030 子集——中文表头/备注的常见编码经此路径免转换直读，
    warning 留痕提示基线转存 UTF-8。两种编码都失败时抛 UnicodeDecodeError。
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    logging.warning(
        "case file contains Chinese text but is not UTF-8 encoded (GBK/GB2312); "
        "decoded as GB18030. Convert case files to UTF-8 to avoid this."
    )
    return data.decode("gb18030")


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
