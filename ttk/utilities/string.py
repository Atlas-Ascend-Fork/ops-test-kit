#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
String Related Utilities
"""

__all__ = [
    "process_kernel_string",
    "camel_to_snake",
    "longest_match",
    "render_table",
    "render_frameless_table",
]

# Standard Packages
import hashlib
import logging
import math
from typing import List, Tuple


def process_kernel_string(value: str) -> str:
    """Sanitize a kernel name: keep [A-Za-z0-9_], '-' -> 'neg', other chars -> '_';
    names over 120 chars truncate to 80 + a deterministic 8-hex digest suffix."""
    if not isinstance(value, str):
        raise TypeError("Testcase name must be a string.")
    if len(value) == 0:
        raise RuntimeError("Testcase name must not be empty.")
    valid_char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not all(c in valid_char for c in value):
        logging.warning(f"{value} is not a valid kernel name, all invalid character will be converted to _")
    result = "".join(c if c in valid_char else ("neg" if c == "-" else "_") for c in value)
    if len(result) > 120:
        # deterministic across processes (hash() is PYTHONHASHSEED-randomized)
        digest = hashlib.md5(result.encode()).hexdigest()[:8]  # noqa: S324 非安全用途：确定性短摘要
        result = result[:80] + digest
    return result


def camel_to_snake(camel_name: str):
    """Operator Registered Camel name convert to snake name."""
    snake_name = ""
    sub_head = False
    for idx, char in enumerate(camel_name):
        if char.islower():
            sub_head = False
        if char.isdigit():
            sub_head = True
        if char.isupper() and idx != 0:
            if not sub_head:
                snake_name += "_"
                sub_head = True
            elif idx + 1 < len(camel_name) and camel_name[idx + 1].islower():
                snake_name += "_"
        snake_name += char

    return snake_name.lower()


def longest_match(match_str: str, match_list: List[str]) -> str:
    """Return the longest registered prefix of match_str, or an empty string."""
    return max((candidate for candidate in match_list if match_str.startswith(candidate)), key=len, default="")


def render_table(data: List[Tuple[str, ...]]) -> str:
    """Render rows as an ASCII-bordered table; single-column rows span sub-lines."""
    result = ""
    cells_width = [0]
    sub_row_size = []
    minimum_table_width = 0
    # Determine Table Width
    for row in data:
        sub_row_size.append(0)
        if len(row) == 1:
            sub_rows = row[0].split("\n")
            for sub_row in sub_rows:
                minimum_table_width = max(minimum_table_width, len(sub_row) + 2)
            sub_row_size[-1] = len(sub_rows)
        else:
            for idx, column in enumerate(row):
                if len(cells_width) <= idx:
                    cells_width.append(0)
                sub_rows = str(column).split("\n")
                for sub_row in sub_rows:
                    cells_width[idx] = max(cells_width[idx], len(sub_row) + 2)
                sub_row_size[-1] = max(sub_row_size[-1], len(sub_rows))

    # Check for Minimum Table Width Requirements
    if sum(cells_width) < minimum_table_width:
        for idx, cell_width in enumerate(cells_width):
            if cell_width < minimum_table_width // len(cells_width):
                cells_width[idx] = math.ceil(minimum_table_width / len(cells_width))
    char_length = sum(cells_width) + len(cells_width) * 2 - 1
    for idx, row in enumerate(data):
        length = len(row)
        result += "+" + char_length * "-" + "+"
        result += "\n"
        sub_result = []
        split_cells = [str(cell).split("\n") for cell in row]
        for sub_row_idx in range(sub_row_size[idx]):
            if length == 1:
                sub_result.append(
                    "| " + split_cells[0][sub_row_idx].ljust(sum(cells_width) + 2 * len(cells_width) - 2) + "|"
                )
            else:
                sub_row = [cell[sub_row_idx] if sub_row_idx < len(cell) else "" for cell in split_cells]
                sub_result.append("|" + "|".join(" " + sub_row[i].ljust(cells_width[i]) for i in range(length)) + "|")
        result += "\n".join(sub_result)
        result += "\n"
    result += "+" + char_length * "-" + "+"
    return result


def render_frameless_table(data: list) -> str:
    """
    Render frameless table like this:
    ----------------- ------------- ------------- ---------- -------------
                 Name           avg           max        min    # of Calls
    ----------------- ------------- ------------- ---------- -------------
                item1           1.1           2.2        3.3             3
                item2           2.2           3.3        4.4             3
    ----------------------------------------------------------------------
    """
    if not data:
        return ""
    min_cell_width = 10
    cols = len(data[0])
    cols_width = [min_cell_width] * cols
    for row in data:
        for idx, col in enumerate(row):
            cols_width[idx] = max(cols_width[idx], len(str(col)))
    result = "\n" + " ".join([w * "-" for w in cols_width])
    result += "\n" + " ".join([str(col).rjust(cols_width[idx]) for idx, col in enumerate(data[0])])
    result += "\n" + " ".join([w * "-" for w in cols_width])
    for row in data[1:]:
        result += "\n" + " ".join([str(col).rjust(cols_width[idx]) for idx, col in enumerate(row)])
    result += "\n" + "-".join([w * "-" for w in cols_width])
    return result
