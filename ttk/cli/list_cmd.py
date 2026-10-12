#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from .bridge import _split_csv


def register_list_command(subparsers):
    parser = subparsers.add_parser("list", help="List test cases from a test case file (csv/xlsx)")
    parser.add_argument("-i", "--input", required=True, help="Test case file (csv/xlsx)")
    parser.add_argument("--sheet", default=None, help="Excel worksheet name (default: first worksheet)")
    parser.add_argument("--op", "--operator", dest="operator", help="Filter by operator name")
    parser.set_defaults(handler=_handle_list)
    return parser


def _handle_list(args):
    from ttk.core_modules.common.testcase import read_table

    header, rows = read_table(args.input, args.sheet)

    try:
        name_idx = header.index("testcase_name")
    except ValueError as err:
        raise ValueError(f"testcase_name column not found in {args.input}") from err

    for row_idx, row in enumerate(rows, start=1):
        if name_idx >= len(row) or not row[name_idx]:
            raise ValueError(f"empty testcase_name at data row {row_idx} in {args.input}")

    names = [row[name_idx] for row in rows]

    # is_enabled 过滤无条件先行（旧行为: testcase_manager.py:469 在任何选择器之前生效,
    # --op 匹配的禁用行同样不出现）
    if "is_enabled" in header:
        enabled_idx = header.index("is_enabled")
        kept = [
            (name, row)
            for name, row in zip(names, rows)
            if enabled_idx >= len(row) or row[enabled_idx].lower() not in ("false", "0")
        ]
        names, rows = [n for n, _ in kept], [r for _, r in kept]

    if args.operator:
        op_filter = _split_csv(args.operator)
        if "op_name" in header:
            op_idx = header.index("op_name")
        elif "api_name" in header:
            op_idx = header.index("api_name")
        else:
            op_idx = None
        if op_idx is not None:
            names = [name for name, row in zip(names, rows) if op_idx < len(row) and row[op_idx] in op_filter]
        else:
            names = []

    for name in names:
        print(name)
    print(f"\nTotal: {len(names)} case(s)")
