#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""zip 容器展开。"""

__all__ = ["iter_csv_members"]

import io
import logging
import zipfile

from .csv_reader import decode_csv_bytes


def iter_csv_members(path: str):
    """zip 路径 → 迭代器 [(member_name, text_fileobj)]。
    testzip 校验损坏（corrupted → RuntimeError）；非 .csv 成员 warning 跳过；
    成员字节经 decode_csv_bytes 解码（utf-8-sig / GB18030 回退）。"""
    with zipfile.ZipFile(path) as zipped_file:
        corrupted = zipped_file.testzip()
        if corrupted:
            raise RuntimeError(f"Zipfile corrupted on file: {corrupted}")
        for file in zipped_file.infolist():
            if not file.filename.endswith(".csv"):
                logging.warning(f"Skipped zipped non-testcase file {file.filename}")
                continue
            logging.info(f"Reading zipped testcase file {file.filename} ...")
            with zipped_file.open(file) as real_file:
                data = real_file.read()
            yield file.filename, io.StringIO(decode_csv_bytes(data))
