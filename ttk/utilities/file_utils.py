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
File utils
"""

__all__ = [
    "file_md5",
    "read_file",
    "delete_files",
    "read_csv_cols",
    "extract_zip",
]


import csv
import hashlib
import os
import shutil
import tempfile
import zipfile


def file_md5(file_path: str):
    block_size = 2**20
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()  # noqa: S324 非安全用途：确定性校验的文件指纹（非密码学）
        for chunk in iter(lambda: f.read(block_size), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def read_file(file_path: str, size_limit: int = 1024 * 1024 * 1024) -> bytes:
    """
    :param file_path: Path to the file
    :param size_limit: Raise an Exception if the file is too large
    :return: binary object
    """
    file_size = os.stat(file_path).st_size
    if file_size > size_limit:
        raise OSError(f"File is too large! Size of {file_path} exceeds the limit: {size_limit}")
    with open(file_path, "rb") as file:
        file_content = file.read()
    return file_content


def delete_files(folder_path: str, include_fn=None, exclude_fn=None):
    # only delete files, but keep folders.
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return
    for root, _, files in os.walk(folder_path, topdown=False):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                if exclude_fn and exclude_fn(file_path):
                    continue
                if include_fn:
                    if include_fn(file_path):
                        os.unlink(file_path)
                else:
                    os.unlink(file_path)
            except Exception:  # noqa: S110 清理尽力而为：只读文件/权限问题不阻断主流程
                pass


def read_csv_cols(filename, extract_cols, col_name_mapping: dict = None, cmp=None) -> list:
    """
    Extract columns of each row.
    multi-columns fills to a dict.
    multi-rows performs a list.
    """
    results = []
    if col_name_mapping is None:
        col_name_mapping = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if cmp is None or cmp(row):
                data_dict = {}
                for c in extract_cols:
                    map_name = col_name_mapping.get(c, c)
                    try:
                        data_dict.update({map_name: float(row[c])})
                    except (ValueError, TypeError):
                        data_dict.update({map_name: row[c]})
                results.append(data_dict)
    return results


def _move_contents(source_dir, target_dir):
    """
    move files under @source_dir to @target_dir
    """
    os.makedirs(target_dir, mode=0o700, exist_ok=True)

    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        if os.path.isdir(source_path) and os.path.exists(target_path):
            _move_contents(source_path, target_path)
        else:
            shutil.move(source_path, target_path)


def extract_zip(zip_file, extract_to, remove_top_level=False):
    """
    Extract a zip file; @extract_to is always treated as a directory.
    remove_top_level: strip the single top-level folder of the archive.
    """
    if not remove_top_level:
        with zipfile.ZipFile(zip_file, mode="r") as zf:
            zf.extractall(extract_to)
        return
    # 唯一临时目录（mkdtemp 原子创建）：固定名 "{extract_to}_tmp" 会无条件 rmtree
    # 相邻同名目录（已有内容丢失/并发解压路径碰撞），改用随机后缀互不干扰
    parent = os.path.dirname(os.path.abspath(extract_to))
    os.makedirs(parent, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"{os.path.basename(extract_to) or 'extract'}_tmp_", dir=parent)
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(tmp_dir)
        tmp_contents = os.listdir(tmp_dir)
        if len(tmp_contents) == 1:
            top_level_path = os.path.join(tmp_dir, tmp_contents[0])
            if os.path.isdir(top_level_path):
                _move_contents(top_level_path, extract_to)
            else:
                os.makedirs(extract_to, mode=0o700, exist_ok=True)
                shutil.move(top_level_path, os.path.join(extract_to, tmp_contents[0]))
        else:
            _move_contents(tmp_dir, extract_to)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
