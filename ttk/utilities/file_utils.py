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
    "load_array",
    "read_file",
    "delete_files",
    "read_csv_cols",
    "extract_zip",
]


import csv
import hashlib
import logging
import os
import shutil
import zipfile
from typing import Union

import numpy

from .dtypes import is_4bit_dtype, torch_to_numpy_tensor, unpack_4bits
from .shape import shape_numel


def file_md5(file_path: str):
    block_size = 2**20
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()  # noqa: S324 非安全用途：确定性校验的文件指纹（非密码学）
        for chunk in iter(lambda: f.read(block_size), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def _resolve_load_dtype(dtype):
    """Map special dtypes to a numpy-loadable dtype for raw file reading."""
    if dtype in ("uint1", "int1"):
        return "uint8"
    if dtype == "complex32":
        return "float16"
    if is_4bit_dtype(dtype):
        return "uint8"
    return dtype


def _postprocess_4bit(array, ori_dtype, shape):
    """Unpack or view 4-bit data after loading."""
    if array.dtype.name in ("int8", "uint8"):
        size = shape_numel(shape)
        if array.size == size:
            array = array.view(ori_dtype)
        else:
            array = unpack_4bits(array, ori_dtype)
            array = array[:size]
    else:
        array = array.view(ori_dtype)
    return array


def _postprocess_special(array, ori_dtype):
    """View-cast special dtypes loaded from .npy (stored as uint8/int8)."""
    for sd in ("bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e8m0", "hifloat8"):
        if sd in str(ori_dtype) and array.dtype.name != sd:
            array = array.view(ori_dtype)
            break
    return array


def _reshape_storage(array, ori_dtype, shape):
    """Reshape to the dtype's physical storage shape: int1 packs 8 elements per
    byte on the last dim; complex32 stores fp16 real/imag as a trailing [2]."""
    if ori_dtype == "int1":
        shape = list(shape[:-1]) + [shape[-1] // 8]
    elif ori_dtype == "complex32":
        shape = list(shape) + [2]
    try:
        return array.reshape(shape)
    except ValueError:
        # in manual output scenario:
        # output shape may be wrong (needs infer-shaping by golden)
        return array


def load_array(file_name: str, dtype: str, shape: Union[list, tuple]):
    ori_dtype = dtype
    dtype = _resolve_load_dtype(dtype)

    if file_name.endswith(".pt"):
        import inspect

        import torch

        # weights_only 优先（手动数据文件可能来自用户，防 pickle 载荷执行）；
        # 旧版 Torch 无该参数时回退 pickle 并告警（仅可加载可信文件）。
        try:
            supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
        except (TypeError, ValueError):
            supports_weights_only = True
        if supports_weights_only:
            tensor = torch.load(file_name, map_location="cpu", weights_only=True)
        else:
            logging.warning(
                "torch.load on this Torch version does not support weights_only=True; "
                "falling back to pickle-based loading for %s. Only load trusted manual-data files.",
                file_name,
            )
            tensor = torch.load(file_name, map_location="cpu")
        array = torch_to_numpy_tensor(tensor)
    elif file_name.endswith(".npy"):
        array = numpy.load(file_name, allow_pickle=False)
    else:
        array = numpy.fromfile(file_name, dtype=dtype)

    if is_4bit_dtype(ori_dtype):
        array = _postprocess_4bit(array, ori_dtype, shape)
    array = _postprocess_special(array, ori_dtype)
    return _reshape_storage(array, ori_dtype, shape)


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
    tmp_dir = f"{extract_to}_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
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
