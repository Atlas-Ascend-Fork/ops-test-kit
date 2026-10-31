#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""manual XPU 三方输出的本地装载 / 落盘（--dump xpu 回放契约）。

CSV ``manual_xpu_binaries`` 显式路径与 manual-data case 目录槽位双来源；
落盘经 DataDumper.dump_xpu 单份实现，与 input/golden 槽同目录共存。
"""

import logging
from typing import Optional

from ....utilities import load_array, resolve_numpy_dtypes

# manual 装载在三方结果中的伪 provider 名（对齐远端产出的 dict 形态，屏显/报告零改动）
MANUAL_XPU_PROVIDER = "manual_binary"


class ManualXpuLoadError(ValueError):
    """manual_xpu_binaries 装载失败（shape 与声明不符等）。"""


def manual_xpu_results() -> dict:
    """manual 装载的 xpu_results 伪条目（status=PASS，api 标记来源）。"""
    return {MANUAL_XPU_PROVIDER: {"status": "PASS", "api": "manual_xpu_binaries"}}


def has_manual_xpu_source(testcase, manual_case=None) -> bool:
    """manual 三方输出源是否已声明（不读文件）。

    判定口径与 load_manual_third_parties 的双来源一致：CSV
    ``flat_manual_xpu_binaries`` 非空，或 manual-data case 目录含 xpu 槽位
    文件。适用于 dispatch 前置探测等只需要"有没有源"的场景；文件内容仍由
    load_manual_third_parties 在消费点装载。
    """
    if getattr(testcase, "flat_manual_xpu_binaries", None):
        return True
    return manual_case is not None and bool(getattr(manual_case, "has_third_parties", False))


def load_manual_third_parties(testcase, manual_case=None, device_outputs=None) -> Optional[list]:
    """manual 三方输出本地装载（--dump xpu 回放契约的对侧），双来源：

    ① CSV ``manual_xpu_binaries`` 显式路径（NPU_DUMP_PATH 归档回放形态）；
    ② manual-data case 目录槽位（``--manual-data-dirs`` 回放形态）——
       ``xpu_{provider}_{idx}_{dtype}[_shape_{dims}].{ext}`` 按 idx 对位
       （文件名一致即可读取，多 provider 同 idx 取字典序首个）。

    两来源均未声明返回 None（调用方走远端采集）；已声明按 flat 输出槽位对齐
    装载：路径数不足的槽位 / 动态 shape 槽（None）补 None（该输出无三方数据，
    cross_check 判据按既有占位语义处理）。shape/dtype 槽取声明槽
    （``flat_output_shapes``/``flat_output_dtypes``，kernel/aclnn）；e2e 输出
    为执行派生（无声明槽），经 ``device_outputs``（设备输出）取 shape 与 dtype，
    口径同 golden 的 device_outputs 装载。声明/device_outputs 缺席时按文件名 shape
    段（bin 自描述）+ 文件名 dtype 装载——manual xpu 回放不依赖当次执行输出。
    """
    paths = getattr(testcase, "flat_manual_xpu_binaries", None)
    saved_shapes = ()
    filename_dtypes = ()
    if not paths and manual_case is not None and getattr(manual_case, "has_third_parties", False):
        paths = manual_case.third_party_paths()
        saved_shapes = getattr(manual_case, "third_party_saved_shapes", lambda: ())()
        filename_dtypes = getattr(manual_case, "third_party_dtypes", lambda: ())()
    if not paths:
        return None
    shapes = getattr(testcase, "flat_output_shapes", None)
    if shapes is None:
        shapes = getattr(testcase, "flat_output_view_shapes", None)
    if not shapes and device_outputs:
        # e2e：输出无声明槽，shape/dtype 取设备输出（值不参与，仅槽位元数据）
        shapes = tuple(tuple(r.shape) if r is not None and hasattr(r, "shape") else None for r in device_outputs)
    shapes = shapes or ()
    dtypes = resolve_numpy_dtypes(getattr(testcase, "flat_output_dtypes", ()) or ())
    if not dtypes and device_outputs:
        dtypes = tuple(
            getattr(r, "dtype", None) if r is not None and hasattr(r, "dtype") else None for r in device_outputs
        )
    name = getattr(testcase, "testcase_name", None) or getattr(testcase, "op_name", None)
    arrays = []
    for index in range(max(len(paths), len(shapes))):
        path = paths[index] if index < len(paths) else None
        declared_shape = shapes[index] if index < len(shapes) else None
        saved_shape = saved_shapes[index] if index < len(saved_shapes) else None
        if path is None or (declared_shape is None and saved_shape is None):
            arrays.append(None)
            continue
        logging.info("Reading manual XPU file: %s ...", path)
        dtype = dtypes[index] if index < len(dtypes) else None
        filename_dtype = filename_dtypes[index] if index < len(filename_dtypes) else None
        arrays.append(_load_manual_xpu_slot(index, (path, declared_shape, saved_shape), dtype, filename_dtype, name))
    return arrays


def _load_manual_xpu_slot(index, slot, dtype, filename_dtype, name):
    """单槽 xpu 装载：声明/文件名 shape 决议 → load_array → 物理形态核对。

    slot = (path, declared_shape, saved_shape)。load_array 返回 dtype 物理
    存储形态（complex32 尾维 [2] fp16 实/虚、int1 位打包、4bit 解包）——校验
    须按物理形态比，口径同 manual-data 的 physical_array_spec；错文件（元素数
    不符）经 _reshape_storage 宽容回退后由本检查拦截。文件名 shape 段自描述
    路径（无声明槽）按段本身（物理存储 shape）比对。
    """
    from ..dump.manual_data import physical_array_spec

    path, declared_shape, saved_shape = slot
    resolved_shape = declared_shape if declared_shape is not None else (saved_shape or (1,))
    if dtype is None and declared_shape is None:
        dtype = filename_dtype
    array = load_array(path, dtype, resolved_shape)
    if declared_shape is not None:
        _, storage_shape = physical_array_spec(dtype, declared_shape)
        expected_storage = storage_shape
    else:
        expected_storage = resolved_shape
    if tuple(array.shape) != expected_storage:
        raise ManualXpuLoadError(
            f"[{name}] manual XPU output[{index}] shape {tuple(array.shape)} does not match "
            f"declared {tuple(declared_shape if declared_shape is not None else resolved_shape)} "
            f"(storage {expected_storage})"
        )
    return array


def dump_xpu_slot_files(testcase, switches, third_parties, provider) -> None:
    """--dump xpu 开启时，三方输出按 case 目录槽位协议落盘。

    ``xpu_{provider}_{idx}_{dtype}.{ext}``（DataDumper.dump_xpu 单份实现）——
    ``--manual-data-dirs`` 回放侧按文件名 idx 对位读取（ManualDataCase.third_party_paths），
    与 input/golden 槽同目录共存。远端/manual 两来源统一经此入口。
    命名前缀：dump_file_prefix or testcase_name（kernel/aclnn 约定；e2e/geir
    无该字段自然回退 testcase_name）。
    """
    if not third_parties:
        return
    if not getattr(getattr(switches, "dump_config", None), "dump_xpu", False):
        return
    from ..dump.dumper import DataDumper

    name_prefix = getattr(testcase, "dump_file_prefix", None) or getattr(testcase, "testcase_name", None) or "case"
    DataDumper(switches, name_prefix).dump_xpu({provider: list(third_parties)})
