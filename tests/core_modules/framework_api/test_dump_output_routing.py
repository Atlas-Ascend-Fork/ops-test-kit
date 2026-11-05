# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""主执行产物 dump 恒走 output_{segment}_ 调试命名（不分后端）；xpu_* 槽位
只由远端三方采集（DataDumper.dump_xpu）产生，供跨机器 manual-data 回放。

跨机比对场景：XPU 机器经 local/remote xpu-server 采集三方输出落 xpu 槽位，
整体转移后 NPU 环境经 manual-data 回放读取（xpu 槽位即三方真值源，provider
口径同三方比对：torch/tf）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy

from ttk.core_modules.common.dump.config import DumpConfig
from ttk.core_modules.common.dump.dumper import DataDumper
from ttk.core_modules.framework_api.compare import dump_outputs
from ttk.switches import SWITCHES


def _sw(tmp_path, **dump_kwargs) -> SWITCHES:
    sw = SWITCHES()
    sw.root_path = str(tmp_path)
    sw.dump_config = DumpConfig(dump_output=True, **dump_kwargs)
    return sw


def test_output_keeps_output_naming(tmp_path):
    """主执行输出（含 XPU 后端）恒落 output_{segment}_ 命名，不占三方槽位。"""
    case = SimpleNamespace(testcase_name="caseA")

    dump_outputs(case, [numpy.array([1.0, 2.0])], _sw(tmp_path), segment="eager")

    case_dir = tmp_path / "dump" / "caseA"
    assert list(case_dir.glob("output_eager_0_float64*.bin"))
    assert not list(case_dir.glob("xpu_*.bin"))


def test_dump_disabled_writes_nothing(tmp_path):
    sw = _sw(tmp_path)
    sw.dump_config = DumpConfig()
    case = SimpleNamespace(testcase_name="caseA")

    dump_outputs(case, [numpy.array([1.0, 2.0])], sw, segment="eager")

    assert not (tmp_path / "dump").exists()


def test_no_prof_xpu_slots_go_to_manual_store(tmp_path):
    """--no-prof 下三方槽位进 manual-data 数据集（与 input/golden 同目录），
    调试 dump 目录不落任何内容。"""
    from ttk.core_modules.common.dump.manual_data import ManualDataCase, ManualDataStore
    from ttk.core_modules.common.xpu import dump_xpu_slot_files

    sw = _sw(tmp_path, dump_xpu=True)
    sw.no_prof = True
    case = SimpleNamespace(testcase_name="caseA")

    store = ManualDataStore([str(tmp_path)])
    store.case_dir("caseA").mkdir(parents=True)
    dump_case = ManualDataCase([], [], None, "", (), store=store, testcase_name="caseA")
    dump_xpu_slot_files(case, sw, [numpy.array([1.0, 2.0]), numpy.array([3.0])], "torch", dump_case=dump_case)

    case_dir = tmp_path / "caseA"
    assert list(case_dir.glob("xpu_torch_*.bin"))
    assert not (tmp_path / "dump").exists()  # 调试 dump 不落


def test_xpu_slot_roundtrip_manual_replay(tmp_path):
    """三方采集落盘（dump_xpu）→ NPU 侧 manual-data 回放装载，值逐元素还原。"""
    from ttk.core_modules.common.dump.manual_data import (
        _XPU_FILE_NAME,
        ManualDataCase,
        ManualDataStore,
        _ManualDataFile,
    )
    from ttk.core_modules.common.dump.manual_data import (
        decode_shape as _decode_shape,
    )
    from ttk.core_modules.common.xpu import load_third_parties

    sw = _sw(tmp_path, dump_xpu=True)
    case = SimpleNamespace(testcase_name="caseA")
    original = numpy.array([1.5, 2.5, 3.5])

    DataDumper(sw, "caseA").dump_xpu({"torch": [original]})

    case_dir = tmp_path / "dump" / "caseA"
    (slot,) = case_dir.glob("xpu_*.bin")
    match = _XPU_FILE_NAME.fullmatch(slot.name)
    assert match is not None, "落盘文件名必须符合 xpu 槽位回放协议"
    provider, index_text, dtype_name, shape_token, file_format = match.groups()
    entry = _ManualDataFile(
        role="xpu",
        index=int(index_text),
        dtype=dtype_name,
        file_format=file_format,
        path=slot,
        saved_shape=_decode_shape(shape_token),
        provider=provider,
    )
    manual_case = ManualDataCase([], [], case_dir, file_format, (), (entry,), store=ManualDataStore([str(tmp_path)]))

    arrays = load_third_parties(case, manual_case, device_outputs=[original])
    assert arrays is not None
    assert arrays[0] is not None
    assert numpy.array_equal(arrays[0], original)
