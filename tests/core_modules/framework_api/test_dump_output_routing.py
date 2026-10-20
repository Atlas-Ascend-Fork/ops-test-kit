# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""XPU 后端执行时 output dump 改走三方槽位命名（eager：xpu_{framework}_{idx}_{dtype}）。

跨机比对场景：XPU 侧生成 input/golden/xpu 槽位后整体转移，NPU 环境经
manual-data 回放读取（xpu 槽位即三方真值源，provider 口径同三方比对：torch/tf）；
非 XPU 后端保持 output_{segment}_ 调试命名不变。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy

from ttk.core_modules.common.backend.xpu_torch_backend import XpuTorchBackend
from ttk.core_modules.common.dump.config import DumpConfig
from ttk.core_modules.framework_api.compare import dump_outputs
from ttk.switches import SWITCHES


def _sw(tmp_path) -> SWITCHES:
    sw = SWITCHES()
    sw.root_path = str(tmp_path)
    sw.dump_config = DumpConfig(dump_output=True)
    return sw


def test_xpu_backend_dumps_xpu_slot(tmp_path):
    backend = XpuTorchBackend()
    case = SimpleNamespace(testcase_name="caseA")

    dump_outputs(case, [numpy.array([1.0, 2.0])], _sw(tmp_path), segment="eager", backend=backend)

    case_dir = tmp_path / "dump" / "caseA"
    assert list(case_dir.glob("xpu_torch_0_float64*.bin")), "XPU 后端 eager 输出应落 xpu_{framework}_ 槽位（不带模式）"
    assert not list(case_dir.glob("output_*.bin"))


def test_xpu_backend_graph_segment_in_provider(tmp_path):
    backend = XpuTorchBackend()
    backend.torch_lib = "cuda"
    case = SimpleNamespace(testcase_name="caseA")

    dump_outputs(case, [numpy.array([1.0, 2.0])], _sw(tmp_path), segment="graph_cst", backend=backend)

    case_dir = tmp_path / "dump" / "caseA"
    assert list(case_dir.glob("xpu_torch_graph_cst_0_float64*.bin"))
    assert not list(case_dir.glob("xpu_cuda_*.bin"))  # provider 取 framework 而非 torch_lib


def test_non_xpu_backend_keeps_output_naming(tmp_path):
    from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend

    case = SimpleNamespace(testcase_name="caseA")

    dump_outputs(case, [numpy.array([1.0, 2.0])], _sw(tmp_path), segment="eager", backend=CpuTorchBackend())

    case_dir = tmp_path / "dump" / "caseA"
    assert list(case_dir.glob("output_eager_0_float64*.bin"))
    assert not list(case_dir.glob("xpu_*.bin"))


def test_third_party_provider_hook():
    """三方命名是后端能力钩子：XPU 后端返回框架名，NPU/CPU 后端返回 None。"""
    from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend
    from ttk.core_modules.common.backend.npu_torch_backend import NpuTorchBackend

    assert XpuTorchBackend().third_party_provider("torch") == "torch"
    assert XpuTorchBackend().third_party_provider("") == "torch"
    assert CpuTorchBackend().third_party_provider("torch") is None
    assert NpuTorchBackend().third_party_provider("torch") is None


def test_dump_disabled_writes_nothing(tmp_path):
    sw = _sw(tmp_path)
    sw.dump_config = DumpConfig()
    backend = XpuTorchBackend()
    case = SimpleNamespace(testcase_name="caseA")

    dump_outputs(case, [numpy.array([1.0, 2.0])], sw, segment="eager", backend=backend)

    assert not (tmp_path / "dump").exists()


def test_xpu_slot_roundtrip_manual_replay(tmp_path):
    """XPU 侧落盘 → NPU 侧 manual-data 回放装载，值逐元素还原。"""
    from ttk.core_modules.common.dump.manual_data import (
        _XPU_FILE_NAME,
        ManualDataCase,
        _decode_shape,
        _ManualDataFile,
    )
    from ttk.core_modules.common.xpu import load_manual_third_parties

    backend = XpuTorchBackend()
    case = SimpleNamespace(testcase_name="caseA")
    original = numpy.array([1.5, 2.5, 3.5])

    dump_outputs(case, [original], _sw(tmp_path), segment="eager", backend=backend)

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
    manual_case = ManualDataCase([], [], case_dir, file_format, (), (entry,))

    arrays = load_manual_third_parties(case, manual_case, device_outputs=[original])
    assert arrays is not None
    assert arrays[0] is not None
    assert numpy.array_equal(arrays[0], original)
