#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ttk.core_modules.kernel import worker as profiling
from ttk.core_modules.kernel.testcase import TestcaseKernel


def _make_testcase(output_shapes=((8,),), output_dtypes=("float32",)):
    case = TestcaseKernel()
    case.testcase_name = "manual_xpu"
    case.op_name = "add"
    case.input_shapes = ((8,), (8,))
    case.input_dtypes = ("float32", "float32")
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = case.input_shapes
    case.output_ori_shapes = output_shapes
    case.input_formats = ("ND", "ND")
    case.input_ori_formats = ("ND", "ND")
    case.output_formats = ("ND",) * len(output_shapes)
    case.output_ori_formats = ("ND",) * len(output_shapes)
    case.input_data_ranges = ((None, None), (None, None))
    case.attributes = {}
    case.manual_input_binaries = ("input0.bin", "input1.bin")
    case.manual_golden_binaries = tuple(f"golden{i}.bin" for i in range(len(output_shapes)))
    return case


def _validate(case):
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper") as mock:
        mock.return_value.info_of.return_value = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": f"z{i}"} for i in range(len(case.output_shapes))],
        }
        case.validate()


def test_manual_xpu_header_is_supported():
    assert "manual_xpu_binaries" in TestcaseKernel.complete_headers


def test_manual_xpu_single_file_is_normalized():
    case = _make_testcase()
    case.manual_xpu_binaries = "gpu.bin"
    _validate(case)
    assert case.is_valid
    assert case.manual_xpu_binaries == ("gpu.bin",)
    assert case.flat_manual_xpu_binaries == ("gpu.bin",)


def test_manual_xpu_count_mismatch_is_rejected():
    case = _make_testcase(output_shapes=((8,), (8,)), output_dtypes=("float32", "float32"))
    case.manual_xpu_binaries = ("gpu0.bin",)
    _validate(case)
    assert not case.is_valid
    assert case.fail_reason == "MANUAL_XPU_BINARIES_INVALID"


def test_manual_xpu_requires_manual_input():
    case = _make_testcase()
    case.manual_input_binaries = ()
    case.manual_xpu_binaries = ("gpu.bin",)
    _validate(case)
    assert not case.is_valid
    assert case.fail_reason == "MANUAL_XPU_INPUT_REQUIRED"


def test_manual_xpu_requires_manual_golden():
    case = _make_testcase()
    case.manual_golden_binaries = ()
    case.manual_xpu_binaries = ("gpu.bin",)
    _validate(case)
    assert not case.is_valid
    assert case.fail_reason == "MANUAL_XPU_GOLDEN_REQUIRED"


def test_manual_xpu_tensor_list_is_flattened():
    case = _make_testcase(
        output_shapes=(((8,), (8,)),),
        output_dtypes=(("float32", "float32"),),
    )
    case.manual_golden_binaries = (("golden0.bin", "golden1.bin"),)
    case.manual_xpu_binaries = (("gpu0.bin", "gpu1.bin"),)
    _validate(case)
    assert case.is_valid
    assert case.flat_manual_xpu_binaries == ("gpu0.bin", "gpu1.bin")


@pytest.mark.parametrize(
    ("need_3party", "golden_is_xpu", "via_slots"),
    [(True, False, False), (False, True, True)],
    ids=["csv-cross-check", "case-dir-golden-xpu"],
)
def test_manual_xpu_bypasses_xpu_dispatch(tmp_path, monkeypatch, need_3party, golden_is_xpu, via_slots):
    gpu_path = tmp_path / "gpu.bin"
    expected = np.arange(8, dtype=np.float32)
    expected.tofile(gpu_path)

    case = _make_testcase()
    if via_slots:
        manual_case = SimpleNamespace(has_third_parties=True, third_party_paths=lambda: [str(gpu_path)])
    else:
        case.manual_xpu_binaries = (str(gpu_path),)
        _validate(case)
        manual_case = None
    case.dyn_compile_result = SimpleNamespace(workspaces=(), debug_buf_size=0)
    case.cst_compile_result = SimpleNamespace(workspaces=(), debug_buf_size=0)
    case.bin_compile_result = SimpleNamespace(workspaces=(), debug_buf_size=0)

    process_context = SimpleNamespace(notify_status=lambda *_: None)
    monkeypatch.setattr(profiling, "process_context", lambda: process_context)
    monkeypatch.setattr(
        profiling, "get_global_storage", lambda: SimpleNamespace(golden_mode="XPU" if golden_is_xpu else "Enable")
    )
    monkeypatch.setattr(
        profiling,
        "_do_xpu_profiling",
        lambda *_: (_ for _ in ()).throw(AssertionError("XPU dispatch must not run")),
    )

    # dispatch/collect 两段式：manual 源 → ready 句柄，collect 即取（远端零调用）
    xpu = profiling._dispatch_xpu(case, need_3party=need_3party, manual_case=manual_case)
    assert not xpu.pending  # ready 形态
    third_parties, xpu_results, provider = xpu.collect()

    np.testing.assert_array_equal(third_parties[0], expected)
    assert xpu_results == {
        "manual_binary": {
            "status": "PASS",
            "api": "manual_xpu_binaries",
        }
    }
    assert provider == "manual_binary"


def test_manual_xpu_wrong_element_count_is_rejected(tmp_path):
    gpu_path = tmp_path / "gpu.bin"
    np.arange(7, dtype=np.float32).tofile(gpu_path)

    case = _make_testcase()
    case.manual_xpu_binaries = (str(gpu_path),)
    _validate(case)

    from ttk.core_modules.common.xpu import ManualXpuLoadError, load_manual_third_parties

    with pytest.raises(ManualXpuLoadError, match="does not match"):
        load_manual_third_parties(case)
