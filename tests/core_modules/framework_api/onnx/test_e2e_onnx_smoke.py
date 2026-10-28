# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""onnx e2e 全链路 smoke（NPU 设备）。

场景矩阵（默认仅 eager 的运行被双开场景覆盖，不单列）：
--eager -c 双开单次执行且 EAGER 与 GRAPH CST 两列同值（去重）；
--eager=false -c 单开落 GRAPH CST 列。
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
CSV = REPO / "examples" / "case_store" / "e2e" / "onnx_cases.csv"

SIX = ("onnx_add_f32", "onnx_add_f16", "onnx_add_broadcast", "onnx_concat_i32", "onnx_split_f32", "onnx_matmul_bf16")


def _npu_available() -> bool:
    try:
        from ttk.core_modules.common.device.dsmi.interface import DSMIInterface

        return DSMIInterface().get_device_count() > 0
    except Exception:  # noqa: BLE001  # 探测失败按无设备处理
        return False


def _atc_available() -> bool:
    """CANN 工具链在位（驱动-only 的 NPU 机器无 atc，编译必败）。"""
    try:
        from ttk.core_modules.framework_api.onnx.om_compiler import _find_atc

        return _find_atc()[0] is not None
    except Exception:  # noqa: BLE001  # 探测失败按无 atc 处理
        return False


def _onnx_pathway_supported() -> bool:
    """onnx 在位且 opset 不超 atc 插件上限（param_extractor 同款产品门禁，
    不锚定版本号——上限取自 param_extractor.ATC_MAX_OPSET 随 CANN 维护）。"""
    try:
        import onnx.defs

        from ttk.core_modules.framework_api.onnx.param_extractor import ATC_MAX_OPSET

        return onnx.defs.onnx_opset_version() <= ATC_MAX_OPSET
    except Exception:  # noqa: BLE001  # onnx 缺席按不支持处理
        return False


def _split_num_outputs_supported() -> bool:
    """样例 CSV 的 Split 用例依赖 num_outputs 属性（注册表实际声明判定）。"""
    try:
        import onnx.defs

        return "num_outputs" in onnx.defs.get_schema("Split").attributes
    except Exception:  # noqa: BLE001  # onnx 缺席按不支持处理
        return False


pytestmark = [
    pytest.mark.device,
    pytest.mark.slow,
    pytest.mark.skipif(not _npu_available(), reason="no NPU device detected via DSMI"),
    pytest.mark.skipif(not _atc_available(), reason="atc not found (driver-only host without CANN toolkit)"),
    pytest.mark.skipif(
        not _onnx_pathway_supported(), reason="onnx missing or env opset exceeds the atc plugin ceiling"
    ),
    pytest.mark.skipif(
        not _split_num_outputs_supported(), reason="installed Split schema lacks the num_outputs attribute"
    ),
    pytest.mark.skipif(not CSV.is_file(), reason="onnx sample csv missing"),
]


def _run_e2e(*extra_args, timeout=1800):
    cmd = [sys.executable, "-m", "ttk", "e2e", "-i", str(CSV), *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO), check=False)


def _case_statuses(output: str) -> dict:
    return dict(re.findall(r"Testcase (\S+) finished: precision_status=(\w+)", output))


def _assert_all_pass(result, label):
    statuses = _case_statuses(result.stdout + result.stderr)
    assert set(statuses) >= set(SIX), f"{label}: statuses={statuses}"
    for name in SIX:
        assert statuses[name] == "PASS", f"{label}/{name}: {statuses[name]}"


def test_smoke_dual_mode_single_execution():
    """--eager -c 双开：单次执行（无 OnGraphCst 阶段推进）+ 两列同值（镜像）。"""
    result = _run_e2e("--eager", "-c")
    assert result.returncode == 0, result.stdout[-3000:]
    _assert_all_pass(result, "dual")
    combined = result.stdout + result.stderr
    assert "OnGraphCst" not in combined  # cst 通道不再二次执行
    # EAGER 与 GRAPH CST 两列同值（同一份执行镜像；每用例两行相邻打印，序一致）
    eager_perf = re.findall(r"\|\s*EAGER\s*\|\s*([\d.]+)\s*\|", combined)
    cst_perf = re.findall(r"\|\s*GRAPH CST\s*\|\s*([\d.]+)\s*\|", combined)
    assert len(eager_perf) >= len(SIX)
    assert eager_perf == cst_perf


def test_smoke_const_only():
    """--eager=false -c 单开：落 GRAPH CST 列（PASS + perf），EAGER 行不出现。"""
    result = _run_e2e("--eager=false", "-c")
    assert result.returncode == 0, result.stdout[-3000:]
    _assert_all_pass(result, "const-only")
    assert re.search(r"\|\s*GRAPH CST\s*\|\s*\d+\.\d+", result.stdout)
    assert "| EAGER " not in result.stdout.replace("GRAPH CST", "")
