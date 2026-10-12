# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GEIR 通路入口 import 级回归测试。

断链背景：object.py 曾引用已拆分的 .profiling 模块（profiling.py 早前已拆为
worker/output_parse/perf_parse），`python -m ttk geir` 在 setup_profile_object
延迟 import 处才崩，--validate 早退将其掩盖。本文件把入口链的每个模块钉在
import 级，任何未来拆分/改名不再依赖运行期才暴露。
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "ttk.core_modules.geir",
        "ttk.core_modules.geir.worker",
        "ttk.core_modules.geir.result",
        "ttk.core_modules.geir.compiler",
        "ttk.core_modules.geir.graph_builder",
        "ttk.core_modules.geir.proto_loader",
        "ttk.core_modules.geir.output_parse",
        "ttk.core_modules.geir.perf_parse",
        "ttk.core_modules.geir.testcase",
    ],
)
def test_geir_entry_modules_importable(module_name):
    """入口链全部模块可 import（断链回归钉）。"""
    assert importlib.import_module(module_name) is not None


def test_geir_worker_entry_callable():
    """worker 入口函数存在且可调用（Task.func 的 pickle 引用目标，现名 run）。"""
    worker = importlib.import_module("ttk.core_modules.geir.worker")
    assert callable(worker.run)


def test_geir_instance_registered():
    """import geir 包根（Instance 宿主）触发 @register_instance("geir") 注册。"""
    importlib.import_module("ttk.core_modules.geir")
    from ttk.core_modules.common.instance import _INSTANCE_REGISTRY

    assert "geir" in _INSTANCE_REGISTRY


def test_geir_result_pick_data_off_semantics():
    """CST_OFF/DYN_OFF 语义收进 pick_data（模式关闭时 perf 列出 OFF 标记）。"""
    from ttk.core_modules.geir.result import GeirReturnStructure

    result = GeirReturnStructure()
    result.precision = "1.0"
    result.passed = True
    result.precision_status = "PASS"
    titles = GeirReturnStructure.get_titles()
    row = result.pick_data(titles)
    assert row[titles.index("cst_perf_us")] == "CST_OFF"
    assert row[titles.index("dyn_perf_us")] == "DYN_OFF"
    result.cst_perf_us = 12.5
    row = result.pick_data(titles)
    assert row[titles.index("cst_perf_us")] == 12.5


def test_geir_mark_off_modes_switch_driven():
    """_mark_off_modes 按 -c/-d 开关补 OFF 标记（屏显与 CSV 同源）。

    数据面 None/空区分不了"未启用"与"执行异常"，判定必须走开关：
    未启用模式 perf/precision 双列补 token；启用模式槽位原样保留
    （含异常路径的 None/空——失败经 STATUS 行暴露，不误标 OFF）。"""
    import types

    from ttk.core_modules.geir.result import GeirReturnStructure
    from ttk.core_modules.geir.worker import _mark_off_modes

    def _switches(cst, dyn):
        return types.SimpleNamespace(
            cst_switches=types.SimpleNamespace(enabled=cst), dyn_switches=types.SimpleNamespace(enabled=dyn)
        )

    result = GeirReturnStructure()
    result.cst_perf_us = 4.5
    result.cst_precision = "100.0%"
    _mark_off_modes(result, _switches(cst=True, dyn=False))
    assert result.cst_perf_us == 4.5
    assert result.cst_precision == "100.0%"
    assert result.dyn_perf_us == "DYN_OFF"
    assert result.dyn_precision == "DYN_OFF"

    result = GeirReturnStructure()  # 异常路径：模式 RAN 但槽位 None/空
    _mark_off_modes(result, _switches(cst=True, dyn=False))
    assert result.cst_perf_us is None
    assert result.cst_precision == ""
    assert result.dyn_perf_us == "DYN_OFF"
    assert result.dyn_precision == "DYN_OFF"


def test_geir_pack_result_forwards_batch_consistency_id():
    """_pack_result 透传 testcase.batch_consistency_id（dl=3 CSV 列非空）。"""
    import types

    from ttk.core_modules.geir.execute import GraphExecResult
    from ttk.core_modules.geir.worker import _pack_result

    testcase = types.SimpleNamespace(batch_consistency_id="seed1|axis0|0-3")
    result = _pack_result("const", GraphExecResult(), ("1.0", "log", True, {}), None, testcase)
    assert result.batch_consistency_id == "seed1|axis0|0-3"
    titles = type(result).get_titles()
    assert result.pick_data(titles)[titles.index("batch_consistency_id")] == "seed1|axis0|0-3"

    no_bcid = types.SimpleNamespace(spec=[])
    result2 = _pack_result("dynamic", GraphExecResult(), ("1.0", "log", True, {}), None, no_bcid)
    assert result2.batch_consistency_id is None  # 未声明 batch 字段 → 空，不参与分组
