# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""run() 收尾恒触发 Backend.reset（dynamo cache 无淘汰，跨用例累计触顶回退 eager）。

异常路径同样覆盖；清理失败不影响用例结果（mirror 测试的裸 object backend
已隐式覆盖该守卫）。
"""

from types import SimpleNamespace

from ttk.core_modules.common.dump import DumpConfig
from ttk.core_modules.framework_api import worker as e2e_worker
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.switches import SWITCHES


def _switches():
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = False
    switches.manual_data_dirs = ()
    switches.cst_switches.enabled = False
    switches.dyn_switches.enabled = False
    return switches


def _e2e_case(name):
    case = TestcaseE2e()
    case.testcase_name = name
    case.api_name = "torch.add"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.golden_api = ""
    return case


def test_run_finally_calls_backend_reset(monkeypatch, tmp_path):
    """run() 收尾恒触发清理：_run_profile 抛错（异常路径）同样走 finally。"""
    switches = _switches()
    ctx = SimpleNamespace(notify_status=lambda *_: None, change_name=lambda *_: None, storage={})
    released = []

    def _raise_profile(*_a, **_k):
        raise RuntimeError("boom")

    backend = SimpleNamespace(reset=released.append)
    for name, stub in {
        "get_global_storage": lambda: switches,
        "process_context": lambda: ctx,
        "reseed_per_testcase": lambda *a, **k: None,
        "gate_host_memory": lambda *a, **k: None,
        "build_single_log_dir": lambda *a, **k: str(tmp_path),
        "default_logging_config": lambda *a, **k: None,
        "_get_or_create_backend": lambda _sw: backend,
        "resolve_deterministic_level": lambda _sw, _tc: 0,
        "_ensure_deterministic_level": lambda *a, **k: None,
        "_run_profile": _raise_profile,
        "_attach_plog_errors": lambda err: err,
        "_profiling_end_print": lambda *a, **k: None,
    }.items():
        monkeypatch.setattr(e2e_worker, name, stub)

    ret = e2e_worker.run(_e2e_case("cache_release"), None, None, 7)

    assert released == [7], "异常路径也须触发 backend.reset(dev_id)"
    assert ret.precision_status == "FAIL"
