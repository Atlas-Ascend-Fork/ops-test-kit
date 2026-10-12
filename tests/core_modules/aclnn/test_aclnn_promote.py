# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ACLNN golden promote dtype context."""

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

from ttk.core_modules.aclnn.golden import GoldenGenerator
from ttk.core_modules.aclnn.info_keeper import OpApiInfo
from ttk.core_modules.aclnn.testcase import TestcaseAclnn


def _mock_switches(golden_mode="Promote"):
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = golden_mode
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


def _make_op_api_info(tensor_names, scalar_names=()):
    params = OrderedDict()
    for n in tensor_names:
        params[n] = {"type": "aclTensor*"}
    for n in scalar_names:
        params[n] = {"type": "aclScalar*"}
    return OpApiInfo(params=params)


def _make_testcase(api_name, tensors, tensor_dtypes):
    case = TestcaseAclnn()
    case.testcase_name = f"test_{api_name}_promote"
    case.api_name = api_name
    case.tensors = list(tensors)
    case.tensor_dtypes = tensor_dtypes
    case.scalars = None
    case.attributes = {}
    case.output_tensor_indexes = ()
    case._pure_output_indexes = ()
    case.manual_golden_binaries = None
    return case


class _RecordDtype:
    """class-form golden:无自定义 __init__(走 cls() 守卫);__call__ 记录收到的 dtype。"""

    received = {}

    def __call__(self, x):
        type(self).received["dtype"] = str(x.dtype)
        return [x]


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        monkeypatch.delenv(k, raising=False)


@patch("ttk.core_modules.aclnn.golden.OpApiInfoKeeper")
@patch("ttk.core_modules.aclnn.golden.get_global_storage")
@patch("ttk.core_modules.common.plugin.resolver.get_plugin_function")
class TestAclnnPromote:
    def test_promote_float16_to_float32_and_restore(self, mock_get_plugin, mock_sw, mock_op_info):
        """Promote: float16 输入 → golden(class __call__)收 float32;退出后 ctx 还原 float16。"""
        _RecordDtype.received = {}
        mock_get_plugin.return_value = _RecordDtype
        mock_sw.return_value = _mock_switches("Promote")
        mock_op_info.return_value.info_of.return_value = _make_op_api_info(["x"])
        inp = torch.tensor([1.0, 2.0], dtype=torch.float16)
        case = _make_testcase("aclnnRec", [inp], ("float16",))
        GoldenGenerator(case)._generate_golden()
        assert _RecordDtype.received.get("dtype") == "torch.float32"
        assert str(case.tensors[0].dtype) == "torch.float16"
