# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GEIR 用例 SOC 检查的 proto 回落测试。"""

from unittest.mock import MagicMock, patch

from ttk.core_modules.geir.testcase import GeirTestcase


def test_soc_check_falls_back_to_proto_registry():
    """ops-info 无条目时：proto 有 REG_OP 声明（纯图级算子）放行；查不到维持 SOC_NOT_SUPPORT。"""
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper.info_of", return_value=None), patch(
        "ttk.core_modules.geir.proto_loader.ProtoLoader.get_op_info"
    ) as proto_get:
        proto_get.return_value = MagicMock()
        case = GeirTestcase()
        case.op_name = "histogram"
        case._check_op_name()
        assert case.is_valid
        assert case.fail_reason is None

        proto_get.return_value = None
        case2 = GeirTestcase()
        case2.op_name = "no_such_op"
        case2._check_op_name()
        assert not case2.is_valid
        assert case2.fail_reason == "SOC_NOT_SUPPORT"

        # 此前已判无效的用例不得被回落复活（was_valid 守卫）
        proto_get.reset_mock()
        proto_get.return_value = MagicMock()
        case3 = GeirTestcase()
        case3.op_name = "histogram"
        case3.is_valid = False
        case3.fail_reason = "CASE_FIELD_AMBIGUOUS"
        case3._check_op_name()
        assert not case3.is_valid
        proto_get.assert_not_called()
