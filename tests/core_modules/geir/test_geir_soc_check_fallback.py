# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GEIR 用例 SOC 检查对纯图级 IR 算子的回落判定。

背景：_check_op_name 源自 kernel 血统（ops-info 条目 = 本 SOC 有可编译执行的
kernel 实现），但 geir 测的是图编译通路，fusion pass 会在编译前消除部分老 IR
节点（如 Histogram -> HistogramV2），这类算子按官方实践不登记 ops-info
（canndev 各 arch ini 均无其条目）。命中 SOC_NOT_SUPPORT 时回落 proto
注册表：有 REG_OP 声明即视为可测，proto 也查不到才维持 SOC_NOT_SUPPORT。
"""

from unittest.mock import MagicMock, patch

from ttk.core_modules.geir.testcase import GeirTestcase


def _make_case(op_name="histogram"):
    case = GeirTestcase()
    case.op_name = op_name
    return case


def test_soc_check_passes_with_ops_info_entry():
    """ops-info 有条目：维持 kernel 血统判定，直接通过，不碰 proto 回落。"""
    case = _make_case("histogram_v2")
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper.info_of", return_value=MagicMock()) as info_of, patch(
        "ttk.core_modules.geir.proto_loader.ProtoLoader.get_op_info"
    ) as proto_get:
        case._check_op_name()
    assert case.is_valid
    assert case.fail_reason is None
    info_of.assert_called_once_with("histogram_v2")
    proto_get.assert_not_called()


def test_soc_check_falls_back_to_proto_registry():
    """ops-info 无条目但 proto 有 REG_OP 声明（纯图级算子）：放行。"""
    case = _make_case("histogram")
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper.info_of", return_value=None), patch(
        "ttk.core_modules.geir.proto_loader.ProtoLoader.get_op_info", return_value=MagicMock()
    ):
        case._check_op_name()
    assert case.is_valid
    assert case.fail_reason is None


def test_soc_check_keeps_failure_when_proto_also_missing():
    """ops-info 与 proto 都查不到：真不认识，维持 SOC_NOT_SUPPORT。"""
    case = _make_case("no_such_op")
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper.info_of", return_value=None), patch(
        "ttk.core_modules.geir.proto_loader.ProtoLoader.get_op_info", return_value=None
    ):
        case._check_op_name()
    assert not case.is_valid
    assert case.fail_reason == "SOC_NOT_SUPPORT"


def test_soc_check_does_not_rescue_op_name_missing():
    """op_name 缺失是 OP_NAME_MISSING，不属于回落范围。"""
    case = _make_case("")
    with patch("ttk.core_modules.geir.proto_loader.ProtoLoader.get_op_info") as proto_get:
        case._check_op_name()
    assert not case.is_valid
    assert case.fail_reason == "OP_NAME_MISSING"
    proto_get.assert_not_called()


def test_soc_check_does_not_resurrect_case_invalid_for_other_reasons():
    """base 归一化已判无效（如字段歧义）的用例不得被 proto 回落复活。

    基类 _check_op_name 在 ops-info 缺失时无条件覆写 fail_reason=SOC_NOT_SUPPORT，
    若只看最终 fail_reason 会把"用例本身配错"误判为"仅 SOC 失败"而放行。
    """
    case = _make_case("histogram")
    case.is_valid = False
    case.fail_reason = "CASE_FIELD_AMBIGUOUS"
    with patch("ttk.core_modules.kernel.info_keeper.OpInfoKeeper.info_of", return_value=None), patch(
        "ttk.core_modules.geir.proto_loader.ProtoLoader.get_op_info", return_value=MagicMock()
    ) as proto_get:
        case._check_op_name()
    assert not case.is_valid
    # proto 回落未触发；fail_reason 已被基类覆写为 SOC_NOT_SUPPORT（基类既有行为），
    # 关键是用例保持无效、不进入建图/编译
    assert case.fail_reason == "SOC_NOT_SUPPORT"
    proto_get.assert_not_called()
