#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TestcaseBase 公共 CSV 契约：device_ids 解析与 batch_consistency_id 派生。"""

import pytest

from ttk.core_modules.common.testcase.base import TestcaseBase


def _case(**kwargs):
    case = TestcaseBase()
    case.is_valid = True
    for key, value in kwargs.items():
        setattr(case, key, value)
    return case


class TestDeviceIds:
    """device_ids CSV 契约："0,1,2" 多卡 / "3" 单卡 / 空调度自选 / 非法值报错。"""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0,1,2", (0, 1, 2)),
            ("3", (3,)),
            (" 1 , 2 ", (1, 2)),
            ("", None),
            (None, None),
            ((0, 1), (0, 1)),  # 已 tuple 透传
            ([2], (2,)),  # list 归一 tuple
        ],
        ids=["multi", "single", "spaced", "empty", "none", "tuple-passthrough", "list-normalized"],
    )
    def test_parse(self, raw, expected):
        case = _case(device_ids=raw)
        case.validate()
        assert case.device_ids == expected

    @pytest.mark.parametrize(
        ("device_ids", "expected"),
        [
            ("0,1,2", True),
            ("3", False),
            ("", False),
        ],
        ids=["multi", "single", "unset"],
    )
    def test_is_multi_device(self, device_ids, expected):
        case = _case(device_ids=device_ids)
        case.validate()
        assert case.is_multi_device() is expected

    def test_invalid_value_invalidates_case(self):
        """非法 token 不再炸 parse_cases：置 is_valid=False + 明确 fail_reason。"""
        case = _case(device_ids="0,x")
        case.validate()
        assert not case.is_valid
        assert "invalid device_ids" in case.fail_reason
        assert "'all'" in case.fail_reason  # 提示合法语法
        assert case.device_ids is None


class TestDeviceIdsAll:
    """device_ids='all'：展开为探测/--dev 池全量；count 未知时明确报错。"""

    def test_all_expands_to_device_count(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr("ttk.switches.get_global_storage", lambda: SimpleNamespace(device_count=8))
        case = _case(device_ids="all")
        case.validate()
        assert case.is_valid
        assert case.device_ids == tuple(range(8))
        assert case.is_multi_device() is True

    @pytest.mark.parametrize("raw", ["ALL", "All", " all "], ids=["upper", "mixed", "spaced"])
    def test_all_case_insensitive(self, monkeypatch, raw):
        from types import SimpleNamespace

        monkeypatch.setattr("ttk.switches.get_global_storage", lambda: SimpleNamespace(device_count=2))
        case = _case(device_ids=raw)
        case.validate()
        assert case.device_ids == (0, 1)

    def test_all_unknown_count_invalidates(self, monkeypatch):
        """--validate-only 早退于探测：count 未知（-1）时明确报错而非静默空集。"""
        from types import SimpleNamespace

        monkeypatch.setattr("ttk.switches.get_global_storage", lambda: SimpleNamespace(device_count=-1))
        case = _case(device_ids="all")
        case.validate()
        assert not case.is_valid
        assert "requires a known device count" in case.fail_reason
        assert "--dev N" in case.fail_reason
        assert case.device_ids is None


class TestBatchConsistencyId:
    """batch_seed/axis/slice_info 齐备时派生 bcid；任一缺失 → None 不参与分组。"""

    def test_full_relation_derives_id(self):
        case = _case(
            batch_axis=((0,),),
            batch_slice_info=(((((0, 3, 1),)),),),
            batch_seed=(((7,),),),
        )
        case.validate()
        # seed_axis_start_stop_step
        assert case.batch_consistency_id == ((("7_0_0_3_1",),),)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"batch_axis": ((0,),), "batch_slice_info": (((((0, 3, 1),)),),)},
            {"batch_axis": ((0,),), "batch_seed": (((7,),),)},
        ],
        ids=["none", "missing-seed", "missing-slice-info"],
    )
    def test_partial_relation_yields_none(self, kwargs):
        case = _case(**kwargs)
        case.validate()
        assert case.batch_consistency_id is None


class TestFlatIndexOf:
    """flat_index_of：嵌套槽序号 → flat 张量槽序号（TensorList 分布前缀和）。"""

    def test_prefix_sum_with_tensor_list(self):
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn

        case = TestcaseAclnn()
        case.tensor_view_shapes = (((1,), (1,)), (2,), (3,))  # TensorList(2) + 两个单槽
        assert case.tensor_list_dist == (2, 0, 0)
        assert case.flat_index_of(0) == 0
        assert case.flat_index_of(1) == 2  # 跳过 TensorList 的 2 个 flat 槽
        assert case.flat_index_of(2) == 3

    def test_identity_without_distribution(self):
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn

        case = TestcaseAclnn()
        case.tensor_view_shapes = ((1,), (2,), (3,))
        assert case.tensor_list_dist == (0, 0, 0)
        assert [case.flat_index_of(i) for i in range(3)] == [0, 1, 2]
