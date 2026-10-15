# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""container 测试：TensorList 嵌套协议——分布推断/展平/按分布折叠/索引访问。"""

import logging

import pytest

from ttk.utilities.container import (
    deep_flatten,
    flatten_by_distribution,
    flatten_groups,
    get,
    infer_list_distribution_from_nesting,
    nest_by_distribution,
    split_distribution,
)

# -- infer_list_distribution_from_nesting ------------------------------------


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        # 两个 flat tensor → (0, 0)
        (((3, 3), (3, 5)), (0, 0)),
        # 两个 tensor_list → (2, 3)
        ((((1,), (2,)), ((3,), (4,), (5,))), (2, 3)),
        # 首位 None 组照常计数（None 槽位与顺序无关） → (2, 0)
        (((None, (2,)), (4,)), (2, 0)),
        # 组内多 None 槽 → (3,)
        (((None, (2,), None),), (3,)),
        # 全 None 组歧义保持叶子 → (0,)
        (((None, None),), (0,)),
    ],
    ids=["flat", "list+list", "leading-none", "multi-none", "all-none-leaf"],
)
def test_infer_distribution_core_patterns(shapes, expected):
    """推断嵌套结构的 tensor list 分布：flat=0，list=元素数，None=0。"""
    assert infer_list_distribution_from_nesting(shapes) == expected


# -- flatten 族 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        # flat 不变
        (((3, 3), (3, 5)), ((3, 3), (3, 5))),
        # 全嵌套展平
        ((((1,), (2,)), ((3,), (4,))), ((1,), (2,), (3,), (4,))),
        # 首位 None 组照常展开，None 槽位保留
        (((None, (2,)), (4,)), (None, (2,), (4,))),
        # 全 None 组保持叶子（歧义语法不扩展）
        (((None, None),), ((None, None),)),
    ],
    ids=["flat", "all-nested", "leading-none-expanded", "all-none-kept-leaf"],
)
def test_flatten_core_patterns(shapes, expected):
    """展平嵌套序列：flat 不变，一层/全嵌套展平，None 保留。"""
    assert flatten_groups(shapes) == expected


def test_flatten_by_distribution_broadcast():
    """CSV 单值广播约定：标量/单元素序列撑满整个 TensorList 槽，逆操作 nest_by_distribution。"""
    dist = (0, 2, 3)
    # 已展开（len==num）原样消费
    assert flatten_by_distribution((1, (2, 3), (4, 5, 6)), dist) == (1, 2, 3, 4, 5, 6)
    # 单元素序列广播
    assert flatten_by_distribution((1, (2,), (3,)), dist) == (1, 2, 2, 3, 3, 3)
    # 标量广播
    assert flatten_by_distribution((1, 2, 3), dist) == (1, 2, 2, 3, 3, 3)
    # 与 nest_by_distribution 互逆（tuple 切片保持 tuple 组）
    flat = flatten_by_distribution((1, (2, 3), (4, 5, 6)), dist)
    assert nest_by_distribution(flat, dist) == [1, (2, 3), (4, 5, 6)]


def test_flatten_by_distribution_whole_column_broadcast():
    """整列广播：单值（len==1）撑满所有槽。"""
    assert flatten_by_distribution(("f32",), (2, 0)) == ("f32", "f32", "f32")
    assert flatten_by_distribution("f32", (2, 0)) == ("f32", "f32", "f32")  # 裸标量


def test_flatten_by_distribution_pre_flattened():
    """预展平约定：len == 逐张量数（sum(max(d,1))）时按张量序折叠，不再静默错位。"""
    assert flatten_by_distribution(("f32", "i8", "f64"), (2, 0)) == ("f32", "i8", "f64")
    assert flatten_by_distribution(("a", "b"), (2,)) == ("a", "b")


def test_flatten_by_distribution_misaligned_raises():
    """既非槽对齐也非预展平的输入显式报错。"""
    with pytest.raises(ValueError, match="not aligned"):
        flatten_by_distribution(("a", "b", "c", "d", "e"), (2, 0, 0))
    with pytest.raises(ValueError, match="not aligned"):
        flatten_by_distribution((), (2, 0))


def test_flatten_by_distribution_slot_group_mismatch_raises():
    """槽内组长度既非 num 也非 1 时显式报错（原为静默 extend 错位）。"""
    with pytest.raises(ValueError, match="slot 0 expects 3 values, got 2"):
        flatten_by_distribution((("a", "b"), "c"), (3, 0))


# -- get ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("container", "idx", "expected"),
    [
        (["a", "b"], 0, "a"),
        (["a", "b"], -1, "b"),  # 负 idx 在界内合法
        (["only"], 99, "only"),  # 单元素广播
        (["a", "b"], 99, "OOR"),  # 越界 fallback
    ],
    ids=["in-range", "negative", "single-broadcast", "out-of-range"],
)
def test_get_indexing(container, idx, expected):
    assert get(container, idx, out_of_range="OOR" if expected == "OOR" else None) == expected


def test_get_out_of_range_without_fallback_logs_error_and_raises(caplog):
    """契约：无 fallback 时 ERROR 日志带全要素（idx/len/container），IndexError 照常抛出。"""
    with caplog.at_level(logging.ERROR), pytest.raises(IndexError):
        get([1, 2], 5)
    assert "idx=5" in caplog.text
    assert "len=2" in caplog.text
    assert "[1, 2]" in caplog.text

    with caplog.at_level(logging.ERROR), pytest.raises(IndexError):
        get([1, 2], -5)
    assert "idx=-5" in caplog.text


# -- nest_by_distribution / split_distribution ---------------------------------------


def test_apply_as_list_fold():
    assert nest_by_distribution([1, 2, 3], [2]) == [[1, 2], 3]
    assert nest_by_distribution([1, 2, 3, 4], (0, 2, 0)) == [1, [2, 3], 4]
    assert nest_by_distribution([1, 2], ()) == [1, 2]  # 空 dist 原样返回


@pytest.mark.parametrize(
    ("dist", "first_count", "expected"),
    [
        ((0, 2, 0, 2), 3, ((0, 2), (0, 2))),  # 3 个输入元素：前两槽为输入段
        ((0, 2, 0, 2), 4, ((0, 2, 0), (2,))),  # 4 个输入元素：边界恰在槽间
        ((0, 2, 0), 4, ((0, 2, 0), ())),  # 全部为输入：输出段空
        ((), 0, ((), ())),  # 空 dist
    ],
)
def test_split_distribution(dist, first_count, expected):
    """槽不跨段：累计消费达到 first_count 的槽起归输出段（判定先于消费）。"""
    assert split_distribution(dist, first_count) == expected


def test_deep_flatten_atomic_elements_not_traversed():
    """原子元素（ndarray/str/标量）绝不展开内部——只展开 tuple/list/set 容器。"""
    import numpy as np

    arr = np.zeros((2, 2))
    assert deep_flatten(((arr, "ab"), 3)) == (arr, "ab", 3)
