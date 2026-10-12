#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for ttk.remote.server.execution_container helpers.

Covers ``bind_params``, ``match_params_v1`` and ``resolve_callable``.
"""

import numpy as np
import pytest

from ttk.remote.server.execution_container import (
    UnknownParamError,
    bind_params,
    match_params_v1,
    resolve_callable,
)


class TestMatchParamsV1:
    """match_params_v1: 由 schema 与扁平数组还原命名输入。"""

    @pytest.mark.parametrize(
        ("schema", "flat_vals", "expected"),
        [
            pytest.param(
                [{"name": "x", "index": 0}, {"name": "y", "index": 1}],
                [[1.0], [2.0]],
                {"x": 0, "y": 1},
                id="single_tensors",
            ),
            pytest.param(
                [{"name": "x", "index": 0}, {"name": "z", "index": None}, {"name": "y", "index": 1}],
                [[1.0], [2.0]],
                {"x": 0, "z": None, "y": 1},
                id="none_optional",
            ),
        ],
    )
    def test_match_by_index(self, schema, flat_vals, expected):
        """index 绑定单张量；index=None 绑定 None。"""
        flat = [np.array(v) for v in flat_vals]
        result = match_params_v1(schema, flat)
        for name, idx in expected.items():
            if idx is None:
                assert result[name] is None
            else:
                assert result[name] is flat[idx]


class TestBindParams:
    """bind_params: 按名绑定，* 决定位置/关键字，未知即抛错 (spec §7.2-7.4)。"""

    @staticmethod
    def test_required_param_missing_raises_with_leftover():
        """必填参数缺失时不得贪婪绑定未消费项(否则静默绑错值)。"""

        class Impl:
            @staticmethod
            def __call__(x, scale):
                return (x, scale)

        # 'scale' 非 input/attr; 'alpha' 是未消费项, 不得被贪婪绑定给 'scale'。
        with pytest.raises(UnknownParamError):
            bind_params(Impl.__call__, {"x": 1, "alpha": 0.5})

    @staticmethod
    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            pytest.param({"reduction": "none"}, (2.0, 0.25, "none"), id="first_two_omitted"),
            pytest.param({"gamma": 2.0, "reduction": "none"}, (2.0, 0.25, "none"), id="middle_omitted"),
            pytest.param({"alpha": -1.6e38, "reduction": "none"}, (2.0, -1.6e38, "none"), id="head_omitted"),
            pytest.param({"gamma": 3.689, "alpha": 0.7, "reduction": "none"}, (3.689, 0.7, "none"), id="all_given"),
        ],
    )
    def test_omitted_earlier_optional_keeps_slot_alignment(attrs, expected):
        """省略前序可选属性时, 后面的值按名落位不得左移错位(类形式三方绑定场景)。"""

        class Impl:
            def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
                self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

        pool = dict(attrs)
        pool.update({"pred": "P", "target": "T", "weight": None})
        ia, ik = bind_params(Impl.__init__, pool, warn_leftover=False)
        inst = Impl(*ia, **ik)
        assert (inst.gamma, inst.alpha, inst.reduction) == expected

    @staticmethod
    def test_pre_star_default_holds_slot_for_varargs_and_kwonly():
        """缺省的前置可选参数占位后, *args 收集剩余、keyword-only 仍走 Python 默认。"""

        def f(a=1, b=2, *rest, key=None):
            return (a, b, rest, key)

        ia, ik = bind_params(f, {"b": 9, "zzz": "leftover", "key": "K"})
        assert f(*ia, **ik) == (1, 9, ("leftover",), "K")

        def g(x, *, axis=-1):
            return (x, axis)

        ia, ik = bind_params(g, {"x": 1})
        assert g(*ia, **ik) == (1, -1)


class TestResolveCallable:
    """resolve_callable: 点分 api 字符串 -> 可调用对象；拒绝类 (spec §7.6)。"""

    @pytest.mark.parametrize(
        ("api_str", "expect"),
        [
            pytest.param("numpy.abs", "is_numpy_abs", id="function"),
            pytest.param("numpy.linalg.norm", "is_callable", id="nested_attr"),
        ],
    )
    def test_resolves_dotted(self, api_str, expect):
        """点分路径解析为函数。"""
        fn = resolve_callable(api_str)
        if expect == "is_numpy_abs":
            assert fn is np.abs
        else:
            assert callable(fn)
