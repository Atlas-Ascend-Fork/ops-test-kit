# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""params 测试：算子参数拾取——pickup 分拣（保键名）与 adapt 签名适配（改键名）共享三变换。"""

import pytest

from ttk.core_modules.kernel.params import adapt_pickup_by_names, pickup_by_names


@pytest.mark.parametrize(
    ("input_names", "attrs", "expected_extract"),
    [
        (["x"], {}, {}),
        ([], {"axis": 0}, {}),
        # direct name match
        (
            ["input_size", "filter", "out_backprop"],
            {"input_size": [2, 3, 18, 130, 130], "strides": [1, 1, 1, 1, 1]},
            {"input_size": [2, 3, 18, 130, 130]},
        ),
        # _in__ suffix matches input name
        (
            ["input_size", "filter"],
            {"input_size_in__": [2, 3, 18], "strides": [1]},
            {"input_size_in__": [2, 3, 18]},
        ),
        # both direct and suffixed present — union
        (
            ["input_size", "filter"],
            {"input_size": [2, 3], "input_size_in__": [1, 1], "strides": [1]},
            {"input_size": [2, 3], "input_size_in__": [1, 1]},
        ),
        # axis/axes alias, both directions
        (["x", "axes"], {"axis": 1}, {"axis": 1}),
        (["x", "axis"], {"axes": [1, 2]}, {"axes": [1, 2]}),
        # axis alias with _in__ suffix
        (["x", "axes"], {"axis_in__": 1}, {"axis_in__": 1}),
        # no match — nothing extracted
        (["x", "y"], {"axis": 0, "keep_dims": True}, {}),
        # conv3d style: 常量输入声明混在属性里
        (
            ["input_size", "filter", "out_backprop"],
            {
                "input_size": [2, 3, 18, 130, 130],
                "strides": [1, 1, 1, 1, 1],
                "pads": [0, 0, 0, 0, 0, 0],
                "dilations": [1, 1, 1, 1, 1],
                "groups": 1,
                "data_format": "NCDHW",
            },
            {"input_size": [2, 3, 18, 130, 130]},
        ),
        # ctc loss style: 多常量输入
        (
            ["log_probs", "targets", "input_lengths", "target_lengths"],
            {"input_lengths": (70,) * 5, "target_lengths": (26, 11), "blank": 0, "reduction": "none"},
            {"input_lengths": (70,) * 5, "target_lengths": (26, 11)},
        ),
    ],
    ids=[
        "empty-attributes",
        "empty-input-names",
        "direct-name-match",
        "in-suffix-matches-input",
        "both-direct-and-suffixed-present",
        "axis-axes-alias-direct",
        "axes-axis-alias-direct",
        "axis-alias-with-in-suffix",
        "no-match-all-attrs",
        "conv3d-style-mixed",
        "ctc-loss-style",
    ],
)
def test_pickup_by_names(input_names, attrs, expected_extract):
    """按输入名单拣出输入声明：直配/_in__/axis 别名三路命中，键名原样保留。"""
    result = pickup_by_names(attrs, input_names)
    assert result == expected_extract
    assert set(result) <= set(attrs)  # 拣出键必来自原 attrs（无幻影键）


@pytest.mark.parametrize(
    ("signatures", "params", "expected"),
    [
        # already-canonical passthrough
        (("axes", "strides"), {"axes": 1, "strides": 2}, {"axes": 1, "strides": 2}),
        # axis ↔ axes rename, both directions
        (("axes",), {"axis": 1}, {"axes": 1}),
        (("axis",), {"axes": [1, 2]}, {"axis": [1, 2]}),
        # _in__ suffix add / strip
        (("beta_in__",), {"beta": 2}, {"beta_in__": 2}),
        (("gamma",), {"gamma_in__": 3}, {"gamma": 3}),
        # alias composed with _in__ suffix
        (("axes_in__",), {"axis": 1}, {"axes_in__": 1}),
        # both forms present — canonical wins, raw form dropped
        (("axes",), {"axis": 1, "axes": 2}, {"axes": 2}),
        # non-matching params removed
        (("axes",), {"axis": 1, "keep_dims": True}, {"axes": 1}),
        # empty signatures — nothing matches
        ((), {"axis": 1}, {}),
    ],
    ids=[
        "passthrough",
        "axis-to-axes",
        "axes-to-axis",
        "add-in-suffix",
        "strip-in-suffix",
        "alias-with-in-suffix",
        "both-forms-canonical-wins",
        "non-match-removed",
        "empty-signatures",
    ],
)
def test_adapt_pickup_by_names(signatures, params, expected):
    """键名改写到签名形态后构造 kwargs：axis↔axes、_in__ 加/剥，不匹配删除。"""
    assert adapt_pickup_by_names(params, signatures) == expected
