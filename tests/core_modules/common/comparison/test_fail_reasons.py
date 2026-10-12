# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for FAIL_REASONS registry keys and stat_rel_err fail reason."""

import numpy as np

# 触发各比对类的注册（装饰器在 import 时执行）
import ttk.core_modules.common.comparison.binary_equal  # noqa: F401
import ttk.core_modules.common.comparison.cosine_similarity  # noqa: F401
import ttk.core_modules.common.comparison.is_close  # noqa: F401
import ttk.core_modules.common.comparison.re_quantize  # noqa: F401
import ttk.core_modules.common.comparison.stat_rel_err  # noqa: F401
from ttk.core_modules.common.comparison.registry import FAIL_REASONS


def test_fail_reasons_keys_exist():
    """FAIL_REASONS 覆盖所有 standard 的失败模式。"""
    expected_keys = {
        "nan_inf_mismatch",
        "threshold_exceeded",
        "bitwise_mismatch",
        "cross_dtype_uncomparable",
        "tolerance_exceeded",
        "similarity_below_threshold",
        "precision_exceeded",
        "third_party_count_mismatch",
        "third_party_unavailable",
        "ratio_exceeded",
        "small_value_exceeded",
    }
    assert expected_keys <= set(FAIL_REASONS.keys())


def test_stat_rel_err_fail_has_reason():
    """stat_rel_err 失败 metrics 有 reason（threshold_exceeded 带 mere/mare 值）。"""
    from ttk.core_modules.common.comparison.registry import ComparisonRegister
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    standards = resolve_tolerance(
        {"float32": {"standard": "stat_rel_err", "threshold": 1e-10}}, None, 1e-8, ["float32"], None
    )
    cls = ComparisonRegister.registry["stat_rel_err"]
    c = cls(np.array([1.0, 2.0]), np.array([1.1, 2.1]), 0, "float32", standards[0].params)
    r = c.compare()
    # mere/mare 超小 threshold → FAIL
    assert not r[2]
    assert "reason" in r[3]
    assert "threshold exceeded" in r[3]["reason"]
    assert "mere(" in r[3]["reason"]
