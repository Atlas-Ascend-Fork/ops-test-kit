# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""precision_metrics / metrics 字段的参数化测试。

覆盖字段存在性：各结构（ComparisonResult / KernelReturnStructure /
GeirReturnStructure）均应含 precision_metrics（或 ComparisonResult 的 metrics）字段。
"""

import pytest

from ttk.core_modules.geir.result import GeirReturnStructure
from ttk.core_modules.kernel.result import ComparisonResult, KernelReturnStructure


@pytest.mark.parametrize(
    ("cls", "factory", "field_name", "is_geir"),
    [
        pytest.param(ComparisonResult, lambda: ComparisonResult(None), "metrics", False, id="comparison_result"),
        pytest.param(
            KernelReturnStructure,
            lambda: KernelReturnStructure(),
            "precision_metrics",
            False,
            id="profiling_structure",
        ),
        pytest.param(
            GeirReturnStructure, lambda: GeirReturnStructure(), "precision_metrics", True, id="geir_structure"
        ),
    ],
)
def test_precision_metrics_slot_exists(cls, factory, field_name, is_geir):
    """各结构均应含 precision_metrics（或 ComparisonResult 的 metrics）字段。

    slots-based 结构检查 __slots__；GeirReturnStructure 是 dataclass，
    检查 get_titles() 含该字段且位于 xpu_metrics 之后。
    """
    instance = factory()
    assert hasattr(instance, field_name), f"{cls.__name__} missing {field_name}"
    if is_geir:
        titles = cls.get_titles()
        assert field_name in titles
        assert titles.index(field_name) == titles.index("xpu_metrics") + 1
    else:
        assert field_name in cls.__slots__
