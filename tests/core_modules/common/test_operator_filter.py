# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.common.instance import _check_testcase_operator_selection


@pytest.mark.parametrize(
    ("selected", "excluded", "operator", "expected"),
    [
        (("add", "sub"), ("sub",), "add", True),
        (("add", "sub"), ("sub",), "sub", False),
        (("add", "sub"), ("sub",), "mul", False),
        (None, ("sub",), "add", True),
        (None, ("sub",), "sub", False),
        (("add",), None, "add", True),
        (("add",), None, "sub", False),
        (None, None, "mul", True),
    ],
)
def test_operator_include_and_exclude_filters_intersect(selected, excluded, operator, expected):
    storage = MagicMock()
    storage.selected_operators = selected
    storage.excluded_operators = excluded
    with patch("ttk.core_modules.common.instance.get_global_storage", return_value=storage):
        assert _check_testcase_operator_selection(operator) is expected
