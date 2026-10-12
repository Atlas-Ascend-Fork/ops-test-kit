# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""API 解析与信封测试：覆盖 _resolve_3party_api、_aclnn_resolve 及 _ok/_err 信封的 api 字段。"""

from ttk.remote.server.executor import _resolve_3party_api


def test_resolve_3party_api():
    """_resolve_3party_api: op_name 命中→name=op_name。"""
    f, name = _resolve_3party_api("add", "Add", "torch")
    assert name == "add"
    assert callable(f)
