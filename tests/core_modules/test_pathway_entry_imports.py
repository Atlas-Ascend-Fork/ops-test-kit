# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""通路 object/instance 模块导入冒烟：CLI 入口链（Instance → ProfileObject → worker）
任何一环 import 断链都会让整条通路启动即挂。"""

import importlib

import pytest

_PATHWAY_ENTRY_MODULES = (
    "ttk.core_modules.common.instance",
    "ttk.core_modules.kernel.instance",
    "ttk.core_modules.aclnn.instance",
    "ttk.core_modules.geir",
    "ttk.core_modules.framework_api.instance",
)


@pytest.mark.parametrize("module_name", _PATHWAY_ENTRY_MODULES)
def test_pathway_entry_module_importable(module_name):
    importlib.import_module(module_name)
