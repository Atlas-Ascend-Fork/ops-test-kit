# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""通路注册表契约：register_instance / get_instance_by_name（真实链路，非 mock）。

注册是 import 副作用（bridge 按 test_mode import 通路 instance 模块触发），
本文件显式导入四通路 instance 模块以填充注册表。kernel 双名注册（"op" 旧名
+ "kernel" CLI 名）。"""

import pytest

import ttk.core_modules.aclnn.instance  # noqa: F401  注册副作用
import ttk.core_modules.framework_api.instance  # noqa: F401
import ttk.core_modules.geir  # noqa: F401  # Instance 宿主为包根
import ttk.core_modules.kernel.instance  # noqa: F401
from ttk.core_modules.common.instance import _INSTANCE_REGISTRY, InstanceBase, get_instance_by_name, register_instance


def test_four_modes_registered():
    """CLI 子命令 ↔ Instance 注册接线：四模式名全部在册（不实例化，避免设备依赖）。"""
    for mode in ("op", "kernel", "aclnn", "framework-api", "geir"):
        assert mode in _INSTANCE_REGISTRY, mode
        assert issubclass(_INSTANCE_REGISTRY[mode], InstanceBase)


def test_get_instance_by_name_unknown_raises_with_listing():
    with pytest.raises(KeyError, match="Unknown test mode 'nope'"):
        get_instance_by_name("nope")


def test_register_and_lookup_roundtrip():
    """注册 → 惰性单例查找 → 重复注册拒绝（用假通路名，不碰真实四模式）。"""

    @register_instance("fake-mode-for-registry-test")
    class FakeInstance:
        pass

    try:
        first = get_instance_by_name("fake-mode-for-registry-test")
        second = get_instance_by_name("fake-mode-for-registry-test")
        assert first is second  # 惰性单例：首查实例化，之后复用

        with pytest.raises(ValueError, match="already registered"):

            @register_instance("fake-mode-for-registry-test")
            class Duplicate:
                pass

    finally:
        _INSTANCE_REGISTRY.pop("fake-mode-for-registry-test", None)
