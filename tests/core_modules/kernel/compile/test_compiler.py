# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""编译器注册制回归：注册表查找/单例/duplicate 防护 + core limit 资源映射。"""

import pytest

from ttk.core_modules.kernel.compile import compiler as compiler_mod
from ttk.core_modules.kernel.compile.compiler import Compiler, get_compiler, register_compiler


class TestRegistry:
    def test_default_stacks_registered(self):
        get_compiler()  # 注册触发（_fill_registry 懒执行）
        assert "tbe" in compiler_mod._COMPILER_REGISTRY
        assert "asc" in compiler_mod._COMPILER_REGISTRY

    def test_get_compiler_returns_singleton(self):
        assert get_compiler("tbe") is get_compiler("tbe")
        assert get_compiler("asc") is get_compiler("asc")

    def test_get_compiler_default_is_tbe(self):
        assert get_compiler() is get_compiler("tbe")

    def test_unknown_compiler_raises_keyerror(self):
        with pytest.raises(KeyError, match="bisheng"):
            get_compiler("bisheng")

    def test_duplicate_registration_rejected(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_compiler("tbe")
            class _Dup(Compiler):
                def is_initialized(self):
                    return False

                def set_compile_soc_info(self, soc_version, core_type="AiCore"): ...

                def set_platform_info_res(self, device_id, res: dict): ...

                def get_soc_spec(self, key: str): ...

                def get_context(self): ...

                def get_compile_info(self): ...

                def get_param_generalization(self, op_type: str): ...

                def do_op_tiling(self, *args, **kwargs): ...

                def build_config(self, **kwargs): ...

                def get_tiling_op_type(self): ...

                @property
                def op_context(self): ...

                @property
                def op_info(self): ...

    def test_stack_implements_are_compiler_subclasses(self):
        assert isinstance(get_compiler("tbe"), Compiler)
        assert isinstance(get_compiler("asc"), Compiler)


class TestCoreLimitRes:
    @pytest.mark.parametrize(
        ("core_type", "core_limit", "expected"),
        [
            ("AiCore", (8, 48), {"ai_core_cnt": "8"}),
            ("VectorCore", (8, 48), {"vector_core_cnt": "48"}),
            ("CubeCore", (8, 48), {"ai_core_cnt": "8"}),
            ("MIX", (None, 48), {"vector_core_cnt": "48"}),
            ("AiCore", (None, 48), None),
            ("AiCore", None, None),
            ("AiCore", "bad", None),
        ],
    )
    def test_core_limit_res_mapping(self, core_type, core_limit, expected):
        assert compiler_mod._core_limit_res(core_type, core_limit) == expected
