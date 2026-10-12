#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""asc（AscendC）编译栈：包装 asc_op_compile_base 包。"""

__all__ = ["AscCompiler"]


# Standard Packages
import logging

# Third-Party Packages
from .compiler import Compiler, register_compiler


@register_compiler("asc")
class AscCompiler(Compiler):
    """
    Class Interface for via `asc_op_compile_base` module
    """

    def __init__(self):
        self._initialized = False
        try:
            self._asc = __import__("asc_op_compile_base")
            logging.debug(f"Using asc_op_compile_base module from {self._asc.__file__}")
        except ModuleNotFoundError as e:
            logging.info(f"Import `asc_op_compile_base` failed. Maybe it is not activated: {e}")
        else:
            self._initialized = True
            try:
                self._asc_common = __import__("asc_op_compile_base.common", fromlist=["register"])
            except BaseException as e:
                logging.critical(f"Module `AscCompiler` init failed: {e}")
                raise e

    def is_initialized(self) -> bool:
        return self._initialized

    def set_compile_soc_info(self, soc_version, core_type="AiCore"):
        self._asc.common.platform.platform_info.set_current_compile_soc_info(soc_version, core_type)

    def set_platform_info_res(self, device_id, res: dict):
        self._asc.common.platform.platform_info.set_platform_info_res(device_id, res)

    def get_soc_spec(self, key: str):
        return self._asc.common.platform.platform_info.get_soc_spec(key)

    def get_context(self):
        return self._asc.common.context.op_context.get_context()

    def get_compile_info(self):
        return {}

    def get_param_generalization(self, op_type: str):
        return self._asc_common.register.get_param_generalization(op_type)

    def do_op_tiling(self, *args, **kwargs):
        return self._asc.asc_op_compiler.op_tiling.do_op_tiling(*args, **kwargs)

    def build_config(self, **kwargs):
        return self._asc.common.buildcfg.build_config(**kwargs)

    def get_tiling_op_type(self):
        return self.get_context().get_addition("op_name")

    @property
    def op_context(self):
        return self._asc.common.context.op_context

    @property
    def op_info(self):
        return self._asc.common.context.op_info
