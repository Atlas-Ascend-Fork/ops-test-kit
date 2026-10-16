#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""tbe 编译栈：包装 tbe python 包的平台信息 / 编译上下文 / tiling 接口。"""

__all__ = ["TbeCompiler"]


# Standard Packages
import logging

# Third-Party Packages
from .compiler import Compiler, register_compiler


@register_compiler("tbe")
class TbeCompiler(Compiler):
    """
    Class Interface for via `tbe` module
    """

    def __init__(self):
        self._initialized = False
        self._import_error = None
        try:
            self._tbe = __import__("tbe")
            logging.debug(f"Using tbe module from {self._tbe.__file__}")
        except ModuleNotFoundError as e:
            self._import_error = e
            if e.name == "tbe":
                logging.info(f"Import `tbe` failed. Maybe it has been removed: {e}")
            else:
                logging.warning(
                    f"Import `tbe` failed: tbe's dependency `{e.name}` is missing. "
                    f"Install it with `pip3 install {e.name}` and retry."
                )
        else:
            self._initialized = True

    def __getattr__(self, name):
        if name == "_tbe":
            err = self.__dict__.get("_import_error")
            hint = (
                f" ({err}). Install the missing dependency and retry"
                if err is not None and err.name != "tbe"
                else ". Check that CANN is installed and its environment is sourced (set_env.sh)"
            )
            raise RuntimeError(
                f"`tbe` compiler stack is unavailable: `import tbe` failed{hint}. "
                "Kernel compilation requires the CANN `tbe` python package."
            ) from err
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def is_initialized(self) -> bool:
        return self._initialized

    def set_platform_info_res(self, device_id, res: dict):
        self._tbe.common.platform.platform_info.set_platform_info_res(0, res)

    def get_soc_spec(self, key: str):
        return self._tbe.common.platform.platform_info.get_soc_spec(key)

    def get_context(self):
        return self._tbe.dsl.base.operation.get_context()

    def get_computes(self):
        return self.get_context().get_computes()

    def get_tiling_op_type(self):
        return self.get_context().get_op_type()

    def get_compile_info(self):
        return self._tbe.dsl.base.operation.get_compile_info()

    def set_compile_soc_info(self, soc_version, core_type="AiCore"):
        self._tbe.common.platform.platform_info.set_current_compile_soc_info(soc_version, core_type)

    def get_param_generalization(self, op_type: str):
        return self._tbe.common.register.get_param_generalization(op_type)

    def do_op_tiling(self, *args, **kwargs):
        return self._tbe.common.utils.op_tiling.do_op_tiling(*args, **kwargs)

    def build_config(self, **kwargs):
        return self._tbe.common.buildcfg.build_config(**kwargs)

    @property
    def op_context(self):
        return self._tbe.common.context.op_context

    @property
    def op_info(self):
        return self._tbe.common.context.op_info

    @property
    def api_config(self):
        return self._tbe.tvm.api_config
