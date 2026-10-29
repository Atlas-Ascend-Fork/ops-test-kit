#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""
FrameworkApiInfoKeeper — cached API parameter info for torch/torch_npu/tf.

参数解析经 adapter 分发（framework_api/adapter.py）：torch → torch/param_extractor、
tf → tf/param_extractor；框架路由单源于 detect_framework，keeper 不再持有前缀分支。
manual 登记走 torch/param_extractor 的 _MANUAL_OVERRIDES 注册表（对任意 api_name
字符串生效——与既有行为一致）。Validates testcase parameters against API signatures.
"""

import logging
from typing import Dict, Optional

from ttk.utilities import Singleton

from .api_model import APIParamInfo


class FrameworkApiInfoKeeper(metaclass=Singleton):
    def __init__(self):
        self._cache: Dict[str, Optional[APIParamInfo]] = {}
        self._failure_reasons: Dict[str, str] = {}

    def get(self, api_name: str) -> Optional[APIParamInfo]:
        if api_name in self._cache:
            return self._cache[api_name]
        reason = None
        try:
            from .adapter import adapter_for_api_name

            info = adapter_for_api_name(api_name).extract_params(api_name)
        except ModuleNotFoundError as e:
            # 可选框架（torch_npu/tensorflow 等）未安装属预期环境形态：
            # 单行告警即可，不刷 ERROR 级整栈回溯污染日志。
            logging.warning("API signature parsing failed: %s: %s: %s", api_name, type(e).__name__, e)
            reason = f"{type(e).__name__}: {e}"
            info = None
        except Exception as e:
            logging.exception("API signature parsing failed: %s: %s: %s", api_name, type(e).__name__, e)
            reason = f"{type(e).__name__}: {e}"
            info = None
        self._cache[api_name] = info
        if info:
            logging.debug(f"Parsed {api_name}: {len(info.params)} params from {info.source}")
        elif info is None:
            self._failure_reasons[api_name] = reason or self._inspect_failure_reason(api_name)
            logging.debug("Could not parse %s; see API signature parsing errors above", api_name)
        return info

    @staticmethod
    def _inspect_failure_reason(api_name):
        from .adapter import adapter_for_api_name

        try:
            adapter_for_api_name(api_name).load_api(api_name)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return "API resolved, but parameter inspection did not produce a supported signature"

    def get_failure_reason(self, api_name: str) -> Optional[str]:
        """Read a cached parse failure without loading or inspecting the API."""
        return self._failure_reasons.get(api_name)

    def register(self, api_name: str, params, source="manual"):
        if isinstance(params, APIParamInfo):
            self._cache[api_name] = params
        elif isinstance(params, list) and params and isinstance(params[0], list):
            info = APIParamInfo(api_name=api_name, overloads=params, source=source)
            self._cache[api_name] = info
        else:
            # 手工签名登记属 torch 域（_MANUAL_OVERRIDES 注册表），经 adapter 钩子
            # 触达——共享层不直连框架子包模块（与 get 的分发口径一致）。
            from .adapter import get_adapter

            self._cache[api_name] = get_adapter("torch").register_params(api_name, params, source)
        self._failure_reasons.pop(api_name, None)

    def clear_cache(self):
        self._cache.clear()
        self._failure_reasons.clear()
