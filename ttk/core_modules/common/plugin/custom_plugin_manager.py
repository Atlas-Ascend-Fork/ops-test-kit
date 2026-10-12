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
旧插件系统管理器：__golden__/__input__ 约定注册的算子函数的扫描与加载。

AST 扫描不执行文件（custom_plugin_scanner），命中后按 (module, func, file)
惰性 exec 并缓存。实例按 plugin_path 键控单例（utilities.Singleton）。
"""

import importlib.util
import logging
from typing import Callable, List, Optional

from ....utilities import Singleton
from .custom_plugin_scanner import PluginScanner


class CustomPluginManager(metaclass=Singleton):
    """Custom plugin manager（旧 __golden__/__input__ 约定的兼容层）"""

    PLUGIN_TYPE_GOLDEN = "golden"
    PLUGIN_TYPE_INPUTS = "input"
    LEVEL_KERNEL = "kernel"
    LEVEL_ACLNN = "aclnn"
    LEVEL_E2E = "e2e"
    FIELD_GOLDEN = "__golden__"
    FIELD_INPUTS = "__input__"

    PLUGIN_CONFIG = {
        PLUGIN_TYPE_GOLDEN: {
            "levels": [LEVEL_KERNEL, LEVEL_ACLNN, LEVEL_E2E],
            "field_name": FIELD_GOLDEN,
        },
        PLUGIN_TYPE_INPUTS: {
            "levels": [LEVEL_KERNEL, LEVEL_ACLNN, LEVEL_E2E],
            "field_name": FIELD_INPUTS,
        },
    }

    def __init__(self, plugin_path=None):
        self.plugin_path = plugin_path

        self._scan_cache = {
            self.PLUGIN_TYPE_GOLDEN: {self.LEVEL_KERNEL: None, self.LEVEL_ACLNN: None, self.LEVEL_E2E: None},
            self.PLUGIN_TYPE_INPUTS: {self.LEVEL_KERNEL: None, self.LEVEL_ACLNN: None, self.LEVEL_E2E: None},
        }

        self._loaded_funcs = {
            self.PLUGIN_TYPE_GOLDEN: {self.LEVEL_KERNEL: {}, self.LEVEL_ACLNN: {}, self.LEVEL_E2E: {}},
            self.PLUGIN_TYPE_INPUTS: {self.LEVEL_KERNEL: {}, self.LEVEL_ACLNN: {}, self.LEVEL_E2E: {}},
        }

        logging.info(f"CustomPluginManager initialized with plugin_path: {plugin_path}")

    @classmethod
    def supports(cls, plugin_type: str) -> bool:
        """旧系统是否服务该 plugin_type（仅 golden/input 有约定）。"""
        return plugin_type in cls.PLUGIN_CONFIG

    def _get_config(self, plugin_type: str) -> dict:
        """Get plugin type configuration"""
        if plugin_type not in self.PLUGIN_CONFIG:
            raise ValueError(f"Unsupported plugin type: {plugin_type}")
        return self.PLUGIN_CONFIG[plugin_type]

    def _get_field_name(self, plugin_type: str) -> str:
        """Get field name"""
        return self._get_config(plugin_type)["field_name"]

    def _get_supported_levels(self, plugin_type: str) -> List[str]:
        """Get supported level types"""
        return self._get_config(plugin_type)["levels"]

    def _is_valid_level(self, plugin_type: str, level_type: str) -> bool:
        """Validate level type"""
        return level_type in self._get_supported_levels(plugin_type)

    def _ensure_scanned(self, plugin_type: str, level_type: str):
        """Ensure the (plugin_type, level_type) scan has run (cached in _scan_cache)."""
        if not self._is_valid_level(plugin_type, level_type):
            raise ValueError(f"Invalid level type {level_type} for plugin type {plugin_type}")

        if self._scan_cache[plugin_type][level_type] is not None:
            return

        if not self.plugin_path:
            self._scan_cache[plugin_type][level_type] = {}
            logging.info(f"No plugin_path, custom {plugin_type} cache empty")
            return

        logging.info(f"Scanning custom {plugin_type} for type: {level_type}")

        field_name = self._get_field_name(plugin_type)
        scanner = PluginScanner(self.plugin_path, plugin_type, field_name, level_type)
        self._scan_cache[plugin_type][level_type] = scanner.scan()

        logging.info(
            f"Scanned {len(self._scan_cache[plugin_type][level_type])} custom {plugin_type}.{level_type} functions"
        )

    def load_if_available(self, operator_name: str, plugin_type: str, level_type: str) -> Optional[Callable]:
        """Return the registered function for operator_name, or None if absent/unloadable."""
        if not self._is_valid_level(plugin_type, level_type):
            raise ValueError(f"Invalid level type {level_type} for plugin type {plugin_type}")

        if operator_name in self._loaded_funcs[plugin_type][level_type]:
            return self._loaded_funcs[plugin_type][level_type][operator_name]

        self._ensure_scanned(plugin_type, level_type)

        scan_results = self._scan_cache[plugin_type][level_type]

        if operator_name in scan_results:
            module_name, func_name, file_path = scan_results[operator_name]

            try:
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                func = getattr(module, func_name)
                self._loaded_funcs[plugin_type][level_type][operator_name] = func
                logging.info(f"Loaded custom {plugin_type}: {level_type}.{operator_name} from {file_path}")
                return func

            except Exception as e:
                logging.error(f"Failed to load custom {plugin_type} {operator_name}: {e}")

        return None
