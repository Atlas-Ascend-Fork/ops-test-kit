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
Plugin scanner using AST to parse plugin declarations without importing modules
"""

import ast
import logging
from pathlib import Path
from typing import Dict, Tuple


class PluginScanner:
    """Scan plugin declarations using AST without importing modules"""

    def __init__(self, scan_dir, plugin_type: str, field_name: str, level_type: str):
        """
        Args:
            scan_dir: Directory or file to scan (str, Path, or tuple of them)
            plugin_type: "golden" or "input"
            field_name: "__golden__" or "__input__"
            level_type: "kernel" / "aclnn" / "e2e" — only registrations under
                this level key are extracted (per-level gating)
        """
        if isinstance(scan_dir, (str, Path)):
            scan_dir = (scan_dir,)
        self.scan_dir = scan_dir
        self.plugin_type = plugin_type
        self.field_name = field_name
        self.level_type = level_type

    def scan(self) -> Dict[str, Tuple[str, str, Path]]:
        """
        Scan directory and return {operator_name: (module_name, function_name, file_path)}
        Uses AST parsing without importing modules. Only the declarations
        registered under self.level_type are returned.
        """
        results = {}

        for raw_f in self.scan_dir:
            f = raw_f if isinstance(raw_f, Path) else Path(raw_f)
            if not f.exists():
                logging.debug(f"Scan directory or file not found: {f}")
                continue
            if f.is_file():
                if f.name.endswith(".py"):
                    module_info = self._scan_module(f)
                    if module_info:
                        results.update(module_info)
            else:
                for py_file in f.rglob("*.py"):
                    if py_file.name.startswith("_") or py_file.name == "__init__.py":
                        continue

                    module_info = self._scan_module(py_file)
                    if module_info:
                        results.update(module_info)

        return results

    def _scan_module(self, py_file: Path) -> Dict[str, Tuple[str, str, Path]]:
        """Parse module using AST"""
        try:
            module_name = py_file.stem

            with open(py_file, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=str(py_file))
            plugin_decl = self._find_plugin_declaration(tree)

            if not plugin_decl:
                return {}

            operators = self._parse_operators_dict(plugin_decl)

            if not operators:
                return {}

            valid_operators = {}
            for op_name, func_name in operators.items():
                valid_operators[op_name] = (module_name, func_name, py_file)
                logging.debug(f"Found {self.plugin_type}: {op_name} -> {module_name}.{func_name}")

            return valid_operators

        except SyntaxError as e:
            logging.error(f"Syntax error in {py_file}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Error scanning {py_file}: {e}")
            return {}

    def _find_plugin_declaration(self, tree: ast.AST) -> ast.Dict:
        """Find __golden__ or __input__ declaration.

        普通赋值检查全部目标（链式赋值 alias = __golden__ = {...} 的非首
        目标同样合法）；带类型注解的赋值（__golden__: dict = {...}）为
        AnnAssign 单目标。仅注解未赋值（无 value）与非字典值仍忽略。
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id == self.field_name and isinstance(node.value, ast.Dict):
                    return node.value
        return None

    def _parse_operators_dict(self, dict_node: ast.Dict) -> Dict[str, str]:
        """Parse the sub-dict registered under self.level_type"""
        operators = {}

        for key, value in zip(dict_node.keys, dict_node.values):
            if isinstance(key, ast.Constant):
                key_value = key.value
                if key_value == self.level_type and isinstance(value, ast.Dict):
                    for op_key, op_value in zip(value.keys, value.values):
                        if isinstance(op_key, ast.Constant) and isinstance(op_value, ast.Constant):
                            operators[op_key.value] = op_value.value

        return operators
