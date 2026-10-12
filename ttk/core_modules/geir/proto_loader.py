#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os
import re
from typing import Dict, List, Optional

from ttk.core_modules.common.plugin.spec_loader import snake_to_pascal
from ttk.utilities import Singleton


class OpProtoInfo:
    __slots__ = ("op_class", "proto_file", "inputs", "outputs", "dynamic_inputs", "dynamic_outputs", "attrs")

    def __init__(
        self,
        op_class: str,
        proto_file: str,
        inputs: List[str],
        outputs: List[str],
        dynamic_inputs: Optional[List[str]] = None,
        dynamic_outputs: Optional[List[str]] = None,
        attrs: Optional[List[tuple]] = None,
    ):
        self.op_class = op_class
        self.proto_file = proto_file
        self.inputs = inputs
        self.outputs = outputs
        self.dynamic_inputs = dynamic_inputs or []
        self.dynamic_outputs = dynamic_outputs or []
        self.attrs = attrs or []


class ProtoLoader(metaclass=Singleton):
    _REG_OP_PATTERN = re.compile(
        r"REG_OP\s*\((\w+)\)"
        r"(.*?)"
        r"\.OP_END_FACTORY_REG\s*\(\1\)",
        re.DOTALL,
    )
    # 按声明位置取输入序：少数算子(如 DropOutV3)的 OPTIONAL_INPUT 夹在必选中间，
    # 分两次 findall 拼接会假定"可选总在末尾"，导致输入错位。
    _ANY_INPUT_PATTERN = re.compile(r"\.(?:OPTIONAL_INPUT|DYNAMIC_INPUT|INPUT)\(\s*(\w+),")
    _DYNAMIC_INPUT_PATTERN = re.compile(r"\.DYNAMIC_INPUT\(\s*(\w+),")
    _ANY_OUTPUT_PATTERN = re.compile(r"\.(?:OUTPUT|DYNAMIC_OUTPUT)\(\s*(\w+),")
    _DYNAMIC_OUTPUT_PATTERN = re.compile(r"\.DYNAMIC_OUTPUT\(\s*(\w+),")
    _ATTR_PATTERN = re.compile(r"\.ATTR\(\s*(\w+)\s*,\s*(\w+)")
    _REQUIRED_ATTR_PATTERN = re.compile(r"\.REQUIRED_ATTR\(\s*(\w+)\s*,\s*(\w+)")

    def __init__(self, ascend_path=None):
        from ttk._env import _find_ascend_root
        from ttk.core_modules.common.device import get_opp_paths

        root = ascend_path or _find_ascend_root() or ""

        # Built-in proto dir: merged ops_proto_*.h headers
        builtin_proto_dir = os.path.join(root, "opp", "built-in", "op_graph", "inc")

        # Collect proto dirs in priority order: custom > vendor > built-in.
        # Custom/vendor packages store individual *_proto.h files in op_proto/inc.
        proto_dirs: List[tuple] = []
        for copp in get_opp_paths("custom"):
            d = os.path.join(copp, "op_proto", "inc")
            if os.path.isdir(d):
                proto_dirs.append(("custom", d))
        for vopp in get_opp_paths("vendor"):
            d = os.path.join(vopp, "op_proto", "inc")
            if os.path.isdir(d):
                proto_dirs.append(("vendor", d))
        if os.path.isdir(builtin_proto_dir):
            proto_dirs.append(("builtin", builtin_proto_dir))

        self._proto_dirs = proto_dirs
        self._cache: Dict[str, OpProtoInfo] = {}

    def get_op_info(self, op_name: str) -> Optional[OpProtoInfo]:
        if op_name in self._cache:
            return self._cache[op_name]

        # GE REG_OP uses PascalCase; try pascal first, then original
        candidates = []
        pascal = snake_to_pascal(op_name)
        if pascal != op_name:
            candidates.append(pascal)
        candidates.append(op_name)

        for name in candidates:
            info = self._scan_proto_files(name)
            if info:
                self._cache[op_name] = info
                return info

        searched = ", ".join(d for _, d in self._proto_dirs) or "<no proto dir found>"
        logging.warning(
            f"Operator '{op_name}': no REG_OP declaration found in any op IR header. "
            f"Searched (candidate names {candidates}): {searched}"
        )
        return None

    def _scan_proto_files(self, op_name: str) -> Optional[OpProtoInfo]:
        import glob

        for category, proto_dir in self._proto_dirs:
            pattern = "ops_proto_*.h" if category == "builtin" else "*_proto.h"
            for proto_path in sorted(glob.glob(os.path.join(proto_dir, pattern))):
                if "ops_proto_legacy.h" in proto_path:
                    continue
                try:
                    with open(proto_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    info = self._parse_reg_op(content, os.path.basename(proto_path), op_name)
                    if info:
                        return info
                except Exception:  # noqa: S112 proto 头逐个探测，坏文件跳过属预期
                    continue
        return None

    def _parse_reg_op(self, content: str, proto_file: str, op_name: str) -> Optional[OpProtoInfo]:
        pattern = re.compile(
            r"REG_OP\s*\(" + re.escape(op_name) + r"\)"
            r"(.*?)"
            r"\.OP_END_FACTORY_REG\s*\(" + re.escape(op_name) + r"\)",
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(content)
        if not m:
            return None

        body = m.group(1)

        # Extract canonical PascalCase name from REG_OP (RoiAlign→ROIAlign)
        canonical = op_name
        name_m = self._REG_OP_PATTERN.search(content, m.start())
        if name_m and name_m.group(1).lower() == op_name.lower():
            canonical = name_m.group(1)

        inputs = self._ANY_INPUT_PATTERN.findall(body)
        dynamic_inputs = self._DYNAMIC_INPUT_PATTERN.findall(body)
        outputs = self._ANY_OUTPUT_PATTERN.findall(body)
        dynamic_outputs = self._DYNAMIC_OUTPUT_PATTERN.findall(body)
        attrs = self._ATTR_PATTERN.findall(body)
        attrs += self._REQUIRED_ATTR_PATTERN.findall(body)

        return OpProtoInfo(canonical, proto_file, inputs, outputs, dynamic_inputs, dynamic_outputs, attrs)
