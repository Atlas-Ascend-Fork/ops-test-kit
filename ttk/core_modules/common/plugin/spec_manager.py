#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Optional, Set

from ....utilities import Singleton
from .spec_loader import SpecLoader
from .spec_validator import validate as _validate


class TestSpecManager(metaclass=Singleton):
    """Operator test spec manager — load/inspect/validate spec classes.

    Usage:
        mgr = TestSpecManager(search_paths=("/path/to/specs",))
        cls = mgr.load("softmax_v2")
        golden_fn = getattr(cls, "golden", None)
    """

    def __init__(self, search_paths):
        self._loader = SpecLoader(search_paths)
        self._validated: Set[int] = set()  # spec classes already validated (by id)

    # -- Loading --

    def load(self, op_name: str) -> Optional[type]:
        """Load spec class. __spec__ dict preferred, naming convention fallback. Cached.
        First load of a class triggers validate (raises InvalidSpecError on mismatch)."""
        cls = self._loader.load(op_name)
        if cls is not None and id(cls) not in self._validated:
            _validate(cls)  # raises before marking validated → retry re-validates
            self._validated.add(id(cls))
        return cls

    # -- Validation --

    def validate(self, cls: type) -> None:
        """Type check. Raises InvalidSpecError on mismatch."""
        _validate(cls)

    # -- Cache management --

    def clear_cache(self):
        """Clear cache (for testing or hot reload)."""
        self._loader.clear_cache()
        self._validated.clear()


def _load_spec_class(op_name: str, plugin_path):
    """按算子名加载 spec 类；无 plugin_path / 无 spec → None。路径归一为 tuple。"""
    if not plugin_path:
        return None
    paths = tuple(plugin_path) if isinstance(plugin_path, (list, tuple)) else (plugin_path,)
    return TestSpecManager(search_paths=paths).load(op_name)


def get_spec_attr(op_name: str, attr: str, plugin_path) -> Any:
    """Get an attribute from a TestSpec class by operator name.

    Args:
        op_name: operator name (e.g. "softmax_v2")
        attr: attribute name (e.g. "pre_compare", "compare", "tolerance")
        plugin_path: search paths (str/path/tuple/list) or None

    Returns:
        Attribute value (callable, dict, str, type, ...) or None if not found.
    """
    cls = _load_spec_class(op_name, plugin_path)
    return getattr(cls, attr, None) if cls is not None else None


def get_spec_class_meta(op_name: str, plugin_path):
    """返回 spec 类元数据（供 XPU/远端 sync+实例化）。无 spec 或无 plugin_path → None。

    Returns:
        {"spec_file": <abs .py path or None>, "class_name": <cls.__name__>} or None
    """
    cls = _load_spec_class(op_name, plugin_path)
    if cls is None:
        return None
    return {"spec_file": getattr(cls, "__ttk_spec_file__", None), "class_name": cls.__name__}
