#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import contextlib
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import InvalidSpecError


def snake_to_pascal(name: str) -> str:
    """snake_case -> PascalCase: 'softmax_v2' -> 'SoftmaxV2'

    Limitation: does not handle already-PascalCase names (e.g. 'BatchMatMul'
    would become 'Batchmatmul'). Use __spec__ dict registration for
    those cases.
    """
    return "".join(part.capitalize() for part in name.split("_"))


class SpecLoader:
    """Spec file discovery, import and cache.

    Discovery uses AST static scan to build __spec_index (0 exec on scan).
    Load then exec's lazily: __spec__-indexed ops exec exactly one file (the
    indexed source); naming-convention fallback exec's candidate files in
    search order until the class is found (each file exec'd at most once,
    cached per-file). __spec__ dict values must be string class names.

    Discovery priority: __spec__ dict > naming convention
    Multiple search paths: first match wins (same as sys.path)
    """

    def __init__(self, search_paths: List[str]):
        self._search_paths = [Path(p).resolve() for p in search_paths]
        self._cache: Dict[str, Optional[type]] = {}
        self._loaded_modules: Dict[Path, object] = {}  # exec cache (Path key)
        self.__spec_index: Dict[str, Tuple[Path, str]] = {}  # op -> (file, classname)
        self._index_built = False

    def load(self, op_name: str) -> Optional[type]:
        """Load spec class. Returns class or None."""

        # 1. Cache hit
        if op_name in self._cache:
            return self._cache[op_name]

        # 2. Build index lazily (once), then resolve via __spec__ or convention
        self._ensure_index_built()
        if op_name in self.__spec_index:
            py_file, cls_name = self.__spec_index[op_name]
            cls, src_file = self._load_class_from(py_file, cls_name)
        else:
            cls_name = snake_to_pascal(op_name) + "TestSpec"
            cls, src_file = self._find_by_convention(cls_name)

        # _load_class_from / _find_by_convention 均保证返回 type 或 None
        if cls is not None:
            self._mark_source(cls, src_file)
            self._cache[op_name] = cls
            return cls

        # 3. Not found
        self._cache[op_name] = None
        return None

    def _ensure_index_built(self):
        """Build __spec_index on first load (memoized)."""
        if self._index_built:
            return
        for py_file in self._iter_py_files():
            self._index_file(py_file)
        self._index_built = True

    def _index_file(self, py_file: Path):
        """Statically scan module-level __spec__ assignments in one file."""
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError:
            return  # skip unreadable files
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            # 只对 spec 文件(含 __spec__ 文本)报语法错; 普通py静默(不关 loader 事)
            if "__spec__" in source:
                logging.error("spec loader: spec file %s syntax error: %s", py_file, e)
            return
        file_spec: Dict[str, str] = {}  # this file's __spec__, last-wins
        for node in tree.body:  # module-level only, NOT ast.walk
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                target = node.target
            else:
                continue
            if target.id != "__spec__" or not isinstance(node.value, ast.Dict):
                continue
            for key, val in zip(node.value.keys, node.value.values):
                if key is None:  # dict spread {**x}
                    continue
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
                    raise InvalidSpecError(
                        f"only string class names accepted in __spec__, "
                        f"got {type(val).__name__} for op {key.value!r} in {py_file}"
                    )
                file_spec[key.value] = val.value  # last-wins within file
        for op, cls_name in file_spec.items():
            if op not in self.__spec_index:  # first-wins across files
                self.__spec_index[op] = (py_file, cls_name)

    def _load_class_from(self, py_file: Path, cls_name: str):
        """Exec (cached per-file) and pull a named class out.

        Uses the per-file module cache directly so a file already imported
        for another op is NOT re-executed (and _import_file is only invoked
        on a genuine cache miss — it is the single exec entry point).
        """
        module = self._loaded_modules.get(py_file)
        if module is None:
            module = self._import_file(py_file)  # exec + cache
        cls = getattr(module, cls_name, None) if module else None
        return (cls, py_file) if (cls is not None and isinstance(cls, type)) else (None, py_file)

    def _find_by_convention(self, cls_name: str):
        """Naming-convention fallback: scan files for a class of this name."""
        for py_file in self._iter_py_files():
            module = self._import_file(py_file)  # per-file cache
            cls = getattr(module, cls_name, None) if module else None
            if cls is not None and isinstance(cls, type):
                return cls, py_file
        return None, None

    def _iter_py_files(self):
        """Iterate all .py files in search path order (exclude _-prefixed hidden files).

        Each search path can be a directory (rglob *.py) or a single .py file
        (loaded directly, _-prefix filter not applied since user chose it explicitly).
        """
        seen = set()
        for search_path in self._search_paths:
            if search_path.is_file():
                resolved = search_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved
            elif search_path.is_dir():
                for py_file in search_path.rglob("*.py"):
                    name = py_file.name
                    if name.startswith("_") and name != "__init__.py":
                        continue
                    resolved = py_file.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved

    @staticmethod
    def _mark_source(spec_cls: type, py_file: Path) -> None:
        """Record the source file on the class so remote spec-mode can locate it."""
        with contextlib.suppress(AttributeError, TypeError):
            spec_cls.__ttk_spec_file__ = str(py_file)

    def _import_file(self, py_file: Path) -> Optional[object]:
        """Dynamically import .py file as module object.
        Does NOT register in sys.modules — spec modules are ephemeral.
        Results cached per-file in _loaded_modules."""
        if py_file in self._loaded_modules:
            return self._loaded_modules[py_file]
        try:
            spec = importlib.util.spec_from_file_location(
                f"ttk_test_spec_{py_file.stem}_{hash(str(py_file))}",
                str(py_file),
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            # Do NOT add to sys.modules — avoids memory leak in long sessions
            spec.loader.exec_module(module)
            self._loaded_modules[py_file] = module
            return module
        except Exception as e:
            # 只对 spec 文件(含 __spec__ 文本)报; 普通py静默
            try:
                if "__spec__" in py_file.read_text(encoding="utf-8"):
                    logging.error("spec loader: spec file %s load failed: %s", py_file, e)
            except OSError:
                pass
            return None

    def clear_cache(self):
        """Clear cache (for testing or hot reload)"""
        self._cache.clear()
        self._loaded_modules.clear()
        self.__spec_index.clear()
        self._index_built = False
