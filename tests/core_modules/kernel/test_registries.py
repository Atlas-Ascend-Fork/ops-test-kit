#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for ttk.core_modules.kernel.operator.registries: SO loading logic.

Validates behavior matches official CANN op_tiling.py:
1. custom/vendor: load op_tiling/liboptiling.so (flat path), TbeLoadSoAndSaveToRegistry, no scene dependency
2. builtin: full if-else chain with lib/{os}/{arch}/ subdirectory, scene via get_ascend_scene_info
3. custom/vendor SO not found: silently skip (OSError caught)
"""

import tempfile
from pathlib import Path
from unittest import mock

import pytest


def _mkdir(p):
    """递归创建目录（已存在则忽略）。"""
    Path(p).mkdir(parents=True, exist_ok=True)


class TestCustomVendorLoading:
    """Custom/vendor 从 op_tiling/liboptiling.so 扁平路径加载，不走 if-else 分支。"""

    @pytest.mark.parametrize(
        "scenario",
        ["flat_path", "no_if_else_branching", "so_not_found"],
    )
    @mock.patch("ctypes.CDLL")
    def test_load_custom_vendor_tiling(self, mock_cdll, scenario):
        """扁平路径加载 / 不加载 libopmaster / SO 缺失静默跳过。"""
        from ttk.core_modules.kernel.operator.registries import _load_custom_vendor_tiling

        with tempfile.TemporaryDirectory() as tmpdir:
            impl = Path(tmpdir) / "op_impl"
            tiling = impl / "ai_core" / "tbe" / "op_tiling"
            _mkdir(tiling)

            if scenario == "so_not_found":
                _load_custom_vendor_tiling(str(impl))
                mock_cdll.assert_not_called()
                return

            (tiling / "liboptiling.so").write_bytes(b"\x00")
            if scenario == "no_if_else_branching":
                (tiling / "libopmaster_rt.so").write_bytes(b"\x00")
                (tiling / "libopmaster_rt2.0.so").write_bytes(b"\x00")

            mock_dll = mock.MagicMock()
            mock_cdll.return_value = mock_dll

            _load_custom_vendor_tiling(str(impl))

            loaded = [c.args[0] for c in mock_cdll.call_args_list]
            assert len(loaded) == 1, f"custom should load exactly 1 SO, got {loaded}"
            assert loaded[0].endswith("op_tiling/liboptiling.so")
            assert "lib/linux" not in loaded[0]
            mock_dll.TbeLoadSoAndSaveToRegistry.assert_called_once()
            if scenario == "no_if_else_branching":
                for p in loaded:
                    assert "libopmaster" not in p


class TestBuiltinLoading:
    """Builtin should use lib/{os}/{arch}/ subdirectory and full if-else chain."""

    @mock.patch("ctypes.CDLL")
    @mock.patch("ttk.core_modules.kernel.operator.registries.get_ascend_scene_info")
    def test_builtin_uses_scene_subdir(self, mock_scene, mock_cdll):
        """Builtin: liboptiling.so is at op_tiling/lib/{os}/{arch}/, not flat."""
        from ttk.core_modules.kernel.operator.registries import _load_builtin_tiling

        mock_scene.return_value = ("linux", "aarch64")

        with tempfile.TemporaryDirectory() as tmpdir:
            impl = Path(tmpdir) / "op_impl"
            subdir = impl / "ai_core" / "tbe" / "op_tiling" / "lib" / "linux" / "aarch64"
            _mkdir(subdir)
            (subdir / "liboptiling.so").write_bytes(b"\x00")

            mock_cdll.return_value = mock.MagicMock()

            _load_builtin_tiling(str(impl))

            loaded = [c.args[0] for c in mock_cdll.call_args_list]
            assert any("lib/linux/aarch64" in p for p in loaded)

    @mock.patch("ctypes.CDLL")
    @mock.patch("ttk.core_modules.kernel.operator.registries.get_ascend_scene_info")
    def test_builtin_liboptiling_no_tbe_load(self, mock_scene, mock_cdll):
        """Builtin liboptiling.so is 1.0 registration — no TbeLoadSoAndSaveToRegistry."""
        from ttk.core_modules.kernel.operator.registries import _load_builtin_tiling

        mock_scene.return_value = ("linux", "aarch64")

        with tempfile.TemporaryDirectory() as tmpdir:
            impl = Path(tmpdir) / "op_impl"
            subdir = impl / "ai_core" / "tbe" / "op_tiling" / "lib" / "linux" / "aarch64"
            _mkdir(subdir)
            (subdir / "liboptiling.so").write_bytes(b"\x00")

            mock_dll = mock.MagicMock()
            mock_cdll.return_value = mock_dll

            _load_builtin_tiling(str(impl))

            mock_dll.TbeLoadSoAndSaveToRegistry.assert_not_called()

    @mock.patch("ctypes.CDLL")
    @mock.patch("ttk.core_modules.kernel.operator.registries.get_ascend_scene_info")
    def test_builtin_op_host_legacy_last(self, mock_scene, mock_cdll):
        """Builtin: op_host legacy SOs loaded after non-legacy."""
        from ttk.core_modules.kernel.operator.registries import _load_builtin_tiling

        mock_scene.return_value = ("linux", "aarch64")

        with tempfile.TemporaryDirectory() as tmpdir:
            impl = Path(tmpdir) / "op_impl"
            host = impl / "ai_core" / "tbe" / "op_host" / "lib" / "linux" / "aarch64"
            _mkdir(host)
            (host / "op_add.so").write_bytes(b"\x00")
            (host / "libophost_legacy.so").write_bytes(b"\x00")

            mock_cdll.return_value = mock.MagicMock()

            _load_builtin_tiling(str(impl))

            loaded = [c.args[0] for c in mock_cdll.call_args_list]
            legacy_idx = next(i for i, p in enumerate(loaded) if "legacy" in p)
            non_legacy_idx = next(i for i, p in enumerate(loaded) if "op_add" in p)
            assert non_legacy_idx < legacy_idx


class TestLoadOpRegistries:
    """Test load_op_registries integration: custom → vendor → builtin order."""

    @mock.patch("ttk.core_modules.kernel.operator.registries._load_builtin_tiling")
    @mock.patch("ttk.core_modules.kernel.operator.registries._load_custom_vendor_tiling")
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("ttk.core_modules.kernel.operator.registries.get_op_impl_paths")
    def test_priority_order(self, mock_paths, mock_isdir, mock_cv, mock_builtin):
        """load_op_registries calls custom, vendor, then builtin loaders in order."""
        from ttk.core_modules.kernel.operator.registries import load_op_registries

        call_log = []

        def _cv_side_effect(p):
            call_log.append(("custom_vendor", p))

        def _builtin_side_effect(p):
            call_log.append(("builtin", p))

        mock_cv.side_effect = _cv_side_effect
        mock_builtin.side_effect = _builtin_side_effect

        def fake_paths(source):
            if source == "custom":
                return ["/custom/op_impl"]
            if source == "vendor":
                return ["/vendor/op_impl"]
            return ["/builtin/op_impl"]

        mock_paths.side_effect = fake_paths

        load_op_registries()

        sources = [c[0] for c in call_log]
        assert sources == ["custom_vendor", "custom_vendor", "builtin"]

    @mock.patch("ttk.core_modules.kernel.operator.registries._load_builtin_tiling")
    @mock.patch("ttk.core_modules.kernel.operator.registries._load_custom_vendor_tiling")
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("ttk.core_modules.kernel.operator.registries.get_op_impl_paths")
    def test_custom_vendor_loaded_without_opp(self, mock_paths, mock_isdir, mock_cv, mock_builtin):
        """custom/vendor should load even if OPP (scene_info) is unavailable."""
        from ttk.core_modules.kernel.operator.registries import load_op_registries

        call_log = []

        def _cv_log(p):
            call_log.append(p)

        mock_cv.side_effect = _cv_log
        mock_builtin.side_effect = RuntimeError("ASCEND_OPP_PATH is not set")

        def fake_paths(source):
            if source == "custom":
                return ["/custom/op_impl"]
            if source == "vendor":
                return ["/vendor/op_impl"]
            return []

        mock_paths.side_effect = fake_paths

        load_op_registries()

        assert len(call_log) == 2
        mock_builtin.assert_not_called()
