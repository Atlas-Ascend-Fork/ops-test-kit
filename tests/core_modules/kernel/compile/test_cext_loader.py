# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""cext_loader 测试：_find_so 搜索路径优先级、_build_cext flock+marker 并发保护、load_cext 错误处理。

不依赖 CANN 环境，通过 mock subprocess.check_call 模拟 cmake 编译。
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import pytest


def _import_cext_loader():
    """重新 import cext_loader（清缓存确保拿到全新模块实例）。"""
    import importlib
    import sys

    mod_name = "ttk.core_modules.kernel.compile.cext_loader"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


class TestFindSo(unittest.TestCase):
    """_find_so: 搜索 .so 文件，whl 优先于源码 build 目录。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_finds_in_whl_lib_dir(self):
        """whl 安装的 .so 在 ttk/lib 目录下能被找到。"""
        loader = _import_cext_loader()
        lib_dir = os.path.join(self.tmpdir, "ttk", "lib")
        os.makedirs(lib_dir, exist_ok=True)
        so_path = os.path.join(lib_dir, "libfake.so")
        with open(so_path, "w") as f:
            f.write("fake")

        assert loader._find_so(self.tmpdir, "libfake.so", "unused") == so_path

    def test_returns_none_when_missing(self):
        """.so 不存在 → 返回 None。"""
        loader = _import_cext_loader()
        assert loader._find_so(self.tmpdir, "libnonexistent.so", "nope") is None


class TestBuildCextMarkerAndFlock(unittest.TestCase):
    """_build_cext: flock + marker 并发保护逻辑。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.build_dir = os.path.join(self.tmpdir, "build")
        os.makedirs(self.build_dir, exist_ok=True)
        self.so_name = "libttk_op_registry_accessor.so"
        self.so_path = os.path.join(self.build_dir, self.so_name)
        self.marker_path = os.path.join(self.build_dir, f".build.{self.so_name}.done")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _fake_build_produces_so(self, *args, **kwargs):
        """模拟 cmake 生成 .so 文件。"""
        with open(self.so_path, "w") as f:
            f.write("fake_so")

    @patch("subprocess.check_call")
    def test_build_fails_no_marker(self, mock_check_call):
        """编译失败（CalledProcessError）→ 不写 marker。"""
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "cmake")

        loader = _import_cext_loader()
        with pytest.raises(subprocess.CalledProcessError):
            loader._build_cext(self.tmpdir, self.so_name)

        assert not os.path.isfile(self.marker_path)

    @patch("subprocess.check_call")
    def test_concurrent_calls_only_one_builds(self, mock_check_call):
        """5 线程并发 _build_cext → cmake 只执行 1 次（flock 保护）。"""
        build_count = 0
        lock = threading.Lock()

        def fake_build(*args, **kwargs):
            nonlocal build_count
            with lock:
                build_count += 1
            time.sleep(0.05)  # 模拟编译耗时，增加并发窗口
            self._fake_build_produces_so()

        mock_check_call.side_effect = fake_build

        loader = _import_cext_loader()
        errors = []

        def worker():
            try:
                loader._build_cext(self.tmpdir, self.so_name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Worker errors: {errors}"
        assert build_count == 2, "只应有 2 次 cmake 调用（configure + build）"
        assert os.path.isfile(self.marker_path)


class TestLoadCext(unittest.TestCase):
    """load_cext: .so 找不到且无源码时的错误处理。"""

    def test_raises_file_not_found_when_so_missing(self):
        """load_cext 在 .so 不存在且无源码目录时抛 FileNotFoundError。"""
        loader = _import_cext_loader()
        with pytest.raises(FileNotFoundError):
            loader.load_cext("libnonexistent.so", "nonexistent_dir")
