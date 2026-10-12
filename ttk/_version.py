#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import subprocess

__version__ = "3.0.0"


def get_build():
    try:
        from ttk._build_hash import build

        return build
    except ImportError:
        pass
    try:
        ttk_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ttk_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _get_git_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


try:
    from setuptools.command.build_py import build_py as _build_py

    class BuildPyWithBuild(_build_py):
        def run(self):
            build_hash = _get_git_hash()
            if build_hash:
                target = os.path.join(self.get_package_dir("ttk"), "_build_hash.py")
                with open(target, "w") as f:
                    f.write(f'build = "{build_hash}"\n')
            super().run()
except ImportError:
    pass
