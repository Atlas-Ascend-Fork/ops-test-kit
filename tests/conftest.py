# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Shared pytest fixtures for the test suite."""

# 必须最先执行: 用 RTLD_DEEPBIND 把 tensorflow 装进来, 之后 torch/torch_npu 再导入
# 就不会与它抢弱符号。对照实验(本机 torch 2.10 + torch_npu + tf 2.21):
#   A) import torch_npu -> import tensorflow            → dumped core
#   B) import 本 shim   -> import torch_npu/tensorflow  → 两者都可用
# 不做这一步, 任何在同一进程里同时碰到两个框架的用例都只能 skip 掉。
from pathlib import Path

import pytest

import ttk.remote.server._framework_preload  # noqa: F401  仅为触发受控预载


@pytest.fixture(autouse=True)
def isolate_ttk_environment(monkeypatch):
    """Ensure tests do not accidentally pick up real device or config."""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
    monkeypatch.delenv("ASCEND_ROOT", raising=False)
    # 用户级 ~/.config/ttk/ttk.conf.yaml 会被 loader 读入(真实机器上常配着远端
    # endpoints),使断言"未配 endpoints"的测试随环境红。只保留 cwd 相对路径,
    # 测试用 tmp_path + chdir 自己铺配置。
    import ttk.config.loader as _loader

    monkeypatch.setattr(_loader, "_CONFIG_PATHS", [Path("ttk.conf.yaml")])


@pytest.fixture
def make_testcase():
    """Factory: TestcaseAclnn instances (shared global; test_testcase_e2e.py
    overrides locally with TestcaseE2e)."""

    def _make(api_name="aclnnDummy", **kwargs):
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn

        case = TestcaseAclnn()
        case.api_name = api_name
        case.is_valid = True
        case.fail_reason = None
        case.attributes = kwargs.pop("attributes", {})
        for k, v in kwargs.items():
            setattr(case, k, v)
        return case

    return _make


@pytest.fixture(scope="session", autouse=True)
def _load_default_config():
    """所有测试前加载默认 config 一次（替代被删的 get_config lazy fallback）。

    删 lazy fallback 后 get_config() 不再自动加载；这个 session fixture 保证
    _config 已加载默认配置链（default.yaml + ~/.config/ttk + ./ttk.conf.yaml），
    测试可直接 get_config()。需要自定义 config 的测试在测试体内显式
    load_config(yaml)，测完 load_config() 恢复默认。
    """
    from ttk.config.loader import load_config

    load_config()
