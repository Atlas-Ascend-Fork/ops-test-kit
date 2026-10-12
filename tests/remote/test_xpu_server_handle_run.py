# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_handle_run 执行隔离 + provider 归一化 单测。

测试 _build_device_opts 纯函数（只返回 device 三件：device_id/docker_args/env，
不构造完整 kwargs——避免 BaseHTTPRequestHandler mock）+ get_framework provider 归一化。
"""

from types import SimpleNamespace

from ttk.remote.server.xpu_server import _build_device_opts


def _h(**ov):
    """造一个类 handler 的 SimpleNamespace（_build_device_opts 只读 3 属性）。"""
    d = {
        "use_device": True,
        "sandbox": "none",
        "profile": {"torch_lib": "cuda", "torch_profiler": {"activities": ["CPU"]}},
    }
    d.update(ov)
    return SimpleNamespace(**d)


# ---- _build_device_opts（执行隔离）----


def test_device_id_zero():
    """非cpu 分支 device_id 固定 0（容器内域，executor 见 cuda:0）。"""
    assert _build_device_opts(_h(), n=0)["device_id"] == 0


def test_docker_fail_fast():
    """sandbox=docker 但 profile 缺 docker_args → fail-fast（http_status=500，render 前检查）。"""
    h = _h(sandbox="docker", profile={"torch_lib": "cuda", "torch_profiler": {"activities": ["CPU"]}})
    opts = _build_device_opts(h, n=0)
    assert opts.get("ok") is False
    assert opts.get("http_status") == 500


def test_cpu_device_id():
    """cpu 分支 n 被忽略（None），device_id="cpu"（device_id 双语义）。"""
    h = _h(use_device=False)
    assert _build_device_opts(h, n=None)["device_id"] == "cpu"
