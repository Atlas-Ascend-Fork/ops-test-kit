# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Backend ABC + TorchBackend + 三个硬件后端测试。

Task 7: device_name() 返回硬件 MODEL（torch.<lib>.get_device_name）；
alias() 携带段名；soc_version 已移除（合并入 device_name）。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from ttk.core_modules.common.backend.base import Backend
from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend
from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend
from ttk.core_modules.common.backend.npu_torch_backend import NpuTorchBackend
from ttk.core_modules.common.backend.torch_backend import TorchBackend

BACKEND_DIR = Path(importlib.import_module("ttk.core_modules.common.backend").__file__).resolve().parent


def _torch_import_lines(path, top_level_only=False):
    """ast 全量（含函数体内惰性 import）收集 torch/torch_npu 导入行号。

    top_level_only=True 仅扫模块层（tree.body）——用于装配面 __init__.py：
    其函数内的惰性 torch import（_npu_torch_available，torch 分支独占）不违规。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    nodes = tree.body if top_level_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in ("torch", "torch_npu") for alias in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0] if node.level == 0 else ""
            if root in ("torch", "torch_npu"):
                hits.append(node.lineno)
    return hits


def test_backend_abc_has_new_methods():
    """Backend ABC 必须包含 device_name/device_type/has_device/is_npu 方法。"""
    assert all(hasattr(Backend, m) for m in ["device_name", "device_type", "has_device", "is_npu"])


def test_backend_base_is_framework_neutral():
    """契约层框架中立（base.py 零 torch 引用）+ TF 家族零 torch + 装配面顶层零 torch。

    分层契约：Backend 是 torch/tf 共用抽象基类，torch 家族特化（torch_lib 注入、
    torch.<lib>.get_device_name）归 TorchBackend——torch_npu 与 TF C 扩展同进程
    冲突，非 torch 后端不得经任何层传递加载 torch（get_backend 装配面顶层亦然）。
    """
    assert _torch_import_lines(BACKEND_DIR / "base.py") == [], "base.py must not import torch/torch_npu"

    for name in ("tf_backend.py", "npu_tf_backend.py", "cpu_tf_backend.py"):
        hits = _torch_import_lines(BACKEND_DIR / name)
        assert hits == [], f"{name} must not import torch/torch_npu: lines {hits}"

    hits = _torch_import_lines(BACKEND_DIR / "__init__.py", top_level_only=True)
    assert hits == [], f"backend/__init__.py top level must stay torch-free: lines {hits}"


def test_tf_backend_is_npu_only_prefixes():
    """TfBackend.is_npu_only：tf.npu_* 扩展与 npu_device/npu_bridge 自定义算子
    均判 NPU-only（无 CPU kernel，golden 路径快速失败判据）；TF 原生 API 不受影响。"""
    backend = CpuTfBackend()
    assert backend.is_npu_only("tf.npu_fused_op")
    assert backend.is_npu_only("npu_device.compat.v1.tbe.npu_vector_ops.lamb_apply_optimizer_assign")
    assert backend.is_npu_only("npu_bridge.tbe.npu_vector_ops.some_op")
    assert not backend.is_npu_only("tf.raw_ops.Add")
    assert not backend.is_npu_only("tf.nn.relu")


def test_device_name_layering():
    """device_name 分层：Backend 默认退化 device_type；torch 版实现归 TorchBackend；
    CpuTorchBackend 保持退化覆写（无 torch.cpu.get_device_name）。"""
    assert "device_name" in TorchBackend.__dict__, "torch 版 device_name 应实现于 TorchBackend"
    assert "device_name" in CpuTorchBackend.__dict__, "CpuTorchBackend 须覆写（CPU 无 get_device_name）"
    assert Backend.__dict__["device_name"] is not TorchBackend.__dict__["device_name"]
    # NpuTorch 未覆写 → 继承 TorchBackend 的 torch 版实现
    assert "device_name" not in NpuTorchBackend.__dict__

    class _MinimalBackend(Backend):
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def synchronize(self, dev_id=0):
            pass

        def from_numpy(self, arr):
            return arr

        def to_numpy(self, tensor, safe=False):
            return tensor

        def has_device(self):
            return False

    minimal = _MinimalBackend()
    minimal._segment_name = "seg"
    assert minimal.device_name() == "seg", "基类默认应退化到 device_type()（框架中立）"


def test_cpu_no_device(monkeypatch):
    """CpuTorchBackend: device_type()='cpu'，has_device=False。"""
    cb = CpuTorchBackend()
    cb.torch_lib = "cpu"
    cb.profile = {}
    # CPU never goes through _build; _segment_name = 'cpu' is a class attribute.
    assert cb.device_type() == "cpu"
    assert cb.has_device() is False


def test_npu_backend_is_npu_device_type():
    """NpuTorchBackend: is_npu()=True, device_type()='npu'（类属性）。"""
    nb = NpuTorchBackend()
    assert nb.is_npu() is True
    assert nb.device_type() == "npu"
