#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
torch 框架子包 —— Adapter + 域文件（api_loader / param_extractor / tensor_binding /
eager_runner / graph_runner / hccl_apis / multi_device）。

模块加载均为惰性（方法内 import）：TF 运行经 adapter 只触达本 __init__，
不加载 torch 域实现（torch_npu 与 TF C 扩展同进程冲突）。
"""

from ..adapter import FrameworkAdapter


class TorchAdapter(FrameworkAdapter):
    """torch/torch_npu 框架适配器。"""

    framework = "torch"

    # ── 三件套 ──

    def load_api(self, api_name: str):
        from .api_loader import load_api

        return load_api(api_name)

    def extract_params(self, api_name: str):
        from .api_loader import ensure_registered
        from .param_extractor import get_api_params

        ensure_registered(api_name)
        return get_api_params(api_name)

    def bind_tensors(self, testcase, backend, dev_tensors, raw_inputs):
        from .tensor_binding import bind_tensors

        return bind_tensors(testcase, backend, dev_tensors, raw_inputs)

    # ── 执行器 ──

    def run_graph(
        self,
        testcase,
        backend,
        dev_id,
        switches,
        plan,
        resolved,
        is_tensor_method,
        is_inplace,
        raw_inputs,
        dynamic,
        is_aclgraph=False,
        deterministic_level=None,
    ):
        from .graph_runner import execute_graph

        return execute_graph(
            testcase,
            backend,
            dev_id,
            switches,
            plan,
            resolved,
            is_tensor_method,
            is_inplace,
            raw_inputs,
            dynamic,
            is_aclgraph=is_aclgraph,
            deterministic_level=deterministic_level,
        )

    # ── eager 模式钩子 ──

    def eager_call(self, resolved, args, kwargs, is_tensor_method):
        from .eager_runner import call

        return call(resolved, args, kwargs, is_tensor_method)

    def eager_prepare_backward(self, backend, args):
        from .eager_runner import enable_requires_grad

        enable_requires_grad(backend, args)

    def eager_backward(self, backend, output, args):
        from .eager_runner import run_backward

        return run_backward(backend, output, args)

    # ── 手工签名登记（manual 表属 torch 域：hccl 生态） ──

    def register_params(self, api_name: str, params, source: str = "manual"):
        from .param_extractor import get_api_params, register_api_params

        register_api_params(api_name, params, source)
        return get_api_params(api_name)

    # ── 设备态 ──

    def setup_device_state(self, backend, testcase, process_ctx):
        from .cube_math_type import setup_cube_math_type

        setup_cube_math_type(backend, testcase, process_ctx)


ADAPTER = TorchAdapter()
