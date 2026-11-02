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
tf 框架子包 —— Adapter + 域文件（api_loader / param_extractor /
tensor_binding / eager_runner / graph_runner）。

包导入即完成 L1 张量构造注册（tensor_binding → common/backend/framework_hooks）。
顶层零 torch/torch_npu import（结构测试钉住）。
"""

from ..adapter import FrameworkAdapter
from . import tensor_binding  # noqa: F401  # 注册输入工厂（framework_hooks）


class TfAdapter(FrameworkAdapter):
    """TensorFlow 框架适配器。"""

    framework = "tf"

    # ── 三件套 ──

    def load_api(self, api_name: str):
        from .api_loader import load_api

        return load_api(api_name)

    def extract_params(self, api_name: str):
        from .param_extractor import extract_tf_params

        return extract_tf_params(api_name)

    def bind_tensors(self, testcase, backend, dev_tensors, raw_inputs):
        from .tensor_binding import bind_tensors

        return bind_tensors(testcase, backend, dev_tensors, raw_inputs)

    # ── 执行器 ──

    def wrap_eager(self, resolved, api_name, backend):
        from .eager_runner import wrap_eager_callable

        return wrap_eager_callable(resolved, api_name, backend)

    def run_graph(
        self,
        ctx,
        *,
        testcase,
        plan,
        is_inplace,
        raw_inputs,
        dynamic,
        is_aclgraph=False,
    ):
        from .graph_runner import execute_tf_graph

        return execute_tf_graph(
            testcase,
            ctx.backend,
            ctx.dev_id,
            ctx.switches,
            plan,
            ctx.resolved,
            ctx.is_tensor_method,
            is_inplace,
            raw_inputs,
            dynamic,
            is_aclgraph=is_aclgraph,
            deterministic_level=ctx.deterministic_level,
        )

    # ── mutability（统一钩子的 tf 实现） ──

    def tensor_param_names(self, api_name):
        from .param_extractor import flat_tensor_param_names

        return flat_tensor_param_names(api_name)

    def mutable_readback(self, api_name, args, kwargs):
        from .eager_runner import readback_mutable

        return readback_mutable(api_name, args, kwargs)

    # ── eager 模式钩子 ──

    def eager_call(self, resolved, args, kwargs, is_tensor_method):  # noqa: ARG001
        from .eager_runner import call

        return call(resolved, args, kwargs, is_tensor_method)

    def eager_rounds(
        self,
        ctx,
        *,
        api_name,
        args,
        kwargs,
        inplace_input_indexes,
        mutable_loader,
        profiler,
        backward_mode,
    ):
        # 关采集时 run_tf_eager_rounds 只跑正确性轮（不进计时段），不再委托基类
        from .eager_runner import run_tf_eager_rounds

        return run_tf_eager_rounds(
            self,
            ctx,
            api_name=api_name,
            args=args,
            kwargs=kwargs,
            inplace_input_indexes=inplace_input_indexes,
            mutable_loader=mutable_loader,
            profiler=profiler,
            backward_mode=backward_mode,
        )

    # ── golden 复位钩子 ──

    def reset_mutable_inputs(self, testcase, dev_tensors, raw_inputs):  # noqa: ARG002
        from .param_extractor import reset_ref_variables

        reset_ref_variables(dev_tensors, raw_inputs)


ADAPTER = TfAdapter()
