#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
FrameworkAdapter —— torch / tf / onnx 框架适配协议 + 运行级注册表。

每个框架子包（framework_api/torch、framework_api/tf、framework_api/onnx）提供
自己的 Adapter，自治三件套：
  - load_api        模块加载：api_name → (callable | 方法名, is_tensor_method)
  - extract_params  参数解析：api_name → APIParamInfo | None
  - bind_tensors    tensor 绑定（L2 特化）：format_cast / ref 复位 / const 位
以及执行器（run_graph、wrap_eager）、eager 模式钩子（eager_call / backward，execute
消费）、mutability 钩子（is_mutable_api / mutable_slots / mutable_readback，模型驱动）
与复位钩子（reset_mutable_inputs，golden.py 消费）。

路由规则：框架选择经 common/backend.framework_detector.detect_framework（单源），
本模块不做 api_name 前缀解析；get_adapter 惰性加载对应子包——torch 与 tf
运行时互不加载（torch_npu 与 TF C 扩展同进程冲突，见 tf/param_extractor.py 模块注记）。
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..common.backend import detect_framework

_ADAPTERS = {}
_ADAPTER_PACKAGE = {"torch": ".torch", "tf": ".tf", "onnx": ".onnx"}


class FrameworkAdapter(ABC):
    """框架适配协议。默认实现为通用/空行为，子包按需覆写。"""

    framework: str

    # ── 三件套 ──────────────────────────────────────────────

    @abstractmethod
    def load_api(self, api_name: str):
        """api_name → (callable | Tensor 方法名, is_tensor_method)。"""

    @abstractmethod
    def extract_params(self, api_name: str):
        """api_name → APIParamInfo | None（解析失败返回 None，由调用方记日志）。"""

    def bind_tensors(self, testcase, backend, dev_tensors, raw_inputs):
        """L2 绑定特化：to_device 之后的框架修正（默认原样返回）。

        torch：npu_format_cast；tf：mutable-ref 变量复位 + const 位以 Python 值传入。
        L1 张量构造（np → 框架张量）不走本钩子——由 backend.inputs_from_numpy 承担。
        """
        return dev_tensors

    # ── 执行器 ──────────────────────────────────────────────

    def wrap_eager(self, resolved, api_name, backend):
        """eager 调用包装（默认原样返回）。

        tf：CPU 侧 Resource* handle 转换 / NPU 侧 tf.function 包装（npu_device
        只对 tf.function 内的 op 触发 GE 编译）；onnx：backend 注入点（soc/dev_id）。
        """
        return resolved

    def eager_graph_alias(self) -> Optional[str]:
        """eager 与图模式的等价声明：'cst' 表示该框架的 eager 执行即静态图执行
        （onnx：om 恒为静态图）。双开（--eager + -c）时 worker 只执行一次，
        结果镜像到 EAGER 与 GRAPH CST 两列。None = 无等价（torch/tf）。
        """
        return None

    @abstractmethod
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
        """图模式执行器：torch → torch.compile + torchair；tf → tf.function。

        ctx 为 RunCtx（deterministic_level 取自 ctx）。返回 4 元组
        (nps, ProfileResult, det_status, err_msg)：失败时 nps=[]、err_msg
        为异常描述；成功或跳过时 err_msg 为 None。
        """

    # ── eager 模式钩子（worker / onnx 委托消费） ─────────

    @abstractmethod
    def eager_call(self, resolved, args, kwargs, is_tensor_method):
        """单次 API 调用（torch Tensor 方法经 args[0] 分发；tf 直调）。"""

    def configure_eager_compile_mode(self, backend, switches):  # noqa: B027, ARG002  # 可选钩子：默认无操作
        """eager 设备前置（默认无操作）。torch：torch_npu 编译模式设置。"""

    def run_eager(self, ctx, *, testcase, plan, is_inplace, raw_inputs):
        """eager 执行器：公共编排（设备置位 / 输入装配 / 采集 / 逐轮载体）+
        内层逐轮策略 self.eager_rounds（torch 单窗口；tf 两阶段，关采集时
        只跑正确性轮）。onnx 无 eager，不走本执行器（经 run_graph 自持编排）。

        ctx 为 RunCtx（worker 构建，eager/graph 共用）。返回
        (result_nps, perf, det_status)。
        """
        from ..common.invoke import Invoke
        from ..common.profiler import ProfilerConfig, get_profiler
        from .utils import (
            _declared_inplace_keys,
            _eager_mutable_keys,
            finalize_det_status,
            prepare_device_args,
            resolve_npu_preprocess_hook,
            unpack_4bit_outputs,
        )

        backend = ctx.backend
        switches = ctx.switches
        dev_id = ctx.dev_id
        backend.set_device(dev_id)
        self.configure_eager_compile_mode(backend, switches)
        args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs, adapter=self)

        # 声明式对象参数物化（generator spec → torch.Generator）；
        # 轮前复位句柄由 eager 轮循环逐轮调用
        from .object_spec import materialize_specs

        ctx.round_reset = materialize_specs(args, kwargs, backend)

        if backend.is_npu():
            result = Invoke.npu_preprocess(
                testcase,
                resolve_npu_preprocess_hook(testcase, switches),
                switches,
                device_args=(args, kwargs),
                device_scope=lambda: backend.device_scope(dev_id),
            )
            testcase.absorb_preprocess(result, plan, args, kwargs)

        profiling_enabled = bool(getattr(switches, "task_profiling", True))
        profiler = get_profiler(
            testcase.api_name,
            backend,
            ProfilerConfig(
                testcase_name=testcase.testcase_name,
                root_path=switches.root_path,
                dev_id=dev_id,
                enabled=profiling_enabled,
                warmup_count=ctx.warmup_count,
            ),
        )

        inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
        # 键约定: int = args 下标; str = kwargs 参数名(TF raw_ops 全部按名传参, args 为空)
        inplace_idxs = _eager_mutable_keys(self, testcase.api_name, args, kwargs, is_inplace)
        inplace_idxs |= _declared_inplace_keys(self, testcase.api_name, args, kwargs, inplace_input_indexes)

        total = ctx.warmup_count + ctx.run_count
        # 逐轮可变槽载体预置（backend 封装：torch 逐轮克隆 / tf 同变量 assign 复位）
        mutable_loader = backend.prepare_mutable_rounds(args, kwargs, inplace_idxs, total)

        backward_mode = getattr(testcase, "backward_mode", False)
        if backward_mode:
            # backward：浮点输入自动 requires_grad=True（整数索引类天然无梯度，
            # autograd.grad allow_unused 兜底）。梯度是第二组返回值（unused 位过滤
            # 丢弃，golden 侧同构），追加在 outputs 尾部——比对/判据/ dump/结果打包零改动。
            self.eager_prepare_backward(backend, args)

        result, result_nps, md5_list = self.eager_rounds(
            ctx,
            api_name=testcase.api_name,
            args=args,
            kwargs=kwargs,
            inplace_input_indexes=inplace_input_indexes,
            mutable_loader=mutable_loader,
            profiler=profiler,
            backward_mode=backward_mode,
        )

        perf = profiler.result(ctx.run_count)
        det_status = finalize_det_status(md5_list, testcase.testcase_name)
        result_nps = unpack_4bit_outputs(testcase, result_nps)
        return result_nps, perf, det_status

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
        """eager 内层逐轮策略（各框架自行实现）。

        torch：单窗口内 warmup + 正确性轮（run_torch_eager_rounds）；tf：
        两阶段（run_tf_eager_rounds）——复位 assign 是真实设备 op，进了采集
        窗会被 msprof 算成额外 kernel，所以正确性轮留在窗外，窗口内纯调用
        不复位。返回 (result, result_nps, md5_list)，result_nps 为末轮输出
        的 numpy。onnx 无 eager，不实现本钩子，误触抛 NotImplementedError。
        """
        raise NotImplementedError(f"{self.framework} adapter does not implement eager_rounds")

    def eager_prepare_backward(self, backend, args):  # noqa: B027  # 可选钩子：默认无操作
        """backward 前置（默认无操作）。torch：浮点输入置 requires_grad。"""

    def eager_backward(self, backend, output, args):
        """backward 采集（默认无梯度）。torch：autograd.grad 函数式。"""
        return

    # ── mutability（"输出=被写输入位"，ParamInfo.is_mutable 模型驱动） ──

    def _mutable_model(self, api_name: str):
        """api_name 的参数模型；异框架名返回 None（不触达对方框架的解析器）。"""
        if not isinstance(api_name, str) or detect_framework(api_name) != self.framework:
            return None
        from .info_keeper import FrameworkApiInfoKeeper

        return FrameworkApiInfoKeeper().get(api_name)

    def is_mutable_api(self, api_name: str) -> bool:
        """参数表含被写位：输出=被写输入位（torch inplace / tf mutable-ref）。"""
        info = self._mutable_model(api_name)
        return bool(info) and any(p.is_mutable for p in info.params)

    def mutable_slots(self, api_name: str):
        """被写参数槽 → (args 位置下标集, kwargs 参数名集)；位置=参数表位次（kw-only 不占 args 位）。"""
        info = self._mutable_model(api_name)
        if not info:
            return frozenset(), frozenset()
        positions = frozenset(i for i, p in enumerate(info.params) if p.is_mutable and not p.is_keyword_only)
        names = frozenset(p.name for p in info.params if p.is_mutable)
        return positions, names

    def tensor_param_names(self, api_name: str) -> Optional[tuple]:  # noqa: ARG002  # 可选钩子：默认 None
        """张量序参数名表（flat 位次 → 参数名）。tf 实现（raw_ops 声明下标
        转 kwargs 键用）；torch 声明下标即 args 位，无需实现。"""
        return None

    def mutable_readback(self, api_name, args, kwargs):  # noqa: ARG002
        """被写参数位的当前值列表（原地更新后的结果）；默认无，框架按需覆写。"""
        return []

    # ── 谓词 ────────────────────────────────────────────────

    def is_inplace(self, resolved, is_tensor_method):
        """原地判定：Tensor 方法看方法名尾下划线，普通函数看 __name__ 尾下划线。"""
        if is_tensor_method:
            return resolved.endswith("_")
        return getattr(resolved, "__name__", "").endswith("_")

    def reset_mutable_inputs(self, testcase, dev_tensors, raw_inputs):  # noqa: B027  # 可选钩子：默认无操作
        """mutable 输入复位（golden 生成前按 raw_inputs 回写；默认无操作）。

        tf：ref 变量被 eager/graph 模式原地改写后，golden 须从初值起重算；torch 无
        mutable 位，继承默认 no-op。
        """

    def register_params(self, api_name: str, params, source: str = "manual"):  # noqa: ARG002, B027  # 可选钩子：默认无操作
        """手工签名登记（默认不支持，返回 None 由调用方忽略）。

        torch：入 torch/param_extractor 的 _MANUAL_OVERRIDES 注册表并回读；
        tf 无手工登记表。manual 登记属 torch 域（hccl 生态），共享层经本钩子触达，
        不直连子包模块。
        """

    # ── 设备态 ──────────────────────────────────────────────

    def setup_device_state(self, backend, testcase, process_ctx):  # noqa: B027  # 可选钩子：默认无操作
        """执行段前的进程级设备态配置（默认无操作）。

        torch：torch_npu CubeMathType 精度模式（torch/cube_math_type.py，
        执行锁内、golden NPU 段与被测段共用同一配置）；tf/onnx 无对应概念。
        由 worker 编排层经本钩子触达，不直连子包模块。
        """


def get_adapter(framework: str = "torch") -> FrameworkAdapter:
    """按框架名取 Adapter 单例（惰性加载子包并缓存）。"""
    framework = framework or "torch"
    if framework not in _ADAPTER_PACKAGE:
        raise ValueError(f"Unknown framework: {framework!r} (known: {sorted(_ADAPTER_PACKAGE)})")
    if framework not in _ADAPTERS:
        import importlib

        module = importlib.import_module(_ADAPTER_PACKAGE[framework], __package__)
        _ADAPTERS[framework] = module.ADAPTER
    return _ADAPTERS[framework]


def adapter_for_api_name(api_name: str) -> FrameworkAdapter:
    """api_name 前缀（detect_framework 单源）→ Adapter。"""
    return get_adapter(detect_framework(api_name))


def adapter_for_testcase(testcase) -> FrameworkAdapter:
    """用例 → Adapter（api_name 为空时按 torch 处理，与 detect_framework 一致）。"""
    return get_adapter(detect_framework(getattr(testcase, "api_name", "") or ""))
