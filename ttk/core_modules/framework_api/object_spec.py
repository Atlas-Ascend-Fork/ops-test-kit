#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""声明式对象参数：CSV 传 spec dict，执行期构造真实对象传入 API。

例：attributes = {'generator': {'seed': 603, 'offset': 1024}}。
解析层（_coerce_value）校验归一为 GeneratorSpec，不构造对象；求值层
（本模块）在 backend 就绪后按 SPEC_BUILDERS 构造并回填 args/kwargs。
golden 侧不构造，spec 原样透传，插件读 seed/offset。
"""

import struct

from ..common.testcase.api_param import GeneratorSpec

__all__ = [
    "GeneratorSpec",
    "SPEC_BUILDERS",
    "build_generator",
    "materialize_specs",
    "strip_specs",
    "make_default_generator_reset",
]


def build_generator(spec, ctx):
    """{'seed':..,'offset':..} → 已播种 torch.Generator。非 NPU 拒绝（CPU MT19937 不可复刻）。"""
    if not ctx.backend.is_npu():
        raise RuntimeError(
            "GENERATOR_BACKEND_UNSUPPORTED: explicit generator spec requires a "
            "Philox backend (npu); CPU (MT19937) stream cannot be replicated"
        )
    import torch

    gen = torch.Generator(device=ctx.backend.device_type())
    gen.manual_seed(spec["seed"])
    gen.set_offset(spec.get("offset", 0))
    return gen


# builder 注册表：spec 标记类 → 构造函数（插件可扩展）
SPEC_BUILDERS = {GeneratorSpec: build_generator}


class _SpecCtx:
    """求值上下文：backend（device/角色判定经其契约接口）。"""

    def __init__(self, backend):
        self.backend = backend


def _spec_slots(args, kwargs):
    """扫描 args/kwargs 中的 spec，返回 [(container, key, spec), ...]。"""
    slots = [(args, i, v) for i, v in enumerate(args or ()) if type(v) in SPEC_BUILDERS]
    slots += [(kwargs, k, v) for k, v in (kwargs or {}).items() if type(v) in SPEC_BUILDERS]
    return slots


def materialize_specs(args, kwargs, backend):
    """args/kwargs 中的 spec → 构造对象原地替换，返回轮前复位句柄（无 spec 为 None）。

    复位 = 每轮按 spec 重新构造（有状态对象回到声明态），eager 轮循环逐轮调用。
    """
    slots = _spec_slots(args, kwargs)
    if not slots:
        return None
    ctx = _SpecCtx(backend)
    for container, key, spec in slots:
        container[key] = SPEC_BUILDERS[type(spec)](spec, ctx)

    def round_reset():
        for container, key, spec in slots:
            container[key] = SPEC_BUILDERS[type(spec)](spec, ctx)

    return round_reset


def strip_specs(args, kwargs):
    """摘除 args/kwargs 中的 spec（图模式用），返回 spec 列表。

    位置槽置 None（不得摘除，避免后续参数错位）。
    """
    specs = []
    for i, v in enumerate(args or ()):
        if type(v) in SPEC_BUILDERS:
            args[i] = None
            specs.append(v)
    for k, v in list((kwargs or {}).items()):
        if type(v) in SPEC_BUILDERS:
            kwargs[k] = None
            specs.append(v)
    return specs


def make_default_generator_reset(spec, backend, dev_id):
    """图模式腿（-c/-d）用：默认 generator 的轮前复位闭包。

    Dynamo 无法 proxy Generator（断图），图腿摘除显式 generator 后随机算子
    回落默认 generator；TorchAir 的 seed/offset 为图输入语义（每 replay 重读
    当前状态），每次 compiled() 调用前重播种到 (seed, offset) 即与 eager
    轮前复位同效。aclgraph 不适用（捕获期禁读 RNG 状态）。
    """
    import torch

    seed = int(spec["seed"])
    offset = int(spec.get("offset", 0))

    def round_reset():
        with backend.device_scope(dev_id):
            torch.npu.manual_seed(seed)
            state = torch.npu.get_rng_state().clone()
            state[8:16] = torch.tensor(
                list(struct.pack("<Q", offset & 0xFFFFFFFFFFFFFFFF)),
                dtype=torch.uint8,
            )
            torch.npu.set_rng_state(state)

    return round_reset
