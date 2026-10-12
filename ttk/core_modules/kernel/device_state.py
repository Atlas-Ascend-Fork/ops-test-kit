#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""设备状态控制：clear_ub/l1/l0 的多通路共享门面。

物理 SM 局部内存（UB/L1/L0）是设备级资源——裸 aclrt（RTSInterface）与
torch_npu 同进程共存（E2E 场景），谁启 helper kernel 清的都是同一份。
kernel/aclnn 在执行轮内调用；e2e 同一 worker 进程轮前调用；geir 在轮间
（C++ 子进程执行中不可介入，由父进程在轮间清，语义等价"每轮执行前清"）。

helper kernel `.o` 现场编译：warmup/clear_ub/clear_l1 走 tik
（kernel/compile/helper_kernels）；clear_l0 在 David 后平台走 AscendC ccec
（kernel/compile/ascendc_kernels，旧平台 tik fallback）。产物在
switches.kernel_meta 下——首次使用时触发编译（各通路进程首次调用，
编译产物路径复用 kernel 机制）。"""

__all__ = ["clear_device_state", "clear_ub", "clear_l1", "clear_l0"]


def _rts():
    from .runtime.rts_interface import RTSInterface

    return RTSInterface()


def _ensure_helper_kernels(switches):
    """首用触发 UB/L1/L0 清零 kernel 编译（产物落 kernel_meta）。

    三个 clear kernel 一并编译：e2e/geir 等非 kernel 通路首用 --clear-l0 时
    无 KernelInstance.setup 预编译，此处漏编会导致 RTS launch 缺 .o 失败
    （kernel 通路自身 setup 已预编译，此前掩盖了该缺口）。
    """
    from .compile.compiler import get_compiler

    get_compiler().compile_ub_clear()
    get_compiler().compile_l1_clear()
    get_compiler().compile_l0_clear()


def clear_ub(switches) -> None:
    """Clear UB 为 force_clear_ub 指定值（None/model 模式跳过）。"""
    if getattr(switches, "force_clear_ub", None) is None or switches.mode.is_model():
        return
    _ensure_helper_kernels(switches)
    _rts().clear_ub(switches)


def clear_l1(switches) -> None:
    """Clear L1（None/model 模式跳过）。"""
    if getattr(switches, "force_clear_l1", None) is None or switches.mode.is_model():
        return
    _ensure_helper_kernels(switches)
    _rts().clear_l1(switches)


def clear_l0(switches) -> None:
    """Clear L0A/L0B/L0C（None/model 模式跳过）。"""
    if getattr(switches, "force_clear_l0", None) is None or switches.mode.is_model():
        return
    _ensure_helper_kernels(switches)
    _rts().clear_l0(switches)


def clear_device_state(switches, before_each_run: bool = True) -> None:
    """--clear-* 三件统一入口。before_each_run=False 时仅 ub/l1/l0 任一配置才触发。"""
    clear_ub(switches)
    clear_l1(switches)
    clear_l0(switches)
