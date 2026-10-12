#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""kernel 通路：RTS 单算子内核测试——编译（tbe/asc）+ 执行 + 比对。

- instance.py：KernelInstance（通路编排 + 编译结果 apply 语义）
- worker.py（run + run_compile）
- execute.py（execute_on_device 三模式分派 + KernelExecutor 逐轮序列）
- result.py（RTSProfilingParam / KernelExecuteResult / 结果结构契约）
- golden.py / input_generate.py / compare.py / params.py / device_state.py
- compile/（Opc + dyn/cst/bin + helper kernels + cext_loader）
- operator/（op_interface + KB server + registries）
- runtime/（RTS + adump + rts_info + hccl helper）
"""

from .instance import KernelInstance as KernelInstance  # noqa: F401

__all__ = ["KernelInstance"]
