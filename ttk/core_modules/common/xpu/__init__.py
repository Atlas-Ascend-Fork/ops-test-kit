#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/xpu：三方标杆执行器（参数绑定 + 编排 + manual 回放 + resolver）。

- binding.py：bind_params / invoke_third_party —— 三方 API 直调的绑定单份
  实现（整仓部署后 remote server 同源使用）
- dispatch.py：XpuDispatch 句柄 + 重叠预算 + remote client 门面（三方采集
  仅 remote：ttk.conf.yaml remote.endpoints）
- manual.py：manual 三方输出的本地装载/落盘（--dump xpu 回放契约）
- resolver.py：op_name → 三方 API 搜索注册制（仅 Kernel/ACLNN 通路）与
  采集结果桥 extract_third_party

三套装配分层（并存不合并）：
  ① spec 方法注入（invoke.py 私有 _bind_by_name）——spec 作者的 golden/compare
  ② invoke_third_party（binding.py bind_params）——三方 API 直调（server + spec 类形态）
  ③ remote client（HTTP/npz 传输）——spec third_party 声明经远端执行

本包为汇总 re-export 面，消费方 import 路径保持不变。
"""

from .binding import UnknownParamError, bind_params, invoke_third_party
from .dispatch import (
    XPU_DATA,
    XPU_PERF,
    XpuDispatch,
    collect_third_party,
    overlap_deadline_s,
    snapshot_xpu_inputs,
    xpu_join_timeout,
    xpu_mode,
    xpu_overlap_enabled,
)
from .manual import (
    MANUAL_XPU_PROVIDER,
    ManualXpuLoadError,
    dump_xpu_slot_files,
    has_manual_xpu_source,
    load_manual_third_parties,
    manual_xpu_results,
)
from .resolver import (
    XpuResolver,
    extract_third_party,
    get_xpu_resolver,
    register_xpu_resolver,
)

__all__ = [
    "XpuResolver",
    "get_xpu_resolver",
    "register_xpu_resolver",
    "invoke_third_party",
    "UnknownParamError",
    "bind_params",
    "xpu_mode",
    "XPU_DATA",
    "XPU_PERF",
    "XpuDispatch",
    "overlap_deadline_s",
    "xpu_join_timeout",
    "snapshot_xpu_inputs",
    "xpu_overlap_enabled",
    "extract_third_party",
    # manual_xpu 装载/落盘（--dump xpu 回放契约）
    "MANUAL_XPU_PROVIDER",
    "ManualXpuLoadError",
    "load_manual_third_parties",
    "has_manual_xpu_source",
    "manual_xpu_results",
    "dump_xpu_slot_files",
    "collect_third_party",
]
