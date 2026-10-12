#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/testcase：用例体系——解析/校验/筛选的完整实现 + worker 各阶段数据源。

文件布局：
- reader/：输入格式适配（path → Table，本目录唯一 IO）
- base.py：TestcaseBase 结构体根 + 字段声明语言
- api_base.py：ApiTestcaseBase API 级中间基类（命名主轴是 API 级 vs 算子级，
  tensor 字段丰富是实现细节）
- factory.py：UniversalTestcaseFactory（Table → 用例对象，参数化 + 纯函数化）
- api_param.py：ResolveApiParamBase 契约 + match_overload/coerce_value
  （重载匹配/值强转公共件；装配器 ParamPlan 在 framework_api/api_param.py）
- infershape.py：CSV 声明式输出推导（解析对账用）
- field_types/field_parser：字段声明语言解释器
- tensor.py：Tensor/TensorSlot 槽模型（AoS）——运行时数据基底
  （tensors/np_storages 为 Tensor.data/Tensor.storage 的 write-through 视图）
- input_data.py：输入数据生成期共享件（存储生成 / attributes 值覆盖）

类层次：
  TestcaseBase（base.py）
    ├── TestcaseKernel（kernel/testcase.py）
    │     └── GeirTestcase（geir/testcase.py）
    └── ApiTestcaseBase（api_base.py）
          ├── TestcaseAclnn（aclnn/testcase.py）
          └── TestcaseE2e（framework_api/testcase.py）

运行时数据（tensors/np_storages）以 Tensor.data/Tensor.storage 的
write-through 视图提供；声明侧（SoA 列 + flat 族）仍为权威源。
"""

__all__ = [
    "Table",
    "iter_tables",
    "read_table",
    "resolved_sheet",
    "TestcaseBase",
    "ApiTestcaseBase",
    "UniversalTestcaseFactory",
    "ResolveApiParamBase",
    "coerce_value",
    "match_overload",
    "Tensor",
    "TensorSlot",
]

from .api_base import ApiTestcaseBase
from .api_param import ResolveApiParamBase, coerce_value, match_overload
from .base import TestcaseBase
from .factory import UniversalTestcaseFactory
from .reader import Table, iter_tables, read_table, resolved_sheet
from .tensor import Tensor, TensorSlot
