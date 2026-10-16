#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""core_modules/common：四通路共享的公共能力层。

消费规则：
1. 通路经 common 子包公开面消费（``from ..common.device import ...``；
   子包 ``__init__`` 的 ``__all__`` 即契约，包根 re-export 亦可
   ``from ..common import Task, ...``）；跨目录 import 下划线成员违规。
2. 新增公开名 = 同时改本文件 ``__all__`` 与 scripts/common_public_surface.json
   （经 scripts/check_conventions.py 断言）。
"""

__all__ = [
    # device/
    "DSMIInterface",
    "PLATFORM_BEFORE_DAVID",
    "get_opp_paths",
    "get_op_impl_paths",
    "get_impl_base_paths",
    "get_op_info_paths",
    "get_ascend_lib64_path",
    "get_npu_hw_info",
    "get_l0_clear_tile_params",
    "get_ascend_scene_info",
    # logging.py
    "default_logging_config",
    "build_single_log_dir",
    # process/
    "Task",
    "TaskKeeper",
    "ProcessGroup",
    "process_context",
    "ProcessContext",
    "DeviceLock",
    "MultiDeviceLock",
    # dump/
    "DataDumper",
    "DumpConfig",
    "register_format",
    "get_formatter",
    "ManualDataStore",
    "ManualDataCase",
    "resolve_manual_case",
    "load_csv_manual_case",
    "snapshot_manual_values",
    "case_directory_name",
    "ManualDataError",
    # simulator/
    "run_kernel_sim",
    "run_aclnn_sim",
    "case_dir",
    "maybe_generate_sim_report",
    "resolve_platform_soc",
    "resolve_camodel_lib_dir",
    # comparison/
    "CompareHooks",
    "compare_with_hooks",
    "invoke_pre_compare",
    "register_comparison",
    "compare_batch_consistency",
    # result/
    "ReturnStructureBase",
    "CompareResult",
    # plugin/
    "get_plugin_function",
    "SpecResolver",
    "get_spec_class_meta",
    "InvalidSpecError",
    # backend/
    "Backend",
    "get_backend",
    "resolve_api",
    # xpu.py
    "XpuResolver",
    "get_xpu_resolver",
    "register_xpu_resolver",
    "invoke_third_party",
    "UnknownParamError",
    "xpu_mode",
    "extract_third_party",
    # invoke.py
    "Invoke",
    "resolve_callable_str",
    # instance.py
    "InstanceBase",
    "register_instance",
    "get_instance_by_name",
    # profiler/
    "Profiler",
    "ProfilerConfig",
    "ProfileResult",
    "KernelDetails",
    "KernelInfo",
    "get_profiler",
    "MsProfiler",
    "MsProfOpDfx",
    # testcase/
    "Table",
    "iter_tables",
    "read_table",
    "resolved_sheet",
    "TestcaseBase",
    "ApiTestcaseBase",
    "UniversalTestcaseFactory",
    "coerce_value",
    "match_overload",
]

from . import backend as backend
from . import comparison as comparison
from . import device as device
from . import dump as dump
from . import process as process
from . import profiler as profiler
from . import result as result
from . import simulator as simulator
from . import testcase as testcase
from .backend import Backend, get_backend, resolve_api
from .comparison import (
    CompareHooks,
    compare_batch_consistency,
    compare_with_hooks,
    invoke_pre_compare,
    register_comparison,
)
from .device import (
    PLATFORM_BEFORE_DAVID,
    DSMIInterface,
    get_ascend_lib64_path,
    get_ascend_scene_info,
    get_impl_base_paths,
    get_l0_clear_tile_params,
    get_npu_hw_info,
    get_op_impl_paths,
    get_op_info_paths,
    get_opp_paths,
    validate_core_limit,
)
from .dump import (
    DataDumper,
    DumpConfig,
    ManualDataCase,
    ManualDataError,
    ManualDataStore,
    case_directory_name,
    get_formatter,
    load_csv_manual_case,
    register_format,
    resolve_manual_case,
    snapshot_manual_values,
)
from .instance import InstanceBase, get_instance_by_name, register_instance
from .invoke import Invoke, resolve_callable_str
from .logging import build_single_log_dir, default_logging_config
from .process import (
    DeviceLock,
    MultiDeviceLock,
    ProcessContext,
    ProcessGroup,
    Task,
    TaskKeeper,
    process_context,
)
from .profiler import (
    KernelDetails,
    KernelInfo,
    MsProfiler,
    MsProfOpDfx,
    Profiler,
    ProfilerConfig,
    ProfileResult,
    get_profiler,
)
from .result import CompareResult, ReturnStructureBase
from .simulator import (
    case_dir,
    maybe_generate_sim_report,
    resolve_camodel_lib_dir,
    resolve_platform_soc,
    run_aclnn_sim,
    run_kernel_sim,
)
from .testcase import (
    ApiTestcaseBase,
    Table,
    TestcaseBase,
    UniversalTestcaseFactory,
    coerce_value,
    iter_tables,
    match_overload,
    read_table,
    resolved_sheet,
)
from .xpu import (
    UnknownParamError,
    XpuResolver,
    extract_third_party,
    get_xpu_resolver,
    invoke_third_party,
    register_xpu_resolver,
    xpu_mode,
)

# 懒导出：兑现 __all__ 声明但不 eager import（plugin/ 公开名经子包
# __getattr__ 二级懒加载），首次访问后缓存到模块字典。
_LAZY_EXPORTS = {
    "get_plugin_function": "plugin",
    "SpecResolver": "plugin",
    "get_spec_class_meta": "plugin",
    "InvalidSpecError": "plugin",
}


def __getattr__(name):
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    attr = getattr(importlib.import_module(f".{submodule}", __name__), name)
    globals()[name] = attr  # 缓存，后续访问不再走 __getattr__
    return attr
