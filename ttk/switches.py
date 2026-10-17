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
Global switches：运行形态（MODE）/ 全局开关单一大类（SWITCHES）/
Kernel 三模式子开关（OPTestSwitch）/ simt 栈配置（SoCSimtCfg）。
全局访问：SWITCHES.active() 只读 + sw.activate() 激活；
get/set_global_storage 为兼容函数（消费方迁移完成后删）。
"""

# Standard Packages
import os
import pathlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

__all__ = ["MODE", "SoCSimtCfg", "SWITCHES", "OPTestSwitch", "get_global_storage", "set_global_storage"]


class MODE(Enum):
    """Model Type"""

    ASCEND_ONBOARD = auto()
    ASCEND_CAMODEL = auto()
    ASCEND_PEMMODEL = auto()
    ASCEND_PVMODEL = auto()
    ASCEND_ESLMODEL = auto()

    def is_model(self) -> Optional[str]:
        if self in [MODE.ASCEND_ESLMODEL, MODE.ASCEND_CAMODEL, MODE.ASCEND_PEMMODEL, MODE.ASCEND_PVMODEL]:
            return self.name.split("_")[-1]
        return None

    def is_online_board(self) -> bool:
        return self is MODE.ASCEND_ONBOARD

    def has_device(self) -> bool:
        return self.is_online_board()

    def is_esl_model(self) -> bool:
        return self is MODE.ASCEND_ESLMODEL


@dataclass
class SoCSimtCfg:
    """simt 栈大小配置（进程级，单位 bytes）。"""

    dcu_stack: Optional[int] = None
    dvg_stack: Optional[int] = None


class SWITCHES:
    """全局运行配置：CLI 解析一次，主进程构造后 activate，worker 经 pickle 恢复，
    全程只读。字段按消费域分六组（见 _init_* 与 __slots__ 分节注释）。"""

    # 全局存储：bridge / worker_bootstrap / simulator wrapper 经 activate() 注入；
    # 消费方读当前激活实例（active()，get_global_storage 为兼容委托）
    _active: Optional["SWITCHES"] = None

    @classmethod
    def active(cls) -> "SWITCHES":
        """当前激活实例；未激活抛 RuntimeError（早失败优于静默错配）。"""
        if cls._active is None:
            raise RuntimeError("SWITCHES not activated; call activate() on an instance first")
        return cls._active

    def activate(self) -> None:
        """激活本实例为全局当前配置（bridge / worker_bootstrap / simulator）。"""
        SWITCHES._active = self

    # 顺序与 _init_* 六组一致；分组语义见各 _init 方法
    __slots__ = [
        # 平台与通路：决定走哪条测试链、什么硬件形态
        "root_path",
        "test_mode",
        "mode",
        "backend",
        "dev_plat",
        "short_soc_version",
        "framework",
        # IO 与配置：输入输出文件与外部配置
        "input_files",
        "sheet",
        "output_file_name",
        "append_mode",
        "logging_to_file",
        "config_path",
        "provider_filter",
        "plugin_path",
        # 用例筛选
        "selected_testcases",
        "selected_testcase_indexes",
        "selected_testcase_count",
        "selected_operators",
        "excluded_operators",
        "priorities",
        "rerun_targets",
        "random_seed",
        # 设备与执行：设备选择/进程模型/Kernel 编译执行主链
        "device_count",
        "device_count_is_limit",
        "device_blacklist",
        "process_per_device",
        "run_timeout",
        "proc_timeout",
        "proc_no_reuse",
        "no_memory_check",
        "warmup",
        "tiling_run_time",
        "kernel_meta",
        "compile_options",
        "op_impl_mode",
        "dyn_switches",
        "cst_switches",
        "bin_switches",
        "force_clear_atomic",
        "force_clear_ub",
        "force_clear_l1",
        "force_clear_l0",
        "force_block_dim",
        "expect_impl_source",
        "force_simt_ub_size",
        "simt_cfg",
        "reuse_hbm",
        "reserve_hbm",
        "hbm_size_limit",
        "task_profiling",
        "summary_print",
        "progress_output",
        "validate_only",
        "deterministic_level",
        "_run_time",
        "_compile_only",
        # 精度与数据
        "input_distribution",
        "golden_mode",
        "compare_method",
        "xpu_perf",
        "xpu_zero_upload",
        "dump_config",
        # 通路/模式专属：仅特定 test_mode / backend 消费（按通路分节）
        # framework-api（E2E）
        "force_cpu",
        "fullgraph",
        "aclgraph_enabled",
        "e2e_binary",
        "eager_enabled",
        "super_kernel_enabled",
        "core_limit",
        # GEIR
        "geir_binary",
        # 手动数据（kernel/aclnn/e2e 三通路共用）：no_prof=不上板标记（关 prof 执行，
        # 配合 dump 门控触发 manual-data 落盘）；manual_data_dirs=数据目录集
        # （load 方向搜索空间；no-prof 落盘时为单目录写目标）
        "no_prof",
        "manual_data_dirs",
        # NPUSim 仿真后端
        "sim_soc_version",
        "sim_output_dir",
        "sim_report",
        "sim_cores",
        "sim_object_file",
    ]

    def __init__(self):
        self._init_platform()
        self._init_io()
        self._init_case_selection()
        self._init_execution()
        self._init_precision()
        self._init_mode_specific()

    def __getstate__(self):
        """Pickle 支持：仅导出 __slots__ 中已赋值属性（私有槽 _run_time/_compile_only 含内；property 不在 slots 不含）。"""
        return {k: getattr(self, k, None) for k in self.__slots__ if hasattr(self, k)}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def overflow_mode(self) -> int:
        """溢出语义恒为 INF/NAN 模式（1）；saturation（0）通路仅 golden 侧 mock 测试保留。"""
        return 1

    @property
    def run_time(self) -> int:
        if self.mode.is_model():
            return self._run_time or 1
        return self._run_time or 3

    @run_time.setter
    def run_time(self, val):
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ValueError(f"run_time must be a positive int, got {val!r}")
        self._run_time = val

    @property
    def compile_only(self) -> bool:
        return self._compile_only

    @compile_only.setter
    def compile_only(self, val: bool):
        if val:
            self.dyn_switches.prof = False
            self.cst_switches.prof = False
            self.bin_switches.prof = False
            self._compile_only = True
        else:
            self.dyn_switches.prof = True
            self.cst_switches.prof = True
            self.bin_switches.prof = True
            self._compile_only = False

    def oom_enabled(self) -> bool:
        return "oom" in self.compile_options.get("op_debug_config", "")

    def _init_platform(self):
        """平台与通路：op(kernel)/aclnn/geir/framework-api 四链 + 板端/仿真形态。"""
        # ttk/switches.py → 2 层 dirname 即仓库根（自 utilities 迁出后少一层）
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_mode: str = "op"
        self.mode: MODE = MODE.ASCEND_ONBOARD
        self.backend: str = "npu"  # "npu" | "npusim"；"cpu" 仅 e2e（配合 force_cpu）
        self.dev_plat: str = "AUTO"
        self.short_soc_version: Optional[str] = None  # Instance 启动时探测回写；None in non-NPU mode
        self.framework: str = "torch"

    def _init_io(self):
        """IO 与配置：输入 CSV/输出报告文件、外部 yaml 配置与 golden 插件路径。"""
        self.input_files: Optional[List[str]] = None
        self.sheet: Optional[str] = None
        self.output_file_name: Optional[str] = None
        self.append_mode: bool = False
        self.logging_to_file: bool = False
        self.config_path: Optional[str] = None  # --config yaml 路径，经 pickle 传 worker
        self.provider_filter: Optional[str] = None  # --provider 过滤器，remote/client 消费
        self.plugin_path: Optional[Tuple[pathlib.Path]] = None  # --plugin golden/输入插件目录

    def _init_case_selection(self):
        """用例筛选：按名/序号/算子/优先级圈选 + 随机抽取种子。"""
        self.selected_testcases = []
        self.selected_testcase_indexes = []
        self.selected_testcase_count = -1
        self.selected_operators = None
        self.excluded_operators = None
        self.priorities: Optional[tuple] = None
        self.rerun_targets = None
        self.random_seed = None

    def _init_execution(self):
        """设备与执行：设备/进程模型、Kernel 三模式编译执行、设备状态强制项。"""
        self.device_count = -1
        # --dev 是数量上限：物理卡不足时压缩而非报错；--devices 点名越界仍报错
        self.device_count_is_limit = False
        self.device_blacklist: tuple = ()
        self.process_per_device: Optional[int] = None
        self.run_timeout = 0
        self.proc_timeout = 3600
        self.proc_no_reuse = False
        self.no_memory_check = False  # --no-memory-check：跳过 host 内存闸门（宽松版，process/memory_gate）
        self.warmup = True
        self.tiling_run_time = 3
        self.kernel_meta = os.path.join(self.root_path, "kernel_meta")
        self.compile_options: dict = {}
        self.op_impl_mode: Optional[str] = None
        self.dyn_switches: OPTestSwitch = OPTestSwitch("Dynamic Shape", True, True, True)
        self.cst_switches: OPTestSwitch = OPTestSwitch("Const Shape", False, True, True)
        self.bin_switches: OPTestSwitch = OPTestSwitch("Binary Release", False, True, True)
        self.force_clear_atomic = [None, None, None]
        self.force_clear_ub = None
        self.force_clear_l1 = None
        self.force_clear_l0 = None
        self.force_block_dim = [None, None, None]
        # --expect-impl-source——实现来源与预期不符时显式失败（builtin/vendor/custom[=name]）
        self.expect_impl_source: str = None
        self.force_simt_ub_size = [None, None, None]  # dynamic/const/binary
        self.simt_cfg: SoCSimtCfg = SoCSimtCfg()
        self.reuse_hbm: bool = False
        self.reserve_hbm: int = 0
        self.hbm_size_limit = 30  # GB，输入张量总量守门上限
        self.task_profiling = True
        self.summary_print = True
        self.progress_output = None
        self.validate_only: bool = False  # --validate 仅解析校验用例，不执行
        self.deterministic_level: int = 0  # 确定性等级，kernel/aclnn/geir/e2e 全链消费
        # 私有槽（property run_time / compile_only 的存储——公开读写走 property：
        # run_time 有默认值回退+校验，compile_only 有三模式 prof 联动）
        self._run_time: Optional[int] = None
        self._compile_only: bool = False

    def _init_precision(self):
        """精度与数据：输入分布、golden 模式、比对方法、XPU 采集、dump。"""
        self.input_distribution: str = "uniform"
        self.golden_mode: str = "Enable"
        self.compare_method = None
        self.xpu_perf: bool = False
        self.xpu_zero_upload: bool = False
        # DumpConfig 实例由 bridge 注入（类型定义在 core_modules/common/dump——本模块不反向 import）
        self.dump_config = None

    def _init_mode_specific(self):
        """通路/模式专属：仅特定 test_mode 或 backend 消费（跨通路的已归
        execution：validate_only / deterministic_level）。"""
        # framework-api（E2E）
        self.force_cpu: bool = False
        self.fullgraph: int = 0
        self.aclgraph_enabled: bool = False
        # -b/--binary：三种模式（eager jitCompile / -c/-d ge.jit_compile / aclgraph
        # static_kernel_compile）统一二进制复用开关，默认开；-b=false 恢复在线编译
        self.e2e_binary: bool = True
        self.eager_enabled: bool = True
        self.super_kernel_enabled: bool = False
        self.core_limit: Optional[tuple] = None  # (ai_core_limit, vector_core_limit)，--core-limit 解析产物
        # GEIR
        self.geir_binary: bool = False
        # 手动数据（kernel/aclnn/e2e 三通路共用）
        self.no_prof: bool = False
        self.manual_data_dirs: Tuple[str, ...] = ()
        # NPUSim 仿真后端
        self.sim_soc_version: str = "Ascend950"
        self.sim_output_dir: str = ""  # 空 -> root_path/sim_output
        self.sim_report: bool = False
        self.sim_cores: str = ""
        self.sim_object_file: str = ""


class OPTestSwitch:
    """
    e.g. dynamic_shape, static_shape
    """

    REUSE_BINARY_RELEASE_KERNEL = "release"

    def __init__(self, name, switch, realtime_compilation, profiling):
        self.name = name
        self.enabled = switch
        self.realtime = realtime_compilation
        self.prof = profiling

    def __str__(self):
        return (
            f"{self.name}: "
            f"{'ENABLED' if self.enabled else 'DISABLED'}, "
            f"{'MANUAL_COMPILE' if not self.realtime else 'RELEASE' if self.realtime == self.REUSE_BINARY_RELEASE_KERNEL else 'TE_COMPILE'}, "
            f"{'ONLINE' if self.prof else 'OFFLINE'}"
        )

    def use_release_bin(self):
        return self.realtime == self.REUSE_BINARY_RELEASE_KERNEL


# ---- 全局存储兼容层 ----------------------------------------------------------
# get_global_storage()：读当前激活实例，未激活回退共享默认实例；正式 API 为
# SWITCHES.active()，本层为其兼容入口。
_default_storage: Optional["SWITCHES"] = None  # 惰性共享默认实例（get_global_storage 回退用）


def get_global_storage() -> "SWITCHES":
    """读当前激活实例；未激活回退共享默认实例（兼容语义，正身为 SWITCHES.active()）。"""
    if SWITCHES._active is not None:
        return SWITCHES._active
    global _default_storage
    if _default_storage is None:
        _default_storage = SWITCHES()
    return _default_storage


def set_global_storage(new_storage: Optional["SWITCHES"]):
    """激活实例（兼容语义，正身为 sw.activate()）；None 清除激活（回退默认实例）。"""
    SWITCHES._active = new_storage
