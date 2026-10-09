#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Precious Utility Classes
"""

# Standard Packages
import copy
import json
import os
import pathlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy

if TYPE_CHECKING:
    from ..core_modules.testcase_manager.testcase_op import TestcaseOp

# Third-party Packages


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
        return self in [MODE.ASCEND_ONBOARD]

    def has_device(self) -> bool:
        return self.is_online_board()

    def is_esl_model(self):
        return str(self.is_model()) == "ESLMODEL"


class DumpLevel(Enum):
    """Dump data level"""

    NO = 0b000
    INPUT = 0b100
    OUTPUT = 0b010
    GOLDEN = 0b001
    FULL = 0b111


@dataclass
class DumpConfig:
    mode: int = DumpLevel.NO.value
    file_format: str = "bin"
    dump_on_fail: bool = False
    xpu: bool = False

    def is_input_enabled(self):
        return self.mode & DumpLevel.INPUT.value

    def is_output_enabled(self):
        return self.mode & DumpLevel.OUTPUT.value

    def is_golden_enabled(self):
        return self.mode & DumpLevel.GOLDEN.value

    def is_xpu_enabled(self):
        return self.xpu

    def enable_input(self):
        self.mode |= DumpLevel.INPUT.value

    def enable_output(self):
        self.mode |= DumpLevel.OUTPUT.value

    def enable_golden(self):
        self.mode |= DumpLevel.GOLDEN.value

    def enable_xpu(self):
        self.xpu = True

    def enable_all(self):
        self.mode = DumpLevel.FULL.value


@dataclass
class SoCSimtCfg:
    # simt stack size configuration per-process in uint: bytes.
    dcu_stack: Optional[int] = None
    dvg_stack: Optional[int] = None


class SWITCHES:
    """
    Control Panel
    """

    __slots__ = [
        "root_path",
        "mode",
        "input_files",
        "sheet",
        "output_file_name",
        "append_mode",
        "logging_to_file",
        "single_testcase_log_mode",
        "dev_plat",
        "short_soc_version",
        "custom_columns",
        "print_help",
        "process_per_device",
        "dyn_switches",
        "cst_switches",
        "bin_switches",
        "rerun_targets",
        "TASK_PROFILING",
        "dump_config",
        "device_count",
        "device_blacklist",
        "device_whitelist",
        "run_timeout",
        "proc_timeout",
        "tiling_run_time",
        "kernel_meta",
        "warmup",
        "summary_print",
        "DAVINCI_HBM_SIZE_LIMIT",
        "selected_testcases",
        "selected_testcase_indexes",
        "selected_testcase_count",
        "selected_operators",
        "excluded_operators",
        "preserve_original_csv",
        "random_seed",
        "no_memory_check",
        "force_clear_atomic",
        "force_block_dim",
        "core_limit",
        "super_kernel_enabled",
        "force_clear_ub",
        "force_clear_l1",
        "force_clear_l0",
        "force_simt_ub_size",
        "progress_output",
        "proc_no_reuse",
        "op_impl_mode",
        "simt_cfg",
        "input_distribution",
        "golden_mode",
        "compare_method",
        "xpu_perf",
        "xpu_zero_upload",
        "geir_serve",
        "precision_report",
        "reuse_hbm",
        "reserve_hbm",
        "priorities",
        "compile_options",
        "plugin_path",
        "test_mode",
        "force_cpu",
        "fullgraph",
        "aclgraph_enabled",
        "aclgraph_online_compile",
        "validate_only",
        "manual_data_mode",
        "manual_data_dirs",
        # private properties
        "_run_time",
        "_compile_only",
        "config_path",  # NEW: --config CLI 值（yaml 路径），经 SWITCHES pickle 传 worker
        "provider_filter",  # NEW: --provider CLI 值（provider 过滤器），经 SWITCHES pickle 传 worker
        # GEIR mode
        "geir_binary",
        "deterministic_level",
        # NPUSim simulator backend
        "backend",
        "sim_soc_version",
        "sim_output_dir",
        "sim_report",
        "sim_cores",
        "sim_object_file",
        "framework",
    ]

    def __init__(self):
        self._init_paths_and_mode()
        self._init_device_and_run()
        self._init_testcases_and_filters()
        self._init_modes_and_backend()

    def __getstate__(self):
        """Pickle 支持：仅导出 __slots__ 中已赋值的属性（跳过 property/私有）。"""
        return {k: getattr(self, k, None) for k in self.__slots__ if hasattr(self, k)}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def overflow_mode(self) -> int:
        try:
            from .platform import get_npu_hw_info

            hw_info = get_npu_hw_info(self.dev_plat)
            support_bf16 = hw_info.get("support_bf16", False)
            return 1 if support_bf16 else 0
        except Exception:
            return 1

    @property
    def run_time(self) -> int:
        if self.mode.is_model():
            return self._run_time or 1
        return self._run_time or 3

    @run_time.setter
    def run_time(self, val):
        if not isinstance(val, int) or val <= 0:
            raise RuntimeError(f"You must specify a int value for run_time. But got {val}")
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

    def _init_paths_and_mode(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.mode: MODE = MODE.ASCEND_ONBOARD
        self.input_files: Optional[List[str]] = None
        self.sheet: Optional[str] = None
        self.output_file_name: Optional[str] = None
        self.append_mode: bool = False
        self.logging_to_file: bool = False
        self.single_testcase_log_mode = False
        self.dev_plat: str = "AUTO"
        self.short_soc_version: Optional[str] = None  # None in non-NPU mode
        self.custom_columns = None
        self.print_help: bool = False
        self.process_per_device = None
        self.dyn_switches: OPTestSwitch = OPTestSwitch("Dynamic Shape", True, True, True)
        self.cst_switches: OPTestSwitch = OPTestSwitch("Const Shape", False, True, True)
        self.bin_switches: OPTestSwitch = OPTestSwitch("Binary Release", False, True, True)
        self.rerun_targets = None
        self.TASK_PROFILING = True
        self.dump_config = DumpConfig()

    def _init_device_and_run(self):
        self.device_count = -1
        self.device_blacklist = []
        self.device_whitelist = []
        self.run_timeout = 0
        self.proc_timeout = 3600
        self.tiling_run_time = 3
        self.no_memory_check = False
        self.force_clear_atomic = [None, None, None]
        self.force_block_dim = [None, None, None]
        self.core_limit: Optional[Tuple[Optional[int], Optional[int]]] = None
        self.super_kernel_enabled: bool = False
        self.force_clear_ub = None
        self.force_clear_l1 = None
        self.force_clear_l0 = None
        self.force_simt_ub_size = [None, None, None]  # dynamic/const/binary
        self.proc_no_reuse = False
        # Hidden switches
        self.kernel_meta = os.path.join(self.root_path, "kernel_meta")
        self.warmup = True
        self.summary_print = True
        # Constants
        self.DAVINCI_HBM_SIZE_LIMIT = 30  # GB

    def _init_testcases_and_filters(self):
        self.selected_testcases = []
        self.selected_testcase_indexes = []
        self.selected_testcase_count = -1
        self.selected_operators = None
        self.excluded_operators = None
        self.preserve_original_csv = False
        self.random_seed = None
        self.progress_output = None
        self.op_impl_mode: Optional[str] = None
        self.simt_cfg: SoCSimtCfg = SoCSimtCfg()
        self.input_distribution: str = "uniform"
        self.golden_mode: str = "Enable"
        self.compare_method = None
        self.xpu_perf: bool = False
        self.xpu_zero_upload: bool = False
        self.geir_serve: bool = False
        self.precision_report: Optional[str] = None
        self.reuse_hbm: bool = False
        self.reserve_hbm: int = 0
        self.priorities: Optional[tuple] = None
        self.compile_options: dict = {}
        self.plugin_path: Optional[Tuple[pathlib.Path]] = None

    def _init_modes_and_backend(self):
        self.test_mode: str = "op"
        self.force_cpu: bool = False
        self.fullgraph: int = 0
        self.aclgraph_enabled: bool = False
        self.aclgraph_online_compile: bool = True
        self.validate_only: bool = False
        self.manual_data_mode: Optional[str] = None
        self.manual_data_dirs: Tuple[str, ...] = ()
        # private properties below
        self._run_time: Optional[int] = None
        self._compile_only: bool = False
        self.config_path: Optional[str] = None
        self.provider_filter: Optional[str] = None
        # GEIR mode
        self.geir_binary: bool = False
        self.deterministic_level: int = 0
        # NPUSim simulator backend
        self.backend: str = "npu"  # "npu" | "npusim"
        self.sim_soc_version: str = "Ascend950"
        self.sim_output_dir: str = ""  # 空 -> root_path/sim_output
        self.sim_report: bool = False
        self.sim_cores: str = ""
        self.sim_object_file: str = ""
        self.framework: str = "torch"


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


@dataclass
class SubKernelJsonInfo:
    kernel_name: str
    parameters: Optional[tuple] = None
    magic: Optional[str] = None
    core_type: Optional[str] = None
    task_ration: Optional[tuple] = None  # aic:aiv

    @classmethod
    def from_dict(cls, json_dict: dict):
        parameters = json_dict.get("parameters")
        if parameters is not None:
            parameters = tuple(parameters)
        magic = json_dict.get("magic")
        core_type = json_dict.get("coreType")
        kernel_name = json_dict["kernelName"]
        task_ration = json_dict.get("taskRation")
        if task_ration is not None:
            if not isinstance(task_ration, str) or ":" not in task_ration:
                raise ValueError(f"task_ration [{task_ration}] is invalid. It may be a bug of compiler.")
            task_ration = tuple(int(i) for i in task_ration.split(":"))
        return SubKernelJsonInfo(kernel_name, parameters, magic, core_type, task_ration)


@dataclass
class KernelJsonInfo:
    """info parsed from ***.json after compile completed"""

    block_dim: int = 0
    workspaces: Optional[tuple] = None
    parameters: Optional[tuple] = None
    magic: Optional[str] = None
    global_workspace_size: int = 0
    kernel_name: Optional[str] = None
    core_type: str = "AiCore"
    task_ration: Optional[tuple] = None  # aic:aiv
    inter_core_sync: bool = False
    op_original_para_size: int = 0
    optional_input_mode: str = "no_placeholder"
    optional_output_mode: str = "no_placeholder"
    dynamic_param_mode: str = "unfolded"
    local_memory_size: int = 0
    sub_kernels: Optional[Dict[str, SubKernelJsonInfo]] = None
    debug_options: Tuple[str] = field(default_factory=tuple)
    debug_buf_size: int = 0
    schedule_mode: int = 0
    op_debug_config: Tuple[str] = field(default_factory=tuple)
    _sub_kernel_cache: Dict[str, "KernelJsonInfo"] = field(init=False, default_factory=dict)
    _is_fat_bin: bool = field(init=False, default=False)

    def __post_init__(self):
        self._is_fat_bin = self.sub_kernels and len(self.sub_kernels) > 0
        if self.sub_kernels and self.kernel_name in self.sub_kernels:
            # migrate from sub kernel
            sk = self.sub_kernels[self.kernel_name]
            self._migrate(sk, self)

    @property
    def clear_atomic(self) -> bool:
        return self.parameters and (1 in self.parameters or any(self.parameters))

    @property
    def is_mix_kernel(self) -> bool:
        return self.core_type == "MIX" or self.inter_core_sync

    @property
    def is_fat_bin(self) -> bool:
        return self._is_fat_bin

    @classmethod
    def from_file(cls, json_file: Union[str, pathlib.Path]):
        """
        Get operator compile info from json
        :param json_file:
        :return:
        """
        with open(json_file, encoding="UTF-8") as f:
            raw_json_data = f.read()
            # noinspection PyBroadException
            try:
                json_data = json.loads(raw_json_data)
            except Exception as err:
                raise RuntimeError(f"Json read failure, received json:\n{raw_json_data}") from err
        return cls.from_dict(json_data)

    @classmethod
    def from_dict(cls, json_dict: dict):
        block_dim = int(json_dict["blockDim"])
        workspaces = tuple(json_dict["workspace"]["size"]) if "workspace" in json_dict else ()
        parameters = tuple(json_dict.get("parameters", ()))
        magic = json_dict.get("magic", "RT_DEV_BINARY_MAGIC_ELF")
        global_workspace_size = int(
            json_dict["globalworkspace_spec_workspace"]["size"] if "globalworkspace_spec_workspace" in json_dict else 0
        )
        core_type = json_dict.get("coreType", "AiCore")
        kernel_name = json_dict["kernelName"]
        sub_kernels = {}
        if "kernelList" in json_dict and len(json_dict["kernelList"]) > 0:
            for k in json_dict["kernelList"]:
                sub_kernel_name = k["kernelName"]
                sub_kernels[sub_kernel_name] = SubKernelJsonInfo.from_dict(k)
        task_ration = json_dict.get("taskRation")
        if task_ration is not None:
            if not isinstance(task_ration, str) or (":" not in task_ration and task_ration != "tilingKey"):
                raise ValueError(f"task_ration [{task_ration}] is invalid. It may be a bug of compiler.")
            task_ration = tuple(int(i) for i in task_ration.split(":")) if task_ration != "tilingKey" else ()
        inter_core_sync = bool(json_dict.get("interCoreSync", False))
        op_original_para_size = int(json_dict.get("oriOpParaSize", 0))
        optional_input_mode = json_dict.get("optionalInputMode", "no_placeholder")
        optional_output_mode = json_dict.get("optionalOutputMode", "no_placeholder")
        dynamic_param_mode = json_dict.get("dynamicParamMode", "unfolded")
        local_memory_size = int(numpy.int32(json_dict.get("localMemorySize", -1)))
        debug_options = tuple(x.strip() for x in json_dict.get("debugOptions", "").lower().split(","))
        debug_buf_size = int(json_dict.get("debugBufSize", 0))
        schedule_mode = int(json_dict.get("schedule_mode", 0))
        support_info = json_dict.get("supportInfo", {})
        # only binary released kernel has this option in json.
        op_debug_config = tuple(x.strip() for x in support_info.get("op_debug_config", "").lower().split(","))
        return KernelJsonInfo(
            block_dim,
            workspaces,
            parameters,
            magic,
            global_workspace_size,
            kernel_name,
            core_type,
            task_ration,
            inter_core_sync,
            op_original_para_size,
            optional_input_mode,
            optional_output_mode,
            dynamic_param_mode,
            local_memory_size,
            sub_kernels,
            debug_options,
            debug_buf_size,
            schedule_mode,
            op_debug_config,
        )

    @staticmethod
    def _migrate(_sk: SubKernelJsonInfo, _target: "KernelJsonInfo"):
        def _migrate_attr(_attr: str):
            v = getattr(_sk, _attr)
            if v is not None:
                setattr(_target, _attr, v)

        for attr in ("parameters", "magic", "core_type", "task_ration", "kernel_name"):
            _migrate_attr(attr)
        # reset inter core sync. mix based on core_type in sub kernel only.
        _target.inter_core_sync = False
        _target.sub_kernels = None

    def sub_kernel_json_info(self, tiling_key: Optional[int] = None) -> "KernelJsonInfo":
        if tiling_key is None or not self.is_fat_bin or not self.sub_kernels:
            return self
        sub_kernel_name = f"{self.kernel_name}_{tiling_key}"
        if sub_kernel_name not in self.sub_kernels:
            return self
        if sub_kernel_name in self._sub_kernel_cache:
            return self._sub_kernel_cache[sub_kernel_name]
        sk = self.sub_kernels[sub_kernel_name]
        ret = copy.deepcopy(self)
        self._migrate(sk, ret)
        self._sub_kernel_cache[sub_kernel_name] = ret
        return ret

    def dynamic_param_is_folded(self):
        return self.dynamic_param_mode == "folded_with_desc"

    def optional_input_gen_placeholder(self):
        return self.optional_input_mode == "gen_placeholder"

    def optional_output_gen_placeholder(self):
        return self.optional_output_mode == "gen_placeholder"

    def printf_enabled(self) -> bool:
        return "printf" in self.debug_options

    def assert_enabled(self) -> bool:
        return "assert" in self.debug_options

    def oom_enabled(self) -> bool:
        return "oom" in self.op_debug_config


@dataclass
class BaseCompilationResult:
    """Compilation Result Base"""

    compile_result: Optional[str] = None
    compile_time: Optional[Union[str, float]] = None
    func_params: Optional[tuple] = None  # parameters of the op implement
    kernel_name: Optional[str] = None
    kernel_dir: Optional[str] = None
    _kernel_json_info: Optional[KernelJsonInfo] = None

    def all_set(self, value: Optional[str]):
        """Set all values"""
        self.compile_result = value
        self.compile_time = value
        self.func_params = None
        self.kernel_name = value
        self.kernel_dir = None
        self._kernel_json_info = None

    def base_standard_set(
        self,
        compile_result: Optional[str],
        compile_time: Optional[Union[str, float]],
        func_params: Optional[tuple],
        kernel_json_info: Optional[KernelJsonInfo],
        kernel_name: str,
        kernel_dir: Optional[str] = None,
    ):
        """Set standard value"""
        self.compile_result = compile_result
        self.compile_time = compile_time
        self.func_params = func_params
        self._kernel_json_info = kernel_json_info
        self.kernel_name = kernel_name
        self.kernel_dir = kernel_dir

    def base_standard_get(self):
        """Get standard value"""
        return (
            self.compile_result,
            self.compile_time,
            self.func_params,
            self._kernel_json_info,
            self.kernel_name,
            self.kernel_dir,
        )

    def get_json(self):
        json_parsed = {"func_params": self.func_params, "kernel_name": self.kernel_name}
        if self.kernel_dir:
            json_parsed["kernel_dir"] = self.kernel_dir
        return json_parsed

    def apply(self, testcase: "TestcaseOp"):
        pass

    def printf_enabled(self) -> bool:
        return False if not self._kernel_json_info else self._kernel_json_info.printf_enabled()

    def assert_enabled(self) -> bool:
        return False if not self._kernel_json_info else self._kernel_json_info.assert_enabled()

    def compile_success(self):
        return self.compile_result == "SUCC"

    @property
    def block_dim(self) -> int:
        return 0 if not self._kernel_json_info else self._kernel_json_info.block_dim

    @block_dim.setter
    def block_dim(self, value):
        if self._kernel_json_info is not None:
            self._kernel_json_info.block_dim = value

    @property
    def workspaces(self) -> Optional[tuple]:
        return None if not self._kernel_json_info else self._kernel_json_info.workspaces

    @workspaces.setter
    def workspaces(self, value):
        if self._kernel_json_info is not None:
            self._kernel_json_info.workspaces = value

    @property
    def tiling_key(self) -> Optional[int]:
        return None

    @tiling_key.setter
    def tiling_key(self, value):
        pass

    @property
    def tiling_data(self):
        return None

    @tiling_data.setter
    def tiling_data(self, value):
        pass

    @property
    def kernel_json_info(self) -> Optional[KernelJsonInfo]:
        return None if not self._kernel_json_info else self._kernel_json_info.sub_kernel_json_info(self.tiling_key)

    @kernel_json_info.setter
    def kernel_json_info(self, value):
        self._kernel_json_info = value

    @property
    def simt_ub_size(self) -> int:
        return 0 if not self._kernel_json_info else self._kernel_json_info.local_memory_size

    @simt_ub_size.setter
    def simt_ub_size(self, value: int):
        if self._kernel_json_info:
            self._kernel_json_info.local_memory_size = value

    @property
    def debug_buf_size(self) -> int:
        return 0 if not self._kernel_json_info else self._kernel_json_info.debug_buf_size


class DynamicCompilationResult(BaseCompilationResult):
    """For Dynamic Compilation"""

    def __init__(self):
        super().__init__()
        self.tiling_result: Optional[DynamicOpTilingResult] = None
        self.compile_info: Optional[dict] = None
        self.tiling_op_type: Optional[str] = None

    def all_set(self, value: Optional[str]):
        """Set all values"""
        super().all_set(value)
        self.compile_info = {}
        self.tiling_op_type = value

    def standard_set(
        self,
        compile_info: Optional[dict],
        tiling_op_type: Optional[str],
        # below for base class
        compile_result: Optional[str],
        compile_time: Optional[Union[str, float]],
        func_params: Optional[tuple],
        kernel_json_info: Optional[KernelJsonInfo],
        kernel_name: str,
        kernel_dir: Optional[str] = None,
    ):
        """Set standard value"""
        self.compile_info = compile_info
        self.tiling_op_type = tiling_op_type
        super().base_standard_set(compile_result, compile_time, func_params, kernel_json_info, kernel_name, kernel_dir)

    def write_json(self, path: Optional[str], override_kernel_name=None):
        """Write compile info json"""
        json_parsed = super().get_json()
        json_parsed.update({"compile_info": self.compile_info, "tiling_op_type": self.tiling_op_type})
        with open(
            pathlib.Path(path, "%s.ttk" % (self.kernel_name if override_kernel_name is None else override_kernel_name)),
            "w+",
            encoding="UTF-8",
        ) as json_file:
            json_file.write(json.dumps(json_parsed, indent=4))

    def standard_get(self):
        """Get standard value"""
        return (self.compile_info, self.tiling_op_type, *self.base_standard_get())

    def apply(self, testcase: "TestcaseOp"):
        """Apply dynamic result to testcase"""
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.func_params
        testcase.dyn_compile_result = self
        testcase.compile_done += 1

    @property
    def block_dim(self) -> int:
        if not self.tiling_result or not self.tiling_result.block_dim:
            return 0
        return self.tiling_result.block_dim

    @block_dim.setter
    def block_dim(self, value):
        if self.tiling_result:
            self.tiling_result.block_dim = value

    @property
    def workspaces(self) -> Optional[tuple]:
        return None if not self.tiling_result else self.tiling_result.workspaces

    @workspaces.setter
    def workspaces(self, value):
        if self.tiling_result:
            self.tiling_result.workspaces = value

    @property
    def tiling_key(self) -> Optional[int]:
        return None if not self.tiling_result else self.tiling_result.tiling_key

    @tiling_key.setter
    def tiling_key(self, value):
        if self.tiling_result:
            self.tiling_result.tiling_key = value

    @property
    def tiling_data(self):
        return None if not self.tiling_result else self.tiling_result.tiling_data

    @tiling_data.setter
    def tiling_data(self, value):
        if self.tiling_result:
            self.tiling_result.tiling_data = value

    @property
    def simt_ub_size(self) -> int:
        return 0 if not self.tiling_result else self.tiling_result.local_memory_size

    @simt_ub_size.setter
    def simt_ub_size(self, value: int):
        if self.tiling_result:
            self.tiling_result.local_memory_size = value


class StaticCompilationResult(BaseCompilationResult):
    """For Static Compilation"""

    def standard_set(
        self,
        # below for base class
        compile_result: Optional[str],
        compile_time: Optional[Union[str, float]],
        func_params: Optional[tuple],
        kernel_json_info: Optional[KernelJsonInfo],
        kernel_name: str,
        kernel_dir: Optional[str] = None,
    ):
        """Set standard value"""
        super().base_standard_set(compile_result, compile_time, func_params, kernel_json_info, kernel_name, kernel_dir)

    def standard_get(self):
        """Get standard value"""
        return self.base_standard_get()

    def apply(self, testcase: "TestcaseOp"):
        pass

    def write_json(self, path: Optional[str]):
        """Write compile info json"""
        json_parsed = super().get_json()
        with open(pathlib.Path(path, f"{self.kernel_name}.ttk"), "w+", encoding="UTF-8") as f:
            f.write(json.dumps(json_parsed, indent=4))


class ConstCompilationResult(StaticCompilationResult):
    """For const Compilation"""

    def apply(self, testcase: "TestcaseOp"):
        """Apply result to testcase"""
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.func_params
        testcase.cst_compile_result = self
        testcase.compile_done += 1


class BinaryCompilationResult(DynamicCompilationResult):
    """For Binary Compilation"""

    def apply(self, testcase: "TestcaseOp"):
        """Apply result to testcase"""
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.func_params
        testcase.bin_compile_result = self
        testcase.compile_done += 1


class DynamicOpTilingResult:
    """For Dynamic Op-tiling"""

    def __init__(self):
        self.block_dim: Optional[int] = None
        self.tiling_key: Optional[int] = None
        self.tiling_data: Optional[bytes] = None
        self.workspaces: Optional[tuple] = None
        self.tiling_time: Optional[Union[str, tuple]] = None
        self.local_memory_size: int = 0

    def standard_set(
        self,
        block_dim: Optional[int],
        tiling_key: Optional[int],
        tiling_data: Optional[bytes],
        workspaces: Optional[tuple],
        tiling_time: Optional[tuple],
        local_memory_size: int = 0,
    ):
        """Set standard value"""
        self.block_dim = block_dim
        self.tiling_key = tiling_key
        self.tiling_data = tiling_data
        self.workspaces = workspaces
        self.tiling_time = tiling_time
        self.local_memory_size = local_memory_size

    def all_set(self, value: Optional[str]):
        """Set standard value"""
        self.block_dim = 0
        self.tiling_key = value
        self.tiling_data = None
        self.workspaces = None
        self.tiling_time = value
        self.local_memory_size = 0


Mode2CompilationResultMap = {
    "Dyn": DynamicCompilationResult,
    "Cst": ConstCompilationResult,
    "Bin": BinaryCompilationResult,
}


def compilation_result(
    mode: str, valid_mode: List[str] = None
) -> Union[BinaryCompilationResult, DynamicCompilationResult, ConstCompilationResult, StaticCompilationResult]:
    if (valid_mode and mode not in valid_mode) or mode not in Mode2CompilationResultMap:
        raise NotImplementedError()
    return Mode2CompilationResultMap[mode]()


def construct_crash_compilation_result(crash_info, mode):
    result = compilation_result(mode)
    result.all_set(crash_info)
    if mode in ("Dyn", "Bin"):
        result.tiling_result = DynamicOpTilingResult()
        result.tiling_result.all_set(crash_info)
    return result
