#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""kernel/compile/result.py：编译结果结构族。

SubKernelJsonInfo / KernelJsonInfo + BaseCompilationResult 族（Dynamic/
Static/Const/Binary）+ DynamicOpTilingResult + compilation_result() 工厂 +
construct_crash_compilation_result。"""

import copy
import json
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy

if TYPE_CHECKING:
    from ..testcase import TestcaseKernel


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

    def apply(self, testcase: "TestcaseKernel"):
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

    def apply(self, testcase: "TestcaseKernel"):
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

    def apply(self, testcase: "TestcaseKernel"):
        pass

    def write_json(self, path: Optional[str]):
        """Write compile info json"""
        json_parsed = super().get_json()
        with open(pathlib.Path(path, f"{self.kernel_name}.ttk"), "w+", encoding="UTF-8") as f:
            f.write(json.dumps(json_parsed, indent=4))


class ConstCompilationResult(StaticCompilationResult):
    """For const Compilation"""

    def apply(self, testcase: "TestcaseKernel"):
        """Apply result to testcase"""
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.func_params
        testcase.cst_compile_result = self
        testcase.compile_done += 1


class BinaryCompilationResult(DynamicCompilationResult):
    """For Binary Compilation"""

    def apply(self, testcase: "TestcaseKernel"):
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
        raise NotImplementedError(
            f"Unknown compilation mode {mode!r}. Valid modes: {sorted(Mode2CompilationResultMap)}"
            + (f" (allowed here: {sorted(valid_mode)})" if valid_mode else "")
        )
    return Mode2CompilationResultMap[mode]()


def construct_crash_compilation_result(crash_info, mode):
    result = compilation_result(mode)
    result.all_set(crash_info)
    if mode in ("Dyn", "Bin"):
        result.tiling_result = DynamicOpTilingResult()
        result.tiling_result.all_set(crash_info)
    return result
