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
common functions in current package
"""

# Standard Packages
import logging
import pathlib
import subprocess
from typing import Optional, Union

# Third-Party Packages
from ttk.switches import get_global_storage

from ....utilities.proc import kernel_debug_compile_enabled
from ...common.device import get_npu_hw_info

# cce 编译开关常量（公开名——operator/ 跨目录引用）
REGBASE_V2_SOURCE_DEBUG_OPTION = "--cce-ignore-always-inline=false"


def normalize_mode(mode: str):
    if not isinstance(mode, str) or len(mode) != 3:
        raise NotImplementedError(f"Mode name must be a string with length 3. But got {mode}")
    return mode[0].upper() + mode[1:].lower()


class CceManualCompile:
    @classmethod
    def compile(
        cls,
        kernel_name: str,
        kernel_main_func: str,
        platform: str,
        core_type: str,
        mix_task_ratio: Optional[tuple] = None,
    ) -> str:
        """
        CCEC Manual Compilation
        """
        if core_type != "MIX":
            return cls._compile_normal_kernel(kernel_name, platform, core_type)
        return cls._compile_mix_kernel(kernel_name, kernel_main_func, platform, mix_task_ratio)

    @classmethod
    def _compile_cce(
        cls,
        cce_file: str,
        i_obj_file: str,
        platform: str,
        core_type: str,
        extra_compile_options: Optional[Union[tuple, list, str]] = None,
    ):
        ccec_cmd = [
            "ccec",
            "-O2",
            cce_file,
            f"--cce-aicore-arch={cls._get_arch_for_ccec(platform, core_type)}",
            "--cce-aicore-only",
        ]
        # Source-level debug info. Driven by --compile-opts op_debug_config=ccec_O0,ccec_g
        # (parsed into compile_options). Under msdebug the debug flags are auto-enabled
        # so breakpoints resolve to source lines.
        debug_config = get_global_storage().compile_options.get("op_debug_config", "")
        debug_compile = kernel_debug_compile_enabled()
        if debug_compile or "ccec_O0" in debug_config or "ccec_g" in debug_config:
            if "ccec_O0" in debug_config or debug_compile:
                ccec_cmd.append("-O0")
            if "ccec_g" in debug_config or debug_compile:
                ccec_cmd.append("-g")
        short_soc_version = get_npu_hw_info(platform).get("short_soc_version")
        if short_soc_version == "Ascend950" and "-O0" in ccec_cmd and "-g" in ccec_cmd:
            ccec_cmd.append(REGBASE_V2_SOURCE_DEBUG_OPTION)
        if extra_compile_options:
            if not isinstance(extra_compile_options, (tuple, list)):
                extra_compile_options = [extra_compile_options]
            ccec_cmd.extend(extra_compile_options)
        ccec_cmd.extend(
            ["-o", i_obj_file, "-mllvm", "--cce-aicore-jump-expand=true", "-mllvm", "-cce-aicore-addr-transform"]
        )
        if short_soc_version not in ("Ascend950", "MC62CM12A"):
            ccec_cmd.extend(
                ["-mllvm", "-cce-aicore-function-stack-size=16000", "-mllvm", "-cce-aicore-record-overflow=true"]
            )
        else:
            ccec_cmd.extend(
                [
                    "-mllvm",
                    "-cce-aicore-stack-size=0x8000",
                    "-mllvm",
                    "-cce-aicore-function-stack-size=0x8000",
                    "-mllvm",
                    "-cce-aicore-record-overflow=false",
                    "-mllvm",
                    "-cce-aicore-dcci-insert-for-scalar=false",
                    "-mllvm",
                    "-cce-aicore-dcci-before-kernel-end=false",
                ]
            )
        ccec = subprocess.Popen(ccec_cmd, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ccec_return = ccec.communicate()[1].decode("UTF-8")
        if ccec.returncode != 0:
            logging.error(f"CCEC Compilation Failure {ccec.returncode}: {ccec_return}")
            return ccec_return
        return "SUCC"

    @classmethod
    def _ld(cls, obj_file: str, i_obj_files: Union[list, tuple, str]) -> str:
        if not isinstance(i_obj_files, (tuple, list)):
            i_obj_files = [i_obj_files]
        ld_cmd = ["ld.lld", "-m", "aicorelinux", "-Ttext=0"]
        ld_cmd.extend(i_obj_files)
        ld_cmd.extend(["-static", "-o", obj_file])
        ld = subprocess.Popen(ld_cmd, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ld_return = ld.communicate()[1].decode("UTF-8")
        if ld.returncode != 0:
            logging.error(f"CCEC-LD-LLD Compilation Failure {ld.returncode}: {ld_return}")
            return ld_return
        return "SUCC"

    @classmethod
    def _compile_normal_kernel(cls, kernel_name: str, platform: str, core_type: str) -> str:
        cce_file = f"{kernel_name}.cce"
        i_obj_file = f"{kernel_name}.i"
        obj_file = f"{kernel_name}.o"

        if not pathlib.Path(cce_file).is_file():
            if pathlib.Path(obj_file).is_file():
                return "SUCC"
            return "FOUND_NOTHING"

        ret = cls._compile_cce(cce_file, i_obj_file, platform, core_type)
        if ret != "SUCC":
            return ret
        return cls._ld(obj_file, i_obj_file)

    @classmethod
    def _compile_mix_kernel(cls, kernel_name: str, kernel_main_func: str, platform: str, mix_task_ratio: tuple) -> str:
        obj_file = f"{kernel_name}.o"
        if all(mix_task_ratio):
            aic_cce_file, aiv_cce_file = f"{kernel_name}_mix_aic.cce", f"{kernel_name}_mix_aiv.cce"
            aic_i_obj_file, aiv_i_obj_file = f"{kernel_name}_mix_aic.i", f"{kernel_name}_mix_aiv.i"

            if any(not pathlib.Path(x).is_file() for x in [aic_cce_file, aiv_cce_file]):
                if pathlib.Path(obj_file).is_file():
                    return "SUCC"
                return "FOUND_NOTHING"

            ret = cls._compile_cce(
                aic_cce_file, aic_i_obj_file, platform, "AiCore", f"-D{kernel_main_func}={kernel_main_func}_mix_aic"
            )
            if ret != "SUCC":
                return ret
            ret = cls._compile_cce(
                aiv_cce_file, aiv_i_obj_file, platform, "VectorCore", f"-D{kernel_main_func}={kernel_main_func}_mix_aiv"
            )
            if ret != "SUCC":
                return ret
            return cls._ld(obj_file, [aic_i_obj_file, aiv_i_obj_file])
        core_type, suffix = ("AiCore", "aic") if mix_task_ratio[0] > 0 else ("VectorCore", "aiv")
        cce_file, i_obj_file = f"{kernel_name}.cce", f"{kernel_name}.i"

        if not pathlib.Path(cce_file).is_file():
            if pathlib.Path(obj_file).is_file():
                return "SUCC"
            return "FOUND_NOTHING"
        ret = cls._compile_cce(
            cce_file, i_obj_file, platform, core_type, f"-D{kernel_main_func}={kernel_main_func}_mix_{suffix}"
        )
        if ret != "SUCC":
            return ret
        return cls._ld(obj_file, i_obj_file)

    @classmethod
    def _get_arch_for_ccec(cls, platform_name: str, core_type: str = "VectorCore"):
        """
        Get architecture abbreviation for ccec by platform
        """
        if core_type == "VectorCore":
            core_type = "vec"
        elif core_type == "AiCore":
            core_type = "cube"
        else:
            core_type = "cube"

        hw_info = get_npu_hw_info(platform_name)
        arch = hw_info.get("ccec_aic_version", "")
        return arch.replace("$core_type", core_type)
