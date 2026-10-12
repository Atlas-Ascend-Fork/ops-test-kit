#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""ACLNN 通路 Instance。

DSMI 直连探测；建池前注入 OpApiInfoKeeper 缓存
快照：forkserver worker 不继承主进程内存，纯数据 keeper 的解析结果经引导
载荷随 worker 拉起传递，worker 内查询全为命中、零重复解析。
"""

import os
from typing import Any, List

from ttk.switches import get_global_storage

from ..common.comparison.compare_log import print_compare_log_failures, read_compare_log_failures
from ..common.device import DSMIInterface, get_npu_hw_info
from ..common.instance import InstanceBase, register_instance
from ..common.logging import COMPARE_LOG_FILE_NAME, build_single_log_dir
from ..common.process import SimpleCommandProcess, Task
from .result import AclnnReturnStructure
from .testcase import TestcaseAclnn
from .worker import run


@register_instance("aclnn")
class AclnnInstance(InstanceBase):
    """ACLNN 通路 Instance。"""

    name = "aclnn"
    testcase_class = TestcaseAclnn

    def prepare_env(self):
        opp_path = os.getenv("ASCEND_OPP_PATH", "")
        if not opp_path:
            raise RuntimeError("Install path of opp or compiler is not found. Please check Ascend installation.")

    def detect_device_count(self):
        if self.switches.no_prof and self.switches.manual_data_dirs or self.switches.mode.is_model():
            self.switches.device_count = 1
        elif self.switches.device_count <= 0:
            self.switches.device_count = DSMIInterface().get_device_count()

    def detect_platform(self):
        if self.switches.no_prof and self.switches.manual_data_dirs:
            if self.switches.dev_plat == "AUTO":
                self.switches.dev_plat = "CPU"
            self.switches.short_soc_version = "cpu"
            return
        if self.switches.dev_plat == "AUTO":
            try:
                self.switches.dev_plat = DSMIInterface().get_chip_info(0).get_complete_platform()
            except Exception as e:
                if self.switches.validate_only or self.switches.no_prof and self.switches.manual_data_dirs:
                    raise RuntimeError(
                        "Try to get Ascend platform failed. Please specify it with option like: --plat=Ascend910A"
                    ) from e
                raise
        hw_info = get_npu_hw_info(self.switches.dev_plat)
        self.switches.short_soc_version = hw_info.get("short_soc_version")
        os.environ["TTK_FULL_SOC_VERSION"] = self.switches.dev_plat
        os.environ["TTK_SHORT_SOC_VERSION"] = self.switches.short_soc_version

    def setup(self):
        if self.switches.mode.is_model():
            os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
        if (
            not self.switches.compile_only
            and not (self.switches.no_prof and self.switches.manual_data_dirs)
            and getattr(self.switches, "backend", None) != "npusim"
        ):
            self._compile_help_kernels()

    @staticmethod
    def _compile_help_kernels():
        # Helper kernels (warmup + UB/L1/L0 clear) feed the aclnn execution
        # chain's per-round hygiene; compiled through the shared operator
        # registry.
        from ..kernel.compile.compiler import get_compiler

        get_compiler().compile_ub_clear()
        get_compiler().compile_l1_clear()
        get_compiler().compile_l0_clear()
        get_compiler().compile_warmup_kernel()

    def prepare_workers(self):
        """Inject the keeper cache snapshot before worker pools are created.

        Testcase parsing has already filled the main-process cache; the
        snapshot travels with the forkserver bootstrap payload and is
        restored inside each worker.
        """
        from .info_keeper import snapshot_keeper

        SimpleCommandProcess._bootstrap_keeper_state = {"ttk.core_modules.aclnn.info_keeper": snapshot_keeper()}
        super().prepare_workers()

    def build_tasks(self, testcases: List[TestcaseAclnn]) -> List[Task]:
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        sorted_cases = sorted(testcases, key=lambda t: getattr(t, "_csv_row_index", -1))
        tasks = []
        for t in sorted_cases:
            device_ids = list(t.device_ids) if hasattr(t, "device_ids") and t.device_ids else None
            tasks.append(
                Task(
                    t,
                    run,
                    (t, grant_events, granted_indices),
                    task_type="profile",
                    pool="profile",
                    device_ids=device_ids,
                )
            )
        return tasks

    def handle_result(self, task: Task, result: Any) -> tuple:
        if not isinstance(result, AclnnReturnStructure):
            raise RuntimeError(f"Only AclnnReturnStructure is valid. But got {type(result)}")
        self._print_new_compare_failures(task.testcase)
        return result.pick_data(self.case_result_title)

    def result_titles(self) -> tuple:
        return AclnnReturnStructure.get_titles()

    def get_device_info(self, dev_id: int) -> str:
        return _dsmi_device_info(self.switches, dev_id)

    def _print_new_compare_failures(self, testcase) -> None:
        # COMPARE 级进单一 ttk.log（per-case 覆写）——按用例日志文件
        # 读 Index/Diff 行，失败明细随结果打屏。
        switches = get_global_storage()
        log_dir = build_single_log_dir(switches.test_mode, testcase.testcase_name, switches.root_path)
        diff_lines, _end = read_compare_log_failures(os.path.join(log_dir, COMPARE_LOG_FILE_NAME))
        print_compare_log_failures(diff_lines, testcase.testcase_name)


def _dsmi_device_info(switches, dev_id: int) -> str:
    """DSMI 设备信息行（进度打印用；与 kernel 通路同源的 DSMI 直连语义）。"""
    if switches.no_prof and switches.manual_data_dirs:
        return f"manual-data:{dev_id} {switches.dev_plat}"
    if not switches.mode.is_online_board() or switches.compile_only:
        phyid = str(dev_id)
        platform = switches.dev_plat
        chip_ver = "MODEL"
        health = "ERR"
        temperature = "???"

        aicore_util = "???"
        mem_util = "???"
        memband_util = "???"
        hbm_util = "???"
        ddr_util = "???"
        hbmband_util = "???"
        veccore_util = "???"

        mem_freq = "????"
        hbm_freq = "????"
        aicore_freq = "????"
        aicore_maxfreq = "????"
        veccore_freq = "????"
    else:
        dsmi = DSMIInterface()
        phyid = str(dsmi.get_physical_id_from_logical_id(dev_id) or dev_id)
        try:
            dsmi_platform = dsmi.get_chip_info(dev_id)
            platform = dsmi_platform.get_complete_platform()
            chip_ver = dsmi_platform.get_ver()
        except Exception:
            platform = "ERR"
            chip_ver = "ERR"

        try:
            health = DSMIInterface().get_device_health_state(dev_id).name
        except Exception:
            health = "ERR"

        temperature = str(DSMIInterface().get_device_temperature(dev_id) or "???")
        # Utils
        mem_util = str(DSMIInterface().get_device_util(dev_id, 1, platform)).replace("None", "ERR")
        aicore_util = str(dsmi.get_device_util(dev_id, 2, platform)).replace("None", "ERR")
        memband_util = str(DSMIInterface().get_device_util(dev_id, 5, platform)).replace("None", "ERR")
        hbm_util = str(DSMIInterface().get_device_util(dev_id, 6, platform)).replace("None", "ERR")
        ddr_util = str(DSMIInterface().get_device_util(dev_id, 8, platform)).replace("None", "ERR")
        hbmband_util = str(DSMIInterface().get_device_util(dev_id, 10, platform)).replace("None", "ERR")
        veccore_util = str(DSMIInterface().get_device_util(dev_id, 12, platform)).replace("None", "ERR")

        mem_freq = str(DSMIInterface().get_device_frequency(dev_id, 1, platform) or "????")
        hbm_freq = str(DSMIInterface().get_device_frequency(dev_id, 6, platform) or "????")
        aicore_freq = str(DSMIInterface().get_device_frequency(dev_id, 7, platform) or "????")
        aicore_maxfreq = str(DSMIInterface().get_device_frequency(dev_id, 9, platform) or "????")
        veccore_freq = str(DSMIInterface().get_device_frequency(dev_id, 12, platform) or "????")

    result = []

    def _add_info(info: str):
        if any(c.isdigit() for c in info):
            result.append(info)

    _add_info(f"{phyid.ljust(3)} {platform.ljust(13)} {chip_ver.ljust(2)} {temperature.ljust(2)}C {health.ljust(8)}")
    _add_info(f"AIC   {aicore_util.ljust(3)}% {aicore_freq.ljust(5)}Mhz / {aicore_maxfreq.ljust(5)}Mhz")
    _add_info(f"VEC   {veccore_util.ljust(3)}% {veccore_freq.ljust(5)}Mhz")
    _add_info(f"MEM   {mem_util.ljust(3)}% {mem_freq.ljust(5)}Mhz Bandwidth: {memband_util.ljust(3)}%")
    _add_info(f"DDR   {ddr_util.ljust(3)}%")
    _add_info(f"HBM   {hbm_util.ljust(3)}% {hbm_freq.ljust(5)}Mhz Bandwidth: {hbmband_util.ljust(3)}%")
    return "\n".join(result)
