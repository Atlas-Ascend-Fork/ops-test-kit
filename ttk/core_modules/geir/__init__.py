#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir 包根 —— GeirInstance 宿主（Instance 定义于 `__init__.py`）。

GeirInstance 基于 InstanceBase 契约（run() 模板 + hook），承担 GEIR 通路的
生命周期编排。
"""

__all__ = ["GeirInstance", "GeirTestcase"]

import logging
import os
from typing import Any, List

from ttk.core_modules.common.device import DSMIInterface, get_npu_hw_info
from ttk.core_modules.common.process import SimpleCommandProcess, Task
from ttk.utilities import cpu_count

from ..common.instance import InstanceBase, register_instance
from .result import GeirReturnStructure
from .testcase import GeirTestcase
from .worker import run


@register_instance("geir")
class GeirInstance(InstanceBase):
    """GEIR 通路 Instance（消解历史耦合，换新契约基座）。"""

    name = "geir"
    testcase_class = GeirTestcase

    # ═══ Abstract hooks（新契约 8 hook）═══

    def prepare_env(self):
        """环境检查：定位 Ascend 安装根（_find_ascend_root）；无安装根且无 ASCEND_OPP_PATH 时显式报错。"""
        from ttk._env import _find_ascend_root

        asc_path = _find_ascend_root()
        if not asc_path:
            opp_path = os.getenv("ASCEND_OPP_PATH", "")
            if not opp_path:
                raise RuntimeError("Install path of opp or compiler is not found. Please check Ascend installation.")

    def detect_platform(self):
        """SoC 探测：DSMI 首卡 complete_platform（失败回退 Ascend910B），写 TTK_*_SOC_VERSION 环境变量。"""
        if self.switches.dev_plat == "AUTO":
            try:
                self.switches.dev_plat = DSMIInterface().get_chip_info(0).get_complete_platform()
            except Exception:
                logging.warning("Cannot detect platform, using default.")
                self.switches.dev_plat = "Ascend910B"
        hw_info = get_npu_hw_info(self.switches.dev_plat)
        self.switches.short_soc_version = hw_info.get("short_soc_version", "")
        os.environ["TTK_FULL_SOC_VERSION"] = self.switches.dev_plat
        os.environ["TTK_SHORT_SOC_VERSION"] = self.switches.short_soc_version

    def detect_device_count(self):
        """设备数解析：DSMI 探测；无 NPU 时回退 CPU 数 min(max(cpu-1,1),4) 并告警。"""
        if self.switches.device_count <= 0:
            self.switches.device_count = DSMIInterface().get_device_count()
        if self.switches.device_count <= 0:
            self.switches.device_count = min(max(cpu_count() - 1, 1), 4)
            logging.warning(f"No NPU devices detected, using {self.switches.device_count} workers")

    def setup(self):
        """通路初始化 no-op：GEIR 无 KB server / 辅助 kernel 编译需求。

        设计所述"setup: ttk.log 偏移"经 logging 演进为 per-case 覆写布局后不再需要
        （compare_log 保留 start_size 机制作增量读兼容，见 common/comparison/compare_log.py）。"""

    def build_tasks(self, testcases: List[GeirTestcase]) -> List[Task]:
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        # GEIR mode: gen + compile + profile in one task (like E2E)
        tasks = []
        for testcase in testcases:
            if not testcase.is_valid:
                self.skipped_cases += 1
                continue
            tasks.append(
                Task(
                    testcase,
                    run,
                    (testcase, grant_events, granted_indices),
                    task_type="profile",
                    pool="profile",
                )
            )
        return tasks

    def handle_result(self, task: Task, result: Any) -> Any:
        """worker 结果 → 结果行 tuple（pick_data 单一实现；CST_OFF/DYN_OFF 语义在 result）；
        BatchSummary 的收集在 InstanceBase（_collect_batch_summary，先于本方法执行）。"""
        if isinstance(result, GeirReturnStructure):
            return result.pick_data(self.case_result_title)
        fallback = (str(result),) + ("",) * (len(self.case_result_title) - 1)
        return fallback

    def result_titles(self) -> tuple:
        is_binary = getattr(self.switches, "geir_binary", False)
        return GeirReturnStructure.get_titles(is_binary=is_binary)

    def get_device_info(self, dev_id: int) -> str:
        try:
            chip = DSMIInterface().get_chip_info(dev_id)
            platform = chip.get_complete_platform()
        except Exception:
            platform = "N/A"
        return f"NPU:{dev_id} {platform}"

    # ═══ 通用方法覆写 ═══

    def prepare_workers(self):
        """建池前注入 OpInfoKeeper 缓存快照（主进程读取一次，worker 直接使用）。

        kernel 生态 keeper（geir 经 _geir_param_order 消费 def.cpp 参数序）；
        forkserver worker 不继承主进程内存，解析期完成的全量 op_info 扫描经
        引导载荷随 worker 拉起传递——worker 查询全命中，零重复扫描（与
        kernel/aclnn 同机制，快照通道）。"""
        from ttk.core_modules.kernel.info_keeper import snapshot_keeper

        SimpleCommandProcess._bootstrap_keeper_state = {"ttk.core_modules.kernel.info_keeper": snapshot_keeper()}
        super().prepare_workers()
