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
FrameworkApiInstance — InstanceBase implementation for framework_api tests.
"""

import logging
import os
from typing import Any, List

from ttk.switches import get_global_storage

from ..common.backend import get_backend
from ..common.comparison.compare_log import (
    print_compare_log_failures,
    print_structured_compare_failures,
    read_compare_log_failures,
)
from ..common.instance import InstanceBase, execution_sorted, register_instance
from ..common.logging import COMPARE_LOG_FILE_NAME, build_single_log_dir
from ..common.process import SimpleCommandProcess, Task
from .result import FrameworkApiReturnStructure
from .testcase import TestcaseE2e
from .worker import run


@register_instance("framework-api")
class FrameworkApiInstance(InstanceBase):
    """Entry instance for framework-level API testing."""

    name = "framework-api"
    testcase_class = TestcaseE2e

    def __init__(self):
        super().__init__()
        switches = get_global_storage()
        if switches.backend == "npusim":
            self._inject_camodel_env(switches)
        framework = getattr(switches, "framework", "torch")
        self.backend = get_backend(switches.force_cpu, framework=framework)
        if not self.backend.has_device():
            switches.proc_no_reuse = True
            # 探测失败回落 CPU 后把运行定格为 CPU：设备池 id 是 CPU 槽位而非
            # NPU id（DSMI 健康过滤按 force_cpu 跳过），worker 也免于二次探测。
            switches.force_cpu = True
        logging.info(f"Framework API mode: backend={self.backend.device_type()}, framework={framework}")

    @staticmethod
    def _inject_camodel_env(switches):
        """Prepare the camodel simulation environment for the E2E profiling workers.

        E2E profiling workers are forked (forkserver) from this process and
        inherit its ``LD_LIBRARY_PATH``, so prepending the camodel lib dir here
        makes every worker load the camodel runtime instead of the real device
        runtime. Must run before ``get_backend()`` probes
        ``torch.npu.is_available()`` so the NPU backend is selected.
        """
        if switches.force_cpu:
            raise ValueError("--cpu cannot be combined with --backend npusim")
        from ..common.simulator.config import resolve_camodel_lib_dir

        camodel = resolve_camodel_lib_dir(switches.sim_soc_version)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        camodel_str = str(camodel)
        # Exact entry check on the ":"-split list (mirrors _env._restore_ld_paths):
        # substring matching on the joined string would skip injection when an
        # unrelated entry merely contains the camodel path (e.g. ".../camodel_bak").
        if camodel_str not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{camodel_str}:{existing}" if existing else camodel_str
        log_dir = os.path.join(switches.sim_output_dir, "camodel_log")
        os.environ.setdefault("CAMODEL_LOG_PATH", log_dir)
        os.environ.setdefault("STARS_LOG_PATH", log_dir)
        logging.info("E2E npusim: camodel runtime injected from %s", camodel)

    def prepare_env(self):
        """环境前置检查（no-op）——hccl 手工表已改 param_extractor 惰性装载，
        无需 eager 注册（基类抽象方法要求实现）。"""

    def detect_device_count(self):
        switches = get_global_storage()
        if switches.device_count == -1:
            switches.device_count = self.backend.device_count()

    def detect_platform(self):
        switches = get_global_storage()
        if switches.dev_plat == "AUTO":
            switches.dev_plat = self.backend.device_name()
        switches.short_soc_version = self.backend.soc_series()
        if switches.core_limit and self.backend.is_npu():
            from ttk.core_modules.common.device import get_npu_hw_info, validate_core_limit

            try:
                hw_info = get_npu_hw_info(switches.dev_plat)
            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"Skip --core-limit validation, platform info unavailable: {e}")
            else:
                validate_core_limit(switches.core_limit, hw_info)

    def setup(self):
        pass

    def build_tasks(self, testcases: List[TestcaseE2e]) -> List[Task]:
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        # -t/--ti 圈选时按传入序（getattr 兼容免实例化调用）
        sorted_cases = execution_sorted(getattr(self, "_selection_order_key", None), testcases)
        tasks = []
        for testcase in sorted_cases:
            # 单值 device_ids（CSV "3" 指定单卡）透传——dispatch 层 pin 路由
            # 到指定设备组；多值经 _try_push_multi_device_task 交集解析。
            device_ids = list(testcase.device_ids) if testcase.device_ids else None
            tasks.append(
                Task(
                    testcase,
                    run,
                    (testcase, grant_events, granted_indices),
                    task_type="profile",
                    pool="profile",
                    device_ids=device_ids,
                )
            )
        return tasks

    def handle_result(self, task: Task, result: Any) -> tuple:
        self._print_new_compare_failures(
            task.testcase, result if isinstance(result, FrameworkApiReturnStructure) else None
        )
        if isinstance(result, FrameworkApiReturnStructure):
            return result.pick_data(self.result_titles())
        # worker 结果类型异常 → 显式 FAIL 行（不静默降级为字符串行）
        logging.error(
            "[%s] INVALID_WORKER_RESULT: unexpected worker result type %s",
            task.testcase.testcase_name,
            type(result).__name__,
        )
        failure = FrameworkApiReturnStructure()
        failure.construct("INVALID_WORKER_RESULT", "FAIL", None)
        return failure.pick_data(self.result_titles())

    def handle_error(self, task: Task, kind: str, detail):
        if kind == "none" and not task.task_type.startswith("compile"):
            # profile 任务无结果 → NO_WORKER_RESULT FAIL 行
            logging.error("[%s] NO_WORKER_RESULT: worker returned no result", task.testcase.testcase_name)
            failure = FrameworkApiReturnStructure()
            failure.construct("NO_WORKER_RESULT", "FAIL", None)
            return self._assemble_result_row(task, failure.pick_data(self.result_titles()))
        return super().handle_error(task, kind, detail)

    def result_titles(self) -> tuple:
        return FrameworkApiReturnStructure.get_titles()

    def get_device_info(self, dev_id: int) -> str:
        return f"{self.backend.device_type()}:{dev_id}"

    def _print_new_compare_failures(self, testcase, result=None) -> None:
        # 自定义比对失败详情：结构化条目随返回结构回传（不依赖日志格式扫描）
        if result is not None:
            print_structured_compare_failures(getattr(result, "precision_metrics", None), testcase.testcase_name)
        # COMPARE 级进单一 ttk.log（per-case 覆写）——按用例日志文件
        # 读 Index/Diff 行，失败明细随结果打屏。
        switches = get_global_storage()
        log_dir = build_single_log_dir(switches.test_mode, testcase.testcase_name, switches.root_path)
        diff_lines, _end = read_compare_log_failures(os.path.join(log_dir, COMPARE_LOG_FILE_NAME))
        print_compare_log_failures(diff_lines, testcase.testcase_name)
