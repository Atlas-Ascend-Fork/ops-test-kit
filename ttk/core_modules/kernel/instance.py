#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Kernel 通路 Instance。

- 双池 prepare_workers：编译池（tbe 编译栈独占）+ 执行池（每卡一组）——
  进程级隔离 tbe 与 torch/tf 的原生库符号冲突
- build_tasks：hash_cases_to_groups 编译分组去重 + compile/profile 双池 Task
- handle_result/handle_error：编译结果 apply 语义 + profile 行 pick
- setup/teardown：KB server 生命周期 + helper kernel 预编译
"""

__all__ = ["KernelInstance"]


# Standard Packages
import logging
import os
import shutil
import time
from typing import Any, List, Optional

from ttk.switches import get_global_storage

from ...utilities import cpu_count, limit_cann_compile_jobs
from ...utilities.proc import msdebug_runtime_injection_enabled
from ..common.comparison.compare_log import (
    print_compare_log_failures,
    print_structured_compare_failures,
    read_compare_log_failures,
)
from ..common.device import DSMIInterface, get_npu_hw_info, validate_core_limit
from ..common.instance import InstanceBase, register_instance
from ..common.logging import COMPARE_LOG_FILE_NAME, build_single_log_dir
from ..common.process import ProcessGroup, SimpleCommandProcess, Task, describe_exit
from .compile.compiler import get_compiler
from .compile.result import BaseCompilationResult, compilation_result, construct_crash_compilation_result
from .operator.knowledge_base import knowledge_base_sequence
from .result import KernelReturnStructure
from .testcase import TestcaseKernel
from .worker import run, run_compile


@register_instance("op")
@register_instance("kernel")
class KernelInstance(InstanceBase):
    """Kernel 通路 Instance（CLI 子命令 kernel/op 共用——"op" 为 CLI 传值兼容名）。"""

    name = "kernel"

    testcase_class = TestcaseKernel

    def __init__(self):
        super().__init__()
        self.kb: Optional[SimpleCommandProcess] = None

    # ═══ 探测 hook ═══

    def prepare_env(self):
        # check environment.
        opp_path = os.getenv("ASCEND_OPP_PATH", "")
        if not opp_path:
            raise RuntimeError("Install path of opp or compiler is not found. Please check Ascend installation.")
        # AscendC 构建链（CANN 内部经 make 驱动）前置检查：缺失时编译日志深处仅
        # `make: not found` 一行，紧随大批 `ld.lld: cannot open ./kernel_meta/*.o`，
        # 最终 DYN_COMPILE_FAILURE 不指向根因，须在此 fail-fast
        if (
            not self.switches.mode.is_model()
            and not self.switches.validate_only
            and not (self.switches.no_prof and self.switches.manual_data_dirs)
            and shutil.which("make") is None
        ):
            raise RuntimeError(
                "System `make` is required by the AscendC build chain but not found in PATH. "
                "Install it first (Debian/Ubuntu: apt-get install make; CentOS/EulerOS: yum install make). "
                "Without it, compilation buries `make: not found` under massive "
                "`ld.lld: error: cannot open ./kernel_meta/*.o` and ends in DYN_COMPILE_FAILURE."
            )
        # 限额容器下收紧 CANN 编译内部并行度（make -j）；须在 setup() 拉起任何
        # 子进程之前注入——forkserver 子进程不继承之后才设置的环境变量
        limit_cann_compile_jobs()

    def detect_device_count(self):
        """Resolve worker count and persist it on ``switches.device_count``.

        Manual-data prepare has no device execution, but the normal scheduler
        still needs one logical worker. That mode deliberately sets the stored
        count to one and avoids querying DSMI hardware.
        """
        if self.switches.no_prof and self.switches.manual_data_dirs or self.switches.mode.is_model():
            self.switches.device_count = 1
        elif self.switches.device_count <= 0:
            if self.switches.compile_only:
                count = min(max(cpu_count() - 1, 1), 8)
            else:
                count = DSMIInterface().get_device_count()
            self.switches.device_count = count

    def detect_platform(self):
        # NPUSim backend sets dev_plat to the platform ini name (e.g.
        # Ascend950PR_9589) in sim_args._normalize_sim_switches, so the normal
        # path below resolves short_soc_version / ccec versions from the ini.
        # --no-prof（含写方向）同样走统一探测：有卡自动 DSMI 解析真实平台，
        # 无卡准备机探测失败时经下方 except 提示显式 --plat——op_info 扫描与
        # 编译栈均依赖本处导出的 TTK_FULL/SHORT_SOC_VERSION。
        if self.switches.dev_plat == "AUTO":
            if self.switches.mode.is_model():
                raise RuntimeError(f"Please specify your platform type with --plat in {self.switches.mode.name} mode")
            try:
                self.switches.dev_plat = DSMIInterface().get_chip_info(0).get_complete_platform()
            except Exception as e:
                if (
                    self.switches.compile_only
                    or self.switches.validate_only
                    or self.switches.no_prof
                    and self.switches.manual_data_dirs
                ):
                    raise RuntimeError(
                        "Try to get Ascend platform failed. Please specify it with option like: --plat=Ascend910A"
                    ) from e
                raise
        hw_info = get_npu_hw_info(self.switches.dev_plat)
        self.switches.short_soc_version = hw_info.get("short_soc_version")
        validate_core_limit(self.switches.core_limit, hw_info)
        os.environ["TTK_FULL_SOC_VERSION"] = self.switches.dev_plat
        os.environ["TTK_SHORT_SOC_VERSION"] = self.switches.short_soc_version

    # ═══ setup / teardown（KB server + helper kernel）═══

    def setup(self):
        if msdebug_runtime_injection_enabled():
            logging.warning("Running under msdebug, skipping KnowledgeBase server startup")
        else:
            self._launch_knowledge_server(self.mp_context)
        if self.switches.mode.is_model():
            os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
        self._check_expected_impl_source()
        if (
            not self.switches.compile_only
            and not (self.switches.no_prof and self.switches.manual_data_dirs)
            and getattr(self.switches, "backend", None) != "npusim"
        ):
            self._compile_help_kernels()

    def _check_expected_impl_source(self):
        """--expect-impl-source 校验——实现来源与预期不符时显式失败，
        防自定义包未生效（未注册 / 被旧 vendor 包抢占）时静默回落内置实现而误报 PASS。"""
        expectation = getattr(self.switches, "expect_impl_source", None)
        if not expectation:
            return
        from .info_keeper import OpInfoKeeper

        source, _, vendor = expectation.partition("=")
        source = source.strip().lower()
        vendor = vendor.strip()
        if source not in ("builtin", "vendor", "custom"):
            raise ValueError(
                f"Invalid --expect-impl-source: {expectation!r} (expected builtin | vendor | custom[=name])"
            )
        keeper = OpInfoKeeper()
        mismatches = []
        seen_ops = set()
        for tc in self.flatten_testcases:
            if tc.op_name in seen_ops:
                continue
            seen_ops.add(tc.op_name)
            info = keeper.impl_source_of(tc.op_name)
            if info is None:
                mismatches.append(f"{tc.op_name}: no op info found for expected source {expectation!r}")
            elif info["source"] != source or (vendor and info["vendor"] != vendor):
                mismatches.append(
                    f"{tc.op_name}: expected source {expectation!r}, resolved source={info['source']!r} "
                    f"vendor={info['vendor']!r} base={info['impl_base_path']!r}"
                )
        if mismatches:
            raise RuntimeError(
                "IMPL_SOURCE_MISMATCH: tested operators do not resolve from the expected "
                f"implementation source ({expectation!r}):\n  " + "\n  ".join(mismatches)
            )
        logging.info(f"All {len(seen_ops)} tested operator(s) resolve from expected impl source: {expectation!r}")

    def teardown(self):
        """KB server 轮询退出（switch=False → RUNNING 结束 → close）；进程死亡即止，不无限等待。"""
        if self.kb:
            self.kb.data["switch"] = False
            while self.kb.status == self.kb.status.RUNNING:
                self.kb.update()
                if self.kb.is_dead():
                    logging.error(
                        f"KnowledgeBaseServer died during teardown "
                        f"(exit: {describe_exit(self.kb.get_exitcode())}, last stage: {self.kb.current_stage()})."
                    )
                    break
                time.sleep(1)
            self.kb.close()

    def _launch_knowledge_server(self, mp_context):
        logging.info("Launching knowledge base Server process")
        self.kb = SimpleCommandProcess(mp_context, name="KBS")
        self.kb.data["switch"] = True
        self.kb.send_action(knowledge_base_sequence, (), {})
        while self.kb.status != self.kb.status.RUNNING:
            # 逐秒轮询等待打屏无信息量（正常 2-3s 即绪）；异常路径走下方
            # RuntimeError / teardown 的 error，均有完整现场
            logging.debug(f"Waiting for KnowledgeBaseServer to be ready (status: {self.kb.status}) ...")
            self.kb.update()
            if self.kb.is_dead():
                raise RuntimeError(
                    "Process KnowledgeBaseServer is DEAD "
                    f"(exit: {describe_exit(self.kb.get_exitcode())}, "
                    f"last stage: {self.kb.current_stage()}). "
                    "Please check exception raised by KnowledgeBaseServer."
                )
            time.sleep(1)
        logging.info(f"Knowledge base Server Pid: {self.kb.get_pid()}")

    @staticmethod
    def _compile_help_kernels():
        get_compiler().compile_ub_clear()
        get_compiler().compile_l1_clear()
        get_compiler().compile_l0_clear()
        get_compiler().compile_warmup_kernel()

    # ═══ 双池 ═══

    def prepare_workers(self):
        """Kernel 双池：基类先行（锁表初始化 + 各卡执行池——golden/执行，
        永不加载 tbe），再建编译池（不绑卡，tbe 编译栈独占）。两类 worker
        进程级隔离，tbe 与 torch/tf 原生库符号冲突由此根治。

        建池前注入 OpInfoKeeper 缓存快照：forkserver worker 不继承主进程
        内存，主进程解析期完成的全量 op_info 扫描经引导载荷随 worker 拉起
        传递——各 worker（编译/执行）查询全命中，零重复扫描。"""
        from .info_keeper import snapshot_keeper

        SimpleCommandProcess._bootstrap_keeper_state = {"ttk.core_modules.kernel.info_keeper": snapshot_keeper()}
        super().prepare_workers()
        compile_count = min(max(cpu_count() - 1, 1), 8)
        compile_count = min(compile_count, max(len(self.flatten_testcases), 1))
        # --pc 联动封顶：编译池不越过用户并行度——多路重型 tbe 编译叠加
        # 会打爆限额容器（cgroup OOM kill），见 prepare_env 的作业数收敛
        compile_count = min(compile_count, max(int(self.switches.process_per_device or 1), 1))
        compile_pool = ProcessGroup("compile", 0, compile_count, self.mp_context, timeout=self.switches.proc_timeout)
        self.process_groups["compile"] = compile_pool
        while not compile_pool.is_ready():
            compile_pool.update()

    # ═══ task 构建（编译分组去重 + 双池 Task）═══

    def build_tasks(self, testcases) -> List[Task]:
        """编译 + profile task 一次性建齐：profile 经 depends_on 依赖本用例
        全部 compile task——编译完成（含失败）才分发；失败用例由 worker 前置
        检查短路产出编译失败行，不白算 golden。"""
        for case in testcases:
            case.kb_pid = self.kb.get_pid() if self.kb else 0
        grouped_testcases = TestcaseKernel.hash_cases_to_groups(testcases)
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        tasks: List[Task] = []
        for cases in grouped_testcases.values():
            anchor_compile_ids: List[str] = []
            for t in cases:
                compile_tasks = []
                for mode in ("Dyn", "Cst", "Bin"):
                    switch = getattr(self.switches, f"{mode.lower()}_switches")
                    if switch.enabled:
                        compile_tasks.append(
                            Task(
                                t,
                                run_compile,
                                (t, mode),
                                task_type=f"compile_{mode.lower()}",
                                pool="compile",
                                is_final=False,
                            )
                        )
                    else:
                        result = compilation_result(mode)
                        result.all_set(f"{mode.upper()}_OFF")
                        self.apply_compile_success_result(t, result)
                if not compile_tasks:
                    self.skipped_cases += 1
                    continue
                if anchor_compile_ids:
                    # 同 hash 兄弟用例：编译 task 依赖锚点完成后再弹出——semaphore
                    # 结果已就位（毫秒级取回），不占编译池槽位空等；锚点失败时
                    # 经 semaphore_dead_sequence 传播失败值，兄弟用例照样短路
                    for ct in compile_tasks:
                        ct.depends_on = list(anchor_compile_ids)
                else:
                    anchor_compile_ids = [ct.task_id for ct in compile_tasks]
                profile_task = Task(
                    t,
                    run,
                    (t, grant_events, granted_indices),
                    task_type="profile",
                    pool="profile",
                    depends_on=[ct.task_id for ct in compile_tasks],
                )
                tasks.extend(compile_tasks)
                tasks.append(profile_task)
        return tasks

    # ═══ 结果处理（handle_result / handle_error / recycle_worker）═══

    def handle_result(self, task: Task, result: Any) -> Optional[tuple]:
        """compile task 结果 apply 到 testcase 后返回 None（暂不落行）；
        profile task 结果 pick_data 出行 tuple。"""
        if task.task_type.startswith("compile"):
            self.apply_compile_success_result(task.testcase, result)
            return self.compile_done(task.testcase)
        return self._profile_row(task.testcase, result)

    def handle_error(self, task: Task, kind: str, detail) -> Optional[tuple]:
        """kernel 三分支扩展：compile+none = 用例 invalid 的编译短路。"""
        if kind == "none":
            if task.task_type.startswith("compile"):
                self._compile_invalid_case(task)
                return self.compile_done(task.testcase)
            raise RuntimeError(
                f"Profile worker returned None for testcase {task.testcase.testcase_name} "
                "(task_type: profile) — this should never happen; likely a TTK bug"
            )
        return super().handle_error(task, kind, detail)

    def recycle_worker(self, task: Task, result: Any) -> bool:
        """profile 执行失败（RTS 系错误）时回收 worker 进程清 ErrorMessage。"""
        if task.task_type.startswith("compile"):
            return False
        return result.kernel_execute_failed()

    def result_titles(self) -> tuple:
        return KernelReturnStructure.get_titles()

    def apply_compile_fail_result(self, testcase: TestcaseKernel, fail_info: str, compile_mode: str):
        result = construct_crash_compilation_result(fail_info, compile_mode)
        testcase.apply_compile_result(result)

    def apply_compile_success_result(self, testcase: TestcaseKernel, result: Any):
        if not isinstance(result, BaseCompilationResult):
            raise RuntimeError(f"Only subtype of BaseCompilationResult is valid. But got {type(result)}")
        testcase.apply_compile_result(result)

    def compile_done(self, testcase: TestcaseKernel) -> Optional[tuple]:
        return None

    def _profile_row(self, testcase: TestcaseKernel, result: Any) -> tuple:
        if not isinstance(result, KernelReturnStructure):
            raise RuntimeError(f"Only KernelReturnStructure is valid. But got {type(result)}")
        self._print_new_compare_failures(testcase, result)
        return result.pick_data(self.case_result_title)

    def _print_new_compare_failures(self, testcase, result=None) -> None:
        # 自定义比对失败详情：结构化条目随返回结构回传（不依赖日志格式扫描）
        if result is not None:
            print_structured_compare_failures(getattr(result, "precision_metrics", None), testcase.testcase_name)
        # COMPARE 级日志落在 per-case 追加档 ttk-compare.log（跨复跑保留，
        # "===== run start" 标记分段）——read_compare_log_failures 只读最后
        # 一个标记之后的本运行段 Index/Diff 行，失败明细随结果打屏。
        switches = get_global_storage()
        log_dir = build_single_log_dir(switches.test_mode, testcase.testcase_name, switches.root_path)
        diff_lines, _end = read_compare_log_failures(os.path.join(log_dir, COMPARE_LOG_FILE_NAME))
        print_compare_log_failures(diff_lines, testcase.testcase_name)

    @staticmethod
    def _compile_invalid_case(task: Task):
        # only when testcase is invalid
        if not isinstance(task.testcase, TestcaseKernel):
            raise RuntimeError(f"Only TestcaseKernel instance is valid. But got {type(task.testcase)}")
        testcase: TestcaseKernel = task.testcase
        reason = testcase.fail_reason
        logging.warning(
            f"Compilation process of mode {task.compile_mode} skipped for "
            f"testcase {testcase.testcase_name} because of {reason}"
        )
        result = construct_crash_compilation_result(reason, task.compile_mode)
        testcase.apply_compile_result(result)

    # ═══ 设备信息（进度打印）═══

    def get_device_info(self, dev_id: int) -> str:
        if self.switches.no_prof and self.switches.manual_data_dirs:
            return f"manual-data:{dev_id} {self.switches.dev_plat}"
        if not self.switches.mode.is_online_board() or self.switches.compile_only:
            phyid = str(dev_id)
            platform = self.switches.dev_plat
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
            physical_id = dsmi.get_physical_id_from_logical_id(dev_id)
            phyid = str(physical_id if physical_id is not None else dev_id)
            try:
                dsmi_platform = dsmi.get_chip_info(dev_id)
                platform = dsmi_platform.get_complete_platform()
                chip_ver = dsmi_platform.get_ver()
            except Exception:
                platform = "ERR"
                chip_ver = "ERR"

            try:
                health = dsmi.get_device_health_state(dev_id).name
            except Exception:
                health = "ERR"

            def _util(kind: int) -> str:
                val = dsmi.get_device_util(dev_id, kind, platform)
                return str(val if val is not None else "ERR")

            def _freq(kind: int) -> str:
                val = dsmi.get_device_frequency(dev_id, kind, platform)
                return str(val if val is not None else "????")

            temperature_val = dsmi.get_device_temperature(dev_id)
            temperature = str(temperature_val if temperature_val is not None else "???")

            mem_util = _util(1)
            aicore_util = _util(2)
            memband_util = _util(5)
            hbm_util = _util(6)
            ddr_util = _util(8)
            hbmband_util = _util(10)
            veccore_util = _util(12)

            mem_freq = _freq(1)
            hbm_freq = _freq(6)
            aicore_freq = _freq(7)
            aicore_maxfreq = _freq(9)
            veccore_freq = _freq(12)

        result = []

        def _add_info(info: str):
            if any(c.isdigit() for c in info):
                result.append(info)

        _add_info(
            f"{phyid.ljust(3)} {platform.ljust(13)} {chip_ver.ljust(2)} {temperature.ljust(2)}C {health.ljust(8)}"
        )
        _add_info(f"AIC   {aicore_util.ljust(3)}% {aicore_freq.ljust(5)}Mhz / {aicore_maxfreq.ljust(5)}Mhz")
        _add_info(f"VEC   {veccore_util.ljust(3)}% {veccore_freq.ljust(5)}Mhz")
        _add_info(f"MEM   {mem_util.ljust(3)}% {mem_freq.ljust(5)}Mhz Bandwidth: {memband_util.ljust(3)}%")
        _add_info(f"DDR   {ddr_util.ljust(3)}%")
        _add_info(f"HBM   {hbm_util.ljust(3)}% {hbm_freq.ljust(5)}Mhz Bandwidth: {hbmband_util.ljust(3)}%")
        return "\n".join(result)
