#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/instance.py：InstanceBase 模板方法 + 通路注册表。

- run() 模板方法（子类不可覆盖阶段序）：prepare_env → detect_platform →
  [--validate 早退] → detect_device_count → parse_cases → setup → open_output
  → prepare_workers → create_tasks → run_loop → cleanup
  （switches 经构造期全局存储注入，run 无参）
- 通路子类契约：类属性 name / testcase_class + 5 个 abstract hook
  （setup / build_tasks / handle_result / result_titles / get_device_info），
  @register_instance("xxx") 注册，get_instance_by_name 首查惰性单例
- handle_error(task, kind, detail)：worker 错误统一三分支（crash/fail/none；
  编译 task 失败 apply 到 testcase 后返回 None）
- recycle_worker(task, result)：结果后 worker 回收判定（kernel 执行失败清
  错误态；多卡/proc_no_reuse 由调度器另行强制）
- teardown()：setup 的对称收尾钩子（kernel：KB server 退出）
"""

__all__ = ["InstanceBase", "register_instance", "get_instance_by_name"]

# Standard Packages
import csv
import logging
import multiprocessing
import os
import shutil
import subprocess
import time
from abc import ABCMeta, abstractmethod
from multiprocessing.context import BaseContext
from typing import IO, Any, Dict, List, Optional, Tuple, Union

import numpy

from ttk.switches import get_global_storage

from ...utilities import VERSION, cpu_count, render_table
from .comparison import compare_batch_consistency
from .device import DSMI_HEALTH_STATE, DSMIInterface
from .process import ProcessGroup, SimpleCommandProcess, Task, TaskKeeper
from .testcase import TestcaseBase
from .testcase.batch import BATCH_RELATION_FIELDS, resolve_deterministic_level
from .testcase.factory import UniversalTestcaseFactory
from .testcase.reader import iter_tables

# ═══ 通路注册表（bridge 经 get_instance_by_name 分发，无 if-else）═══

_INSTANCE_REGISTRY: Dict[str, "InstanceBase"] = {}


def register_instance(name: str):
    """类装饰器：注册通路。name 即 CLI 子命令名与 switches.test_mode 值。

    注册存类（不实例化）——InstanceBase.__init__ 读 get_global_storage()，
    bridge set_global_storage 之前不可构造；单例在 get_instance_by_name
    首查时惰性实例化（彼时 switches 已就绪）。"""

    def decorator(cls):
        if name in _INSTANCE_REGISTRY:
            raise ValueError(f"instance name {name!r} already registered: {_INSTANCE_REGISTRY[name]!r}")
        _INSTANCE_REGISTRY[name] = cls
        return cls

    return decorator


_INSTANCE_SINGLETONS: Dict[str, "InstanceBase"] = {}


def get_instance_by_name(name: str) -> "InstanceBase":
    """按名称获取通路单例（首查惰性实例化）。未注册抛 KeyError（列出已注册名）。"""
    if name in _INSTANCE_SINGLETONS:
        return _INSTANCE_SINGLETONS[name]
    try:
        cls = _INSTANCE_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown test mode {name!r}; registered: {sorted(_INSTANCE_REGISTRY)}") from None
    _INSTANCE_SINGLETONS[name] = cls()
    return _INSTANCE_SINGLETONS[name]


# 跑批 worker 是 forkserver 子进程 —— 它们从一个极简进程起, **不继承父进程已导入的模块**,
# 于是每个 worker 都要自己 import 一遍重模块。本机实测: torch 2.37s、tbe 0.64s、te 0.67s,
# 这笔开销与用例数无关, 每次跑批都付(小批量调试时占比最高)。
# ``set_forkserver_preload`` 让 forkserver 进程自己先导入一次, 子进程 fork 出来即已具备,
# 实测同一机制在 xpu_server 上把子进程启动从 2.3s 压到 0.03s。
# 安全性: CPython 的 forkserver 对 preload 列表里的模块是 ``try: __import__ except ImportError: pass``,
# 模块缺失只会被忽略, 不会让跑批起不来; 故这里只列"装了就该预载"的重模块。
# torch 系预载对 TF 用例是毒药: torch/tbe/te 会连带加载 torch_npu, 而已加载 torch 的
# 进程再 import tensorflow 必定 SIGSEGV(libstdc++ 弱符号合并冲突, 见
# ttk/remote/server/_framework_preload.py 模块注记)。TF 预载 tensorflow 先行,
# 后续任意 torch 系导入均安全(实测 tf → torch/torch_npu/tbe/te 全序正常)。
_WORKER_PRELOAD_TORCH = ("numpy", "torch", "tbe", "te")
_WORKER_PRELOAD_TF = ("numpy", "tensorflow")


def _worker_preload_modules(switches) -> tuple:
    """按通路/框架选预载表。框架检测(run_with_switches)先于 Instance 构建完成。"""
    if getattr(switches, "test_mode", "") == "framework-api" and getattr(switches, "framework", "") == "tf":
        return _WORKER_PRELOAD_TF
    return _WORKER_PRELOAD_TORCH


def _preload_worker_modules(ctx: BaseContext, preload: tuple = _WORKER_PRELOAD_TORCH) -> None:
    """给 worker 的 forkserver 设预载模块。必须在任何子进程启动**之前**调用。"""
    try:
        ctx.set_forkserver_preload(list(preload))
    except Exception as e:  # noqa: BLE001  # 预载只是提速, 失败绝不该让跑批不可用
        logging.debug("set_forkserver_preload skipped: %s", e)


def intersect_device_ids(requested, grantable):
    """CSV device_ids ∩ 可用池（保序）。返回 (effective, failure_reason)。

    - 全部可用 → (requested, None)
    - 部分交集且剩余 ≥2 → (交集, None)（调用方截断继续并记 warning）
    - 交集为空 / 仅剩 1 台 → (交集, 明确失败原因)——多卡用例不降级单卡
      （hccl 通信域在单卡上无意义）
    """
    requested = list(requested)
    effective = [d for d in requested if d in grantable]
    if len(effective) == len(requested):
        return effective, None
    if not effective:
        return effective, (
            f"device_ids {requested} ∩ available devices {sorted(grantable)} = ∅ "
            "(check CSV 'device_ids' vs --devices/--dev and device health)"
        )
    if len(effective) == 1:
        return effective, (
            f"multi-device case requires >= 2 devices; device_ids {requested} ∩ "
            f"available devices {sorted(grantable)} = {effective} (only 1 device left)"
        )
    return effective, None


class InstanceBase(metaclass=ABCMeta):
    """所有测试通路的基类。子类要求：
      1. 设置类属性 name / testcase_class
      2. 实现 5 个 @abstractmethod hook（setup / build_tasks / handle_result /
         result_titles / get_device_info），按需 override 通用方法与探测 hook
      3. 用 @register_instance("xxx") 注册
      4. 单例由 get_instance_by_name 首查惰性保证
    worker 逻辑不在此类——Task.func 引用各通路 worker.py 的模块级函数。
    """

    # 通路身份（子类必须设置）
    name: str = ""
    testcase_class: type = None  # TestcaseBase 子类——聚合层 Factory 参数化来源

    def __init__(self):
        self.switches = get_global_storage()
        self.flatten_testcases: List[TestcaseBase] = []
        self.total_case_count = 0
        self.completed_case_count = 0
        # build_tasks 期间计数的跳过用例（无任务产出：geir invalid / kernel 无编译模式）
        self.skipped_cases: int = 0
        # Final summary counters (driven by precision_status in result rows)
        self.pass_count = 0
        self.fail_count = 0
        self.other_count = 0
        self._header_flushed = False
        self._precision_status_idx: Optional[int] = None
        # Test Result CSV Storage
        self.result_csv_writer = None
        self.result_csv_file: Optional[IO[Any]] = None
        self.result_path: str = ""
        # Testcase
        self.case_original_headers: List[str] = []
        self.case_result_titles: Tuple[str] = ()
        self.case_input_title: Tuple[str] = ()
        self._front_input_count: int = 0
        # Multiprocessing
        self.mp_context: BaseContext = multiprocessing.get_context("forkserver")
        _preload_worker_modules(self.mp_context, _worker_preload_modules(self.switches))
        self.mp_manager = None
        self.device_locks = ()
        self.used_device = []
        self._health_excluded_devices: tuple = ()
        # dev_id → 执行池组（pool="profile"）；"compile" → 编译池组（Kernel 双池）
        self.process_groups: Dict[Union[int, str], ProcessGroup] = {}
        # Tasks
        self.task_keeper: TaskKeeper = TaskKeeper()
        # Multi-device task tracking
        self._multi_device_completed: Dict[str, dict] = {}
        self._multi_device_running = False
        # DFX
        self.start_timestamp = time.time()
        self.last_print_timestamp = time.time() - 20
        self.print_cycle = 10
        if self.switches.random_seed:
            numpy.random.seed(self.switches.random_seed)
        self._commit_id: Optional[str] = None
        self.heartbeat_manager = None  # HeartbeatManager or None
        # dl=3 batch 一致性：worker 回传的 BatchSummary 收集（cleanup 组装比对）
        self._batch_summaries: Dict[str, Any] = {}
        # dl=3 缺字段提示的按例去重（派发期逐 case 打，deferred 重派/双池不重复）
        self._batch_relation_warned: set = set()
        # -t/--ti 圈选时的传入序 key（None = 未圈选，按 CSV 行序）
        self._selection_order_key = None

    @staticmethod
    def _read_existing_header(path: str):
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                return next(reader, None)
        except (UnicodeDecodeError, csv.Error, OSError):
            return None

    # ═══ Abstract hooks（通路必须实现）═══

    @abstractmethod
    def prepare_env(self):
        """环境检查（Kernel/ACLNN：ASCEND_OPP_PATH；GEIR：ascend root）。
        E2E 无需环境前置——hccl 手工表已改 param_extractor 惰性装载。"""

    @abstractmethod
    def detect_platform(self):
        """SoC 探测：写 switches.dev_plat / short_soc_version（E2E 经 Backend）。"""

    @abstractmethod
    def detect_device_count(self):
        """设备数探测：写 switches.device_count（E2E 经 Backend）。"""

    @abstractmethod
    def setup(self):
        """通路初始化。Kernel：KB server 启动 + helper kernel 预编译；
        其余通路：注册/环境类动作。"""

    @abstractmethod
    def build_tasks(self, testcases: List[TestcaseBase]) -> List[Task]:
        """testcase 列表 → Task 列表（Task.func 引用本通路 worker.py 模块级函数）。
        无任务产出的用例自行累加 self.skipped_cases。Kernel 在此实现编译分组
        去重 + compile/profile 双池 Task；其余通路每用例一个 profile task。"""

    @abstractmethod
    def handle_result(self, task: Task, result: Any) -> Optional[Tuple]:
        """worker 结果 → CSV 行 tuple；None = 暂不输出（如 compile task 完成，
        结果已 apply 到 testcase）。"""

    @abstractmethod
    def result_titles(self) -> tuple:
        """该通路结果 CSV 的结果列 tuple（open_output 写表头/取数用）。"""

    @abstractmethod
    def get_device_info(self, dev_id: int) -> str:
        """设备信息字符串（进度打印用），如温度/利用率/频率。"""

    # ═══ 模板方法（基类实现，子类不可覆盖）═══

    def run(self):
        """主入口，CLI 调用。完整测试生命周期编排——阶段序不可覆盖。"""
        logging.info("TTK Profiling Start!")
        logging.info(f"Mode: {self.switches.mode.name}")
        self.prepare_env()
        self.detect_platform()
        logging.info(f"Device Platform: {self.switches.dev_plat}")
        if self.switches.validate_only:
            self._validate_only()
            return
        self.detect_device_count()
        if self.switches.device_count <= 0:
            raise RuntimeError(f"Device count is invalid: {self.switches.device_count}")
        logging.info(f"Device Count: {self.switches.device_count}")
        self.parse_cases()
        self._start_heartbeat_process()  # after validate_only early-return, before setup
        self.setup()
        self.open_output()
        self.prepare_workers()
        self.create_tasks()
        self._reset_xpu_breaker()
        self.run_loop()
        self.cleanup()

    def _reset_xpu_breaker(self):
        from ttk.remote import xpu_breaker

        try:
            xpu_breaker.reset(self.switches.root_path)
        except OSError as e:
            logging.warning("xpu breaker reset failed (ignored): %s", e)

    def _xpu_breaker_reason(self) -> str:
        from ttk.remote import xpu_breaker

        try:
            return xpu_breaker.tripped(self.switches.root_path)[1]
        except OSError as e:
            logging.warning("xpu breaker check failed (ignored): %s", e)
            return ""

    # ═══ 通用方法（基类实现，通路可 override）═══

    def parse_cases(self):
        """用例解析：iter_tables → UniversalTestcaseFactory(testcase_class) →
        去重/筛选 → 逐例校验 → self.flatten_testcases / case_original_headers。

        校验后置于筛选：selector（-t/--op/--priority/--ti/rerun/enabled）只消费
        解析期就绪的原始字段（testcase_name/op_name/priority/_csv_row_index），
        而校验可能触发重导入（如 e2e 的 torch.ops 扩展包签名探测会 import
        扩展包并 JIT 编译其携带的全部算子）——未选中的行不该付这笔代价。"""
        logging.info("Preparing testcases ...")
        for case_file in self.switches.input_files:
            logging.info(f"Reading testcase file {case_file} ...")
            self._load_cases(case_file)
        if len(self.flatten_testcases) <= 0:
            raise RuntimeError(
                "No testcases left after parsing and filtering. Check input CSV content and filter options."
            )
        logging.info("Filtering testcases ...")
        self._filter_testcases()
        logging.info("Checking testcase name...")
        self._check_duplicate_case()
        logging.info("Validating testcases ...")
        self._validate_testcases()
        self.total_case_count = len(self.flatten_testcases)
        logging.info(f"Case number: {self.total_case_count}")

    def _validate_testcases(self):
        """筛选后的逐例校验（skip_validate 延迟到此）。"""
        from ttk.utilities import set_thread_name

        for testcase in self.flatten_testcases:
            set_thread_name(testcase.testcase_name)
            testcase.validate()

    def prepare_workers(self):
        """建 ProcessGroup。基类默认单池（pool="profile"，每卡一组）；
        Kernel override 为编译池 + 执行池双池。"""
        self._prepare_device_locks()
        # Create process for every usable device
        if self.switches.process_per_device is None:
            # Use 80% of total cpu cores, not exceed 4
            self.switches.process_per_device = min(max(int(cpu_count() * 0.8) // len(self.used_device), 1), 4)
        # Not exceed testcase count
        self.switches.process_per_device = min(self.switches.process_per_device, len(self.flatten_testcases))
        self.used_device = self.used_device[: len(self.flatten_testcases)]
        logging.info(f"Process per device: {self.switches.process_per_device}")
        # Prepare SubProcesses（执行池，每卡一组；Kernel 另建编译池——见 KernelInstance）
        logging.info(
            f"Starting worker processes: {len(self.used_device)} device group(s), "
            f"{self.switches.process_per_device} process(es) per device"
        )
        for dev_id in self.used_device:
            self.process_groups[dev_id] = ProcessGroup(
                "profile",
                dev_id,
                self.switches.process_per_device,
                self.mp_context,
                timeout=self.switches.proc_timeout,
            )
        for pg in self.process_groups.values():
            while not pg.is_ready():
                pg.update()

    def create_tasks(self):
        """调通路 build_tasks() → TaskKeeper（扁平入队，组内顺序即派发优先序）。"""
        self._reset_case_logs()
        tasks = self.build_tasks(self.flatten_testcases)
        self.task_keeper.append(tasks)
        self.completed_case_count += self.skipped_cases

    def _reset_case_logs(self):
        """运行前重置各用例 ttk.log——worker 追加写（编译/执行多任务共写），
        复跑覆写语义由本重置实现；compare 档是跨复跑历史，不重置。"""
        from .logging import build_single_log_dir, reset_case_log

        for tc in self.flatten_testcases:
            reset_case_log(build_single_log_dir(self.switches.test_mode, tc.testcase_name, self.switches.root_path))

    def run_loop(self):
        """调度主循环：收结果（handle_error 三分支 / handle_result）→ 按池派发
        （pop(pool) 路由）→ 进度，直至 all_done。"""
        while True:
            self._supervise_heartbeat()  # respawn HB if it died; None-safe + throttled ~1s
            self._update_processes()
            self._handle_completed_process()
            self._push_task_to_process()
            self._close_idle_processes()
            self._summary_print(self.print_cycle)
            if self.total_case_count == self.completed_case_count:
                logging.info("All testcases completed")
                break
            time.sleep(
                0.01
            )  # pacing before next iteration; avoids busy-spin 100% CPU on one core. Placed after the break-check so we don't sleep on the exiting iteration.

    def handle_error(self, task: Task, kind: str, detail) -> Optional[Tuple]:
        """worker 错误统一 hook（三分支默认实现，通路可 override）。

        kind: "crash"=SystemError（detail=(last_stage, pid, result)）；
              "fail"=RuntimeError（detail=(pid, result)）；"none"=结果为 None。
        编译 task（compile_* 前缀）失败 apply 到 testcase 后返回 None（不落行，
        依赖它的 profile task 分发时短路）；执行 task 返回错误结果行；
        "none" 默认无行（防御路径——kernel 覆写为 raise/编译 invalid 分支）。"""
        if task.task_type.startswith("compile"):
            if kind == "crash":
                last_stage, pid, result = detail
                logging.fatal(
                    f"Compilation process of mode {task.compile_mode} crashed at stage {last_stage} "
                    f"for testcase {task.testcase.testcase_name} with pid {pid}. "
                    f"System error: {result}"
                )
                self.apply_compile_fail_result(
                    task.testcase, f"Crashed at stage {last_stage}: {result}", task.compile_mode
                )
            elif kind == "fail":
                pid, result = detail
                logging.error(
                    f"Compilation process of mode {task.compile_mode} failed "
                    f"for testcase {task.testcase.testcase_name} with pid {pid}, "
                    f"fail reason: \n{result.args[0]}"
                )
                self.apply_compile_fail_result(task.testcase, "COMPILE_FAILURE", task.compile_mode)
            return self.compile_done(task.testcase)
        if kind == "crash":
            last_stage, pid, result = detail
            logging.fatal(
                f"Profile process crashed at stage {last_stage} "
                f"for testcase {task.testcase.testcase_name} with pid {pid}. "
                f"System error: {result}"
            )
            return self._error_row(task, "PROFILE_CRASH", f"Crashed at profiling stage: {result}")
        if kind == "fail":
            pid, result = detail
            logging.error(
                f"Profiling process failed "
                f"for testcase {task.testcase.testcase_name} with pid {pid}, "
                f"fail reason:\n{result.args[0]}"
            )
            return self._error_row(task, "FAILURE", str(result.args[0]))
        return None

    def recycle_worker(self, task: Task, result: Any) -> bool:  # noqa: B027
        """结果后 worker 回收判定（默认不复用即 False）。Kernel override：
        profile 结果 kernel_execute_failed 时回收 worker 进程清 ErrorMessage。"""
        return False

    def teardown(self):  # noqa: B027
        """setup 的对称收尾钩子（默认 no-op；Kernel：KB server 轮询退出）。"""

    def cleanup(self):
        """关池 + dl=3 batch 一致性组装比对 + 按需停止心跳 + teardown + 进度落盘 + CSV 关闭 + 最终摘要。"""
        self.close_subprocesses()
        self._compare_batch_summaries()
        self.teardown()
        if self.switches.progress_output:
            self._output_progress()
        if self.result_csv_file:
            self.result_csv_file.close()
        self._print_final_summary()

    def _collect_batch_summary(self, task: Task, result: Any) -> None:
        """dl=3 worker 回传的 BatchSummary 收集（worker 侧 assemble 已按
        deterministic_level==3 门控，此处无摘要即跳过）。"""
        summary = getattr(result, "batch_summary", None)
        if summary is not None:
            self._batch_summaries[task.testcase.testcase_name] = summary

    def _compare_batch_summaries(self) -> None:
        """batch 一致性组装比对：按 bcid 分组 → 同关系位置 hash 集合基数判定
        （四通路共同尾巴；无收集即 no-op）。"""
        if not self._batch_summaries:
            return
        consistent, failures = compare_batch_consistency(self._batch_summaries)
        if consistent:
            logging.info(
                "batch consistency PASS: %d case(s) in %d group(s)",
                len(self._batch_summaries),
                len({s.batch_consistency_id for s in self._batch_summaries.values()}),
            )
        else:
            logging.error("batch consistency FAIL: %s", failures)
        logging.info(
            "Level-3 phase one compared batch relations by per-sample output hashes (see the "
            "batch consistency summary above). For raw-dump slice comparison across cases with "
            "different relation sets, run with --dump out --dump-format bin and NPU_DUMP_PATH "
            "set, then run scripts/compare_batch_consistency.py with this result CSV."
        )

    def _warn_incomplete_batch_relation(self, testcase) -> None:
        """dl=3 但 batch 三字段缺一：派发期按用例提示一次。

        挂在 run_loop 的任务派发点（_push_task_to_process 的 pop choke point），
        不做解析后全量扫描——缺字段是用例级信息，随派发打印即可；deferred
        重派 / kernel 双池（compile+profile 同 case）经名字去重不重复打。
        --validate-only 不经 run_loop，按文档约定不输出该执行提示。"""
        name = getattr(testcase, "testcase_name", None)
        if name is None or name in self._batch_relation_warned:
            return
        self._batch_relation_warned.add(name)
        if self.switches.no_prof and self.switches.manual_data_dirs:
            return
        if not testcase.is_enabled or not testcase.is_valid:
            return
        try:
            level = resolve_deterministic_level(self.switches, testcase)
        except ValueError:
            # The worker reports malformed levels through the normal case failure path.
            return
        if level != 3:
            return
        missing = [n for n in BATCH_RELATION_FIELDS if getattr(testcase, n, None) is None]
        if missing:
            logging.warning(
                "%s: deterministic level 3 without complete batch metadata (missing: %s); "
                "continuing ordinary level-3 execution without batch consistency comparison.",
                name,
                ", ".join(missing),
            )

    def open_output(self):
        """结果 CSV 打开 + 列名计算 + 写表头（append 模式表头对账）。"""
        logging.info("Initialize output csv file...")
        self.result_path = self.switches.output_file_name
        if not self.switches.input_files:
            raise RuntimeError("No input testcase files specified. Pass -i/--input.")
        if not self.result_path:
            root, _ = os.path.splitext(self.switches.input_files[0])
            sheet = self._resolved_output_sheet()
            tag = ""
            if sheet:
                # Escape the escape marker too, so distinct legal sheet names cannot collapse to one path.
                tag = "_" + sheet.translate(str.maketrans({char: f"%{ord(char):02X}" for char in '\\/:*?"<>| %'}))
            self.result_path = root + tag + "_result.csv"
            logging.info(f"Output csv file is not specified. It will be set as {self.result_path}")
        if not self.result_path.endswith(".csv"):
            self.result_path += ".csv"
        parent_dir = os.path.dirname(self.result_path)
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logging.info(f"Created output directory: {parent_dir}")
        append_mode = self.switches.append_mode
        file_exists = os.path.exists(self.result_path) and os.path.getsize(self.result_path) > 0
        if append_mode and file_exists:
            self._prepare_output_titles()
            existing_header = self._read_existing_header(self.result_path)
            header_match = existing_header is not None and tuple(existing_header) == self.case_result_titles
            if header_match:
                self.result_csv_file = open(self.result_path, newline="", mode="a+")  # noqa: SIM115
                self.result_csv_writer = csv.writer(self.result_csv_file)
                self._header_flushed = True
                self._precision_status_idx = self._resolve_precision_status_idx(self.case_result_titles)
                logging.info(f"Append mode: appending to existing {self.result_path}")
            else:
                logging.warning(
                    f"Append mode: existing file header does not match "
                    f"(file has {len(existing_header) if existing_header else 0} columns, "
                    f"current expects {len(self.case_result_titles)}). "
                    f"Overwriting {self.result_path}"
                )
                self.result_csv_file = open(self.result_path, newline="", mode="w+")  # noqa: SIM115
                self.result_csv_writer = csv.writer(self.result_csv_file)
                self.write_row(self.case_result_titles)
        else:
            self.result_csv_file = open(self.result_path, newline="", mode="w+")  # noqa: SIM115
            self.result_csv_writer = csv.writer(self.result_csv_file)
            self._prepare_output_titles()
            self.write_row(self.case_result_titles)

    def write_row(self, row: tuple) -> None:
        """写 CSV 行 + 摘要计数（首行为表头只落盘不计数）。"""
        self.result_csv_writer.writerow(row)
        self.result_csv_file.flush()
        if not self._header_flushed:
            self._header_flushed = True
            self._precision_status_idx = self._resolve_precision_status_idx(row)
            return
        self._update_summary(row)

    def close_subprocesses(self):
        self._stop_heartbeat_process()
        for pg in self.process_groups.values():
            pg.close_all()

    def testcase_complete(self):
        self.completed_case_count += 1

    # ═══ 编译 task 通用 apply 语义（默认 no-op；Kernel 覆写——其余通路无编译 task）═══

    def apply_compile_fail_result(self, testcase: TestcaseBase, crash_info: str, compile_mode: str):  # noqa: B027
        pass

    def apply_compile_success_result(self, testcase: TestcaseBase, result: Any):  # noqa: B027
        pass

    def compile_done(self, testcase: TestcaseBase) -> Optional[tuple]:  # noqa: B027
        return None

    # ═══ 心跳（远端 XPU 配置时）═══

    def _start_heartbeat_process(self) -> None:
        """Spawn heartbeat subprocess if remote execution is configured.

        Called from run() before setup. Heartbeat starts sending GET
        /v1/heartbeat (merged health+detect+register) to all configured
        endpoints; writes health state to TTK_XPU_HEALTH_PATH.
        """
        try:
            from ttk.remote import get_tenant_id, is_remote_configured
            from ttk.remote.config import get_remote_config
            from ttk.remote.heartbeat import heartbeat_loop
            from ttk.remote.heartbeat_manager import HeartbeatManager
        except ImportError:
            return  # ttk.remote not available

        if not is_remote_configured():
            return

        config = get_remote_config()
        if not config or not config.endpoints:
            return

        # Shared TLS module (same source as dispatcher): tls_from_config does the
        # cert/key pair check (raises on mismatch, fail-loud); returns {} when no
        # TLS configured — build_tls_connection treats {} as plain HTTP.
        # NB: tls_from_config call is OUTSIDE the ImportError try/except above so a
        # cert/key mismatch raises as a loud startup failure (not swallowed).
        from ttk.remote.tls import tls_from_config

        tls = tls_from_config(config)
        self.heartbeat_manager = HeartbeatManager(
            heartbeat_target=heartbeat_loop,
            root_path=self.switches.root_path,
            tenant_id=get_tenant_id(),
            endpoints=config.endpoints,
            tls=tls,
        )
        self.heartbeat_manager.start()

    def _stop_heartbeat_process(self) -> None:
        """Terminate heartbeat subprocess and cleanup state.

        Called from close_subprocesses(). The subprocess sends DELETE to all
        endpoints before exiting, cleaning up tenant state on the server.
        """
        if self.heartbeat_manager is not None:
            self.heartbeat_manager.stop()
            self.heartbeat_manager = None

    def _supervise_heartbeat(self) -> None:
        """Respawn HB if it died. None-safe + throttled (~1s, not every 10ms loop tick).

        Called at the top of the run_loop() before _update_processes.
        heartbeat_manager is None when remote execution is off — guarded.
        """
        if self.heartbeat_manager is None:
            return
        now = time.time()
        if now - getattr(self, "_last_supervise_ts", 0.0) < 1.0:
            return
        self._last_supervise_ts = now
        self.heartbeat_manager.supervise()

    # ═══ 进度与摘要 ═══

    def _summary_print(self, print_cycle: int = 0):
        now = time.time()
        if now - self.last_print_timestamp > print_cycle:
            self.last_print_timestamp = now
            self._output_progress()
            if self.switches.summary_print:
                self._get_head_commit_id()
                try:
                    percentage = int(self.completed_case_count / self.total_case_count * 100)
                except Exception:
                    percentage = "?"
                title = (
                    f"Version: {VERSION} "
                    f"Summary (Device Total: {self.switches.device_count}) "
                    f"Progress: {percentage}% "
                    f"{self.completed_case_count} / {self.total_case_count} "
                    f"ET: {int(now - self.start_timestamp)}s "
                    f"Rev: {self._commit_id}",
                )

                loop_count = len(self.used_device) // 2 if self.used_device else self.switches.device_count // 2
                remain_count = len(self.used_device) % 2 if self.used_device else self.switches.device_count % 2
                lines = [title]
                # 编译池行（Kernel 双池）：不绑卡不占设备位，独立成行——长编译阶段
                # 执行 worker 全 idle，没有此行时表格无可观测信息
                compile_pg = self.process_groups.get("compile")
                if compile_pg is not None:
                    compile_info = compile_pg.info()
                    if compile_info:
                        lines.append(("CompilePool", compile_info))
                for loop in range(loop_count):
                    if self.used_device:
                        dev_id_0 = self.used_device[loop * 2]
                        dev_id_1 = self.used_device[loop * 2 + 1]
                    else:
                        dev_id_0, dev_id_1 = loop * 2, loop * 2 + 1
                    lines.append(
                        (
                            *(
                                self.get_device_info(dev_id_0),
                                self.process_groups[dev_id_0].info() if dev_id_0 in self.process_groups else "",
                            ),
                            *(
                                self.get_device_info(dev_id_1),
                                self.process_groups[dev_id_1].info() if dev_id_1 in self.process_groups else "",
                            ),
                        )
                    )
                if remain_count:
                    dev_id = self.used_device[-1] if self.used_device else loop_count * 2
                    lines.append(
                        (
                            *(
                                self.get_device_info(dev_id),
                                self.process_groups[dev_id].info() if dev_id in self.process_groups else "",
                            ),
                        )
                    )
                logging.info("\n" + render_table(lines))

    def _print_final_summary(self):
        total = self.pass_count + self.fail_count + self.other_count
        if total <= 0:
            return
        pass_rate = (self.pass_count / total * 100) if total else 0.0
        # Total 是**载入条数**。喂入多于载入时把差额显式列出来——
        # 喂 200 跑 181 也报 PassRate 98.34%，覆盖被砍掉却看不出来。
        from ..common.testcase.factory import LOAD_STATS

        dropped = LOAD_STATS.get("dropped", 0)
        lines = [
            ("Total", "PASS", "FAIL", "SKIP/NA", "PassRate"),
            (
                str(total),
                str(self.pass_count),
                str(self.fail_count),
                str(self.other_count),
                f"{pass_rate:.2f}% ({self.pass_count}/{total})",
            ),
        ]
        if dropped:
            lines.append(
                (
                    f"FED {LOAD_STATS.get('fed', 0)}",
                    f"DROPPED {dropped}",
                    "(未载入未执行; 原因见 'Testcase load:' 行)",
                    "",
                    "",
                )
            )
        logging.info("\n" + render_table(lines))

    def _log_batch_execution_mode(self, *, before_execution=False):
        """Warn before dispatch; leave the phase-two handoff at the end of execution."""
        from ttk.core_modules.common.testcase.batch import BATCH_RELATION_FIELDS, resolve_deterministic_level

        if self.switches.no_prof and self.switches.manual_data_dirs:
            return
        needs_comparison = False
        for testcase in self.flatten_testcases:
            if not testcase.is_enabled or not testcase.is_valid:
                continue
            try:
                level = resolve_deterministic_level(self.switches, testcase)
            except ValueError:
                # The worker reports malformed levels through the normal case failure path.
                continue
            if level != 3:
                continue
            missing = [name for name in BATCH_RELATION_FIELDS if getattr(testcase, name, None) is None]
            if missing and before_execution:
                logging.warning(
                    "%s: deterministic level 3 without complete batch metadata (missing: %s); "
                    "continuing ordinary level-3 execution without batch consistency comparison.",
                    testcase.testcase_name,
                    ", ".join(missing),
                )
            elif not missing:
                needs_comparison = True
        if needs_comparison and not before_execution:
            logging.info(
                "Level-3 phase one compared batch relations by per-sample output hashes (see the "
                "batch consistency summary above). For raw-dump slice comparison across cases with "
                "different relation sets, run with --dump out --dump-format bin and NPU_DUMP_PATH "
                "set, then run scripts/compare_batch_consistency.py with this result CSV."
            )

    # ═══ 设备锁与建池前置 ═══

    def _prepare_device_locks(self):
        # TODO: device-id not start from 0 to max-count in docker.
        # Prepare device locks
        self._validate_device_count_against_hardware()
        self.device_locks = self._initialize_device_lock()
        for dev_id, dev_lock in enumerate(self.device_locks):
            if dev_lock is not None:
                self.used_device.append(dev_id)
        if len(self.used_device) <= 0:
            health_part = (
                f", unhealthy: {sorted(self._health_excluded_devices)}" if self._health_excluded_devices else ""
            )
            raise RuntimeError(
                f"No available device: total {self.switches.device_count}, "
                f"excluded by --devices: {sorted(self.switches.device_blacklist)}{health_part}. "
                f"Adjust --devices or check device health (npu-smi info)."
            )

    def _validate_device_count_against_hardware(self):
        """请求的设备数量与本机物理卡数对账。

        语义分开：`--dev N` 是数量上限，物理卡不足时压缩到实际卡数（告警留痕）；
        `--devices` 的点名/正选 ID 是显式请求，越界物理卡数按配置错误启动即报错。
        容错边界：模型/仿真/编译态不依赖真卡；DSMI 探测失败或探得 0 卡
        （无驱动容器、远端执行）时跳过——只在"本机确有卡且确实不够"时处理。
        """
        if not self.switches.mode.is_online_board():
            return
        if getattr(self.switches, "compile_only", False) or getattr(self.switches, "backend", None) == "npusim":
            return
        if self.switches.device_count <= 0:
            return
        try:
            physical = DSMIInterface().get_device_count()
        except Exception:
            return
        if not 0 < physical < self.switches.device_count:
            return
        if getattr(self.switches, "device_count_is_limit", False):
            logging.warning(
                f"--dev {self.switches.device_count} exceeds {physical} physical Ascend device(s); "
                f"scheduling on {physical} device(s) instead"
            )
            self.switches.device_count = physical
            return
        raise RuntimeError(
            f"--dev/--devices resolve to device IDs 0-{self.switches.device_count - 1}, but only {physical} "
            f"Ascend device(s) are present on this machine (valid device IDs: 0-{physical - 1})"
        )

    def _initialize_device_lock(self) -> tuple:
        self.mp_manager = self.mp_context.Manager()
        # --devices 表达式已在 bridge 解析为黑名单
        if self.switches.device_blacklist:
            logging.info(f"Device {self.switches.device_blacklist} has been blacklisted, removing...")
        available = tuple(
            True if n not in self.switches.device_blacklist else None for n in range(self.switches.device_count)
        )
        available_devices = [i for i, v in enumerate(available) if v is not None]
        self._health_excluded_devices = self._filter_unhealthy_devices(available_devices)
        if self._health_excluded_devices:
            available = tuple(None if i in self._health_excluded_devices else v for i, v in enumerate(available))
            available_devices = [i for i, v in enumerate(available) if v is not None]
        SimpleCommandProcess.initialize_device_locks(available_devices, self.mp_manager)
        return available

    def _filter_unhealthy_devices(self, device_ids):
        """DSMI 健康态过滤（仅在线板态）：OFF 及未知健康码的卡不进调度池
        （授权域/派发池/CSV 校验域同步收缩）；OK/WARNING 直接可用，
        IMPORTANT/CRITICAL 打告警放行（结果可信度由告警留痕）；单卡查询失败
        保留该卡；查询能力缺失（无驱动/.so——asys 同源层）打告警不做检查。
        CPU 运行（--backend cpu 或 torch_npu 缺失回落）时池内是 CPU 槽位而非
        NPU id，不做健康检查。"""
        if not device_ids:
            return ()
        if not self.switches.mode.is_online_board():
            return ()
        if getattr(self.switches, "compile_only", False) or getattr(self.switches, "backend", None) == "npusim":
            return ()
        if getattr(self.switches, "force_cpu", False):
            return ()
        if self.switches.no_prof and self.switches.manual_data_dirs:
            return ()
        states = DSMIInterface().get_device_health_states(device_ids)
        if states is None:
            logging.warning("Device health check skipped (DSMI unavailable); all detected devices scheduled")
            return ()
        _ALARM_STATES = (DSMI_HEALTH_STATE.IMPORTANT, DSMI_HEALTH_STATE.CRITICAL)
        alarmed = {d: h for d, h in states.items() if h in _ALARM_STATES}
        if alarmed:
            detail = {d: getattr(h, "name", h) for d, h in sorted(alarmed.items())}
            logging.warning(
                f"Device(s) with health alarm scheduled anyway: {detail}; "
                f"results may be unreliable (check dsmi_get_device_error)"
            )
        schedulable = (
            DSMI_HEALTH_STATE.OK,
            DSMI_HEALTH_STATE.WARNING,
            DSMI_HEALTH_STATE.IMPORTANT,
            DSMI_HEALTH_STATE.CRITICAL,
        )
        unhealthy = {d: h for d, h in states.items() if h is not None and h not in schedulable}
        if unhealthy:
            detail = {d: getattr(h, "name", h) for d, h in sorted(unhealthy.items())}
            logging.warning(f"Unhealthy device(s) excluded from scheduling: {detail}")
            return tuple(sorted(unhealthy))
        return ()

    # ═══ 多卡 task 派发 ═══

    def _try_push_multi_device_task(self, task: Task) -> bool:
        """多卡任务派发：先做用例级设备解析（CSV device_ids ∩ 可用池），再按
        主卡路由到对应池。

        交集语义：CSV 声明与 CLI 池（--dev/--devices）+ 健康剔除取交集——
        剩余 ≥2 按交集截断继续；剩余 ≤1 判失败（多卡用例不降级单卡，
        hccl 通信域在单卡上无意义）。True=已派发至 worker；False=暂无空闲
        （已回队）或已判死。
        """
        grantable = {i for i, locked in enumerate(self.device_locks) if locked is not None}
        effective, failure = intersect_device_ids(task.device_ids, grantable)
        if failure:
            self._fail_task_without_worker(task, failure)
            return False
        if len(effective) != len(task.device_ids):
            logging.warning(
                f"Task {task.label}: device_ids {list(task.device_ids)} ∩ available devices "
                f"{sorted(grantable)} = {effective}; proceeding with {len(effective)} devices"
            )
            task.device_ids = effective
            # 结果 CSV 回显实跑设备集（截断前的声明值只留在 warning 日志）
            task.testcase.device_ids = tuple(effective)
        primary = task.device_ids[0]
        pg = self.process_groups.get(primary)
        if pg is None:
            self._fail_task_without_worker(
                task,
                f"primary device {primary} outside dispatch pool {sorted(self.process_groups)}",
            )
            return False
        if pg.idle_count() <= 0:
            self.task_keeper.insert(task)
            return False
        pg.push(task)
        return True

    def _fail_task_without_worker(self, task: Task, reason: str) -> None:
        """调度期判死：不占用 worker，走运行期失败同路径落结果行并计数。"""
        message = f"Unavailable device(s) requested via CSV 'device_ids' field: {reason}"
        logging.error(f"Task {task.label} failed at dispatch: {message}")
        output = self.handle_error(task, "fail", (0, RuntimeError(message)))
        if output:
            self.write_row(output)
        self.task_keeper.done(task.task_id)
        self.testcase_complete()

    # ═══ 用例解析 ═══

    def _filter_testcases(self):
        """聚合层筛选：enabled/soc、名称、索引、
        算子、优先级、rerun；count 抽样。选择器状态天然在 instance（全局视角），
        跨表重名在 _check_duplicate_case。"""
        kept = []
        for testcase_idx, tc in enumerate(self.flatten_testcases):
            if not _check_testcase_enabled(tc):
                continue
            if not _check_testcase_name_selection(tc.testcase_name):
                continue
            if not _check_testcase_indexes_selection(getattr(tc, "_csv_row_index", testcase_idx)):
                continue
            if not _check_testcase_operator_selection(tc.op_name):
                continue
            if not _check_testcase_priority_selection(tc.priority):
                continue
            if not _check_testcase_rerun(tc):
                continue
            kept.append(tc)
        self.flatten_testcases = kept
        self._apply_selection_order()
        count = getattr(self.switches, "selected_testcase_count", -1)
        if 0 < count < len(self.flatten_testcases):
            import random

            logging.info(f"Selecting {count} cases from all testcases")
            all_indexes = random.sample(tuple(range(len(self.flatten_testcases))), k=count)
            self.flatten_testcases = [tc for idx, tc in enumerate(self.flatten_testcases) if idx in all_indexes]

    def _apply_selection_order(self):
        """-t/--ti 圈选时按传入参数顺序重排执行序（其余选择器仍按 CSV 序）。

        key 同时存于 self._selection_order_key——通路 build_tasks 经
        _execution_sorted 消费（aclnn/e2e 原先固定按 _csv_row_index 重排会抵消此处重排）。"""
        names = list(get_global_storage().selected_testcases or [])
        indexes = list(get_global_storage().selected_testcase_indexes or [])
        if names:
            order = {name: pos for pos, name in enumerate(names)}
            self._selection_order_key = lambda tc: order.get(tc.testcase_name, len(order))
            self.flatten_testcases.sort(key=self._selection_order_key)
            logging.info("Testcases reordered by -t argument order")
        elif indexes:
            order = {idx: pos for pos, idx in enumerate(indexes)}
            self._selection_order_key = lambda tc: order.get(getattr(tc, "_csv_row_index", -1), len(order))
            self.flatten_testcases.sort(key=self._selection_order_key)
            logging.info("Testcases reordered by --ti argument order")

    def _execution_sorted(self, testcases):
        """通路 build_tasks 统一排序入口：-t/--ti 圈选时按传入序，否则按 CSV 行序。"""
        return execution_sorted(self._selection_order_key, testcases)

    def _validate_only(self):
        self.parse_cases()
        valid_count = 0
        invalid_cases = []
        reason_counts = {}
        for tc in sorted(self.flatten_testcases, key=lambda t: t.testcase_name):
            if tc.is_valid:
                valid_count += 1
            else:
                reason = tc.fail_reason or "UNKNOWN"
                invalid_cases.append((tc.testcase_name, getattr(tc, "api_name", ""), reason))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        invalid_count = len(invalid_cases)
        total = valid_count + invalid_count
        logging.info(f"Validate Complete: {total} total, {valid_count} valid, {invalid_count} invalid")
        if reason_counts:
            logging.info("Invalid reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                logging.info(f"  {reason}: {count}")
        if invalid_cases:
            logging.info("Invalid cases:")
            for name, api, reason in invalid_cases:
                api_str = f" ({api})" if api else ""
                logging.info(f"  {name}{api_str}: {reason}")
        if self.switches.output_file_name:
            self._write_validate_result(invalid_cases, valid_count, invalid_count)
        if invalid_count > 0:
            logging.warning(f"{invalid_count} testcase(s) have validation errors")
        else:
            logging.info("All testcases passed validation")

    def _write_validate_result(self, invalid_cases, valid_count, invalid_count):
        result_path = self.switches.output_file_name
        if not result_path.endswith(".csv"):
            result_path += ".csv"
        header = ("testcase_name", "api_name", "status", "fail_reason")
        append = False
        if self.switches.append_mode and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            existing_header = self._read_existing_header(result_path)
            append = existing_header is not None and tuple(existing_header) == header
            if not append:
                logging.warning(f"Append mode: existing file header does not match, overwriting {result_path}")
        with open(result_path, "a" if append else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(header)
            for tc in sorted(self.flatten_testcases, key=lambda t: t.testcase_name):
                api = getattr(tc, "api_name", "")
                if tc.is_valid:
                    writer.writerow((tc.testcase_name, api, "VALID", ""))
                else:
                    writer.writerow((tc.testcase_name, api, "INVALID", tc.fail_reason or ""))
        logging.info(f"Validate result written to {result_path}")

    def _load_cases(self, testcase_file: str):
        self._load_case_from_table(testcase_file)

    def _check_duplicate_case(self):
        testcase_names = set()
        duplicates = set()
        for t in self.flatten_testcases:
            if t.testcase_name in testcase_names:
                duplicates.add(t)
            else:
                testcase_names.add(t.testcase_name)
        for t in duplicates:
            logging.warning(f"Testcase duplicate: {t.testcase_name}. Skip it ...")
            self.flatten_testcases.remove(t)

    def _load_case_from_table(self, testcase_path: str):
        # iter_tables 统一分发（.xlsx/.zip/.csv——zip 展开为 N 张 Table，唯一入口）；
        # Factory 参数化：testcase_class 经 _testcase_class_for_mode 解析；
        # skip_validate：校验后置于 parse_cases 的筛选之后（未选中行不付
        # 校验代价，如 torch.ops 扩展包的 import + JIT 编译）。
        testcase_class = self._testcase_class_for_mode()
        for table in iter_tables(testcase_path, getattr(self.switches, "sheet", None)):
            factory = UniversalTestcaseFactory(table, testcase_class, skip_validate=True)
            self.flatten_testcases.extend(factory.get())
            self.case_original_headers = list(dict.fromkeys(self.case_original_headers + factory.header))

    def _testcase_class_for_mode(self) -> type:
        """testcase_class 解析（类属性契约）：实例属性优先（类属性或
        property——通路按 test_mode 动态切）；未声明时经注册表查
        （common 不 import 通路——通路注册时声明）。"""
        tc_class = getattr(self, "testcase_class", None)
        if tc_class is not None:
            return tc_class
        entry = _INSTANCE_REGISTRY.get(self.switches.test_mode)
        tc_class = getattr(entry, "testcase_class", None) if entry else None
        if tc_class is None:
            raise RuntimeError(
                f"pathway {type(self).__name__} must set class attribute testcase_class "
                "(TestcaseBase subclass) for the aggregation layer"
            )
        return tc_class

    # ═══ 结果 CSV ═══

    def _resolved_output_sheet(self):
        """Worksheet name to embed in the default output name (xlsx only)."""
        src = self.switches.input_files[0] if self.switches.input_files else ""
        if not src.lower().endswith((".xlsx", ".xlsm")):
            return None
        from .testcase import resolved_sheet

        return resolved_sheet(src, getattr(self.switches, "sheet", None))

    def _prepare_output_titles(self):
        first_testcase = next(iter(self.flatten_testcases))
        self.case_result_titles = self.output_titles(first_testcase, self.case_original_headers)

    def output_titles(self, testcase: TestcaseBase, case_original_headers: list) -> tuple:
        """结果列组装：可见输入 header（身份列前置）+ 通路 result_titles()。"""
        input_titles = testcase.get_all_visible_headers()

        result_titles = self.result_titles()
        input_titles = self._reorder_input_titles(input_titles)
        self.case_input_title = tuple(input_titles)
        self.case_result_title = tuple(result_titles)
        return self._compose_header()

    _FRONT_IDENTITY_HEADERS = ("testcase_name", "op_name", "api_name")

    def _reorder_input_titles(self, input_titles: tuple) -> tuple:
        """Move identity headers (testcase_name, op_name/api_name) to the front,
        preserving the relative order of the remaining titles. Returns a tuple;
        caller reads ``self._front_input_count`` for the count of moved headers."""
        self._front_input_count = 0
        if not input_titles or input_titles[0] != "testcase_name":
            return input_titles
        front = ["testcase_name"]
        rest = list(input_titles[1:])
        for h in self._FRONT_IDENTITY_HEADERS[1:]:
            if h in rest:
                rest.remove(h)
                front.append(h)
        self._front_input_count = len(front)
        return tuple(front + rest)

    def _compose_header(self) -> tuple:
        return self._assemble_row(self.case_input_title, self.case_result_title)

    def _assemble_row(self, inputs: tuple, results: tuple) -> tuple:
        n = self._front_input_count
        if n > 0 and len(inputs) > n:
            return (*inputs[:n], *results, *inputs[n:])
        return (*inputs, *results)

    def _error_row(self, task: Task, main: str, details: str) -> tuple:
        """执行 task 错误行（主列失败态 + 明细，其余列占位）。"""
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. But got {task.testcase.__class__}")
        results = [main] * len(self.case_result_title)
        if len(results) > 0:
            results[0] = details  # store details.
        inputs = task.testcase.pick_data(self.case_input_title)
        return self._assemble_row(inputs, tuple(results))

    def _resolve_precision_status_idx(self, header: tuple) -> Optional[int]:
        try:
            return header.index("precision_status")
        except ValueError:
            return None

    def _update_summary(self, row: tuple):
        idx = self._precision_status_idx
        if idx is None or idx >= len(row):
            return
        status = str(row[idx]).upper()
        if status == "PASS":
            self.pass_count += 1
        elif status in {"FAIL", "FAILURE", "PROFILE_CRASH", "COMPILE_FAILURE", "TIMEOUT"} or status.startswith(
            ("CRASHED AT PROFILING STAGE:", "ERROR:")
        ):
            self.fail_count += 1
        else:
            self.other_count += 1
        # 用例完成行（主日志检索锚点：grep 用例名即得结论）
        name = str(row[0]) if self.case_input_title and self.case_input_title[0] == "testcase_name" else "<unknown>"
        logging.info(f"Testcase {name} finished: precision_status={status}")

    # ═══ 调度循环内件 ═══

    def _update_processes(self):
        for pg in self.process_groups.values():
            pg.update()

    def _close_idle_processes(self):
        if self.task_keeper.empty():
            for pg in self.process_groups.values():
                pg.close_idles()

    def _output_progress(self):
        if self.switches.progress_output:
            parent = os.path.dirname(self.switches.progress_output)
            if parent:
                os.makedirs(parent, exist_ok=True)
            now = time.time()
            with open(self.switches.progress_output, "w") as f:
                f.write(
                    f"StartAt:{str(self.start_timestamp)}\n"
                    f"ElapsedTime:{str(now - self.start_timestamp)}\n"
                    f"TotalCases:{self.total_case_count}\n"
                    f"CompletedCases:{self.completed_case_count}"
                )

    def _most_idle_group_in_pool(self, pool: str) -> Optional[ProcessGroup]:
        """池内最空闲组（无空闲返回 None）。"""
        max_idle, max_pg = 0, None
        for pg in self.process_groups.values():
            if pg.pool != pool:
                continue
            idles = pg.idle_count()
            if idles > max_idle:
                max_idle, max_pg = idles, pg
        return max_pg

    def _push_task_to_process(self):
        """按池分发：每池独立填本池空闲 worker（pop(pool=) 池键路由，编译/
        执行 task 不进对方进程池）。执行池优先派发；多卡 task 派发成功即置
        闸门，本轮不再派发其它池。

        单值 device_ids（CSV "3" 指定单卡）：路由到指定设备组而非最闲组
        （文档语义恢复）；指定组无空闲时延后（deferred 回队首，不阻塞同池
        其它 task），指定设备不在池内 → 调度期判死。
        """
        if self._multi_device_running:
            return
        pools = list(dict.fromkeys(pg.pool for pg in self.process_groups.values()))
        pools.sort(key=lambda pool: pool != "profile")
        for pool in pools:
            deferred = []
            while True:
                target = self._most_idle_group_in_pool(pool)
                if target is None:
                    break
                task = self.task_keeper.pop(target.pool)
                if task is None:
                    break
                # dl=3 缺字段提示：派发 choke point 逐 case 打（名字去重，含
                # 多卡/pin/普通三条 push 路径与 kernel 双池）
                self._warn_incomplete_batch_relation(task.testcase)
                if task.is_multi_device:
                    if self._try_push_multi_device_task(task):
                        self._multi_device_running = True
                        for t in reversed(deferred):
                            self.task_keeper.insert(t)
                        return
                    break
                if task.device_ids:
                    # CSV 单值 pin：device_ids=[d] 固定路由到 d 的设备组
                    pinned = self.process_groups.get(task.device_ids[0])
                    if pinned is None or pinned.pool != task.pool:
                        self._fail_task_without_worker(
                            task,
                            f"pinned device {task.device_ids[0]} not present or excluded by "
                            f"--devices/health check (available devices: {self._grantable_devices()})",
                        )
                        continue
                    if pinned.idle_count() <= 0:
                        deferred.append(task)
                        continue
                    pinned.push(task)
                    continue
                target.push(task)
            for t in reversed(deferred):
                self.task_keeper.insert(t)

    def _grantable_devices(self):
        """调度期可用设备集（device_locks 全量，供报错信息引用）。"""
        return sorted(i for i, locked in enumerate(self.device_locks) if locked is not None)

    def _handle_completed_process(self):
        completed_process = []
        for pg in self.process_groups.values():
            completed_process.extend(pg.completed())
        for pt_pair in completed_process:
            proc, task = pt_pair
            if task.is_multi_device:
                self._multi_device_running = False
                self._handle_multi_device_task_result(task, proc)
            else:
                self._handle_task_result(task, proc)
                if task.is_final:
                    self.testcase_complete()
            self.task_keeper.done(task.task_id)  # 解锁 depends_on 它的 task

    def _handle_task_result(self, task: Task, proc: SimpleCommandProcess):
        result = proc.get_result()
        pid = proc.get_pid()
        if result is None:
            output_data = self.handle_error(task, "none", None)
        elif isinstance(result, SystemError):
            last_stage = proc.current_stage()  # recycle 会重置 data，崩溃 stage 须先取
            proc.recycle()
            output_data = self.handle_error(task, "crash", (last_stage, pid, result))
        elif isinstance(result, RuntimeError):
            proc.recycle()
            output_data = self.handle_error(task, "fail", (pid, result))
        else:
            self._collect_batch_summary(task, result)
            output_data = self._assemble_result_row(task, self.handle_result(task, result))
            if self.switches.proc_no_reuse or self.recycle_worker(task, result):
                proc.close()
                proc.resurrect()
        if output_data:
            self.write_row(output_data)

    def _handle_multi_device_task_result(self, task: Task, proc: SimpleCommandProcess):
        result = proc.get_result()
        pid = proc.get_pid()
        if isinstance(result, (SystemError, RuntimeError)):
            proc.recycle()
            if isinstance(result, SystemError):
                output_data = self.handle_error(task, "crash", ("MultiDevice", pid, result))
            else:
                output_data = self.handle_error(task, "fail", (pid, result))
        elif result is not None:
            self._collect_batch_summary(task, result)
            output_data = self._assemble_result_row(task, self.handle_result(task, result))
            # 多卡 worker 不复用（跨卡资源随进程退出统一回收）
            proc.close()
            proc.resurrect()
        else:
            output_data = self.handle_error(task, "none", None)
        if output_data:
            self.write_row(output_data)
        self.testcase_complete()

    def _assemble_result_row(self, task: Task, results: Optional[tuple]) -> Optional[tuple]:
        """结果列 + 输入列组装：handle_result 产出结果列，输入列（用例身份与
        声明字段）由基类统一拼装——通路 hook 只关心结果。"""
        if results is None:
            return None

        inputs = task.testcase.pick_data(self.case_input_title)
        return self._assemble_row(inputs, results)

    def _get_head_commit_id(self):
        if self._commit_id is not None:
            return
        if shutil.which("git") is None:
            self._commit_id = "UNKNOWN"
            return

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, shell=False, check=False
            )
            if result.returncode != 0:
                self._commit_id = "UNKNOWN"
            else:
                self._commit_id = result.stdout.split("\n")[0]
        finally:
            if self._commit_id is None:
                self._commit_id = "UNKNOWN"
        return


# ── 聚合层选择器（parse_cases 侧消费）──


def execution_sorted(selection_order_key, testcases):
    """通路 build_tasks 统一排序——-t/--ti 圈选时按传入参数序，否则按 CSV 行序。

    selection_order_key 为 None（未圈选/伪实例）时回退 CSV 行序，供 build_tasks
    以 getattr(self, "_selection_order_key", None) 免实例化调用。"""
    if selection_order_key is not None:
        return sorted(testcases, key=selection_order_key)
    return sorted(testcases, key=lambda t: getattr(t, "_csv_row_index", -1))


def _check_testcase_rerun(testcase_struct: TestcaseBase) -> bool:
    re_run_titles = testcase_struct.supported_rerun_title()
    if get_global_storage().rerun_targets:
        for title in get_global_storage().rerun_targets:
            if title in re_run_titles:
                original_result = testcase_struct.original_dict[title]
                try:
                    float(original_result)
                except Exception:
                    if original_result not in ("PASS",):
                        return True
            else:
                raise RuntimeError(f"Unsupported rerun targets: {title}")
    else:
        return True
    return False


def _check_testcase_enabled(testcase_struct: TestcaseBase) -> bool:
    if not testcase_struct.is_enabled:
        logging.debug(f"Testcase {testcase_struct.testcase_name} skipped bcz it's disabled")
        return False
    current_soc = get_global_storage().short_soc_version
    if testcase_struct.soc_series and current_soc:
        enabled_soc = [s for s in testcase_struct.soc_series if not s.startswith("-")]
        disabled_soc = [s[1:] for s in testcase_struct.soc_series if s.startswith("-")]
        if not enabled_soc and not disabled_soc:
            enabled = True
        elif not disabled_soc:
            enabled = current_soc in enabled_soc
        elif not enabled_soc:
            enabled = current_soc not in disabled_soc
        else:
            enabled = current_soc in enabled_soc and current_soc not in disabled_soc
        if not enabled:
            logging.debug(
                f"Testcase {testcase_struct.testcase_name} skipped bcz it's disabled in current soc {current_soc}."
            )
        return enabled
    return True


def _check_testcase_name_selection(testcase_name: str) -> bool:
    if get_global_storage().selected_testcases:
        return testcase_name in list(get_global_storage().selected_testcases)
    return True


def _check_testcase_indexes_selection(testcase_idx: int) -> bool:
    if get_global_storage().selected_testcase_indexes:
        return testcase_idx in get_global_storage().selected_testcase_indexes
    return True


def _check_testcase_operator_selection(testcase_op_name: str) -> bool:
    storage = get_global_storage()
    selected = storage.selected_operators
    excluded = storage.excluded_operators
    return (not selected or testcase_op_name in selected) and (not excluded or testcase_op_name not in excluded)


def _check_testcase_priority_selection(priority: int) -> bool:
    if not get_global_storage().priorities:
        return True
    return any(p[0] <= priority <= p[1] for p in get_global_storage().priorities)


def _rename_duplicate_case_name(ori_name: str, op_name: str, conflict_names: set):
    i = 0
    while True:
        new_name = f"{ori_name}_{op_name}_dup{i}"
        if new_name in conflict_names:
            i = i + 1
            continue
        logging.warning(f"Detected duplicate testcase name: {ori_name}. Rename it to {new_name}")
        return new_name
