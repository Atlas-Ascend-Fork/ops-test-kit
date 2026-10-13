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
Interface for msprof
"""

__all__ = ["MsProfiler"]


# Standard Packages
import contextlib
import csv
import ctypes
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from typing import List, Optional

# Third-Party Packages
from .base import KernelDetails, KernelInfo, Profiler, ProfileResult
from .types import (
    ACL_ERROR_DESC_DICT,
    AclProfType,
    AiCoreProfMetrics,
    MsprofCompactInfo,
    MsProfOpDfx,
    MsprofTaskType,
    TtkMsProfType,
)

# msprof.py export (summary/timeline) 的挂起上限——单算子级数据导出通常为
# 秒级~分钟级，600s 足够宽裕；防 worker 被卡死（subprocess.run timeout）。
_MSPROF_EXPORT_TIMEOUT_S = 600

# 落盘 CSV 的算子任务类型（AI Core / Vector 族）——op_summary 的 Task Type
# 与 task_time 的 kernel_type 列共用此集合
_KERNEL_TASK_TYPES = frozenset(
    {
        "AI_CORE",
        "AIV_SQE",
        "AI_VECTOR_CORE",
        "MIX_AIC",
        "MIX_AIV",
        "KERNEL_MIX_AIC",
        "KERNEL_MIX_AIV",
        "KERNEL_AIVEC",
        "KERNEL_AICORE",
    }
)


def _resolve_msprof_tool() -> Optional[str]:
    msprof = f"{os.getenv('ASCEND_OPP_PATH', '')}/../tools/profiler/profiler_tool/analysis/msprof/msprof.py"
    if not os.path.exists(msprof):
        logging.warning(f"File [{msprof}] is not found. Please install CANN-toolkit RUN package.")
        return None
    return msprof


def export_msprof_summary(prof_dir, exports=("summary", "timeline"), timeout=_MSPROF_EXPORT_TIMEOUT_S):
    """msprof.py 子进程导出 PROF_* 原始数据为 CSV（谁采集谁解析的导出侧）。

    MsProfiler.__exit__ 与 geir/perf_parse 共用；sys.executable 而非
    "python3"（venv/conda 下 PATH 解释器可能缺 msprof 依赖）；超时记日志
    不抛（__exit__ 必须保持干净）。msprof.py 缺失（未装 CANN-toolkit
    RUN 包）warning 后静默返回。
    """
    msprof = _resolve_msprof_tool()
    if msprof is None:
        return
    for t in exports:
        try:
            child = subprocess.run(
                [sys.executable, msprof, "export", t, "-dir", prof_dir],
                shell=False,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logging.error(f"call msprof.py to export {t} timed out after {timeout}s")
            continue
        if child.returncode != 0:
            logging.error(f"call msprof.py to export {t} failed: {child.stdout}, {child.stderr}")


def _stat_mtime(path: str):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0  # 消失的文件排到候选末尾


def find_newest_csv(prof_dir, prefixes) -> Optional[str]:
    """prof_dir（递归）下匹配任一前缀的最新 mtime CSV。

    mtime 降序、文件名降序 tie-break（同秒场景，03d1234 geir 语义）——
    prof 目录跨运行复用时按路径序会永远命中最旧一档文件。
    """
    matches = []
    for root, _, files in os.walk(prof_dir):
        for f in files:
            if f.endswith(".csv") and any(f.startswith(p) for p in prefixes):
                matches.append(os.path.join(root, f))
    if not matches:
        return None
    return max(matches, key=lambda p: (_stat_mtime(p), os.path.basename(p)))


def _extract_durations(csv_path, duration_col, type_col) -> List[float]:
    """按 KERNEL_TASK_TYPES 过滤后抽取时长列（列缺失/坏值行跳过）。"""
    results = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(type_col, "") not in _KERNEL_TASK_TYPES:
                continue
            with contextlib.suppress(ValueError, KeyError):
                results.append(float(row[duration_col]))
    return results


def read_task_durations(prof_dir) -> List[float]:
    """每 launch 的设备 Task Duration(us)——kernel/geir 通路的 op 级视图。

    候选 op_summary_（Task Duration(us)/Task Type）与 task_time_
    （task_time(us)/kernel_type）按 mtime 降序；空结果文件回退到下一候选
    （kernel 通路的 fall-through 语义：无 KERNEL_TYPE 行的文件视为无数据）。
    """
    if not os.path.isdir(prof_dir):
        return []
    candidates = []
    for root, _, files in os.walk(prof_dir):
        for f in files:
            if f.endswith(".csv") and f.startswith(("op_summary_", "task_time_")):
                candidates.append(os.path.join(root, f))
    candidates.sort(key=lambda p: (_stat_mtime(p), os.path.basename(p)), reverse=True)
    for path in candidates:
        if os.path.basename(path).startswith("op_summary_"):
            duration_col, type_col = "Task Duration(us)", "Task Type"
        else:
            duration_col, type_col = "task_time(us)", "kernel_type"
        try:
            durations = _extract_durations(path, duration_col, type_col)
        except OSError as e:
            logging.warning(f"Failed to read {path}: {e}")
            continue
        if durations:
            return durations
    return []


def _parse_op_statistic_row(row) -> Optional[KernelInfo]:
    """单行 op_statistic → KernelInfo；Avg<=0 / 空 OP Type 剔除。"""
    try:
        avg_us = float(row.get("Avg Time(us)", 0))
        total_us = float(row.get("Total Time(us)", 0))
        count = int(row.get("Count", 0))
        min_us = float(row.get("Min Time(us)", 0))
        max_us = float(row.get("Max Time(us)", 0))
    except (ValueError, TypeError):
        return None
    if avg_us <= 0:
        return None
    name = row.get("OP Type", "").strip()
    if not name:
        return None
    return KernelInfo(
        name=name,
        device_us=total_us,
        calls=count,
        avg_us=avg_us,
        max_us=max_us,
        min_us=min_us,
    )


def read_op_statistic(csv_path) -> List[KernelInfo]:
    """op_statistic CSV → KernelInfo 列表（op 级聚合视图）。"""
    kernels = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            info = _parse_op_statistic_row(row)
            if info is not None:
                kernels.append(info)
    return kernels


class MsProfiler(Profiler):
    """
    interfaces for msprof
    """

    def __init__(
        self,
        device_id: int,
        result_path: str,
        prof_type: TtkMsProfType,
        extra_acl_prof_type: Optional[List[AclProfType]] = None,
        start_step: int = 1,
        is_model: bool = False,
    ):
        self._dvc_id: int = device_id
        self._result_path: str = result_path
        self._prof_type = prof_type
        self._start_step = start_step
        self._current_step = 0
        self._is_model: bool = is_model
        self._prof_cfg: Optional[ctypes.c_void_p] = None
        self._data_type_cfg = self._build_data_type_config(prof_type, extra_acl_prof_type)
        self._prof_inited: bool = False
        self._prof_started: bool = False
        if self._is_model or (prof_type == TtkMsProfType.NONE and not extra_acl_prof_type):
            self._msprof_dll = None
        else:
            if os.path.exists(result_path):
                shutil.rmtree(result_path)
            self._msprof_dll = ctypes.CDLL("libmsprofiler.so")
        self._prof_api_dll = None  # initialize when needed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._msprof_dll is not None:
            self._stop()
            self._analysis_profile_data()

    def step(self):
        if self._msprof_dll is None:
            return
        if self._current_step == self._start_step:
            self._start()
        self._current_step += 1

    def report_op_dfx(self, op_dfx: MsProfOpDfx):
        if self._is_model:
            return
        if self._msprof_dll is None:
            return
        if not self._prof_started:
            # start_step 隔离（run>1 首轮不进采集窗口）时，首轮 launch 时尚无
            # aclprof 会话可标注——no-op（该轮本就不在采集窗口内）
            return
        if self._prof_api_dll is None:
            self._prof_api_dll = ctypes.CDLL("libprofapi.so")
        api = getattr(self._prof_api_dll, "MsprofReportCompactInfo", None)
        if api is None:
            logging.warning("MsProf API [MsprofReportCompactInfo] is not defined.")
            return

        timestamp = self._get_sys_cycle()
        kernel_name = self._get_hash_id(op_dfx.kernel_name)
        task_type = self._get_op_task_type(op_dfx.core_type, op_dfx.task_ration)
        compact_info = MsprofCompactInfo(timestamp, kernel_name, task_type, op_dfx.block_dim)
        start_time = time.time()

        api.restype = ctypes.c_int32
        api.argtypes = [ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32]
        rt_error = api(1, ctypes.c_void_p(ctypes.addressof(compact_info)), ctypes.sizeof(compact_info))
        self.parse_error(rt_error, "MsprofReportCompactInfo", f"costs {round(time.time() - start_time, 3)} seconds")

    @staticmethod
    def parse_error(acl_ret: ctypes.c_uint64, api_name: str, extra_info: str):
        # Convert aclError to int if received ctypes object
        if isinstance(acl_ret, ctypes.c_uint64):
            acl_ret = acl_ret.value
        elif isinstance(acl_ret, int):
            pass
        else:
            raise TypeError(f"Invalid acl_ret type {type(acl_ret)}")
        # Success
        if acl_ret == 0x0:
            logging.debug(f"MsProf API Call {api_name}() Success, {extra_info}")
            return
        # Fail
        desc = ACL_ERROR_DESC_DICT.get(acl_ret, str(acl_ret))
        raise RuntimeError(f"MsProf API Call {api_name}() Failed: {desc}, {extra_info}")

    def _start(self):
        self._acl_prof_init()
        self._acl_prof_create_config(self._data_type_cfg)
        self._acl_prof_start()

    def _stop(self):
        self._acl_prof_stop()
        self._acl_prof_destroy_config()
        self._acl_prof_finalize()

    def _acl_prof_init(self):
        """
        aclError aclprofInit(const char *profilerResultPath, size_t length);
        """
        c_path = self._result_path.encode("UTF-8")
        # length must be the encoded byte count: non-ASCII paths (e.g. Chinese
        # directory names) make len(str) != len(bytes) and hand C a wrong
        # buffer size.
        c_size = ctypes.c_size_t(len(c_path))
        self._api_call("aclprofInit", None, c_path, c_size)
        self._prof_inited = True

    def _acl_prof_create_config(self, data_type_cfg: int):
        """
        aclprofConfig *aclprofCreateConfig(
            uint32_t *deviceIdList, uint32_t deviceNums,
            aclprofAicoreMetrics aicoreMetrics,
            aclprofAicoreEvents *aicoreEvents,  # always nullptr
            uint64_t dataTypeConfig);
        """
        c_device_id_lst = (ctypes.c_uint32 * 1)(self._dvc_id)
        c_device_nums = ctypes.c_uint32(1)
        c_aicore_metrics = ctypes.c_int(AiCoreProfMetrics.PIPE_UTILIZATION.value)
        c_data_type_config = ctypes.c_uint64(data_type_cfg)
        self._prof_cfg = self._api_call_with_ptr_return(
            "aclprofCreateConfig", None, c_device_id_lst, c_device_nums, c_aicore_metrics, None, c_data_type_config
        )

    def _acl_prof_start(self):
        """
        aclError aclprofStart(const aclprofConfig *profilerConfig);
        """
        self._api_call("aclprofStart", None, self._prof_cfg)
        self._prof_started = True

    def _acl_prof_stop(self):
        """
        aclError aclprofStop(const aclprofConfig *profilerConfig);
        """
        if self._prof_started:
            self._api_call("aclprofStop", None, self._prof_cfg)
            self._prof_started = False

    def _acl_prof_destroy_config(self):
        """
        aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig)
        """
        if self._prof_cfg is not None:
            self._api_call("aclprofDestroyConfig", None, self._prof_cfg)
            self._prof_cfg = None

    def _acl_prof_finalize(self):
        """
        aclError aclprofFinalize();
        """
        if self._prof_inited:
            self._api_call("aclprofFinalize", None)
            self._prof_inited = False

    def _get_hash_id(self, hash_info: str):
        api = getattr(self._prof_api_dll, "MsprofGetHashId", None)
        if api is None:
            raise RuntimeError("MsProf API [MsprofGetHashId] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        api.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        hash_info = hash_info.encode("utf-8")
        rt_val = api(hash_info, len(hash_info))
        logging.debug(
            f"MsProf API Call MsprofGetHashId() Success, hash_info: {hash_info}. "
            f"costs {round(time.time() - start_time, 3)} seconds"
        )
        return rt_val

    def _get_sys_cycle(self):
        api = getattr(self._prof_api_dll, "MsprofSysCycleTime", None)
        if api is None:
            raise RuntimeError("MsProf API [MsprofSysCycleTime] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        api.argtypes = []
        rt_val = api()
        logging.debug(
            f"MsProf API Call MsprofSysCycleTime() Success, costs {round(time.time() - start_time, 3)} seconds"
        )
        return rt_val

    def result(self, repeat_count=1) -> ProfileResult:
        """谁采集谁解析：解析自家导出的 CSV → ProfileResult。

        视图按 prof_type：
        - OP：read_task_durations 每 launch 时长，elapsed = median
          （warmup 处理归通路——kernel 的 start_step 首轮隔离是编排，
          不在采集器职责内）
        - API：op_statistic 聚合，elapsed = Σ Avg（单轮口径——Avg 已被
          msprof 按实际采集次数平均），total_device = Σ Avg × repeat
        - NONE / is_model（未采集）或无数据：空结果
        """
        if self._msprof_dll is None:
            return ProfileResult(source="disabled")
        try:
            if self._prof_type == TtkMsProfType.OP:
                durations = read_task_durations(self._result_path)
                if not durations:
                    return ProfileResult(source="msprof")
                return ProfileResult(
                    elapsed_us=round(statistics.median(durations), 3),
                    kernel_details=None,
                    source="msprof",
                )
            csv_path = find_newest_csv(self._result_path, ("op_statistic_",))
            if not csv_path:
                return ProfileResult(source="msprof")
            kernels = read_op_statistic(csv_path)
            total_avg = round(sum(k.avg_us for k in kernels), 3)
            if total_avg <= 0:
                return ProfileResult(source="msprof")
            return ProfileResult(
                elapsed_us=total_avg,
                kernel_details=KernelDetails(
                    kernels=kernels,
                    total_device_us=total_avg * max(repeat_count, 1),
                    total_cpu_us=0.0,
                ),
                source="msprof",
            )
        except Exception as e:
            logging.warning(f"MsProfiler result parse failed under {self._result_path}: {e}")
            return ProfileResult(source="msprof")

    def _analysis_profile_data(self):
        if not os.path.exists(self._result_path):
            return
        export_msprof_summary(self._result_path)

    def _api_call(self, api_name: str, extra_log: Optional[str], *api_args):
        if extra_log is None:
            extra_log = ""
        api = getattr(self._msprof_dll, api_name, None)
        if not api:
            raise RuntimeError(f"MsProf API [{api_name}] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        rt_error = api(*api_args)
        self.parse_error(rt_error, api_name, f"{extra_log} costs {round(time.time() - start_time, 3)} seconds")

    def _api_call_with_ptr_return(self, api_name: str, extra_log: Optional[str], *api_args) -> ctypes.c_void_p:
        if extra_log is None:
            extra_log = ""
        api = getattr(self._msprof_dll, api_name, None)
        if not api:
            raise RuntimeError(f"MsProf API [{api_name}] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_void_p
        rt_ptr = api(*api_args)
        if not rt_ptr:
            raise RuntimeError(f"MsProf API {api_name} return nullptr. {extra_log}")
        logging.debug(
            f"MsProf API Call {api_name}() Success, {extra_log} costs {round(time.time() - start_time, 3)} seconds"
        )
        return ctypes.c_void_p(rt_ptr)

    @staticmethod
    def _build_data_type_config(
        prof_type: TtkMsProfType, extra_acl_prof_type: Optional[List[AclProfType]] = None
    ) -> int:
        if not extra_acl_prof_type:
            return prof_type.value
        data_type_cfg = prof_type.value
        for acl_pt in extra_acl_prof_type:
            data_type_cfg |= acl_pt.value
        return data_type_cfg

    @staticmethod
    def _get_op_task_type(core_type, task_ration):
        if core_type in ("AiCore", "CubeCore"):
            return MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
        if core_type in ("VectorCore",):
            return MsprofTaskType.MSPROF_TASK_TYPE_AIV.value
        # task_ration is the compiler JSON's "aic:aiv" ratio; guard against
        # None / malformed tuples so a DFX annotation cannot crash the launch
        # loop (report_op_dfx runs inside the per-repeat kernel launch try).
        if not task_ration or len(task_ration) < 2:
            logging.warning(
                "cannot resolve msprof task type: core_type=%r task_ration=%r; defaulting to AI_CORE",
                core_type,
                task_ration,
            )
            return MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
        if all(task_ration):
            return MsprofTaskType.MSPROF_TASK_TYPE_MIX_AIC.value
        if task_ration[1] == 0:
            return MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
        return MsprofTaskType.MSPROF_TASK_TYPE_AIV.value
