#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir/execute.py —— GEIR 图执行段。

- 输入 bin 落盘（None 跳过、TensorList 逐元素展开与 graph_builder 的
  elements 对齐）+ subprocess 执行 + os.pipe 数据协议 + 多轮确定性 MD5
  + 轮间设备状态清理 + perf 触发。
- 执行协议：数据通道 os.pipe()，读线程 _drain_pipe（64KB 管道缓冲防死锁），
  逐输出 [8B num_outputs][8B byte_count][data]*；stdout/stderr 留给 GE
  运行时日志（不与数据混流），Popen.communicate() 并行收两端。
- dl=1 时 run_time 轮：每轮重写输入 bin（防图算子原地写污染，Kernel
  inplace 逐轮恢复的同构防御）；--clear-ub/-l1/-l0 时轮间由父进程 RTS
  helper kernel 清（C++ 子进程执行中不可介入，语义等价"每轮执行前清"）。
"""

__all__ = ["GraphExecPlan", "GraphExecResult", "execute_graph"]

import contextlib
import glob
import hashlib
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ttk.utilities import is_4bit_dtype, pack_4bits

from ..common.comparison import invoke_pre_compare
from ..kernel.device_state import clear_device_state
from .output_parse import _parse_stdout
from .perf_parse import _parse_msprof_task_duration
from .testcase import GeirTestcase


@dataclass
class GraphExecPlan:
    """图执行段输入（worker 编排层装配）。"""

    binary: str
    config_path: str
    build_dir: str
    prof_path: str
    input_prefix: str


@dataclass
class GraphExecResult:
    """图执行段产出：最后一轮输出 + 确定性状态 + 设备耗时（µs）。"""

    outputs: List[np.ndarray] = field(default_factory=list)
    det_status: Optional[str] = None
    device_perf_us: Optional[float] = None


def execute_graph(
    testcase: GeirTestcase, dev_id: int, switches, plan: GraphExecPlan, pre_compare=None
) -> GraphExecResult:
    """逐轮执行编译出的 C++ 图测试程序（dl=1 时 run_time 轮）。

    pre_compare（双消费点之二）：dl=1 时逐轮作为确定性校验的规范化阶段——
    双半边签名（outputs + goldens），hash 只用 outputs 半边；GEIR 变换在轮间
    Python 侧，天然在 C++ msprof 采集进程之外（零窗口约束）。
    """
    result = GraphExecResult()
    _write_all_input_bins(testcase.input_arrays, plan.input_prefix)
    testcase.original_input_arrays = None
    deterministic = getattr(switches, "deterministic_level", 0) == 1
    run_count = switches.run_time if deterministic else 1
    md5_list = []
    try:
        for run_idx in range(run_count):
            clear_device_state(switches)  # 轮间父进程 RTS 清（子进程执行中不可介入）
            if run_idx > 0:
                _write_all_input_bins(testcase.input_arrays, plan.input_prefix)  # 防图算子原地写污染
            outputs, perf_us = _run_once(testcase, dev_id, switches, plan)
            result.outputs = outputs
            if plan.prof_path:
                result.device_perf_us = perf_us
            if deterministic:
                md5_list.append(_det_round_hash(testcase, outputs, testcase.golden_arrays, pre_compare))
    finally:
        _cleanup_bins(plan.input_prefix)
    result.det_status = _det_status(md5_list, deterministic, testcase.testcase_name)
    return result


def _snapshot_prof_entries(prof_path: str) -> frozenset:
    """执行前 prof_path 既有子目录快照（空路径返回空集）。"""
    if not prof_path or not os.path.isdir(prof_path):
        return frozenset()
    return frozenset(entry for entry in os.listdir(prof_path) if os.path.isdir(os.path.join(prof_path, entry)))


def _new_prof_dirs(prof_path: str, before: frozenset) -> list:
    """执行前后差集 → 本轮子进程新建的 PROF 目录（绝对路径）。

    msprof 导出只往既有 PROF 内写文件、不新建顶层目录，差集即本轮全部新 PROF。
    """
    if not prof_path or not os.path.isdir(prof_path):
        return []
    current = {entry for entry in os.listdir(prof_path) if os.path.isdir(os.path.join(prof_path, entry))}
    return [os.path.join(prof_path, entry) for entry in sorted(current - before)]


def _run_once(testcase, dev_id, switches, plan: GraphExecPlan) -> tuple:
    """单轮执行：Popen + pipe 数据通道 + plog 捕获 + 输出/perf 解析。"""
    prof_before = _snapshot_prof_entries(plan.prof_path)
    data_r, data_w = os.pipe()
    data_holder = []

    def _drain_pipe(_dr=data_r, _dh=data_holder):
        chunks = []
        while True:
            chunk = os.read(_dr, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        _dh.append(b"".join(chunks))

    reader = threading.Thread(target=_drain_pipe, name=f"geir_data_{testcase.testcase_name}")
    reader.start()
    try:
        try:
            proc = subprocess.Popen(
                [plan.binary, str(dev_id), plan.input_prefix, str(data_w), plan.prof_path, plan.config_path],
                pass_fds=(data_w,),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=plan.build_dir,
            )
        finally:
            os.close(data_w)  # parent closes write-end so reader gets EOF
    except Exception:
        os.close(data_r)
        reader.join(timeout=5)
        raise
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=switches.proc_timeout or 1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        os.close(data_r)
        raise RuntimeError("GEIR execution timed out") from None
    reader.join(timeout=30)
    os.close(data_r)

    _log_process_output(testcase.testcase_name, stdout_bytes, stderr_bytes)
    if proc.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:10000]
        raise RuntimeError(f"GEIR execution failed (rc={proc.returncode}): {stderr}")

    new_profs = _new_prof_dirs(plan.prof_path, prof_before)
    device_perf_us = _parse_msprof_task_duration(new_profs) if plan.prof_path else None
    run_outputs = _parse_stdout(
        data_holder[0] if data_holder else b"",
        testcase.flat_output_dtypes,
        testcase.flat_output_shapes,
        case_name=testcase.testcase_name,
    )
    data_holder.clear()
    return run_outputs, device_perf_us


def _log_process_output(case_name, stdout_bytes, stderr_bytes):
    """Capture plog to TTK logging. stdout holds plog echo (when
    ASCEND_SLOG_PRINT_TO_STDOUT=1); stderr holds [TTK-GEIR] markers."""
    if stdout_bytes:
        for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                logging.info("[GEIR/%s] %s", case_name, line)
    if stderr_bytes:
        logging.debugc("[GEIR/%s] stderr:\n%s", case_name, stderr_bytes.decode("utf-8", errors="replace"))


def _write_input_bin(arr, path):
    """写 GEIR 输入 bin。NumPy 4-bit dtype 是 unpacked（1 字节/元素），而 C++ 执行程序
    按设备 packed 表示读取（每字节 2 个 4-bit 元素，见 geir_op_template.cpp.j2 的
    LoadInputFromFile），写入前必须打包，否则输入数据错位。"""
    if is_4bit_dtype(arr.dtype):
        pack_4bits(np.ascontiguousarray(arr)).tofile(path)
    else:
        arr.tofile(path)


def _write_all_input_bins(input_arrays, input_prefix) -> None:
    """Write input files for C++ program (skip None inputs, use contiguous index).

    DYNAMIC_INPUT(TensorList): 逐元素写 bin，索引与 graph_builder 的 elements 对齐。
    """
    data_idx = 0
    for _i, arr in enumerate(input_arrays):
        if arr is None:
            continue
        if isinstance(arr, (list, tuple)):
            for sub in arr:
                if sub is not None:
                    _write_input_bin(sub, f"{input_prefix}_{data_idx}.bin")
                    data_idx += 1
            continue
        _write_input_bin(arr, f"{input_prefix}_{data_idx}.bin")
        data_idx += 1


def _md5_of(outputs) -> str:
    return hashlib.md5(  # noqa: S324  # 非安全用途：确定性校验和
        b"".join(arr.tobytes() if isinstance(arr, np.ndarray) else b"" for arr in outputs)
    ).hexdigest()


def _det_round_hash(testcase, outputs, goldens, pre_compare) -> str:
    """dl=1 逐轮规范化哈希（双消费点）：pre_compare 双半边签名（outputs +
    goldens），hash 只用 outputs 半边——轮间一致性只关被测侧，goldens 半边每轮
    重算弃用。浅拷贝隔离：执行段返回的 outputs 保持 raw，比对侧由 compare_case 自行规范化。"""
    if pre_compare is None:
        return _md5_of(outputs)
    norm_outputs = list(outputs)
    norm_goldens = list(goldens)
    invoke_pre_compare(testcase, norm_outputs, norm_goldens, pre_compare)
    return _md5_of(norm_outputs)


def _det_status(md5_list, deterministic, case_name) -> Optional[str]:
    """dl=1 轮间一致性判定：MD5 全一致 PASS / 不一致 FAIL（单轮恒 PASS）。"""
    if not deterministic or not md5_list:
        return None
    if len(md5_list) > 1:
        if len(set(md5_list)) != 1:
            logging.error("[%s] GEIR MD5 mismatch across %s runs: %s", case_name, len(md5_list), md5_list)
            return "FAIL"
        logging.info("[%s] GEIR MD5 consistent across %s runs: %s", case_name, len(md5_list), md5_list[0])
        return "PASS"
    return "PASS"


def _cleanup_bins(input_prefix):
    """清临时输入 bin（全部退出路径统一经 finally）。"""
    for f in glob.glob(f"{glob.escape(input_prefix)}_*.bin"):
        with contextlib.suppress(OSError):
            os.remove(f)
