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
- run_time 轮（--run 透传，默认 3，与 kernel/aclnn 同口径）：每轮重写输入 bin
  （防图算子原地写污染，Kernel inplace 逐轮恢复的同构防御）；--clear-ub/-l1/
  -l0 时轮间由父进程 RTS helper kernel 清（C++ 子进程执行中不可介入，语义等价
  "每轮执行前清"）。
- perf 整批结束后聚合（_collect_round_perf）：本次运行新增的 PROF 目录按目录
  名内嵌时间戳升序，剔除时间戳最早的首轮（图编译/装载冷启动，与 kernel/aclnn
  首轮剔除同口径），剩余轮逐目录 Task Duration median 入样后跨轮 median；
  run=1 单轮冷启动口径；差集捕获不足总轮数时跳过剔除防误剔有效轮。
"""

__all__ = ["GraphExecPlan", "GraphExecResult", "execute_graph"]

import atexit
import contextlib
import glob
import hashlib
import logging
import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ttk.utilities import is_4bit_dtype, pack_4bits

from ..common.comparison import invoke_pre_compare
from ..kernel.device_state import clear_device_state
from .output_parse import _parse_stdout
from .perf_parse import _parse_msprof_task_duration
from .testcase import GeirTestcase

_SERVE_CASE_ERROR = "GEIR execution failed (serve):"


def _set_pdeathsig():
    """Linux PR_SET_PDEATHSIG：主进程死亡时子进程收 SIGKILL（覆盖 GE 内部卡死场景）。"""
    try:
        import ctypes

        ctypes.CDLL(None).prctl(1, 9)
    except Exception as e:  # noqa: BLE001  # 非 Linux 平台无 prctl 时静默降级为仅 EOF 兜底
        logging.debug("[GEIR-serve] set pdeathsig unavailable: %s", e)


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
    """图执行段产出：最后一轮输出 + 确定性状态 + 设备耗时（µs，跨轮 median）。"""

    outputs: List[np.ndarray] = field(default_factory=list)
    det_status: Optional[str] = None
    device_perf_us: Optional[float] = None


def execute_graph(
    testcase: GeirTestcase, dev_id: int, switches, plan: GraphExecPlan, pre_compare=None
) -> GraphExecResult:
    """逐轮经常驻 serve 子进程执行图（--run 透传）。

    轮数语义：与 kernel/aclnn 同口径——执行 run_time 轮，首个（图编译/装载
    冷启动）不进性能窗口，剩余 run_time-1 轮为有效样本；run=1 单轮冷启动口径
    （唯一一轮计入）。
    perf 采集在轮循环整体结束后统一完成（_collect_round_perf）：本次运行新增
    PROF 目录按时间戳排序、剔除最早一轮后聚合，首轮识别不依赖循环序号。
    pre_compare（双消费点之二）：dl=1 时逐轮作为确定性校验的规范化阶段——
    双半边签名（outputs + goldens），hash 只用 outputs 半边；GEIR 变换在轮间
    Python 侧，天然在 C++ msprof 采集进程之外（零窗口约束）。
    """
    result = GraphExecResult()
    _write_all_input_bins(testcase.input_arrays, plan.input_prefix)
    testcase.original_input_arrays = None
    deterministic = getattr(switches, "deterministic_level", 0) == 1
    run_count = switches.run_time
    md5_list = []
    prof_before = _snapshot_prof_entries(plan.prof_path)  # 本次运行前快照（整批一次）
    try:
        for run_idx in range(run_count):
            clear_device_state(switches)  # 轮间父进程 RTS 清（子进程执行中不可介入）
            if run_idx > 0:
                _write_all_input_bins(testcase.input_arrays, plan.input_prefix)  # 防图算子原地写污染
            result.outputs = _run_once_serve(testcase, dev_id, switches, plan)
            if deterministic:
                md5_list.append(_det_round_hash(testcase, result.outputs, testcase.golden_arrays, pre_compare))
    finally:
        _cleanup_bins(plan.input_prefix)
    result.device_perf_us = _collect_round_perf(plan, prof_before, run_count, testcase.testcase_name)
    result.det_status = _det_status(md5_list, deterministic, testcase.testcase_name)
    return result


class _GeirServe:
    """每 worker 一个常驻 GE 子进程（harness argv[5]="-" serve 模式）。

    GEInitialize 整批一次；数据管道逐例协议：成功 [8B num_outputs][8B len][data]*；
    失败 [8B -1][8B len][msg]。以 (binary, mtime) 为键——算子重编译后自动重启常驻进程。
    """

    def __init__(self, binary: str, build_dir: str, dev_id: int):
        self.binary = binary
        data_r, data_w = os.pipe()
        self._data_r = data_r
        self.proc = subprocess.Popen(
            [binary, str(dev_id), "", str(data_w), "", "-"],
            pass_fds=(data_w,),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=build_dir,
            preexec_fn=_set_pdeathsig,  # noqa: PLW1509  # 仅 libc prctl 设置，无 Python 对象操作
        )
        os.close(data_w)  # 父进程关写端, 子进程 EOF 即收工
        self._stderr_tail = bytearray()
        self._stdout_tail = bytearray()

        def _drain(stream, sink):
            try:
                for chunk in iter(lambda: stream.read(4096), b""):
                    sink.extend(chunk)
                    del sink[:-16384]
            except Exception as e:  # noqa: BLE001  # 排空线程: 进程退出后 read 抛错属预期
                logging.debug("[GEIR-serve] drain thread stopped: %s", e)

        threading.Thread(target=_drain, args=(self.proc.stderr, self._stderr_tail), daemon=True).start()
        threading.Thread(target=_drain, args=(self.proc.stdout, self._stdout_tail), daemon=True).start()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read_exact(self, n: int, deadline: float) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("GEIR serve timeout: incomplete frame")
            ready, _, _ = select.select([self._data_r], [], [], remaining)
            if not ready:
                raise RuntimeError("GEIR serve timeout: incomplete frame")
            chunk = os.read(self._data_r, n - len(buf))
            if not chunk:
                raise RuntimeError(
                    "GEIR serve process exited unexpectedly: " + self._stderr_tail[-2000:].decode("utf-8", "replace")
                )
            buf += chunk
        return bytes(buf)

    def run_case(self, testcase, plan: GraphExecPlan, switches) -> list:
        """喂一行 config，读一例完整帧，返回解析后的输出列表。"""
        entry = plan.config_path + "\t" + plan.input_prefix
        if plan.prof_path:
            entry += "\t" + plan.prof_path
        self.proc.stdin.write((entry + "\n").encode())
        self.proc.stdin.flush()
        deadline = time.time() + (switches.proc_timeout or 1800)
        head = self._read_exact(8, deadline)
        num_outputs = int(np.frombuffer(head, dtype=np.int64)[0])
        if num_outputs < 0:
            msg_len = int(np.frombuffer(self._read_exact(8, deadline), dtype=np.int64)[0])
            msg = self._read_exact(msg_len, deadline).decode("utf-8", errors="replace")
            raise RuntimeError(f"{_SERVE_CASE_ERROR}{msg}")
        stream = bytearray(head)
        for _ in range(num_outputs):
            len_raw = self._read_exact(8, deadline)
            stream.extend(len_raw)
            byte_count = int(np.frombuffer(len_raw, dtype=np.int64)[0])
            stream.extend(self._read_exact(byte_count, deadline))
        return _parse_stdout(
            bytes(stream),
            testcase.flat_output_dtypes,
            testcase.flat_output_shapes,
            case_name=testcase.testcase_name,
        )

    def close(self):
        with contextlib.suppress(Exception):
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            os.close(self._data_r)


_SERVE_INST: Optional[_GeirServe] = None
_SERVE_KEY: Optional[tuple] = None


def _serve_instance(plan: GraphExecPlan, dev_id: int) -> _GeirServe:
    """取当前 worker 的常驻实例；键（binary, mtime）变化或进程死亡则重启。"""
    global _SERVE_INST, _SERVE_KEY
    key = (plan.binary, os.path.getmtime(plan.binary))
    if _SERVE_INST is not None and (key != _SERVE_KEY or not _SERVE_INST.alive()):
        _SERVE_INST.close()
        _SERVE_INST = None
    if _SERVE_INST is None:
        _SERVE_INST = _GeirServe(plan.binary, plan.build_dir, dev_id)
        _SERVE_KEY = key
    return _SERVE_INST


def _close_geir_serve():
    global _SERVE_INST, _SERVE_KEY
    if _SERVE_INST is not None:
        _SERVE_INST.close()
        _SERVE_INST = None
        _SERVE_KEY = None


atexit.register(_close_geir_serve)


def _run_once_serve(testcase, dev_id: int, switches, plan: GraphExecPlan) -> list:
    """常驻模式单轮：喂一行 config，读一例完整帧。

    perf 不在本轮解析——轮循环整体结束后由 _collect_round_perf 按目录
    时间戳统一聚合，首轮识别与循环序号解耦。
    """
    serve = _serve_instance(plan, dev_id)
    try:
        return serve.run_case(testcase, plan, switches)
    except RuntimeError as e:
        if not str(e).startswith(_SERVE_CASE_ERROR):
            _close_geir_serve()
        raise
    except Exception:
        _close_geir_serve()
        raise


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


def _prof_dir_timestamp(prof_dir: str) -> Optional[str]:
    """PROF_000001_<YYYYMMDDHHMMSSmmm>_<pid+rand> 目录名内嵌时间戳；无法解析返回 None。"""
    parts = os.path.basename(prof_dir).split("_")
    if len(parts) >= 4 and len(parts[2]) >= 14 and parts[2].isdigit():
        return parts[2]
    return None


def _collect_round_perf(plan: GraphExecPlan, prof_before: frozenset, run_count: int, case_name: str):
    """整批运行后的性能聚合：本次新增 PROF 目录按时间戳升序，剔除最早一轮。

    run_count 为总执行轮数（execute_graph 传入 run_time，与 kernel/aclnn 同
    口径——首轮执行但首轮数据不进性能窗口）：
    - run_count>1：时间戳最早的目录即首轮产物（图编译/装载冷启动），不进
      性能窗口，剩余 run_count-1 轮为有效样本；
    - run_count=1：唯一一轮计入（单轮冷启动口径）；
    - 差集捕获目录数不足 run_count 时跳过剔除——误剔有效轮的风险优先于
      口径完整，同时告警暴露 PROF 目录缺失；
    - 时间戳不可解析时回退目录名序（PROF 序号前缀为 msprof 进程内单调）；
    - 逐目录 Task Duration median 入样（op_summary 优先/task_time 兜底，见
      perf_parse），跨轮 median 聚合；无有效样本返回 None。
    """
    if not plan.prof_path:
        return None
    new_dirs = _new_prof_dirs(plan.prof_path, prof_before)
    if not new_dirs:
        return None
    keys = [_prof_dir_timestamp(prof_dir) for prof_dir in new_dirs]
    if all(keys):
        ordered = [prof_dir for _, prof_dir in sorted(zip(keys, new_dirs))]
    else:
        logging.warning("[%s] GEIR perf: PROF dir name without timestamp, fallback to name order", case_name)
        ordered = new_dirs
    if len(ordered) < run_count:
        logging.warning(
            "[%s] GEIR perf: %s/%s rounds' PROF dirs captured; first-round exclusion skipped",
            case_name,
            len(ordered),
            run_count,
        )
        included = ordered
    elif run_count > 1:
        included = ordered[1:]
    else:
        included = ordered
    samples = []
    for prof_dir in included:
        value = _parse_msprof_task_duration([prof_dir])
        if value is not None:
            samples.append(value)
    if not samples:
        return None
    median_us = round(float(np.median(samples)), 3)
    logging.info(
        "[%s] GEIR device perf: rounds=%s, included=%s, samples=%s, median=%.3f us",
        case_name,
        run_count,
        len(included),
        samples,
        median_us,
    )
    return median_us


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
