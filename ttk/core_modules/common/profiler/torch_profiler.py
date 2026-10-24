#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""torch 系 profiler。

torch 生态的两个互斥分支（factory 按 torch.* 前缀二选一，永不独立被选，
故同族同居一文件）+ 共享引擎函数：

- TorchProfiler：torch.profiler（torch.* 非 NPU——CPU 本地 / 远端加速卡同构）
- TorchNpuProfiler：torch_npu.profiler（torch.*/torch_npu.* × NPU——dispatcher
  可见、schedule 采集时 step 过滤、aclgraph Model ID 过滤）
- resolve_activities / summarize_events / _is_device_kernel：共享引擎，
  E2E TorchProfiler 与 server 的 torch_xpu_profiler_pass 同源消费
  （activities 数据驱动解析 + key_averages 事件树拍平聚合）
"""

import csv
import logging
import os
import time
from typing import Optional

from .base import (
    FrameworkProfiler,
    KernelDetails,
    KernelInfo,
    ProfileResult,
    _self_device_time,
)


def _is_device_kernel(evt) -> bool:
    """只保留真实设备内核条目，供 device_us 求和使用。

    key_averages() 是把 profiler 的**事件树拍平**后的列表。树上同一段设备
    时间会以三种不同身份各出现一次，若不加区分地全部相加，等于把它重复计入：

        key_averages()
        │
        ├─ ProfilerStep*        容器 span（schedule() 每轮自动插入，覆盖整轮）
        │                       剔除：量纲不同（跨度≠忙碌）且层级上已包住下面两类
        ├─ aten::xxx            CPU 侧算子（主机端记录）
        │                       剔除：与内核条目是同一段时间的两种视角
        └─ xxx（内核名）         真正的设备内核条目
                                保留：唯一应该求和的对象
    """
    name = getattr(evt, "key", "")
    return not name.startswith(("ProfilerStep", "aten::", "cuda"))


def resolve_activities(activities_cfg):
    """Data-driven ProfilerActivity resolution（属性名列表 → 枚举列表）。

    未知名 raise ValueError 并列出合法值——E2E TorchProfiler 与 server
    torch pass 的配置错误同型（config error 上抛，非 NA 降级）。
    """
    from torch.profiler import ProfilerActivity

    activities = []
    for name in activities_cfg:
        try:
            activities.append(getattr(ProfilerActivity, name))
        except AttributeError:
            valid = [n for n in dir(ProfilerActivity) if not n.startswith("_")]
            raise ValueError(f"unknown ProfilerActivity '{name}'; valid: {valid}") from None
    return activities


def summarize_events(events, device: str, device_time_attr: Optional[str] = None):
    """Flatten torch.profiler key_averages() into KernelInfo aggregates.

    过滤：_is_device_kernel 剔除容器 span / CPU 算子 / host API 的重复
    视角；self_* 纯自身时间经 _self_device_time 三候选回退（见 base）。
    device_us == 0 的条目不入列（对总和贡献亦为 0）。

    Returns (kernels, total_device_us, total_cpu_us)。total_cpu_us 对
    **所有**事件求和（含非内核条目——CPU 时间不按设备身份剔除）。
    """
    kernels = []
    total_device_us = 0.0
    total_cpu_us = 0.0
    for evt in events:
        total_cpu_us += getattr(evt, "cpu_time_total", 0.0)
        if not _is_device_kernel(evt):
            continue
        device_us = _self_device_time(evt, device, device_time_attr)
        if device_us > 0:
            kernels.append(
                KernelInfo(
                    name=evt.key,
                    device_us=device_us,
                    calls=evt.count,
                    avg_us=device_us / max(evt.count, 1),
                )
            )
            total_device_us += device_us
    return kernels, total_device_us, total_cpu_us


class TorchProfiler(FrameworkProfiler):
    """Profiler for torch using torch.profiler. Works on both CPU and accelerators.

    Hardware-neutral: activities come from
    ``backend.profile["profiler"]["activities"]`` (a list of ProfilerActivity
    attribute names like "CPU"/"CUDA"/"MLU"/"MUSA") rather than a string compare
    on torch_lib. ``_device_acts`` is the non-CPU subset; ``result`` reports
    device kernels via the shared ``summarize_events`` engine.
    """

    def __init__(self, backend):
        from torch.profiler import profile

        cfg = backend.profile["profiler"]
        self._prof = profile(activities=resolve_activities(cfg["activities"]), record_shapes=True)
        # Non-CPU activity names (e.g. ["MLU"]); empty for CPU-only profiles.
        self._device_acts = [a for a in cfg["activities"] if a != "CPU"]
        self._device_time_attr = cfg.get("device_time_attr")
        # device key for _self_device_time legacy fallback (self_{device}_time_total).
        # device = profile["torch_lib"] (mlu/musa/...), NOT the activity name
        # (activity name is a ProfilerActivity enum; torch_lib is the torch module attr).
        self._device = backend.profile["torch_lib"]

    def __enter__(self):
        self._prof.__enter__()
        return self

    def __exit__(self, *exc):
        return self._prof.__exit__(*exc)

    def result(self, repeat_count) -> ProfileResult:
        events = self._prof.key_averages()
        kernels, total_device_us, total_cpu_us = summarize_events(events, self._device, self._device_time_attr)

        if self._device_acts:
            return ProfileResult(
                elapsed_us=total_device_us / max(repeat_count, 1),
                kernel_details=KernelDetails(
                    kernels=kernels,
                    total_device_us=total_device_us,
                    total_cpu_us=total_cpu_us,
                ),
                source="torch",
            )

        # CPU backend: no device timing, only CPU wall-clock
        return ProfileResult(
            elapsed_us=0.0,
            kernel_details=KernelDetails(
                kernels=[],
                total_device_us=0.0,
                total_cpu_us=total_cpu_us / max(repeat_count, 1),
            ),
            source="torch",
        )


def torch_npu_profile_start(outdir: str, warmup_count: int = 0, active_steps: int = 1):
    """Start a torch_npu.profiler session; returns (prof, start_time_ns).

    E2E TorchNpuProfiler 与 remote server torch pass（torch_lib=npu 分支）
    共享——单一实现防漂移。schedule: wait=0, warmup=warmup_count,
    active=active_steps, repeat=1（每轮 prof.step() 推进；prof.stop() 由
    调用方负责）。导出走 tensorboard_trace_handler(outdir)。
    """
    from torch_npu.profiler import (
        AiCMetrics,
        ExportType,
        ProfilerActivity,
        ProfilerLevel,
        _ExperimentalConfig,
        profile,
        schedule,
        tensorboard_trace_handler,
    )

    experimental_config = _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1,
        aic_metrics=AiCMetrics.PipeUtilization,
        export_type=ExportType.Text,
    )
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        record_shapes=True,
        experimental_config=experimental_config,
        schedule=schedule(wait=0, warmup=warmup_count, active=active_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(outdir),
    )
    start_time_ns = time.time_ns()
    prof.start()
    return prof, start_time_ns


def find_npu_profile_csv(outdir: str, start_time_ns: int, filename: str, context_name: str = "unknown"):
    """Only consider CSVs written since start_time_ns, never old traces.

    E2E TorchNpuProfiler 与 remote server torch pass 共享的查找逻辑
    （mtime ≥ start 过滤，多 trace 子目录取最新 mtime）。
    """
    matches = []
    for root, _, files in os.walk(outdir):
        if filename in files:
            path = os.path.join(root, filename)
            try:
                if os.stat(path).st_mtime_ns >= start_time_ns:
                    matches.append(path)
            except FileNotFoundError:
                continue  # removed between listdir and stat (concurrent cleanup)
    if not matches:
        logging.warning(
            "NPU profiler has no new %s for %s in %s; historical traces are not reused. "
            "Check profiler export logs and directory permissions.",
            filename,
            context_name,
            outdir,
        )
        return None
    return max(matches, key=os.path.getmtime)


def _kernel_row_value(row, graph=False):
    """Return (name, duration_us) for a row worth counting, else None.

    两条规则（按采集形态；graph 内不区分 torchair cst/dyn 与 aclgraph）：
    - eager：空 Step Id 行是 warmup 逃逸（实测偶发），丢弃；不看 Model ID
      ——eager 算子本在图外、CANN 恒标 UINT32_MAX，按它过滤会全数误丢。
    - graph：warmup 调用在 schedule 的 warmup step 内执行、不进导出（实测
      4 次执行只导出 3 行），空 Step Id 行只可能是计时调用的归因缺失（CST
      首个回放确定性缺失），保留；图外杂活（staging/填充/复位拷贝）CANN
      标 UINT32_MAX Model ID，丢弃。
    """
    if graph:
        try:
            model_id = int(row.get("Model ID", ""))
        except (ValueError, TypeError):
            return None
        if not 0 <= model_id < 0xFFFFFFFF:
            return None
    elif not row.get("Step Id", "").strip():
        return None
    name = row.get("Name", "").strip()
    try:
        duration = float(row.get("Duration(us)", 0))
    except (ValueError, TypeError):
        return None
    return (name, duration) if name and duration > 0 else None


def parse_kernel_details_csv(csv_path, aclgraph=False, graph=False):
    """Parse kernel_details.csv for per-kernel device timing.

    过滤规则见 _kernel_row_value（eager 丢空 Step Id / graph 统一丢图外
    Model ID，aclgraph 归 graph 规则）。ACLGraph timing includes all
    captured models, but not graph-external input copies.

    E2E TorchNpuProfiler 与 remote server torch pass 共享（单一实现防漂移）。
    """
    use_graph = graph or aclgraph
    kernels_map = {}  # name -> {total_us, calls, max_us, min_us}
    total_device_us = 0.0

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if use_graph and "Model ID" not in (reader.fieldnames or ()):
                label = "ACLGraph" if aclgraph else "Graph-mode"
                logging.warning("%s performance unavailable: Model ID missing in %s", label, csv_path)
                return [], 0.0
            for row in reader:
                value = _kernel_row_value(row, use_graph)
                if value is None:
                    continue
                name, duration = value
                total_device_us += duration
                info = kernels_map.get(name)
                if info is None:
                    kernels_map[name] = {
                        "total_us": duration,
                        "calls": 1,
                        "max_us": duration,
                        "min_us": duration,
                    }
                else:
                    info["total_us"] += duration
                    info["calls"] += 1
                    info["max_us"] = max(info["max_us"], duration)
                    info["min_us"] = min(info["min_us"], duration)
    except Exception as e:
        logging.warning(f"Failed to parse {csv_path}: {e}")

    if aclgraph:
        if kernels_map:
            logging.info(
                "ACLGraph device performance includes captured graph kernels only; "
                "graph-external tasks are excluded. Full trace: %s",
                csv_path,
            )
        else:
            logging.warning(
                "ACLGraph performance unavailable: no captured graph kernels with valid Step Id/Model ID in %s; "
                "check graph capture and profiler output",
                csv_path,
            )
    kernels = [
        KernelInfo(
            name=name,
            device_us=info["total_us"],
            calls=info["calls"],
            avg_us=info["total_us"] / info["calls"],
            max_us=info["max_us"],
            min_us=info["min_us"],
        )
        for name, info in kernels_map.items()
    ]
    return kernels, total_device_us


def parse_operator_cpu_time(csv_path):
    """Parse operator_details.csv for total CPU time."""
    total_cpu_us = 0.0
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cpu_us = float(row.get("Host Self Duration(us)", 0))
                except (ValueError, TypeError):
                    continue
                total_cpu_us += cpu_us
    except Exception as e:
        logging.warning(f"Failed to parse {csv_path}: {e}")
    return total_cpu_us


class TorchNpuProfiler(FrameworkProfiler):
    """Profiler for NPU using torch_npu.profiler.

    Uses torch_npu.profiler.profile (start/step/stop) and parses the exported
    kernel_details.csv / operator_details.csv for device-side timing.
    采集/解析内核已提为模块级共享函数（torch_npu_profile_start /
    find_npu_profile_csv / parse_kernel_details_csv / parse_operator_cpu_time），
    供 remote server torch pass（torch_lib=npu）复用。
    """

    def __init__(self, backend, config):
        """config: ProfilerConfig（工厂已就绪；enabled 分支在工厂先行裁决）。"""
        self._testcase_name = config.testcase_name or "unknown"
        self._outdir = os.path.join(config.root_path, "msprof", "e2e", self._testcase_name)
        os.makedirs(self._outdir, exist_ok=True)
        self._warmup_count = config.warmup_count
        self._aclgraph = config.aclgraph
        # aclgraph 是图模式的子集（torch.compile + npugraph_ex），过滤规则共用 graph
        self._graph = config.graph or config.aclgraph
        self._start_time_ns = 0
        self._prof = None

    def __enter__(self):
        self._prof, self._start_time_ns = torch_npu_profile_start(self._outdir, self._warmup_count)
        return self

    def step(self):
        if self._prof:
            self._prof.step()

    def __exit__(self, *exc):
        if self._prof:
            try:
                self._prof.step()
            finally:
                self._prof.stop()

    def result(self, repeat_count) -> ProfileResult:
        kernel_csv = self._find_csv("kernel_details.csv")
        operator_csv = self._find_csv("operator_details.csv")

        kernels = []
        total_device_us = 0.0
        total_cpu_us = 0.0

        if kernel_csv:
            kernels, total_device_us = parse_kernel_details_csv(kernel_csv, aclgraph=self._aclgraph, graph=self._graph)
            if not kernels and not self._aclgraph:
                logging.warning(
                    "NPU performance unavailable: no device kernel records with valid Step Id in %s", kernel_csv
                )
        if operator_csv:
            total_cpu_us = parse_operator_cpu_time(operator_csv)

        return ProfileResult(
            elapsed_us=total_device_us / max(repeat_count, 1),
            kernel_details=KernelDetails(
                kernels=kernels,
                total_device_us=total_device_us,
                total_cpu_us=total_cpu_us,
            ),
            source="torch_npu",
        )

    def _find_csv(self, filename):
        return find_npu_profile_csv(self._outdir, self._start_time_ns, filename, self._testcase_name)

    # ── 兼容封装：历史调用点/测试直接以类方法消费（语义与模块级共享函数一致）──
    @staticmethod
    def _kernel_row_value(row, graph=False):
        return _kernel_row_value(row, graph)

    @classmethod
    def _parse_kernel_details(cls, csv_path, aclgraph=False, graph=False):
        return parse_kernel_details_csv(csv_path, aclgraph=aclgraph, graph=graph)

    @staticmethod
    def _parse_operator_cpu_time(csv_path):
        return parse_operator_cpu_time(csv_path)
