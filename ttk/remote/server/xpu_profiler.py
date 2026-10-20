#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""XPU profiler pass——server _run_perf 的采集层。

- torch_xpu_profiler_pass：torch.profiler Self device time（schedule wait/
  warmup/active=runtime）——activities 解析与事件聚合复用 common/profiler/
  torch_profiler 的 resolve_activities / summarize_events 共享引擎（与 E2E
  TorchProfiler 同源，单一实现防漂移）
- tf_xpu_profiler_pass：tf.profiler.experimental + xplane.pb device plane 解析

错误策略：OP 执行失败 **propagate**（→ FAIL）；仅 profiler 机制失败
（start/stop/readout）degrade 为 NA 并 log——错误分类属 _run_perf 的 HTTP
handler 编排契约，不属采集语义。

peak_memory 不在此层——reset_peak 再执行一遍取峰值是编排动作，由 _run_perf
的 pass 2 完成。"""

__all__ = ["torch_xpu_profiler_pass", "tf_xpu_profiler_pass"]

# Standard Packages
import logging
import shutil
import tempfile
from pathlib import Path

from ttk.core_modules.common.profiler.torch_profiler import resolve_activities, summarize_events


def torch_xpu_profiler_pass(invoke, runtime: int, device: str, activities_cfg: list):
    """torch.profiler Self device time. Returns (outputs, device_us).

    invoke: 无参 callable（执行一次被测 op；OP 错误由 invoke 内部传播）。
    device_us = per-iteration average（active window 总 self device time / runtime），
    仅累加真实 设备 内核条目（summarize_events 的 _is_device_kernel 过滤）。

    配置错误（activities 缺失/未知）raise ValueError（共享 resolve_activities）——
    server config error → 500，非 NA degrade；profiler 机制失败
    （start/stop/readout）→ NA + log。
    """
    import torch

    activities = resolve_activities(activities_cfg)
    device_us = 0.0
    outputs = None
    try:
        _ctx = torch.profiler.profile(
            activities=activities, schedule=torch.profiler.schedule(wait=1, warmup=1, active=runtime, repeat=1)
        )
        prof = _ctx.__enter__()
    except Exception:
        logging.exception("torch.profiler start failed; device_us=NA")
        return outputs, device_us
    try:
        for _ in range(runtime + 2):
            outputs = invoke()
            prof.step()
    finally:
        try:
            _ctx.__exit__(None, None, None)
        except Exception:
            logging.exception("torch.profiler stop failed")
    try:
        _, total_device_us, _ = summarize_events(prof.key_averages(), device)
        if runtime > 0:
            total_device_us /= runtime
        device_us = total_device_us
    except Exception:
        logging.exception("torch.profiler key_averages failed; device_us=NA")
    return outputs, device_us


_TF_XPLANE_EVENT_CAP = 100000


def _tf_profile_window(invoke, runtime: int, logdir: str):
    """采集窗口：warmup×2（丢弃）→ start → runtime 轮 → stop。

    返回 (started, outputs)：started=False 表示 profiler 启动失败（device_us
    置 NA，跳过解析）；OP 执行错误 propagate（→ FAIL）；stop 失败仅 log。
    """
    import tensorflow as tf

    for _ in range(2):  # warmup (dropped) — errors propagate
        invoke()
    try:
        tf.profiler.experimental.start(logdir)
    except Exception:
        logging.exception("tf.profiler start failed; device_us=NA")
        return False, None
    try:
        outputs = None
        for _ in range(runtime):
            outputs = invoke()
        return True, outputs
    finally:
        try:
            tf.profiler.experimental.stop()
        except Exception:
            logging.exception("tf.profiler stop failed")


def _sum_xplane_device_us(pb_path, xplane_pb2) -> float:
    """device plane 事件 duration_ps/1e6 求和（copy/memcpy 拷贝事件剔除）。

    num_occurrences 不乘（proto3 unset = 0）；扫描有安全上限（病态 proto 防爆）。"""
    xs = xplane_pb2.XSpace()
    xs.ParseFromString(pb_path.read_bytes())
    total = 0.0
    seen = 0
    truncated = False
    for plane in xs.planes:
        if not plane.name.startswith("/device"):
            continue
        meta = {mid: m.name for mid, m in plane.event_metadata.items()}
        for line in plane.lines:
            for ev in line.events:
                if seen >= _TF_XPLANE_EVENT_CAP:
                    truncated = True
                    break
                ev_name = meta.get(ev.metadata_id, "").lower()
                if "copy" in ev_name or "memcpy" in ev_name:
                    continue
                total += ev.duration_ps / 1e6
                seen += 1
            if truncated:
                break
        if truncated:
            break
    if truncated:
        logging.warning("TF xplane event count exceeded %d, truncating", _TF_XPLANE_EVENT_CAP)
    return total


def tf_xpu_profiler_pass(invoke, runtime: int):
    """tf.profiler.experimental + xplane.pb. Returns (outputs, device_us).

    device_us = per-iteration average（device plane events duration_ps/1e6 总和
    / runtime）。两次 warmup 先于 profiled runtime invokes；logdir finally 清理。"""
    try:
        from tensorflow.core.profiler.protobuf import xplane_pb2
    except ImportError:
        xplane_pb2 = None

    device_us = 0.0
    logdir = tempfile.mkdtemp(prefix="tfprof_")
    try:
        started, outputs = _tf_profile_window(invoke, runtime, logdir)
        if not started:
            return outputs, device_us
        pb_path = next(Path(logdir).rglob("*.xplane.pb"), None)
        if pb_path is None:
            logging.warning("TF profiler wrote no xplane.pb under %s; device_us=NA", logdir)
        elif xplane_pb2 is None:
            logging.warning(
                "tensorflow.core.profiler.protobuf.xplane_pb2 not importable; cannot parse %s; device_us=NA",
                pb_path,
            )
        if pb_path and xplane_pb2:
            try:
                total = _sum_xplane_device_us(pb_path, xplane_pb2)
                if runtime > 0:
                    total /= runtime
                device_us = total
            except Exception:
                logging.exception("tf xplane parse failed; device_us=NA")
        return outputs, device_us
    finally:
        shutil.rmtree(logdir, ignore_errors=True)
