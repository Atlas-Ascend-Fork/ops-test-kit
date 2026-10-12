#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""cgroup 内存限额感知：容器里宿主 free 是假象（如 754G 宿主 / 32G 限额）。

两条消费路径：
- memory_gate：闸门快照按 min(cgroup 限额, 宿主) 计算（防 OOM 前拦截）
- limit_cann_compile_jobs：按限额收紧 CANN tbe 编译内部并行度
  （ascendc_compile_base 单用例默认 make -j8，多路重型 ccec 会打爆限额）
"""

import logging
import os

__all__ = [
    "cgroup_mem_current",
    "cgroup_mem_limit",
    "cgroup_memory_snapshot",
    "limit_cann_compile_jobs",
]

_GB = 1024**3
_CGROUP_V2_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_V2_CURRENT = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V1_CURRENT = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
# v2 "max"（读出非整数→None）；v1 以近 2^63 的巨值表示无限额
_UNLIMITED = 1 << 60

# CANN tbe 编译（ascendc_compile_base._get_parallel_job_without_op_project）
# 单用例内部默认 make -j8；单个 ccec/bisheng 重型编译估算 ~4GiB，
# 常驻（主进程 / KBS / worker preload）预留 8GiB
_CANN_DEFAULT_JOBS = 8
_COMPILE_JOB_MEM = 4 * _GB
_COMPILE_RESERVE_MEM = 8 * _GB


def _read_int(path):
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def cgroup_mem_limit():
    """cgroup 内存限额（字节）；无 cgroup / 未限额（v2 "max"、v1 巨值）→ None。"""
    raw = _read_int(_CGROUP_V2_MAX)
    if raw is None:
        raw = _read_int(_CGROUP_V1_LIMIT)
    if raw is None:
        return None
    return raw if 0 < raw < _UNLIMITED else None


def cgroup_mem_current():
    """cgroup 当前用量（字节）；无 cgroup → None。"""
    raw = _read_int(_CGROUP_V2_CURRENT)
    if raw is None:
        raw = _read_int(_CGROUP_V1_CURRENT)
    return raw


def cgroup_memory_snapshot():
    """(limit, current)；无 cgroup → (None, None)。"""
    return cgroup_mem_limit(), cgroup_mem_current()


def limit_cann_compile_jobs(job_mem=_COMPILE_JOB_MEM, reserve_mem=_COMPILE_RESERVE_MEM, max_concurrent_compiles=8):
    """按 cgroup 限额收紧 CANN 编译内部并行度（注入 TILINGKEY_PARALLEL_JOB）。

    预算 = (限额 - 当前用量 - 常驻预留) / (最大并发编译数 × 单 job 内存)，
    仅当推算值低于 CANN 默认(8)时注入；用户显式设置 TILINGKEY_PARALLEL_JOB /
    TILINGKEY_PAR_COMPILE=1（CANN 串行开关）/ 无限额（裸机）一律不干预。

    必须在任何 worker / 子进程拉起之前调用——forkserver 子进程不继承
    之后才设置的环境变量。
    """
    if "TILINGKEY_PARALLEL_JOB" in os.environ or os.environ.get("TILINGKEY_PAR_COMPILE") == "1":
        return None
    limit, current = cgroup_memory_snapshot()
    if limit is None:
        return None
    budget = max(0, limit - (current or 0) - reserve_mem)
    jobs = max(1, int(budget // (max(1, max_concurrent_compiles) * job_mem)))
    if jobs < _CANN_DEFAULT_JOBS:
        os.environ["TILINGKEY_PARALLEL_JOB"] = str(jobs)
        logging.info(
            "cgroup memory limit %.1f GB: CANN compile jobs throttled to %d "
            "(TILINGKEY_PARALLEL_JOB; budget %.1f GB, %d concurrent compiles x %.0f GB/job)",
            limit / _GB,
            jobs,
            budget / _GB,
            max_concurrent_compiles,
            job_mem / _GB,
        )
        return jobs
    return None
