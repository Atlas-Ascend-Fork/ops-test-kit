#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn HCCL multi-device execution framework (原 mc2.py，命名收口 hccl 前缀).

The execution path only knows the machine list here: ``device_ids`` declares
which devices participate, and TTK sets up the HCCL communicators (world
group, plus EP/TP sub-groups when the CSV attributes ``epWorldSize`` /
``tpWorldSize`` declare a dual topology), launches one execution thread per
rank with per-rank seeds and runs the generic two-phase aclnn executor.
The execution path carries no operator names: operator-specific input
preparation (rank-specific input regeneration such as quantized-input
correlation or weight sharing, and attribute derivation) lives in
per-operator plugin assets injected via --plugin, so changing an operator
never changes this framework.

Multi-device plugin contract (execution-side hooks, same spec attributes
as the single-device path):
- input (``customize_inputs``): invoked per rank thread inside
  InputGenerator.gen with the built C-signature args; the hook kwargs
  carry ``testcase`` — the rank's TestcaseAclnn copy whose ``my_rank`` /
  ``device_ids`` give the rank context and whose slot-table writers
  (set_flat_tensor / set_flat_storage / attributes) enable storage-level
  regeneration.

The multi-device golden dispatch is plugin-first: spec plugin golden,
then the hccl_dispatcher framework entry as the no-plugin fallback —
called as ``golden(thread_contexts, device_ids, all_precision)``. The golden runs after execution by construction: it
consumes each rank's execution result (thread contexts), so it is exempt
from the golden-before-execution ordering of the single-device path.
"""

__all__ = [
    "profile_multi_device",
    "release_retained_multi_device_resources",
    "compare_multi_device",
    "HcclResourceScope",
]


# Standard Packages
import contextlib
import copy
import ctypes
import logging
import threading
from typing import Dict, List

import numpy

# Third-Party Packages
from ttk.switches import get_global_storage

from ..common.plugin import SpecResolver
from .execute import AclOpExecutor, get_aclnn_device
from .hccl_dispatcher import golden_multi_device_compare
from .info_keeper import ensure_api_so_path
from .input_generate import InputGenerator
from .result import ApiComparisonResult, ApiProfilingResult
from .runtime import AclInterface
from .testcase import TestcaseAclnn


def _dual_comm_requested(attributes) -> bool:
    """Dual (EP/TP) communicator topology is declared by testcase data.

    The topology is a CSV attributes declaration (epWorldSize/tpWorldSize),
    not an operator-name match: any communication case that declares a dual
    topology gets EP/TP sub-groups; the framework stays operator-agnostic.
    Dual means BOTH domains declared: single-domain ops (e.g. grouped GMM
    declares only epWorldSize over the full world) fall back to the world
    communicator for both groupEp/groupTp (see _launch_one_thread), so they
    must not enter the dual-topology subgroup setup.
    """
    if not attributes:
        return False
    return "epWorldSize" in attributes and "tpWorldSize" in attributes


# numpy.random is process-global state: rank threads interleaving their
# draws would cross-contaminate the per-rank seed streams (rank A's later
# tensors drawn from rank B's stream after a GIL switch), breaking the
# same-seed reproducibility contract in multi-device mode. Serializing
# seed+generation keeps each rank's inputs solely from its own stream.
_INPUT_GEN_LOCK = threading.Lock()


def _launch_one_thread(
    context: TestcaseAclnn,
    dev_id: int,
    rank_idx: int,
    hccl_comm_val: int,
    c_context_val: int,
    c_stream_val: int,
    results: Dict[int, ApiProfilingResult],
    thread_contexts: Dict[int, TestcaseAclnn],
    errors: Dict[int, Exception],
    main_device: "AclInterface",
    ep_comm_val: int = 0,
    tp_comm_val: int = 0,
    pre_exec_barrier: threading.Barrier = None,
):
    try:
        logging.info(f"[Thread dev={dev_id}] starting, rank_idx={rank_idx}")
        acl_dll = ctypes.CDLL("libascendcl.so")
        acl_dll.aclrtSetCurrentContext.restype = ctypes.c_int
        acl_dll.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
        acl_dll.aclrtSetCurrentContext(ctypes.c_void_p(c_context_val))
        hccl_dll = ctypes.CDLL("libhcomm.so")
        hccl_dll.HcclGetCommName.restype = ctypes.c_int
        hcom_name_buf = ctypes.create_string_buffer(128)
        ret = hccl_dll.HcclGetCommName(ctypes.c_void_p(hccl_comm_val), hcom_name_buf)
        if ret != 0:
            raise RuntimeError(f"HcclGetCommName failed with ret={ret} for dev {dev_id}")
        hcom_name = hcom_name_buf.value.decode("utf-8")
        logging.info(f"[Thread dev={dev_id}] HcclGetCommName = '{hcom_name}'")
        thread_ctx = copy.deepcopy(context)
        thread_ctx.my_rank = dev_id
        if "group" in thread_ctx.attributes:
            thread_ctx.attributes["group"] = hcom_name
        # Communicator attribute names are testcase declarations.  When no
        # topology-specific sub-group was requested, the primary world
        # communicator is the valid fallback for both names.
        if "groupEp" in thread_ctx.attributes and ep_comm_val == 0:
            thread_ctx.attributes["groupEp"] = hcom_name
        if "groupTp" in thread_ctx.attributes and tp_comm_val == 0:
            thread_ctx.attributes["groupTp"] = hcom_name
        if ep_comm_val != 0 or tp_comm_val != 0:
            if ep_comm_val != 0:
                ep_name_buf = ctypes.create_string_buffer(128)
                ret = hccl_dll.HcclGetCommName(ctypes.c_void_p(ep_comm_val), ep_name_buf)
                if ret == 0:
                    thread_ctx.attributes["groupEp"] = ep_name_buf.value.decode("utf-8")
                    logging.info(f"[Thread dev={dev_id}] groupEp = '{thread_ctx.attributes['groupEp']}'")
            if tp_comm_val != 0:
                tp_name_buf = ctypes.create_string_buffer(128)
                ret = hccl_dll.HcclGetCommName(ctypes.c_void_p(tp_comm_val), tp_name_buf)
                if ret == 0:
                    thread_ctx.attributes["groupTp"] = tp_name_buf.value.decode("utf-8")
                    logging.info(f"[Thread dev={dev_id}] groupTp = '{thread_ctx.attributes['groupTp']}'")
        switches = get_global_storage()
        base_seed = switches.random_seed or 0
        per_rank_seed = base_seed + rank_idx * 1000
        try:
            # Standard generation plus the per-rank plugin input hook: any
            # operator-specific rank input regeneration is plugin territory.
            # The hook receives the rank context through the ``testcase``
            # kwarg (my_rank/device_ids + slot-table writers), see
            # InputGenerator._call_custom_input. Seed+gen under the lock:
            # numpy.random is global, interleaved draws would leak across
            # rank seed streams.
            with _INPUT_GEN_LOCK:
                numpy.random.seed(per_rank_seed)
                logging.info(f"[Thread dev={dev_id}] using seed={per_rank_seed} for input generation")
                InputGenerator(thread_ctx).gen()
        except Exception:
            logging.exception(f"[Thread dev={dev_id}] input generation failed:")
            _abort_barrier(pre_exec_barrier)
            errors[dev_id] = RuntimeError("INPUT_GEN_FAILURE")
            results[dev_id] = ApiProfilingResult.fail("INPUT_GEN_FAILURE")
            return
        # 线程级设备视图：DLL/RTS 句柄与主设备共享，acl 内存账本线程私有
        # （收口 AclInterface.clone_for_thread——原手工 __dict__ 克隆漏重置
        # 三个容器且账本位写过错名属性，共享容器并发写靠 GIL 侥幸；线程
        # 不 reset/finalize，HcclResourceScope 统一清理——线程提前 reset
        # 会 double-free 主线程句柄 → SIGSEGV，A5 实测）。
        thread_device = main_device.clone_for_thread(dev_id)
        # SO resolution is generic (info_keeper.ensure_api_so_path): valid
        # paths (incl. the builtin nm symbol scan) are kept, empty or dangling
        # ones fall back to libopapi candidates in the arch-specific lib64 dir.
        so_path = ensure_api_so_path(thread_ctx.api_name)
        logging.info(f"[Thread dev={dev_id}] so_path='{so_path}'")
        c_stream = ctypes.c_void_p(c_stream_val)
        if pre_exec_barrier is not None:
            logging.info(f"[Thread dev={dev_id}] waiting at pre-exec barrier")
            pre_exec_barrier.wait()
            logging.info(f"[Thread dev={dev_id}] passed pre-exec barrier")
        thread_ctx.prof_result = AclOpExecutor(thread_ctx, thread_device).do(
            stream=c_stream, skip_context_creation=True
        )
        logging.info(f"[Thread dev={dev_id}] execution done, success={not thread_ctx.prof_result.failed()}")
        thread_contexts[dev_id] = thread_ctx
        results[dev_id] = thread_ctx.prof_result
    except Exception as e:
        logging.exception(f"[Thread dev={dev_id}] failed:")
        # Fail-fast: a failed rank must not leave sibling ranks blocked at
        # the pre-exec barrier for the full barrier timeout (1800s).
        _abort_barrier(pre_exec_barrier)
        errors[dev_id] = e
        results[dev_id] = ApiProfilingResult.fail("THREAD_EXECUTION_FAILED")


def _abort_barrier(barrier) -> None:
    if barrier is None:
        return
    with contextlib.suppress(Exception):
        barrier.abort()


class HcclResourceScope:
    """HCCL 多卡资源生命周期上下文管理器（形态套用 MultiDeviceLock）。

    ``__enter__`` 全量获取：预复位 + ``aclrtSetDevice`` 全部设备 → 世界域
    ``HcclCommInitAll`` → EP/TP 子域 → 逐设备 context/stream（返回码全检
    查，句柄增量入账）。``__exit__`` 统一释放（extra/世界域
    ``HcclCommDestroy`` → sync → ``DestroyStream`` → ``DestroyContext`` →
    ``ResetDevice``），setup 任意一步抛错都会清理已建的部分资源。成功路
    径调用 ``retain()`` 把句柄转存 context 供 golden 阶段
    （``release_retained_multi_device_resources`` 延迟释放），``__exit__``
    跳过释放。

    不继承 MultiDeviceLock（评估结论）：设备锁属进程调度域（worker 在
    golden 前经 process_ctx RPC 释放，生命周期止于执行块），HCCL 句柄保
    留至 golden 分发前，两者生命周期与归属不同；继承会把 grant events
    调度管线耦合进本模块。此处仅套用其 acquire-all → body → release 的
    上下文管理器形态。
    """

    def __init__(self, context: TestcaseAclnn, device_ids: List[int], acl_dll=None, hccl_dll=None):
        self._context = context
        self._device_ids = list(device_ids)
        self._acl_dll = acl_dll if acl_dll is not None else ctypes.CDLL("libascendcl.so")
        self._hccl_dll = hccl_dll if hccl_dll is not None else ctypes.CDLL("libhcomm.so")
        self._hccl_dll.HcclCommInitAll.restype = ctypes.c_uint32
        self._retained = False
        self.comm_handles: List[int] = []
        self.ctx_vals: list = []
        self.stm_vals: list = []
        self.extra_comms: List[int] = []
        self.ep_comm_map: Dict[int, int] = {}
        self.tp_comm_map: Dict[int, int] = {}

    def __enter__(self) -> "HcclResourceScope":
        # __exit__ only runs for a successfully entered context: a failure
        # inside setup must clean up the partial state HERE and re-raise.
        try:
            self._acquire_all()
        except BaseException:
            self._release()
            raise
        return self

    def _acquire_all(self) -> None:
        ndev = len(self._device_ids)
        logging.info(f"Multi-device: aclrtSetDevice for all {self._device_ids}")
        for did in self._device_ids:
            with contextlib.suppress(Exception):
                self._acl_dll.aclrtResetDevice(ctypes.c_int32(did))
        for did in self._device_ids:
            ret = self._acl_dll.aclrtSetDevice(ctypes.c_int32(did))
            if ret != 0:
                raise RuntimeError(f"aclrtSetDevice failed with ret={ret} for device {did}")
        c_devices = (ctypes.c_int32 * ndev)(*self._device_ids)
        hccl_comms = (ctypes.c_void_p * ndev)()
        logging.info(f"Multi-device: HcclCommInitAll with {self._device_ids}")
        ret = self._hccl_dll.HcclCommInitAll(ctypes.c_uint32(ndev), c_devices, hccl_comms)
        if ret != 0:
            raise RuntimeError(f"HcclCommInitAll failed with ret={ret}")
        logging.info("Multi-device: HcclCommInitAll success")
        self.comm_handles = [v if isinstance(v, int) else v.value if hasattr(v, "value") else 0 for v in hccl_comms]
        self._init_dual_comm_groups()
        self._create_contexts_streams()

    def _init_comm_for_devs(self, dev_list: List[int]) -> List[int]:
        n = len(dev_list)
        c_devs = (ctypes.c_int32 * n)(*dev_list)
        c_comms = (ctypes.c_void_p * n)()
        ret = self._hccl_dll.HcclCommInitAll(ctypes.c_uint32(n), c_devs, c_comms)
        if ret != 0:
            raise RuntimeError(f"HcclCommInitAll failed with ret={ret} for devices {dev_list}")
        return [v if isinstance(v, int) else v.value if hasattr(v, "value") else 0 for v in c_comms]

    def _init_dual_comm_groups(self) -> None:
        if not _dual_comm_requested(self._context.attributes):
            return
        attrs = self._context.attributes or {}
        device_ids = self._device_ids
        ndev = len(device_ids)
        ep_ws = int(attrs.get("epWorldSize", 0))
        tp_ws = int(attrs.get("tpWorldSize", 0))
        if ep_ws <= 0 or tp_ws <= 0 or ndev % ep_ws or ndev % tp_ws:
            raise ValueError(
                f"Invalid dual-comm topology: device_count={ndev}, epWorldSize={ep_ws}, tpWorldSize={tp_ws}"
            )
        n_ep_groups = ndev // ep_ws
        n_tp_groups = ndev // tp_ws

        # A full-world EP/TP group is exactly the communicator created above.
        # Re-initializing the same full rank set with HcclCommInitAll can block
        # indefinitely on Ascend950, so reuse the primary handles instead.
        if ep_ws == ndev:
            self.ep_comm_map = dict(zip(device_ids, self.comm_handles))
        else:
            # EP groups stride by TP width: with ep=2, tp=2 this is grid ranks
            # [0, 2] and [1, 3]. Grid ranks are positions in device_ids;
            # HcclCommInitAll takes device ids, so map before initializing —
            # non-0-based device lists previously initialized subgroups on
            # the wrong devices (grid indices fed to HcclCommInitAll as-is).
            for g in range(n_ep_groups):
                grid_base = (g // tp_ws) * ep_ws * tp_ws
                tp_rank = g % tp_ws
                group_ranks = [grid_base + tp_rank + e * tp_ws for e in range(ep_ws)]
                group_devs = [device_ids[r] for r in group_ranks]
                handles = self._init_comm_for_devs(group_devs)
                for dev, h in zip(group_devs, handles):
                    self.ep_comm_map[dev] = h
                self.extra_comms.extend(handles)

        if tp_ws == ndev:
            self.tp_comm_map = dict(zip(device_ids, self.comm_handles))
        else:
            # TP groups are contiguous in the same EP/TP grid: grid ranks
            # [0, 1] and [2, 3] for ep=2, tp=2.
            for g in range(n_tp_groups):
                grid_base = (g // ep_ws) * ep_ws * tp_ws
                ep_rank = g % ep_ws
                group_ranks = [grid_base + ep_rank * tp_ws + t for t in range(tp_ws)]
                group_devs = [device_ids[r] for r in group_ranks]
                handles = self._init_comm_for_devs(group_devs)
                for dev, h in zip(group_devs, handles):
                    self.tp_comm_map[dev] = h
                self.extra_comms.extend(handles)
        logging.info(f"Dual-comm: ep_groups={n_ep_groups} tp_groups={n_tp_groups} ep_ws={ep_ws} tp_ws={tp_ws}")

    def _create_contexts_streams(self) -> None:
        for did in self._device_ids:
            ret = self._acl_dll.aclrtSetDevice(ctypes.c_int32(did))
            if ret != 0:
                raise RuntimeError(f"aclrtSetDevice failed with ret={ret} for device {did}")
            c_ctx = ctypes.c_void_p()
            ret = self._acl_dll.aclrtCreateContext(ctypes.pointer(c_ctx), ctypes.c_int32(did))
            if ret != 0:
                raise RuntimeError(f"aclrtCreateContext failed with ret={ret} for device {did}")
            c_stm = ctypes.c_void_p()
            ret = self._acl_dll.aclrtCreateStream(ctypes.pointer(c_stm))
            if ret != 0:
                raise RuntimeError(f"aclrtCreateStream failed with ret={ret} for device {did}")
            # 增量入账：中途失败时 __exit__ 按"已建到哪算哪"清理。
            self.ctx_vals.append(c_ctx.value)
            self.stm_vals.append(c_stm.value)

    def retain(self) -> None:
        """Keep handles past this scope: stash them on the testcase for the
        golden stage (released later by release_retained_multi_device_resources)."""
        self._retained = True
        self._context._multi_device_hccl_handles = {
            "comm_handles": self.comm_handles,
            "ctx_vals": self.ctx_vals,
            "stm_vals": self.stm_vals,
            "extra_comms": self.extra_comms,
        }

    def __exit__(self, *exc) -> None:
        if self._retained:
            return
        self._release()

    def _release(self) -> None:
        _cleanup_multi_device_resources(
            self._hccl_dll,
            self._acl_dll,
            self._device_ids,
            self.comm_handles,
            self.ctx_vals,
            self.stm_vals,
            self.extra_comms,
        )


def profile_multi_device(context: TestcaseAclnn, device_ids: List[int], dev_id: int) -> ApiProfilingResult:
    """
    Single-process multi-device execution via threads.
    Each thread gets its own context copy with rank-specific seed.
    Supports dual communication groups (ep + tp) declared via the
    epWorldSize/tpWorldSize testcase attributes. The HCCL/context/stream
    lifecycle is owned by HcclResourceScope: any setup-phase failure
    releases partial resources on scope exit; success retains the handles
    for the golden stage.
    """
    device = get_aclnn_device(dev_id)
    device_ids = list(device_ids)
    ndev = len(device_ids)
    pre_exec_barrier = threading.Barrier(ndev, timeout=1800)
    thread_results: Dict[int, ApiProfilingResult] = {}
    thread_contexts: Dict[int, TestcaseAclnn] = {}
    thread_errors: Dict[int, Exception] = {}
    threads = []
    with HcclResourceScope(context, device_ids, acl_dll=device._acl_dll) as scope:
        for idx, rank_id in enumerate(device_ids):
            t = threading.Thread(
                target=_launch_one_thread,
                args=(
                    context,
                    rank_id,
                    idx,
                    scope.comm_handles[idx],
                    scope.ctx_vals[idx],
                    scope.stm_vals[idx],
                    thread_results,
                    thread_contexts,
                    thread_errors,
                    device,
                    scope.ep_comm_map.get(rank_id, 0),
                    scope.tp_comm_map.get(rank_id, 0),
                    pre_exec_barrier,
                ),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        logging.info("Multi-device: all threads joined")
        if thread_errors:
            # 执行失败：抛首个错误，__exit__ 立即清理（fail-fast），避免泄漏
            first_err_dev = min(thread_errors.keys())
            raise thread_errors[first_err_dev]
        # 成功：保留 comm/context/stream 透传给 golden 阶段，
        # 由 release_retained_multi_device_resources（golden 分发前）释放。
        context._multi_device_thread_contexts = thread_contexts
        scope.retain()
    return thread_results.get(device_ids[0], ApiProfilingResult.fail("UNKNOWN"))


def _cleanup_multi_device_resources(hccl_dll, acl_dll, device_ids, comm_handles, ctx_vals, stm_vals, extra_comms):
    """统一清理 HCCL comm / stream / context / device。

    句柄列表可能是 setup 中途失败的部分状态：逐设备按"已建到哪算哪"
    清理，未建到的资源跳过，保证 HcclResourceScope.__exit__ 对任意失败
    点都能安全回退。
    """
    for h in extra_comms:
        try:
            ret = hccl_dll.HcclCommDestroy(ctypes.c_void_p(h))
            logging.info(f"Cleanup: extra HcclCommDestroy ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: extra HcclCommDestroy exception: {e}")
    for i, did in enumerate(device_ids):
        # 必须在每个 device 操作前先 aclrtSetDevice(did)：
        # 多设备场景下主线程 current context 是创建循环里最后 set 的 device（device_ids[-1]），
        # 不切回本 did 就调 HcclCommDestroy / DestroyStream / DestroyContext 会因
        # context 与 handle 不匹配触发 SIGSEGV (exit -11)。
        try:
            ret = acl_dll.aclrtSetDevice(ctypes.c_int32(did))
            logging.info(f"Cleanup: aclrtSetDevice device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtSetDevice device {did} exception: {e}")
        if i < len(comm_handles):
            try:
                ret = hccl_dll.HcclCommDestroy(ctypes.c_void_p(comm_handles[i]))
                logging.info(f"Cleanup: HcclCommDestroy device {did} ret={ret}")
            except Exception as e:
                logging.warning(f"Cleanup: HcclCommDestroy device {did} exception: {e}")
        if i < len(stm_vals):
            # 销毁 stream 前先 sync，避免异步 aclnnRun 任务未完成时 destroy 触发 SIGSEGV
            try:
                ret = acl_dll.aclrtSynchronizeStream(ctypes.c_void_p(stm_vals[i]))
                logging.info(f"Cleanup: aclrtSynchronizeStream device {did} ret={ret}")
            except Exception as e:
                logging.warning(f"Cleanup: aclrtSynchronizeStream device {did} exception: {e}")
            try:
                ret = acl_dll.aclrtDestroyStream(ctypes.c_void_p(stm_vals[i]))
                logging.info(f"Cleanup: aclrtDestroyStream device {did} ret={ret}")
            except Exception as e:
                logging.warning(f"Cleanup: aclrtDestroyStream device {did} exception: {e}")
        if i < len(ctx_vals):
            try:
                ret = acl_dll.aclrtDestroyContext(ctypes.c_void_p(ctx_vals[i]))
                logging.info(f"Cleanup: aclrtDestroyContext device {did} ret={ret}")
            except Exception as e:
                logging.warning(f"Cleanup: aclrtDestroyContext device {did} exception: {e}")
    for did in device_ids:
        try:
            ret = acl_dll.aclrtResetDevice(ctypes.c_int32(did))
            logging.info(f"Cleanup: aclrtResetDevice device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtResetDevice device {did} exception: {e}")


def release_retained_multi_device_resources(context: TestcaseAclnn, device_ids: List[int]):
    """Release resources retained for the multi-device golden stage."""
    handles = getattr(context, "_multi_device_hccl_handles", None)
    if handles is None:
        return
    try:
        _cleanup_multi_device_resources(
            ctypes.CDLL("libhcomm.so"),
            ctypes.CDLL("libascendcl.so"),
            device_ids,
            handles["comm_handles"],
            handles["ctx_vals"],
            handles["stm_vals"],
            handles["extra_comms"],
        )
    except Exception:
        logging.exception("Multi-device resource cleanup failed")
    finally:
        context._multi_device_hccl_handles = None


def compare_multi_device(context: TestcaseAclnn, device_ids: List[int]) -> ApiComparisonResult:
    """Multi-device golden dispatch (runs after execution by construction).

    Plugin-first: the operator's spec plugin golden, then the
    hccl_dispatcher framework entry as the no-plugin fallback.
    """
    switches = get_global_storage()
    thread_contexts = context._multi_device_thread_contexts
    all_passed = True
    all_precision = []
    for did in device_ids:
        tc = thread_contexts.get(did)
        if tc is None or tc.prof_result.failed():
            all_passed = False
            logging.error(f"Multi-device: rank dev={did} execution failed, skipping golden")
    if not all_passed:
        return ApiComparisonResult(None).set("MULTI_DEVICE_EXECUTION_FAILURE", "FAIL")
    release_retained_multi_device_resources(context, device_ids)
    plugin_golden = SpecResolver(context.api_name, "aclnn", getattr(switches, "plugin_path", None)).resolve_golden()
    if plugin_golden is not None:
        # Inject proc_timeout into thread_contexts attributes so
        # cascade workers can read it via _get_timeout().
        timeout_val = int(getattr(switches, "proc_timeout", 0) or 3600)
        for did in device_ids:
            tc = thread_contexts.get(did)
            if tc is not None and tc.attributes is not None:
                tc.attributes["proc_timeout"] = timeout_val
        logging.info(f"Multi-device: cascade timeout = {timeout_val}s, using plugin golden for {context.api_name}")
        plugin_golden(thread_contexts, device_ids, all_precision)
    else:
        golden_multi_device_compare(thread_contexts, device_ids, all_precision)
    has_fail = any("FAIL" in p or "EXCEPTION" in p for p in all_precision)
    return ApiComparisonResult(None).set(
        ",".join(all_precision) if all_precision else "MULTI_DEVICE_COMPARE_FAILURE",
        "PASS" if not has_fail else "FAIL",
    )
