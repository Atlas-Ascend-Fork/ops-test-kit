#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""三方采集编排：dispatch/collect 两段式句柄 + 重叠预算 + remote client 门面。

三方采集仅 remote 通路（ttk.conf.yaml remote.endpoints）；远端配置/端点
不可用由 client 既有语义降级。
"""

# 数据位模式（need_data / need_perf 按位组合）——正本在 ttk.remote（协议层
# client/server 共享），此处 re-export 保持 worker 侧统一从 common.xpu 导入
from ....remote import DATA as XPU_DATA  # noqa: F401
from ....remote import PERF as XPU_PERF  # noqa: F401
from ....remote import xpu_mode  # noqa: F401

# 重叠请求的 NPU 预算默认缓冲（--proc-timeout 未配置时）
_DEFAULT_NPU_BUDGET_S = 600.0

# collect 收割上界的固定余量：重试退避/建连等残余开销
_JOIN_SLACK_S = 300.0


def xpu_overlap_enabled() -> bool:
    """重叠总闸（ttk.conf.yaml remote 段 xpu_overlap，缺省 true）——XPU 服务端
    负载紧张时运维全局关闭，回到串行时序。"""
    from ....config.loader import get_remote_config

    try:
        config = get_remote_config()
    except Exception:  # noqa: BLE001  配置不可用时保守取默认（不因配置问题关掉功能）
        return True
    return bool(getattr(config, "xpu_overlap", True)) if config is not None else True


def overlap_deadline_s(switches) -> float:
    """重叠请求的 deadline 预算：dispatch_deadline + NPU 预算。

    NPU 预算锚点 ``--proc-timeout``（SWITCHES 默认 3600s = 用例全程上界，
    语义即 NPU 段上界；测试桩无该字段时取 600s 缓冲）——重叠请求横跨整个
    NPU 执行期，deadline 须覆盖。XPU 侧自身的执行上界由服务端 run_deadline
    兜底（与等待期无关），客户端预算只需覆盖"等待 NPU + XPU 执行 + 传输"。
    """
    from ....config.loader import get_remote_config

    try:
        config = get_remote_config()
        dispatch_deadline = float(getattr(config, "dispatch_deadline_s", 300.0) or 300.0) if config else 300.0
    except Exception:  # noqa: BLE001
        dispatch_deadline = 300.0
    proc_timeout = getattr(switches, "proc_timeout", None)
    npu_budget = float(proc_timeout) if proc_timeout else _DEFAULT_NPU_BUDGET_S
    return dispatch_deadline + npu_budget


def xpu_join_timeout(deadline_s: float) -> float:
    """重叠句柄 collect 的 join 上界：2×deadline + 余量。

    线程上界 ≈ deadline（重试主循环）+ 一份 socket timeout（deadline
    之后的末次在途请求）；超界 collect 抛 TimeoutError，不无限等待。
    """
    return 2.0 * float(deadline_s) + _JOIN_SLACK_S


def snapshot_xpu_inputs(values):
    """输入深快照（numpy 叶子 copy、嵌套容器递归、其余原样）。

    重叠路径的竞争防线：后台线程序列化读输入 vs NPU 执行原地改写输入
    （geir 图执行 backing storage / e2e inplace 算子）。派生与快照必须在
    主线程 dispatch 之前完成。
    """
    import numpy

    if isinstance(values, (list, tuple)):
        return type(values)(snapshot_xpu_inputs(v) for v in values)
    if isinstance(values, numpy.ndarray):
        return values.copy()
    return values


class XpuDispatch:
    """三方采集统一句柄（dispatch/collect 两段式，NPU 执行期重叠）。

    ready：同步路径（golden-XPU / manual 源 / 自动降级 / 总闸关闭）——dispatch
    时已内联完成采集，collect 即取。background：重叠路径——现同步采集函数
    原样入 daemon 线程，结果/异常装箱，collect 时 join 收割；daemon 保证 NPU
    失败早退不背长等待、进程退出不挂死（线程上界 = join_timeout）。

    collect 幂等（首查缓存）；线程内异常在 collect 处重放；带 join_timeout
    的 background 形态收割有界，超时抛 TimeoutError。
    """

    def __init__(self):
        self._result = None
        self._error = None
        self._thread = None
        self._collected = False
        self._discarded = False
        self._join_timeout = None

    @property
    def pending(self) -> bool:
        """background 形态且未收割（通路据此发 OnXpuCollect 阶段通知）。"""
        return self._thread is not None and not self._collected

    @classmethod
    def ready(cls, result) -> "XpuDispatch":
        handle = cls()
        handle._result = result
        handle._collected = True
        return handle

    @classmethod
    def background(cls, fn, *args, join_timeout=None, **kwargs) -> "XpuDispatch":
        import threading

        handle = cls()
        handle._join_timeout = join_timeout

        def _run():
            try:
                handle._result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001  装箱重放，collect 侧统一处理
                handle._error = exc

        handle._thread = threading.Thread(target=_run, daemon=True, name="xpu-dispatch")
        handle._thread.start()
        return handle

    def collect(self):
        """收割（幂等）：ready 即取；background join 后异常重放；
        join_timeout 超界抛 TimeoutError（线程 daemon 自然消亡）。"""
        if self._discarded:
            raise RuntimeError("XpuDispatch already discarded")
        if not self._collected:
            if self._thread is not None:
                self._thread.join(self._join_timeout)
                if self._thread.is_alive():
                    self._collected = True
                    raise TimeoutError(
                        f"XPU collect join timeout ({self._join_timeout:.0f}s): background dispatch stuck"
                    )
            self._collected = True
            if self._error is not None:
                raise self._error
        return self._result

    def discard(self, timeout: float = 5.0) -> None:
        """弃置句柄：短超时 join 后放弃（daemon 线程自然消亡）。

        供编排分支提前结束时调用，不阻塞失败上报；弃置后 collect 抛错。
        """
        if self._collected:
            return
        self._discarded = True
        if self._thread is not None:
            self._thread.join(timeout)


def collect_third_party(**kwargs):
    """三方采集门面——与 ttk.remote.client.collect_third_party 同签名同返回（仅 remote 通路）。

    远端配置/端点不可用由 client 既有语义降级：(None, None, None) → 调用方
    GOLDEN_FAILURE（cross_check）或空指标（xpu_perf，其缺配置在启动期由
    normalize_xpu_options 硬错）。"""
    from ttk.remote.client import collect_third_party as remote_collect

    return remote_collect(**kwargs)
