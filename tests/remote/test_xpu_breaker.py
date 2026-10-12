# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""三方采集连续失败熔断器。

背景：三方采集失败是被设计成"降级不致命"的，对偶发抖动是对的；但对配置性不
兼容（免传输打到旧服务端、端点整段不可用、服务端缺 golden）每一例都会失败，
降级就变成了跑满全程产出空数据。熔断器只管后者：连续失败到阈值就中止跑批。
"""

import multiprocessing as mp

import pytest

from ttk.remote import xpu_breaker as br


@pytest.fixture
def root(tmp_path):
    br.reset(str(tmp_path))
    return str(tmp_path)


def test_consecutive_failures_trip(root):
    for _ in range(br.DEFAULT_LIMIT - 1):
        br.record(root, ok=False, reason="third_party unavailable")
    assert br.tripped(root)[0] is False, "未到阈值不应熔断"
    br.record(root, ok=False, reason="third_party unavailable")
    tripped, reason = br.tripped(root)
    assert tripped is True
    assert "third_party unavailable" in reason


def test_success_resets_counter(root):
    for _ in range(br.DEFAULT_LIMIT - 1):
        br.record(root, ok=False, reason="boom")
    br.record(root, ok=True)
    for _ in range(br.DEFAULT_LIMIT - 1):
        br.record(root, ok=False, reason="boom")
    assert br.tripped(root)[0] is False, "成功一例必须清零连续计数"


def test_scattered_failures_do_not_trip(root):
    # 偶发抖动：失败/成功交替，永远不该熔断
    for _ in range(br.DEFAULT_LIMIT * 3):
        br.record(root, ok=False, reason="flaky")
        br.record(root, ok=True)
    assert br.tripped(root)[0] is False


def test_limit_zero_disables(root, monkeypatch):
    monkeypatch.setenv("TTK_XPU_FAIL_LIMIT", "0")
    for _ in range(50):
        br.record(root, ok=False, reason="boom")
    assert br.tripped(root)[0] is False, "阈值 0 = 关闭熔断"


def test_env_overrides_limit(root, monkeypatch):
    monkeypatch.setenv("TTK_XPU_FAIL_LIMIT", "2")
    br.record(root, ok=False, reason="boom")
    assert br.tripped(root)[0] is False
    br.record(root, ok=False, reason="boom")
    assert br.tripped(root)[0] is True


def test_reset_clears_tripped_state(root):
    for _ in range(br.DEFAULT_LIMIT):
        br.record(root, ok=False, reason="boom")
    assert br.tripped(root)[0] is True
    br.reset(root)
    assert br.tripped(root)[0] is False


def test_tripped_is_safe_when_never_recorded(tmp_path):
    # 没有 xpu 的跑批不会 record，tripped 必须安静地返回 False
    assert br.tripped(str(tmp_path)) == (False, "")


def _worker(root, n):
    for _ in range(n):
        br.record(root, ok=False, reason="boom")


def test_counts_across_processes(root, monkeypatch):
    # 跑批的用例跑在各自的 worker 进程里，计数必须跨进程累加
    monkeypatch.setenv("TTK_XPU_FAIL_LIMIT", "6")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(root, 3)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    assert br.tripped(root)[0] is True, "两个进程各失败 3 例，合计 6 例应熔断"


# ---- 客户端接线: dispatch_xpu 把每例三方结果喂进熔断器 ----------------------


class _Sw:
    def __init__(self, root):
        self.root_path = root


def test_judge_pass_when_any_provider_passes():
    from ttk.remote.client import _judge_xpu_results

    assert _judge_xpu_results({"torch": {"status": "PASS"}, "tf": {"status": "FAIL", "error": "x"}})["ok"] is True


def test_judge_fail_carries_first_error():
    from ttk.remote.client import _judge_xpu_results

    got = _judge_xpu_results({"torch": {"status": "FAIL", "error": "Server returned 400: ..."}})
    assert got["ok"] is False
    assert "400" in got["reason"]


def test_judge_empty_results_is_failure():
    from ttk.remote.client import _judge_xpu_results

    # 43 例那种 xpu_metrics={} —— 三方压根没发起, 必须计入失败
    got = _judge_xpu_results({})
    assert got["ok"] is False
    assert got["reason"] == "empty xpu_results"


def test_record_breaker_writes_state(root):
    from ttk.remote.client import _record_breaker

    _record_breaker(_Sw(root), ok=False, reason="no usable provider")
    assert br._read(br._state_path(root))["consec_fail"] == 1
