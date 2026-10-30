# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TorchProfiler 数据驱动 activities + 共享引擎 summarize_events 测试。

TorchProfiler.__init__ 从 backend.profile['profiler'] 读取 activities
（数据驱动，无 =='gpu'/torch_lib=='cuda' 字符串比较）；result() 经
summarize_events 聚合（_is_device_kernel 三身份剔除 + _self_device_time
三候选纯自身回退），覆盖 torch 2.7+（self_device_time_total）和 legacy
（self_{device}_time_total，如 self_cuda_time_total / self_mlu_time_total）。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ttk.core_modules.common.profiler import get_profiler
from ttk.core_modules.common.profiler.torch_profiler import TorchProfiler


class _Evt:
    """Minimal stand-in for a torch.profiler.Event for fallback tests."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.key = "op"
        self.count = 1


class _FakeBackend:
    """Minimal backend stand-in carrying only what TorchProfiler reads."""

    def __init__(self, profile):
        self.profile = profile

    def device_type(self):
        return "fake"

    def is_npu(self):
        return False


class _FakeEvent:
    """Stand-in for a torch.profiler.Event for result() tests."""

    def __init__(self, key, count, cpu_time_total=0.0, **device_attrs):
        self.key = key
        self.count = count
        self.cpu_time_total = cpu_time_total
        self.__dict__.update(device_attrs)


class _FakeKeyAverages:
    """Fake _prof whose key_averages() returns a fixed event list."""

    def __init__(self, events):
        self._events = events

    def key_averages(self):
        return self._events


def _prof_for_result(events, device_acts, device="cuda", device_time_attr=None):
    """Build a TorchProfiler (bypassing __init__) wired for result()."""
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._prof = _FakeKeyAverages(events)
    prof._device_acts = device_acts
    prof._device = device
    prof._device_time_attr = device_time_attr
    return prof


def test_self_device_time_candidate_order():
    """_self_device_time 三候选顺序：显式 attr → self_device_time_total
    (torch 2.7+) → self_{device}_time_total (legacy)。is not None 守卫保证
    candidate1 的合法 0.0 不被跳过而误入 fallback。"""
    from ttk.core_modules.common.profiler.base import _self_device_time

    # explicit device_time_attr wins over both self_ candidates
    evt = _Evt(cuda_time_total=5.0, self_device_time_total=1.0, self_cuda_time_total=2.0)
    assert _self_device_time(evt, "cuda", "cuda_time_total") == 5.0
    # torch 2.7+ self_device_time_total wins over legacy self_cuda_time_total
    evt = _Evt(self_device_time_total=1.0, self_cuda_time_total=2.0)
    assert _self_device_time(evt, "cuda") == 1.0
    # legacy fallback when self_device_time_total is absent
    evt = _Evt(self_cuda_time_total=2.0)
    assert _self_device_time(evt, "cuda") == 2.0
    # genuine 0.0 on candidate1 is NOT skipped (not truthiness)
    evt = _Evt(self_device_time_total=0.0, self_cuda_time_total=2.0)
    assert _self_device_time(evt, "cuda") == 0.0
    # nothing present -> 0.0
    assert _self_device_time(_Evt(), "cuda") == 0.0


def test_summarize_events_aggregates_and_filters():
    """共享引擎 summarize_events：容器 span/aten::/cuda 剔除、self_* 三候选
    回退、>0 过滤；total_cpu_us 对全部事件求和（不按设备身份剔除）。"""
    from ttk.core_modules.common.profiler.torch_profiler import summarize_events

    events = [
        _FakeEvent("k1", 2, cpu_time_total=10.0, self_device_time_total=200.0),
        _FakeEvent("k2", 1, cpu_time_total=5.0, self_cuda_time_total=80.0),  # legacy 候选
        _FakeEvent("idle", 1, cpu_time_total=1.0, self_device_time_total=0.0),  # >0 过滤
        _FakeEvent("ProfilerStep#3", 1, cpu_time_total=3.0, self_device_time_total=99.0),  # 容器 span 剔除
        _FakeEvent("aten::add", 1, cpu_time_total=4.0, self_device_time_total=0.0),  # CPU 算子视角剔除
        _FakeEvent("cudaLaunchKernel", 1, cpu_time_total=2.0, self_device_time_total=0.0),  # host API 剔除
    ]
    kernels, total_device_us, total_cpu_us = summarize_events(events, "cuda")
    assert [k.name for k in kernels] == ["k1", "k2"]
    assert total_device_us == 280.0
    assert total_cpu_us == 25.0  # 10+5+1+3+4+2 全量
    assert kernels[0].calls == 2
    assert kernels[0].avg_us == 100.0


# --- A3: CannMsProfiler（tf/onnx 共享）msprof 采集 ---


class _FakeCannMsProfiler:
    """CannMsProfiler 内部包 MsProfiler 替身（记录调用序 + 结果桩）。"""

    def __init__(self, *args, **kwargs):
        self.entered = False
        self.stepped = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def step(self):
        self.stepped = True

    def __exit__(self, *exc):
        self.exited = True

    def result(self, repeat_count):
        from ttk.core_modules.common.profiler.base import KernelDetails, KernelInfo, ProfileResult

        k = KernelInfo(name="Add", device_us=2.0 * repeat_count, calls=repeat_count, avg_us=2.0, max_us=2.5, min_us=1.5)
        return ProfileResult(
            elapsed_us=2.0,
            kernel_details=KernelDetails(kernels=[k], total_device_us=2.0 * repeat_count, total_cpu_us=0.0),
            source="msprof",
        )


def _cann_prof(monkeypatch, tmp_path, warmup_count, fake_cls, backend=None):
    import ttk.core_modules.common.profiler.ms_profiler as ms_profiler_mod
    from ttk.core_modules.common.profiler.base import ProfilerConfig
    from ttk.core_modules.common.profiler.cann_profiler import CannMsProfiler

    created = []
    monkeypatch.setattr(ms_profiler_mod, "MsProfiler", lambda *a, **k: created.append(fake_cls(*a, **k)) or created[-1])
    if backend is None:
        backend = _SyncBackend()
    return (
        CannMsProfiler(
            ProfilerConfig(testcase_name="cann_case", root_path=str(tmp_path), warmup_count=warmup_count), backend
        ),
        created,
    )


class _SyncBackend:
    """CannMsProfiler 开窗前同步用的最小 backend 替身。"""

    def __init__(self):
        self.synced = 0

    def synchronize(self, dev_id=0):  # noqa: ARG002
        self.synced += 1


def test_cann_ms_profiler_init_failure_leaves_perf_empty(tmp_path, monkeypatch, caplog):
    """libmsprofiler.so 缺失 / MsProfiler 构造失败不得杀掉用例——结果留空
    （disabled 空结果、无 kernel 明细），与 kernel/aclnn/geir 同口径，
    不把墙钟数混进设备耗时列。"""
    import ttk.core_modules.common.profiler.ms_profiler as ms_profiler_mod
    from ttk.core_modules.common.profiler.base import ProfilerConfig
    from ttk.core_modules.common.profiler.cann_profiler import CannMsProfiler

    def _boom(*args, **kwargs):
        raise OSError("libmsprofiler.so: cannot open shared object file")

    monkeypatch.setattr(ms_profiler_mod, "MsProfiler", _boom)
    prof = CannMsProfiler(ProfilerConfig(testcase_name="fb", root_path=str(tmp_path), warmup_count=0))
    with prof:
        pass
    assert prof._prof is None
    result = prof.result(repeat_count=2)
    assert result.source == "disabled"
    assert result.elapsed_us == 0.0  # 结果列落 "----"，不塞墙钟
    assert result.kernel_details is None
    assert "perf columns left empty" in caplog.text


def test_cann_ms_profiler_cleans_up_partial_msprof_init(tmp_path, monkeypatch):
    """step() 中途失败（如 aclprofInit 成功、CreateConfig 失败）时，部分初始化的
    会话必须经 __exit__ 清理（异常发生在内部 _start_msprof，with 不会替它收）。"""

    created = []

    class _FailingStepMsProfiler:
        def __init__(self, *args, **kwargs):
            self.exited = False
            created.append(self)

        def __enter__(self):
            return self

        def step(self):
            raise RuntimeError("aclprofCreateConfig failed")

        def __exit__(self, *exc):
            self.exited = True

    prof, _ = _cann_prof(monkeypatch, tmp_path, warmup_count=0, fake_cls=_FailingStepMsProfiler)
    with prof:
        pass
    assert prof._prof is None  # degraded
    assert created  # MsProfiler was constructed before step() failed
    assert created[0].exited  # partial init cleaned up


def test_cann_ms_profiler_window_open_semantics(tmp_path, monkeypatch):
    """采集窗开启点与 e2e 循环协议对齐：开窗前先 synchronize 设备（TF 异步
    执行下 warmup 轮 kernel 可能仍在飞），warmup>0 时 __enter__ 不开窗、
    首个 step()（warmup 轮尾）开窗且幂等；warmup=0 时 __enter__ 即开窗。"""
    backend = _SyncBackend()
    prof, created = _cann_prof(monkeypatch, tmp_path, warmup_count=1, fake_cls=_FakeCannMsProfiler, backend=backend)
    with prof:
        assert created == []  # enter 未开窗（warmup 存在）
        assert backend.synced == 0  # 未开窗不触发同步
        prof.step()
        assert created[0].entered  # warmup 轮尾开窗
        assert created[0].stepped
        assert backend.synced == 1  # 开窗前先同步设备
        prof.step()
        assert len(created) == 1  # 幂等：不重建 MsProfiler
        assert backend.synced == 1
    assert created[0].exited  # with 结束：aclprof 会话关闭

    backend = _SyncBackend()
    prof, created = _cann_prof(monkeypatch, tmp_path, warmup_count=0, fake_cls=_FakeCannMsProfiler, backend=backend)
    with prof:
        assert created[0].entered  # 无 warmup：enter 即开窗
        assert created[0].stepped
        assert backend.synced == 1
    assert created[0].exited


def test_cann_ms_profiler_sync_failure_does_not_block_window(tmp_path, monkeypatch):
    """synchronize 抛异常为 best-effort（不得吞掉采集窗开启）。"""

    class _BrokenSyncBackend(_SyncBackend):
        def synchronize(self, dev_id=0):
            raise RuntimeError("sync failed")

    prof, created = _cann_prof(
        monkeypatch, tmp_path, warmup_count=0, fake_cls=_FakeCannMsProfiler, backend=_BrokenSyncBackend()
    )
    with prof:
        assert created[0].stepped  # 同步失败仍照常开窗
    assert prof._prof is not None


def test_cann_ms_profiler_result_delegates_msprof(tmp_path, monkeypatch):
    """msprof 命中：elapsed/kernel_details 取 MsProfiler.result；采集无数据
    （空 op_statistic）→ 空结果直通，不降级墙钟。"""
    prof, created = _cann_prof(monkeypatch, tmp_path, warmup_count=1, fake_cls=_FakeCannMsProfiler)
    with prof:
        prof.step()
    res = prof.result(repeat_count=10)
    assert res.source == "msprof"
    assert res.elapsed_us == 2.0
    assert [k.name for k in res.kernel_details.kernels] == ["Add"]

    class _EmptyMsProfiler(_FakeCannMsProfiler):
        def result(self, repeat_count):
            from ttk.core_modules.common.profiler.base import ProfileResult

            return ProfileResult(source="msprof")

    prof, _ = _cann_prof(monkeypatch, tmp_path, warmup_count=0, fake_cls=_EmptyMsProfiler)
    with prof:
        pass
    empty = prof.result(repeat_count=2)
    assert empty.source == "msprof"
    assert empty.elapsed_us == 0.0  # 空结果直通（Device/us 落 "----"）


# --- C3: find_newest_csv mtime 择新（不依赖 rmtree 副作用）——见 profiler/test_msprof_readers.py ---


def test_result_device_branch_collects_kernels():
    """result() with _device_acts: device kernels via _device_time, elapsed = total/repeat."""
    events = [
        _FakeEvent("k1", 2, cpu_time_total=10.0, self_cuda_time_total=200.0),
        _FakeEvent("k2", 1, cpu_time_total=5.0, self_cuda_time_total=80.0),
        _FakeEvent("idle", 1, cpu_time_total=1.0, self_cuda_time_total=0.0),  # skipped
    ]
    prof = _prof_for_result(events, device_acts=["CUDA"], device="cuda")
    res = prof.result(repeat_count=2)
    # total_device_us = 200+80 = 280; elapsed = 280/2 = 140
    assert res.elapsed_us == 140.0
    assert res.kernel_details.total_device_us == 280.0
    names = [k.name for k in res.kernel_details.kernels]
    assert names == ["k1", "k2"]  # idle (0.0) skipped
    k1 = res.kernel_details.kernels[0]
    assert k1.device_us == 200.0
    assert k1.calls == 2
    assert k1.avg_us == 100.0


def test_result_cpu_branch_reports_cpu_only():
    """result() without _device_acts（CPU backend）：elapsed=0、无 device kernels，
    total_cpu_us 按 repeat 摊销。"""
    events = [_FakeEvent("cpu_op", 3, cpu_time_total=30.0), _FakeEvent("cpu_op2", 1, cpu_time_total=6.0)]
    prof = _prof_for_result(events, device_acts=[])
    res = prof.result(repeat_count=3)
    assert res.elapsed_us == 0.0
    assert res.kernel_details.kernels == []
    assert res.kernel_details.total_device_us == 0.0
    assert res.kernel_details.total_cpu_us == 12.0  # (30+6)/3


def test_torch_profiler_init_resolves_activities_and_rejects_unknown():
    """__init__ 数据驱动 activities（backend.profile['profiler']['activities']→
    ProfilerActivity 属性名），未知名 raise ValueError 并列出合法值。"""
    backend = _FakeBackend({"profiler": {"activities": ["CPU"], "device_time_attr": "my_time"}, "torch_lib": "cpu"})
    prof = TorchProfiler(backend)
    assert prof._device_acts == []  # CPU-only: 非 CPU activities 为空
    assert prof._device_time_attr == "my_time"
    assert prof._device == "cpu"

    bad = _FakeBackend({"profiler": {"activities": ["WARP_DRIVE"]}, "torch_lib": "cuda"})
    with pytest.raises(ValueError, match="unknown ProfilerActivity 'WARP_DRIVE'"):
        TorchProfiler(bad)


# --- 兜底 profiler：WallClock / Disabled ---


def test_wall_clock_profiler_result_semantics():
    """未 exit 前 result() → 0；exit 后 elapsed/repeat、无 kernel_details。"""
    from ttk.core_modules.common.profiler.base import WallClockProfiler

    prof = WallClockProfiler(None)
    assert prof.result(1).elapsed_us == 0.0  # result before exit
    with prof:
        pass
    res = prof.result(repeat_count=2)
    assert res.kernel_details is None
    assert res.elapsed_us >= 0.0


def test_disabled_profiler_result_is_empty():
    from ttk.core_modules.common.profiler.base import DisabledProfiler

    prof = DisabledProfiler()
    with prof:
        prof.step()  # no-op
    res = prof.result(4)
    assert res.elapsed_us == 0.0
    assert res.kernel_details is None


# --- I5: get_profiler routing ---


_TORCH_CPU_PROFILE = {"profiler": {"activities": ["CPU"]}, "torch_lib": "cpu"}


@pytest.mark.parametrize(
    ("api_name", "npu", "profile", "expected"),
    [
        ("tf.raw_ops.Add", True, None, "CannMsProfiler"),
        ("tf.raw_ops.Add", False, None, "WallClockProfiler"),
        ("tensorflow.add", True, None, "CannMsProfiler"),
        ("onnx.Add", True, None, "CannMsProfiler"),
        ("onnx.Add", False, None, "WallClockProfiler"),
        ("torch.add", True, {"profiler": "builtin"}, "TorchNpuProfiler"),
        ("torch.add", False, _TORCH_CPU_PROFILE, "TorchProfiler"),
        ("torch_npu.npu_rms_norm", True, None, "TorchNpuProfiler"),
        ("torch_npu.npu_rms_norm", False, None, "RuntimeError"),
        ("custom_op", True, None, "TorchNpuProfiler"),
        ("custom_op", False, None, "WallClockProfiler"),
    ],
)
def test_get_profiler_routing_matrix(api_name, npu, profile, expected):
    """factory 全路由矩阵（api_name 前缀 × backend）——每条分支一例，
    torch_npu.* + 非 NPU 是唯一 raise 路由。"""
    from ttk.core_modules.common.profiler.base import Profiler

    backend = SimpleNamespace(
        is_npu=lambda: npu,
        device_type=lambda: "npu" if npu else "xpu",
        profile=profile if profile is not None else {},
    )
    if expected == "RuntimeError":
        with pytest.raises(RuntimeError, match="requires NPU backend"):
            get_profiler(api_name, backend)
        return
    profiler = get_profiler(api_name, backend)
    assert type(profiler).__name__ == expected
    assert isinstance(profiler, Profiler)


def test_get_profiler_disabled_returns_disabled_profiler():
    from ttk.core_modules.common.profiler import ProfilerConfig
    from ttk.core_modules.common.profiler.base import DisabledProfiler

    backend = SimpleNamespace(is_npu=lambda: True, profile={})
    profiler = get_profiler("torch.add", backend, ProfilerConfig(enabled=False))
    assert isinstance(profiler, DisabledProfiler)


def test_get_profiler_rejects_non_profiler_config():
    backend = SimpleNamespace(is_npu=lambda: False, profile={})
    with pytest.raises(TypeError, match="must be ProfilerConfig"):
        get_profiler("torch.add", backend, config=object())


def test_npu_find_csv_picks_latest_mtime(tmp_path, caplog):
    """_find_csv returns the newest-mtime CSV when multiple trace subdirs exist.

    torch_npu.profiler's tensorboard_trace_handler writes each run into a new
    timestamped subdir under _outdir, so _outdir accumulates multiple copies
    of kernel_details.csv / operator_details.csv. os.walk yields subdirs in
    filesystem (os.listdir) order, not by time; _find_csv must pick by mtime to
    avoid reading a stale trace from a previous run.
    """
    import os

    from ttk.core_modules.common.profiler import ProfilerConfig
    from ttk.core_modules.common.profiler.torch_profiler import TorchNpuProfiler

    prof = TorchNpuProfiler(None, ProfilerConfig(testcase_name="fresh_capture", root_path=str(tmp_path)))
    prof._outdir = str(tmp_path)

    older_dir = tmp_path / "trace_older"
    newer_dir = tmp_path / "trace_newer"
    older_dir.mkdir()
    newer_dir.mkdir()
    older_csv = older_dir / "kernel_details.csv"
    newer_csv = newer_dir / "kernel_details.csv"
    older_csv.write_text("Name,Duration(us)\nold,10\n")
    newer_csv.write_text("Name,Duration(us)\nnew,20\n")

    older_str = str(older_csv)
    newer_str = str(newer_csv)
    os.utime(older_str, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer_str, ns=(2_000_000_000, 2_000_000_000))

    found = prof._find_csv("kernel_details.csv")
    assert found == newer_str
    prof._start_time_ns = 1_500_000_000
    assert prof._find_csv("kernel_details.csv") == newer_str
    prof._start_time_ns = 3_000_000_000
    assert prof._find_csv("kernel_details.csv") is None
    assert prof.result(3).elapsed_us == 0
    assert "historical traces are not reused" in caplog.text


def test_npu_find_csv_returns_none_when_missing(tmp_path, caplog):
    """_find_csv returns None when no matching CSV exists in the tree."""
    from ttk.core_modules.common.profiler import ProfilerConfig
    from ttk.core_modules.common.profiler.torch_profiler import TorchNpuProfiler

    prof = TorchNpuProfiler(None, ProfilerConfig(testcase_name="missing_capture", root_path=str(tmp_path)))
    prof._outdir = str(tmp_path)
    (tmp_path / "trace").mkdir()
    (tmp_path / "trace" / "other.csv").write_text("x\n1\n")

    assert prof._find_csv("kernel_details.csv") is None
    assert "has no new kernel_details.csv" in caplog.text
    assert "historical traces are not reused" in caplog.text


def test_npu_parse_operator_cpu_time_skips_invalid_rows(tmp_path):
    from ttk.core_modules.common.profiler.torch_profiler import TorchNpuProfiler

    path = tmp_path / "operator_details.csv"
    path.write_text("Host Self Duration(us)\n1.5\n2.5\nnot-a-number\n")
    assert TorchNpuProfiler._parse_operator_cpu_time(path) == 4.0


@pytest.mark.parametrize("api_name", ["torch.add", "torch_npu.npu_rms_norm", "custom_op"])
@pytest.mark.parametrize("aclgraph", [False, True])
def test_aclgraph_kernel_accounting(tmp_path, api_name, aclgraph):
    from ttk.core_modules.common.profiler import ProfilerConfig
    from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

    backend = SimpleNamespace(is_npu=lambda: True, profile={"profiler": "builtin"})
    config = ProfilerConfig(testcase_name="capture", root_path=str(tmp_path), aclgraph=aclgraph)
    profiler = get_profiler(api_name, backend, config)

    (Path(profiler._outdir) / "kernel_details.csv").write_text(
        "Step Id,Name,Model ID,Duration(us)\n"
        "1,ForeachCopy,4294967295,40\n"
        "1,MainKernel,48,20\n"
        "1,MainKernel,48,22\n"
        "1,SmallKernel,49,1\n"
        "1,ForeachCopy,0,2\n"
        ",MainKernel,48,24\n"
        "1,Invalid,-1,3\n"
        "1,Unknown,N/A,4\n"
        "1,Malformed,broken,5\n"
        "1,BadDuration,48,not-a-number\n"
    )
    result = profiler.result(2)
    assert result.elapsed_us == (34.5 if aclgraph else 48.5)
    details = {kernel.name: kernel for kernel in result.kernel_details.kernels}
    assert details["MainKernel"].calls == (3 if aclgraph else 2)
    assert details["ForeachCopy"].device_us == (2 if aclgraph else 42)
    if aclgraph:
        assert set(details) == {"MainKernel", "SmallKernel", "ForeachCopy"}
    report = FrameworkApiReturnStructure()
    report.construct("PASS", "PASS", result, mode="aclgraph" if aclgraph else None)
    prefix = "graph_aclgraph_" if aclgraph else "eager_"
    assert getattr(report, prefix + "device_perf_us") == ("34.500" if aclgraph else "48.500")
    assert sum(k["total"] for k in json.loads(getattr(report, prefix + "kernel_details"))) == result.elapsed_us * 2


@pytest.mark.parametrize(
    "contents",
    [
        "Step Id,Name,Duration(us)\n1,Copy,40\n",
        "Step Id,Name,Model ID,Duration(us)\n1,Copy,4294967295,40\n",
        "Step Id,Name,Model ID,Duration(us)\n1,Copy,,40\n",
    ],
)
def test_aclgraph_unknown_capture_does_not_fall_back_to_total(tmp_path, caplog, contents):
    from ttk.core_modules.common.profiler.torch_profiler import TorchNpuProfiler

    path = tmp_path / "kernel_details.csv"
    path.write_text(contents)
    assert TorchNpuProfiler._parse_kernel_details(path, aclgraph=True) == ([], 0.0)
    assert "ACLGraph performance unavailable" in caplog.text


def test_kernel_row_filter_unified_graph_rule(tmp_path):
    """统一过滤规则：eager 丢空 Step Id（warmup 逃逸，不看 Model ID——eager
    算子恒为图外 UINT32_MAX）；graph（torchair/aclgraph 不区分）丢图外
    Model ID、保留空 Step Id（CST 首个计时回放的归因缺失，实测 StepId=''
    ——丢弃会把 # of Calls 计成 run_time-1）。"""
    from ttk.core_modules.common.profiler.torch_profiler import TorchNpuProfiler

    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Step Id,Name,Model ID,Duration(us)\n"
        ",Add,1,1.342\n"  # CST 首调归因缺失
        "1,Add,1,1.083\n"
        "1,Add,1,1.100\n"
        "1,ForeachCopy,4294967295,3.0\n"  # 图外杂活（staging/复位拷贝）
    )
    kernels, total = TorchNpuProfiler._parse_kernel_details(path, graph=True)
    assert [k.calls for k in kernels if k.name == "Add"] == [3]
    assert total == 3.525  # 图外杂活不计

    # eager：空 Step Id 行丢弃（warmup 逃逸），图外 Model ID 不影响
    kernels, total = TorchNpuProfiler._parse_kernel_details(path)
    assert [k.calls for k in kernels if k.name == "Add"] == [2]
    assert [k.calls for k in kernels if k.name == "ForeachCopy"] == [1]
    assert total == 5.183
