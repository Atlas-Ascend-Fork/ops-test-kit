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


# --- A3: TfNpuProfiler msprof 机制失败 → wall-clock 降级 ---


def test_tf_npu_profiler_falls_back_to_wall_clock_on_msprof_init_failure(tmp_path, monkeypatch, caplog):
    """libmsprofiler.so 缺失 / MsProfiler 构造失败不得杀掉用例——降级 wall-clock，
    兑现 TfNpuProfiler 'Eager mode falls back to wall-clock timing' 语义。"""
    import ttk.core_modules.common.profiler.ms_profiler as ms_profiler_mod
    from ttk.core_modules.common.profiler.tf_npu_profiler import TfNpuProfiler

    def _boom(*args, **kwargs):
        raise OSError("libmsprofiler.so: cannot open shared object file")

    monkeypatch.setattr(ms_profiler_mod, "MsProfiler", _boom)
    backend = SimpleNamespace(device_type=lambda: "npu", is_npu=lambda: True)
    prof = TfNpuProfiler(backend, testcase_name="fb", root_path=str(tmp_path))
    with prof:
        pass
    assert prof._prof is None
    result = prof.result(repeat_count=2)
    assert result.kernel_details is None  # wall-clock shape（msprof 路径会带 KernelDetails）
    assert result.elapsed_us >= 0.0
    assert "falling back to wall-clock" in caplog.text


def test_tf_npu_profiler_cleans_up_partial_msprof_init(tmp_path, monkeypatch):
    """step() 中途失败（如 aclprofInit 成功、CreateConfig 失败）时，部分初始化的
    会话必须经 __exit__ 清理（__enter__ 抛异常场景 with 不会调 TfNpuProfiler.__exit__）。"""
    import ttk.core_modules.common.profiler.ms_profiler as ms_profiler_mod
    from ttk.core_modules.common.profiler.tf_npu_profiler import TfNpuProfiler

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

    monkeypatch.setattr(ms_profiler_mod, "MsProfiler", _FailingStepMsProfiler)
    backend = SimpleNamespace(device_type=lambda: "npu", is_npu=lambda: True)
    prof = TfNpuProfiler(backend, testcase_name="partial", root_path=str(tmp_path))
    with prof:
        pass
    assert prof._prof is None  # degraded
    assert created  # MsProfiler was constructed before step() failed
    assert created[0].exited  # partial init cleaned up


# --- B3: op_statistic 空 OP Type 拦截 ---


def test_tf_npu_result_aggregates_op_statistic(tmp_path):
    """op_statistic happy path（解析统一在 ms_profiler.read_op_statistic）：
    Avg 求和为 elapsed（单轮口径），total_device_us = avg × repeat；
    avg<=0 行剔除。"""
    from ttk.core_modules.common.profiler.tf_npu_profiler import TfNpuProfiler

    backend = SimpleNamespace(device_type=lambda: "npu", is_npu=lambda: True)
    prof = TfNpuProfiler(backend, testcase_name="agg", root_path=str(tmp_path))
    (Path(prof._outdir) / "op_statistic_20240101000000.csv").write_text(
        "OP Type,Avg Time(us),Total Time(us),Count,Min Time(us),Max Time(us)\n"
        "Add,2.0,20.0,10,1.0,3.0\n"
        "Mul,1.5,15.0,10,0.5,2.5\n"
        "ZeroAvg,0,0,0,0,0\n"
    )
    res = prof.result(repeat_count=10)
    assert res.elapsed_us == 3.5  # 2.0 + 1.5
    assert res.source == "msprof"
    assert res.wall_us is not None  # wall 侧路结果透出
    assert res.kernel_details.total_device_us == 35.0  # 3.5 × 10
    assert res.kernel_details.total_cpu_us == 0.0
    names = [k.name for k in res.kernel_details.kernels]
    assert names == ["Add", "Mul"]


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
        ("tf.raw_ops.Add", True, None, "TfNpuProfiler"),
        ("tf.raw_ops.Add", False, None, "WallClockProfiler"),
        ("tensorflow.add", True, None, "TfNpuProfiler"),
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
