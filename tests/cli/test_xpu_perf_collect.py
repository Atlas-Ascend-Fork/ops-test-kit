#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""XPU 性能采集测试：--xpu-perf 参数解析、开关映射与前置校验；
normalize_xpu_options 前置校验、_extract_third_party fail-closed、
run 中 XPU gate 的 OPEN/SHUT 回归看护。"""

import argparse
import logging
from types import SimpleNamespace

import pytest

from ttk.core_modules.common.dump import DumpConfig
from ttk.switches import SWITCHES

# -- xpu_perf slot + CLI 参数解析 --------------------------------------------


def test_xpu_perf_slot_default_and_assignable():
    """xpu_perf slot 默认 False，可赋值 True。"""
    sw = SWITCHES()
    assert sw.xpu_perf is False
    sw.xpu_perf = True
    assert sw.xpu_perf is True


def test_dump_config_xpu_default_and_assignable():
    sw = SWITCHES()
    sw.dump_config = DumpConfig()
    sw.dump_config = DumpConfig()
    assert sw.dump_config.dump_xpu is False
    sw.dump_config.dump_xpu = True
    assert sw.dump_config.dump_xpu is True


def test_xpu_perf_flag_registered_on_kernel_parser():
    """--xpu-perf 经共享 add_perf_args 组在 kernel 子命令注册；不传时默认 False。"""
    from ttk.cli.kernel import register_kernel_command

    sub = argparse.ArgumentParser().add_subparsers()
    p = register_kernel_command(sub)
    assert p.parse_args(["-i", "x.csv", "--xpu-perf"]).xpu_perf is True
    assert p.parse_args(["-i", "x.csv"]).xpu_perf is False


def test_xpu_perf_mapped_to_switches():
    """args.xpu_perf 经 args_to_switches 映射到 sw.xpu_perf。"""
    from ttk.cli.bridge import args_to_switches

    sw_on = args_to_switches(SimpleNamespace(input="x.csv", output="o.csv", xpu_perf=True))
    assert sw_on.xpu_perf is True
    sw_off = args_to_switches(SimpleNamespace(input="x.csv", output="o.csv", xpu_perf=False))
    assert sw_off.xpu_perf is False


def test_dump_xpu_option_and_combined_switch_mapping():
    from ttk.cli.bridge import args_to_switches
    from ttk.cli.common import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["-i", "x.csv", "--dump", "xpu"])
    assert args.dump == "xpu"
    assert args_to_switches(args).dump_config.dump_xpu is True

    combined = parser.parse_args(["-i", "x.csv", "--dump", "in,out,golden,xpu"])
    dump_config = args_to_switches(combined).dump_config
    assert dump_config.dump_input
    assert dump_config.dump_output
    assert dump_config.dump_golden
    assert dump_config.dump_xpu

    with pytest.raises(SystemExit):
        parser.parse_args(["-i", "x.csv", "--dump-xpu"])


def test_dump_full_includes_xpu_slot():
    from ttk.cli.bridge import args_to_switches
    from ttk.cli.common import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    dump_config = args_to_switches(parser.parse_args(["-i", "x.csv", "--dump"])).dump_config
    assert dump_config.dump_input
    assert dump_config.dump_output
    assert dump_config.dump_golden
    assert dump_config.dump_xpu is True


# -- normalize_xpu_options 前置校验 -----------------------------------------


def test_normalize_xpu_options_xpu_perf_requires_remote(monkeypatch):
    """--xpu-perf 无远端配置 → 报错（显式性能请求，缺配置 = 配置错误）。"""
    from ttk.cli import common as common_mod

    sw = SWITCHES()
    sw.xpu_perf = True

    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: False)
    with pytest.raises(RuntimeError, match="xpu-perf"):
        common_mod.normalize_xpu_options(sw)

    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: True)
    common_mod.normalize_xpu_options(sw)


def test_normalize_xpu_options_golden_xpu_requires_remote(monkeypatch):
    """--golden-mode XPU 无远端配置 → 启动期报错（而非整轮跑完逐用例 GOLDEN_FAILURE）。"""
    from ttk.cli import common as common_mod

    sw = SWITCHES()
    sw.golden_mode = "XPU"

    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: False)
    with pytest.raises(RuntimeError, match="golden-mode XPU"):
        common_mod.normalize_xpu_options(sw)

    sw.golden_mode = "Enable"
    common_mod.normalize_xpu_options(sw)


def test_normalize_xpu_options_dump_xpu_degrades_without_remote(monkeypatch, caplog):
    """--dump xpu（含 full 隐式开启）无远端配置 → 关闭 + 警告，不报错。

    有数据才落盘：无远端配置就没有 XPU 数据，静默跳过比报错更符合
    in/out/golden 的一致语义。
    """
    from ttk.cli import common as common_mod

    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: False)
    sw = SWITCHES()
    sw.dump_config = DumpConfig()
    sw.dump_config.dump_xpu = True

    with caplog.at_level(logging.WARNING):
        common_mod.normalize_xpu_options(sw)

    assert sw.dump_config.dump_xpu is False
    assert "XPU output dump skipped" in caplog.text

    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: True)
    sw.dump_config.dump_xpu = True
    common_mod.normalize_xpu_options(sw)
    assert sw.dump_config.dump_xpu is True


# -- _extract_third_party fail-closed ---------------------------------------


def test_extract_third_party_fail_closed():
    """_extract_third_party fail-closed：非 PASS / 无 outputs / 无 priority → None。"""
    from ttk.core_modules.kernel import worker as prof

    # 无 results / 无 priority → None
    assert prof._extract_third_party(None, "torch") is None
    assert prof._extract_third_party({}, "torch") is None
    assert prof._extract_third_party({"torch": {"status": "PASS", "outputs": [1]}}, None) is None
    # 非 PASS / 无 outputs key → None
    assert prof._extract_third_party({"torch": {"status": "FAIL", "outputs": [1]}}, "torch") is None
    assert prof._extract_third_party({"torch": {"status": "PASS"}}, "torch") is None
    # happy path：PASS + outputs → 返回 outputs 引用
    outs = [1, 2]
    assert prof._extract_third_party({"torch": {"status": "PASS", "outputs": outs}}, "torch") is outs


# -- run XPU gate 回归看护 ---------------------------------------


# _fake_do_xpu 被调用时抛此异常，短路 run 证明 gate 已开。


class _SpecResolverStub:
    """get_spec_attr(name, attr, path) 键控 monkeypatch 的 SpecResolver 等价物。"""

    def __init__(self, get_attr):
        self._get_attr = get_attr

    def __call__(self, name, pathway, path):
        self._name, self._path = name, path
        return self

    def resolve_tolerance(self):
        return self._get_attr(self._name, "tolerance", self._path)

    def resolve_pre_compare(self):
        return self._get_attr(self._name, "pre_compare", self._path)

    def resolve_compare(self):
        return self._get_attr(self._name, "compare", self._path)


class _XpuCalled(Exception):
    pass


def test_profile_process_gate_open_and_shut(monkeypatch):
    """run 的 XPU gate 回归看护。

    gate = _xpu_mode(sw, need_3party_outputs)；若 gate != 0 则调 _do_xpu_profiling。
    本测试隔离 gate 上游所有依赖（parse/gen_input/gen_output/resolve 等），
    使 need_3party=False，gate 完全由 xpu_perf 决定：

    - gate OPEN（xpu_perf=True）：_do_xpu_profiling 被调用 1 次，短路抛 _XpuCalled。
    - gate SHUT（xpu_perf=False）：_do_xpu_profiling 调用 0 次，run
      越过 gate 到 __gen_workspaces，因 context 无 dyn_compile_result 抛 AttributeError。

    若 gate 被回退为 is_remote_configured()（无条件开启），SHUT 分支会失败。
    """
    from ttk.core_modules.kernel import worker as prof

    sw = SWITCHES()

    context = SimpleNamespace(
        testcase_name="t_gate",
        op_name="DummyOp",
        is_valid=True,
        compile_failed=lambda: False,
        input_bytes=0,
        output_bytes=0,
        flat_precision_tolerances=(),
        flat_absolute_precision=(),
        flat_output_dtypes=(),
        flat_input_dtypes=(),
    )

    # 隔离 gate 上游：所有 parse/gen 步骤为 no-op
    monkeypatch.setattr(prof, "get_global_storage", lambda: sw)
    monkeypatch.setattr(
        prof,
        "process_context",
        lambda: SimpleNamespace(
            change_name=lambda _name: None,
            notify_status=lambda _s: None,
        ),
    )
    monkeypatch.setattr(prof, "__apply_force_overrides", lambda _ctx: None)
    monkeypatch.setattr(prof, "__parse_dynamic_tiling_data", lambda _ctx: None)
    monkeypatch.setattr(prof, "__parse_binary_tiling_data", lambda _ctx: None)
    monkeypatch.setattr(prof, "gen_input", lambda _ctx: None)
    monkeypatch.setattr(prof, "gen_output", lambda _ctx: None)
    # resolve 路径：tolerance None → resolve 返回 [] → need_3party=False
    monkeypatch.setattr(prof, "SpecResolver", _SpecResolverStub(lambda *_a, **_k: None))
    import ttk.core_modules.common.comparison.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "resolve_tolerance", lambda *_a, **_k: [])
    # clear_error_manager 在 run 内 lazy import
    import ttk.core_modules.kernel.error_cleaner as ec

    monkeypatch.setattr(ec, "clear_error_manager", lambda: None)

    # 记录 _do_xpu_profiling 调用；OPEN 时抛异常短路
    xpu_calls = []

    def _fake_do_xpu(_ctx, _mode):
        xpu_calls.append((_ctx, _mode))
        raise _XpuCalled()

    monkeypatch.setattr(prof, "_do_xpu_profiling", _fake_do_xpu)

    # --- gate OPEN：xpu_perf=True → _xpu_mode 返回 PERF → 调 _do_xpu_profiling ---
    xpu_calls.clear()
    sw.xpu_perf = True
    with pytest.raises(_XpuCalled):
        prof.run(context, {}, {}, 0)
    assert len(xpu_calls) == 1, "gate open must invoke _do_xpu_profiling"

    # --- gate SHUT：xpu_perf=False, need_3party=False → _xpu_mode 返回 0 ---
    # _do_xpu_profiling 不应被调用；run 越过 gate 到 __gen_workspaces，
    # 因 context 无 dyn_compile_result 而抛 AttributeError（证明 gate 未开）。
    xpu_calls.clear()
    sw.xpu_perf = False
    with pytest.raises(AttributeError, match="dyn_compile_result"):
        prof.run(context, {}, {}, 0)
    assert len(xpu_calls) == 0, "gate shut must NOT call _do_xpu_profiling"
