#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit tests for the E2E (framework-api) NPUSim backend support.

These tests never run cannsim / camodel — they mock ``ASCEND_TOOLKIT_HOME`` and
the camodel directory resolution, mirroring the existing simulator backend
tests (``test_simulator_backend.py``).
"""

import os
from types import SimpleNamespace

import pytest

from ttk.core_modules.common.simulator import config as sim_config
from ttk.core_modules.framework_api.instance import FrameworkApiInstance
from ttk.switches import SWITCHES, get_global_storage, set_global_storage


class TestResolveCamodelLibDir:
    """resolve_camodel_lib_dir 从 toolkit 解析 camodel 库目录，覆盖各类缺失/异常。"""

    def test_returns_camodel_path(self, tmp_path, monkeypatch):
        """toolkit 下存在 camodel 目录时返回该路径。"""
        toolkit = tmp_path / "toolkit"
        camodel = toolkit / "tools" / "simulator" / "Ascend950PR_9599" / "camodel"
        camodel.mkdir(parents=True)
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
        monkeypatch.setattr(sim_config, "_cannsim_model_name", lambda soc: "Ascend950PR_9599")

        assert sim_config.resolve_camodel_lib_dir("Ascend950") == camodel

    def test_missing_toolkit_home_raises(self, monkeypatch):
        """缺少 ASCEND_TOOLKIT_HOME 时报错。"""
        monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
        with pytest.raises(RuntimeError, match="ASCEND_TOOLKIT_HOME"):
            sim_config.resolve_camodel_lib_dir("Ascend950")

    def test_unknown_soc_raises(self, tmp_path, monkeypatch):
        """未知 SoC 无 camodel 模型名时报错。"""
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path))
        monkeypatch.setattr(sim_config, "_cannsim_model_name", lambda soc: "")
        with pytest.raises(RuntimeError, match="Unknown NPUSim SoC"):
            sim_config.resolve_camodel_lib_dir("Ascend999")

    def test_missing_camodel_dir_raises(self, tmp_path, monkeypatch):
        """camodel 目录不存在时报错。"""
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path))
        monkeypatch.setattr(sim_config, "_cannsim_model_name", lambda soc: "Ascend950PR_9599")
        with pytest.raises(RuntimeError, match="camodel directory not found"):
            sim_config.resolve_camodel_lib_dir("Ascend950")

    def test_cannsim_model_name_falls_back_to_platform_map(self, tmp_path, monkeypatch):
        """When the cannsim package is unavailable, SIM_PLATFORM_BY_SOC is used."""
        toolkit = tmp_path / "toolkit"
        camodel = toolkit / "tools" / "simulator" / "Ascend950PR_9589" / "camodel"
        camodel.mkdir(parents=True)
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
        monkeypatch.setattr(sim_config, "_cannsim_model_name", lambda soc: "")
        assert sim_config.resolve_camodel_lib_dir("Ascend950") == camodel


class _FakeBackend:
    """Minimal backend stand-in for FrameworkApiInstance.__init__."""

    def __init__(self):
        self._dev = True

    def has_device(self):
        return self._dev

    def device_type(self):
        return "npu"


class TestFrameworkApiInstanceNpusim:
    """FrameworkApiInstance 在 npusim 下注入 camodel 环境并拒绝 --cpu。"""

    def _switches(self, tmp_path):
        """构造一份 npusim SWITCHES。"""
        sw = SWITCHES()
        sw.backend = "npusim"
        sw.sim_soc_version = "Ascend950"
        sw.sim_output_dir = str(tmp_path / "sim_output")
        sw.force_cpu = False
        return sw

    def test_injects_camodel_env(self, tmp_path, monkeypatch):
        """npusim 实例化时向 LD_LIBRARY_PATH 注入 camodel 目录。"""
        camodel = tmp_path / "camodel"
        camodel.mkdir()
        # instance._inject_camodel_env imports resolve_camodel_lib_dir from the
        # config module lazily, so patch it at its definition site.
        monkeypatch.setattr(
            "ttk.core_modules.common.simulator.config.resolve_camodel_lib_dir",
            lambda soc: camodel,
        )
        monkeypatch.setattr(
            "ttk.core_modules.framework_api.instance.get_backend",
            lambda force_cpu, framework="torch", profile=None: _FakeBackend(),
        )
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        sw = self._switches(tmp_path)

        original = get_global_storage()
        try:
            set_global_storage(sw)
            FrameworkApiInstance()
        finally:
            set_global_storage(original)

        assert str(camodel) in os.environ["LD_LIBRARY_PATH"].split(":")
        assert os.environ["CAMODEL_LOG_PATH"] == str(tmp_path / "sim_output" / "camodel_log")

    def test_force_cpu_rejected(self, tmp_path, monkeypatch):
        """npusim 下 --cpu 被拒绝。"""
        monkeypatch.setattr(
            "ttk.core_modules.common.simulator.config.resolve_camodel_lib_dir",
            lambda soc: tmp_path / "camodel",
        )
        sw = self._switches(tmp_path)
        sw.force_cpu = True
        with pytest.raises(ValueError, match="--cpu"):
            FrameworkApiInstance._inject_camodel_env(sw)


class TestFrameworkApiInstanceOnnxGates:
    """onnx 域专属开关 fail-fast：npusim / 三方 XPU（golden-XPU / xpu-perf / cross_check）。

    onnx 用例声明 ``--backend npusim`` 时有卡环境会静默真机执行（DSMI 仍见
    真卡）、无卡环境逐用例报误导性的 ONNX_CPU_EXECUTION_UNSUPPORTED——两类
    都必须在启动期拒绝，且拒绝须先于 camodel 环境注入（不污染
    LD_LIBRARY_PATH）。三方 XPU 远端按框架 dotted path 分发，无 onnx 通道。
    """

    def _switches(self, tmp_path):
        """构造一份 onnx 域 SWITCHES（backend 默认 npu）。"""
        sw = SWITCHES()
        sw.framework = "onnx"
        sw.backend = "npu"
        sw.sim_output_dir = str(tmp_path / "sim_output")
        sw.force_cpu = False
        return sw

    def test_npusim_onnx_rejected_and_torch_injects(self, tmp_path, monkeypatch):
        """onnx + npusim 启动期拒绝且 camodel 环境零注入；torch 域 npusim 不受
        onnx 门禁影响（camodel 照常注入）——同一初始化路径的对比钉住。"""
        injected = []
        monkeypatch.setattr(
            FrameworkApiInstance,
            "_inject_camodel_env",
            staticmethod(injected.append),
        )

        sw = self._switches(tmp_path)
        sw.backend = "npusim"
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        original = get_global_storage()
        try:
            set_global_storage(sw)
            with pytest.raises(ValueError, match="ONNX_NPUSIM_UNSUPPORTED"):
                FrameworkApiInstance()
        finally:
            set_global_storage(original)
        assert injected == []  # 拒绝先于 camodel 注入
        assert "LD_LIBRARY_PATH" not in os.environ  # 环境零污染

        monkeypatch.setattr(
            "ttk.core_modules.framework_api.instance.get_backend",
            lambda force_cpu, framework="torch", profile=None: _FakeBackend(),
        )
        sw = SWITCHES()
        sw.backend = "npusim"
        sw.sim_soc_version = "Ascend950"
        sw.sim_output_dir = str(tmp_path / "sim_output")
        sw.force_cpu = False
        original = get_global_storage()
        try:
            set_global_storage(sw)
            FrameworkApiInstance()  # 不抛：torch 域 npusim 语义不变
        finally:
            set_global_storage(original)
        assert len(injected) == 1

    @pytest.mark.parametrize(
        ("field", "value", "reason"),
        [
            ("golden_mode", "XPU", "--golden-mode XPU"),
            ("xpu_perf", True, "--xpu-perf"),
            ("compare_method", "cross_check", "--compare cross_check"),
        ],
    )
    def test_xpu_switches_rejected(self, tmp_path, field, value, reason):
        """三方 XPU 开关逐项拒绝，错误信息带开关名与 ONNX_XPU_UNSUPPORTED 码。"""
        sw = self._switches(tmp_path)
        setattr(sw, field, value)
        with pytest.raises(ValueError, match="ONNX_XPU_UNSUPPORTED") as excinfo:
            FrameworkApiInstance._reject_onnx_exclusive_options(sw)
        assert reason in str(excinfo.value)

    def test_xpu_reasons_listed_together(self, tmp_path):
        """多个 XPU 开关同时开启时错误信息逐一列出。"""
        sw = self._switches(tmp_path)
        sw.golden_mode = "XPU"
        sw.xpu_perf = True
        sw.compare_method = "cross_check"
        with pytest.raises(ValueError, match="ONNX_XPU_UNSUPPORTED") as excinfo:
            FrameworkApiInstance._reject_onnx_exclusive_options(sw)
        message = str(excinfo.value)
        for reason in ("--golden-mode XPU", "--xpu-perf", "--compare cross_check"):
            assert reason in message

    def test_dump_xpu_degraded_and_clean_switches_pass(self, tmp_path, caplog):
        """非致命路径：--dump xpu 告警降级（对齐 normalize_xpu_options 范式）；
        无冲突开关的 onnx 配置照常放行（门禁无误伤）。"""
        sw = self._switches(tmp_path)
        sw.dump_config = SimpleNamespace(dump_xpu=True)
        with caplog.at_level("WARNING"):
            FrameworkApiInstance._reject_onnx_exclusive_options(sw)
        assert sw.dump_config.dump_xpu is False
        assert any("dump xpu" in rec.message for rec in caplog.records)

        sw = self._switches(tmp_path)
        sw.dump_config = SimpleNamespace(dump_xpu=False)
        FrameworkApiInstance._reject_onnx_exclusive_options(sw)  # 不抛即通过


class TestFrameworkApiInstanceCpuFallback:
    """torch_npu/npu_device 缺失探测回落 CPU 时，实例把运行定格为 CPU（force_cpu）。"""

    def _run_instance(self, monkeypatch, backend):
        monkeypatch.setattr(
            "ttk.core_modules.framework_api.instance.get_backend",
            lambda force_cpu, framework="torch": backend,
        )
        sw = SWITCHES()
        sw.backend = "npu"
        sw.force_cpu = False
        original = get_global_storage()
        try:
            set_global_storage(sw)
            FrameworkApiInstance()
        finally:
            set_global_storage(original)
        return sw

    def test_fallback_pins_force_cpu(self, monkeypatch):
        """CPU 回落（has_device=False）时置 force_cpu，健康过滤随之跳过。"""
        backend = _FakeBackend()
        backend._dev = False
        sw = self._run_instance(monkeypatch, backend)
        assert sw.force_cpu is True
        assert sw.proc_no_reuse is True

    def test_npu_backend_keeps_switches(self, monkeypatch):
        """NPU 后端不动 force_cpu/proc_no_reuse。"""
        sw = self._run_instance(monkeypatch, _FakeBackend())
        assert sw.force_cpu is False
        assert sw.proc_no_reuse is False


class TestE2eCliSimArgs:
    """E2E CLI 解析器暴露 --backend 并默认 npu。"""

    def test_e2e_parser_exposes_backend(self):
        """--backend npusim 可被解析。"""
        import argparse

        from ttk.cli.e2e import register_e2e_command

        parser = argparse.ArgumentParser()
        register_e2e_command(parser.add_subparsers())
        args = parser.parse_args(["e2e", "-i", "cases.csv", "--backend", "npusim"])
        assert args.backend == "npusim"

    def test_e2e_parser_default_backend_npu(self):
        """未指定 --backend 时默认为 npu。"""
        import argparse

        from ttk.cli.e2e import register_e2e_command

        parser = argparse.ArgumentParser()
        register_e2e_command(parser.add_subparsers())
        args = parser.parse_args(["e2e", "-i", "cases.csv"])
        assert args.backend == "npu"


class TestCollectSimReport:
    """E2E npusim instr.bin collection + --sim-report trigger."""

    @staticmethod
    def _switches(tmp_path):
        """构造一份带 sim_report 的 npusim SWITCHES。"""
        sw = SWITCHES()
        sw.backend = "npusim"
        sw.sim_soc_version = "Ascend950"
        sw.sim_output_dir = str(tmp_path / "sim_output")
        sw.sim_report = True
        sw.root_path = str(tmp_path)  # worker cwd
        return sw

    @staticmethod
    def _testcase():
        """构造一个最小测试用例桩。"""
        return SimpleNamespace(testcase_name="add_f32_01")

    def test_move_instr_and_report(self, tmp_path, monkeypatch):
        """npusim 下 instr.bin 移入用例目录并触发 sim-report。"""
        from ttk.core_modules.common.simulator.case_writer import case_dir
        from ttk.core_modules.framework_api import worker as e2e_profiling

        sw = self._switches(tmp_path)
        (tmp_path / "instr.bin").write_bytes(b"\x00" * 64)
        reported = []
        monkeypatch.setattr(
            "ttk.core_modules.common.simulator.report.maybe_generate_sim_report",
            lambda sw_, cp, er: reported.append(cp),
        )

        e2e_profiling._collect_sim_report(self._testcase(), sw)

        case_path = case_dir(sw, "add_f32_01")
        assert (case_path / "instr.bin").is_file()  # moved into case dir
        assert not (tmp_path / "instr.bin").exists()  # removed from worker cwd
        assert reported == [case_path]  # report triggered

    def test_collect_without_report(self, tmp_path, monkeypatch):
        """sim_report=False 时仍收集 instr.bin 但不生成报告。"""
        from ttk.core_modules.common.simulator.case_writer import case_dir
        from ttk.core_modules.framework_api import worker as e2e_profiling

        sw = self._switches(tmp_path)
        sw.sim_report = False
        (tmp_path / "instr.bin").write_bytes(b"\x00" * 64)
        reported = []
        monkeypatch.setattr(
            "ttk.core_modules.common.simulator.report.maybe_generate_sim_report",
            lambda sw_, cp, er: reported.append(cp),
        )

        e2e_profiling._collect_sim_report(self._testcase(), sw)

        case_path = case_dir(sw, "add_f32_01")
        assert (case_path / "instr.bin").is_file()  # still collected
        assert not reported  # but no report

    def test_skip_non_npusim(self, tmp_path, monkeypatch):
        """非 npusim 后端时 instr.bin 保持不动且不触发报告。"""
        from ttk.core_modules.framework_api import worker as e2e_profiling

        sw = self._switches(tmp_path)
        sw.backend = "npu"
        (tmp_path / "instr.bin").write_bytes(b"\x00" * 64)
        reported = []
        monkeypatch.setattr(
            "ttk.core_modules.common.simulator.report.maybe_generate_sim_report",
            lambda sw_, cp, er: reported.append(cp),
        )

        e2e_profiling._collect_sim_report(self._testcase(), sw)

        assert (tmp_path / "instr.bin").exists()  # untouched
        assert not reported

    def test_skip_missing_or_empty_instr(self, tmp_path, monkeypatch):
        """instr.bin 缺失或为空时不触发报告。"""
        from ttk.core_modules.framework_api import worker as e2e_profiling

        sw = self._switches(tmp_path)
        reported = []
        monkeypatch.setattr(
            "ttk.core_modules.common.simulator.report.maybe_generate_sim_report",
            lambda sw_, cp, er: reported.append(cp),
        )
        # no instr.bin at all
        e2e_profiling._collect_sim_report(self._testcase(), sw)
        # empty instr.bin
        (tmp_path / "instr.bin").write_bytes(b"")
        e2e_profiling._collect_sim_report(self._testcase(), sw)

        assert not reported
        assert (tmp_path / "instr.bin").exists()  # empty file left in place
