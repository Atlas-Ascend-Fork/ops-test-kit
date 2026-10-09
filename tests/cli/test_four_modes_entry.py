# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""四模式 CLI 入口测试：kernel / geir / aclnn / e2e 子命令注册、参数→SWITCHES 映射、
run_with_switches 按 test_mode 分派到正确 Instance 类。"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from ttk.cli.aclnn import register_aclnn_command
from ttk.cli.bridge import run_with_switches
from ttk.cli.e2e import register_e2e_command
from ttk.cli.geir import register_geir_command
from ttk.cli.kernel import register_kernel_command


def _build_parser():
    """构造带四个子命令的完整 parser。"""
    parser = argparse.ArgumentParser(prog="ttk")
    sub = parser.add_subparsers(dest="command")
    register_kernel_command(sub)
    register_geir_command(sub)
    register_aclnn_command(sub)
    register_e2e_command(sub)
    return parser


@pytest.fixture
def captured(monkeypatch):
    """拦截四个子命令模块里的 run_with_switches，捕获实际生成的 SWITCHES。"""
    box = {}

    def _record(sw):
        box["sw"] = sw

    for mod in ("ttk.cli.kernel", "ttk.cli.geir", "ttk.cli.aclnn", "ttk.cli.e2e"):
        monkeypatch.setattr(f"{mod}.run_with_switches", _record)
    return box


def _run(argv, captured):
    """解析 argv → 调 handler → 返回捕获的 SWITCHES。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return captured["sw"]


# -- 子命令注册 --------------------------------------------------------------


class TestSubcommandRegistration:
    def test_four_modes_registered(self):
        """四个子命令 kernel/geir/aclnn/e2e 均已注册。"""
        parser = _build_parser()
        sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        names = set(sub_actions[0].choices.keys()) if sub_actions else set()
        assert {"kernel", "geir", "aclnn", "e2e"} <= names

    def test_input_required(self):
        """-i/--input 为必需参数，缺失时 argparse 退出。"""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["kernel"])


# -- kernel 模式 -------------------------------------------------------------


class TestKernelMode:
    def test_sets_test_mode_op(self, captured):
        """kernel 子命令将 test_mode 设为 'op'。"""
        sw = _run(["kernel", "-i", "x.csv"], captured)
        assert sw.test_mode == "op"

    def test_dynamic_shape_disabled_by_flag(self, captured):
        """-d=false 关闭动态 shape 测试。"""
        sw = _run(["kernel", "-i", "x.csv", "-d=false"], captured)
        assert sw.dyn_switches.enabled is False

    def test_binary_release_mode(self, captured):
        """-b=release 启用二进制模式并复用预编译 release kernel。"""
        sw = _run(["kernel", "-i", "x.csv", "-b=release"], captured)
        assert sw.bin_switches.enabled is True
        assert sw.bin_switches.realtime == "release"

    def test_compile_opts_key_value_passthrough(self, captured):
        """--compile-opts KEY=VALUE 可多次指定，累积成 dict。"""
        sw = _run(
            ["kernel", "-i", "x.csv", "--compile-opts", "op_debug_config=oom", "--compile-opts", "k=v"],
            captured,
        )
        assert sw.compile_options == {"op_debug_config": "oom", "k": "v"}


# -- geir 模式 --------------------------------------------------------------


class TestGeirMode:
    def test_sets_test_mode_geir(self, captured):
        """geir 子命令将 test_mode 设为 'geir'。"""
        sw = _run(["geir", "-i", "x.csv"], captured)
        assert sw.test_mode == "geir"

    def test_default_const_on_dynamic_off(self, captured):
        """geir 默认开 const、关 dynamic（与 kernel 相反）。"""
        sw = _run(["geir", "-i", "x.csv"], captured)
        assert sw.cst_switches.enabled is True
        assert sw.dyn_switches.enabled is False

    def test_dynamic_enabled_by_flag(self, captured):
        """-d=true 打开 geir 动态 shape 测试。"""
        sw = _run(["geir", "-i", "x.csv", "-d=true"], captured)
        assert sw.dyn_switches.enabled is True

    def test_binary_release_sets_geir_binary(self, captured):
        """-b=release 将 geir_binary 设为 True。"""
        sw = _run(["geir", "-i", "x.csv", "-b=release"], captured)
        assert sw.geir_binary is True


# -- aclnn 模式 -------------------------------------------------------------


class TestAclnnMode:
    def test_sets_test_mode_aclnn(self, captured):
        """aclnn 子命令将 test_mode 设为 'aclnn'。"""
        sw = _run(["aclnn", "-i", "x.csv"], captured)
        assert sw.test_mode == "aclnn"


# -- e2e 模式 ---------------------------------------------------------------


class TestE2eMode:
    def test_sets_test_mode_framework_api(self, captured):
        """e2e 子命令将 test_mode 设为 'framework-api'。"""
        sw = _run(["e2e", "-i", "x.csv"], captured)
        assert sw.test_mode == "framework-api"

    def test_cpu_flag_sets_force_cpu(self, captured):
        """--cpu 将 force_cpu 设为 True（强制 CPU 后端）。"""
        sw = _run(["e2e", "-i", "x.csv", "--cpu"], captured)
        assert sw.force_cpu is True

    def test_fullgraph_flag(self, captured):
        """--fullgraph 1 打开 torch.compile 全图捕获。"""
        sw = _run(["e2e", "-i", "x.csv", "--fullgraph", "1"], captured)
        assert sw.fullgraph == 1

    def test_aclgraph_flag(self, captured):
        """--aclgraph 启用 aclgraph reduce-overhead 模式。"""
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph"], captured)
        assert sw.aclgraph_enabled is True
        assert sw.aclgraph_online_compile is True

    @pytest.mark.parametrize(
        "binary_args", [["-b"], ["-b=release"], ["-b", "release"], ["--binary"], ["--binary=true"]]
    )
    def test_aclgraph_can_disable_online_compile(self, captured, binary_args):
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph", *binary_args], captured)
        assert sw.aclgraph_enabled is True
        assert sw.aclgraph_online_compile is False

    @pytest.mark.parametrize("value", ["false", "0", "no", "off"])
    def test_binary_false_keeps_online_compile(self, captured, value):
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph", f"-b={value}"], captured)
        assert sw.aclgraph_online_compile is True

    @pytest.mark.parametrize("extra", [[], ["-c"], ["--aclgraph", "--super-kernel"]])
    def test_binary_rejects_invalid_combinations(self, captured, extra):
        with pytest.raises(ValueError, match="--binary"):
            _run(["e2e", "-i", "x.csv", "-b", *extra], captured)

    def test_binary_false_allows_super_kernel(self, captured):
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph", "--super-kernel", "-b=false"], captured)
        assert sw.aclgraph_online_compile is True
        assert sw.super_kernel_enabled is True

    def test_binary_rejects_invalid_value(self, captured):
        with pytest.raises(SystemExit):
            _run(["e2e", "-i", "x.csv", "--aclgraph", "-b=invalid"], captured)


# -- run_with_switches 分派 --------------------------------------------------


class TestRunWithSwitchesDispatch:
    @pytest.mark.parametrize(
        ("test_mode", "instance_path"),
        [
            ("op", "ttk.core_modules.npu.instance_refactor.NpuInstance"),
            ("aclnn", "ttk.core_modules.npu.instance_refactor.NpuInstance"),
            ("geir", "ttk.core_modules.geir.instance.GeirInstance"),
            ("framework-api", "ttk.core_modules.framework_api.instance.FrameworkApiInstance"),
        ],
    )
    @pytest.mark.parametrize("fail_count", [0, 1])
    def test_dispatches_to_correct_instance(self, test_mode, instance_path, monkeypatch, fail_count):
        """run_with_switches 按 test_mode 分派到对应 Instance 类并调用 profile()。"""
        sw = MagicMock()
        sw.test_mode = test_mode
        sw.logging_to_file = False
        sw.input_files = ["x.csv"]

        monkeypatch.setattr("ttk.utilities.set_global_storage", lambda _sw: None)
        monkeypatch.setattr("ttk.core_modules.tbe_logging.default_logging_config", lambda **kw: None)
        monkeypatch.setattr("ttk.utilities.set_process_name", lambda: None)
        monkeypatch.setattr("ttk.utilities.set_thread_name", lambda: None)
        monkeypatch.setattr("ttk.cli.bridge._detect_framework_from_csv", lambda files, sheet=None: "torch")

        with patch(instance_path) as mock_cls:
            mock_cls.return_value.fail_count = fail_count
            if fail_count:
                with pytest.raises(SystemExit) as exc:
                    run_with_switches(sw)
                assert exc.value.code == 1
            else:
                run_with_switches(sw)
            mock_cls.assert_called_once()
            mock_cls.return_value.profile.assert_called_once()
