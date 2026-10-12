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
from unittest.mock import MagicMock

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

    def test_fullgraph_flag(self, captured):
        """--fullgraph 1 打开 torch.compile 全图捕获。"""
        sw = _run(["e2e", "-i", "x.csv", "--fullgraph", "1"], captured)
        assert sw.fullgraph == 1

    def test_aclgraph_flag(self, captured):
        """--aclgraph 启用 aclgraph reduce-overhead 模式（算子编译默认二进制复用）。"""
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph"], captured)
        assert sw.aclgraph_enabled is True
        assert sw.e2e_binary is True

    def test_binary_default_on_all_legs(self, captured):
        """-b 默认二进制：不依赖 --aclgraph，eager/-c/-d/aclgraph 三种模式全生效。"""
        sw = _run(["e2e", "-i", "x.csv"], captured)
        assert sw.e2e_binary is True
        sw = _run(["e2e", "-i", "x.csv", "-c"], captured)
        assert sw.e2e_binary is True
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph"], captured)
        assert sw.e2e_binary is True

    @pytest.mark.parametrize(
        "binary_args", [["-b"], ["-b=release"], ["-b", "release"], ["--binary"], ["--binary=true"]]
    )
    def test_binary_explicit_forms_enable_binary(self, captured, binary_args):
        """显式 -b（含 release 等价写法）与默认等价，无需 --aclgraph。"""
        sw = _run(["e2e", "-i", "x.csv", *binary_args], captured)
        assert sw.e2e_binary is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off"])
    def test_binary_false_restores_online_compile(self, captured, value):
        """-b=false 恢复在线编译。"""
        sw = _run(["e2e", "-i", "x.csv", f"-b={value}"], captured)
        assert sw.e2e_binary is False

    def test_binary_rejects_super_kernel(self, captured):
        """显式 -b 与 --super-kernel 同开报错（SuperKernel 要求在线编译）。"""
        with pytest.raises(ValueError, match="--binary"):
            _run(["e2e", "-i", "x.csv", "-b", "--aclgraph", "--super-kernel"], captured)

    def test_super_kernel_forces_online_compile(self, captured):
        """隐式默认二进制 + --super-kernel：自动回退在线编译（不破坏既有裸用法）。"""
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph", "--super-kernel"], captured)
        assert sw.e2e_binary is False
        assert sw.super_kernel_enabled is True

    def test_binary_false_allows_super_kernel(self, captured):
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph", "--super-kernel", "-b=false"], captured)
        assert sw.e2e_binary is False
        assert sw.super_kernel_enabled is True

    def test_binary_lenient_truthy_token(self, captured):
        """-b 为布尔风格参数（对齐 -c/-d/--eager）：非 falsy 串一律视为 True。"""
        sw = _run(["e2e", "-i", "x.csv", "--aclgraph", "-b=invalid"], captured)
        assert sw.e2e_binary is True

    def test_binary_forced_off_under_npusim(self, captured):
        """npusim 仿真要求算子在线编译：默认/显式二进制均强制关闭。"""
        sw = _run(["e2e", "-i", "x.csv", "--backend", "npusim"], captured)
        assert sw.e2e_binary is False
        sw = _run(["e2e", "-i", "x.csv", "--backend", "npusim", "-b"], captured)
        assert sw.e2e_binary is False


# -- run_with_switches 分派 --------------------------------------------------


class TestRunWithSwitchesDispatch:
    @pytest.mark.parametrize(
        "test_mode",
        ["op", "aclnn", "geir", "framework-api"],
    )
    @pytest.mark.parametrize("fail_count", [0, 1])
    def test_dispatches_to_correct_instance(self, test_mode, monkeypatch, fail_count):
        """run_with_switches 经注册表按 test_mode 分派（get_instance_by_name 惰性单例）
        并调用 profile()；失败计数非零时以退出码 1 退出。"""
        from ttk.core_modules.common import instance as instance_mod

        mock_ins = MagicMock()
        mock_ins.fail_count = fail_count
        monkeypatch.setattr(instance_mod, "get_instance_by_name", lambda name: mock_ins)

        sw = MagicMock()
        sw.test_mode = test_mode
        sw.logging_to_file = False
        sw.input_files = ["x.csv"]

        monkeypatch.setattr("ttk.switches.set_global_storage", lambda _sw: None)
        monkeypatch.setattr("ttk.core_modules.common.logging.default_logging_config", lambda **kw: None)
        monkeypatch.setattr("ttk.utilities.set_process_name", lambda: None)
        monkeypatch.setattr("ttk.utilities.set_thread_name", lambda: None)
        monkeypatch.setattr("ttk.cli.bridge._detect_framework_from_csv", lambda files, sheet=None: "torch")

        if fail_count:
            with pytest.raises(SystemExit) as exc:
                run_with_switches(sw)
            assert exc.value.code == 1
        else:
            run_with_switches(sw)
        mock_ins.run.assert_called_once()


def test_cli_top_level_error_is_single_line(capsys, monkeypatch, tmp_path):
    """bridge/handler 的校验 ValueError → 单行 ttk: error（非 traceback）；TTK_DEBUG=1 保留全栈。"""
    import sys

    monkeypatch.chdir(tmp_path)  # 隔离用户级 ttk.conf.yaml
    monkeypatch.setattr(sys, "argv", ["ttk", "kernel", "-i", "x.csv", "--dump", "in,foo"])
    from ttk.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("ttk: error:")
    assert "unknown token" in err
    assert "Traceback" not in err

    monkeypatch.setenv("TTK_DEBUG", "1")
    with pytest.raises(ValueError, match="unknown token"):
        main()
