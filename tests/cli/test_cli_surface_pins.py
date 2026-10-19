# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""CLI 参数面 pin 测试。

目的：把**易被增删/合并/删除的 CLI 表面**钉住——变更时必须
**有意识地**更新本文件（原卡号标注已去除），防止无意识漂移。钉"参数存在性 +
真实默认值 + choices 值域 + bridge 落点 + 组内注册序（按 dest 字母序）"，
不钉 help 文案。
"""

import argparse

import pytest

from ttk.cli import build_parser
from ttk.cli.aclnn import register_aclnn_command
from ttk.cli.bridge import args_to_switches
from ttk.cli.e2e import register_e2e_command
from ttk.cli.geir import register_geir_command
from ttk.cli.kernel import register_kernel_command


def _build_parser():
    parser = argparse.ArgumentParser(prog="ttk")
    sub = parser.add_subparsers(dest="command")
    register_kernel_command(sub)
    register_geir_command(sub)
    register_aclnn_command(sub)
    register_e2e_command(sub)
    return parser


def _parse(argv):
    return _build_parser().parse_args(argv)


def _switches(argv):
    from ttk.cli.bridge import apply_kernel_args
    from ttk.cli.sim_args import apply_sim_args

    args = _parse(argv)
    sw = args_to_switches(args)
    apply_sim_args(sw, args)
    if args.command == "kernel":
        apply_kernel_args(sw, args)
    return sw


# ── 历史删除/合并对象：现状必须存在且语义如钉 ─────────────────────────────


class TestLegacyCliSurface:
    """参数变更落地后的翻转断言（原 pins 按设计更新）。"""

    def test_removed_flags_rejected(self):
        # 历史翻转：--single-log（已默认化）/
        # --title/--csv-preserve（低频删除）——--no-memory-check 已恢复（宽松闸门）
        for flag in ("--single-log", "--title", "--csv-preserve"):
            with pytest.raises(SystemExit):
                _parse(["kernel", "-i", "cases.csv", flag])

    def test_summary_print_restored(self):
        # --print 恢复：默认 true，--print false 关闭周期性摘要（重构期一度无条件打印）
        sw = _switches(["kernel", "-i", "cases.csv"])
        assert sw.summary_print is True
        sw = _switches(["kernel", "-i", "cases.csv", "--print", "false"])
        assert sw.summary_print is False

    def test_no_memory_check_restored(self):
        # --no-memory-check 恢复（实施修正：多进程拉起防 OOM 直杀；闸门宽松化）
        sw = _switches(["kernel", "-i", "cases.csv", "--no-memory-check"])
        assert sw.no_memory_check is True
        sw = _switches(["kernel", "-i", "cases.csv"])
        assert sw.no_memory_check is False  # 默认闸门开启

    def test_devices_unified(self):
        # --dev 数量 / --devices 纯 ID 表达式（数量回归 --dev/--device，
        # --devices 纯数字数量分支已废除）
        sw = _switches(["kernel", "-i", "cases.csv", "--dev", "2"])
        assert sw.device_count == 2
        assert sw.device_blacklist == ()
        sw = _switches(["kernel", "-i", "cases.csv", "--device", "3"])
        assert sw.device_count == 3
        sw = _switches(["kernel", "-i", "cases.csv", "--devices", "2"])
        assert sw.device_count == 3
        assert sw.device_blacklist == (0, 1)
        sw = _switches(["kernel", "-i", "cases.csv", "--devices", "0,1,2"])
        assert sw.device_count == 3
        assert sw.device_blacklist == ()
        sw = _switches(["kernel", "-i", "cases.csv", "--devices", "0-7,^3"])
        assert sw.device_blacklist == (3,)
        sw = _switches(["kernel", "-i", "cases.csv", "--devices", "^4"])
        assert sw.device_blacklist == (4,)
        # 组合：--dev 定池上限，--devices 正选池内取交集 / ^ 剔除
        sw = _switches(["kernel", "-i", "cases.csv", "--dev", "4", "--devices", "0,3"])
        assert sw.device_count == 4
        assert sw.device_blacklist == (1, 2)
        sw = _switches(["kernel", "-i", "cases.csv", "--dev", "4", "--devices", "^2"])
        assert sw.device_count == 4
        assert sw.device_blacklist == (2,)
        # --device-whitelist = --devices 的旧拼写（同 dest 同表达式，纯兼容别名）
        sw = _switches(["kernel", "-i", "cases.csv", "--dev", "4", "--device-whitelist", "0,3"])
        assert sw.device_count == 4
        assert sw.device_blacklist == (1, 2)
        # 旧拼写同样吃完整表达式（功能一致）
        sw = _switches(["kernel", "-i", "cases.csv", "--device-whitelist", "0-7,^3"])
        assert sw.device_blacklist == (3,)
        # blacklist 不保留（剔除统一走 --devices '^'）
        with pytest.raises(SystemExit):
            _parse(["kernel", "-i", "cases.csv", "--device-blacklist", "0"])

    def test_e2e_cpu_merged_into_backend(self):
        # 历史翻转：--cpu 并入 --backend cpu（e2e 专属删除）
        with pytest.raises(SystemExit):
            _parse(["e2e", "-i", "cases.csv", "--cpu"])
        sw = _switches(["e2e", "-i", "cases.csv", "--backend", "cpu"])
        assert sw.force_cpu is True

    def test_backend_cpu_is_e2e_only(self):
        # cpu 仅 e2e 提供（kernel/aclnn 主执行需设备或 npusim）
        for mode in ("kernel", "aclnn"):
            with pytest.raises(SystemExit):
                _parse([mode, "-i", "cases.csv", "--backend", "cpu"])
        with pytest.raises(SystemExit):
            _parse(["geir", "-i", "cases.csv", "--backend", "npusim"])

    def test_backend_xpu_rejected(self):
        # xpu 本地三方执行已下线；三方采集仅 remote（endpoints 配置）
        for mode in ("kernel", "aclnn", "e2e", "geir"):
            with pytest.raises(SystemExit):
                _parse([mode, "-i", "cases.csv", "--backend", "xpu"])

    def test_geir_has_no_sim_args(self):
        # geir 无 npusim 形态——--sim-* 不注册（避免死参数面）
        for flag in ("--sim-soc", "--sim-report", "--sim-cores"):
            with pytest.raises(SystemExit):
                _parse(["geir", "-i", "cases.csv", flag])

    def test_comma_separated_args_tolerate_spaces(self):
        # 逗号传参带空格：strip + 滤空段（裸 split 会留 ' b' 导致静默失配）
        sw = _switches(
            [
                "kernel",
                "-i",
                "cases.csv",
                "-t",
                "case1, case2",
                "--op",
                "add, sub",
                "--no-op",
                "mul, div",
                "--rerun",
                "a, B",
                "--priority",
                "1, 3",
                "--dump",
                "in, out",
            ]
        )
        assert sw.selected_testcases == ["case1", "case2"]
        assert sw.selected_operators == ("add", "sub")
        assert sw.excluded_operators == ("mul", "div")
        assert sw.rerun_targets == ["a", "b"]
        assert sw.priorities == ((1, 1), (3, 3))
        assert (sw.dump_config.dump_input, sw.dump_config.dump_output) == (True, True)

    def test_dump_rejects_unknown_token(self):
        # --dump 未知 token fail-fast（原先静默吞掉）
        with pytest.raises(ValueError, match="unknown token"):
            _switches(["kernel", "-i", "cases.csv", "--dump", "in, foo"])

    def test_manual_data_dirs_on_all_four_modes(self):
        # 四通路均暴露 manual-data 工作流：no-prof 单遍落盘
        # + load 逐模式恢复
        for mode in ("kernel", "aclnn", "e2e", "geir"):
            assert _parse([mode, "-i", "cases.csv", "--manual-data-dirs", "dir"]).manual_data_dirs == ["dir"]
            assert _parse([mode, "-i", "cases.csv", "--no-prof"]).no_prof is True


class TestExtendedChoicesCurrent:
    def test_compare_choices_contain_mixed(self):
        # 3675093 后 --compare 含 mixed 别名；argparse dest=compare，bridge 落 compare_method
        for token in ("close", "stat_rel_err", "mixed", "cosine", "binary", "requant", "cross_check"):
            assert _parse(["kernel", "-i", "cases.csv", "--compare", token]).compare == token
            assert _switches(["kernel", "-i", "cases.csv", "--compare", token]).compare_method == token

    def test_golden_mode_choices_include_xpu(self):
        # XPU 态已加入（GPU 输出作 golden）
        for mode in ("Enable", "Disable", "Promote", "XPU"):
            assert _parse(["kernel", "-i", "cases.csv", "--golden-mode", mode]).golden_mode == mode

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("enable", "Enable"), ("DISABLE", "Disable"), ("promote", "Promote"), ("xpU", "XPU")],
    )
    def test_golden_mode_cli_is_case_insensitive(self, raw, expected):
        assert _parse(["kernel", "-i", "cases.csv", "--golden-mode", raw]).golden_mode == expected

    def test_golden_mode_cli_rejects_unknown_value(self):
        with pytest.raises(SystemExit):
            _parse(["kernel", "-i", "cases.csv", "--golden-mode", "xxx"])

    @pytest.mark.parametrize(
        ("override", "expected"),
        [("xpu", "XPU"), ("DISABLE", "Disable"), ("Promote", "Promote"), ("bogus", "Enable")],
    )
    def test_golden_mode_resolver_is_case_insensitive(self, override, expected):
        # 用例级 golden_mode_override 同样大小写不敏感；未知值告警后回 Enable
        from types import SimpleNamespace

        from ttk.core_modules.common.invoke import golden_mode

        class _Case:
            golden_mode_override = None

        case = _Case()
        case.golden_mode_override = override
        assert golden_mode(case, SimpleNamespace(golden_mode="Enable")) == expected

    def test_dump_tokens_and_xpu_slot(self):
        # DumpConfig bool 位 + xpu 位；full 四类全开（无远端配置时 CLI 层降级关闭 xpu）
        sw = _switches(["kernel", "-i", "cases.csv", "--dump", "in,out,golden"])
        assert (sw.dump_config.dump_input, sw.dump_config.dump_output, sw.dump_config.dump_golden) == (True, True, True)
        assert sw.dump_config.dump_xpu is False
        sw = _switches(["kernel", "-i", "cases.csv", "--dump", "xpu"])
        assert (sw.dump_config.dump_xpu, sw.dump_config.dump_input) == (True, False)
        # --dump 缺省 const="full" = 四类全开（含 xpu；无远端配置时自动跳过）
        sw = _switches(["kernel", "-i", "cases.csv", "--dump"])
        assert (sw.dump_config.dump_input, sw.dump_config.dump_output, sw.dump_config.dump_golden) == (True, True, True)
        assert sw.dump_config.dump_xpu is True


# ── 稳定默认值锚点（argparse 层多为 None，默认值在 bridge 落位）────────────


class TestStableDefaults:
    @pytest.mark.parametrize("mode", ["kernel", "aclnn", "e2e", "geir"])
    def test_common_defaults_via_bridge(self, mode):
        sw = _switches([mode, "-i", "cases.csv"])
        assert sw.run_time == 3
        assert sw.proc_timeout == 3600
        assert sw.deterministic_level == 0
        assert sw.warmup is True
        assert sw.golden_mode == "Enable"
        assert sw.compare_method is None  # None = per-output 默认路由（浮点 mix_tolerance）

    def test_kernel_mode_defaults_via_bridge(self):
        sw = _switches(["kernel", "-i", "cases.csv"])
        assert sw.dyn_switches.enabled is True
        assert sw.cst_switches.enabled is False
        assert sw.bin_switches.enabled is False


def test_bare_b_enables_binary_mode():
    """裸 -b 启用 binary 模式（默认 realtime）；-b=release 才切预编译复用。

    回归：bridge 曾以 `elif args.binary is not True` 吞掉裸旗标——-b 后
    bin_switches.enabled 仍为 False，用例静默 BIN_OFF。
    """
    sw = _switches(["kernel", "-i", "x.csv", "-b"])
    assert sw.bin_switches.enabled is True
    assert sw.bin_switches.realtime != "release"

    sw = _switches(["kernel", "-i", "x.csv", "-b=release"])
    assert sw.bin_switches.enabled is True
    assert sw.bin_switches.realtime == "release"

    sw = _switches(["kernel", "-i", "x.csv", "-b=false"])
    assert sw.bin_switches.enabled is False

    sw = _switches(["kernel", "-i", "x.csv"])
    assert sw.bin_switches.enabled is False


# ── 组内注册序 pin（业界惯例：功能分组 + 组内定序）─────────────────────────


class TestHelpGroupOrderPins:
    """各组内参数按 dest（规范名）字母序注册——防新增参数错位漂移。

    比较键 = 去掉 -/_ 分隔符的 dest（裸 ASCII 序有 "_" < 字母 的陷阱：
    proc_timeout 会排到 process_count 前，违反显示直觉）。
    """

    @pytest.mark.parametrize("mode", ["kernel", "geir", "aclnn", "e2e"])
    def test_within_group_registration_is_dest_sorted(self, mode):
        _, sub_map = build_parser()

        def order_key(dest):
            return dest.replace("_", "").replace("-", "")

        for group in sub_map[mode]._action_groups:
            dests = [action.dest for action in group._group_actions]
            assert dests == sorted(dests, key=order_key), f"[{mode}] group {group.title!r} out of order: {dests}"

    def test_subcommand_display_order_is_layered(self):
        """主 help 子命令展示序 = 架构分层序（kernel 算子 → geir 图 → aclnn 引擎 API → e2e 框架）。"""
        parser, _ = build_parser()
        choices = parser._subparsers._group_actions[0]._choices_actions
        assert [c.dest for c in choices][:4] == ["kernel", "geir", "aclnn", "e2e"]
