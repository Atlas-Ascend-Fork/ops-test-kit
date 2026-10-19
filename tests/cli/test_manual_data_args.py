# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""manual-data 两阶段模式测试：--manual-data-dirs / --no-prof 参数解析、
prepare/replay 模式分发、互斥校验、--clear-ub/--clear-l1/--clear-l0 数值解析。"""

import argparse
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.cli.aclnn import register_aclnn_command
from ttk.cli.bridge import (
    _log_manual_data_configuration,
    _parse_clean_val,
    configure_manual_data,
)
from ttk.cli.e2e import register_e2e_command
from ttk.cli.geir import register_geir_command
from ttk.cli.kernel import register_kernel_command
from ttk.core_modules.common.dump import DumpConfig
from ttk.switches import SWITCHES


def _parser():
    """构造带 e2e/aclnn/kernel/geir 四子命令的 parser。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_e2e_command(subparsers)
    register_aclnn_command(subparsers)
    register_kernel_command(subparsers)
    register_geir_command(subparsers)
    return parser


def _args(**overrides):
    """构造 configure_manual_data 所需的 args（manual_data_dirs + no_prof）。"""
    values = {"manual_data_dirs": None, "no_prof": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepare_switches(file_format="bin"):
    """构造 prepare 模式所需 SWITCHES：开 input+golden dump、指定 file_format。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.dyn_switches.enabled = False
    switches.dump_config.dump_input = True
    switches.dump_config.dump_golden = True
    switches.dump_config.file_format = file_format
    return switches


# -- 参数暴露 ----------------------------------------------------------------


def test_manual_data_dirs_exposed_on_four_commands():
    """e2e/aclnn/kernel/geir 四个子命令都暴露 --manual-data-dirs 和 --no-prof。"""
    parser = _parser()
    e2e = parser.parse_args(
        [
            "e2e",
            "-i",
            "case.csv",
            "--no-prof",
            "--dump",
            "in,golden",
            "--manual-data-dirs",
            "prepared",
        ]
    )
    aclnn = parser.parse_args(
        [
            "aclnn",
            "-i",
            "case.csv",
            "--manual-data-dirs",
            "first",
            "second",
        ]
    )
    kernel = parser.parse_args(
        [
            "kernel",
            "-i",
            "case.csv",
            "--manual-data-dirs",
            "prepared",
        ]
    )
    geir = parser.parse_args(
        [
            "geir",
            "-i",
            "case.csv",
            "--no-prof",
            "--dump",
            "in,golden",
            "--manual-data-dirs",
            "prepared",
        ]
    )

    assert e2e.no_prof is True
    assert e2e.manual_data_dirs == ["prepared"]
    assert aclnn.manual_data_dirs == ["first", "second"]
    assert kernel.manual_data_dirs == ["prepared"]
    assert geir.no_prof is True
    assert geir.manual_data_dirs == ["prepared"]


# -- dump 方向（--no-prof 落盘） ----------------------------------------------


def test_prepare_defaults_to_plugin_manual_data_dir(tmp_path):
    """--no-prof 落盘默认输出目录 = <plugin_path>/manual_data。"""
    switches = _prepare_switches()
    switches.plugin_path = (tmp_path / "assets",)

    configure_manual_data(switches, _args(no_prof=True), "e2e")

    assert switches.no_prof is True
    assert switches.manual_data_dirs == (str((tmp_path / "assets" / "manual_data").resolve()),)


def test_prepare_without_plugin_defaults_to_cwd(tmp_path, monkeypatch, caplog):
    """无 plugin 时 --no-prof 落盘默认输出 cwd/manual_data 并打 INFO 日志。"""
    switches = _prepare_switches()
    monkeypatch.chdir(tmp_path)

    configure_manual_data(switches, _args(no_prof=True), "e2e")
    with caplog.at_level(logging.INFO):
        _log_manual_data_configuration(switches)

    assert switches.manual_data_dirs == (str((tmp_path / "manual_data").resolve()),)
    assert "using current-directory manual-data output" in caplog.text


def test_prepare_multiple_plugins_requires_explicit_output(tmp_path):
    """多 plugin 路径 + 无显式输出目录 → 报错（歧义）。"""
    switches = _prepare_switches()
    switches.plugin_path = (tmp_path / "first", tmp_path / "second")

    with pytest.raises(ValueError, match="multiple --plugin paths"):
        configure_manual_data(switches, _args(no_prof=True), "e2e")


def test_prepare_accepts_explicit_output_directory(tmp_path):
    """显式 --manual-data-dirs 覆盖默认推导路径。"""
    switches = _prepare_switches("pt")
    switches.plugin_path = (tmp_path / "first", tmp_path / "second")

    configure_manual_data(
        switches,
        _args(no_prof=True, manual_data_dirs=[str(tmp_path / "data")]),
        "aclnn",
    )

    assert switches.no_prof is True
    assert switches.manual_data_dirs == (str((tmp_path / "data").resolve()),)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        # dump 格式不可恢复（print 无法回放）
        (lambda sw: setattr(sw.dump_config, "file_format", "print"), "not restorable"),
        # --golden-mode Disable 关闭 CPU golden 生成，golden 无从落盘
        (lambda sw: setattr(sw, "golden_mode", "Disable"), "CPU golden"),
        # e2e/aclnn --no-prof 必须开 input 落盘门
        (lambda sw: setattr(sw.dump_config, "dump_input", False), "requires --dump in"),
    ],
    ids=["unrestorable-format", "golden-disable", "no-input-dump"],
)
def test_prepare_rejects_incompatible_combinations(mutate, message):
    """--no-prof 落盘拒绝 3 种不兼容组合（落什么由 dump 门控自然决定）。"""
    switches = _prepare_switches()
    mutate(switches)

    with pytest.raises(ValueError, match=message):
        configure_manual_data(switches, _args(no_prof=True), "e2e")


def test_prepare_allows_full_dump_combo():
    """--no-prof --dump full 不再报错：output 门自然无产物（不上板）。"""
    switches = _prepare_switches()
    switches.dump_config.dump_output = True

    configure_manual_data(switches, _args(no_prof=True), "e2e")

    assert switches.no_prof is True
    assert switches.manual_data_dirs  # 默认目录已设置


def test_e2e_prepare_accepts_input_only_dump(tmp_path):
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.dyn_switches.enabled = False
    switches.dump_config.dump_input = True
    switches.golden_mode = "Disable"

    configure_manual_data(switches, _args(no_prof=True), "e2e")

    assert switches.no_prof is True
    assert switches.manual_data_dirs  # 默认目录已设置
    assert switches.dump_config.dump_input
    assert not switches.dump_config.dump_golden


# -- load 方向（--manual-data-dirs 搜索） ------------------------------------


def test_replay_preserves_directory_order(tmp_path):
    """load 方向按 --manual-data-dirs 指定顺序搜索（顺序敏感）。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    directories = [str(tmp_path / "one"), str(tmp_path / "two")]

    configure_manual_data(switches, _args(manual_data_dirs=directories), "aclnn")

    assert switches.no_prof is False
    assert switches.manual_data_dirs == tuple(str((tmp_path / name).resolve()) for name in ("one", "two"))


def test_e2e_replay_rejects_cpu_backend(tmp_path):
    """e2e load 是设备阶段，不能与 --cpu 共存。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.force_cpu = True

    with pytest.raises(ValueError, match="device stage"):
        configure_manual_data(switches, _args(manual_data_dirs=[str(tmp_path)]), "e2e")


def test_geir_no_prof_accepts_xpu_golden_mode():
    """geir --no-prof + golden-mode XPU 对齐 kernel：合法组合，占位 golden 过滤后仅落 input。"""
    switches = _prepare_switches()
    switches.golden_mode = "XPU"

    configure_manual_data(switches, _args(no_prof=True), "geir")

    assert switches.no_prof is True
    assert switches.manual_data_dirs  # 默认目录已设置


def test_geir_no_prof_accepts_input_only_dump():
    """geir --no-prof 与 aclnn/e2e 同构：仅 --dump in 也接受（golden 缺席时复跑回退现算）。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.dyn_switches.enabled = False
    switches.dump_config.dump_input = True
    switches.golden_mode = "Disable"

    configure_manual_data(switches, _args(no_prof=True), "geir")

    assert switches.no_prof is True
    assert switches.manual_data_dirs  # 默认目录已设置
    assert not switches.dump_config.dump_golden


# -- kernel 特有语义 ---------------------------------------------------------


def test_kernel_no_prof_without_dump_pair_keeps_legacy():
    """kernel --no-prof 但无 --dump in,golden → 不落盘，保持旧 dry-run 语义。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()

    configure_manual_data(switches, _args(no_prof=True), "kernel")

    assert switches.no_prof is True
    assert switches.manual_data_dirs == ()


def test_kernel_prepare_requires_exact_dump_pair(tmp_path):
    """kernel 落盘需要精确的 --dump in,golden 组合 + 可恢复格式。"""
    switches = _prepare_switches("npy")

    configure_manual_data(
        switches,
        _args(no_prof=True, manual_data_dirs=[str(tmp_path)]),
        "kernel",
    )

    assert switches.no_prof is True
    assert switches.manual_data_dirs == (str(tmp_path.resolve()),)


def test_kernel_prepare_without_dump_pair_rejected(tmp_path):
    """kernel 有目录但无 --dump in,golden → 报错。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()

    with pytest.raises(ValueError, match="exactly --no-prof --dump in,golden"):
        configure_manual_data(
            switches,
            _args(no_prof=True, manual_data_dirs=[str(tmp_path)]),
            "kernel",
        )


def test_kernel_manual_data_rejects_compile_only(tmp_path):
    """kernel manual-data（dump 方向）不能与 --compile-only 共存。"""
    switches = _prepare_switches()
    switches.compile_only = True

    with pytest.raises(ValueError, match="compile-only"):
        configure_manual_data(switches, _args(no_prof=True, manual_data_dirs=[str(tmp_path)]), "kernel")


# -- pickle 往返 -------------------------------------------------------------


def test_manual_data_fields_survive_worker_pickle(tmp_path):
    """no_prof / manual_data_dirs 经 pickle 往返不丢（worker 传递保障）。"""
    switches = SWITCHES()
    switches.dump_config = DumpConfig()
    switches.no_prof = True
    switches.manual_data_dirs = (str(tmp_path),)

    restored = pickle.loads(pickle.dumps(switches))  # noqa: S301

    assert restored.no_prof is True
    assert restored.manual_data_dirs == (str(tmp_path),)


# -- --clear-ub / --clear-l1 / --clear-l0 数值解析 ----------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_type", "expected"),
    [
        ("7", np.int32, 7),
        ("0xff", np.int32, 255),
        ("-1.25", np.float32, -1.25),
        ("float16(1.5)", np.float16, 1.5),
        ("uint8(0xff)", np.uint8, 255),
    ],
    ids=["int", "hex", "float", "typed-float16", "typed-uint8"],
)
def test_clear_value_parser_accepts_numeric_literals(value, expected_type, expected):
    """--clear-ub/--clear-l1/--clear-l0 解析十进制/十六进制/浮点/带 dtype 前缀的数值字面量。"""
    parsed = _parse_clean_val("UB", value)

    assert isinstance(parsed, expected_type)
    assert parsed == expected


def test_clear_value_parser_accepts_inf_and_nan():
    """--clear-ub/--clear-l1/--clear-l0 解析特殊浮点值 inf/nan。"""
    assert np.isinf(_parse_clean_val("L1", "float32(inf)"))
    assert np.isnan(_parse_clean_val("L1", "nan"))
    assert np.isinf(_parse_clean_val("L0", "float32(inf)"))
    assert np.isnan(_parse_clean_val("L0", "nan"))


def test_clear_value_parser_rejects_code_injection():
    """--clear-ub 拒绝 __import__ 等代码注入（安全看护）。"""
    with pytest.raises(ValueError, match="Cannot parse UB clean value"):
        _parse_clean_val("UB", "float32(__import__('os').getcwd())")


def test_clear_value_parser_rejects_non_numeric_dtype():
    """--clear-l1 拒绝非数值 dtype（如 object）。"""
    with pytest.raises(ValueError, match="Unsupported L1 clean value dtype"):
        _parse_clean_val("L1", "object(1)")
