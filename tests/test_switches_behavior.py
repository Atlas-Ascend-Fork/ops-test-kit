# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ttk/switches.py 公开接口行为 pin（对象自身语义；CLI→SWITCHES 解析映射在 tests/cli/）。

覆盖：run_time / compile_only / overflow_mode / MODE 谓词 / active 激活 /
pickle 往返（worker 传输契约）/ oom_enabled / OPTestSwitch / 全局存储兼容层。
"""

import pickle

import pytest

from ttk.switches import MODE, SWITCHES, OPTestSwitch, get_global_storage, set_global_storage


def test_root_path_points_to_repo_root():
    """root_path 必须是仓库根（含 pyproject.toml）——曾因模块目录深度变化
    （utilities 3 层 → ttk 2 层 dirname）算偏一级而测试未察觉，此为 pin。"""
    import os

    from ttk import switches as switches_mod

    sw = SWITCHES()
    assert sw.root_path == os.path.dirname(os.path.dirname(os.path.abspath(switches_mod.__file__)))
    assert os.path.isfile(os.path.join(sw.root_path, "pyproject.toml"))


def test_run_time_default_and_model_fallback():
    """板端默认 3 次、仿真形态默认 1 次；显式设定后两者一致。"""
    sw = SWITCHES()
    assert sw.run_time == 3
    sw.mode = MODE.ASCEND_CAMODEL
    assert sw.run_time == 1
    sw.run_time = 2
    assert sw.run_time == 2
    sw.mode = MODE.ASCEND_ONBOARD
    assert sw.run_time == 2


@pytest.mark.parametrize("bad", [0, -1, "3", 1.5, True])
def test_run_time_rejects_non_positive_int(bad):
    """run_time 仅接受正 int（bool 显式排除）；违规抛 ValueError。"""
    with pytest.raises(ValueError, match="positive int"):
        SWITCHES().run_time = bad


def test_compile_only_toggles_prof():
    """compile_only=True 关三模式 prof；=False 强制全开（存量语义 pin——
    False 分支会覆盖用户对 prof 的本意，生产只写 True，改动前先钉住）。"""
    sw = SWITCHES()
    assert not sw.compile_only
    assert (sw.dyn_switches.prof, sw.cst_switches.prof, sw.bin_switches.prof) == (True, True, True)
    sw.compile_only = True
    assert sw.compile_only
    assert (sw.dyn_switches.prof, sw.cst_switches.prof, sw.bin_switches.prof) == (False, False, False)
    sw.compile_only = False
    assert not sw.compile_only
    assert (sw.dyn_switches.prof, sw.cst_switches.prof, sw.bin_switches.prof) == (True, True, True)


def test_overflow_mode_pinned_to_inf_nan():
    """overflow_mode 设计恒 1（INF/NAN 溢出语义）；saturation 通路仅 golden mock 保留。"""
    assert SWITCHES().overflow_mode == 1


@pytest.mark.parametrize(
    ("mode", "is_model", "online", "has_device", "esl"),
    [
        (MODE.ASCEND_ONBOARD, None, True, True, False),
        (MODE.ASCEND_CAMODEL, "CAMODEL", False, False, False),
        (MODE.ASCEND_PEMMODEL, "PEMMODEL", False, False, False),
        (MODE.ASCEND_PVMODEL, "PVMODEL", False, False, False),
        (MODE.ASCEND_ESLMODEL, "ESLMODEL", False, False, True),
    ],
)
def test_mode_predicates(mode, is_model, online, has_device, esl):
    """MODE 谓词真值表：板端唯一有设备；仿真形态返回短名。"""
    assert mode.is_model() == is_model
    assert mode.is_online_board() == online
    assert mode.has_device() == has_device
    assert mode.is_esl_model() == esl


def test_active_lifecycle():
    """未激活 active() 早失败；activate() 后可读；恢复现场防跨测试污染。"""
    saved = SWITCHES._active
    try:
        SWITCHES._active = None
        with pytest.raises(RuntimeError):
            SWITCHES.active()
        sw = SWITCHES()
        sw.activate()
        assert SWITCHES.active() is sw
    finally:
        SWITCHES._active = saved


def test_global_storage_compat_fallback_and_clear():
    """兼容层语义 pin（迁移 active() 完成前删除）：未激活回退同一只共享默认
    实例；激活后优先返回激活实例；set_global_storage(None) 清除回退。"""
    saved = SWITCHES._active
    try:
        SWITCHES._active = None
        assert get_global_storage() is get_global_storage()  # 回退 = 惰性单例
        sw = SWITCHES()
        sw.activate()
        assert get_global_storage() is sw
        set_global_storage(None)
        assert get_global_storage() is not sw  # 清除后回退默认实例
    finally:
        SWITCHES._active = saved


def test_pickle_roundtrip_worker_contract():
    """pickle 往返（parent→worker 传输契约）：property 存储槽与 config 字段不丢，
    run_time/compile_only 语义恢复。"""
    sw = SWITCHES()
    sw.run_time = 5
    sw.compile_only = True
    sw.config_path = "/tmp/x.yaml"
    sw.provider_filter = "torch"
    revived = pickle.loads(pickle.dumps(sw))
    assert revived.run_time == 5
    assert revived.compile_only is True
    assert revived.config_path == "/tmp/x.yaml"
    assert revived.provider_filter == "torch"


@pytest.mark.parametrize(
    ("debug_config", "expected"),
    [
        ("oom,dump_cce", True),
        ("dump_cce", False),
        ("", False),
    ],
)
def test_oom_enabled_reads_compile_options(debug_config, expected):
    """oom_enabled = compile_options['op_debug_config'] 含 'oom' 子串（--compile-opts 透传）。"""
    sw = SWITCHES()
    sw.compile_options["op_debug_config"] = debug_config
    assert sw.oom_enabled() is expected


@pytest.mark.parametrize(
    ("raw", "expected", "expect_raise"),
    [
        ("-DFOO=1 -DBAR=2", ["-DFOO=1", "-DBAR=2"], None),
        ('-DX="a b"', ["-DX=a b"], None),
        ("", [], None),
        (None, [], None),
        (["-DFOO=1"], None, "must be a string"),
    ],
)
def test_macros_parses_compile_options(raw, expected, expect_raise):
    """macros = shlex 解析 --compile-opts macros 值；未设置返回空列表；非字符串报错。"""
    sw = SWITCHES()
    if raw is not None:
        sw.compile_options["macros"] = raw
    if expect_raise:
        with pytest.raises(ValueError, match=expect_raise):
            _ = sw.macros
    else:
        assert sw.macros == expected


def test_op_test_switch_str_and_release_bin():
    """OPTestSwitch：__str__ 三段式格式（worker 日志消费）；realtime='release'
    时 use_release_bin（--binary release 编译复用二进制）。"""
    dyn = SWITCHES().dyn_switches
    assert str(dyn) == "Dynamic Shape: ENABLED, TE_COMPILE, ONLINE"
    release = OPTestSwitch("Binary Release", True, OPTestSwitch.REUSE_BINARY_RELEASE_KERNEL, False)
    assert str(release) == "Binary Release: ENABLED, RELEASE, OFFLINE"
    assert release.use_release_bin() is True
    assert dyn.use_release_bin() is False
