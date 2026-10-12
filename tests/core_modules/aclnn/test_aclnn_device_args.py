# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn 缺失 device args 补齐的回归：-l 守门 + --simt-stack-* rtLimit 接线。

--clear-ub/l1/l0 经核对本就有效（AclInterface→RTSInterface 委托 + instance 预编译
helper kernels）——此前审计误判为未接线。"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _wiring_args(**kwargs):
    base = {"provider": None, "config": None, "input": "x.csv", "output": None, "run": 1}
    base.update(kwargs)
    return SimpleNamespace(**base)


# ── CLI 注册与解析 ──────────────────────────────────────────────────────────
def test_hbm_simt_args_registered_on_kernel_and_aclnn():
    """-l/--simt-stack-*（运行期 rtDeviceSetLimit）在 kernel/aclnn 注册；
    --simt-ub（编译期）kernel 专属；e2e/geir 均不注册。"""
    import argparse

    from ttk.cli.aclnn import register_aclnn_command
    from ttk.cli.e2e import register_e2e_command
    from ttk.cli.geir import register_geir_command
    from ttk.cli.kernel import register_kernel_command

    sub = argparse.ArgumentParser().add_subparsers()
    matrix = {
        register_kernel_command: ("--limit", "--simt-ub", "--simt-stack-dcu", "--simt-stack-dvg"),
        register_aclnn_command: ("--limit", "--simt-stack-dcu", "--simt-stack-dvg"),
    }
    for register, expect_set in matrix.items():
        parser = register(sub)
        opts = {o for a in parser._actions for o in a.option_strings}  # noqa: SLF001
        for expect in expect_set:
            assert expect in opts, (register.__module__, expect)
        assert "--simt-ub" not in opts or register is register_kernel_command
    for register in (register_e2e_command, register_geir_command):
        parser = register(sub)
        opts = {o for a in parser._actions for o in a.option_strings}  # noqa: SLF001
        assert "--limit" not in opts
        assert "--simt-ub" not in opts
        assert "--simt-stack-dcu" not in opts


def test_limit_and_simt_wired_via_args_to_switches():
    """-l/--simt-* 经 args_to_switches（kernel/aclnn 共享路径）落到 switches。"""
    from ttk.cli.bridge import args_to_switches

    sw = args_to_switches(_wiring_args(limit=16, simt_ub=100000, simt_stack_dcu=8192, simt_stack_dvg=4096))
    assert sw.hbm_size_limit == 16
    assert sw.force_simt_ub_size == [100000] * 3
    assert sw.simt_cfg.dcu_stack == 8192
    assert sw.simt_cfg.dvg_stack == 4096


def test_simt_ub_cli_delivers_int_not_str_slice():
    """--simt-ub 补 type=int 的回归：字符串切片（['1','0','0']）旧 bug 不复现。"""
    import argparse

    from ttk.cli.device import add_device_control_args

    p = argparse.ArgumentParser()
    add_device_control_args(p, kernel_only=True)
    assert p.parse_args(["--simt-ub", "100000"]).simt_ub == 100000


# ── aclnn HBM 守门 ──────────────────────────────────────────────────────────
def _aclnn_case(shapes, dtypes):
    from ttk.core_modules.aclnn.testcase import TestcaseAclnn

    case = TestcaseAclnn()
    case.testcase_name = "hbm_gate_probe"
    case.api_name = "aclnnAdd"
    case.tensor_view_shapes = shapes
    case.tensor_dtypes = dtypes
    case.input_data_ranges = ((None, None),) * len(shapes)
    case.is_valid = True
    return case


def test_aclnn_shape_out_of_bound_fails_validate():
    """输入总量超 -l 上限 → STC_SHAPE_OUT_OF_BOUND（对齐 kernel 语义）。"""
    case = _aclnn_case((((999999999,),),), (("float32",),))
    mock_gs = MagicMock()
    mock_gs.hbm_size_limit = 0
    with patch("ttk.core_modules.aclnn.testcase.get_global_storage", return_value=mock_gs):
        # 守门在 validate 链尾，前置检查失败不阻断守门验证——直接调守门方法
        case._stc_shape_size_check()
    assert not case.is_valid
    assert case.fail_reason == "STC_SHAPE_OUT_OF_BOUND"


def test_aclnn_shape_within_bound_keeps_valid():
    """正常体量输入不触发守门。"""
    case = _aclnn_case((((2, 3),),), (("float32",),))
    mock_gs = MagicMock()
    mock_gs.hbm_size_limit = 30
    with patch("ttk.core_modules.aclnn.testcase.get_global_storage", return_value=mock_gs):
        case._stc_shape_size_check()
    assert case.is_valid


def test_aclnn_dynamic_slot_skipped_in_gate():
    """-1 动态槽（None shape）不计入守门总量、不抛 shape_numel。"""
    case = _aclnn_case((None, (2, 3)), ("float32", "float32"))
    mock_gs = MagicMock()
    mock_gs.hbm_size_limit = 0
    # None 槽跳过 + (2,3) float32=24B > 0GB 仍超限——但 None 槽不参与 numel 计算
    with patch("ttk.core_modules.aclnn.testcase.get_global_storage", return_value=mock_gs):
        case._stc_shape_size_check()
    assert not case.is_valid  # (2,3) 位 24B > 0B 上限
    assert case.fail_reason == "STC_SHAPE_OUT_OF_BOUND"


# ── aclnn SIMT rtLimit 接线 ─────────────────────────────────────────────────
def _executor_stub(switches, device):
    from ttk.core_modules.aclnn.execute import AclOpExecutor

    ex = AclOpExecutor.__new__(AclOpExecutor)
    ex._switches = switches
    ex._dvc = device
    return ex


def test_apply_simt_stack_config_real_device():
    """配置了 --simt-stack-* → rts_context 阶段下发 rtLimit（真机不带 device_id）。"""
    switches = SimpleNamespace(simt_cfg=SimpleNamespace(dcu_stack=8192, dvg_stack=4096))
    device = MagicMock()
    device.is_model.return_value = False
    device.device_id = 3
    _executor_stub(switches, device)._apply_simt_stack_config()
    device.set_simt_stack_size.assert_called_once_with(8192, 4096, None)


def test_apply_simt_stack_config_model_passes_device_id():
    """npusim（model）形态传 device_id（kernel/execute 同款语义）。"""
    switches = SimpleNamespace(simt_cfg=SimpleNamespace(dcu_stack=8192, dvg_stack=None))
    device = MagicMock()
    device.is_model.return_value = True
    device.device_id = 3
    _executor_stub(switches, device)._apply_simt_stack_config()
    device.set_simt_stack_size.assert_called_once_with(8192, None, 3)


def test_apply_simt_stack_config_noop_when_unset():
    """未配置（None/None）不触达 rtLimit——每用例无谓调用为零。"""
    switches = SimpleNamespace(simt_cfg=SimpleNamespace(dcu_stack=None, dvg_stack=None))
    device = MagicMock()
    _executor_stub(switches, device)._apply_simt_stack_config()
    device.set_simt_stack_size.assert_not_called()


@pytest.mark.parametrize("missing", ["dcu_stack", "dvg_stack", "simt_cfg"])
def test_apply_simt_stack_config_tolerates_partial(missing):
    """simt_cfg 缺失或单边配置不报错（单边 None 由 RTSInterface 内部跳过）。"""
    if missing == "simt_cfg":
        switches = SimpleNamespace()
    elif missing == "dcu_stack":
        switches = SimpleNamespace(simt_cfg=SimpleNamespace(dcu_stack=None, dvg_stack=4096))
    else:
        switches = SimpleNamespace(simt_cfg=SimpleNamespace(dcu_stack=8192, dvg_stack=None))
    device = MagicMock()
    device.is_model.return_value = False
    _executor_stub(switches, device)._apply_simt_stack_config()
    if missing != "simt_cfg":
        device.set_simt_stack_size.assert_called_once()
