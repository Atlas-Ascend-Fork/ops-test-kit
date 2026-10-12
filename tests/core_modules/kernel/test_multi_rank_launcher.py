# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MultiRankLauncher 编排逻辑回归（mock 设备，无真机依赖）：
单卡退化走 rank0 直发；多卡齐发后逐 rank 清理；rank 失败 abort barrier
且不再发射 rank0；_pack_launch_op_args 打包序与 placeholder 语义。"""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy
import pytest

import ttk.core_modules.kernel.execute as execute_mod
from ttk.core_modules.kernel.execute import MultiRankLauncher, _pack_launch_op_args
from ttk.core_modules.kernel.runtime.hccl_helper import is_hccl_op


def _fake_switches(run_timeout=60):
    sw = MagicMock()
    sw.run_timeout = run_timeout
    return sw


def _fake_param(fat_bin=False, output_placeholder=True, tensor_list=False):
    inputs = [numpy.ones((4,), dtype=numpy.float16)]
    if tensor_list:
        inputs.append([numpy.ones((2,), dtype=numpy.float16), numpy.ones((2,), dtype=numpy.float16)])
    outputs = [numpy.zeros((4,), dtype=numpy.float16)]
    workspaces = [numpy.zeros((8,), dtype=numpy.uint8)]
    return SimpleNamespace(
        input_arrays=tuple(inputs),
        output_arrays=tuple(outputs),
        workspace_arrays=tuple(workspaces),
        dfx_arrays=(),
        output_placeholder=output_placeholder,
        kernel_dir="/tmp/kernel_meta",
        kernel_name="dyn_op_case",
        kernel_main_func_name="dyn_op_case",
        tiling_key=1024 if fat_bin else None,
        tiling_data=b"\x01\x02",
        block_dim=8,
        simt_share_memory_size=0,
        op_name="all_gather_matmul",
        kernel_json_info=SimpleNamespace(
            is_fat_bin=fat_bin, is_mix_kernel=False, schedule_mode=0, magic="RT_DEV_BINARY_MAGIC_ELF"
        ),
    )


def _fake_device(device_id=0, fail_on=None):
    dev = MagicMock()
    dev.device_id = device_id
    dev.is_model.return_value = False
    dev.copy_nparray_to_hbm.side_effect = lambda arr, fill_oob_flag=False: ctypes.c_void_p(id(arr) & 0xFFFFFFFFFF)
    dev.register_device_binary_kernel.return_value = ctypes.c_void_p(0xB1)
    dev.register_function.return_value = ctypes.c_void_p(0xF1)
    dev.register_all_kernel.return_value = ctypes.c_void_p(0xAB)
    dev.int_magic.return_value = 0x100
    if fail_on:
        getattr(dev, fail_on).side_effect = RuntimeError(f"{fail_on} failed")
    return dev


def _fake_executor(param, device, switches):
    ex = MagicMock()
    ex._param = param
    ex._device = device
    ex._switches = switches
    ex._launch_with.side_effect = lambda dev, stream, handle, op_args, dfx: dev.launch_kernel_with_handle(None, stream)
    return ex


def test_pack_args_order_placeholder():
    param = _fake_param(output_placeholder=True)
    dvc_mem = [[1, 2], [3], [4], []]
    with patch.object(execute_mod.RTSInterface, "prepare_tensor_list_info", side_effect=lambda a, b: "TLI"):
        assert _pack_launch_op_args(dvc_mem, param) == [1, 2, 3, 4]
    param = _fake_param(output_placeholder=False)
    assert _pack_launch_op_args(dvc_mem, param) == [1, 2, 4]


def test_pack_args_tensor_list_slot():
    param = _fake_param(output_placeholder=False, tensor_list=True)
    dvc_mem = [[1, [11, 12]], [5], [6], []]
    with patch.object(execute_mod.RTSInterface, "prepare_tensor_list_info", side_effect=lambda a, b: "TLI") as tl:
        packed = _pack_launch_op_args(dvc_mem, param)
    assert packed == [1, "TLI", 6]
    tl.assert_called_once_with([11, 12], param.input_arrays[1])


def test_launch_single_rank_degenerate(monkeypatch):
    param = _fake_param()
    device = _fake_device(0)
    executor = _fake_executor(param, device, _fake_switches())
    monkeypatch.setattr(execute_mod, "init_hccl", lambda: ([0, 1], "ep", 1))
    MultiRankLauncher(executor).launch(stream=ctypes.c_void_p(7))
    (stream_arg,) = executor._launch_single_rank.call_args.args
    assert stream_arg.value == 7
    executor._launch_with.assert_not_called()


def test_launch_two_ranks_happy_path(monkeypatch):
    param = _fake_param()
    device = _fake_device(0)
    executor = _fake_executor(param, device, _fake_switches())
    rank1 = _fake_device(1)
    device.clone_for_thread.return_value = rank1
    monkeypatch.setattr(execute_mod, "init_hccl", lambda: ([0, 1], "ep", 2))

    MultiRankLauncher(executor).launch(stream=ctypes.c_void_p(7))

    executor._launch_single_rank.assert_called_once()
    device.clone_for_thread.assert_called_once_with(1)
    rank1.set_device.assert_called_once_with(1)
    rank1.create_context.assert_called_once()
    rank1.create_stream.assert_called_once()
    rank1.register_device_binary_kernel.assert_called_once()
    func_name = rank1.register_function.call_args.args[1]
    binary_hdl = rank1.register_function.call_args.args[0]
    assert func_name == "dyn_op_case"
    assert binary_hdl.value == 0xB1
    rank1.synchronize_with_stream.assert_called_once()
    executor._launch_with.assert_called_once()
    assert rank1.api_call.call_args.args[0] == "rtCtxDestroy"
    rank1.unregister_device_binary_kernel.assert_called_once()
    rank1.destroy_stream_force.assert_called_once()
    copied = [c.args[0] for c in rank1.copy_nparray_to_hbm.call_args_list]
    assert copied[0] is param.input_arrays[0]
    assert copied[1] is param.output_arrays[0]
    assert copied[2] is param.workspace_arrays[0]


def test_launch_rank_setup_failure_aborts(monkeypatch):
    param = _fake_param()
    device = _fake_device(0)
    executor = _fake_executor(param, device, _fake_switches())
    device.clone_for_thread.return_value = _fake_device(1, fail_on="create_context")
    monkeypatch.setattr(execute_mod, "init_hccl", lambda: ([0, 1], "ep", 2))

    with pytest.raises(RuntimeError, match="aborted before launch"):
        MultiRankLauncher(executor).launch(stream=ctypes.c_void_p(7))
    executor._launch_single_rank.assert_not_called()


def test_launch_rank1_kernel_failure_raises(monkeypatch):
    param = _fake_param()
    device = _fake_device(0)
    executor = _fake_executor(param, device, _fake_switches())
    rank1 = _fake_device(1, fail_on="synchronize_with_stream")
    device.clone_for_thread.return_value = rank1
    monkeypatch.setattr(execute_mod, "init_hccl", lambda: ([0, 1], "ep", 2))

    with pytest.raises(RuntimeError, match="synchronize_with_stream failed"):
        MultiRankLauncher(executor).launch(stream=ctypes.c_void_p(7))
    executor._launch_single_rank.assert_called_once()
    rank1.unregister_device_binary_kernel.assert_called_once()


def test_hccl_op_set_is_frozen_constant():
    assert is_hccl_op("all_gather_matmul")
    assert is_hccl_op("moe_ep_combine")
    assert not is_hccl_op("add")
    assert not is_hccl_op("")
