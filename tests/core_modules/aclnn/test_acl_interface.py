#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for ACL runtime initialization interoperability."""

import ctypes

import numpy
import pytest

from ttk.core_modules.aclnn.runtime import (
    ACL_ERROR_REPEAT_INITIALIZE,
    AclInterface,
)
from ttk.core_modules.kernel.runtime import RTSInterface
from ttk.utilities.views import numpy_as_strided_safe


class FakeRtsInterface(RTSInterface):
    @classmethod
    def create(cls, captured):
        instance = cls.__new__(cls)
        instance.captured = captured
        instance.skip_teardown = True
        return instance

    def copy_nparray_to_hbm(self, array):
        self.captured["copied_array"] = array
        self.captured["flattened_array"] = self._flatten_numpy_array(array)
        return ctypes.c_void_p(0x1000)


class NumpyTensorAclInterface(AclInterface):
    @classmethod
    def create(cls, captured):
        instance = cls.__new__(cls)
        instance.captured = captured
        instance._rts_interface = FakeRtsInterface.create(captured)
        instance._acl_tensor_to_device_mem = {}
        instance._acl_tensors = set()
        return instance

    def _opbase_api_call_with_ptr_return(self, api_name, extra_log, *args):
        self.captured["api_name"] = api_name
        self.captured["extra_log"] = extra_log
        self.captured["args"] = args
        return ctypes.c_void_p(0x2000)

    def _on_exit(self):
        self.captured["closed"] = True


def make_numpy_tensor_device():
    captured = {}
    return NumpyTensorAclInterface.create(captured), captured


def _noop():
    pass


def test_parse_error_accepts_repeat_initialization():
    AclInterface.parse_error(
        ACL_ERROR_REPEAT_INITIALIZE,
        "aclInit",
        "runtime already initialized by a custom input callback",
        accepted_errors=(ACL_ERROR_REPEAT_INITIALIZE,),
    )


def test_parse_error_keeps_other_acl_errors_strict():
    with pytest.raises(RuntimeError, match="100003"):
        AclInterface.parse_error(
            100003,
            "aclInit",
            "unexpected initialization failure",
            accepted_errors=(ACL_ERROR_REPEAT_INITIALIZE,),
        )


def test_acl_init_tracks_external_runtime(monkeypatch):
    device = AclInterface.__new__(AclInterface)
    device._acl_inited = False
    device._owns_acl_runtime = False

    def _noop():
        pass

    device._on_exit = _noop
    call = {}

    def record_call(kind, api_name, extra_log, *args, accepted_errors=()):
        call.update(
            {
                "kind": kind,
                "api_name": api_name,
                "extra_log": extra_log,
                "args": args,
                "accepted_errors": accepted_errors,
            }
        )
        return ACL_ERROR_REPEAT_INITIALIZE

    monkeypatch.setattr(device, "_api_call", record_call)

    device._acl_init()

    assert device._acl_inited is True
    assert device._owns_acl_runtime is False
    assert call == {
        "kind": "ACL",
        "api_name": "aclInit",
        "extra_log": None,
        "args": (None,),
        "accepted_errors": (ACL_ERROR_REPEAT_INITIALIZE,),
    }


@pytest.mark.parametrize(("owns_runtime", "expected_reset_calls"), [(False, 0), (True, 1)])
def test_reset_only_resets_owned_runtime_device(monkeypatch, owns_runtime, expected_reset_calls):
    device = AclInterface.__new__(AclInterface)
    device._owns_acl_runtime = owns_runtime
    device._device_id = 0
    device._rts_interface = None
    device._on_exit = _noop
    reset_calls = []

    monkeypatch.setattr(device, "_release_acl_memory", _noop)

    def _record_reset():
        reset_calls.append(True)

    monkeypatch.setattr(device, "_acl_reset_device", _record_reset)

    device.reset()

    assert len(reset_calls) == expected_reset_calls
    if not owns_runtime:
        assert device._device_id is None


@pytest.mark.parametrize(("owns_runtime", "expected_finalize_calls"), [(False, 0), (True, 1)])
def test_finalize_only_finalizes_owned_runtime(monkeypatch, owns_runtime, expected_finalize_calls):
    device = AclInterface.__new__(AclInterface)
    device._acl_inited = True
    device._owns_acl_runtime = owns_runtime
    device._on_exit = _noop
    finalize_calls = []

    monkeypatch.setattr(
        device,
        "_acl_api_call",
        lambda api_name, extra_log, *args: finalize_calls.append(api_name),
    )

    device._acl_finalize()

    assert len(finalize_calls) == expected_finalize_calls
    assert device._acl_inited is False
    assert device._owns_acl_runtime is False


def test_create_acl_tensor_from_numpy_copies_complete_storage_with_offset():
    storage = numpy.arange(64, dtype=numpy.uint8).reshape(4, 16)
    view = numpy_as_strided_safe(storage.ravel()[3:], shape=(2, 2, 4), strides=(16, 4, 1))
    device, captured = make_numpy_tensor_device()

    device.create_acl_tensor(view, "ND", storage.shape)

    copied = captured["copied_array"]
    assert copied.flags.c_contiguous
    assert copied.nbytes == storage.nbytes
    assert numpy.array_equal(copied.reshape(storage.shape), storage)
    assert tuple(captured["args"][3]) == (16, 4, 1)
    assert captured["args"][4].value == 3


def test_create_acl_tensor_from_numpy_rejects_ambiguous_parent_storage():
    parent = numpy.arange(128, dtype=numpy.uint8)
    storage_shape = (4, 16)
    # The 93-byte tail and 128-byte parent are both larger than the declared storage.
    view = numpy_as_strided_safe(parent[35:], shape=(2, 2, 4), strides=(16, 4, 1))
    device, _ = make_numpy_tensor_device()

    with pytest.raises(ValueError, match="exact contiguous numpy storage"):
        device.create_acl_tensor(view, "ND", storage_shape)


def test_get_op_api_resolves_so_via_generic_helper(monkeypatch, tmp_path):
    """_get_op_api 走 ensure_api_so_path 通用解析：空 so_path 回退 libopapi 候选，
    单卡与多卡共用同一条解析链。"""
    from collections import OrderedDict
    from types import SimpleNamespace

    from ttk.core_modules.aclnn import info_keeper
    from ttk.core_modules.aclnn.info_keeper import OpApiInfo, OpApiInfoKeeper
    from ttk.utilities.singleton import Singleton

    so = tmp_path / "libopapi.so"
    so.touch()
    params = OrderedDict([("x", {"type": "aclTensor*", "default": None})])
    info = OpApiInfo(params)
    info._api_name = "aclnnFake"
    info.category = "custom"
    info._so_path = ""
    keeper = OpApiInfoKeeper.__new__(OpApiInfoKeeper)
    keeper._hdr_dirs = []
    keeper._api_info = {"aclnnFake": info}
    key = (OpApiInfoKeeper, (), ())
    saved = dict(Singleton._instances)
    Singleton._instances[key] = keeper
    try:
        monkeypatch.setattr(info_keeper, "get_ascend_lib64_path", lambda: str(tmp_path))
        marker = lambda a, b: None  # noqa: E731
        monkeypatch.setattr(AclInterface, "_SO_CACHE", {str(so): SimpleNamespace(aclnnFake=marker)})
        monkeypatch.setattr(AclInterface, "_OP_API_CACHE", {})
        device = AclInterface.__new__(AclInterface)

        assert device._get_op_api("aclnnFake") is marker
        assert info._so_path == str(so)
        # 二次调用命中函数指针缓存（同一对象，restype 已预置）
        import ctypes as _ctypes

        cached = device._get_op_api("aclnnFake")
        assert cached is marker
        assert cached.restype == _ctypes.c_uint64
    finally:
        Singleton._instances.clear()
        Singleton._instances.update(saved)


class TestCloneForThread:
    """多卡线程设备克隆契约：账本线程私有、teardown 抑制、RTS 委托。

    原手工 __dict__ 克隆（hccl_multi_device）漏重置 _acl_tensor_lists/
    _acl_scalar_lists/_acl_tensor_to_device_mem，且 _acl_device_memories 是
    不存在的属性名——共享容器并发写只能靠 GIL 侥幸，收口后由本测试看守。
    """

    _ACCOUNTS = (
        "_acl_tensors",
        "_acl_tensor_lists",
        "_acl_scalars",
        "_acl_scalar_lists",
        "_acl_bool_arrays",
        "_acl_int_arrays",
        "_acl_float_arrays",
        "_acl_tensor_to_device_mem",
    )

    @staticmethod
    def _fake_rts():
        rts = RTSInterface.__new__(RTSInterface)
        rts.__dict__.update(
            rtsdll=None,
            camodel=None,
            skip_teardown=False,
            kernel_binary_storage={0x1: "bin"},
            context_storage=["ctx"],
            memory_manager={0x2: "mem"},
            size_info_storage=["tpl"],
            module_id=58,
            short_soc_version="Ascend950",
            device_id=0,
            context="ctx",
        )
        return rts

    @classmethod
    def _main_device(cls):
        main = AclInterface.__new__(AclInterface)
        main.__dict__.update(
            _acl_tensors={0x11},
            _acl_tensor_lists={0x12: ()},
            _acl_scalars={0x13},
            _acl_scalar_lists={0x14},
            _acl_bool_arrays={0x15},
            _acl_int_arrays={0x16},
            _acl_float_arrays={0x17},
            _acl_tensor_to_device_mem={0x18: None},
            _acl_inited=True,
            _owns_acl_runtime=True,
            _device_id=0,
            skip_teardown=False,
            _rts_interface=cls._fake_rts(),
        )
        return main

    def test_acl_clone_resets_all_accounts_and_suppresses_teardown(self):
        main = self._main_device()
        clone = main.clone_for_thread(3)

        for name in self._ACCOUNTS:
            fresh = getattr(clone, name)
            assert fresh is not getattr(main, name), f"{name} must not be shared with the main device"
            assert len(fresh) == 0, f"{name} must start empty"
        # 容器类型不得混淆（曾以 set 覆盖 dict 账本位）
        assert isinstance(clone._acl_tensor_lists, dict)
        assert isinstance(clone._acl_tensor_to_device_mem, dict)
        assert isinstance(clone._acl_tensors, set)
        assert clone._device_id == 3
        assert clone._acl_inited is True
        assert clone._owns_acl_runtime is False
        assert clone._suppress_finalize is True
        assert clone._suppress_reset is True

    def test_acl_clone_delegates_to_rts_clone(self):
        main = self._main_device()
        clone = main.clone_for_thread(5)
        rts = clone._rts_interface

        assert rts is not main._rts_interface
        assert rts.device_id == 5
        assert rts.skip_teardown is True
        assert rts.context is None
        for name in ("kernel_binary_storage", "memory_manager", "context_storage", "size_info_storage"):
            fresh = getattr(rts, name)
            assert fresh is not getattr(main._rts_interface, name), f"rts.{name} must not be shared"
            assert len(fresh) == 0, f"rts.{name} must start empty"

    def test_rts_clone_reset_is_noop(self):
        """RTS 克隆 reset 为空操作（skip_teardown）——线程 __del__ 不会 rtDeviceReset。

        rtsdll 置 None：若 reset 未被 skip_teardown 拦截，api_call 会在此炸出，
        测试即失败。
        """
        clone = self._fake_rts().clone_for_thread(2)
        clone.reset()
        clone.__del__()
