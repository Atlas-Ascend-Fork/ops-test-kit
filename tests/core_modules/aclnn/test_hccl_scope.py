# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""HcclResourceScope 生命周期 pin（形态套用 MultiDeviceLock，不继承）。

- __enter__ 全量获取（返回码检查），__exit__ 统一释放；
- setup 中途失败：按已建部分资源回退，不泄漏（原早退泄漏修复）；
- retain()：句柄转存 context 供 golden 阶段，__exit__ 跳过释放；
- EP/TP 子域 HcclCommInitAll 收到真实设备号（原网格索引 bug 修复：
  非 0 基设备号列表会把子域建到错误设备上）。

全部用假 DLL 驱动，不依赖 CANN 安装。
"""

import pytest

from ttk.core_modules.aclnn.hccl_multi_device import HcclResourceScope


class _FakeFunc:
    """Callable with settable attributes (ctypes restype/argtypes pattern)."""

    def __init__(self, impl):
        self.impl = impl

    def __call__(self, *args):
        return self.impl(*args)


class _FakeDll:
    """Attribute-miss dispatcher: every api name resolves to a recording fake."""

    def __init__(self, impls):
        self._impls = impls
        self.calls = []

    def __getattr__(self, name):
        if name not in self._impls:
            raise AttributeError(name)
        impl = self._impls[name]

        def _record(*args):
            self.calls.append((name, args))
            return impl(*args)

        return _FakeFunc(_record)


def _api_calls(dll, name):
    return [tuple(a.value if hasattr(a, "value") else a for a in args) for api, args in dll.calls if api == name]


def _hccl_init_impl(fail_devices=()):
    state = {"next": 0x10000}

    def impl(n, devs, comms):
        dev_list = list(devs)
        if any(d in fail_devices for d in dev_list):
            return 1
        for i in range(len(dev_list)):
            comms[i] = state["next"]
            state["next"] += 1
        return 0

    return impl


def _acl_impls(fail_create_context_on=None):
    state = {"ctx": 0x20000, "stm": 0x30000}

    def create_context(pctx, dev):
        if dev.value == fail_create_context_on:
            return 1
        pctx[0] = state["ctx"]
        state["ctx"] += 1
        return 0

    def create_stream(pstm):
        pstm[0] = state["stm"]
        state["stm"] += 1
        return 0

    return {
        "aclrtResetDevice": lambda dev: 0,
        "aclrtSetDevice": lambda dev: 0,
        "aclrtCreateContext": create_context,
        "aclrtCreateStream": create_stream,
        "aclrtSynchronizeStream": lambda stm: 0,
        "aclrtDestroyStream": lambda stm: 0,
        "aclrtDestroyContext": lambda ctx: 0,
    }


def _scope(context, device_ids, acl_impls=None, hccl_impl=None):
    acl = _FakeDll(acl_impls or _acl_impls())
    hccl = _FakeDll({"HcclCommInitAll": hccl_impl or _hccl_init_impl(), "HcclCommDestroy": lambda ch: 0})
    scope = HcclResourceScope(context, device_ids, acl_dll=acl, hccl_dll=hccl)
    return scope, acl, hccl


def test_scope_exit_releases_full_state():
    """正常退出（未 retain）：comm/stream/context/设备全量释放。"""
    scope, acl, hccl = _scope(type("C", (), {"attributes": None})(), [0, 1])
    with scope:
        assert len(scope.comm_handles) == 2
        assert len(scope.ctx_vals) == 2
        assert len(scope.stm_vals) == 2
    destroyed = [c[0] for c in _api_calls(hccl, "HcclCommDestroy")]
    assert sorted(destroyed) == sorted(scope.comm_handles)
    assert _api_calls(acl, "aclrtDestroyStream") == [(s,) for s in scope.stm_vals]
    assert _api_calls(acl, "aclrtDestroyContext") == [(c,) for c in scope.ctx_vals]
    # __enter__ 预复位 2 次 + __exit__ 清理复位 2 次
    assert sorted(_api_calls(acl, "aclrtResetDevice")) == [(0,), (0,), (1,), (1,)]


def test_scope_retain_stashes_handles_and_skips_cleanup():
    """retain()：句柄转存 context 供 golden 阶段，__exit__ 不释放。"""
    context = type("C", (), {"attributes": None})()
    scope, acl, hccl = _scope(context, [0, 1])
    with scope:
        scope.retain()
    assert context._multi_device_hccl_handles == {
        "comm_handles": scope.comm_handles,
        "ctx_vals": scope.ctx_vals,
        "stm_vals": scope.stm_vals,
        "extra_comms": scope.extra_comms,
    }
    assert _api_calls(hccl, "HcclCommDestroy") == []
    # 仅 __enter__ 预复位，无 __exit__ 清理复位（retain 跳过释放）
    assert _api_calls(acl, "aclrtResetDevice") == [(0,), (1,)]


def test_scope_setup_failure_cleans_partial_state():
    """context 创建中途失败：世界域 comm 全销毁、已建 stream/context 销毁、设备全复位。"""
    scope, acl, hccl = _scope(type("C", (), {"attributes": None})(), [0, 1], _acl_impls(fail_create_context_on=1))
    with pytest.raises(RuntimeError, match="aclrtCreateContext"), scope:
        raise AssertionError("unreachable")
    # 世界域 comm 创建早于 context：两个句柄都必须销毁
    destroyed = [c[0] for c in _api_calls(hccl, "HcclCommDestroy")]
    assert sorted(destroyed) == sorted(scope.comm_handles)
    # device 0 已建 context/stream，device 1 失败：只销毁已建的
    assert _api_calls(acl, "aclrtDestroyContext") == [(scope.ctx_vals[0],)]
    assert _api_calls(acl, "aclrtDestroyStream") == [(scope.stm_vals[0],)]
    # __enter__ 预复位 2 次 + __exit__ 清理复位 2 次（清理不因部分状态而漏设备）
    assert sorted(_api_calls(acl, "aclrtResetDevice")) == [(0,), (0,), (1,), (1,)]


def test_world_comm_failure_propagates_without_partial_destroy():
    """世界域初始化失败：异常透传，无已建 comm 可清理。"""
    scope, acl, hccl = _scope(
        type("C", (), {"attributes": None})(), [0, 1], hccl_impl=_hccl_init_impl(fail_devices={0})
    )
    with pytest.raises(RuntimeError, match="HcclCommInitAll"), scope:
        raise AssertionError("unreachable")
    assert scope.comm_handles == []
    assert _api_calls(hccl, "HcclCommDestroy") == []


def test_dual_comm_subgroups_map_grid_ranks_to_device_ids():
    """非 0 基设备号：子域 HcclCommInitAll 收到真实设备号（网格索引 bug 修复）。"""
    context = type("C", (), {"attributes": {"epWorldSize": 2, "tpWorldSize": 2}})()
    init_calls = []

    def impl(n, devs, comms):
        init_calls.append(list(devs))
        for i in range(len(list(devs))):
            comms[i] = 0x1000 + len(init_calls) * 0x10 + i
        return 0

    scope, acl, hccl = _scope(context, [4, 5, 6, 7], hccl_impl=impl)
    with scope:
        pass
    # 世界域 + EP 子域（跨 TP 步长 [4,6]/[5,7]）+ TP 子域（连续 [4,5]/[6,7]）
    assert init_calls == [[4, 5, 6, 7], [4, 6], [5, 7], [4, 5], [6, 7]]
    # ep/tp map 以设备号为键，线程可按 rank_id（真实设备号）取句柄
    assert set(scope.ep_comm_map) == {4, 5, 6, 7}
    assert set(scope.tp_comm_map) == {4, 5, 6, 7}
    # 8 个子域句柄全部登记进 extra_comms 待清理
    assert len(scope.extra_comms) == 8


def test_full_world_dual_comm_reuses_primary_handles():
    """ep/tp 等于全量：复用世界域句柄，不重复初始化（Ascend950 挂起规避）。"""
    context = type("C", (), {"attributes": {"epWorldSize": 4, "tpWorldSize": 4}})()
    init_calls = []

    def impl(n, devs, comms):
        init_calls.append(list(devs))
        for i in range(len(list(devs))):
            comms[i] = 0x1000 + i
        return 0

    scope, acl, hccl = _scope(context, [0, 1, 2, 3], hccl_impl=impl)
    with scope:
        pass
    assert init_calls == [[0, 1, 2, 3]]
    assert scope.ep_comm_map == dict(zip([0, 1, 2, 3], scope.comm_handles))
    assert scope.tp_comm_map == dict(zip([0, 1, 2, 3], scope.comm_handles))
    assert scope.extra_comms == []
