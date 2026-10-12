# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""OpInfoKeeper 缓存快照随行：snapshot → pickle → restore 闭环。

forkserver worker 不继承主进程内存；全量 op_info 扫描产物（ops_info）经
引导载荷传递，worker 内 restore 后查询命中、不再重扫。此处用捏造快照验证
恢复语义（不依赖 CANN 环境与真实 op_info 文件）。
"""

import pickle

import pytest

from ttk.core_modules.kernel.info_keeper import (
    OpInfoKeeper,
    restore_keeper_snapshot,
    snapshot_keeper,
)
from ttk.utilities.singleton import Singleton

_KEY = (OpInfoKeeper, (), ())


def _fake_ops_info():
    return {
        "fake_add": {
            "op_type": "FakeAdd",
            "coreType.value": "AiCore",
            "_impl_base_path": "/impl/fake",
            "_ops_group": "fake_group",
        }
    }


@pytest.fixture
def _clean_singleton():
    saved = dict(Singleton._instances)
    yield
    Singleton._instances.clear()
    Singleton._instances.update(saved)


@pytest.mark.usefixtures("_clean_singleton")
def test_restore_answers_queries_from_snapshot():
    """恢复后查询全命中——不触发 _load_op_info_cfg 重扫。"""
    restore_keeper_snapshot({"ops_info": _fake_ops_info()})

    keeper = OpInfoKeeper()
    # 重扫路径封死：任何触发 _load 的查询都会炸
    keeper._load_op_info_cfg = lambda: (_ for _ in ()).throw(AssertionError("rescan triggered"))
    assert keeper.info_of("fake_add")["op_type"] == "FakeAdd"
    assert keeper.op_type_of("fake_add") == "FakeAdd"
    assert keeper.info_of("no_such_op") is None
    # alias_index 从恢复数据派生（CamelCase 别名命中）
    assert keeper.info_of("FakeAdd")["op_type"] == "FakeAdd"


@pytest.mark.usefixtures("_clean_singleton")
def test_snapshot_roundtrip_through_pickle():
    snapshot = {"ops_info": _fake_ops_info()}
    payload = pickle.loads(pickle.dumps(snapshot))
    restore_keeper_snapshot(payload)

    assert OpInfoKeeper().info_of("fake_add") is not None


@pytest.mark.usefixtures("_clean_singleton")
def test_snapshot_keeper_triggers_lazy_parse():
    """snapshot_keeper 在缓存未填时先触发解析，再产出快照。"""
    keeper = OpInfoKeeper.__new__(OpInfoKeeper)
    keeper._ops_info = None
    keeper._alias_index = None
    keeper._op_func_cache = {}
    keeper._kernel_info_cache = {}
    keeper._binary_static_cache = {}
    keeper._binary_info_cache = {}
    keeper._load_op_info_cfg = lambda: _fake_ops_info()
    Singleton._instances[_KEY] = keeper

    snap = snapshot_keeper()
    assert snap == {"ops_info": _fake_ops_info()}


@pytest.mark.usefixtures("_clean_singleton")
def test_restore_keeps_per_op_caches_empty():
    """逐算子缓存不随行：恢复后为空、按需重建。"""
    restore_keeper_snapshot({"ops_info": _fake_ops_info()})
    keeper = OpInfoKeeper()
    assert keeper._op_func_cache == {}
    assert keeper._kernel_info_cache == {}
    # 按需重建走恢复的 ops_info
    keeper._load_op_info_cfg = lambda: (_ for _ in ()).throw(AssertionError("rescan triggered"))
    info = keeper.kernel_info_of("fake_add")
    assert info["ops_group"] == "fake_group"
    assert keeper._kernel_info_cache["fake_add"] is info


def test_worker_bootstrap_restores_by_module_path():
    """worker_bootstrap 的恢复通道：按模块路径数据驱动恢复。"""
    from ttk.core_modules.common.process.simple_command import _restore_keeper_snapshots

    saved = dict(Singleton._instances)
    try:
        _restore_keeper_snapshots({"ttk.core_modules.kernel.info_keeper": {"ops_info": _fake_ops_info()}})
        assert OpInfoKeeper().info_of("fake_add") is not None
    finally:
        Singleton._instances.clear()
        Singleton._instances.update(saved)


def test_restore_degrades_on_empty_payload():
    """空载荷/坏载荷不抛（worker 惰性重解析兜底）。"""
    restore_keeper_snapshot({})
    restore_keeper_snapshot(None)
    restore_keeper_snapshot({"unexpected": 1})
