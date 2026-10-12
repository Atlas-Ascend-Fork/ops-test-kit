# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""OpApiInfoKeeper 缓存快照随行：snapshot → pickle → restore 闭环。

forkserver worker 不继承主进程内存；keeper 纯数据缓存经引导载荷传递，
worker 内 restore 后查询全为命中。此处用捏造快照验证恢复语义（不依赖
CANN 头文件环境）。
"""

import pickle
from collections import OrderedDict

import pytest

from ttk.core_modules.aclnn.info_keeper import (
    OpApiInfo,
    OpApiInfoKeeper,
    restore_keeper_snapshot,
    snapshot_keeper,
)
from ttk.utilities.singleton import Singleton

_KEY = (OpApiInfoKeeper, (), ())


def _fake_snapshot():
    params = OrderedDict(
        [
            ("x", {"type": "aclTensor*", "default": None}),
            ("alpha", {"type": "aclScalar*", "default": None}),
            ("out", {"type": "aclTensor*", "default": None}),
        ]
    )
    info = OpApiInfo(params)
    info._api_name = "aclnnFakeAdd"
    info.category = "builtin"
    return {"api_info": {"aclnnFakeAdd": info}}


@pytest.fixture
def _clean_singleton():
    saved = dict(Singleton._instances)
    yield
    Singleton._instances.clear()
    Singleton._instances.update(saved)


@pytest.mark.usefixtures("_clean_singleton")
def test_restore_answers_queries_from_snapshot():
    snapshot = _fake_snapshot()
    restore_keeper_snapshot(snapshot)

    keeper = OpApiInfoKeeper()
    assert keeper.has_api("aclnnFakeAdd")
    info = keeper.info_of("aclnnFakeAdd")
    assert info.tensors == ["x", "out"]
    assert info.scalars == ["alpha"]
    assert keeper.info_of("aclnnUnknown") is None


@pytest.mark.usefixtures("_clean_singleton")
def test_snapshot_roundtrip_through_pickle():
    snapshot = _fake_snapshot()
    payload = pickle.loads(pickle.dumps(snapshot))
    restore_keeper_snapshot(payload)

    assert OpApiInfoKeeper().info_of("aclnnFakeAdd") is not None


@pytest.mark.usefixtures("_clean_singleton")
def test_snapshot_keeper_returns_parsed_cache(monkeypatch):
    """snapshot_keeper 在缓存未填时先触发解析，再产出快照。"""
    parsed = _fake_snapshot()["api_info"]
    monkeypatch.setattr(OpApiInfoKeeper, "_parse_op_api_header_files", staticmethod(lambda: parsed), raising=False)
    # 构造一个不触头文件目录的实例（绕过 __init__ 的目录检查）
    keeper = OpApiInfoKeeper.__new__(OpApiInfoKeeper)
    keeper._hdr_dirs = []
    keeper._api_info = None
    Singleton._instances[_KEY] = keeper

    snap = snapshot_keeper()
    assert snap == {"api_info": parsed}


def test_worker_bootstrap_restores_by_module_path():
    """worker_bootstrap 的恢复通道：按模块路径数据驱动恢复，不 import 通路。"""
    from ttk.core_modules.common.process.simple_command import _restore_keeper_snapshots

    saved = dict(Singleton._instances)
    try:
        snapshot = _fake_snapshot()
        _restore_keeper_snapshots({"ttk.core_modules.aclnn.info_keeper": snapshot})
        assert OpApiInfoKeeper().has_api("aclnnFakeAdd")
    finally:
        Singleton._instances.clear()
        Singleton._instances.update(saved)


def test_worker_bootstrap_degrades_on_bad_payload():
    """恢复失败降级为日志，不抛（worker 惰性重解析兜底）。"""
    from ttk.core_modules.common.process.simple_command import _restore_keeper_snapshots

    _restore_keeper_snapshots({"ttk.core_modules.aclnn.info_keeper": {}})
    _restore_keeper_snapshots({"ttk.core_modules.no.such.module": {"x": 1}})
