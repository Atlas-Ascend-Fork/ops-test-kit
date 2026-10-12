# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ensure_api_so_path 通用 SO 路径解析：有效值保留 + 空悬回退候选探测。

原候选探测内嵌多卡线程函数（仅多卡路径可用），收敛为 info_keeper 通用
能力后单卡（runtime._get_op_api）与多卡共用。此处 pin 解析语义：
- 有效 so_path（文件存在）一律保留，不被候选探测覆盖；
- 空/悬空回退 libopapi_transformer.so → libopapi.so，并回写 info 缓存；
- CANN 场景不可解析（无 ASCEND_OPP_PATH）时静默返回，不抛。
全部用捏造环境，不依赖 CANN 安装。
"""

from collections import OrderedDict

import pytest

from ttk.core_modules.aclnn import info_keeper
from ttk.core_modules.aclnn.info_keeper import OpApiInfo, OpApiInfoKeeper, ensure_api_so_path
from ttk.utilities.singleton import Singleton

_KEY = (OpApiInfoKeeper, (), ())
_API = "aclnnFakeComm"


@pytest.fixture
def _clean_singleton():
    saved = dict(Singleton._instances)
    yield
    Singleton._instances.clear()
    Singleton._instances.update(saved)


def _install_info(so_path="", category="custom"):
    params = OrderedDict([("x", {"type": "aclTensor*", "default": None})])
    info = OpApiInfo(params)
    info._api_name = _API
    info.category = category
    info._so_path = so_path
    keeper = OpApiInfoKeeper.__new__(OpApiInfoKeeper)
    keeper._hdr_dirs = []
    keeper._api_info = {_API: info}
    Singleton._instances[_KEY] = keeper
    return info


def _fake_lib64(monkeypatch, tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    monkeypatch.setattr(info_keeper, "get_ascend_lib64_path", lambda: str(tmp_path))


@pytest.mark.usefixtures("_clean_singleton")
def test_valid_so_path_is_kept(monkeypatch, tmp_path):
    """头文件派生/符号扫描得到的有效路径不被候选探测覆盖（x86 修复语义）。"""
    existing = tmp_path / "libcust_opapi.so"
    existing.touch()
    info = _install_info(so_path=str(existing))
    _fake_lib64(monkeypatch, tmp_path, ("libopapi.so", "libopapi_transformer.so"))

    assert ensure_api_so_path(_API) == str(existing)
    assert info._so_path == str(existing)


@pytest.mark.usefixtures("_clean_singleton")
def test_empty_builtin_falls_back_transformer_first(monkeypatch, tmp_path):
    """builtin 符号扫描未命中 → libopapi_transformer.so 优先，并回写缓存。"""
    info = _install_info(so_path="", category="builtin")
    monkeypatch.setattr(info_keeper, "_builtin_api_so_map", {})
    _fake_lib64(monkeypatch, tmp_path, ("libopapi.so", "libopapi_transformer.so"))

    resolved = ensure_api_so_path(_API)

    assert resolved == str(tmp_path / "libopapi_transformer.so")
    assert info._so_path == resolved


@pytest.mark.usefixtures("_clean_singleton")
def test_fallback_to_base_libopapi(monkeypatch, tmp_path):
    """无 transformer 变体 → 退回 libopapi.so。"""
    _install_info(so_path="", category="builtin")
    monkeypatch.setattr(info_keeper, "_builtin_api_so_map", {})
    _fake_lib64(monkeypatch, tmp_path, ("libopapi.so",))

    assert ensure_api_so_path(_API) == str(tmp_path / "libopapi.so")


@pytest.mark.usefixtures("_clean_singleton")
def test_dangling_path_is_replaced(monkeypatch, tmp_path):
    """so_path 指向不存在文件 → 视为无效，回退候选。"""
    info = _install_info(so_path=str(tmp_path / "missing.so"))
    _fake_lib64(monkeypatch, tmp_path, ("libopapi.so",))

    assert ensure_api_so_path(_API) == str(tmp_path / "libopapi.so")
    assert info._so_path == str(tmp_path / "libopapi.so")


@pytest.mark.usefixtures("_clean_singleton")
def test_no_candidates_keeps_current(monkeypatch, tmp_path):
    """无候选可探测 → 返回原值，不抛不写。"""
    dangling = str(tmp_path / "missing.so")
    _install_info(so_path=dangling)
    _fake_lib64(monkeypatch, tmp_path, ())

    assert ensure_api_so_path(_API) == dangling


@pytest.mark.usefixtures("_clean_singleton")
def test_unknown_api_returns_empty(monkeypatch, tmp_path):
    _fake_lib64(monkeypatch, tmp_path, ("libopapi.so",))
    keeper = OpApiInfoKeeper.__new__(OpApiInfoKeeper)
    keeper._hdr_dirs = []
    keeper._api_info = {}
    Singleton._instances[_KEY] = keeper

    assert ensure_api_so_path("aclnnUnknown") == ""


@pytest.mark.usefixtures("_clean_singleton")
def test_unresolvable_scene_during_symbol_scan_returns_empty(monkeypatch):
    """ASCEND_OPP_PATH 缺失：符号扫描内部 get_ascend_lib64_path 抛 → 返回空。"""
    _install_info(so_path="", category="builtin")
    monkeypatch.setattr(info_keeper, "_builtin_api_so_map", None)

    def _raise():
        raise RuntimeError("Environment `ASCEND_OPP_PATH` is not set.")

    monkeypatch.setattr(info_keeper, "get_ascend_lib64_path", _raise)

    assert ensure_api_so_path(_API) == ""


@pytest.mark.usefixtures("_clean_singleton")
def test_unresolvable_scene_during_fallback_returns_current(monkeypatch, tmp_path):
    """ASCEND_OPP_PATH 缺失：候选探测阶段 get_ascend_lib64_path 抛 → 返回原值。"""
    dangling = str(tmp_path / "missing.so")
    info = _install_info(so_path=dangling)
    monkeypatch.setattr(info_keeper, "_builtin_api_so_map", {})

    def _raise():
        raise RuntimeError("Environment `ASCEND_OPP_PATH` is not set.")

    monkeypatch.setattr(info_keeper, "get_ascend_lib64_path", _raise)

    assert ensure_api_so_path(_API) == dangling
    assert info._so_path == dangling
