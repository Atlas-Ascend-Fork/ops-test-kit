# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AclModelInterface 单测（mock CDLL，无 Ascend 环境可跑）。

覆盖：懒加载语义（import 零 .so 触碰）、set_device 幂等与切卡、错误码抛错、
账本登记/释放、模型 IO 信息快照、execute 的 dataset 组装序。
"""

import ctypes
import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from ttk.core_modules.framework_api.onnx.runtime import AclModelInterface, MdlIoDesc


def _fake_dll(**overrides):
    """MagicMock DLL：默认全部成功（返回 0 / 非空指针 / 计数）。"""
    dll = mock.MagicMock()
    dll.aclInit.return_value = 0
    dll.aclFinalize.return_value = 0
    dll.aclrtSetDevice.return_value = 0
    dll.aclrtResetDevice.return_value = 0
    dll.aclrtGetDeviceCount.return_value = 0
    dll.aclrtMalloc.return_value = 0
    dll.aclrtFree.return_value = 0
    dll.aclrtMemcpy.return_value = 0
    dll.aclmdlLoadFromFile.return_value = 0
    dll.aclmdlUnload.return_value = 0
    dll.aclmdlCreateDesc.return_value = 0x1000
    dll.aclmdlDestroyDesc.return_value = 0
    dll.aclmdlGetDesc.return_value = 0
    dll.aclmdlGetNumInputs.return_value = 1
    dll.aclmdlGetNumOutputs.return_value = 1
    dll.aclmdlGetInputSizeByIndex.return_value = 24
    dll.aclmdlGetOutputSizeByIndex.return_value = 24
    dll.aclmdlGetInputDataType.return_value = 0
    dll.aclmdlGetOutputDataType.return_value = 0
    dll.aclmdlGetInputFormat.return_value = 2
    dll.aclmdlGetOutputFormat.return_value = 2
    dll.aclmdlCreateDataset.return_value = 0x2000
    dll.aclmdlDestroyDataset.return_value = 0
    dll.aclmdlAddDatasetBuffer.return_value = 0
    dll.aclCreateDataBuffer.return_value = 0x3000
    dll.aclDestroyDataBuffer.return_value = 0
    dll.aclmdlExecute.return_value = 0

    def _load(_path, out):
        out._obj.value = 7
        return 0

    dll.aclmdlLoadFromFile.side_effect = _load

    def _malloc(out, _size, _policy):
        out._obj.value = 0xDEADB000
        return 0

    dll.aclrtMalloc.side_effect = _malloc

    def _get_count(out):
        out._obj.value = 2
        return 0

    dll.aclrtGetDeviceCount.side_effect = _get_count

    def _fill_dims(_desc, _idx, dims):
        # byref 产物是 CArgObject：经 _obj 触达目标结构体
        target = dims._obj
        target.name = b"input_0"
        target.dimCount = 2
        target.dims[0] = 2
        target.dims[1] = 3
        return 0

    dll.aclmdlGetInputDims.side_effect = _fill_dims
    dll.aclmdlGetOutputDims.side_effect = _fill_dims
    dll.aclmdlGetInputNameByIndex.return_value = b"input_0"
    dll.aclmdlGetOutputNameByIndex.return_value = b"add_0:0:output_0"

    for name, value in overrides.items():
        setattr(dll, name, value)
    return dll


@pytest.fixture
def iface():
    """真实构造（注册 atexit）但 DLL 指向 fake 的实例。

    类级 _DLL 打补丁（_dll() classmethod 读类属性，实例属性不生效）；
    teardown 前清空账本并复位补丁，防 atexit 触碰真实 .so。
    """
    prev = AclModelInterface._DLL
    AclModelInterface._DLL = _fake_dll()
    inst = AclModelInterface(skip_teardown=True)
    try:
        yield inst
    finally:
        inst._models.clear()
        inst._dev_ptrs.clear()
        inst._acl_inited = False
        AclModelInterface._DLL = prev


# ── 懒加载 ─────────────────────────────────────────────────────────────────


def test_module_import_never_loads_so():
    """模块 import 零 .so 触碰：类级 _DLL 保持 None。

    子进程剥离 CANN 环境条目（ASCEND_* 变量与 LD_LIBRARY_PATH/PYTHONPATH
    中的 Ascend 路径，模拟无 CANN 容器）后仍可安全 import——若 import 期
    触碰 libascendcl.so 将因找不到 .so 直接失败。LD_LIBRARY_PATH 的非 CANN
    条目（Python 自身 lib 目录）必须保留，否则解释器连 numpy 都载不动。
    """
    code = (
        "from ttk.core_modules.framework_api.onnx import runtime;"
        "assert runtime.AclModelInterface._DLL is None, 'DLL loaded at import time'"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASCEND_")}
    for var in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        value = env.get(var)
        if value is None:
            continue
        kept = [p for p in value.split(os.pathsep) if p and "ascend" not in p.lower() and "cann" not in p.lower()]
        if kept:
            env[var] = os.pathsep.join(kept)
        else:
            del env[var]
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr


# ── 运行时 / 设备 / 账本 ────────────────────────────────────────────────────


def test_set_device_semantics(iface):
    """幂等 + 切卡重新 SetDevice + 错误码带 API 名抛错。"""
    iface.set_device(0)
    assert iface.device_id == 0
    assert iface._dll().aclrtSetDevice.call_count == 1
    iface.set_device(0)
    assert iface._dll().aclrtSetDevice.call_count == 1  # 幂等
    iface.set_device(1)
    assert iface.device_id == 1
    assert iface._dll().aclrtSetDevice.call_count == 2  # 切卡重新 SetDevice
    iface._dll().aclrtSetDevice.return_value = 100000
    with pytest.raises(RuntimeError, match="aclrtSetDevice.*100000"):
        iface.set_device(2)


def test_memory_and_model_ledgers(iface):
    """H2D/D2H 指针账本与模型账本：登记/释放/账本外幂等。"""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    ptr = iface.h2d(arr)
    assert ptr.value == 0xDEADB000
    assert 0xDEADB000 in iface._dev_ptrs
    got = iface.d2h(ptr, arr.nbytes)
    assert got.shape == (24,)
    assert got.dtype == np.uint8
    iface.free(ptr)
    assert 0xDEADB000 not in iface._dev_ptrs
    iface.free(ptr)  # 双重 free 幂等（账本外无操作）

    model_id = iface.load_model("/tmp/fake.om")
    assert model_id == 7
    assert 7 in iface._models
    iface.unload_model(model_id)
    assert 7 not in iface._models
    iface.unload_model(model_id)  # 账本外幂等
    assert iface._dll().aclmdlUnload.call_count == 1


def test_ledger_retained_on_release_failure(iface):
    """释放失败保留账本条目（供 reset() 兜底重试），成功后才摘除。"""
    model_id = iface.load_model("/tmp/fake.om")
    iface._dll().aclmdlUnload.return_value = 100000
    with pytest.raises(RuntimeError, match="aclmdlUnload"):
        iface.unload_model(model_id)
    assert model_id in iface._models  # 失败不摘账本
    iface._dll().aclmdlUnload.return_value = 0
    iface.unload_model(model_id)
    assert model_id not in iface._models

    ptr = iface.h2d(np.arange(3, dtype=np.float32))
    iface._dll().aclrtFree.return_value = 100000
    with pytest.raises(RuntimeError, match="aclrtFree"):
        iface.free(ptr)
    assert ptr.value in iface._dev_ptrs  # 失败不摘账本
    iface._dll().aclrtFree.return_value = 0
    iface.free(ptr)
    assert ptr.value not in iface._dev_ptrs


def test_model_io_info_snapshot(iface):
    info = iface.model_io_info(7)
    assert isinstance(info["inputs"][0], MdlIoDesc)
    assert info["inputs"][0].name == "input_0"
    assert info["inputs"][0].dims == (2, 3)
    assert info["inputs"][0].size == 24
    # 输出名被 atc 改写：按索引消费、不按名字匹配
    assert info["outputs"][0].name == "add_0:0:output_0"
    assert info["outputs"][0].format_id == 2
    iface._dll().aclmdlDestroyDesc.assert_called_once()


# ── 执行 ───────────────────────────────────────────────────────────────────


def test_execute_dataset_lifecycle(iface):
    ptr = ctypes.c_void_p(0xDEADB000)
    iface.execute(7, [(ptr, 24)], [(ptr, 32)])
    dll = iface._dll()
    assert dll.aclmdlCreateDataset.call_count == 2
    assert dll.aclCreateDataBuffer.call_count == 2
    assert dll.aclmdlAddDatasetBuffer.call_count == 2
    dll.aclmdlExecute.assert_called_once()
    # dataset/buffer 不缓存：每次执行后销毁
    assert dll.aclmdlDestroyDataset.call_count == 2
    assert dll.aclDestroyDataBuffer.call_count == 2
