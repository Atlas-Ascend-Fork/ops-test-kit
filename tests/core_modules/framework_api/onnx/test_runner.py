# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""OmSession 单测（FakeAcl，无 Ascend 环境可跑）。

会话生命周期与输出哨兵预填（假 PASS 防御）、动态输出守卫、会话复用判定
（身份/内容双层）。
"""

import ctypes

import numpy as np
import pytest

from ttk.core_modules.framework_api.onnx.runner import OmSession
from ttk.core_modules.framework_api.onnx.runtime import MdlIoDesc


class FakeAcl:
    """OmSession 消费面的最小 fake（计数 + 可编程 desc）。"""

    def __init__(self, input_descs, output_descs):
        self.device_id = None
        self.input_descs = input_descs
        self.output_descs = output_descs
        self.set_device_calls = []
        self.loaded = []
        self.unloaded = []
        self.mallocs = []
        self.frees = []
        self.memsets = []  # (ptr_value, value, count)
        self.h2ds = []  # (ptr_value, nbytes)
        self.executes = []
        self.host_writes = {}  # ptr_value -> np.ndarray（d2h 数据源）

    def set_device(self, dev_id):
        self.set_device_calls.append(dev_id)
        self.device_id = dev_id

    def load_model(self, om_path):
        self.loaded.append(om_path)
        return 7

    def unload_model(self, model_id):
        self.unloaded.append(model_id)

    def model_io_info(self, model_id):
        return {"inputs": list(self.input_descs), "outputs": list(self.output_descs)}

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p(0x1000 + 0x100 * len(self.mallocs))
        self.mallocs.append((ptr.value, nbytes))
        self.host_writes[ptr.value] = np.full(nbytes, 0x7F, np.uint8)  # 哨兵初值
        return ptr

    def free(self, ptr):
        value = ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr
        self.frees.append(value)

    def memset(self, ptr, value, count):
        self.memsets.append((ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr, value, count))
        self.host_writes[ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr][:] = value

    def h2d(self, arr, size=None):
        ptr = self.malloc(size or arr.nbytes)
        self.host_writes[ptr.value][: arr.nbytes] = np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8)
        self.h2ds.append((ptr.value, arr.nbytes))
        return ptr

    def d2h(self, ptr, nbytes):
        value = ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr
        return self.host_writes[value][:nbytes].copy()

    def execute(self, model_id, input_bufs, output_bufs):
        self.executes.append((model_id, len(input_bufs), len(output_bufs)))
        # 模拟 kernel 写输出：f(x)=x+1 逐字节不可行——用首输入字节 +1 覆盖首输出逻辑区
        in_ptr, _in_size = input_bufs[0]
        src = self.host_writes[in_ptr.value]
        out_ptr, out_size = output_bufs[0]
        self.host_writes[out_ptr.value][: min(out_size, len(src))] = src[: min(out_size, len(src))]


def _desc(name, dims, dtype_id, size, format_id=2):
    return MdlIoDesc(name=name, dims=dims, dtype_id=dtype_id, size=size, format_id=format_id)


@pytest.fixture
def acl():
    return FakeAcl(
        input_descs=[_desc("input_0", (2, 3), 0, 24)],
        output_descs=[_desc("add_0:0:output_0", (2, 3), 0, 24)],
    )


def test_session_lifecycle_and_sentinel_prefill(acl):
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    sess = OmSession(acl, 0, "/tmp/fake.om", [arr], "Add")
    assert acl.set_device_calls == [0]
    assert acl.loaded == ["/tmp/fake.om"]
    # 输出缓冲：malloc + 0x7F 哨兵预填（假 PASS 防御：漏写 kernel 读到确定性模式）
    assert len(acl.memsets) == 1
    ptr_value, fill_value, count = acl.memsets[0]
    assert fill_value == 0x7F
    assert count == 24
    # 输入 H2D：按 desc size 分配、按实际字节拷贝
    assert acl.h2ds == [(ptr_value - 0x100, 24)]
    out = sess.execute_round()
    assert len(acl.executes) == 1
    assert out[0].shape == (2, 3)
    sess.unload()
    assert acl.unloaded == [7]
    assert len(acl.frees) == 2  # 1 输入 + 1 输出


def test_guard_dynamic_dim_output(acl):
    acl.output_descs = [_desc("nz_0:0:output_0", (2, -1), 9, 96)]
    with pytest.raises(RuntimeError, match="ONNX_DYNAMIC_OUTPUT_UNSUPPORTED.*-1"):
        OmSession(acl, 0, "/tmp/fake.om", [np.zeros((2, 3), np.float32)], "NonZero")
    assert acl.unloaded == [7]  # 异常路径仍卸载模型


def test_acl_failure_prefixes_at_lifecycle_points(acl):
    """load/desc/execute 三处 ACL 异常统一加 OM_*_FAILED 前缀透传；异常路径仍卸载模型。"""
    orig_load, orig_io, orig_exec = acl.load_model, acl.model_io_info, acl.execute
    zeros = np.zeros((2, 3), np.float32)

    def load_boom(om_path):
        raise RuntimeError(f"Acl API Call aclmdlLoadFromFile() Failed: 107111, om={om_path}")

    acl.load_model = load_boom
    with pytest.raises(RuntimeError, match="OM_LOAD_FAILED: Acl API Call aclmdlLoadFromFile.*107111"):
        OmSession(acl, 0, "/tmp/fake.om", [zeros], "Add")

    def desc_boom(model_id):
        raise RuntimeError(f"Acl API Call aclmdlGetDesc() Failed: 107077, model_id={model_id}")

    acl.load_model, acl.model_io_info = orig_load, desc_boom
    with pytest.raises(RuntimeError, match="OM_LOAD_FAILED: Acl API Call aclmdlGetDesc.*107077"):
        OmSession(acl, 0, "/tmp/fake.om", [zeros], "Add")
    assert acl.unloaded == [7]  # desc 失败路径仍卸载模型（load 失败无 model_id，不卸载）

    acl.model_io_info = orig_io
    sess = OmSession(acl, 0, "/tmp/fake.om", [zeros], "Add")

    def execute_boom(model_id, input_bufs, output_bufs):
        raise RuntimeError(f"Acl API Call aclmdlExecute() Failed: 107001, model_id={model_id}")

    acl.execute = execute_boom
    with pytest.raises(RuntimeError, match="OM_EXECUTE_FAILED: Acl API Call aclmdlExecute.*107001"):
        sess.execute_round()
    acl.execute = orig_exec


def test_matches_inputs_identity_and_content():
    acl = FakeAcl(
        input_descs=[_desc("input_0", (2, 3), 0, 24), _desc("input_1", (2, 3), 0, 24)],
        output_descs=[_desc("out", (2, 3), 0, 24)],
    )
    a = np.zeros((2, 3), np.float32)
    b = np.ones((2, 3), np.float32)
    sess = OmSession(acl, 0, "/tmp/fake.om", [a, b], "Add")
    assert sess.matches_inputs([a, b]) is True  # 身份快路径
    assert sess.matches_inputs([a]) is False  # 在场数不同
    a2, b2 = a.copy(), b.copy()
    assert sess.matches_inputs([a2, b2]) is True  # 内容慢路径（同值新对象）
    b2[0, 0] = 9.0
    assert sess.matches_inputs([a2, b2]) is False  # 同签名不同值 → 驱逐重建
    sess.rebind_inputs([a2, b2])
    assert sess.matches_inputs([a2, b2]) is True  # 重绑后身份快路径恢复
