# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""XU round-trip 结构保真看护。嵌套组合 ((A,B), C, None, (D,E)) 两个方向。"""

import importlib.util
import io

import numpy as np
import pytest

from ttk.remote.dispatcher import (
    _build_input_schema,
    _load_npz_outputs,
    _serialize_to_file,
)
from ttk.remote.server.executor import _outputs_to_numpy

# torch_npu._C + tensorflow C extension 在同一进程 import 时符号冲突 → segfault。
_has_torch_npu = importlib.util.find_spec("torch_npu") is not None
_has_tf = importlib.util.find_spec("tensorflow") is not None


def _sample():
    A = np.array([1.0, 2.0])
    B = np.array([3.0, 4.0])
    C = np.array([5.0])
    D = np.array([6.0, 7.0])
    E = np.array([8.0, 9.0])
    return A, B, C, D, E


def test_input_direction_client_to_server(tmp_path):
    """方向 1: client 序列化 → server match_params_v1 恢复嵌套+None。"""
    from ttk.remote.server.executor import match_params_v1

    A, B, C, D, E = _sample()
    inputs = [(A, B), C, None, (D, E)]
    names = ["in0", "in1", "in2", "in3"]
    schema = _build_input_schema(inputs, names)
    path = _serialize_to_file(inputs, dir=str(tmp_path))
    npz = np.load(path)
    flat = [
        npz[f"a{i}"]
        for i in range(
            sum(len(e["indices"]) if "indices" in e else (1 if e.get("index") is not None else 0) for e in schema)
        )
    ]
    named = match_params_v1(schema, flat)
    assert isinstance(named["in0"], list)
    assert len(named["in0"]) == 2
    assert np.array_equal(named["in0"][0], A)
    assert np.array_equal(named["in0"][1], B)
    assert np.array_equal(named["in1"], C)
    assert named["in2"] is None
    assert isinstance(named["in3"], list)
    assert len(named["in3"]) == 2
    assert np.array_equal(named["in3"][0], D)
    assert np.array_equal(named["in3"][1], E)


def test_output_direction_server_to_client():
    """方向 2: server _outputs_to_numpy → npz → client _load_npz_outputs 嵌套保真（同 dtype）。"""
    A, B, C, D, E = _sample()
    raw_outputs = ((A, B), C, None, (D, E))
    schema, arrays = _outputs_to_numpy(raw_outputs, "numpy")
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": o for i, o in enumerate(arrays)})
    buf.seek(0)
    loaded = _load_npz_outputs(buf, schema)
    # loaded[0] = [A, B], loaded[1] = C, loaded[2] = None, loaded[3] = [D, E]
    assert isinstance(loaded[0], list)
    assert len(loaded[0]) == 2
    assert np.array_equal(loaded[0][0], A)
    assert np.array_equal(loaded[0][1], B)
    assert np.array_equal(loaded[1], C)
    assert loaded[2] is None
    assert isinstance(loaded[3], list)
    assert len(loaded[3]) == 2
    assert np.array_equal(loaded[3][0], D)
    assert np.array_equal(loaded[3][1], E)


def _complex32_storage():
    """fp16 [2,3,2] 交错存储（real=1, imag=2），等价 complex32 logical [2,3]。"""
    arr = np.empty((2, 3, 2), dtype=np.float16)
    arr[..., 0] = 1.0
    arr[..., 1] = 2.0
    return arr


def _custom_wire_samples():
    """npz 存不回的 dtype 样例：(dtype_name, arr)。e4m3fn/e5m2/int4 来自
    ml_dtypes，e8m0/hifloat8 来自 en_dtypes（未安装则跳过）。"""
    import ml_dtypes

    samples = [
        ("float8_e4m3fn", np.array([1.0, -2.0, 0.5, 3.0], dtype=ml_dtypes.float8_e4m3fn)),
        ("float8_e5m2", np.array([1.0, -2.0, 0.5, 3.0], dtype=ml_dtypes.float8_e5m2)),
        ("int4", np.array([-8, -1, 0, 7], dtype=ml_dtypes.int4)),
    ]
    try:
        import en_dtypes

        samples.extend(
            [
                ("float8_e8m0", np.array([1.0, 2.0, 4.0], dtype=en_dtypes.float8_e8m0)),
                ("hifloat8", np.array([0.5, -1.0, 2.0], dtype=en_dtypes.hifloat8)),
            ]
        )
    except ImportError:
        pass
    return samples


def test_custom_wire_dtype_input_direction(tmp_path):
    """自定义 wire dtype 输入方向：schema 声明元数据 → server 还原逻辑 dtype。

    complex32 以 fp16 交错位传输、还原 torch.complex32 逻辑张量；float8 家族
    wire 只传原始字节（npz 存不下其 dtype），server 按声明重铸——torch 有原生
    存储的 e4m3fn/e5m2 转 torch 张量，e8m0/hifloat8 与 int4 numpy 直通。
    所有路径均须与 wire 字节逐位一致。
    """
    torch = pytest.importorskip("torch")
    from ttk.remote.server.executor import _to_vendor_tensor, match_params_v1

    # complex32
    storage = _complex32_storage()
    schema = _build_input_schema(inputs=[storage], input_names=["x"], input_dtypes=["complex32"])
    assert schema[0]["dtype"] == "float16"  # 物理 dtype
    assert schema[0]["logical_dtype"] == "complex32"
    path = _serialize_to_file([storage], dir=str(tmp_path))
    npz = np.load(path)
    named = match_params_v1(schema, [npz["a0"]])
    t = _to_vendor_tensor(named["x"], "torch", "cpu", schema[0]["dtype"], schema[0]["logical_dtype"])
    assert t.dtype == torch.complex32
    assert tuple(t.shape) == (2, 3)  # 尾维 [2] 被吸收为复数
    ref = t.contiguous().view(torch.float16).reshape(2, 3, 2).numpy()
    assert np.all(ref[..., 0] == 1.0)
    assert np.all(ref[..., 1] == 2.0)
    # 无 logical_dtype（旧客户端）：保持 fp16 [..., 2] 原样（向后兼容）
    t_plain = _to_vendor_tensor(named["x"], "torch", "cpu", schema[0]["dtype"], None)
    assert t_plain.dtype == torch.float16
    assert tuple(t_plain.shape) == (2, 3, 2)

    # float8 家族 / int4：wire 原始字节 + schema dtype → server 重铸
    for name, arr in _custom_wire_samples():
        schema = _build_input_schema(inputs=[arr], input_names=["x"])
        assert schema[0]["dtype"] == name  # 元数据走 schema，不随数据
        path = _serialize_to_file([arr], dir=str(tmp_path))
        wire = np.load(path)["a0"]
        named = match_params_v1(schema, [wire])
        out = _to_vendor_tensor(named["x"], "torch", "cpu", schema[0]["dtype"], None)
        if name in ("float8_e4m3fn", "float8_e5m2"):
            assert isinstance(out, torch.Tensor)
            assert out.dtype == getattr(torch, name)
            got = out.view(torch.uint8).cpu().numpy()
        else:
            assert isinstance(out, np.ndarray)
            assert out.dtype.name == name
            got = np.ascontiguousarray(out).view(np.uint8)
        np.testing.assert_array_equal(got, wire.view(np.uint8))


def test_complex32_output_direction():
    """complex32 输出方向：server 转 fp16 [...,2] 交错布局回传（与 golden/NPU 输出对齐）。"""
    torch = pytest.importorskip("torch")
    from ttk.remote.server.executor import _to_numpy_pair

    c = torch.complex(torch.ones(2, 3), torch.full((2, 3), 2.0)).to(torch.complex32)
    arr, dt = _to_numpy_pair(c, "torch")
    assert dt == "complex32"
    assert arr.dtype == np.float16
    assert arr.shape == (2, 3, 2)
    assert np.all(arr[..., 0] == 1.0)
    assert np.all(arr[..., 1] == 2.0)
    # 端到端：_outputs_to_numpy → npz → _load_npz_outputs 布局保真
    schema, arrays = _outputs_to_numpy([c], "torch")
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": o for i, o in enumerate(arrays)})
    buf.seek(0)
    loaded = _load_npz_outputs(buf, schema)
    assert loaded[0].shape == (2, 3, 2)
    assert loaded[0].dtype == np.float16
    assert np.all(loaded[0][..., 0] == 1.0)
    assert np.all(loaded[0][..., 1] == 2.0)
