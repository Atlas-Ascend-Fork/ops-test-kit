# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""dtypes 公开面行为锚点：27 个对外接口逐一覆盖。

用例原则：一条用例锚一个契约——别名归一、双向不变量（round-trip/表互逆）、
修复点回归（e8m0 别名、width(None)、double 浮点分类）、拒绝路径（KeyError/RuntimeError）。
"""

import importlib.util

import numpy as np
import pytest
import torch

from ttk.utilities.dtypes import (
    ACL_DTYPE_TO_ID,
    ACL_ID_TO_DTYPE,
    DTYPE_PROMOTE_MAP,
    DTYPE_WIDTH_MAP,
    acl_to_torch_dtype,
    dtype_to_name,
    get_dtype_range,
    get_dtype_width,
    is_4bit_dtype,
    is_float_dtype,
    is_torch_native_dtype,
    map_dtypes,
    mx_quantize,
    normalize_dtype,
    numpy_to_torch_tensor,
    pack_4bits,
    resolve_numpy_dtype,
    resolve_numpy_dtypes,
    str_to_torch_dtype,
    torch_to_numpy_tensor,
    unpack_4bits,
)


def _missing_ml_dtypes():
    try:
        import ml_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


def _missing_en_dtypes():
    try:
        import en_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


_TF_INSTALLED = importlib.util.find_spec("tensorflow") is not None
_TORCH_NPU_TF_CONFLICT = importlib.util.find_spec("torch_npu") is not None and _TF_INSTALLED
_requires_tf = pytest.mark.skipif(
    not _TF_INSTALLED or _TORCH_NPU_TF_CONFLICT,
    reason="tensorflow not installed"
    if not _TF_INSTALLED
    else "torch_npu._C + tensorflow C extension conflict → segfault",
)


# ── 表不变量 ────────────────────────────────────────────────────────────────


def test_acl_dtype_tables_round_trip():
    """ACL 双表互逆；同义别名指向同一枚举 id。"""
    for acl_id, name in ACL_ID_TO_DTYPE.items():
        assert ACL_DTYPE_TO_ID[name] == acl_id
    assert ACL_DTYPE_TO_ID["bf16"] == ACL_DTYPE_TO_ID["bfloat16"] == 27
    assert ACL_DTYPE_TO_ID["double"] == ACL_DTYPE_TO_ID["float64"] == 11
    assert ACL_DTYPE_TO_ID["float8_e8m0"] == ACL_DTYPE_TO_ID["float8_e8m0fnu"] == 37


def test_dtype_promote_map_widens():
    """Promote 目标必须严格更宽（抬精度才有意义）。"""
    for src, dst in DTYPE_PROMOTE_MAP.items():
        assert DTYPE_WIDTH_MAP[src] < DTYPE_WIDTH_MAP[dst], (src, dst)


# ── 名字归一 / 分类 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("alias", "canonical"), [("fp16", "float16"), ("bf16", "bfloat16"), ("f8_e4m3fn", "float8_e4m3fn"), (None, None)]
)
def test_normalize_dtype(alias, canonical):
    assert normalize_dtype(alias) is canonical or normalize_dtype(alias) == canonical


def test_normalize_dtype_rejects_unknown():
    """方括号访问 = 已知 dtype 白名单，CSV typo 在解析期 fail-fast。"""
    with pytest.raises(KeyError):
        normalize_dtype("flaot16")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float16", True),
        ("fp16", True),
        ("bf16", True),
        ("double", True),
        ("half", True),
        ("float8_e8m0", True),
        ("float4_e2m1", True),
        ("hifloat8", True),
        ("int8", False),
        ("int4", False),
        ("uint1", False),
        ("bool", False),
        ("complex64", False),
        (None, False),
    ],
)
def test_is_float_dtype_matrix(name, expected):
    """别名与规范名统一分类；"double"=True 是对旧 profiling 子串匹配漏判的修正锚点。"""
    assert is_float_dtype(name) is expected


def test_is_float_dtype_objects():
    assert is_float_dtype(torch.float16) is True
    assert is_float_dtype(np.dtype("float32")) is True
    assert is_float_dtype(np.dtype("int32")) is False
    assert is_float_dtype(np.dtype("bool")) is False


def test_dtype_to_name_robust():
    """torch dtype / numpy dtype / ml_dtypes 类对象 / 纯字符串统一产出裸名。

    类对象直接 str().split('.') 会得到 "bfloat16'>"（尾部残留），必须被剥掉。
    """
    assert dtype_to_name(torch.float16) == "float16"
    assert dtype_to_name(np.dtype("float32")) == "float32"
    assert dtype_to_name("float16") == "float16"
    assert dtype_to_name("int4") == "int4"
    ml_dtypes = pytest.importorskip("ml_dtypes")
    assert dtype_to_name(ml_dtypes.bfloat16) == "bfloat16"
    assert dtype_to_name(np.dtype(ml_dtypes.bfloat16)) == "bfloat16"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("int4", True), ("float4_e2m1", True), ("hifloat4", True), ("int8", False), ("float16", False)],
)
def test_is_4bit_dtype(name, expected):
    assert is_4bit_dtype(name) is expected


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_is_4bit_dtype_accepts_numpy_dtype():
    assert is_4bit_dtype(np.dtype(resolve_numpy_dtype("float4_e2m1"))) is True


def test_map_dtypes_preserves_nesting_as_tuples():
    assert map_dtypes([("fp16", "int8"), "bf16"], normalize_dtype) == (("float16", "int8"), "bfloat16")
    assert map_dtypes((), normalize_dtype) == ()
    assert map_dtypes(None, normalize_dtype) is None


# ── 宽度 / 范围 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("dtype", "width"),
    [
        ("float16", 2),
        ("float32", 4),
        ("int8", 1),
        ("int4", 0.5),
        ("hifloat4", 0.5),
        (None, 0),
    ],
)
def test_get_dtype_width(dtype, width):
    """逻辑位宽（非宿主存储宽度）；hifloat4（子串兜底）与 None（短路）是修复锚点。"""
    assert get_dtype_width(dtype) == width


def test_get_dtype_width_rejects_unknown():
    with pytest.raises(RuntimeError, match="Unknown dtype"):
        get_dtype_width("qint8")


@pytest.mark.parametrize(
    ("dtype", "low", "high"),
    [
        ("int8", -128, 127),
        ("int4", -8, 7),
        ("bool", 0, 1),
        ("float8_e8m0", 2**-127, 2**127),
        ("bfloat16", -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")),
        ("float16", np.finfo(np.float16).min, np.finfo(np.float16).max),
    ],
)
def test_get_dtype_range(dtype, low, high):
    assert get_dtype_range(dtype) == (low, high)


# ── 自定义 dtype 解析 ───────────────────────────────────────────────────────


def test_resolve_numpy_dtype_bfloat16():
    assert dtype_to_name(resolve_numpy_dtype("bfloat16")) == "bfloat16"


def test_resolve_numpy_dtype_rejects_non_custom():
    """原生 numpy dtype 不是此口职责——错误信息须直说。"""
    with pytest.raises(RuntimeError, match="Unsupported custom numpy dtype"):
        resolve_numpy_dtype("float16")


def test_resolve_numpy_dtypes_resolves_and_tuples():
    resolved = resolve_numpy_dtypes(["float32", ("bfloat16", "int4")])
    assert isinstance(resolved, tuple)
    assert resolved[0] == np.dtype("float32")
    inner = resolved[1]
    assert isinstance(inner, tuple)
    assert dtype_to_name(inner[0]) == "bfloat16"
    assert dtype_to_name(inner[1]) == "int4"


# ── 4bit 打包 / 解包 ────────────────────────────────────────────────────────


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_pack_4bits_layout_and_roundtrip():
    """e2m1 编码 0/1/2/3 → 半字节 0/2/4/5；奇数尾部补零；pack↔unpack 互逆。"""
    from en_dtypes import float4_e2m1

    src = np.array([0, 1, 2, 3], dtype=float4_e2m1)
    packed = pack_4bits(src)
    assert packed.dtype == np.uint8
    assert packed.tolist() == [0 | 2 << 4, 4 | 5 << 4]
    assert np.array_equal(unpack_4bits(packed, "float4_e2m1"), src)

    odd = np.array([0, 1, 2, 3, 4], dtype=float4_e2m1)
    assert pack_4bits(odd).tolist() == [0 | 2 << 4, 4 | 5 << 4, 6]  # e2m1(4) 编码为 0x6


# ── numpy ↔ torch 转换 ──────────────────────────────────────────────────────


@pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.int32])
def test_torch_numpy_roundtrip_native(np_dtype):
    src = (np.arange(12) - 6).astype(np_dtype)
    back = torch_to_numpy_tensor(numpy_to_torch_tensor(src))
    assert back.dtype == src.dtype
    assert np.array_equal(back, src)


@pytest.mark.skipif(_missing_ml_dtypes(), reason="ml_dtypes is not installed")
@pytest.mark.parametrize("name", ["bfloat16", "float8_e5m2", "float8_e4m3fn"])
def test_torch_numpy_roundtrip_ml_dtypes(name):
    src = np.linspace(-1.0, 1.0, 14).astype(resolve_numpy_dtype(name))
    t = numpy_to_torch_tensor(src)
    assert t.dtype == getattr(torch, name)
    back = torch_to_numpy_tensor(t)
    assert back.dtype == src.dtype
    assert np.array_equal(back.view(np.uint8), src.view(np.uint8))


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_torch_numpy_roundtrip_e8m0():
    """e8m0 正数刻度域；torch 侧拼写是 float8_e8m0fnu。"""
    src = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=resolve_numpy_dtype("float8_e8m0"))
    t = numpy_to_torch_tensor(src)
    assert t.dtype == torch.float8_e8m0fnu
    back = torch_to_numpy_tensor(t)
    assert back.dtype == src.dtype
    assert np.array_equal(back.view(np.uint8), src.view(np.uint8))


@pytest.mark.skipif(
    _missing_en_dtypes() or not hasattr(torch, "float4_e2m1fn_x2"), reason="en_dtypes/float4 torch dtype unavailable"
)
def test_torch_numpy_roundtrip_float4():
    src = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 0.5, 1.5], dtype=resolve_numpy_dtype("float4_e2m1"))
    t = numpy_to_torch_tensor(src)
    assert t.dtype == torch.float4_e2m1fn_x2
    assert tuple(t.shape) == (4,)
    back = torch_to_numpy_tensor(t)
    assert back.dtype == src.dtype
    assert np.array_equal(back.view(np.uint8), src.view(np.uint8))


@pytest.mark.skipif(_missing_ml_dtypes(), reason="ml_dtypes is not installed")
def test_int4_converts_to_packed_uint8():
    """int4 走 fp4 同款 packed uint8 降级分支（不 raise）：末维减半、按 pack_4bits 打包。"""
    src = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=resolve_numpy_dtype("int4"))
    t = numpy_to_torch_tensor(src)
    assert t.dtype == torch.uint8
    assert tuple(t.shape) == (4,)
    assert t.tolist() == pack_4bits(src).tolist()


@pytest.mark.skipif(_missing_ml_dtypes(), reason="ml_dtypes is not installed")
def test_int4_2d_shape_and_odd_last_dim():
    src = np.arange(12, dtype=resolve_numpy_dtype("int4")).reshape(2, 6)
    t = numpy_to_torch_tensor(src)
    assert tuple(t.shape) == (2, 3)
    odd = np.array([0, 1, 2], dtype=resolve_numpy_dtype("int4"))
    with pytest.raises(RuntimeError, match="even last dimension"):
        numpy_to_torch_tensor(odd)


def test_torch_numpy_complex32_roundtrip():
    src = np.zeros(8, dtype=np.float16)
    src[::2] = 1.0
    t = numpy_to_torch_tensor(src, is_complex32=True)
    assert t.dtype == torch.complex32
    back = torch_to_numpy_tensor(t)
    assert back.dtype == np.float16
    assert np.array_equal(back, src)


@pytest.mark.skipif(_missing_ml_dtypes(), reason="ml_dtypes is not installed")
def test_int4_packs_to_uint8_like_other_4bit():
    """int4 与 HIF4 同路：无可用 view 目标时保持 packed uint8（不 raise）。"""
    t = numpy_to_torch_tensor(np.zeros(4, dtype=resolve_numpy_dtype("int4")))
    assert t.dtype == torch.uint8
    assert tuple(t.shape) == (2,)


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_hifloat4_packs_to_uint8_without_torch_dtype():
    """HIF4 逻辑存储在 torch 缺 E1M2 dtype 时保持 packed uint8。"""
    logical = np.zeros((2, 4), dtype=resolve_numpy_dtype("hifloat4"))
    logical.view(np.uint8)[:] = np.arange(logical.size, dtype=np.uint8).reshape(logical.shape)
    packed = numpy_to_torch_tensor(logical)
    assert tuple(packed.shape) == (2, 2)
    expected = getattr(torch, "float4_e1m2fn_x2", None)
    assert packed.dtype == (expected if isinstance(expected, torch.dtype) else torch.uint8)


def test_torch_to_numpy_tensor_rejects_non_tensor():
    with pytest.raises(RuntimeError, match="Only support torch.Tensor"):
        torch_to_numpy_tensor([1, 2, 3])


def test_acl_to_torch_dtype():
    assert acl_to_torch_dtype([0, 1, 27, 11]) == (torch.float32, torch.float16, torch.bfloat16, torch.double)


# ── 框架原生判定 / 名字→对象 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("float16", torch.float16),
        ("float8_e8m0", torch.float8_e8m0fnu),
        ("torch.float32", torch.float32),
        ("torch.float8_e8m0", torch.float8_e8m0fnu),
        ("bogus", None),
        ("a.b.c", None),
        ("numpy.float16", None),
    ],
)
def test_str_to_torch_dtype(raw, expected):
    """e8m0 两行是修复锚点：别名必须命中 torch.float8_e8m0fnu（曾误返 None）。"""
    assert str_to_torch_dtype(raw) == expected


def test_str_to_torch_dtype_passthrough_non_str():
    sentinel = object()
    assert str_to_torch_dtype(sentinel) is sentinel


@pytest.mark.parametrize(
    ("name", "expected"), [("float16", True), ("int4", False), ("float8_e8m0", True), ("bogus", False)]
)
def test_is_torch_native_dtype(name, expected):
    """int4 显式排除：torch.int4 存在但 CPU 上不可用。"""
    assert is_torch_native_dtype(name) is expected


# ── MX 量化 ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_mx_quantize_zero_block():
    """全零块：share_exp=-inf 被 clamp 到 -127 → 最小 scale，ele 全零。"""
    scale, ele = mx_quantize(np.zeros((1, 32), dtype=np.float16), "float4_e2m1")
    assert scale.shape == (1, 1, 2)
    assert float(scale.ravel()[0]) == 2.0**-127
    assert not ele.any()


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_mx_quantize_saturated_block():
    """块内全为 e2m1 max(6.0)：share_exp=0 → scale=1.0，ele 精确表示。"""
    scale, ele = mx_quantize(np.full((1, 32), 6.0, dtype=np.float16), "float4_e2m1")
    assert float(scale.ravel()[0]) == 1.0
    assert np.all(ele.astype(np.float32) == 6.0)


def test_mx_element_dtypes_gate():
    """MX_ELEMENT_DTYPES 与 mx_quantize 支持集一致，非成员直接拒绝。"""
    from ttk.utilities.dtypes import MX_ELEMENT_DTYPES

    assert "float4_e2m1" in MX_ELEMENT_DTYPES
    with pytest.raises(NotImplementedError):
        mx_quantize(np.zeros((2, 32), dtype=np.float16), mx_ele_dtype="float16")


# ── TF 桥接（torch_npu 与 tf C 扩展同进程冲突，按仓内惯例跳过） ─────────────


@_requires_tf
def test_tf_dtype_bridge():
    import tensorflow as tf

    from ttk.utilities.dtypes import is_tf_native_dtype, normalize_from_tf_dtype, normalize_to_tf_dtype, str_to_tf_dtype

    assert str_to_tf_dtype("fp16") == tf.float16
    assert str_to_tf_dtype("not-a-dtype") is None
    assert is_tf_native_dtype("fp16") is True
    assert is_tf_native_dtype("float8_e4m3fn") is False

    src = np.linspace(-1.0, 1.0, 8).astype(resolve_numpy_dtype("bfloat16"))
    to_tf = normalize_to_tf_dtype(src)
    assert to_tf.dtype == tf.bfloat16.as_numpy_dtype
    back = normalize_from_tf_dtype(to_tf)
    assert back.dtype == src.dtype
    assert np.array_equal(back.view(np.uint16), src.view(np.uint16))
