# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""file_utils 公开面：load_array 装载契约。

- float8_e5m2 三后缀（bin/npy/pt）全通：npy 侧按 dump 存储约定（uint8 view
  落盘）读回后按 dtype view 还原
- dtype 参数两种形态均受支持：str 名与 resolve_numpy_dtypes 解析出的 dtype
  对象（manual_data / input_generate / golden 的真实传入形态）
- 物理 shape 变换：int1 尾维 //8、complex32 追加实虚尾维 [2]；声明 shape 与
  数据不符时回退原数组（manual output 场景，待 golden infer-shape）
"""

import numpy
import pytest

from ttk.core_modules.common.dump.formatter import get_formatter
from ttk.utilities import extract_zip, load_array
from ttk.utilities.dtypes import resolve_numpy_dtype


def _e5m2(values):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    return numpy.array(values, dtype=ml_dtypes.float8_e5m2)


def _e5m2_dtype(form):
    if form == "str":
        return "float8_e5m2"
    ml_dtypes = pytest.importorskip("ml_dtypes")
    return ml_dtypes.float8_e5m2


@pytest.mark.parametrize("form", ["str", "object"])
def test_load_array_e5m2_npy_roundtrip(form, tmp_path):
    """dump 侧 uint8 存储约定 + 读侧 view 还原，dtype 两形态等价。"""
    arr = _e5m2([1, 2, 4, 8])
    get_formatter("npy").dump(arr, str(tmp_path), "x")
    back = load_array(str(tmp_path / "x.npy"), _e5m2_dtype(form), [4])
    assert back.dtype == resolve_numpy_dtype("float8_e5m2")
    numpy.testing.assert_array_equal(back.view(numpy.uint8), arr.view(numpy.uint8))


def test_load_array_e5m2_bin(tmp_path):
    arr = _e5m2([1, 2, 4, 8])
    arr.view(numpy.uint8).tofile(tmp_path / "x.bin")
    back = load_array(str(tmp_path / "x.bin"), "float8_e5m2", [4])
    assert back.dtype == resolve_numpy_dtype("float8_e5m2")
    numpy.testing.assert_array_equal(back.view(numpy.uint8), arr.view(numpy.uint8))


def test_load_array_e5m2_pt(tmp_path):
    torch = pytest.importorskip("torch")
    t = torch.tensor([1, 2, 4, 8], dtype=torch.float8_e5m2)
    torch.save(t, tmp_path / "x.pt")
    back = load_array(str(tmp_path / "x.pt"), "float8_e5m2", [4])
    numpy.testing.assert_array_equal(back.view(numpy.uint8), t.view(torch.uint8).numpy())


def test_load_array_storage_shape_int1(tmp_path):
    """int1：尾维 //8（8 元素/字节）。"""
    numpy.arange(8, dtype=numpy.uint8).tofile(tmp_path / "x.bin")
    back = load_array(str(tmp_path / "x.bin"), "int1", [8, 8])
    assert back.shape == (8, 1)
    assert back.dtype == numpy.uint8


def test_load_array_storage_shape_complex32(tmp_path):
    """complex32：fp16 实虚尾维 [2]。"""
    numpy.arange(8, dtype=numpy.float16).tofile(tmp_path / "x.bin")
    back = load_array(str(tmp_path / "x.bin"), "complex32", [4])
    assert back.shape == (4, 2)
    assert back.dtype == numpy.float16


def test_load_array_reshape_fallback_on_wrong_shape(tmp_path):
    """声明 shape 与数据不符：回退原数组（manual output 待 golden infer-shape）。"""
    numpy.arange(4, dtype=numpy.int32).tofile(tmp_path / "x.bin")
    back = load_array(str(tmp_path / "x.bin"), "int32", [3])
    assert back.shape == (4,)


# -- extract_zip -------------------------------------------------------------


def _make_zip(path, entries):
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)


def test_extract_zip_strips_single_top_level(tmp_path):
    zip_path = tmp_path / "z.zip"
    _make_zip(zip_path, [("top/a.txt", b"aaa"), ("top/b.txt", b"bbb")])
    out = tmp_path / "out"
    out.mkdir()
    extract_zip(str(zip_path), str(out), remove_top_level=True)
    assert sorted(p.name for p in out.iterdir()) == ["a.txt", "b.txt"]
    assert (out / "a.txt").read_bytes() == b"aaa"


def test_extract_zip_multiple_top_levels_keeps_structure(tmp_path):
    zip_path = tmp_path / "z.zip"
    _make_zip(zip_path, [("a.txt", b"1"), ("b/c.txt", b"2")])
    out = tmp_path / "out"
    out.mkdir()
    extract_zip(str(zip_path), str(out), remove_top_level=True)
    assert (out / "a.txt").read_bytes() == b"1"
    assert (out / "b" / "c.txt").read_bytes() == b"2"


def test_extract_zip_keeps_adjacent_fixed_name_tmp_dir(tmp_path):
    """相邻 <目标>_tmp 目录不被清理（原实现固定名 + 无条件 rmtree 会丢已有内容）。"""
    zip_path = tmp_path / "z.zip"
    _make_zip(zip_path, [("top/a.txt", b"aaa")])
    out = tmp_path / "out"
    out.mkdir()
    neighbor = tmp_path / "out_tmp"
    neighbor.mkdir()
    (neighbor / "precious.txt").write_bytes(b"keep")
    extract_zip(str(zip_path), str(out), remove_top_level=True)
    assert (neighbor / "precious.txt").read_bytes() == b"keep"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("out_tmp_")]
