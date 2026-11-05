#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""dump 格式注册制。

一格式一实现，dump 写 / load 读成对。数据三形态 × 格式的语义矩阵：
  ndarray : bin=tofile（4bit 先 pack）/ npy=直存 / pt=转 torch 存 / print=print
  bytes   : bin=原样写文件 / npy·pt=按 dtype frombuffer 解码（uint1→uint8 别名、
            complex32→float16、4bit unpack）再存，无 dtype 按一维 uint8 数组序列化 /
            print=print(bytes) 或解码后 print
  Tensor  : bin=转 numpy tofile / npy=转 numpy 存 / pt=torch.save / print 同 ndarray

load 读回 store 布局文件（manual_data 消费）：npy 自定义 dtype 以等宽 void
落盘，float8_e5m2 兼容既有 uint8 载体，读侧 view 还原；pt 的 torch 不支持
dtype 走 ttk_raw_bytes/ttk_shape 原始字节载荷。读取失败抛 DumpLoadError。

新增格式 = 新增 formatter 实现 + @register_format 一行。
"""

__all__ = ["DumpFormatter", "DumpLoadError", "register_format", "get_formatter"]

# Standard Packages
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import numpy

from ....utilities import is_4bit_dtype, pack_4bits, resolve_numpy_dtypes, unpack_4bits
from ....utilities.dtypes import (
    NUMPY_DTYPE_SOURCES,
    dtype_to_name,
    numpy_to_torch_tensor,
    resolve_numpy_dtype,
    torch_to_numpy_tensor,
)
from ....utilities.shape import shape_numel

_FORMAT_REGISTRY: Dict[str, "DumpFormatter"] = {}


class DumpLoadError(ValueError):
    """格式读侧失败（文件损坏/shape·dtype 不符/载荷非法），由消费方映射为自身错误类型。"""


def register_format(name: str) -> Callable[[type], type]:
    """类装饰器：注册 dump 格式实现（单例实例入表）。"""

    def decorator(cls):
        _FORMAT_REGISTRY[name] = cls()
        return cls

    return decorator


def get_formatter(name: str) -> "DumpFormatter":
    """注册表查找；未知格式抛 ValueError（列出已注册格式）。"""
    try:
        return _FORMAT_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown dump format {name!r}; registered: {', '.join(sorted(_FORMAT_REGISTRY))}") from None


def _is_torch_tensor(data) -> bool:
    return "torch.Tensor" in str(type(data))


def _as_npy_savable(array):
    """ml_dtypes float8_e5m2 的 .npy descr 序列化为非法的 '<f1'（numpy 无 1 字节
    float，写出即不可读回），落盘前降为 uint8 view；读侧 NpyFormatter.load 的
    _CUSTOM_NUMPY_DTYPES view 还原。其余 fp8/bf16 的 descr（|V1/|V2）合法，直存。"""
    if array is not None and array.dtype.name == "float8_e5m2":
        return array.view(numpy.uint8)
    return array


# 读侧可与等宽 void 存储互认的自定义 dtype（写侧 view 落盘，读侧 view 还原）
_CUSTOM_NUMPY_DTYPES = {
    "bfloat16",
    "int4",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e8m0",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat8",
    "hifloat4",
}


def _decode_binary_to_array(data, dtype):
    """Decode raw bytes to numpy array based on dtype.

    返回 (array, pt_supported) 或 array（4bit 解包——pt 不可承载，调用方回退 npy）。"""
    if dtype is None:
        return None
    if str(dtype) == "uint1":
        return numpy.frombuffer(data, dtype="uint8")
    if not is_4bit_dtype(dtype):
        dtype = "float16" if str(dtype) == "complex32" else dtype
        return numpy.frombuffer(data, dtype=dtype), True
    return unpack_4bits(numpy.frombuffer(data, dtype=numpy.uint8), dtype)


class DumpFormatter(ABC):
    """一格式一实现。dump 写 {file_path}/{file_name}.{ext}；dtype 仅 bytes 形态需要。

    load 读回 store 布局的单文件（path 为完整文件路径，非目录+名）：dtype 为
    文件名声明的 numpy dtype，shape 为期望形状（None = 不校验），返回 ndarray。
    """

    ext: str

    @abstractmethod
    def dump(self, data, file_path: str, file_name: str, dtype: Optional[str] = None) -> None: ...

    def load(self, path, dtype, shape=None) -> numpy.ndarray:
        raise DumpLoadError(f"format {type(self).__name__} does not support load")

    def _abspath(self, file_path: str, file_name: str) -> str:
        dump_root = os.path.realpath(file_path)
        dump_path = os.path.realpath(os.path.join(dump_root, f"{file_name}.{self.ext}"))
        if os.path.commonpath((dump_root, dump_path)) != dump_root:
            raise ValueError(f"dump file path escapes dump directory: {file_name!r}")
        return dump_path


@register_format("bin")
class BinFormatter(DumpFormatter):
    ext = "bin"

    def dump(self, data, file_path: str, file_name: str, dtype: Optional[str] = None) -> None:
        path = self._abspath(file_path, file_name)
        if _is_torch_tensor(data):
            data = torch_to_numpy_tensor(data)
        if isinstance(data, numpy.ndarray):
            if is_4bit_dtype(data.dtype):
                pack_4bits(data).tofile(path)
            else:
                data.tofile(path)
        elif not isinstance(data, str) and data is not None:
            with open(path, "wb+") as f:
                f.write(data)

    def load(self, path, dtype, shape=None) -> numpy.ndarray:
        """bin 原始字节读取；4bit/特殊 dtype 与物理形态由 _postprocess 还原。"""
        if shape is None:
            return numpy.fromfile(str(path), dtype=_resolve_load_dtype(dtype or "uint8"))
        array = numpy.fromfile(str(path), dtype=_resolve_load_dtype(dtype or "uint8"))
        return _postprocess(array, dtype or "uint8", shape)


@register_format("npy")
class NpyFormatter(DumpFormatter):
    ext = "npy"

    def dump(self, data, file_path: str, file_name: str, dtype: Optional[str] = None) -> None:
        if isinstance(data, numpy.ndarray) or _is_torch_tensor(data):
            array = data if isinstance(data, numpy.ndarray) else torch_to_numpy_tensor(data)
            # e5m2 的 descr 为非法 '<f1'（写出即不可读回），先降 uint8 载体；
            # 其余自定义 dtype 的 descr 不可直存，view 等宽 V 落盘
            array = _as_npy_savable(array)
            if array.dtype.name in _CUSTOM_NUMPY_DTYPES:
                array = array.view(numpy.dtype(f"V{array.dtype.itemsize}"))
            numpy.save(self._abspath(file_path, file_name), array)
        elif not isinstance(data, str) and data is not None:
            self._dump_bytes(data, dtype, file_path, file_name)

    def _dump_bytes(self, data, dtype, file_path, file_name) -> None:
        path = self._abspath(file_path, file_name)
        if dtype is None:
            numpy.save(path, numpy.frombuffer(data, dtype=numpy.uint8))
            return
        result = _decode_binary_to_array(data, dtype)
        result = result if not isinstance(result, tuple) else result[0]
        numpy.save(path, _as_npy_savable(result))

    def load(self, path, dtype, shape=None) -> numpy.ndarray:
        """读回 .npy；自定义 dtype 的等宽 void 或 e5m2 的 uint8 载体按存储约定还原。"""
        array = numpy.load(str(path), allow_pickle=False)
        if shape is not None and tuple(array.shape) != tuple(shape):
            raise DumpLoadError(f"{path} stored shape {tuple(array.shape)} != expected {tuple(shape)}")

        expected_dtype = numpy.dtype(dtype)
        if array.dtype.name == expected_dtype.name:
            return array
        if expected_dtype.name not in _CUSTOM_NUMPY_DTYPES:
            raise DumpLoadError(f"{path} stored dtype {array.dtype.name!r} != filename dtype {expected_dtype.name!r}")
        if array.dtype.kind != "V" and not (
            expected_dtype.name == "float8_e5m2" and array.dtype == numpy.dtype(numpy.uint8)
        ):
            raise DumpLoadError(f"{path} stored dtype {array.dtype.name!r} != filename dtype {expected_dtype.name!r}")
        if array.dtype.itemsize != expected_dtype.itemsize:
            raise DumpLoadError(
                f"{path} itemsize {array.dtype.itemsize} != filename dtype itemsize {expected_dtype.itemsize}"
            )
        return array.view(dtype)


@register_format("pt")
class PtFormatter(DumpFormatter):
    ext = "pt"

    def dump(self, data, file_path: str, file_name: str, dtype: Optional[str] = None) -> None:
        if isinstance(data, numpy.ndarray) or _is_torch_tensor(data):
            import torch

            if _is_torch_tensor(data):
                payload = data
            else:
                try:
                    payload = numpy_to_torch_tensor(data)
                except Exception:
                    # torch 不承载的自定义 dtype：原始字节载荷（ttk_raw_bytes/ttk_shape）
                    raw = numpy.frombuffer(data.tobytes(order="C"), dtype=numpy.uint8).copy()
                    payload = {"ttk_raw_bytes": torch.from_numpy(raw), "ttk_shape": tuple(data.shape)}
            torch.save(payload, self._abspath(file_path, file_name))
        elif not isinstance(data, str) and data is not None:
            self._dump_bytes(data, dtype, file_path, file_name)

    def _dump_bytes(self, data, dtype, file_path, file_name) -> None:
        import torch

        path = self._abspath(file_path, file_name)
        if dtype is None:
            torch.save(numpy_to_torch_tensor(numpy.frombuffer(data, dtype=numpy.uint8)), path)
            return
        result = _decode_binary_to_array(data, dtype)
        # 4bit 解包产物 pt 不可承载——回退写 npy 文件
        if isinstance(result, tuple):
            torch.save(numpy_to_torch_tensor(result[0]), path)
        else:
            numpy.save(self._abspath(file_path, file_name).removesuffix(".pt") + ".npy", result)

    def load(self, path, dtype, shape=None) -> numpy.ndarray:
        payload = _load_pt_payload(path)
        if isinstance(payload, dict) and set(payload) == {"ttk_raw_bytes", "ttk_shape"}:
            return _pt_raw_bytes_to_array(payload, dtype, shape, path)
        if hasattr(payload, "detach"):
            array = torch_to_numpy_tensor(payload.detach().cpu())
            if shape is not None and tuple(array.shape) != tuple(shape):
                raise DumpLoadError(f"{path} stored shape {tuple(array.shape)} != expected {tuple(shape)}")
            return array
        raise DumpLoadError(f"invalid pt payload: {path}")


@register_format("print")
class PrintFormatter(DumpFormatter):
    ext = "print"  # 不落文件

    def dump(self, data, file_path: str, file_name: str, dtype: Optional[str] = None) -> None:
        if isinstance(data, (numpy.ndarray,)) or _is_torch_tensor(data):
            print(data)
        elif not isinstance(data, str) and data is not None:
            if dtype is None:
                print(data)
            else:
                result = _decode_binary_to_array(data, dtype)
                print(result if not isinstance(result, tuple) else result[0])


def _load_pt_payload(path):
    """torch.load（weights_only 优先；旧版 Torch 回退 pickle 并告警）。"""
    import inspect

    try:
        import torch

        try:
            supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
        except (TypeError, ValueError):
            supports_weights_only = True
        if supports_weights_only:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        logging.warning(
            "torch.load on this Torch version does not support weights_only=True; "
            "falling back to pickle-based loading for %s. Only load trusted manual-data files.",
            path,
        )
        return torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise DumpLoadError(f"cannot load pt file {path}: {exc}") from exc


def _pt_raw_bytes_to_array(payload, dtype, shape, path) -> numpy.ndarray:
    """ttk_raw_bytes/ttk_shape 原始字节载荷 → 按 dtype 解码 reshape。"""
    raw_tensor = payload["ttk_raw_bytes"]
    if not hasattr(raw_tensor, "detach"):
        raise DumpLoadError(f"invalid raw-byte pt payload: {path}")
    try:
        stored_shape = tuple(int(dimension) for dimension in payload["ttk_shape"])
    except Exception as exc:
        raise DumpLoadError(f"invalid raw-byte pt shape: {path}") from exc
    if shape is not None and stored_shape != tuple(shape):
        raise DumpLoadError(f"{path} stored shape {stored_shape} != expected {tuple(shape)}")
    raw = raw_tensor.detach().cpu().contiguous().numpy().tobytes()
    try:
        return numpy.frombuffer(raw, dtype=dtype).copy().reshape(stored_shape)
    except ValueError as exc:
        raise DumpLoadError(f"{path} raw bytes cannot be reshaped to stored shape {stored_shape}") from exc


def storage_layout(logical_dtype, logical_shape) -> Tuple[str, Tuple[int, ...]]:
    """逻辑 dtype+shape → 落盘的物理存储 (dtype, shape)。"""
    dtype_name = dtype_to_name(logical_dtype)
    shape = tuple(int(dimension) for dimension in logical_shape)
    if dtype_name == "complex32":
        return "float16", shape + (2,)
    if dtype_name in ("uint1", "int1"):
        element_count = math.prod(shape) if shape else 1
        return "uint8", ((element_count + 7) // 8,)
    try:
        resolved = resolve_numpy_dtypes((dtype_name,))[0]
        return numpy.dtype(resolved).name, _saved_array_shape(shape)
    except Exception as exc:
        raise DumpLoadError(f"cannot resolve dtype {dtype_name!r}") from exc


class ManualDataError(RuntimeError):
    """Manual data is missing, corrupt, or incompatible with the CSV case."""


def _saved_array_shape(shape) -> Tuple[int, ...]:
    result = tuple(int(dimension) for dimension in shape)
    return result if result else (1,)


_NONE_DTYPE = "none"


_RAW_UINT8_COMPATIBLE_DTYPES = {
    "int4",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e8m0",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat8",
    "hifloat4",
}


def _resolve_file_dtype(dtype_name: str):
    try:
        return resolve_numpy_dtypes((dtype_name,))[0]
    except Exception as exc:
        raise ManualDataError(f"cannot resolve dtype {dtype_name!r} from manual-data filename") from exc


def _expected_bin_size(dtype_name: str, shape: Tuple[int, ...]) -> int:
    count = math.prod(shape)
    if is_4bit_dtype(dtype_name):
        return (count + 1) // 2
    try:
        return count * numpy.dtype(_resolve_file_dtype(dtype_name)).itemsize
    except Exception as exc:
        if isinstance(exc, ManualDataError):
            raise
        raise ManualDataError(f"cannot determine byte size for dtype {dtype_name!r}") from exc


def _resolve_load_dtype(dtype):
    """Map special dtypes to a numpy-loadable dtype for raw file reading."""
    if dtype in ("uint1", "int1"):
        return "uint8"
    if dtype == "complex32":
        return "float16"
    if is_4bit_dtype(dtype):
        return "uint8"
    if isinstance(dtype, str) and dtype in NUMPY_DTYPE_SOURCES:
        return resolve_numpy_dtype(dtype)
    return dtype


def _postprocess_4bit(array, ori_dtype, shape):
    """Unpack or view 4-bit data after loading."""
    if array.dtype.name in ("int8", "uint8"):
        size = shape_numel(shape)
        if array.size == size:
            array = array.view(ori_dtype)
        else:
            array = unpack_4bits(array, ori_dtype)
            array = array[:size]
    else:
        array = array.view(ori_dtype)
    return array


def _postprocess_special(array, ori_dtype):
    """View-cast special dtypes loaded from .npy (stored as uint8/int8)."""
    for sd in ("bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e8m0", "hifloat8"):
        if sd in str(ori_dtype) and array.dtype.name != sd:
            array = array.view(ori_dtype)
            break
    return array


def _reshape_storage(array, ori_dtype, shape):
    """Reshape to the dtype's physical storage shape: int1 packs 8 elements per
    byte on the last dim; complex32 stores fp16 real/imag as a trailing [2]."""
    if ori_dtype == "int1":
        shape = list(shape[:-1]) + [shape[-1] // 8]
    elif ori_dtype == "complex32":
        shape = list(shape) + [2]
    try:
        return array.reshape(shape)
    except ValueError:
        # in manual output scenario:
        # output shape may be wrong (needs infer-shaping by golden)
        return array


def _postprocess(array, ori_dtype, shape):
    """读后处理：4bit 解包 → 特殊 dtype view 还原 → 物理存储形态 reshape。"""
    if is_4bit_dtype(ori_dtype):
        array = _postprocess_4bit(array, ori_dtype, shape)
    array = _postprocess_special(array, ori_dtype)
    return _reshape_storage(array, ori_dtype, shape)
