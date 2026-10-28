#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
OmSession：一个 (算子, 输入形态, 属性, soc, dev_id) 组合的常驻执行体。

生命周期：set_device → load → desc 查询（含动态输出守卫）→ 分配输入输出
HBM + H2D → [execute_round]*N → D2H → unload。首轮（warmup）完成全部重活，
计时轮只走 aclmdlExecute + D2H。

动态输出守卫：数据依赖 shape 算子（NonZero 类）
不在 atc 编译期失败，而是静默产出动态 dim om（-1 / size=0）——desc 查询处
任一输出 dim<0 或 size==0 即显式失败 ONNX_DYNAMIC_OUTPUT_UNSUPPORTED
（信息带算子名与 desc 明细），不与 -d 模式开关的 ONNX_DYNAMIC_GRAPH_UNSUPPORTED
混淆。

输出缓冲建会话时以 0x7F 哨兵预填（假 PASS 防御）：漏写输出的 kernel 读到的
是确定性哨兵模式而非陈旧 HBM 内容——本应 FAIL 的用例不再因缓冲恰含旧值
而假 PASS。

回读按 desc dims/dtype 还原：desc size 可能大于理论字节（对齐 padding），
frombuffer 按 numel 截断。
"""

__all__ = ["OmSession"]

import hashlib
import logging
import weakref

import numpy as np

from ttk.utilities import ACL_ID_TO_DTYPE, resolve_numpy_dtype

# 输出缓冲哨兵字节：0x7F 逐字节填充——float32 位模式 0x7F7F7F7F ≈ 3.4e+38
# （近 float32 上限的大数，不会与常见有效输出混淆）；int8/uint8 视角为 127，
# int32 视角为 0x7F7F7F7F（2139062143）
_OUTPUT_SENTINEL_BYTE = 0x7F


def _desc_numpy_dtype(dtype_id: int):
    """ACL dtype 枚举 → numpy dtype（bfloat16 等经 ml_dtypes 解析）。"""
    name = ACL_ID_TO_DTYPE.get(int(dtype_id))
    if name is None:
        raise RuntimeError(f"OM_LOAD_FAILED: om 输出 desc 的 ACL dtype 枚举 {dtype_id} 无 TTK canonical 映射")
    try:
        return np.dtype(name)
    except TypeError:
        return resolve_numpy_dtype(name)


def _inputs_digest(host_inputs) -> str:
    """输入内容指纹（建会话一次 + 复用判定慢路径；逐轮零开销）。"""
    h = hashlib.md5()  # noqa: S324  # 非安全用途：输入内容指纹
    for arr in host_inputs:
        h.update(f"[{arr.dtype.str}:{arr.shape}]".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class OmSession:
    """om 常驻执行体（OnnxAdapter 持有；会话键见 onnx/__init__.py）。"""

    def __init__(self, acl, dev_id: int, om_path: str, host_inputs, op_type: str):
        """load → desc 查询（动态输出守卫）→ 分配 + H2D（首轮/warmup 全部重活）。"""
        self._acl = acl
        self._dev_id = dev_id
        self._op_type = op_type
        self._model_id = None
        self._input_bufs = []
        self._output_bufs = []
        acl.set_device(dev_id)
        try:
            self._model_id = acl.load_model(om_path)
        except RuntimeError as exc:
            raise RuntimeError(f"OM_LOAD_FAILED: {exc}") from exc
        try:
            try:
                io = acl.model_io_info(self._model_id)
            except RuntimeError as exc:
                raise RuntimeError(f"OM_LOAD_FAILED: {exc}") from exc
            self._input_descs = io["inputs"]
            self._output_descs = io["outputs"]
            self._guard_dynamic_outputs()

            if len(self._input_descs) != len(host_inputs):
                raise RuntimeError(
                    f"OM_LOAD_FAILED: om 输入数 {len(self._input_descs)} 与 host 输入数 "
                    f"{len(host_inputs)} 不一致（om={om_path}）"
                )
            for desc, arr in zip(self._input_descs, host_inputs):
                ptr = acl.h2d(np.ascontiguousarray(arr), size=desc.size)
                self._input_bufs.append((ptr, desc.size))
            for desc in self._output_descs:
                ptr = acl.malloc(desc.size)
                # 哨兵预填：未写输出的 kernel 读到确定性模式而非陈旧 HBM（假 PASS 防御）
                acl.memset(ptr, _OUTPUT_SENTINEL_BYTE, desc.size)
                self._output_bufs.append((ptr, desc.size))
        except Exception:
            self.unload()
            raise
        # 会话键输入指纹（同签名不同值用例的复用判定：
        # 身份快路径 + 内容慢路径）
        self._arg_refs = tuple(weakref.ref(arr) for arr in host_inputs)
        self._input_digest = _inputs_digest(host_inputs)

    # ── 复用判定（身份快路径 + 内容慢路径） ──

    def matches_inputs(self, host_inputs) -> bool:
        """同会话判定：数组对象身份一致（weakref 存活）或内容 md5 一致。

        host_inputs 为 build_input_specs 的在场输入（None 占位已剔除）；
        同一结构键下 None 模式确定，无需在此比对。
        """
        if len(self._arg_refs) != len(host_inputs):
            return False
        if all(ref() is arr for ref, arr in zip(self._arg_refs, host_inputs)):
            return True  # 同对象（无原地写语义，内容必未变）
        return _inputs_digest(host_inputs) == self._input_digest

    def rebind_inputs(self, host_inputs) -> None:
        """复用命中后刷新身份引用（内容未变时快路径持续生效）。"""
        self._arg_refs = tuple(weakref.ref(arr) for arr in host_inputs)

    def _guard_dynamic_outputs(self) -> None:
        for desc in self._output_descs:
            if desc.size == 0 or any(dim < 0 for dim in desc.dims):
                raise RuntimeError(
                    f"ONNX_DYNAMIC_OUTPUT_UNSUPPORTED: 算子 {self._op_type} 的 om 输出 "
                    f"{desc!r} 含动态 dim 或 size=0——数据依赖 shape 的算子不支持"
                    "（atc 静默产出动态 om）；输出 shape 无法从 desc 恢复"
                )

    # ── 执行 ────────────────────────────────────────────────

    def execute_round(self) -> tuple:
        """aclmdlExecute + D2H。前置守卫：跨用例换卡时 set_device 切回；
        执行失败加 OM_EXECUTE_FAILED 前缀（ACL 错误码原文透传）。"""
        if self._acl.device_id != self._dev_id:
            self._acl.set_device(self._dev_id)
        try:
            self._acl.execute(self._model_id, self._input_bufs, self._output_bufs)
        except RuntimeError as exc:
            raise RuntimeError(f"OM_EXECUTE_FAILED: {exc}") from exc
        return tuple(self._read_output(desc, ptr) for desc, (ptr, _size) in zip(self._output_descs, self._output_bufs))

    def _read_output(self, desc, ptr) -> np.ndarray:
        np_dtype = _desc_numpy_dtype(desc.dtype_id)
        numel = 1
        for dim in desc.dims:
            numel *= int(dim)

        raw = self._acl.d2h(ptr, desc.size)

        # desc size 可能含对齐 padding：按 numel 截断（frombuffer 只读语义可接受，
        # 比对/dump/MD5 均只读）
        return np.frombuffer(raw, dtype=np_dtype, count=numel).reshape(desc.dims)

    def unload(self) -> None:
        """释放 HBM + 卸载模型（账本在 AclModelInterface 侧）。"""
        for ptr, _size in tuple(self._input_bufs) + tuple(self._output_bufs):
            try:
                self._acl.free(ptr)
            except RuntimeError:
                logging.warning(f"[{self._op_type}] free device buffer failed during unload", exc_info=True)
        self._input_bufs = []
        self._output_bufs = []
        model_id = self._model_id
        if model_id is not None:
            self._model_id = None
            try:
                self._acl.unload_model(model_id)
            except RuntimeError:
                logging.warning(f"[{self._op_type}] unload model {model_id} failed", exc_info=True)
