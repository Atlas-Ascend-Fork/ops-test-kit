#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
onnx e2e ACL 运行时：libascendcl aclmdl 族的 ctypes 封装。

职责：om 模型的加载 / desc 查询 / 执行 + HBM 管理 + 自持运行时生命周期
（aclInit / aclrtSetDevice / aclFinalize），仅依赖 libascendcl.so，懒加载、
进程内共享。范式对齐 aclnn/runtime.py 的 AclInterface，但更懒——模块 import
零 .so 触碰（AclInterface 在构造期 CDLL，本类首用时才加载）。

onnx e2e worker 进程内只有本类触达 ACL；backend 层零 ACL（DSMI 探测）。
"""

__all__ = ["AclModelInterface", "MdlIoDesc", "aclmdlIODims"]

import atexit
import ctypes
import logging
import threading

import numpy as np

ACL_ERROR_REPEAT_INITIALIZE = 100002
# aclrtMemcpyKind
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
# aclrtMemMallocPolicy：优先大页
ACL_MEM_MALLOC_HUGE_FIRST = 0

ACL_MAX_TENSOR_NAME_LEN = 128
ACL_MAX_DIM_CNT = 128


class aclmdlIODims(ctypes.Structure):
    """aclmdlIODims（acl_mdl.h）：name[128] + dimCount + dims[128]。"""

    _fields_ = [
        ("name", ctypes.c_char * ACL_MAX_TENSOR_NAME_LEN),
        ("dimCount", ctypes.c_size_t),
        ("dims", ctypes.c_int64 * ACL_MAX_DIM_CNT),
    ]


class MdlIoDesc:
    """单个模型输入/输出的 desc 快照（脱离 aclmdlDesc 生命周期可安全持有）。"""

    __slots__ = ("name", "dims", "dtype_id", "size", "format_id")

    def __init__(self, name, dims, dtype_id, size, format_id):
        self.name = name
        self.dims = dims
        self.dtype_id = dtype_id
        self.size = size
        self.format_id = format_id

    def __repr__(self):
        return f"MdlIoDesc(name={self.name!r}, dims={self.dims}, dtype_id={self.dtype_id}, size={self.size})"


def _signature_table():
    """aclmdl/aclrt 族签名表：api_name -> (restype, argtypes)。

    源自 CANN 9.2.0 include/acl/acl_mdl.h / acl_base.h / acl_rt.h。符号导出已经
    实测核实：28 符号全导出，另含
    aclrtResetDevice 与 aclmdlGetInputFormat / aclmdlGetOutputFormat
    （MdlIoDesc.format_id 的数据源），共 31 条。
    """
    c = ctypes
    dims_ptr = ctypes.POINTER(aclmdlIODims)
    return {
        # ── 运行时初始化 / 设备 ──
        "aclInit": (c.c_int32, [c.c_char_p]),
        "aclFinalize": (c.c_int32, []),
        "aclrtSetDevice": (c.c_int32, [c.c_int32]),
        "aclrtResetDevice": (c.c_int32, [c.c_int32]),
        "aclrtGetDeviceCount": (c.c_int32, [c.POINTER(c.c_uint32)]),
        # ── HBM ──
        "aclrtMalloc": (c.c_int32, [c.POINTER(c.c_void_p), c.c_size_t, c.c_int32]),
        "aclrtFree": (c.c_int32, [c.c_void_p]),
        "aclrtMemcpy": (c.c_int32, [c.c_void_p, c.c_size_t, c.c_void_p, c.c_size_t, c.c_int32]),
        "aclrtMemset": (c.c_int32, [c.c_void_p, c.c_size_t, c.c_int32, c.c_size_t]),
        # ── 模型生命周期 ──
        "aclmdlLoadFromFile": (c.c_int32, [c.c_char_p, c.POINTER(c.c_uint32)]),
        "aclmdlUnload": (c.c_int32, [c.c_uint32]),
        # ── desc 查询 ──
        "aclmdlCreateDesc": (c.c_void_p, []),
        "aclmdlDestroyDesc": (c.c_int32, [c.c_void_p]),
        "aclmdlGetDesc": (c.c_int32, [c.c_void_p, c.c_uint32]),
        "aclmdlGetNumInputs": (c.c_size_t, [c.c_void_p]),
        "aclmdlGetNumOutputs": (c.c_size_t, [c.c_void_p]),
        "aclmdlGetInputSizeByIndex": (c.c_size_t, [c.c_void_p, c.c_size_t]),
        "aclmdlGetOutputSizeByIndex": (c.c_size_t, [c.c_void_p, c.c_size_t]),
        "aclmdlGetInputDims": (c.c_int32, [c.c_void_p, c.c_size_t, dims_ptr]),
        "aclmdlGetOutputDims": (c.c_int32, [c.c_void_p, c.c_size_t, dims_ptr]),
        "aclmdlGetInputDataType": (c.c_int32, [c.c_void_p, c.c_size_t]),
        "aclmdlGetOutputDataType": (c.c_int32, [c.c_void_p, c.c_size_t]),
        "aclmdlGetInputNameByIndex": (c.c_char_p, [c.c_void_p, c.c_size_t]),
        "aclmdlGetOutputNameByIndex": (c.c_char_p, [c.c_void_p, c.c_size_t]),
        "aclmdlGetInputFormat": (c.c_int32, [c.c_void_p, c.c_size_t]),
        "aclmdlGetOutputFormat": (c.c_int32, [c.c_void_p, c.c_size_t]),
        # ── dataset 组装 ──
        "aclmdlCreateDataset": (c.c_void_p, []),
        "aclmdlDestroyDataset": (c.c_int32, [c.c_void_p]),
        "aclmdlAddDatasetBuffer": (c.c_int32, [c.c_void_p, c.c_void_p]),
        "aclCreateDataBuffer": (c.c_void_p, [c.c_void_p, c.c_size_t]),
        "aclDestroyDataBuffer": (c.c_int32, [c.c_void_p]),
        # ── 执行 ──
        "aclmdlExecute": (c.c_int32, [c.c_uint32, c.c_void_p, c.c_void_p]),
    }


class AclModelInterface:
    """libascendcl aclmdl 族封装（懒加载；模块 import 零 .so 触碰）。

    与 aclnn/AclInterface 的关系：同为 libascendcl 的 ctypes 消费者，但本类
    自持运行时生命周期（aclInit/SetDevice/Finalize），不依赖 aclnn 通路——
    onnx e2e worker 进程内只有本类触达 ACL。进程内单例（OnnxAdapter 持有）。
    """

    _DLL = None
    _DLL_LOCK = threading.Lock()

    def __init__(self, skip_teardown: bool = False):
        self._device_id = None
        self._acl_inited = False
        self._owns_acl_runtime = False
        self._models: set = set()
        self._dev_ptrs: set = set()
        # skip_teardown：仿真等场景抑制退出清理（对齐 AclInterface 语义）
        self.skip_teardown = skip_teardown
        atexit.register(self._on_exit)

    def __del__(self):
        self._on_exit()

    # ── 懒加载 ──────────────────────────────────────────────

    @classmethod
    def _dll(cls) -> ctypes.CDLL:
        if cls._DLL is None:
            with cls._DLL_LOCK:
                if cls._DLL is None:
                    dll = ctypes.CDLL("libascendcl.so")
                    cls._bind_signatures(dll)
                    cls._DLL = dll
        return cls._DLL

    @staticmethod
    def _bind_signatures(dll) -> None:
        for api_name, (restype, argtypes) in _signature_table().items():
            fn = getattr(dll, api_name)
            fn.restype = restype
            fn.argtypes = argtypes

    @staticmethod
    def parse_error(acl_ret, api_name: str, extra_info: str = "", accepted_errors=()) -> None:
        """非零 ACL 错误码 → RuntimeError（错误语义对齐 AclInterface.parse_error）。"""
        if isinstance(acl_ret, ctypes.c_int32):
            acl_ret = acl_ret.value
        if acl_ret == 0 or acl_ret in accepted_errors:
            logging.debug(f"Acl API Call {api_name}() Success, {extra_info}")
            return
        raise RuntimeError(f"Acl API Call {api_name}() Failed: {acl_ret}, {extra_info}")

    def _api(self, api_name: str, extra_log, *api_args, accepted_errors=()):
        fn = getattr(self._dll(), api_name)
        rt_error = fn(*api_args)
        self.parse_error(rt_error, api_name, extra_log or "", accepted_errors=accepted_errors)
        return rt_error

    def _api_ptr(self, api_name: str, extra_log, *api_args) -> ctypes.c_void_p:
        ptr = getattr(self._dll(), api_name)(*api_args)
        if not ptr:
            raise RuntimeError(f"Acl API Call {api_name}() return nullptr. {extra_log}")
        return ctypes.c_void_p(ptr)

    # ── 运行时 / 设备 ───────────────────────────────────────

    def set_device(self, device_id: int) -> None:
        """aclInit（容错 REPEAT_INITIALIZE）+ aclrtSetDevice；幂等，切卡重新 SetDevice。"""
        self._acl_init()
        if self._device_id != device_id:
            self._api("aclrtSetDevice", f"on device {device_id}", ctypes.c_int32(device_id))
        self._device_id = device_id

    @property
    def device_id(self):
        return self._device_id

    def device_count(self) -> int:
        count = ctypes.c_uint32(0)
        self._api("aclrtGetDeviceCount", "", ctypes.byref(count))
        return count.value

    def reset(self) -> None:
        """释放全部登记的 model / HBM；owns 运行时时复位设备。"""
        if getattr(self, "skip_teardown", False):
            return
        self._release_memory()
        if self._device_id is not None and self._owns_acl_runtime:
            self._api("aclrtResetDevice", f"device {self._device_id}", ctypes.c_int32(self._device_id))
            self._device_id = None

    def _acl_init(self) -> None:
        if self._acl_inited:
            return
        status = self._api("aclInit", "", None, accepted_errors=(ACL_ERROR_REPEAT_INITIALIZE,))
        self._acl_inited = True
        self._owns_acl_runtime = status == 0

    def _acl_finalize(self) -> None:
        if self._acl_inited and self._owns_acl_runtime:
            self._api("aclFinalize", "")
        self._acl_inited = False
        self._owns_acl_runtime = False

    def _release_memory(self) -> None:
        for model_id in tuple(self._models):
            try:
                self.unload_model(model_id)
            except RuntimeError:
                logging.warning(f"unload model {model_id} failed during release", exc_info=True)
        # 快照迭代：free() 内部按账本判重并摘除，直接遍历会跳过元素
        for value in tuple(self._dev_ptrs):
            self.free(value)

    def _on_exit(self) -> None:
        # __new__ 构造的测试替身（无账本属性）无资源可清理，全程 no-op
        if not hasattr(self, "_models"):
            return
        self.reset()
        self._acl_finalize()

    # ── 模型 ────────────────────────────────────────────────

    def load_model(self, om_path: str) -> int:
        """aclmdlLoadFromFile → model_id（账本登记）。"""
        model_id = ctypes.c_uint32(0)
        self._api("aclmdlLoadFromFile", f"om={om_path}", om_path.encode(), ctypes.byref(model_id))
        self._models.add(model_id.value)
        return model_id.value

    def unload_model(self, model_id: int) -> None:
        if model_id not in self._models:
            return
        self._api("aclmdlUnload", f"model_id={model_id}", ctypes.c_uint32(model_id))
        # 卸载成功才摘账本：失败保留条目，供 _release_memory / reset() 兜底重试
        self._models.discard(model_id)

    def model_io_info(self, model_id: int) -> dict:
        """desc 查询 → {"inputs": [MdlIoDesc], "outputs": [MdlIoDesc]}。

        逐条调用 Num*/SizeByIndex/GetDims/GetDataType/GetNameByIndex/GetFormat，
        快照为 MdlIoDesc 后销毁 desc（不跨调用持有）。输出名会被 atc 改写
        （如 add_0:0:output_0），消费方按索引取数、不按名字匹配。
        """
        desc = self._api_ptr("aclmdlCreateDesc", f"model_id={model_id}")
        try:
            self._api("aclmdlGetDesc", f"model_id={model_id}", desc, ctypes.c_uint32(model_id))
            return {
                "inputs": self._query_io(desc, is_input=True),
                "outputs": self._query_io(desc, is_input=False),
            }
        finally:
            self._api("aclmdlDestroyDesc", f"model_id={model_id}", desc)

    def _query_io(self, desc, is_input: bool) -> list:
        dll = self._dll()
        side = "Input" if is_input else "Output"
        count = getattr(dll, f"aclmdlGetNum{side}s")(desc)
        result = []
        for idx in range(count):
            dims = aclmdlIODims()
            self._api(
                f"aclmdlGet{side}Dims",
                f"idx={idx}",
                desc,
                ctypes.c_size_t(idx),
                ctypes.byref(dims),
            )
            name_ptr = getattr(dll, f"aclmdlGet{side}NameByIndex")(desc, ctypes.c_size_t(idx))
            name = name_ptr.decode("utf-8", errors="replace") if name_ptr else f"{side.lower()}_{idx}"
            result.append(
                MdlIoDesc(
                    name=name,
                    dims=tuple(int(d) for d in dims.dims[: dims.dimCount]),
                    dtype_id=int(getattr(dll, f"aclmdlGet{side}DataType")(desc, ctypes.c_size_t(idx))),
                    size=int(getattr(dll, f"aclmdlGet{side}SizeByIndex")(desc, ctypes.c_size_t(idx))),
                    format_id=int(getattr(dll, f"aclmdlGet{side}Format")(desc, ctypes.c_size_t(idx))),
                )
            )
        return result

    # ── HBM ─────────────────────────────────────────────────

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._api(
            "aclrtMalloc",
            f"{nbytes} bytes",
            ctypes.byref(ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_int32(ACL_MEM_MALLOC_HUGE_FIRST),
        )
        self._dev_ptrs.add(ptr.value)
        return ptr

    def free(self, dev_ptr) -> None:
        value = dev_ptr.value if isinstance(dev_ptr, ctypes.c_void_p) else dev_ptr
        if value is None or value not in self._dev_ptrs:
            return
        self._api("aclrtFree", f"ptr={value}", ctypes.c_void_p(value))
        # 释放成功才摘账本：失败保留条目，供 _release_memory / reset() 兜底重试
        self._dev_ptrs.discard(value)

    def h2d(self, np_arr: np.ndarray, size: int = None) -> ctypes.c_void_p:
        """host 数组 → HBM（malloc 按对齐后 size、memcpy 按实际字节数；账本登记）。"""
        arr = np.ascontiguousarray(np_arr)
        nbytes = int(arr.nbytes)
        alloc = int(size) if size is not None else nbytes
        if alloc < nbytes:
            raise ValueError(f"h2d alloc size {alloc} < host bytes {nbytes}")
        ptr = self.malloc(alloc)
        src = ctypes.c_void_p(arr.__array_interface__["data"][0])
        self._api(
            "aclrtMemcpy",
            f"h2d {nbytes}B",
            ptr,
            ctypes.c_size_t(alloc),
            src,
            ctypes.c_size_t(nbytes),
            ctypes.c_int32(ACL_MEMCPY_HOST_TO_DEVICE),
        )
        return ptr

    def d2h(self, dev_ptr, nbytes: int) -> np.ndarray:
        """HBM → host uint8 一维数组。"""
        buf = ctypes.create_string_buffer(int(nbytes))
        self._api(
            "aclrtMemcpy",
            f"d2h {nbytes}B",
            buf,
            ctypes.c_size_t(nbytes),
            dev_ptr,
            ctypes.c_size_t(nbytes),
            ctypes.c_int32(ACL_MEMCPY_DEVICE_TO_HOST),
        )
        return np.frombuffer(buf, dtype=np.uint8)

    def memset(self, dev_ptr, value: int, count: int) -> None:
        """设备内存填充（OmSession 输出缓冲哨兵预填，见 runner）。"""
        self._api(
            "aclrtMemset",
            f"{count}B with 0x{value:02X}",
            dev_ptr,
            ctypes.c_size_t(count),
            ctypes.c_int32(value),
            ctypes.c_size_t(count),
        )

    # ── 执行 ────────────────────────────────────────────────

    def execute(self, model_id: int, input_bufs, output_bufs) -> None:
        """组 dataset → aclmdlExecute → 拆 dataset（不缓存，避免跨轮句柄漂移）。

        input/output_bufs: [(dev_ptr, size), ...]。dataset/dataBuffer 逐次创建、
        finally 统一销毁——执行失败亦不泄漏句柄。
        """
        self._dll()
        datasets = [self._api_ptr("aclmdlCreateDataset", f"model_id={model_id}")]
        data_buffers = []
        try:
            datasets.append(self._api_ptr("aclmdlCreateDataset", f"model_id={model_id}"))
            for dataset, bufs in ((datasets[0], input_bufs), (datasets[1], output_bufs)):
                for ptr, size in bufs:
                    data_buf = self._api_ptr("aclCreateDataBuffer", f"{size}B", ptr, ctypes.c_size_t(size))
                    self._api("aclmdlAddDatasetBuffer", f"{size}B", dataset, data_buf)
                    data_buffers.append(data_buf)
            self._api(
                "aclmdlExecute",
                f"model_id={model_id}",
                ctypes.c_uint32(model_id),
                datasets[0],
                datasets[1],
            )
        finally:
            for dataset in datasets:
                self._api("aclmdlDestroyDataset", f"model_id={model_id}", dataset)
            for data_buf in data_buffers:
                self._api("aclDestroyDataBuffer", "", data_buf)
