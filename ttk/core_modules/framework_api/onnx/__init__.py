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
onnx e2e 子包：OnnxAdapter + OnnxOpHandle。

执行链路：CSV（``onnx.<OpType>``，可带显式 opset 段 ``onnx.<opset>.<OpType>``）
→ detect_framework → OnnxAdapter / NpuOnnxBackend → 参数解析
（param_extractor，schema 单源）→ 输入生成（numpy 直通）→ eager_call（会话缓存）
→ model_builder 组模 → om_compiler（atc）→ OmSession（aclmdl 执行）→
tuple[np.ndarray] → 既有比对/dump/结果列。

会话缓存：结构键（算子+opset+输入形态+属性+soc+dev_id）→
OmSession 的进程级 LRU；命中后经「数组身份 → 内容指纹」双层判定复用（同签名
不同值用例不脏命中），上限驱逐 + 同键替换时卸载旧会话（HBM 不无限累积）。

子包纪律：全树零 torch/tf import（test_structure 钉住，onnx 域是 numpy 直通域）；
ACL so 加载只发生在 runtime.py 的懒 CDLL（模块 import 零 .so 触碰）。
"""

__all__ = ["ADAPTER", "OnnxAdapter", "OnnxOpHandle"]

import contextlib
import hashlib
import logging
import os
import tempfile
from collections import OrderedDict

from ..adapter import FrameworkAdapter

_log = logging.getLogger(__name__)

# CSV 布局列（ND 连续语义，填写即建模期显式失败）
_LAYOUT_FIELDS = ("tensor_formats", "tensor_storage_shapes", "tensor_view_strides", "tensor_view_offsets")


class OnnxOpHandle:
    """load_api 产物：api 级句柄（进程缓存）。

    持 op_type / APIParamInfo（eager_call 反解用）/ opset（api_name 显式
    版本段，None = 裸名建模用环境最高）；attach_backend 注入
    backend 引用（soc_version + dev_id 由 wrap_eager 时机转交——backend.set_device
    仅记录 dev_id，ACL set_device 由 runner 惰性执行）。
    ``__name__ = op_type``（is_inplace 尾下划线判定天然 False）。
    """

    def __init__(self, api_name: str, op_type: str, param_info, opset=None):
        self.api_name = api_name
        self.op_type = op_type
        self.param_info = param_info
        self.opset = opset
        self.__name__ = op_type
        self._backend = None
        self._soc_version = None

    def attach_backend(self, backend) -> None:
        self._backend = backend

    @property
    def soc_version(self) -> str:
        """DSMI 完整芯片名（atc --soc_version 直传值；首用缓存）。"""
        if self._soc_version is None:
            dev_id = self.dev_id
            if dev_id is None:
                dev_id = 0
            self._soc_version = self._backend.device_name(dev_id)
        return self._soc_version

    @property
    def dev_id(self):
        return getattr(self._backend, "_dev_id", None)

    def __repr__(self):
        return f"OnnxOpHandle({self.api_name})"


class OnnxAdapter(FrameworkAdapter):
    """onnx 域适配器：eager ≡ 静态图（cst）；dyn/aclgraph 快速失败。"""

    framework = "onnx"

    # 进程级会话缓存（结构键 → OmSession；内容指纹防脏命中，
    # LRU 上限防 HBM 无限累积）
    _sessions: "OrderedDict[str, object]" = OrderedDict()
    _MAX_SESSIONS = 4
    _acl = None

    # ── 三件套 ──────────────────────────────────────────────

    def load_api(self, api_name):
        handle = _load_handle(api_name)
        return handle, False

    def extract_params(self, api_name):
        from .param_extractor import extract_onnx_params

        return extract_onnx_params(api_name)

    def bind_tensors(self, testcase, backend, dev_tensors, raw_inputs):  # noqa: ARG002
        """L2 绑定（onnx：无 format cast / 无 mutable 复位）+ 布局守卫。

        CSV 布局列（formats/storage/strides/offsets）不支持——ND 连续语义
        （H2D 按连续字节上板），填了显式报 ONNX_LAYOUT_UNSUPPORTED。
        """
        filled = [name for name in _LAYOUT_FIELDS if getattr(testcase, name, None)]
        if filled:
            raise ValueError(
                f"ONNX_LAYOUT_UNSUPPORTED: onnx e2e 仅支持 ND 连续布局，用例 "
                f"[{getattr(testcase, 'testcase_name', '?')}] 填写了 {filled}；请清空这些列"
                "（formats/storage_shapes/strides/offsets）"
            )
        return dev_tensors

    # ── 执行 ────────────────────────────────────────────────

    def wrap_eager(self, resolved, api_name, backend):  # noqa: ARG002
        # soc/dev_id 注入点：execute.py 保证 backend.set_device 先于本调用
        resolved.attach_backend(backend)
        return resolved

    def eager_call(self, resolved, args, kwargs, is_tensor_method):  # noqa: ARG002
        return self._execute(resolved, args, kwargs)

    def eager_graph_alias(self):
        return "cst"  # om 恒为静态图：eager ≡ cst（worker 去重依据）

    def run_graph(
        self,
        testcase,
        backend,
        dev_id,
        switches,
        plan,
        resolved,
        is_tensor_method,
        is_inplace,
        raw_inputs,
        dynamic,
        is_aclgraph=False,
        deterministic_level=None,
    ):
        """图模式执行器：cst 委托共享 _execute_eager（om 执行即静态图）。

        返回 4 元组 (nps, perf, det_status, err)——对齐 worker._run_graph_phase
        的图执行器协议（torch/tf 的 execute_graph 同构；err 位恒 None，执行
        异常经 _run_graph_phase 统一转 GRAPH_EXEC_FAILURE）。
        """
        if is_aclgraph:
            raise ValueError("ONNX_ACLGRAPH_UNSUPPORTED: onnx e2e 不支持 aclgraph 模式")
        if dynamic:
            raise ValueError("ONNX_DYNAMIC_GRAPH_UNSUPPORTED: onnx e2e 不支持动态 shape 图模式")
        from ..execute import _execute_eager  # 延迟 import 防环

        result_nps, perf, det_status = _execute_eager(
            testcase,
            backend,
            dev_id,
            switches,
            plan,
            resolved,
            is_tensor_method,
            is_inplace,
            raw_inputs,
            deterministic_level=deterministic_level,
        )
        return result_nps, perf, det_status, None

    # ── 会话执行 ────────────────────────────────────────────

    def _execute(self, handle, args, kwargs):
        backend = handle._backend
        if backend is None or not backend.is_npu():
            raise RuntimeError(
                "ONNX_CPU_EXECUTION_UNSUPPORTED: onnx e2e 设备执行需要 NPU 后端（--backend cpu 仅用于 golden/输入生成；om 执行经 atc→aclmdl）"
            )
        from .model_builder import build_input_specs

        input_specs, host_inputs = build_input_specs(handle.param_info, args)
        key = self._session_key(handle, input_specs, kwargs)
        session = self._sessions.get(key)
        if session is not None:
            if session.matches_inputs(host_inputs):
                session.rebind_inputs(host_inputs)
                self._sessions.move_to_end(key)
                return session.execute_round()
            # 同签名不同值：旧会话驱逐重建
            _log.info("onnx session %s evicted (input values changed)", handle.op_type)

        session = self._create_session(handle, input_specs, host_inputs, kwargs)
        self._store_session(key, session)
        return session.execute_round()

    def _session_key(self, handle, input_specs, kwargs) -> str:
        raw = "|".join(
            [
                handle.op_type,
                str(handle.opset),  # 显式版本段区分同名算子不同 opset（om 不同不得复用会话）
                handle.param_info.source,
                repr([(name, shape, dtype) for name, shape, dtype in input_specs]),
                repr(sorted((name, repr(value)) for name, value in kwargs.items())),
                handle.soc_version,
                str(handle.dev_id),
            ]
        )
        return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324  # 非安全用途：会话结构键

    @staticmethod
    def _case_dir_name(handle) -> str:
        """onnx_om 按用例名 + opset 归组（对齐 log/msprof 的 per-case 目录惯例：
        <root>/onnx_om/<testcase_name>/<opset>/model.om）。

        worker 进程名即用例名（process_context.change_name →
        set_process_name，编译只发生在 worker 进程）；非 worker 环境
        （单测直调 / MainProcess）回退 op_type 归组（legacy 布局）。
        """
        import multiprocessing

        name = getattr(multiprocessing.current_process(), "name", "") or ""
        if name.startswith("MP "):
            name = name[3:]
        if name and name != "MainProcess":
            return name
        return handle.op_type

    def _create_session(self, handle, input_specs, host_inputs, kwargs):
        from .model_builder import build_model_bytes, resolve_opset
        from .om_compiler import compile_om
        from .runner import OmSession

        model_bytes = build_model_bytes(handle.op_type, input_specs, kwargs, opset=handle.opset)
        root_path = self._root_path()
        # onnx_om/<testcase_name>/<opset>/model.om：opset 层区分同名用例的
        # 显式版本段与裸名（env 最高）——同名不同 opset 互不覆盖
        work_dir = os.path.join(root_path, "onnx_om", self._case_dir_name(handle), str(resolve_opset(handle.opset)))
        os.makedirs(work_dir, exist_ok=True)
        fd, tmp_onnx = tempfile.mkstemp(prefix="ttk_onnx_", suffix=".onnx", dir=work_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(model_bytes)
            om_path = compile_om(
                tmp_onnx,
                {name: shape for name, shape, _dtype in input_specs},
                handle.soc_version,
                work_dir,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_onnx)

        acl = self._acl_interface()
        return OmSession(acl, handle.dev_id or 0, om_path, host_inputs, handle.op_type)

    def _store_session(self, key, session) -> None:
        old = self._sessions.get(key)
        if old is not None and old is not session:
            old.unload()
        self._sessions[key] = session
        self._sessions.move_to_end(key)
        while len(self._sessions) > self._MAX_SESSIONS:
            _evicted_key, evicted = self._sessions.popitem(last=False)
            evicted.unload()

    @staticmethod
    def _root_path() -> str:
        from ttk.switches import get_global_storage

        return getattr(get_global_storage(), "root_path", os.getcwd())

    @classmethod
    def _acl_interface(cls):
        if cls._acl is None:
            from .runtime import AclModelInterface

            cls._acl = AclModelInterface()
        return cls._acl


_HANDLE_CACHE: dict = {}


def _load_handle(api_name: str) -> OnnxOpHandle:
    handle = _HANDLE_CACHE.get(api_name)
    if handle is not None:
        return handle
    from .param_extractor import extract_onnx_params, parse_onnx_api_name

    parsed = parse_onnx_api_name(api_name)
    if parsed is None:
        raise ValueError(f"invalid onnx api_name: {api_name!r}（应为 onnx.<OpType> 或 onnx.<opset>.<OpType>）")
    op_type, opset = parsed
    param_info = extract_onnx_params(api_name)
    if param_info is None:
        raise ValueError(f"onnx op not found in environment registry: {api_name}")
    handle = OnnxOpHandle(api_name, op_type, param_info, opset=opset)
    _HANDLE_CACHE[api_name] = handle
    return handle


ADAPTER = OnnxAdapter()
