#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""kernel/compile 编译器注册制。

- ``Compiler`` ABC：编译栈抽象（栈查询/上下文方法 + helper kernel 编译）；
  新增编译栈（如 bisheng）只需新增文件 + @register_compiler 注册类
- ``register_compiler(name)`` / ``get_compiler(name)``：注册表与栈单例获取
- ``Opc``：跨栈编排门面（switch_opc / core_type 广播 / 查询路由）——
  OperatorInterface 消费；get_compiler 完全替代（含 __getattribute__ 拆除）
  属路由契约卡（真机验证），过渡保留
"""

__all__ = ["Compiler", "Opc", "get_compiler", "register_compiler"]


# Standard Packages
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

# Third-Party Packages
from ttk.switches import get_global_storage

from ....utilities import Singleton
from ...common.device import get_npu_hw_info


class NullApiConfig:
    def __getattr__(self, name):
        class DummyContext:
            def __enter__(self, *args):
                return self

            def __exit__(self, *args):
                return False

        return DummyContext


class Compiler(metaclass=ABCMeta):
    """编译栈抽象。当前实现 tbe.TbeCompiler / asc.AscCompiler。

    ★ 编译池契约：任何 Compiler 实现运行于编译池进程（加载 tbe 等编译栈
    native lib），实现内禁止 import torch/tensorflow——这是编译/执行进程池
    隔离对编译栈的约束，新栈必须继承，不靠各栈自觉。

    栈查询/上下文方法（原 IOpc 的 12 个）由各栈实现；helper kernel 编译
    （warmup / clear_ub / clear_l1 / clear_l0）为基类具体实现——
    tik 现场编译生态与具体编译栈正交，统一经 tbe 栈的平台信息。
    """

    # ═══ 栈查询/上下文（栈实现）═══

    @abstractmethod
    def is_initialized(self) -> bool: ...

    @abstractmethod
    def set_compile_soc_info(self, soc_version, core_type="AiCore"): ...

    @abstractmethod
    def set_platform_info_res(self, device_id, res: dict): ...

    @abstractmethod
    def get_soc_spec(self, key: str): ...

    @abstractmethod
    def get_context(self): ...

    @abstractmethod
    def get_compile_info(self): ...

    @abstractmethod
    def get_param_generalization(self, op_type: str): ...

    @abstractmethod
    def do_op_tiling(self, *args, **kwargs): ...

    @abstractmethod
    def build_config(self, **kwargs): ...

    @abstractmethod
    def get_tiling_op_type(self): ...

    @property
    @abstractmethod
    def op_context(self): ...

    @property
    @abstractmethod
    def op_info(self): ...

    @property
    def api_config(self):
        return NullApiConfig()

    def get_computes(self):
        return None

    # ═══ helper kernel 编译（基类具体——经 tbe 栈平台信息）═══

    def compile_ub_clear(self):
        switches = get_global_storage()
        if switches.force_clear_ub is None:
            return
        opc = get_compiler("tbe")
        obj_file = os.path.join(switches.kernel_meta, "clear_ub.o")
        if os.path.exists(obj_file):
            os.remove(obj_file)
        full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
        core_type = "VectorCore" if get_npu_hw_info(full_soc_version).get("cv_split") else "AiCore"
        clean_val = switches.force_clear_ub
        from .helper_kernels import clear_ub

        with opc.op_context.OpContext("pre-static") as cxt:
            tensor = {"shape": (1,), "range": ((1, None),), "dtype": clean_val.dtype.name, "format": "ND"}
            attrs = {"full_soc_version": full_soc_version, "core_type": core_type, "kernel_name": "clear_ub"}
            op_info = opc.op_info.OpInfo("ClearUB", "ClearUB")
            cxt.add_op_info(op_info)
            cxt.add_addition("op_name", "ClearUB")
            clear_ub(tensor, **attrs)
        if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "clear_ub.json")):
            raise RuntimeError(f"Compile clear_ub failed. kernel or json file does not exist in {switches.kernel_meta}")

    def compile_l1_clear(self):
        switches = get_global_storage()
        if switches.force_clear_l1 is None:
            return
        opc = get_compiler("tbe")
        obj_file = os.path.join(switches.kernel_meta, "clear_l1.o")
        if os.path.exists(obj_file):
            os.remove(obj_file)
        full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
        clean_val = switches.force_clear_l1
        from .helper_kernels import clear_l1

        with opc.op_context.OpContext("pre-static") as cxt:
            tensor = {"shape": (1,), "range": ((1, None),), "dtype": clean_val.dtype.name, "format": "ND"}
            attrs = {"full_soc_version": full_soc_version, "kernel_name": "clear_l1"}
            op_info = opc.op_info.OpInfo("ClearL1", "ClearL1")
            cxt.add_op_info(op_info)
            cxt.add_addition("op_name", "ClearL1")
            clear_l1(tensor, **attrs)
        if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "clear_l1.json")):
            raise RuntimeError(f"Compile clear_l1 failed. kernel or json file does not exist in {switches.kernel_meta}")

    def compile_l0_clear(self):
        switches = get_global_storage()
        if switches.force_clear_l0 is None:
            return
        opc = get_compiler("tbe")
        obj_file = os.path.join(switches.kernel_meta, "clear_l0.o")
        if os.path.exists(obj_file):
            os.remove(obj_file)
        full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
        short_soc_version = opc.get_soc_spec("SHORT_SOC_VERSION")

        from ttk.core_modules.common.device import PLATFORM_BEFORE_DAVID

        if short_soc_version not in PLATFORM_BEFORE_DAVID:
            self._compile_l0_clear_ascendc(switches.kernel_meta, full_soc_version)
        else:
            from .helper_kernels import clear_l0

            with opc.op_context.OpContext("pre-static") as cxt:
                tensor = {"shape": (1,), "range": ((1, None),), "dtype": "float16", "format": "ND"}
                attrs = {"full_soc_version": full_soc_version, "kernel_name": "clear_l0"}
                op_info = opc.op_info.OpInfo("ClearL0", "ClearL0")
                cxt.add_op_info(op_info)
                cxt.add_addition("op_name", "ClearL0")
                clear_l0(tensor, **attrs)
        if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "clear_l0.json")):
            raise RuntimeError(f"Compile clear_l0 failed. kernel or json file does not exist in {switches.kernel_meta}")

    @staticmethod
    def _compile_l0_clear_ascendc(kernel_meta: str, full_soc_version: str):
        """
        David-generation chips have no TIK cube intrinsics (mmad/load2dv2/fixpipe),
        so the clear_l0 helper kernel is implemented with Ascend C tensor_api and
        compiled with ccec directly (see ascendc_kernels/clear_l0_ascendc.cpp).
        Tile dimensions and --npu-arch are derived from the target chip's platform
        config so the kernel adapts to different L0A/L0B/L0C sizes.
        """
        import json
        import subprocess

        import tbe.common.platform as tbe_platform

        from ttk.core_modules.common.device import get_l0_clear_tile_params

        tbe_platform.set_current_compile_soc_info(full_soc_version, "AiCore")
        core_num = tbe_platform.get_soc_spec("CORE_NUM")
        m_dim, k_dim, n_dim, npu_arch = get_l0_clear_tile_params(full_soc_version)

        ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
        asc_dir = os.path.join(ascend_home, "asc")
        os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
        src_file = os.path.join(os.path.dirname(__file__), "ascendc_kernels", "clear_l0_ascendc.cpp")
        i_obj_file = os.path.join(kernel_meta, "clear_l0.i")

        ccec_cmd = [
            "ccec",
            "-O2",
            "--asc-aicore-lang",
            src_file,
            f"-DM_DIM={m_dim}",
            f"-DK_DIM={k_dim}",
            f"-DN_DIM={n_dim}",
            "-I",
            asc_dir,
            "-I",
            os.path.join(asc_dir, "include"),
            "-I",
            os.path.join(asc_dir, "include", "basic_api"),
            f"--npu-arch={npu_arch}",
            "--cce-aicore-only",
            "-o",
            i_obj_file,
            "-mllvm",
            "--cce-aicore-jump-expand=true",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            "-mllvm",
            "-cce-aicore-dcci-before-kernel-end=false",
        ]
        result = subprocess.run(ccec_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"clear_l0 ccec compilation failed: {result.stderr}")

        ld_cmd = [
            "ld.lld",
            "-m",
            "aicorelinux",
            "-Ttext=0",
            i_obj_file,
            "-static",
            "-o",
            os.path.join(kernel_meta, "clear_l0.o"),
        ]
        result = subprocess.run(ld_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"clear_l0 ld.lld failed: {result.stderr}")
        if os.path.exists(i_obj_file):
            os.remove(i_obj_file)

        kernel_info = {
            "binFileName": "clear_l0",
            "binFileSuffix": ".o",
            "blockDim": core_num,
            "coreType": "AiCore",
            "kernelName": "clear_l0",
            "magic": "RT_DEV_BINARY_MAGIC_ELF",
        }
        with open(os.path.join(kernel_meta, "clear_l0.json"), "w", encoding="UTF-8") as f:
            json.dump(kernel_info, f, indent=4)

    def compile_warmup_kernel(self):
        switches = get_global_storage()
        if not switches.warmup:
            return
        opc = get_compiler("tbe")
        obj_file = os.path.join(switches.kernel_meta, "warmup.o")
        if os.path.exists(obj_file):
            os.remove(obj_file)
        full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
        from .helper_kernels import warmup

        with opc.op_context.OpContext("pre-static") as cxt:
            build_cfg = {"op_debug_config": "dump_cce"}
            with opc.build_config(**build_cfg):
                attrs = {"full_soc_version": full_soc_version, "kernel_name": "warmup"}
                op_info = opc.op_info.OpInfo("Warmup", "Warmup")
                cxt.add_op_info(op_info)
                cxt.add_addition("op_name", "Warmup")
                warmup(**attrs)
        if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "warmup.json")):
            raise RuntimeError(
                f"Compile warmup kernel failed. kernel or json file does not exist in {switches.kernel_meta}"
            )


# ═══ 注册表（新增编译栈 = 新增文件 + 注册类，不改正框架）═══

_COMPILER_REGISTRY: dict = {}
_COMPILER_SINGLETONS: dict = {}
_COMPILER_READY: set = set()


def register_compiler(name: str) -> Callable[[type], type]:
    """类装饰器：注册编译栈。name 即 get_compiler 查找键（"tbe" / "asc" / ...）。"""

    def decorator(cls):
        if name in _COMPILER_REGISTRY:
            raise ValueError(f"compiler name {name!r} already registered: {_COMPILER_REGISTRY[name]!r}")
        _COMPILER_REGISTRY[name] = cls
        return cls

    return decorator


def _fill_registry():
    """注册触发：import 栈实现模块（@register_compiler 副作用）。"""
    if not _COMPILER_REGISTRY:
        from . import asc, tbe  # noqa: F401


def _core_limit_res(core_type: str, core_limit) -> Optional[dict]:
    """core limit → 平台资源覆盖（None = 不限制）。"""
    if not core_limit or not isinstance(core_limit, tuple):
        return None
    ai_limit, vec_limit = core_limit
    res = {}
    if core_type == "VectorCore":
        if vec_limit:
            res["vector_core_cnt"] = str(vec_limit)
    elif core_type in ("AiCore", "CubeCore"):
        if ai_limit:
            res["ai_core_cnt"] = str(ai_limit)
    else:
        if ai_limit:
            res["ai_core_cnt"] = str(ai_limit)
        if vec_limit:
            res["vector_core_cnt"] = str(vec_limit)
    return res or None


def _ensure_ready(comp: "Compiler") -> bool:
    """栈首次物化：soc 信息 + core limit 就位。

    平台未探测（dev_plat 缺省/AUTO——测试或导入期上下文）时跳过并返回
    False，下次 get_compiler 重试；真实运行序中 detect_platform 先于任何
    编译栈使用，首次有效获取即就位。
    """
    if not comp.is_initialized():
        return True  # 栈不可用（未安装）——无就位动作，与旧 _all_opc_invoke 跳过一致
    switches = get_global_storage()
    dev_plat = getattr(switches, "dev_plat", None)
    if not dev_plat or dev_plat == "AUTO":
        return False
    comp.set_compile_soc_info(dev_plat, "AiCore")
    res = _core_limit_res("AiCore", switches.core_limit)
    if res:
        comp.set_platform_info_res(0, res)
    return True


def get_compiler(name: str = "tbe") -> Compiler:
    """按名取编译栈单例（首次物化时 soc 信息就位）。

    路由现状：tbe vs asc 由 OpInfoKeeper.impl_type_of(op_name)
    （算子实现目录）决定，消费方 OperatorInterface._switch_opc 经 Opc 门面
    切换；name 缺省 "tbe"（helper kernel 生态默认栈）。
    """
    _fill_registry()
    try:
        cls = _COMPILER_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown compiler {name!r}; registered: {sorted(_COMPILER_REGISTRY)}") from None
    if name not in _COMPILER_SINGLETONS:
        _COMPILER_SINGLETONS[name] = cls()
    if name not in _COMPILER_READY and _ensure_ready(_COMPILER_SINGLETONS[name]):
        _COMPILER_READY.add(name)
    return _COMPILER_SINGLETONS[name]


class Opc(metaclass=Singleton):
    """跨栈编排门面：编译栈切换（switch_opc）+ SoC / core limit 广播 +
    栈查询路由到当前选中栈。OperatorInterface 消费。"""

    def __init__(self):
        self._core_type = None
        self._opc = None

        _fill_registry()
        for name in _COMPILER_REGISTRY:
            get_compiler(name)  # 与旧行为一致：急切物化全部已注册栈

        try:
            self.core_type = None
        except BaseException as e:
            logging.critical(f"Opc init failed: {e}")
            raise e

    def __getattribute__(self, item):
        if item in (
            "get_compile_info",
            "get_tiling_op_type",
            "get_param_generalization",
            "get_soc_spec",
            "do_op_tiling",
            "build_config",
            "op_context",
            "op_info",
            "api_config",
        ):
            return getattr(self._opc, item)
        return super().__getattribute__(item)

    @property
    def core_type(self) -> str:
        return self._core_type

    @core_type.setter
    def core_type(self, val: Optional[str]):
        self._core_type = val or "AiCore"
        self._core_type = self._core_type or "AiCore"
        dev_plat = get_global_storage().dev_plat
        logging.debug(f"Setting soc version to {dev_plat} for {self._core_type}")
        self._all_opc_invoke("set_compile_soc_info", dev_plat, self._core_type)
        self._apply_core_limit()

    def _apply_core_limit(self):
        res = _core_limit_res(self._core_type, get_global_storage().core_limit)
        if res:
            logging.info(f"Applying core limit {get_global_storage().core_limit} for {self._core_type}: {res}")
            self._all_opc_invoke("set_platform_info_res", 0, res)

    def switch_opc(self, opc_type: str):
        if opc_type not in _COMPILER_REGISTRY:
            raise ValueError(f"Invalid opc type: {opc_type}")
        self._opc = get_compiler(opc_type)

    def _get_any_opc(self) -> Optional[Compiler]:
        for name in _COMPILER_REGISTRY:
            opc = get_compiler(name)
            if opc.is_initialized():
                return opc
        return None

    def _all_opc_invoke(self, func, *args, **kwargs):
        for name in _COMPILER_REGISTRY:
            opc: Compiler = get_compiler(name)
            if not opc.is_initialized():
                continue
            getattr(opc, func)(*args, **kwargs)
