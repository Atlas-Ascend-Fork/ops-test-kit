#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/xpu.py：三方标杆执行器 + 参数绑定。

- XpuResolver 注册制（加 provider = 加一行装饰器）
- invoke_third_party + bind_params：三方 API 直调的绑定单份实现
  （整仓部署后 remote server 同源使用）
- xpu_mode / extract_third_party / collect_third_party：worker 三方标杆编排层
  与 remote client 的稳定查询面（三方采集仅 remote：ttk.conf.yaml remote.endpoints）

三套装配分层（并存不合并）：
  ① spec 方法注入（invoke.py 私有 _bind_by_name）——spec 作者的 golden/compare
  ② invoke_third_party（本文件 bind_params）——三方 API 直调（server + spec 类形态）
  ③ remote client（HTTP/npz 传输）——spec third_party 声明经远端执行
"""

__all__ = [
    "generate_gmm_alltoallv_matrix",
    "XpuResolver",
    "get_xpu_resolver",
    "register_xpu_resolver",
    "invoke_third_party",
    "UnknownParamError",
    "bind_params",
    "xpu_mode",
    "XPU_DATA",
    "XPU_PERF",
    "XpuDispatch",
    "overlap_deadline_s",
    "xpu_join_timeout",
    "snapshot_xpu_inputs",
    "xpu_overlap_enabled",
    "extract_third_party",
    # manual_xpu 装载/落盘（--dump xpu 回放契约）
    "ManualXpuLoadError",
    "load_manual_third_parties",
    "has_manual_xpu_source",
    "manual_xpu_results",
    "dump_manual_xpu_archive",
    "dump_xpu_slot_files",
    "collect_third_party",
]

# Standard Packages
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

# 数据位模式（need_data / need_perf 按位组合）——正本在 ttk.remote（协议层
# client/server 共享），此处 re-export 保持 worker 侧统一从 common.xpu 导入
from ...remote import DATA as XPU_DATA  # noqa: F401
from ...remote import PERF as XPU_PERF
from ...remote import xpu_mode
from ...utilities import load_array, resolve_numpy_dtypes

# manual 装载在三方结果中的伪 provider 名（对齐远端产出的 dict 形态，屏显/报告零改动）
MANUAL_XPU_PROVIDER = "manual_binary"

# 重叠请求的 NPU 预算默认缓冲（--proc-timeout 未配置时）
_DEFAULT_NPU_BUDGET_S = 600.0

# collect 收割上界的固定余量：重试退避/建连等残余开销
_JOIN_SLACK_S = 300.0


def xpu_overlap_enabled() -> bool:
    """重叠总闸（ttk.conf.yaml remote 段 xpu_overlap，缺省 true）——XPU 服务端
    负载紧张时运维全局关闭，回到串行时序。"""
    from ...config.loader import get_remote_config

    try:
        config = get_remote_config()
    except Exception:  # noqa: BLE001  配置不可用时保守取默认（不因配置问题关掉功能）
        return True
    return bool(getattr(config, "xpu_overlap", True)) if config is not None else True


def overlap_deadline_s(switches) -> float:
    """重叠请求的 deadline 预算：dispatch_deadline + NPU 预算。

    NPU 预算锚点 ``--proc-timeout``（SWITCHES 默认 3600s = 用例全程上界，
    语义即 NPU 段上界；测试桩无该字段时取 600s 缓冲）——重叠请求横跨整个
    NPU 执行期，deadline 须覆盖。XPU 侧自身的执行上界由服务端 run_deadline
    兜底（与等待期无关），客户端预算只需覆盖"等待 NPU + XPU 执行 + 传输"。
    """
    from ...config.loader import get_remote_config

    try:
        config = get_remote_config()
        dispatch_deadline = float(getattr(config, "dispatch_deadline_s", 300.0) or 300.0) if config else 300.0
    except Exception:  # noqa: BLE001
        dispatch_deadline = 300.0
    proc_timeout = getattr(switches, "proc_timeout", None)
    npu_budget = float(proc_timeout) if proc_timeout else _DEFAULT_NPU_BUDGET_S
    return dispatch_deadline + npu_budget


def xpu_join_timeout(deadline_s: float) -> float:
    """重叠句柄 collect 的 join 上界：2×deadline + 余量。

    线程上界 ≈ deadline（重试主循环）+ 一份 socket timeout（deadline
    之后的末次在途请求）；超界 collect 抛 TimeoutError，不无限等待。
    """
    return 2.0 * float(deadline_s) + _JOIN_SLACK_S


def snapshot_xpu_inputs(values):
    """输入深快照（numpy 叶子 copy、嵌套容器递归、其余原样）。

    重叠路径的竞争防线：后台线程序列化读输入 vs NPU 执行原地改写输入
    （geir 图执行 backing storage / e2e inplace 算子）。派生与快照必须在
    主线程 dispatch 之前完成。
    """
    import numpy

    if isinstance(values, (list, tuple)):
        return type(values)(snapshot_xpu_inputs(v) for v in values)
    if isinstance(values, numpy.ndarray):
        return values.copy()
    return values


class XpuDispatch:
    """三方采集统一句柄（dispatch/collect 两段式，NPU 执行期重叠）。

    ready：同步路径（golden-XPU / manual 源 / 自动降级 / 总闸关闭）——dispatch
    时已内联完成采集，collect 即取。background：重叠路径——现同步采集函数
    原样入 daemon 线程，结果/异常装箱，collect 时 join 收割；daemon 保证 NPU
    失败早退不背长等待、进程退出不挂死（线程上界 = join_timeout）。

    collect 幂等（首查缓存）；线程内异常在 collect 处重放；带 join_timeout
    的 background 形态收割有界，超时抛 TimeoutError。
    """

    def __init__(self):
        self._result = None
        self._error = None
        self._thread = None
        self._collected = False
        self._discarded = False
        self._join_timeout = None

    @property
    def pending(self) -> bool:
        """background 形态且未收割（通路据此发 OnXpuCollect 阶段通知）。"""
        return self._thread is not None and not self._collected

    @classmethod
    def ready(cls, result) -> "XpuDispatch":
        handle = cls()
        handle._result = result
        handle._collected = True
        return handle

    @classmethod
    def background(cls, fn, *args, join_timeout=None, **kwargs) -> "XpuDispatch":
        import threading

        handle = cls()
        handle._join_timeout = join_timeout

        def _run():
            try:
                handle._result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001  装箱重放，collect 侧统一处理
                handle._error = exc

        handle._thread = threading.Thread(target=_run, daemon=True, name="xpu-dispatch")
        handle._thread.start()
        return handle

    def collect(self):
        """收割（幂等）：ready 即取；background join 后异常重放；
        join_timeout 超界抛 TimeoutError（线程 daemon 自然消亡）。"""
        if self._discarded:
            raise RuntimeError("XpuDispatch already discarded")
        if not self._collected:
            if self._thread is not None:
                self._thread.join(self._join_timeout)
                if self._thread.is_alive():
                    self._collected = True
                    raise TimeoutError(
                        f"XPU collect join timeout ({self._join_timeout:.0f}s): background dispatch stuck"
                    )
            self._collected = True
            if self._error is not None:
                raise self._error
        return self._result

    def discard(self, timeout: float = 5.0) -> None:
        """弃置句柄：短超时 join 后放弃（daemon 线程自然消亡）。

        供编排分支提前结束时调用，不阻塞失败上报；弃置后 collect 抛错。
        """
        if self._collected:
            return
        self._discarded = True
        if self._thread is not None:
            self._thread.join(timeout)


class UnknownParamError(ValueError):
    """A parameter name is neither a known input/attribute nor
    self / a framework-reserved name / **kwargs.

    Turns a misspelled parameter name from a silent mis-computation into a
    loud failure.
    """


def extract_third_party(xpu_results, priority):
    """从 priority provider 取 outputs（采集 → 比对的数据桥）。

    fail-closed：无 results / 无 priority / 非 PASS / 无 outputs → None
    （cross_check → GOLDEN_FAILURE）。priority 为 str 或 tuple（多 provider
    优先级序），逐个尝试首个 PASS 且有 outputs 的 provider。
    """
    if not xpu_results or priority is None:
        return None
    if isinstance(priority, str):
        priority = (priority,)
    for provider in priority:
        entry = xpu_results.get(provider, {})
        if entry.get("status") == "PASS" and "outputs" in entry:
            return entry["outputs"]
    return None


def generate_gmm_alltoallv_matrix(a_array_val, exp_per_card, seed):
    """GMM alltoallv 的 token 分布矩阵生成（HCCL 通信类算子共享，E2E 多卡 worker 同源消费）。"""
    import numpy

    n = len(a_array_val)
    rng = numpy.random.default_rng(seed)
    total = sum(a_array_val)
    if total % n != 0:
        return [[total // n] * (exp_per_card * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in a_array_val:
        if a % n != 0:
            return [[col_sum // (exp_per_card)] * (exp_per_card * n) for _ in range(n)]
        k = a // n
        k_values.append(max(k, exp_per_card))
    blocks = []
    for k in k_values:
        block = numpy.zeros((exp_per_card, n), dtype=int)
        for col in range(n):
            counts = rng.multinomial(k - exp_per_card, [1.0 / exp_per_card] * exp_per_card)
            block[:, col] = counts + 1
        blocks.append(block)
    tmp = numpy.vstack(blocks)
    return [list(col) for col in zip(*tmp)]


class XpuResolver(ABC):
    """op_name → 三方 API 搜索（仅 Kernel/ACLNN 通路；E2E 用 resolve_api——
    api 本身就是 torch.xxx，无需翻译）。"""

    @abstractmethod
    def resolve(self, op_name: str, op_type: str) -> tuple: ...  # (callable, resolved_name)


_XPU_RESOLVER_REGISTRY: dict = {}


def register_xpu_resolver(provider: str) -> Callable[[type], type]:
    def decorator(cls):
        _XPU_RESOLVER_REGISTRY[provider] = cls()
        return cls

    return decorator


def get_xpu_resolver(provider: str) -> XpuResolver:
    try:
        return _XPU_RESOLVER_REGISTRY[provider]
    except KeyError:
        raise ValueError(f"Unknown XPU provider {provider!r}; registered: {sorted(_XPU_RESOLVER_REGISTRY)}") from None


# ── bind_params：XPU 执行参数装配的单份契约实现 ──
# （server 直接 import 本模块，client/server 同源）


def _format_device(device):
    return device


def _self_is_instance_slot(func) -> bool:
    """签名中的 ``self`` 形参是否为实例占位（而非 API 真实参数名）。

    仅未绑定的 ``__init__`` 如此——其首参由 ``cls(*args)`` 填充，永不来自
    输入/属性池。普通函数（含 aten 惯例 ``def f(self, ...)`` 形态的三方
    API/插件函数）的 ``self`` 是真实参数名，须按名绑定；绑定方法的签名已
    由 Python 去掉实例参，不在此列。
    """
    import inspect

    return not inspect.ismethod(func) and getattr(func, "__name__", None) == "__init__"


def _self_kwarg_conflicts(func) -> bool:
    """``self=`` 关键字是否会与实例位冲突（self 名输入不可经 **kwargs 送达）。

    绑定方法（实例已占 ``self`` 位，``inst(self=...)`` → multiple values
    for argument 'self'）与未绑定 ``__init__``（``cls(**{'self': ...})``
    同理）都冲突；普通函数无实例位，安全。
    """
    import inspect

    return not inspect.isfunction(func) or _self_is_instance_slot(func)


def bind_params(func, name_to_value: dict, device: Optional[str] = None, warn_leftover: bool = True) -> tuple:
    """Bind a name->value map onto ``func``'s signature BY PARAMETER NAME.

    Returns ``(args, kwargs)``. Each declared param (except an unbound-
    ``__init__`` instance slot / VAR) is resolved by:
      * name in ``name_to_value`` -> take value (positional before ``*``, keyword after);
      * framework-reserved name (``device``) -> inject ``device`` (when not None);
      * has a default -> use the default: pre-star params get the default appended
        positionally so later values stay in their declared slots (skipping the
        slot would shift every later value one param to the left); post-star
        params are simply skipped (Python applies the default);
      * otherwise -> :class:`UnknownParamError` (neither input/attr nor defaulted).
    A ``self`` parameter is a REAL API param name for plain functions (aten
    convention, e.g. ``torch._foreach_addcmul(self, tensor1, ...)``) and binds
    from the pool like any name; only an unbound ``__init__``'s ``self`` (the
    instance slot, filled by ``cls(*args)``) is skipped.
    ``*args`` collects unconsumed entries (in insertion order) as positional
    args; with ``**kwargs`` present, non-``self`` leftovers go to ``**kwargs``
    instead (no double-feeding). A pool entry named ``self`` (an input whose
    schema name is literally ``self``, e.g. aclnnReal's only input) is
    delivered positionally via ``*args``; it goes through ``**kwargs`` only for
    plain functions (no instance slot to collide with), never for bound
    methods / unbound ``__init__``. ``warn_leftover`` controls whether
    unconsumed/undeliverable entries log a warning (default True).
    """
    import inspect

    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_keyword = False
    has_var_positional = False
    consumed = set()
    for name, param in sig.parameters.items():
        # self 双语义：未绑定 __init__ 的首参 self 是实例占位（cls(*args) 填充），
        # 跳过；普通函数签名里的 self 是 API 真实形参名（aten 惯例
        # torch._foreach_addcmul(self, tensor1, ...)），按名绑定——一律跳过会把
        # 第一个输入静默丢掉（aclnnReal 等 self 名输入的远端三方执行即由此回归）。
        if name == "self" and _self_is_instance_slot(func):
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            seen_star = True
            has_var_positional = True
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            seen_star = True
        if name in name_to_value:
            consumed.add(name)
            if seen_star:
                kwargs[name] = name_to_value[name]
            else:
                args.append(name_to_value[name])
        elif name == "device" and device is not None:
            kwargs[name] = _format_device(device)
        elif param.default is not inspect.Parameter.empty:
            if not seen_star:
                # Pre-star optional param missing from the pool: append the default
                # to hold the positional slot — skipping it would shift every later
                # value one param to the left (e.g. alpha's value landing on gamma).
                args.append(param.default)
            # Post-star optional param: leave it to Python (skip, use the default)
        else:
            raise UnknownParamError(
                f"parameter {name!r} of {getattr(func, '__name__', func)} has no value "
                f"(not in inputs/attrs, no default)"
            )
    leftover = {k: v for k, v in name_to_value.items() if k not in consumed}
    if has_var_positional and not has_var_keyword:
        # 仅 *args：全部 leftover（含 self 名输入）按插入序整体位置送达
        args.extend(leftover.values())
        return tuple(args), kwargs
    unset = object()
    self_value = leftover.pop("self", unset)
    if leftover:
        if has_var_keyword:
            kwargs.update(leftover)
        elif warn_leftover:
            logging.warning("bind_params: unconsumed pool entries %s", sorted(leftover))
    if self_value is not unset:
        if has_var_positional:
            # *args + **kwargs 并存：self 只走 *args——bound 方法的 self= 关键字
            # 会与实例位冲突（multiple values for argument 'self'）
            args.append(self_value)
        elif has_var_keyword and not _self_kwarg_conflicts(func):
            kwargs["self"] = self_value
        elif warn_leftover:
            logging.warning(
                "bind_params: pool entry 'self' undeliverable to %s (no *args; **kwargs collides with the instance slot)",
                getattr(func, "__name__", func),
            )
    return tuple(args), kwargs


def invoke_third_party(func, named: dict, attrs: dict, device, param_order: Optional[list] = None):
    """三方 API 装配执行（装配分层 ②：Remote server 端 + spec third_party 类形态共用）。

    merged = attrs ∪ named → param_order 排序（def.cpp 序，Kernel/ACLNN/GEIR
    各自提供）→ bind_params 签名匹配（具名对位，未消耗 UnknownParamError）→
    func/类双形态（类形态经 bind_params 双阶段绑定 + device 注入）→ raw outputs。
    """
    import inspect

    merged = dict(attrs or {})
    merged.update(named)
    if param_order:
        from collections import OrderedDict

        ordered = OrderedDict()
        for name in param_order:
            if name in merged:
                ordered[name] = merged[name]
        for k, v in merged.items():
            if k not in ordered:
                ordered[k] = v
        merged = ordered
    if inspect.isclass(func):
        if func.__init__ is object.__init__:  # 无自定义 __init__: 直接实例化
            inst = func()
        else:
            try:
                init_args, init_kwargs = bind_params(func.__init__, merged, device=device, warn_leftover=False)
            except UnknownParamError:
                init_args, init_kwargs = (), {}
            inst = func(*init_args, **init_kwargs)
        call_args, call_kwargs = bind_params(inst.__call__, merged, device=device)
        return inst(*call_args, **call_kwargs)
    args, kwargs = bind_params(func, merged, device=device)
    return func(*args, **kwargs)


class ManualXpuLoadError(ValueError):
    """manual_xpu_binaries 装载失败（shape 与声明不符等）。"""


def manual_xpu_results() -> dict:
    """manual 装载的 xpu_results 伪条目（status=PASS，api 标记来源）。"""
    return {MANUAL_XPU_PROVIDER: {"status": "PASS", "api": "manual_xpu_binaries"}}


def has_manual_xpu_source(testcase, manual_case=None) -> bool:
    """manual 三方输出源是否已声明（不读文件）。

    判定口径与 load_manual_third_parties 的双来源一致：CSV
    ``flat_manual_xpu_binaries`` 非空，或 manual-data case 目录含 xpu 槽位
    文件。适用于 dispatch 前置探测等只需要"有没有源"的场景；文件内容仍由
    load_manual_third_parties 在消费点装载。
    """
    if getattr(testcase, "flat_manual_xpu_binaries", None):
        return True
    return manual_case is not None and bool(getattr(manual_case, "has_third_parties", False))


def load_manual_third_parties(testcase, manual_case=None, device_outputs=None) -> Optional[list]:
    """manual 三方输出本地装载（--dump xpu 回放契约的对侧），双来源：

    ① CSV ``manual_xpu_binaries`` 显式路径（NPU_DUMP_PATH 归档回放形态）；
    ② manual-data case 目录槽位（``--manual-data-dirs`` 回放形态）——
       ``xpu_{provider}_{idx}_{dtype}[_shape_{dims}].{ext}`` 按 idx 对位
       （文件名一致即可读取，多 provider 同 idx 取字典序首个）。

    两来源均未声明返回 None（调用方走远端采集）；已声明按 flat 输出槽位对齐
    装载：路径数不足的槽位 / 动态 shape 槽（None）补 None（该输出无三方数据，
    cross_check 判据按既有占位语义处理）。shape/dtype 槽取声明槽
    （``flat_output_shapes``/``flat_output_dtypes``，kernel/aclnn）；e2e 输出
    为执行派生（无声明槽），经 ``device_outputs``（设备输出）取 shape 与 dtype，
    口径同 golden 的 device_outputs 装载。声明/device_outputs 缺席时按文件名 shape
    段（bin 自描述）+ 文件名 dtype 装载——manual xpu 回放不依赖当次执行输出。
    """
    paths = getattr(testcase, "flat_manual_xpu_binaries", None)
    saved_shapes = ()
    filename_dtypes = ()
    if not paths and manual_case is not None and getattr(manual_case, "has_third_parties", False):
        paths = manual_case.third_party_paths()
        saved_shapes = getattr(manual_case, "third_party_saved_shapes", lambda: ())()
        filename_dtypes = getattr(manual_case, "third_party_dtypes", lambda: ())()
    if not paths:
        return None
    shapes = getattr(testcase, "flat_output_shapes", None)
    if shapes is None:
        shapes = getattr(testcase, "flat_output_view_shapes", None)
    if not shapes and device_outputs:
        # e2e：输出无声明槽，shape/dtype 取设备输出（值不参与，仅槽位元数据）
        shapes = tuple(tuple(r.shape) if r is not None and hasattr(r, "shape") else None for r in device_outputs)
    shapes = shapes or ()
    dtypes = resolve_numpy_dtypes(getattr(testcase, "flat_output_dtypes", ()) or ())
    if not dtypes and device_outputs:
        dtypes = tuple(
            getattr(r, "dtype", None) if r is not None and hasattr(r, "dtype") else None for r in device_outputs
        )
    name = getattr(testcase, "testcase_name", None) or getattr(testcase, "op_name", None)
    arrays = []
    for index in range(max(len(paths), len(shapes))):
        path = paths[index] if index < len(paths) else None
        declared_shape = shapes[index] if index < len(shapes) else None
        saved_shape = saved_shapes[index] if index < len(saved_shapes) else None
        if path is None or (declared_shape is None and saved_shape is None):
            arrays.append(None)
            continue
        logging.info("Reading manual XPU file: %s ...", path)
        dtype = dtypes[index] if index < len(dtypes) else None
        filename_dtype = filename_dtypes[index] if index < len(filename_dtypes) else None
        arrays.append(_load_manual_xpu_slot(index, (path, declared_shape, saved_shape), dtype, filename_dtype, name))
    return arrays


def _load_manual_xpu_slot(index, slot, dtype, filename_dtype, name):
    """单槽 xpu 装载：声明/文件名 shape 决议 → load_array → 物理形态核对。

    slot = (path, declared_shape, saved_shape)。load_array 返回 dtype 物理
    存储形态（complex32 尾维 [2] fp16 实/虚、int1 位打包、4bit 解包）——校验
    须按物理形态比，口径同 manual-data 的 physical_array_spec；错文件（元素数
    不符）经 _reshape_storage 宽容回退后由本检查拦截。文件名 shape 段自描述
    路径（无声明槽）按段本身（物理存储 shape）比对。
    """
    from .dump.manual_data import physical_array_spec

    path, declared_shape, saved_shape = slot
    resolved_shape = declared_shape if declared_shape is not None else (saved_shape or (1,))
    if dtype is None and declared_shape is None:
        dtype = filename_dtype
    array = load_array(path, dtype, resolved_shape)
    if declared_shape is not None:
        _, storage_shape = physical_array_spec(dtype, declared_shape)
        expected_storage = storage_shape
    else:
        expected_storage = resolved_shape
    if tuple(array.shape) != expected_storage:
        raise ManualXpuLoadError(
            f"[{name}] manual XPU output[{index}] shape {tuple(array.shape)} does not match "
            f"declared {tuple(declared_shape if declared_shape is not None else resolved_shape)} "
            f"(storage {expected_storage})"
        )
    return array


def dump_manual_xpu_archive(testcase, arrays) -> None:
    """--dump xpu 开启时，manual 装载产物按归档回放协议落盘（与远端产物同命名）。

    实现归属 common/dump/dumper.dump_xpu_archive（``{testcase}_xpu_golden_
    {index}.bin``）——manual 回放的用例再 dump 出来仍是可回放形态，闭环完整。
    manual 不经远端，此函数是 manual 侧唯一归档点。
    """
    from ...switches import get_global_storage

    switches = get_global_storage()
    if not getattr(getattr(switches, "dump_config", None), "dump_xpu", False):
        return
    from .dump.dumper import dump_xpu_archive

    testcase_name = getattr(testcase, "testcase_name", None) or getattr(testcase, "op_name", None) or "case"
    dump_xpu_archive(
        {MANUAL_XPU_PROVIDER: {"status": "PASS", "outputs": list(arrays)}},
        testcase_name,
        switches,
    )


def dump_xpu_slot_files(testcase, switches, third_parties, provider) -> None:
    """--dump xpu 开启时，三方输出按 case 目录槽位协议落盘。

    ``xpu_{provider}_{idx}_{dtype}.{ext}``（DataDumper.dump_xpu 单份实现）——
    ``--manual-data-dirs`` 回放侧按文件名 idx 对位读取（ManualDataCase.third_party_paths），
    与 input/golden 槽同目录共存。远端/manual 两来源统一经此入口。
    命名前缀：dump_file_prefix or testcase_name（kernel/aclnn 约定；e2e/geir
    无该字段自然回退 testcase_name）。
    """
    if not third_parties:
        return
    if not getattr(getattr(switches, "dump_config", None), "dump_xpu", False):
        return
    from .dump.dumper import DataDumper

    name_prefix = getattr(testcase, "dump_file_prefix", None) or getattr(testcase, "testcase_name", None) or "case"
    DataDumper(switches, name_prefix).dump_xpu({provider: list(third_parties)})


def collect_third_party(**kwargs):
    """三方采集门面——与 ttk.remote.client.collect_third_party 同签名同返回（仅 remote 通路）。

    远端配置/端点不可用由 client 既有语义降级：(None, None, None) → 调用方
    GOLDEN_FAILURE（cross_check）或空指标（xpu_perf，其缺配置在启动期由
    normalize_xpu_options 硬错）。"""
    from ttk.remote.client import collect_third_party as remote_collect

    return remote_collect(**kwargs)
