#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
XPU 第三方输出采集客户端 —— 面向 core_modules 的门面。

把 TestSpec.third_party 翻译成 ExecutionSpec，经 EndpointView 解析可用 provider，
派发到远端 xpu_server 执行，提取返回的第三方输出数组。

本模块是 Kernel / GEIR / ACLNN / E2E 各模式共用的通用编排层，不耦合任何模式
专属上下文（OpInfoKeeper / OpApiInfoKeeper / TestcaseKernel / TestcaseAclnn 等）。
各模式在调用前自行准备：op_name、inputs（逻辑 ori-shape 数组）、input_names、
op_type、attributes、testcase_name、switches。
"""

__all__ = [
    "extract_spec_providers",
    "build_spec",
    "xpu_mode_of",
    "extract_third_party",
    "dispatch_xpu",
    "collect_third_party",
]

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from . import DATA, PERF, ExecutionSpec, get_tenant_id
from .endpoint_view import _TP_ALIASES


def extract_spec_providers(tp) -> List[str]:
    """Spec-layer provider keys (priority order = insertion order).

    dict -> keys; str -> single provider derived from API prefix; None/empty -> [].
    (Empty -> caller lets EndpointView.resolve_providers use detect∩yaml∩alive.)
    """
    if isinstance(tp, dict):
        return [_TP_ALIASES.get(k, k) for k in tp]
    if isinstance(tp, str):
        from ttk.remote import _derive_provider_from_api

        return [_derive_provider_from_api(tp, "torch")]
    return []


def build_spec(provider: str, tp, spec_file: Optional[str], spec_class: Optional[str]) -> ExecutionSpec:
    """Build one ExecutionSpec. api source: third_party dict | str | None.

    - dict[str, str]      -> type='api', api=value
    - dict[str, type]     -> type='spec' (impl class; server resolves
                             cls.third_party[provider]); spec_file/class
                             carried so the server can sync the module
    - str                 -> type='api', api=tp
    - None / no spec      -> type='api', api=None (server _resolve_3party_api
                             derives api from op_name + op_type)
    """
    if isinstance(tp, dict) and provider in tp:
        v = tp[provider]
        if isinstance(v, str):
            return ExecutionSpec(provider=provider, type="api", api=v)
        return ExecutionSpec(
            provider=provider,
            type="spec",
            spec_file=spec_file,
            spec_module=Path(spec_file).stem if spec_file else None,
            spec_class=spec_class,
        )
    if isinstance(tp, str):
        return ExecutionSpec(provider=provider, type="api", api=tp)
    return ExecutionSpec(provider=provider, type="api", api=None)


def xpu_mode_of(switches, need_data: bool) -> int:
    """按位或：xpu_perf→PERF，need_data/--dump xpu→DATA。返回 0/PERF/DATA/DATA|PERF。"""
    mode = 0
    if getattr(switches, "xpu_perf", False):
        mode |= PERF
    if need_data or switches.dump_config.dump_xpu:
        mode |= DATA
    return mode


def extract_third_party(xpu_results, priority: Optional[str]):
    """从 priority provider 取 outputs（纯函数，直接索引，不靠 dict 序）。

    fail-closed：无 results / 无 priority / 非 PASS / 无 outputs → None
    （cross_check → GOLDEN_FAILURE）。
    """
    if not xpu_results or priority is None:
        return None
    entry = xpu_results.get(priority, {})
    if entry.get("status") != "PASS" or "outputs" not in entry:
        return None
    return entry["outputs"]


def _safe_dump_token(value, default: str) -> str:
    text = str(value or default)
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    if token != text:
        token = f"{token or default}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"
    return token or default


def _dump_xpu_output(output, target_dir, file_name, file_format):
    """按格式落盘单个 XPU 输出（remote 独立部署：不依赖 core_modules 的 formatter 注册制）。"""
    if file_format == "npy":
        import numpy

        numpy.save(os.path.join(target_dir, file_name + ".npy"), output)
    else:
        with open(os.path.join(target_dir, file_name + ".bin"), "wb") as f:
            f.write(output.tobytes() if hasattr(output, "tobytes") else bytes(output))


def _dump_xpu_outputs(xpu_results, testcase_name: str, switches) -> None:
    """Persist successful XPU outputs when --dump xpu is enabled.

    文件名对齐归档回放契约：``{testcase}_xpu_golden_{index}.bin``，可直接作为
    ``manual_xpu_binaries`` 上传归档平台。多个 provider 同时产出时按 provider
    分目录避免同名覆盖，各目录内文件名保持一致。
    """
    if not switches.dump_config.dump_xpu or not xpu_results:
        return

    from ttk.utilities import deep_flatten

    dump_path = os.getenv("NPU_DUMP_PATH") or getattr(switches, "root_path", os.getcwd())
    file_format = getattr(getattr(switches, "dump_config", None), "file_format", "bin")
    case_token = _safe_dump_token(testcase_name, "testcase")

    def _has_outputs(entry):
        return isinstance(entry, dict) and entry.get("status") == "PASS" and entry.get("outputs") is not None

    multiple = sum(1 for entry in xpu_results.values() if _has_outputs(entry)) > 1

    for provider, entry in xpu_results.items():
        if not _has_outputs(entry):
            continue
        provider_token = _safe_dump_token(provider, "xpu")
        target_dir = os.path.join(dump_path, provider_token) if multiple else dump_path
        os.makedirs(target_dir, exist_ok=True)
        for index, output in enumerate(deep_flatten(entry["outputs"])):
            if output is None or isinstance(output, str):
                continue
            file_name = f"{case_token}_xpu_golden_{index}"
            _dump_xpu_output(output, target_dir, file_name, file_format)
            logging.info("[%s] Dumped XPU output: %s/%s", testcase_name or "testcase", target_dir, file_name)


# 端点短暂不可用时的恢复等待: 默认最多等 3 分钟, 环境变量可调(0 表示不等, 恢复旧行为)。
ENDPOINT_RECOVERY_WAIT_S = float(os.environ.get("TTK_ENDPOINT_RECOVERY_WAIT_S", "180"))
# 本进程内是否曾经成功解析过端点 —— 区分"链路瞬断"(值得等)与"从未可用"(不值得等)
_ENDPOINT_EVER_RESOLVED = False


def _probe_endpoints_alive(ev) -> bool:
    """绕过 health 文件直接探一次各端点, 任一活着就返回 True。

    health 文件由心跳子进程周期性刷新; 链路抖动期间它会如实写 alive=false。
    但"此刻读到不可用"不等于"这一整批都不可用" —— 直接探一次, 让恢复立刻被看见,
    而不是干等下一次心跳。
    """
    import http.client

    for ep in getattr(ev, "_endpoints", []) or []:
        try:
            conn = http.client.HTTPConnection(ep.host, ep.port, timeout=10)
            try:
                conn.request("GET", "/v1/heartbeat")
                if 200 <= conn.getresponse().status < 300:
                    return True
            finally:
                conn.close()
        except OSError:
            continue
    return False


def _resolve_with_recovery(ev, spec_providers, cli_providers, label):
    """解析 provider; 端点暂时不可用时退避等待其恢复, 而不是把用例判死。

    背景: 跨公网跳板的链路会瞬时中断。端点一旦判死, 后续每个用例
    都会快速失败并被**消耗掉** —— 等链路几分钟后恢复时, 这些用例已经记成 FAIL,
    整批只能重跑。更糟的是这种 FAIL 与"算子精度真的不达标"在 precision_status 上
    无法区分, 得翻 xpu_metrics 才看得出是基础设施问题。
    """
    import time as _time

    # 只对**曾经解析成功过**的端点等待恢复。理由: 等待是为了扛住"跑批中途链路瞬断"
    # (端点原本活着, 掉线几分钟后会回来); 而"一开始就不可用"是永久状态 —— 配置没写、
    # 服务没起、端口不通, 等多久都不会变, 每个用例空等只是把跑批拖死。
    global _ENDPOINT_EVER_RESOLVED
    if not _ENDPOINT_EVER_RESOLVED:
        try:
            providers = ev.resolve_providers(spec_providers, cli_providers)
        except RuntimeError as e:
            logging.error("[%s] XPU resolve failed: %s", label, e)
            return None
        _ENDPOINT_EVER_RESOLVED = True
        return providers

    deadline = _time.monotonic() + ENDPOINT_RECOVERY_WAIT_S
    delay = 5.0
    while True:
        try:
            providers = ev.resolve_providers(spec_providers, cli_providers)
            _ENDPOINT_EVER_RESOLVED = True
            return providers
        except RuntimeError as e:
            if _time.monotonic() >= deadline:
                logging.error("[%s] XPU resolve failed (endpoint down >%.0fs): %s", label, ENDPOINT_RECOVERY_WAIT_S, e)
                return None
            if _probe_endpoints_alive(ev):
                logging.warning("[%s] endpoint reachable again, retrying provider resolution", label)
                _time.sleep(1.0)  # 给心跳一点时间刷新 health 文件
                continue
            logging.warning("[%s] endpoint unavailable, waiting %.0fs before retry: %s", label, delay, e)
            _time.sleep(delay)
            delay = min(delay * 2, 30.0)


def dispatch_xpu(
    *,
    op_name: str,
    inputs,
    input_names: List[str],
    op_type: Optional[str],
    attributes: dict,
    input_formats: Optional[list] = None,
    input_dtypes: Optional[list] = None,
    testcase_name: str,
    switches,
    need_data: bool,
    param_order: Optional[list] = None,
):
    """Run XPU dispatch，返回 (xpu_results, priority_provider)。

    resolve_providers 失败 → xpu_results={} + priority=None
    （→ extract_third_party None → GOLDEN_FAILURE）。

    参数化：不读 OpInfoKeeper / context.op_name / get_global_storage，
    全部由调用方注入。本地执行路由在 common/xpu.py（RemoteXpuExecutor
    委托）；spec 查询经 ttk.core_modules.common.plugin。
    """
    from ttk.core_modules.common.plugin import SpecResolver, get_spec_class_meta
    from ttk.remote.endpoint_view import EndpointView, _parse_provider_filter

    paths = getattr(switches, "plugin_path", None) or ()
    # pathway 仅参与缓存键：third_party 查询在 spec 系统内通路无关
    tp = SpecResolver(op_name, "kernel", paths).resolve_third_party()
    if isinstance(tp, dict):
        tp = {_TP_ALIASES.get(k, k): v for k, v in tp.items()}
    meta = get_spec_class_meta(op_name, paths)
    spec_file = meta["spec_file"] if meta else None
    spec_class = meta["class_name"] if meta else None

    ev = EndpointView()
    spec_providers = extract_spec_providers(tp)
    cli_providers = _parse_provider_filter(getattr(switches, "provider_filter", None))

    providers = _resolve_with_recovery(ev, spec_providers, cli_providers, testcase_name or op_name)
    if providers is None:
        _record_breaker(switches, ok=False, reason="no usable provider (endpoint not resolvable)")
        return {}, None

    specs = [build_spec(p, tp, spec_file, spec_class) for p in providers]

    # E2E: op_name is a dotted API path (e.g. "torch.add"); use it as api
    # when no explicit third_party is configured, so the server resolves it
    # via resolve_callable (dotted) instead of _resolve_3party_api (snake_case).
    if tp is None and op_name and "." in op_name:
        for s in specs:
            if s.api is None:
                s.api = op_name

    from ttk.remote.xpu_collector import collect_xpu_results

    _tmp_root = os.path.join(getattr(switches, "root_path", os.getcwd()), ".ttk", "xpu_tmp")
    xpu_results = collect_xpu_results(
        specs,
        inputs=inputs,
        input_names=input_names,
        mode=xpu_mode_of(switches, need_data),
        tenant_id=get_tenant_id(),
        op_name=op_name,
        op_type=op_type,
        attrs=attributes or {},
        input_formats=input_formats,
        input_dtypes=input_dtypes,
        tmp_root=_tmp_root,
        runtime=getattr(switches, "run_time", 3),
        param_order=param_order,
        dump_xpu=switches.dump_config.dump_xpu,
    )
    _record_breaker(switches, **_judge_xpu_results(xpu_results))
    _dump_xpu_outputs(xpu_results, testcase_name, switches)
    return xpu_results, (specs[0].provider if specs else None)


def _judge_xpu_results(xpu_results: dict) -> dict:
    """本例三方采集算成功还是失败(喂给熔断器)。任一 provider PASS 即成功。"""
    if any((r or {}).get("status") == "PASS" for r in xpu_results.values()):
        return {"ok": True, "reason": ""}
    errors = [(r or {}).get("error", "") for r in xpu_results.values()]
    return {"ok": False, "reason": next((e for e in errors if e), "empty xpu_results")}


def _record_breaker(switches, *, ok: bool, reason: str) -> None:
    """记一例三方采集结果；熔断器自身出问题绝不能影响跑批。"""
    from ttk.remote import xpu_breaker

    try:
        xpu_breaker.record(getattr(switches, "root_path", None) or os.getcwd(), ok=ok, reason=reason)
    except OSError as e:
        logging.warning("xpu breaker record failed (ignored): %s", e)


def collect_third_party(
    *,
    op_name: str,
    inputs,
    input_names: List[str],
    op_type: Optional[str],
    attributes: dict,
    testcase_name: str,
    switches,
    need_data: bool = True,
    param_order: Optional[list] = None,
    input_formats: Optional[list] = None,
    input_dtypes: Optional[list] = None,
) -> Tuple[Optional[str], Optional[list], Optional[dict]]:
    """门面：采集第三方输出，返回 (priority_provider, flat_third_parties, xpu_results)。

    各模式在 cross_check / xpu_perf 场景调用：传入逻辑 ori-shape 的 inputs、
    算子参数名 input_names，本函数完成 spec 解析 → endpoint 解析 → 派发 → 提取 → 展平。

    input_formats：逐输入 format（与 input_names 位置对齐，可为空）——转发给
    dispatch_xpu → X-Input-Schema，供服务端 compose 解析通道轴等 format 依赖场景

    input_dtypes：逐输入**逻辑** dtype（与 input_names 位置对齐，可为空），即
    CSV 声明值（如 complex32）——嵌入 X-Input-Schema 的 logical_dtype 字段，
    供服务端把 complex32 的 float16+尾维[2] 存储布局还原成 torch.complex32。

    - 远端不可用 / 无 provider / 执行失败 → (None, None, None)
      （调用方传 None 给 compare → cross_check 返回 GOLDEN_FAILURE）
    - cross_check 成功 → (priority, [np.ndarray, ...], xpu_results)
    - PERF-only 成功 → (priority, None, xpu_results)
    """
    from ttk.utilities import deep_flatten

    xpu_mode = xpu_mode_of(switches, need_data)
    if not xpu_mode:
        return None, None, None

    xpu_results, priority = dispatch_xpu(
        op_name=op_name,
        inputs=inputs,
        input_names=input_names,
        op_type=op_type,
        attributes=attributes,
        testcase_name=testcase_name,
        switches=switches,
        need_data=need_data,
        param_order=param_order,
        input_formats=input_formats,
        input_dtypes=input_dtypes,
    )

    if need_data:
        nested = extract_third_party(xpu_results, priority)
        if nested is None:
            if priority is None and not xpu_results:
                logging.warning(
                    "[%s] cross_check configured but no third_party output "
                    "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                    testcase_name or op_name,
                )
            return priority, None, xpu_results
        return priority, list(deep_flatten(nested)), xpu_results

    return priority, None, xpu_results
