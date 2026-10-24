#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Dispatcher - Worker-side remote execution dispatch.

Sends numpy inputs to xpu_server, handles 424 dependency retry with automatic
/v1/sync upload, returns numpy outputs.
"""

import atexit
import base64
import contextlib
import hashlib
import http.client
import io
import json
import logging
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ttk.remote import has_data
from ttk.utilities.container import deep_flatten
from ttk.utilities.dtypes import is_4bit_dtype

CHUNK_SIZE = 64 * 1024  # 64KB chunks for file streaming


class RemoteExecutionError(Exception):
    """Remote execution failed."""


class RemoteConnectionError(Exception):
    """Failed to connect to xpu_server."""


class RemoteBusyError(RemoteExecutionError):
    """Server returned 503 (busy) and the retry budget was exhausted.

    Subclasses RemoteExecutionError so callers catching the base class still
    work, while TTK can tell 'server gave up under load' (this) apart from
    'operator genuinely failed' (plain RemoteExecutionError from a 500).
    """


@dataclass
class RemoteResult:
    """Outputs + perf from a successful /v1/run.

    dispatch_to_remote returns a plain list by default; pass
    return_result=True to get this wrapper carrying both outputs and perf.
    ``api`` carries the server-resolved API (X-API header) so the collector
    can show the real API rather than a client-side guess.
    """

    outputs: list
    perf: Optional[dict] = None
    api: Optional[str] = None


def _parse_client_mode(raw) -> int:
    """Convert mode string ('data'/'data_perf') to bitmask integer.

    Accepts 'data', 'data_perf', or None. Defaults to DATA mode.
    """
    from ttk.remote import DATA, PERF

    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        return {"data": DATA, "perf": PERF, "data_perf": DATA | PERF}.get(raw.strip().lower(), DATA)
    return DATA


def _parse_perf_header(raw):
    """Parse the X-Perf header (JSON); None if absent/invalid."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_api_header(raw):
    """Parse the X-API response header (raw string, NOT json).

    The server echoes back the resolved API as a plain dotted string
    (e.g. 'torch.add'); None if absent/empty/whitespace.
    """
    if not raw:
        return None
    return raw.strip() or None


def _backoff_delay(n: int, base: float, max_delay: float, jitter: float, rng=random.uniform) -> float:
    """Exponential backoff with jitter: min(base*2**n, max_delay) * (1 ± jitter)."""
    capped = min(base * (2**n), max_delay)
    return capped * (1 + rng(-jitter, jitter))


def _cfg(field: str, default, cast=float):
    """Generic config getter from RemoteConfig with fallback.

    Args:
        field: Config field name (e.g., 'backoff_base_s')
        default: Default value if config is None or field is None
        cast: Type cast function (default: float)

    Returns:
        Config value cast to specified type, or default
    """
    from ttk.remote.config import get_remote_config

    config = get_remote_config()
    val = getattr(config, field, default) if config else default
    v = val if val is not None else default
    if field == "backoff_jitter":
        return cast(max(0.0, min(float(v), 1.0)))
    return cast(v)


# Response bodies <= this stay in memory; larger stream to a per-request file.
RESP_MEM_THRESHOLD = 64 << 20


# float8 家族按原始字节传输：dtype 元数据由 X-Input-Schema 声明。
# 与 executor._UINT8_WIRE_DTYPES 同口径。
_UINT8_WIRE_DTYPES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0",
    "hifloat8",
    "int4",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat4",
)


def _wire_leaf(arr):
    """按位传输 numpy 无原生存储的 dtype：float8 家族→uint8、4bit 家族→uint8、
    bfloat16→int16；其余直通。

    dtype 元数据由 X-Input-Schema 声明，服务端据此重铸；0-D 不走
    ascontiguousarray（会把标量升成 (1,)）。
    """
    name = getattr(getattr(arr, "dtype", None), "name", "")
    if name in _UINT8_WIRE_DTYPES or is_4bit_dtype(name):
        wire = np.ascontiguousarray(arr) if arr.ndim else arr
        return wire.view(np.uint8)
    if name == "bfloat16":
        wire = np.ascontiguousarray(arr) if arr.ndim else arr
        return wire.view(np.int16)
    return arr


def _serialize_to_file(inputs: list, dir=None) -> str:
    """Serialize numpy inputs to a temporary .npz file.

    deep_flatten 展开嵌套（每叶子一个 a{i}），过滤 None。float8 家族只传
    uint8 原始位、bfloat16 只传 int16 原始位（npz 无法以原生 dtype 落盘回读），
    元数据由 schema 声明。
    dir: 临时文件目录（默认系统 tempdir）。返回路径，caller 清理。
    """
    leaves = [a for a in deep_flatten(inputs) if a is not None]
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=dir) as tmp:
        np.savez(tmp, **{f"a{i}": _wire_leaf(a) for i, a in enumerate(leaves)})
    return tmp.name


def _reinterpret_dtype(arr, dtype_name):
    from ..utilities.dtypes import resolve_numpy_dtype

    try:
        if dtype_name == "bfloat16" or dtype_name in _UINT8_WIRE_DTYPES or is_4bit_dtype(dtype_name):
            return arr.view(resolve_numpy_dtype(dtype_name))
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
        return arr
    return arr


def _resp_to_npz_source(resp, req_dir, threshold=RESP_MEM_THRESHOLD):
    """Return an np.load()-able source for the response body.
    Large Content-Length -> req_dir/resp.npz; small/missing -> BytesIO."""
    try:
        cl = int(resp.getheader("Content-Length") or 0)
    except (TypeError, ValueError):
        cl = 0
    if cl > threshold:
        path = os.path.join(req_dir, "resp.npz")
        with open(path, "wb") as f:
            shutil.copyfileobj(resp, f, CHUNK_SIZE)
        return path
    return io.BytesIO(resp.read())


def _load_npz_outputs(source, schema):
    """按 schema 恢复嵌套顶层 slots（不 flatten）。
    schema: [{index|indices|null, dtype}, ...]"""
    npz = np.load(source, allow_pickle=False)
    out = []
    for entry in schema:
        idx = entry.get("index")
        if idx is None:
            indices = entry.get("indices")
            if indices is None:
                out.append(None)
            else:
                # tensor-list slot：恢复 list of arrays
                dtype = entry.get("dtype")
                out.append([_reinterpret_dtype(npz[f"a{i}"], dtype) for i in indices])
        else:
            arr = npz[f"a{idx}"]
            out.append(_reinterpret_dtype(arr, entry.get("dtype")))
    return out


def _dtype_name(arr):
    """Best-effort numpy dtype name for the wire (e.g. 'float32', 'bfloat16').

    Carried in X-Input-Schema so the server can convert precisely even when its
    own numpy can't represent the dtype (bfloat16 arrives as raw bytes / void).
    """
    if arr is None:
        return None
    try:
        return arr.dtype.name
    except AttributeError:
        return None


def _slot_logical_dtype(entry):
    """Schema 每 slot 单值；TensorList 声明的嵌套 dtype 取首叶子。

    与物理 dtype 的 leaves[0] 取法一致（同质假设：list 内 dtype 一致）。
    """
    if isinstance(entry, (list, tuple)):
        return entry[0] if entry else None
    return entry


def _build_input_schema(
    inputs: list, input_names: list, input_formats: Optional[list] = None, input_dtypes: Optional[list] = None
) -> list:
    """Build X-Input-Schema from inputs and their names.

    嵌套为真相源：顶层 zip(input_names, inputs) 位置对齐，按 slot 类型分派。
    None slot → index:null；ndarray slot → 单 index；list/tuple slot → indices
    （deep_flatten 取叶子，过滤 None）。deep_flatten 不作用于 ndarray slot
    （其作容器元素时原子保留；裸 ndarray 当 sequence 入参会按行切，故 ndarray
    分支直接用 slot）。

    ``input_formats`` 为逐输入 format（与 input_names 位置对齐，可为空）——
    嵌入每个 schema 条目（name/index/dtype/format），server 侧可据此把
    format 传给三方 compose（广播/归约轴判定用），不另设独立 header。

    ``input_dtypes`` 为逐输入**逻辑** dtype（与 input_names 位置对齐，可为空），
    即 CSV 声明值（如 complex32），区别于条目 ``dtype`` 字段的物理 numpy dtype
    ——complex32 在 npz 里物理上是 float16+尾维[2]，服务端靠 logical_dtype
    才能还原成 torch.complex32 逻辑张量。

    注意：tensor-list slot 的 dtype 取自首个叶子（leaves[0]），假设同一 list
    内 dtype 同质；混合 dtype 的 list slot 未完全支持——server 每个 name 只应用
    一种 dtype。
    """
    if not input_names:
        return []

    assert len(inputs) <= len(input_names), (
        f"inputs({len(inputs)}) > names({len(input_names)}): slots/names length mismatch (caller bug)"
    )

    fmts = list(input_formats) if input_formats else []
    ldts = list(input_dtypes) if input_dtypes else []
    fi = 0
    schema: list = []
    for i, (name, slot) in enumerate(zip(input_names, inputs)):
        fmt = fmts[i] if i < len(fmts) else None
        ldt = _slot_logical_dtype(ldts[i] if i < len(ldts) else None)
        if slot is None:
            schema.append({"name": name, "index": None, "dtype": None, "format": fmt, "logical_dtype": ldt})
        elif isinstance(slot, (list, tuple)):
            leaves = [x for x in deep_flatten(slot) if x is not None]
            schema.append(
                {
                    "name": name,
                    "indices": [fi + i for i in range(len(leaves))],
                    "dtype": _dtype_name(leaves[0]) if leaves else None,
                    "format": fmt,
                    "logical_dtype": ldt,
                }
            )
            fi += len(leaves)
        else:  # ndarray（含 0-d）/ numpy 标量 —— 单叶子,直接用 slot
            schema.append({"name": name, "index": fi, "dtype": _dtype_name(slot), "format": fmt, "logical_dtype": ldt})
            fi += 1
    # names 多于 inputs:尾部补 index:null
    for name in input_names[len(inputs) :]:
        schema.append({"name": name, "index": None, "dtype": None, "format": None, "logical_dtype": None})
    return schema


def _schema_leaf_count(schema: list) -> int:
    """schema 引用的非 None 叶子数 = npz a{i} 数 = X-Input-Count。

    `dispatch_to_remote` 内 `effective_count` 用此派生 header，使 X-Input-Count
    按构造等于 schema 叶子数（非第三个独立 deep_flatten 表达式）。
    """
    return sum(len(e["indices"]) if "indices" in e else (1 if e.get("index") is not None else 0) for e in schema)


def _find_spec_file(module_name: str, search_roots: list) -> Optional[str]:
    """Search for a .py file matching module_name in the spec directory trees.

    Converts dotted module names to paths, tries both as .py file and __init__.py.
    """
    rel_path = module_name.replace(".", os.sep)
    candidates = [f"{rel_path}.py", os.path.join(rel_path, "__init__.py")]

    for root in search_roots:
        for cand in candidates:
            full = os.path.join(root, cand)
            if os.path.isfile(full):
                return full

    # Try a shallow name-only match (e.g. "util" → any "util.py" under root)
    base = module_name.rsplit(".", 1)[-1]
    for root in search_roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if f"{base}.py" in filenames:
                return os.path.join(dirpath, f"{base}.py")

    return None


def _read_file_with_hash(file_path: str) -> tuple:
    """Read file, return (base64_content, sha256_hex).

    Computes the hash over the raw bytes at no extra read cost.
    """
    with open(file_path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def _do_http_sync(
    missing: str, search_roots: list, endpoint_host: str, endpoint_port: int, tenant_id: str, timeout: int
) -> bool:
    """Find and sync a missing dependency file to xpu_server.

    Returns True if the file was found and synced successfully.
    """
    file_path = _find_spec_file(missing, search_roots)
    if not file_path:
        logging.warning(f"424: could not find file for missing module '{missing}'")
        return False

    rel_path = os.path.basename(file_path)
    content_b64, file_hash = _read_file_with_hash(file_path)

    sync_body = json.dumps({"files": {rel_path: {"content": content_b64, "hash": file_hash}}})

    try:
        conn = _create_connection(endpoint_host, endpoint_port, timeout=timeout)
        conn.request(
            "POST", "/v1/sync", body=sync_body, headers={"Content-Type": "application/json", "X-Tenant-ID": tenant_id}
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status == 200:
            logging.info(f"424: synced '{rel_path}' for missing module '{missing}'")
            return True
        logging.warning(f"424: sync failed for '{rel_path}' (status {resp.status})")
        return False
    except Exception as e:
        logging.warning(f"424: sync error for '{missing}': {e}")
        return False


def _sync_missing_dependency(
    missing: str, search_roots: list, endpoint_host: str, endpoint_port: int, tenant_id: str, timeout: int
) -> bool:
    """Sync a missing dependency, serialized across workers by semaphore.

    The first worker to acquire the per-(endpoint, module) lock performs the
    upload; concurrent workers poll ``get_semaphore`` and reuse its result,
    avoiding duplicate uploads and concurrent-write races on xpu_server.

    信号量 RPC 占用 worker 命令管道，仅限主线程调用；XPU 重叠的后台线程
    直接上传（服务端 /v1/sync 原子写，并发重复无害）。
    """
    import threading

    from ttk.core_modules.common.process import process_context

    ctx = process_context()
    if ctx is None:
        # Single-process mode (no pool): sync directly — no concurrent dedup needed.
        return _do_http_sync(missing, search_roots, endpoint_host, endpoint_port, tenant_id, timeout)
    if threading.current_thread() is not threading.main_thread():
        return _do_http_sync(missing, search_roots, endpoint_host, endpoint_port, tenant_id, timeout)
    sync_id = f"xpu_sync_{endpoint_host}:{endpoint_port}:{missing}"

    if ctx.acquire_semaphore(sync_id):
        try:
            ok = _do_http_sync(missing, search_roots, endpoint_host, endpoint_port, tenant_id, timeout)
            ctx.set_semaphore(sync_id, "ok" if ok else "fail")
            return ok
        except Exception as e:
            ctx.set_semaphore(sync_id, f"err:{e}")
            raise

    # Another worker holds the lock — wait for its result. Normally the wait
    # ends quickly: holder set_semaphore on success/failure, or pool.py's
    # dead-worker handler fills the semaphore via _semaphore_dead_sequence if
    # the holder process crashed. The timeout below is a last-resort safety
    # net for edge cases where neither path fires (e.g. holder stuck in
    # zombie state, pool detection delayed).
    logging.debug(f"424: waiting for another worker to sync '{missing}' ({sync_id})")
    wait_deadline = time.monotonic() + timeout
    while (result := ctx.get_semaphore(sync_id)) is None:
        if time.monotonic() >= wait_deadline:
            raise RemoteExecutionError(
                f"timed out waiting for sync of '{missing}' (holder may have crashed); sync_id={sync_id}"
            )
        time.sleep(0.5)
    return result == "ok"


def _create_connection(host, port, timeout):
    """HTTP or HTTPS connection based on RemoteConfig TLS fields (shared tls module)."""
    from ttk.remote.config import get_remote_config
    from ttk.remote.tls import build_tls_connection, tls_from_config

    return build_tls_connection(host, port, timeout, tls_from_config(get_remote_config()))


# ---------------------------------------------------------------------------
# 连接复用(keep-alive)
#
# 每请求新建一条 TCP, 在 SSH 隧道上就是新开一条 channel。实测经公网跳板时
# 新建成本随使用逐步劣化(同隧道连续探测 18ms→57ms→290ms→752ms), 而复用同一
# 连接恒为即时。故按 (host, port) 在**进程内**缓存连接。
#
# 三条上限防止"一条连接挂一整晚": 服务端是 ThreadingHTTPServer, 一条持久连接
# 占住一个线程不放; 长时间空闲的连接还会被 NAT/跳板静默丢弃, 老 channel 本身
# 也会劣化。任一上限触发就主动换新。
#
# 关键前提: **必须假设连接随时会被对端关掉**。所有上限只是"在可控时机主动关",
# 真正兜底的是复用失败时立即换新连接重试一次(见 dispatch_to_remote 的 reused 分支)。
_CONN_POOL: dict = {}
KEEPALIVE_IDLE_S = 60.0  # 空闲这么久就不再复用(批次之间的长间隔不占资源)
KEEPALIVE_MAX_USES = 100  # 定期换新 channel, 避免老化劣化
KEEPALIVE_MAX_AGE_S = 300.0  # 兜底: 任何路径下都不会无限持有


def _acquire_connection(host, port, timeout):
    """取一条可用连接。返回 (conn, reused)；reused=True 表示来自缓存。"""
    key = (host, port)
    entry = _CONN_POOL.pop(key, None)
    if entry is not None:
        now = time.monotonic()
        fresh = (
            now - entry["last"] < KEEPALIVE_IDLE_S
            and now - entry["born"] < KEEPALIVE_MAX_AGE_S
            and entry["uses"] < KEEPALIVE_MAX_USES
        )
        if fresh:
            entry["uses"] += 1
            # 复用连接的 socket timeout 对齐本次请求，入池时的值不泄漏到后续请求
            if getattr(entry["conn"], "sock", None) is not None:
                with contextlib.suppress(OSError):
                    entry["conn"].sock.settimeout(timeout)
            _CONN_POOL[key] = entry
            return entry["conn"], True
        with contextlib.suppress(OSError):
            entry["conn"].close()
    conn = _create_connection(host, port, timeout=timeout)
    _CONN_POOL[key] = {"conn": conn, "uses": 1, "born": time.monotonic(), "last": time.monotonic()}
    return conn, False


def _response_allows_reuse(resp) -> bool:
    """只在服务端明确允许时才复用: HTTP/1.1 且未声明 Connection: close。

    这样遇到未升级的服务端(BaseHTTPRequestHandler 默认 HTTP/1.0)会自动退化成
    每次新建, 不需要开关, 也不需要版本协商。
    """
    if getattr(resp, "version", 10) < 11:
        return False
    return "close" not in (resp.getheader("Connection") or "").lower()


def _settle_connection(host, port, conn, reusable: bool):
    """请求收尾: 可复用则留在池里(并刷新 last), 否则关掉并从池中移除。"""
    key = (host, port)
    if reusable:
        entry = _CONN_POOL.get(key)
        if entry is not None and entry["conn"] is conn:
            entry["last"] = time.monotonic()
            return
    _CONN_POOL.pop(key, None)
    with contextlib.suppress(OSError):
        conn.close()


def close_pooled_connections():
    """进程退出/worker 收尾时关掉缓存连接。SIGKILL 覆盖不到 —— 那只能靠服务端
    自己的空闲超时回收(见 xpu_server 的 handler timeout)。"""
    for key in list(_CONN_POOL):
        entry = _CONN_POOL.pop(key, None)
        if entry:
            with contextlib.suppress(OSError):
                entry["conn"].close()


def _forget_pool_after_fork():
    """fork 后子进程必须丢弃继承来的连接引用(不关闭: 那是父进程的 socket)。

    TTK 的 worker 走 fork/forkserver。若连接在 fork 前入池, 父子会持有**同一个**
    socket fd, 两边往同一条连接上收发 → 流内容交错, 症状是诡异的间歇性失败。
    并行度本来就来自进程数(--pc N)而非连接数, 每进程各自建连才是正确语义。
    """
    _CONN_POOL.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pool_after_fork)

atexit.register(close_pooled_connections)


@dataclass
class _RequestMeta:
    """请求描述——/v1/run 请求头装配的全部输入（api/spec 双形态 + 算子标识）。"""

    op_name: Optional[str]
    op_type: Optional[str]
    provider: str
    attrs: dict
    tenant_id: str
    execution_type: str
    api: Optional[str]
    spec_module: Optional[str]
    spec_class: Optional[str]
    param_order: Optional[list]
    mode_int: int
    runtime: int


def _build_headers(meta: _RequestMeta, schema: list, leaf_count: int) -> dict:
    """装配 /v1/run 请求头：模式位、输入 schema、api/spec 双形态、算子标识。"""
    headers = {
        "X-Execution-Type": meta.execution_type,
        "X-Provider": meta.provider,
        "X-Attrs": json.dumps(meta.attrs),
        "X-Input-Schema": json.dumps(schema),
        "X-Input-Count": str(leaf_count),
        "X-Mode": str(meta.mode_int),
        "X-Runtime": str(meta.runtime),
        "X-Tenant-ID": meta.tenant_id,
        "Content-Type": "application/octet-stream",
    }
    if meta.param_order:
        headers["X-Param-Order"] = json.dumps(meta.param_order)
    if meta.execution_type == "api":
        if meta.api:
            headers["X-API"] = meta.api
    else:  # spec
        if meta.spec_module:
            headers["X-Spec-Module"] = meta.spec_module
        if meta.spec_class:
            headers["X-Spec-Class"] = meta.spec_class
    if meta.op_name:
        headers["X-Op-Name"] = meta.op_name
    if meta.op_type:
        headers["X-Op-Type"] = meta.op_type
    return headers


@dataclass
class _DispatchPlan:
    """一次远端派发的流程状态：请求头 + 端点 + 重试预算（各阶段共享，计数器随重试递增）。"""

    headers: dict
    mode_int: int
    endpoint_host: str
    endpoint_port: int
    timeout: int
    tenant_id: str
    spec_search_roots: list
    return_result: bool
    max_retries: int
    deadline_s: int
    deadline_at: float
    budget_503: int
    budget_conn: int
    budget_424: int
    used_503: int = 0
    used_conn: int = 0

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline_at

    def busy_retry(self) -> bool:
        """503 预算内退避后重试（424 重试不占此预算）。"""
        if self.used_503 >= self.budget_503:
            return False
        self.used_503 += 1
        time.sleep(self._backoff(self.used_503))
        return True

    def transport_retry(self, exc: Exception) -> bool:
        """传输异常预算内退避后重试（端点死亡由 HB 11s 驱逐，不在此标记）。"""
        if self.used_conn >= self.budget_conn:
            return False
        self.used_conn += 1
        logging.warning(f"conn error, retry ({self.used_conn}/{self.budget_conn}): {exc}")
        time.sleep(self._backoff(self.used_conn))
        return True

    def _backoff(self, used: int) -> float:
        return _backoff_delay(
            used - 1, _cfg("backoff_base_s", 0.5), _cfg("backoff_max_s", 10.0), _cfg("backoff_jitter", 0.25)
        )


def _send_request(plan: _DispatchPlan, tmp_path: str):
    """一次 HTTP 尝试：建连 → 64KB 流式上传 → (conn, resp)；conn 由调用方关闭。"""
    conn, reused = _acquire_connection(plan.endpoint_host, plan.endpoint_port, plan.timeout)
    if reused:
        # 复用连接可能已被对端静默关闭——立即换新连接重试一次（keep-alive 兜底语义）
        try:
            return _send_on(conn, plan, tmp_path)
        except (http.client.HTTPException, OSError):
            with contextlib.suppress(Exception):
                conn.close()
            conn = _create_connection(plan.endpoint_host, plan.endpoint_port, timeout=plan.timeout)
    return _send_on(conn, plan, tmp_path)


def _send_on(conn, plan, tmp_path):
    plan.headers["Content-Length"] = str(os.path.getsize(tmp_path))
    conn.putrequest("POST", "/v1/run")
    for key, value in plan.headers.items():
        conn.putheader(key, value)
    conn.endheaders()
    with open(tmp_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            conn.send(chunk)
    return conn, conn.getresponse()


def _finish(plan: _DispatchPlan, outputs, perf, api=None):
    """按 return_result 决定返回形态：裸 outputs 或 RemoteResult 包装。"""
    return RemoteResult(outputs=outputs, perf=perf, api=api) if plan.return_result else outputs


def _handle_ok(resp, plan: _DispatchPlan, req_dir: str):
    """200：PERF-only / 零输出直接收尾；DATA 模式按 X-Output-Schema 回解 npz。"""
    perf = _parse_perf_header(resp.getheader("X-Perf"))
    api = _parse_api_header(resp.getheader("X-API"))
    if not has_data(plan.mode_int):
        resp.read()
        return _finish(plan, [], perf, api)
    output_count = int(resp.getheader("X-Output-Count", "1"))
    if output_count == 0:
        resp.read()
        return _finish(plan, [], perf, api)
    _schema_raw = resp.getheader("X-Output-Schema")
    _schema = json.loads(_schema_raw) if _schema_raw else []
    _source = _resp_to_npz_source(resp, req_dir)
    return _finish(plan, _load_npz_outputs(_source, _schema), perf, api)


def _handle_missing_dep(resp, plan: _DispatchPlan) -> None:
    """424：预算内查找并 /v1/sync 缺失的 spec 依赖，返回后由主循环重试。

    预算耗尽 / 缺失模块不在 spec 搜索路径（server 侧环境依赖）→ 抛
    RemoteExecutionError；同步失败仅告警（重试与预算耗尽兜底）。
    """
    resp_body = resp.read()
    plan.budget_424 -= 1
    if plan.budget_424 < 0:
        raise RemoteExecutionError(f"424 dependency retries ({plan.max_retries}) exceeded: {resp_body.decode()}")
    logging.warning(f"424 retry ({plan.max_retries - plan.budget_424}/{plan.max_retries}): {resp_body.decode()}")
    try:
        missing_module = json.loads(resp_body).get("missing", "")
    except json.JSONDecodeError:
        missing_module = ""
    if not (missing_module and plan.spec_search_roots):
        return
    # Pre-check: can we find this file locally? If not, this is an environment
    # dep (e.g. scipy not installed on the server), not a spec file — retrying
    # won't help.
    if _find_spec_file(missing_module, plan.spec_search_roots) is None:
        raise RemoteExecutionError(
            f"server requires module '{missing_module}' which is not "
            f"in spec search paths; likely a server-side "
            f"environment dependency (pip install) is missing"
        )
    synced = _sync_missing_dependency(
        missing_module,
        plan.spec_search_roots,
        plan.endpoint_host,
        plan.endpoint_port,
        plan.tenant_id,
        plan.timeout,
    )
    if not synced:
        logging.warning(
            f"Could not sync '{missing_module}', retrying ({plan.max_retries - plan.budget_424}/{plan.max_retries})"
        )


def _handle_busy(resp, plan: _DispatchPlan) -> None:
    """503：读空响应体；预算内退避（返回后主循环重试），预算尽抛 RemoteBusyError。"""
    resp.read()
    if not plan.busy_retry():
        raise RemoteBusyError(f"server busy (503), retry budget ({plan.budget_503}) exhausted")


def _dispatch_loop(plan: _DispatchPlan, tmp_path: str, req_dir: str):
    """重试主循环：deadline 检查 → 发送 → 按响应分类（200 回解 / 424 同步 / 503 退避 / 其余终止）。"""
    while True:
        if plan.expired():
            raise RemoteExecutionError(f"dispatch deadline ({plan.deadline_s}s) exceeded")
        conn = None
        resp = None
        try:
            conn, resp = _send_request(plan, tmp_path)
            if resp.status == 200:
                return _handle_ok(resp, plan, req_dir)
            if resp.status == 424:
                _handle_missing_dep(resp, plan)
                continue  # does NOT touch 503 budget
            if resp.status == 503:
                _handle_busy(resp, plan)
                continue  # does NOT touch 424 budget
            # 400 / 500 / other -> genuine failure, do not retry.
            resp_body = resp.read()
            raise RemoteExecutionError(f"Server returned {resp.status}: {resp_body.decode()}")
        except RemoteExecutionError:
            raise
        except (http.client.HTTPException, ConnectionRefusedError, OSError) as e:
            # _mark_endpoint_dead removed: round-robin + HB 11s eviction replace it
            if not plan.transport_retry(e):
                raise RemoteConnectionError(f"connection retries ({plan.budget_conn}) exhausted: {e}") from e
        finally:
            if conn:
                # keep-alive：响应允复用则归还连接池，否则关闭
                with contextlib.suppress(Exception):
                    reusable = _response_allows_reuse(resp) if resp is not None else False
                _settle_connection(plan.endpoint_host, plan.endpoint_port, conn, reusable)


def dispatch_to_remote(
    op_name: str,
    inputs: list,
    *,
    input_names: Optional[list] = None,
    op_type: Optional[str] = None,
    provider: str = "torch",
    attrs: Optional[dict] = None,
    input_formats: Optional[list] = None,
    input_dtypes: Optional[list] = None,
    endpoint_host: str = "127.0.0.1",
    endpoint_port: int = 9090,
    tenant_id: str = "unknown",
    timeout: int = 300,
    max_retries: int = 5,
    mode: str = "data",
    spec_search_roots: Optional[list] = None,
    api: Optional[str] = None,
    execution_type: str = "api",
    spec_module: Optional[str] = None,
    spec_class: Optional[str] = None,
    spec_file: Optional[str] = None,
    return_result: bool = False,
    tmp_root: Optional[str] = None,
    runtime: int = 3,
    param_order: Optional[list] = None,
    deadline_s: Optional[float] = None,
):
    """Send inputs to remote xpu_server, return numpy outputs.

    On 424 (missing dependency), automatically searches spec_search_roots
    for the missing .py file and uploads it via /v1/sync before retrying.
    """
    spec_search_roots = list(spec_search_roots or [])
    # In spec mode, ensure the spec source dir is searchable for 424 dep sync.
    if execution_type == "spec" and spec_file:
        _fdir = os.path.dirname(os.path.abspath(spec_file))
        if _fdir not in spec_search_roots:
            spec_search_roots.insert(0, _fdir)

    # KERNEL no-spec (api=None) relies on the server's _resolve_3party_api to
    # derive the API from op_name/op_type — do NOT infer client-side.
    schema = _build_input_schema(inputs, input_names or [], input_formats, input_dtypes)
    meta = _RequestMeta(
        op_name=op_name,
        op_type=op_type,
        provider=provider,
        attrs=attrs or {},
        tenant_id=tenant_id,
        execution_type=execution_type,
        api=api,
        spec_module=spec_module,
        spec_class=spec_class,
        param_order=param_order,
        mode_int=_parse_client_mode(mode),
        runtime=runtime,
    )
    headers = _build_headers(meta, schema, _schema_leaf_count(schema))

    if deadline_s is None:
        deadline_s = _cfg("dispatch_deadline_s", 300.0, int)
    else:
        # 重叠形态：请求横跨 NPU 执行期，socket 读超时同样须覆盖（至少 = deadline）
        timeout = max(timeout, int(deadline_s))
    plan = _DispatchPlan(
        headers=headers,
        mode_int=meta.mode_int,
        endpoint_host=endpoint_host,
        endpoint_port=endpoint_port,
        timeout=timeout,
        tenant_id=tenant_id,
        spec_search_roots=spec_search_roots,
        return_result=return_result,
        max_retries=max_retries,
        deadline_s=deadline_s,
        deadline_at=time.monotonic() + deadline_s,
        budget_503=_cfg("max_503_retries", 10.0, int),
        budget_conn=_cfg("max_conn_retries", 5.0, int),
        budget_424=max_retries,
    )

    req_dir = None
    try:
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        req_dir = tempfile.mkdtemp(prefix=f"req_{tenant_id}_", dir=tmp_root)
        return _dispatch_loop(plan, _serialize_to_file(inputs, dir=req_dir), req_dir)
    finally:
        if req_dir:
            shutil.rmtree(req_dir, ignore_errors=True)
