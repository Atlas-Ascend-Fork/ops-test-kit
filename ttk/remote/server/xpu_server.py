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
xpu_server - TTK Remote XPU Execution Server.

部署形态：整仓部署（ttk 与 server 同仓同源——executor 经
ttk.core_modules.common.xpu / common.profiler 消费单份契约与共享引擎）。
XPU 机器上克隆/拷贝整个 ttk 仓（或 pip 安装）后启动，命令见 Quick start。

Quick start:
    python -m ttk.remote.server.xpu_server --port 9090

All options:
    python -m ttk.remote.server.xpu_server \\
        --port 9090          # 监听端口（默认：配置或 9090）
        --bind 127.0.0.1     # 绑定地址（默认：127.0.0.1）
        --config xpu_server.yaml  # 配置文件路径
        --devices 0,1        # 设备 ID 列表（默认：0）
        --dry-run            # 空跑模式（返回随机数据）
"""

import argparse
import base64
import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import ssl
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from ttk.core_modules.common.xpu import XPU_DATA as DATA
from ttk.core_modules.common.xpu import XPU_PERF as PERF

from . import executor
from .config import detect_hardware, get_framework, load_server_config
from .container import _run_in_container

CHUNK_SIZE = 64 * 1024  # 64KB

# Module-level variables (set by run_server from config)
SYNC_BASE_DIR = None
TMP_ROOT = None
HEARTBEAT_TIMEOUT_S = 600
# Per-device locks: one Lock per device_id, allowing concurrent execution on
# distinct devices while serializing access to the same device. Initialized by
# run_server. Kill-safe: parent-held around child -> never orphaned even if the
# child crashes.
_device_locks: dict = {}  # {device_id: threading.Lock()}


def _init_device_locks(device_ids):
    """按 device_ids 初始化 per-device Lock 字典。CPU 跳过。

    先 clear 再填——避免残留旧 device 的锁。run_server 启动时调用。
    """
    _device_locks.clear()
    for dev in device_ids:
        if dev != "cpu":
            _device_locks[dev] = threading.Lock()


def _build_device_opts(handler, n):
    """提取 device 分支选项（device_id/docker_args/env）——纯函数，不构造完整 kwargs。

    避免在单测里 mock BaseHTTPRequestHandler：handler 只读 use_device/sandbox/profile
    三个属性。返回 device 相关三件：
      - device_id：0（容器内固定，executor 见 {torch_lib}:0）/ "cpu"
      - docker_args：sandbox=docker 时 render 后的 2-token list（{device_id}→物理 n）
      - env：sandbox=none 时 {visible_env: str(n)}（forkserver 不继承 os.environ，显式传）

    fail-fast：sandbox=docker + 非cpu 但 profile 缺 docker_args → 返回 ok=False/http_status=500
    （检查在 render 前，防 .format(None) 崩）。执行隔离单测已覆盖此分支。
    """
    if not handler.use_device:
        return {"device_id": "cpu"}
    opts = {"device_id": 0}  # 容器内固定（executor 见 {torch_lib}:0）
    if handler.sandbox == "docker":
        if not handler.profile.get("docker_args"):
            return {"ok": False, "http_status": 500, "error": "sandbox=docker but profile missing docker_args"}
        opts["docker_args"] = [a.format(device_id=n) for a in handler.profile["docker_args"]]
    else:
        env_name = handler.profile.get("visible_env") or f"{handler.profile['torch_lib'].upper()}_VISIBLE_DEVICES"
        opts["env"] = {env_name: str(n)}
    return opts


def _parse_json_header(raw):
    """解析可选 JSON 头；缺失或非法一律当作没有（调用方自然走老路）。"""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("ignoring malformed JSON header")
        return None


def _parse_mode(raw):
    """X-Mode: int bitmask (new protocol) or legacy 'data'/'perf' string."""
    raw = (raw or "").strip()
    if raw.isdigit():
        return int(raw)
    return {"data": DATA, "perf": PERF, "data_perf": DATA | PERF}.get(raw.lower(), DATA)


def _clamp_runtime(raw: str, default: int = 3, lo: int = 1, hi: int = 100) -> int:
    """Clamp the X-Runtime header to [lo, hi]; non-numeric -> default.

    Extracted from _handle_run so the guard is unit-testable. Boundary
    behavior: out-of-range clamps to nearest bound; non-int-parseable
    (ValueError/TypeError, incl. None) falls back to ``default``.
    """
    try:
        return max(lo, min(int(raw), hi))
    except (ValueError, TypeError):
        return default


def _receive_body_to_file(handler: BaseHTTPRequestHandler, dir=None) -> Optional[str]:
    """Stream request body into a temporary file.

    Returns path to the temp file, or None if body is empty.
    Uses constant memory regardless of body size.
    """
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length == 0:
        return None

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=dir) as tmp:
        remaining = content_length
        while remaining > 0:
            chunk_size = min(CHUNK_SIZE, remaining)
            chunk = handler.rfile.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
            remaining -= len(chunk)
    return tmp.name


def _resolve_class(module: object, dotted_name: str):
    """Resolve a dotted class name within a module.

    Supports nested classes: 'OuterClass.InnerImpl'.
    """
    obj = module
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


_device_unhealthy: dict = {}  # {device_id: timestamp_marked}


def _mark_device_unhealthy(device_id):
    _device_unhealthy[device_id] = time.time()


def _is_device_healthy(device_id, cooldown_s):
    ts = _device_unhealthy.get(device_id)
    return ts is None or (time.time() - ts) >= cooldown_s


SERVER_FEATURES = ["keepalive"]
_FORKSERVER_CTX = None


def _probe_child(framework):
    """预载自检子进程: 验证"从这个预载过的 forkserver fork 出来的子进程,
    仍能 import 指定框架"。

    每个请求的子进程只 import **一个**框架(它要服务的那个 provider), 所以自检
    也要一个框架起一个子进程 —— 在一个子进程里 import 全部框架是更严苛的条件,
    本机(装了 torch_npu)那样必然 core dump, 会把"不预载"也误判成不可用。

    预载 torch 之后子进程再 import tensorflow 可能撞 C 扩展冲突(本机实测),
    这种组合就该被这道自检挡下来并降级。
    """
    import importlib

    importlib.import_module(framework)


def _forkserver_ctx(provider_hint: str = ""):
    """返回**预载过框架**的 forkserver 上下文(全进程复用一个)。

    forkserver 默认只带极小的模块集, 于是每个请求 fork 出的子进程都要自己
    ``import torch`` —— 实测 2.3 秒/请求, 而请求本身的 GPU 计算往往只有几百微秒。
    预载后子进程启动成本降到 ~0.03 秒(实测降 97%)。

    预载哪些模块**不做任何硬编码假设, 按候选表逐级实测降级**:
    个别环境里某些框架无法共存(如装了 torch_npu 的机器上 torch 与 tensorflow
    的 C 扩展同进程会 core dump), 但这是环境特性不是通例 —— 与其写死规则,
    不如每次启动拿一个空子进程试出来, 谁能起来用谁。

    另: ``set_forkserver_preload`` 必须在 forkserver 启动**之前**调用, 一旦启动
    就改不了; 所以每次降级都要在全新进程上下文里重试(通过 _spawn_ctx 重建)。
    """
    global _FORKSERVER_CTX
    if _FORKSERVER_CTX is not None:
        return _FORKSERVER_CTX
    with _FORKSERVER_LOCK:
        if _FORKSERVER_CTX is None:  # 双检: 等锁期间可能已被别的线程选好
            _FORKSERVER_CTX = _select_forkserver_ctx(provider_hint)
    return _FORKSERVER_CTX


def _select_forkserver_ctx(provider_hint: str = ""):
    """真正的选档过程。必须在 _FORKSERVER_LOCK 内调用 —— 服务端是多线程 HTTP,
    两个 provider 会并发打进来; 若不串行化, 一个线程正用着 forkserver, 另一个
    在降级时把它 _stop 掉, 请求就会永久挂住(实测: 并发双 provider 的用例超时)。"""
    import importlib.util

    # 框架统一由 _framework_preload 这个 shim 加载: 它用 RTLD_DEEPBIND 导入 tensorflow,
    # 从机制上切断"弱符号跨库合并"(torch 由 GCC 编、tf 由 clang 编, 88 个同名模板实例化
    # 被合并后 tf 在 .so 静态初始化阶段即段错误)。详见该模块 docstring 的实测与对照实验。
    # 因此这里**不再依赖加载顺序**, 也不需要按 provider 重排。
    installed = [m for m in ("torch", "tensorflow") if importlib.util.find_spec(m) is not None]
    del provider_hint  # 顺序无关, 保留形参以免调用方改动
    base = ["ttk.remote.server.executor", "ttk.remote.server._framework_preload"]
    # 候选从"全部预载"逐级退到"什么都不预载"
    # 最后一档是"什么都不预载": ttk.remote.server.executor 本身会拉起 torch,
    # 所以在 torch/tf 不能共存的机器上, 连 base 档都过不了自检, 必须能退到空。
    # 候选只有两档: 预载(executor + shim) -> 什么都不预载。
    # 框架由 shim 内部按需加载, 不再用"逐个丢框架"的降级阶梯 —— 那个阶梯的前提是
    # "框架之间必然冲突", 而 DEEPBIND 已经消除了冲突; 万一某环境仍然崩, 自检会把
    # 整档判失败并退到空预载(等价于改动前的行为)。
    candidates = [base, []]
    for preload in candidates:
        # forkserver 进程是 multiprocessing 的**模块级单例**: 一旦以某组 preload 起来,
        # 后续 set_forkserver_preload 改不动它(新建 context 对象也没用)。不显式停掉,
        # 降级就是假的 —— 实测会出现"连空预载档都自检失败", 因为用的还是第一档那个进程。
        _stop_forkserver()
        ctx = multiprocessing.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(preload)
            for framework_name in installed or ["ttk.remote.server.executor"]:
                proc = ctx.Process(target=_probe_child, args=(framework_name,))
                proc.start()
                proc.join(120)
                if proc.exitcode != 0:
                    raise RuntimeError(f"warm-up child for {framework_name} exitcode={proc.exitcode}")
            logging.info("forkserver preload ready: %s", preload)
            return ctx
        except Exception as e:  # noqa: BLE001  预载失败只降级, 绝不让服务不可用
            logging.warning("forkserver preload %s failed (%s), degrading", preload, e)
    _stop_forkserver()  # 全部档位失败: 退回干净的 forkserver(等价于改动前的行为)
    return multiprocessing.get_context("forkserver")


_FORKSERVER_LOCK = threading.Lock()


def _stop_forkserver():
    """停掉已启动的 forkserver 单例, 使下一次 set_forkserver_preload 能真正生效。"""
    try:
        from multiprocessing import forkserver as _fs

        _fs._forkserver._stop()
    except Exception as e:  # noqa: BLE001  没起来过 / 内部结构变化, 都不该影响服务
        # 首次调用(forkserver 还没起)必然走到这里, 属正常路径, 故只记 debug。
        logging.debug("stop forkserver skipped: %s", e)


def _run_in_subprocess(kwargs: dict, deadline: float) -> dict:
    """Run execute_request in a FRESH forkserver child process; return envelope.

    One child per request -> fresh sys.modules (cross-tenant isolation), a crash
    kills only that child. A hard crash (segfault/OOM) means the child exits
    with nonzero code and sends nothing -> 500. Timeout -> kill -> 500.
    """
    ctx = _forkserver_ctx(kwargs.get("provider") or "")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=executor.child_main, args=(child_conn, kwargs))
    logging.info("_run_in_subprocess: starting child for provider=%s", kwargs.get("provider"))
    proc.start()
    logging.info("_run_in_subprocess: waiting pid=%s deadline=%s", proc.pid, deadline)
    proc.join(timeout=deadline)
    logging.info("_run_in_subprocess: done alive=%s exitcode=%s", proc.is_alive(), proc.exitcode)
    if proc.is_alive():
        proc.kill()
        proc.join(5)
        return executor._err(500, f"request timed out after {deadline}s", api=executor._api_from_kwargs(kwargs))
    envelope = None
    if parent_conn.poll(0):
        try:
            envelope = parent_conn.recv()
        except EOFError:
            envelope = None
    parent_conn.close()
    if envelope is None:
        return executor._err(
            500, f"child process exited with code {proc.exitcode}", api=executor._api_from_kwargs(kwargs)
        )
    return envelope


class TenantManager:
    """Thread-safe tenant lifecycle management."""

    def __init__(self, sync_base_dir: str):
        self._lock = threading.Lock()
        self._tenants: dict = {}
        self.sync_base_dir = sync_base_dir

    def heartbeat(self, tenant_id: str):
        with self._lock:
            if tenant_id not in self._tenants:
                tenant_path = os.path.join(self.sync_base_dir, tenant_id)
                with contextlib.suppress(OSError):
                    os.makedirs(tenant_path, exist_ok=True)
                self._tenants[tenant_id] = {
                    "last_heartbeat": time.time(),
                    "path": tenant_path,
                }
            else:
                self._tenants[tenant_id]["last_heartbeat"] = time.time()

    def cleanup(self, tenant_id: str) -> bool:
        with self._lock:
            info = self._tenants.pop(tenant_id, None)
            if info and os.path.isdir(info["path"]):
                self._safe_rmtree(info["path"])
                return True
            return False

    def _safe_rmtree(self, path):
        """仅清理 sync_base_dir 真实路径之内的目录（tenant_id 流入 join，
        入口校验之外的路径不得越界删除——防御纵深）。"""
        base = os.path.realpath(self.sync_base_dir)
        target = os.path.realpath(path)
        if target != base and not target.startswith(base + os.sep):
            logging.error("refuse to rmtree outside sync base: %s (base %s)", path, base)
            return
        shutil.rmtree(target, ignore_errors=True)

    def cleanup_expired(self):
        now = time.time()
        with self._lock:
            expired = [
                (tid, info["path"])
                for tid, info in self._tenants.items()
                if now - info["last_heartbeat"] > HEARTBEAT_TIMEOUT_S
            ]
            for tid, _ in expired:
                del self._tenants[tid]
        # Clean up files outside the lock to avoid holding it during I/O
        for tid, path in expired:
            logging.info(f"Tenant {tid} expired, cleaning up")
            if os.path.isdir(path):
                self._safe_rmtree(path)


def _atomic_write_file(abs_path: str, content: bytes) -> None:
    """Write content to abs_path atomically via temp file + rename.

    Concurrent writes to the same path never produce a torn file: a reader
    sees either the previous version or the complete new version, never a
    half-written one. Missing parent directories are created.
    """
    dir_path = os.path.dirname(abs_path)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".sync_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, abs_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


class XpuRequestHandler(BaseHTTPRequestHandler):
    # 持久连接: 客户端可复用同一条 TCP —— 在 SSH 隧道上就是复用同一条 channel。
    # 实测经公网跳板时每请求新建 channel 的成本随使用逐步劣化
    # (同隧道连续探测 18ms→57ms→290ms→752ms), 复用则恒为即时。
    # 前提是每条响应都带准确 Content-Length(本文件两处出口 + send_error 均满足)。
    protocol_version = "HTTP/1.1"
    # 空闲读超时: 超时后 handle_one_request 置 close_connection, 线程退出、连接关闭。
    # 这是**不依赖客户端清理**的那道保险 —— 客户端被 SIGKILL 时内核虽关了 fd,
    # 但隧道侧 channel 可能滞留在 CLOSE-WAIT(实测), 服务端不能干等着占线程。
    timeout = 30
    tenant_manager: TenantManager
    dry_run: bool = False
    device_count: int = 1
    device_ids = ["cpu"]  # device=[ids]，CPU=["cpu"]；run_server 设置
    _device_rr_counter: int = 0
    _device_rr_lock = threading.Lock()
    use_device: bool = False
    data_gate = None  # threading.BoundedSemaphore (backpressure)
    provider: str = ""  # set by run_server() from config/detect
    hardware: str = ""  # device_str format key; "" = auto-detect
    profile: dict = {}  # hardware_config[role] segment; {} = cpu 兜底
    hardware_config: dict = {}  # full hardware section from cfg
    provider_framework: dict = {}  # provider→framework overrides
    frameworks: dict = {}
    sync_base_dir: str = ""  # set by run_server() from config
    tmp_root: str = ""  # set by run_server() from config
    gate_wait_s: float = 1.0  # set by run_server() from config
    run_deadline_s: int = 300  # set by run_server() from config

    def log_message(self, format, *args):
        logging.info(f"{self.client_address[0]} - {format % args}")

    def _send_json(self, status: int, data: dict, env=None):
        body = json.dumps(data).encode()
        self.send_response(status)
        if env and env.get("api"):
            self.send_header("X-API", env["api"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_header(self, name: str, default: str = "") -> str:
        return self.headers.get(name, default)

    @staticmethod
    def _valid_tenant_id(tid):
        """Reject path-traversal / injection in tenant_id (flows into os.path.join).

        `.`/`..` 匹配字符集但经 join 会逃出 sync base，入口即拒。"""
        import re

        return bool(tid) and tid not in (".", "..") and bool(re.fullmatch(r"[A-Za-z0-9._-]{1,64}", tid))

    def do_GET(self):
        if self.path.startswith("/v1/heartbeat"):
            # Merge of old /health + /v1/detect + /heartbeat:
            #   - tenant register side-effect (old /heartbeat)
            #   - hardware/device_count (old /health)
            #   - providers, incl. docker sandbox branch (old /v1/detect)
            qs = parse_qs(urlparse(self.path).query)
            tenant_id = qs.get("tenant_id", [""])[0]
            if tenant_id:
                if not self._valid_tenant_id(tenant_id):
                    self._send_json(400, {"error": "invalid tenant_id"})
                    return
                self.tenant_manager.heartbeat(tenant_id)
            # Capability set ONLY — order has NO priority meaning (priority is
            # app-side spec order). providers 为无序集合。
            providers = list(self.frameworks.keys()) if self.frameworks else []
            if getattr(self, "sandbox", "none") == "docker":
                docker_images = getattr(self, "docker_images", {}) or {}
                providers = list(docker_images.keys())
            self._send_json(
                200,
                {
                    "status": "ok",
                    "device_count": self.device_count,
                    "hardware": self.hardware,
                    "providers": providers,
                    # 能力声明（keepalive）：客户端连接复用协商依据。
                    "features": SERVER_FEATURES,
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.startswith("/v1/tenant/"):
            tenant_id = self.path.split("/v1/tenant/", 1)[1].split("/")[0]
            if not self._valid_tenant_id(tenant_id):
                self._send_json(400, {"error": "invalid tenant_id"})
                return
            cleaned = self.tenant_manager.cleanup(tenant_id)
            self._send_json(
                200,
                {
                    "cleaned": cleaned,
                    "path": os.path.join(self.sync_base_dir, tenant_id),
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/sync":
            self._handle_sync()
        elif self.path == "/v1/run":
            self._handle_run()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_sync(self):
        tenant_id = self._get_header("X-Tenant-ID", "")
        if not self._valid_tenant_id(tenant_id):
            self._send_json(400, {"error": "invalid or missing tenant_id"})
            return

        self.tenant_manager.heartbeat(tenant_id)
        tenant_path = os.path.join(self.sync_base_dir, tenant_id)

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024 * 1024:
            self._send_json(413, {"error": "sync body too large (max 10GB)"})
            return
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        files = data.get("files", {})
        synced = 0
        skipped = 0
        errors = []

        for rel_path, file_info in files.items():
            # Security: path traversal check
            if ".." in rel_path or os.path.isabs(rel_path):
                errors.append({"path": rel_path, "error": "path traversal rejected"})
                continue
            # Security: .py only
            if not rel_path.endswith(".py"):
                errors.append({"path": rel_path, "error": "only .py files allowed"})
                continue

            abs_path = os.path.join(tenant_path, rel_path)
            content_b64 = file_info.get("content", "")
            expected_hash = file_info.get("hash", "")

            try:
                content_bytes = base64.b64decode(content_b64)
            except Exception:
                errors.append({"path": rel_path, "error": "invalid base64"})
                continue

            # Hash check — skip if unchanged
            if expected_hash and os.path.isfile(abs_path):
                with open(abs_path, "rb") as f:
                    existing = f.read()
                existing_hash = hashlib.sha256(existing).hexdigest()
                if existing_hash == expected_hash.replace("sha256:", ""):
                    skipped += 1
                    continue

            try:
                _atomic_write_file(abs_path, content_bytes)
                synced += 1
            except OSError as e:
                errors.append({"path": rel_path, "error": str(e)})

        if errors:
            self._send_json(400, {"synced": synced, "skipped": skipped, "errors": errors})
        else:
            self._send_json(200, {"synced": synced, "skipped": skipped, "errors": errors})

    def _device_rr_next(self) -> int:
        """Thread-safe RR counter increment. Returns previous value."""
        with self._device_rr_lock:
            self._device_rr_counter += 1
            return self._device_rr_counter - 1

    def _assign_device(self):
        """RR + try-lock: 从 RR 起点出发找空闲且健康的 device，全占则阻塞起点。

        所有请求（PERF + DATA）都 acquire Lock。返回 device_id（int 或 "cpu"）。
        调用方负责 finally release。

        不健康设备（子进程报设备初始化错误时标记）在冷却期内跳过，
        避免反复分配到不可用设备。冷却期过后自动恢复重试。
        """
        device_ids = [d for d in self.device_ids if d != "cpu"]
        if not device_ids:
            return "cpu"

        cooldown = self.device_unhealthy_detect_interval_s
        start = self._device_rr_next() % len(device_ids)
        for offset in range(len(device_ids)):
            dev = device_ids[(start + offset) % len(device_ids)]
            if not _is_device_healthy(dev, cooldown):
                continue
            if _device_locks[dev].acquire(blocking=False):
                return dev
        # 健康设备全占 → 阻塞等第一个健康设备
        for offset in range(len(device_ids)):
            dev = device_ids[(start + offset) % len(device_ids)]
            if _is_device_healthy(dev, cooldown):
                _device_locks[dev].acquire()
                return dev
        # 全部不健康 → 阻塞起点（最后手段，大概率仍失败但不死锁）
        _device_locks[device_ids[start]].acquire()
        return device_ids[start]

    def _handle_run(self):
        req_api = self._get_header("X-API", "") or self._get_header("X-Spec-Class", "") or None
        parsed = self._parse_run_headers(req_api)
        if parsed is None:
            return
        gate = self.data_gate
        gate_held = False
        if gate is not None:
            if not gate.acquire(timeout=self.gate_wait_s):
                self._send_json(503, {"error": "server busy, retry"}, env={"api": req_api})
                return
            gate_held = True
        try:
            self._run_under_gate(parsed["tenant_id"], req_api, parsed)
        finally:
            if gate_held:
                with contextlib.suppress(ValueError):
                    gate.release()

    def _run_under_gate(self, tenant_id, req_api, parsed):
        """data_gate 持有段：mkdtemp → dispatch → 回包 → 清理。

        请求目录创建失败（磁盘满/权限）时已建目录的清理照常执行。"""
        req_dir = None
        try:
            req_dir = tempfile.mkdtemp(prefix=f"req_{tenant_id}_", dir=self.tmp_root)
            result, provider, n = self._dispatch_run(req_dir, req_api, parsed)
            self._send_run_result(result, req_api, provider, n)
        finally:
            if req_dir is not None:
                shutil.rmtree(req_dir, ignore_errors=True)

    def _parse_run_headers(self, req_api):
        """Parse and validate all /v1/run headers. Returns dict or None (error sent)."""
        tenant_id = self._get_header("X-Tenant-ID", "")
        if not self._valid_tenant_id(tenant_id):
            self._send_json(400, {"error": "invalid or missing tenant_id"}, env={"api": req_api})
            return None
        mode = _parse_mode(self._get_header("X-Mode", "data"))
        exec_type = self._get_header("X-Execution-Type", "api")
        runtime = _clamp_runtime(self._get_header("X-Runtime", "3"))
        schema_raw = self._get_header("X-Input-Schema", "[]")
        try:
            input_schema = json.loads(schema_raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid X-Input-Schema JSON"}, env={"api": req_api})
            return None
        try:
            input_count = int(self._get_header("X-Input-Count", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid X-Input-Count (not integer)"}, env={"api": req_api})
            return None
        attrs_raw = self._get_header("X-Attrs", "{}")
        try:
            attrs = json.loads(attrs_raw)
        except json.JSONDecodeError:
            attrs = {}
        param_order_raw = self._get_header("X-Param-Order", "")
        param_order = None
        if param_order_raw:
            try:
                param_order = json.loads(param_order_raw)
            except json.JSONDecodeError:
                param_order = None
        return {
            "tenant_id": tenant_id,
            "mode": mode,
            "exec_type": exec_type,
            "runtime": runtime,
            "input_schema": input_schema,
            "input_count": input_count,
            "attrs": attrs,
            "param_order": param_order,
        }

    def _dispatch_run(self, req_dir, req_api, parsed):
        """Build kwargs and dispatch to subprocess/container. Returns (result, provider, n)."""
        tmp_in = _receive_body_to_file(self, dir=req_dir)
        logging.info("_handle_run: dry_run=%s body_size=%s", self.dry_run, os.path.getsize(tmp_in) if tmp_in else 0)
        n = None
        provider = None
        result = {"ok": False, "http_status": 500, "error": "internal error", "api": req_api}
        try:
            if self.dry_run:
                result = _dry_run_env(req_dir)
            else:
                provider = get_framework(self._get_header("X-Provider", "torch"), self.provider_framework)
                max_attempts = len([d for d in self.device_ids if d != "cpu"]) if self.use_device else 1
                for attempt in range(max_attempts):
                    n = "cpu" if not self.use_device else self._assign_device()
                    opts = _build_device_opts(self, n)
                    if opts.get("ok") is False:
                        result = {**opts, "api": req_api}
                        n = None
                        break
                    kwargs = self._build_run_kwargs(req_dir, req_api, provider, parsed, tmp_in, opts)
                    result = self._exec_run(kwargs, opts, req_api, provider)
                    if result.get("ok") or result.get("http_status", 500) < 500:
                        break
                    _mark_device_unhealthy(n)
                    logging.warning(
                        "device %s marked unhealthy (attempt %d/%d): %s",
                        n,
                        attempt + 1,
                        max_attempts,
                        result.get("error", ""),
                    )
                    if n != "cpu":
                        _device_locks[n].release()
                    n = None
                    if attempt + 1 >= max_attempts:
                        break
        finally:
            if n is not None and n != "cpu":
                _device_locks[n].release()
        return result, provider, n

    def _build_run_kwargs(self, req_dir, req_api, provider, parsed, tmp_in, opts):
        kwargs = {
            "tenant_sync_dir": os.path.join(self.sync_base_dir, parsed["tenant_id"]),
            "exec_type": parsed["exec_type"],
            "provider": provider,
            "profile": self.profile,
            "op_name": self._get_header("X-Op-Name", "") or None,
            "op_type": self._get_header("X-Op-Type", "") or None,
            "api": req_api,
            "spec_module": self._get_header("X-Spec-Module", "") or None,
            "spec_class": self._get_header("X-Spec-Class", "") or None,
            "mode": parsed["mode"],
            "input_schema": parsed["input_schema"],
            "attrs": parsed["attrs"],
            "tmp_in_path": tmp_in,
            "input_count": parsed["input_count"],
            "device_id": opts["device_id"],
            "use_device": self.use_device,
            "output_dir": req_dir,
            "runtime": parsed["runtime"],
            "param_order": parsed["param_order"],
        }
        if "env" in opts:
            kwargs["env"] = opts["env"]
        return kwargs

    def _exec_run(self, kwargs, opts, req_api, provider):
        if self.sandbox == "docker":
            image = self.docker_images.get(provider)
            if image is None:
                return {
                    "ok": False,
                    "http_status": 500,
                    "error": f"no docker image for provider {provider}",
                    "api": req_api,
                }
            return _run_in_container(
                kwargs,
                deadline=self.run_deadline_s,
                image=image,
                memory=self.docker_memory,
                network=self.docker_network,
                use_device=self.use_device,
                docker_args=opts.get("docker_args") or [],
            )
        return _run_in_subprocess(kwargs, deadline=self.run_deadline_s)

    def _send_run_result(self, result, req_api, provider, n):
        if result.get("ok"):
            logging.info("_handle_run ok: api=%s provider=%s", req_api, provider)
            self._send_run_ok(result)
            return
        status = result.get("http_status", 500)
        if status >= 500:
            logging.error(
                "_handle_run fail: status=%s api=%s provider=%s device=%s err=%s",
                status,
                req_api,
                provider,
                n,
                result.get("error"),
            )
        else:
            logging.warning("_handle_run client-err: status=%s api=%s err=%s", status, req_api, result.get("error"))
        self._send_json(status, {"error": result.get("error"), "missing": result.get("missing")}, env=result)

    def _send_run_ok(self, env):
        output_path = env.get("output_path")
        has_body = bool(output_path and os.path.exists(output_path))
        file_size = os.path.getsize(output_path) if has_body else 0
        self.send_response(200)
        self.send_header("X-Output-Count", str(env.get("output_count", 0)))
        if has_body:
            self.send_header("X-Output-Shapes", json.dumps(env.get("shapes", [])))
            self.send_header("X-Output-Schema", json.dumps(env.get("schema", [])))  # 替 X-Output-Dtypes
        if env.get("perf") is not None:
            self.send_header("X-Perf", json.dumps(env["perf"]))
        if env.get("api"):
            self.send_header("X-API", env["api"])
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        if has_body:
            with open(output_path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    self.wfile.write(chunk)


def _dry_run_env(req_dir):
    """Random-output envelope for dry-run, mirroring _run_in_subprocess's return."""
    outs = [np.random.randn(4, 8).astype(np.float32)]
    path = os.path.join(req_dir, "out.npz")
    np.savez(path, **{f"a{i}": o for i, o in enumerate(outs)})
    return {
        "ok": True,
        "http_status": 200,
        "output_path": path,
        "output_count": len(outs),
        "shapes": [list(o.shape) for o in outs],
        "schema": [{"index": i, "dtype": str(o.dtype)} for i, o in enumerate(outs)],  # 替 dtypes
        "perf": None,
        "api": None,
    }


def _heartbeat_watcher(tenant_manager: TenantManager):
    while True:
        time.sleep(60)
        try:
            tenant_manager.cleanup_expired()
        except Exception as e:
            logging.error(f"Heartbeat watcher error: {e}")


def _detect_frameworks() -> dict:
    """Detect installed frameworks by module lookup — no import (fast, no GIL flood).

    Returns {provider_name: True}; keys are PROVIDER names (the convention used
    by --provider, third_party dicts, X-Provider header) — NOT import names.
    tensorflow advertises as 'tf' so the client's provider routing matches.
    Only the keys are used (advertised via /v1/heartbeat). The actual framework
    import is deferred to the executor (subprocess/container) when a request
    needs it — importing here would block startup ~5.5s on tensorflow's C-lib
    init + flood ~700 log lines and starve the HTTP handler.
    """
    import importlib.util

    providers = {}
    if importlib.util.find_spec("torch") is not None:
        providers["torch"] = True
    if importlib.util.find_spec("tensorflow") is not None:
        providers["tf"] = True  # provider name; the import is 'tensorflow'
    return providers


def _resolve_devices(devices, cfg):
    """Resolve device role/ids/use_device from CLI args or /dev auto-detect."""
    if devices is not None:
        use_device = bool(devices) and devices != "cpu"
        if use_device:
            role, _ = detect_hardware(cfg["hardware_config"])
        else:
            role = "cpu"
        device_ids = [int(x) for x in devices.split(",")] if use_device else ["cpu"]
    else:
        role, device_ids = detect_hardware(cfg["hardware_config"])
        # cpu 兜底（/dev 无设备）不得置真：cpu profile 无 torch_lib
        use_device = role != "cpu"
    return role, device_ids, use_device


def _warn_max_concurrent(cfg, device_ids):
    device_count = len([d for d in device_ids if d != "cpu"])
    if device_count > 0 and cfg["max_concurrent"] < device_count:
        logging.warning(
            f"max_concurrent ({cfg['max_concurrent']}) < device count ({device_count}); "
            f"data_gate caps total concurrency, {device_count - cfg['max_concurrent']} "
            f"device(s) may idle"
        )


def _apply_handler_config(handler, cfg, role, device_ids, use_device, dry_run, sync_base_dir, tmp_root):
    """Populate XpuRequestHandler class-level config from cfg + runtime state."""
    handler.hardware = role
    handler.profile = cfg["hardware_config"].get(role, {})
    handler.device_ids = device_ids
    handler.hardware_config = cfg["hardware_config"]
    handler.provider_framework = cfg["provider_framework"]
    handler.tenant_manager = TenantManager(sync_base_dir)
    handler.dry_run = dry_run
    # providers 白名单（README：不配 = 探测到什么暴露什么；配置后仅保留
    # 白名单 ∩ 已探测，未安装的条目静默忽略）。
    detected = _detect_frameworks()
    providers_whitelist = cfg.get("providers")
    if providers_whitelist:
        detected = {name: ok for name, ok in detected.items() if name in providers_whitelist}
    handler.frameworks = detected
    handler.device_count = len([d for d in device_ids if d != "cpu"])
    handler.use_device = use_device
    handler.sandbox = cfg["sandbox"]
    handler.docker_images = cfg["docker_images"]
    handler.docker_memory = cfg["docker_memory"]
    handler.docker_network = cfg["docker_network"]
    handler.data_gate = threading.BoundedSemaphore(cfg["max_concurrent"])
    handler.sync_base_dir = sync_base_dir
    handler.tmp_root = tmp_root
    handler.gate_wait_s = cfg["gate_wait_s"]
    handler.device_unhealthy_detect_interval_s = cfg["device_unhealthy_detect_interval_s"]
    handler.run_deadline_s = cfg["run_deadline_s"]


def run_server(port: int, dry_run: bool = False, devices: str = None, bind: str = "127.0.0.1", config_path: str = None):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import sys

    try:
        cfg = load_server_config(config_path)
        bind = bind or cfg["bind"]
        port = port or cfg["port"]
        role, device_ids, use_device = _resolve_devices(devices, cfg)
        _init_device_locks(device_ids)
        _warn_max_concurrent(cfg, device_ids)
    except ValueError as e:
        logging.critical(f"startup failed: {e}")
        sys.exit(1)

    global SYNC_BASE_DIR, TMP_ROOT
    SYNC_BASE_DIR = cfg["sync_dir"]
    TMP_ROOT = cfg["tmp_dir"]
    os.makedirs(SYNC_BASE_DIR, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)

    _apply_handler_config(XpuRequestHandler, cfg, role, device_ids, use_device, dry_run, SYNC_BASE_DIR, TMP_ROOT)

    watcher = threading.Thread(target=_heartbeat_watcher, args=(XpuRequestHandler.tenant_manager,), daemon=True)
    watcher.start()

    class _Server(ThreadingHTTPServer):
        request_queue_size = 128
        allow_reuse_address = True

    server = _Server((bind, port), XpuRequestHandler)

    _apply_tls(server, cfg)
    logging.info(f"xpu_server listening on {bind}:{port} (dry_run={dry_run})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
        server.shutdown()


def _apply_tls(server, cfg):
    """Wrap server socket with mTLS if enabled, else exit on misconfig."""
    import sys

    if not cfg["tls_enabled"]:
        return
    tls_ca = cfg["tls_ca_cert"]
    tls_cert = cfg["tls_server_cert"]
    tls_key = cfg["tls_server_key"]
    if not (tls_ca and tls_cert and tls_key):
        logging.error("tls.enabled=true but cert paths empty; refusing to start")
        sys.exit(1)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(tls_ca)
    ctx.load_cert_chain(tls_cert, tls_key)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    logging.info("mTLS enabled (client cert required)")


def main():
    parser = argparse.ArgumentParser(description="TTK XPU Remote Execution Server")
    parser.add_argument("--port", type=int, default=None, help="Listen port (default: from config yaml or 9090)")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run mode: return random outputs")
    parser.add_argument("--bind", default="", help="Bind address (default: from config or 127.0.0.1)")
    parser.add_argument("--config", default=None, help="Path to xpu_server.yaml")
    parser.add_argument("--devices", default=None, help="Device IDs (e.g. 0,1) or 'cpu'")
    args = parser.parse_args()
    run_server(args.port, dry_run=args.dry_run, devices=args.devices, bind=args.bind, config_path=args.config)


if __name__ == "__main__":
    main()
