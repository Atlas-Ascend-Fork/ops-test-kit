#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Heartbeat subprocess lifecycle management.

Owns the health file path and the heartbeat Process. Called from
InstanceBase.prepare_subprocesss() / close_subprocesss().
"""

import atexit
import contextlib
import logging
import multiprocessing
import os
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

# Subprocess target signature: must accept (endpoints, tenant_id, health_path, tls) as kwargs
HeartbeatTarget = Callable


class HeartbeatManager:
    """Manages heartbeat subprocess for remote endpoint health tracking.

    Lifecycle:
        start()     — called from InstanceBase.prepare_subprocesss()
        supervise() — respawn HB if it died (preserves health file)
        stop()      — called from InstanceBase.close_subprocesss()

    On start: computes health file path, sets TTK_XPU_HEALTH_PATH env var,
    spawns multiprocessing.Process (name="HB") running heartbeat_target.

    On supervise: checks HB liveness, respawns if dead WITHOUT calling stop
    (so the health file is preserved until the new HB's first probe overwrites it).

    On stop: terminates subprocess with 10s grace, sends DELETE to endpoints,
    removes health file, clears env var.
    """

    def __init__(
        self,
        heartbeat_target: HeartbeatTarget,
        root_path: str,
        tenant_id: str,
        endpoints: List,
        tls: Optional[dict] = None,
    ):
        self._target = heartbeat_target
        self._root_path = root_path
        self._tenant_id = tenant_id
        self._endpoints = endpoints
        self._tls = tls
        self._process: Optional[multiprocessing.Process] = None
        self._atexit_registered = False
        self._respawn_count = 0

    @property
    def health_path(self) -> str:
        """Absolute path to health state JSON file."""
        return os.path.join(self._root_path, ".ttk", f"xpu_health_{self._tenant_id}.json")

    def start(self) -> None:
        """Spawn heartbeat subprocess and set TTK_XPU_HEALTH_PATH env var.

        Idempotent: calling start() on an already-running manager is a no-op.
        When endpoints list is empty, sets the path but does not spawn.
        """
        if self._process is not None and self._process.is_alive():
            return  # already running

        os.environ["TTK_XPU_HEALTH_PATH"] = self.health_path

        if not self._endpoints:
            log.info("HeartbeatManager: no remote endpoints, skipping subprocess")
            return

        self._process = multiprocessing.Process(
            name="HB",
            target=self._target,
            kwargs={
                "endpoints": self._endpoints,
                "tenant_id": self._tenant_id,
                "health_path": self.health_path,
                "tls": self._tls,
            },
            daemon=True,
        )
        self._process.start()
        log.info(f"Heartbeat subprocess started: pid={self._process.pid}, health_path={self.health_path}")

        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

    _MAX_RESPAWN_ATTEMPTS = 5

    def supervise(self) -> None:
        """Check HB is alive; respawn if dead (max 5 consecutive attempts, then give up).

        Does NOT call stop (preserves health file). 旧 health 文件保留至新 HB 首探覆盖。
        """
        if self._process is None:
            return
        if self._process.is_alive():
            self._respawn_count = 0  # reset on success
            return

        self._respawn_count += 1
        if self._respawn_count > self._MAX_RESPAWN_ATTEMPTS:
            log.error(
                "HB subprocess died %d consecutive times (last exitcode=%s), "
                "giving up respawn — remote health will be stale",
                self._respawn_count - 1,
                self._process.exitcode,
            )
            self._process = None
            return
        log.warning(
            "HB subprocess died (exitcode=%s), respawning (attempt %d/%d)",
            self._process.exitcode,
            self._respawn_count,
            self._MAX_RESPAWN_ATTEMPTS,
        )
        self._process = multiprocessing.Process(
            name="HB",
            target=self._target,
            kwargs={
                "endpoints": self._endpoints,
                "tenant_id": self._tenant_id,
                "health_path": self.health_path,
                "tls": self._tls,
            },
            daemon=True,
        )
        self._process.start()
        log.info("HB respawned: pid=%s", self._process.pid)

    def _cleanup_tenants(self) -> None:
        """Best-effort tenant DELETE from the parent process.

        On a graceful stop()/atexit the parent is still alive when HB is
        terminated, so HB's own parent-death DELETE never fires and the tenant
        would linger server-side until cleanup_expired. Send DELETE here so
        cleanup is prompt. Idempotent (server tolerates deleting an absent
        tenant). Best-effort: per-endpoint failures are logged, not raised.
        """
        if not self._endpoints or not self._tenant_id:
            return
        from ttk.remote.tls import build_tls_connection

        for ep in self._endpoints:
            host = port = None
            try:
                # Endpoints are Endpoint objs in production (.host/.port); tests
                # may pass (host, port) tuples — handle both.
                host, port = (ep.host, ep.port) if hasattr(ep, "host") else (ep[0], ep[1])
                conn = build_tls_connection(host, port, 5, self._tls)
                conn.request("DELETE", f"/v1/tenant/{self._tenant_id}")
                conn.getresponse().read()
                conn.close()
                log.info("tenant %s deleted on %s:%s", self._tenant_id, host, port)
            except Exception as e:
                log.debug("tenant cleanup on %s:%s failed: %s", host, port, e)

    def stop(self) -> None:
        """Terminate heartbeat subprocess, cleanup tenant state, remove health file."""
        # Parent-side tenant DELETE first — HB's own DELETE only fires on
        # detected parent death, which terminate() doesn't trigger.
        self._cleanup_tenants()

        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._process = None

        # Remove health file on clean stop
        if os.path.exists(self.health_path):
            try:
                os.unlink(self.health_path)
            except OSError as e:
                log.warning(f"Failed to remove health file {self.health_path}: {e}")

        os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def _atexit_cleanup(self) -> None:
        """atexit handler: terminate subprocess + remove health file.

        Covers the case where close_subprocesss() is not called (e.g.,
        unhandled exception in worker loop). Network tenant DELETE is NOT done
        here — atexit runs during interpreter shutdown where sockets/logging
        are torn down. Graceful stop() sends the parent-side DELETE; abnormal
        termination relies on HB's parent-death DELETE (robust ppid check).
        Health file is a local file (os.unlink, no network) — safe to remove here.
        """
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if os.path.exists(self.health_path):
            with contextlib.suppress(OSError):
                os.unlink(self.health_path)
