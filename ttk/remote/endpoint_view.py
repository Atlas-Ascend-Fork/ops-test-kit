#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""EndpointView: worker-side per-process singleton for XPU endpoint state.

Reads health file (written by HB subprocess) + resolves providers + round-robin
load balancing. No writes. All TTK_XPU_* env READS stay in this module.
"""

import logging
import os
import random
import time
from typing import List, Optional

from ttk.remote.health_file import read_health_file
from ttk.utilities import Singleton

log = logging.getLogger(__name__)
HEALTH_BARRIER_TIMEOUT_S = 15.0

# third_party provider aliases — canonical worker-side copy (spec-layer key
# normalization in client.py and CLI --provider parsing share it); the server
# keeps its own in server/executor.py (separate process). Keep in sync.
_TP_ALIASES = {"tensorflow": "tf", "np": "numpy"}


def _parse_provider_filter(raw: Optional[str]) -> Optional[List[str]]:
    """Parse --provider filter from a raw CLI string (comma-separated).

    空字符串/None → None（resolve_providers 见 None 跳过 cli 过滤）。
    别名归一与 spec 层 third_party 键同口径（tensorflow→tf、np→numpy），
    否则 --provider tensorflow 过滤不到规范名 tf 的 endpoint。
    """
    parsed = [p.strip() for p in (raw or "").split(",") if p.strip()]
    parsed = [_TP_ALIASES.get(p, p) for p in parsed]
    return parsed or None


def _wait_health_file(path: str, timeout: float = HEALTH_BARRIER_TIMEOUT_S, interval: float = 0.5) -> None:
    """Lazy barrier: poll os.path.exists until health file appears (once per process)."""
    if not path:
        log.warning("No TTK_XPU_HEALTH_PATH set, skipping health file barrier")
        return
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() >= deadline:
            log.warning("Health file %s not ready after %ss, continuing anyway", path, timeout)
            return
        time.sleep(min(interval, max(0.0, deadline - time.time())))


class EndpointView(metaclass=Singleton):
    """Per-process singleton. Reads health file + resolves providers + round-robin."""

    def __init__(self):
        from ttk.remote.config import get_remote_config

        config = get_remote_config()  # 从 load_config 缓存取
        self._endpoints = config.endpoints if config else []
        self._health_path = os.environ.get("TTK_XPU_HEALTH_PATH", "")  # 保留 env
        _wait_health_file(self._health_path)
        # Shuffle once so per-provider round-robin start offset is randomized across
        # workers (no thundering-herd on the same first endpoint).
        random.shuffle(self._endpoints)
        self._rr_index: dict = {}  # {provider: next offset}

    def resolve_providers(
        self, spec_providers: Optional[List[str]] = None, cli_providers: Optional[List[str]] = None
    ) -> List[str]:
        """all_effective(alive + detect∩yaml) ∩ spec_providers ∩ cli_providers.

        Sequential intersection. Raises RuntimeError when nothing survives
        (fail-loud: caller turns this into a single-case FAIL, worker survives).
        Error messages carry detect/yaml diagnostics.

        Return order: priority = first spec in input order.
        When ``spec_providers`` is given the survivors are returned in spec
        insertion order — the app controls priority via spec order, this layer
        does not silently sort. When no ``spec_providers`` is declared there is
        no priority to honor, so survivors are returned ``sorted`` (deterministic
        neutral).
        """
        effective = self._all_effective_providers()
        if not effective:
            health = read_health_file(self._health_path)
            ep_keys = [f"{e.host}:{e.port}" for e in self._endpoints]
            raise RuntimeError(
                f"no usable provider (detect∩yaml∩alive empty); "
                f"health_present={health is not None}, endpoints={ep_keys}"
            )
        candidates = set(effective)
        if spec_providers:
            candidates &= set(spec_providers)
        if cli_providers:
            candidates &= set(cli_providers)
        if not candidates:
            raise RuntimeError(
                f"no provider after filter (effective={sorted(effective)}, spec={spec_providers}, cli={cli_providers})"
            )
        if spec_providers:
            # preserve spec insertion order = app-declared priority;
            # priority 控制权留在应用层（spec 顺序），不在 resolve 层偷偷排序
            return [p for p in spec_providers if p in candidates]
        return sorted(candidates)

    def pick_endpoint(self, provider: str):
        """Round-robin: next alive endpoint effectively supporting provider, else None.

        Load balancing, NOT failover: each call returns the next EP in rotation.
        The ONLY endpoint decision point.
        """
        eps = self._alive_effective_endpoints(provider)
        if not eps:
            return None
        idx = self._rr_index.get(provider, 0) % len(eps)
        self._rr_index[provider] = idx + 1
        return eps[idx]

    def _all_effective_providers(self) -> set:
        """Union of detect∩yaml across ALIVE endpoints (dead/empty skipped)."""
        health = read_health_file(self._health_path)
        if not health:
            return set()
        result = set()
        for ep in self._endpoints:
            effective = self._ep_effective(ep, health)
            if effective is not None:
                result |= effective
        return result

    def _alive_effective_endpoints(self, provider: str) -> list:
        """Alive endpoints whose effective set contains provider."""
        health = read_health_file(self._health_path) or {}
        result = []
        for ep in self._endpoints:
            effective = self._ep_effective(ep, health)
            if effective and provider in effective:
                result.append(ep)
        return result

    def _ep_effective(self, ep, health) -> Optional[set]:
        """detect∩yaml for one endpoint if alive, else None. yaml not in detect -> warn+drop."""
        eps_state = health.get("endpoints", {})
        ep_key = f"{ep.host}:{ep.port}"
        state = eps_state.get(ep_key, {})
        if not state.get("alive", False):
            return None
        detect = set(state.get("providers", []))
        yaml_filter = set(ep.providers or [])
        if yaml_filter:
            dropped = yaml_filter - detect
            if dropped:
                log.warning("endpoint %s: yaml %s not in detect %s, dropped", ep_key, sorted(dropped), sorted(detect))
            return detect & yaml_filter
        return detect
