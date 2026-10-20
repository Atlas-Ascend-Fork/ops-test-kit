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
配置加载：default.yaml（基础）→ 用户配置（覆盖）→ CLI（覆盖）。
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── 用户配置文件搜索路径 ──
_CONFIG_PATHS = [
    Path.home() / ".config" / "ttk" / "ttk.conf.yaml",
    Path("ttk.conf.yaml"),
]

_config: Optional[dict] = None


# ====================================================================
# 数据模型
# ====================================================================


@dataclass
class Endpoint:
    """远端 XPU 服务器端点。"""

    host: str
    port: int
    providers: Optional[List[str]] = None
    hardware: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Endpoint":
        return cls(
            host=d["host"],
            port=d["port"],
            providers=d.get("providers"),
            hardware=d.get("hardware"),
        )


@dataclass
class RemoteConfig:
    """远端执行配置。"""

    endpoints: List[Endpoint] = field(default_factory=list)
    backoff_base_s: float = 0.5
    backoff_max_s: float = 10.0
    backoff_jitter: float = 0.25
    max_503_retries: int = 10
    max_conn_retries: int = 5
    dispatch_deadline_s: int = 300
    # XPU/NPU 执行重叠总闸（false = 回串行时序；XPU 服务端负载紧张时关闭）
    xpu_overlap: bool = True
    tls_ca: str = ""
    tls_cert: str = ""
    tls_key: str = ""
    tls_skip_verify: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "RemoteConfig":
        endpoints = [Endpoint.from_dict(ep) for ep in d.get("endpoints", []) if isinstance(ep, dict)]
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "endpoints"}
        return cls(endpoints=endpoints, **fields)


def get_remote_config() -> Optional[RemoteConfig]:
    """从已加载的 _config 取 remote 段。无 endpoints 返回 None。"""
    cfg = get_config()
    remote = cfg.get("remote", {})
    if not remote.get("endpoints"):
        return None
    return RemoteConfig.from_dict(remote)


def get_hardware_config() -> dict:
    """frameworks 段；空返回 {}（仅 cpu 合法默认，区别 remote 的 None）。"""
    return get_config().get("frameworks", {}) or {}


# ── CascadeConfig ──


@dataclass
class CascadeConfig:
    """MC2 算子级联通信端口配置。"""

    port_base: int = 30000
    port_step: int = 13
    port_max: int = 60000
    hccl_port_range: str = "50000-50100"

    @classmethod
    def from_dict(cls, d: dict) -> "CascadeConfig":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


def get_cascade_config() -> CascadeConfig:
    """从已加载的 _config 取 cascade 段；无则返回默认。"""
    raw = get_config().get("cascade", {})
    return CascadeConfig.from_dict(raw)


# ====================================================================
# 加载逻辑
# ====================================================================


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并配置：override 覆盖 base。

    仅 dict 类型逐层合并；list 与标量由 override 整体替换
    （配置语义：list 不做元素级合并，整体替换更可预期，避免静默拼凑）。
    合并结果与 base/override 完全隔离：对结果的嵌套修改不会写回输入。
    """
    merged = copy.deepcopy(base)
    for k, v in override.items():
        cur = merged.get(k)
        if isinstance(cur, dict) and isinstance(v, dict):
            merged[k] = deep_merge(cur, v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _load_yaml(path) -> dict:
    """加载 YAML 文件，返回 dict。文件不存在返回 {}；格式错抛 yaml.YAMLError（fail-loud）。"""
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text("utf-8")) or {}


def _default_config_path() -> Path:
    """默认配置文件路径（包内 default.yaml）。"""
    return Path(__file__).parent / "default.yaml"


def load_config(config_path: str = None) -> dict:
    """加载配置：default.yaml → 用户文件 → CLI 覆盖。

    Args:
        config_path: 可选 CLI 指定的配置文件路径（不存在时报错——显式路径
            拼写错误不应静默回退默认配置；标准搜索路径缺失仍视为正常）

    Returns:
        合并后的配置 dict
    """
    global _config

    # 1. 基础：包内 default.yaml
    cfg = _load_yaml(_default_config_path())

    # 2. 覆盖：用户配置文件（按优先级）
    for path in _CONFIG_PATHS:
        if path.exists():
            logging.debug(f"Loading config: {path}")
            cfg = deep_merge(cfg, _load_yaml(path))

    # 3. 覆盖：CLI 指定的配置文件
    if config_path:
        if not Path(config_path).exists():
            raise ValueError(f"config file not found: {config_path}")
        cfg = deep_merge(cfg, _load_yaml(config_path))

    _config = cfg
    return _config


def get_config() -> dict:
    """返回已加载的配置。_config is None 时 raise（调用方应先 load_config）。

    删 lazy fallback：原行为是未加载时静默 load_config()（丢 --config），
    掩盖了 forkserver worker 拿不到配置的 bug。改为显式 raise 让问题早暴露。
    """
    if _config is None:
        raise RuntimeError("config not loaded — call load_config() first")
    return _config
