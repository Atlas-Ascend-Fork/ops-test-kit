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
platform：纯静态映射——OPP/impl/op_info 路径三源解析（builtin/vendor/custom）、
型号→SoC 参数（platform_config ini）、--core-limit 校验与
clear_l0 tile 推导。零 .so 依赖，纯文件/env 解析。
"""

__all__ = [
    "get_ascend_lib64_path",
    "get_ascend_scene_info",
    "get_opp_paths",
    "get_op_impl_paths",
    "get_impl_base_paths",
    "get_op_info_paths",
    "get_npu_hw_info",
    "validate_core_limit",
    "get_l0_clear_tile_params",
    "PLATFORM_BEFORE_DAVID",
]


# Standard Packages
import configparser
import math
import os
from functools import lru_cache
from typing import Tuple

PLATFORM_BEFORE_DAVID = (
    "Ascend031",
    "Ascend310",
    "Ascend310P",
    "Ascend310B",
    "Ascend610",
    "Ascend610Lite",
    "BS9SX1A",
    "Ascend910B",
    "Ascend910_93",
    "Ascend910",
    "Hi3796CV300CS",
    "Hi3796CV300ES",
    "OPTG",
    "SD3403",
    "TsnsC",
    "TsnsE",
)

_VALID_SOURCES = ("builtin", "vendor", "custom")


@lru_cache(maxsize=None)
def get_opp_paths(source: str) -> list:
    """Return opp root paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        opp_path = os.getenv("ASCEND_OPP_PATH", "")
        if not opp_path:
            raise RuntimeError("Environment `ASCEND_OPP_PATH` is not set.")
        return [opp_path]
    if source == "vendor":
        opp_path = os.getenv("ASCEND_OPP_PATH", "")
        if not opp_path:
            return []
        config_file = os.path.join(opp_path, "vendors", "config.ini")
        if not os.path.isfile(config_file):
            # fallback: no vendors config → treat as custom path
            custom_path = os.path.join(opp_path, "op_impl", "custom")
            return [custom_path] if os.path.isdir(custom_path) else []
        vendors = []
        with open(config_file) as f:
            for raw_line in f:
                stripped_line = raw_line.strip()
                if not stripped_line.startswith("load_priority="):
                    continue
                for raw_name in stripped_line.split("=", 1)[1].split(","):
                    name = raw_name.strip()
                    if not name:
                        continue
                    p = os.path.join(opp_path, "vendors", name)
                    if os.path.isdir(p):
                        vendors.append(p)
                break
        return vendors
    if source == "custom":
        env_val = os.getenv("ASCEND_CUSTOM_OPP_PATH", "")
        if not env_val:
            return []
        return [p for p in env_val.split(":") if p and os.path.isdir(p)]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


def _builtin_op_impl_rel():
    """Return builtin op_impl relative path, compatible with both layouts:
    - vendors/ exists → built-in/op_impl (newer CANN)
    - vendors/ absent → op_impl/built-in (older CANN)
    """
    opp_path = get_opp_paths("builtin")[0]
    vendors_dir = os.path.join(opp_path, "vendors")
    return os.path.join("built-in", "op_impl") if os.path.isdir(vendors_dir) else os.path.join("op_impl", "built-in")


@lru_cache(maxsize=None)
def get_op_impl_paths(source: str) -> list:
    """Return {opp}/op_impl paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        return [os.path.join(get_opp_paths("builtin")[0], _builtin_op_impl_rel())]
    if source == "vendor":
        return [os.path.join(p, "op_impl") for p in get_opp_paths("vendor")]
    if source == "custom":
        return [os.path.join(p, "op_impl") for p in get_opp_paths("custom")]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


@lru_cache(maxsize=None)
def get_impl_base_paths(source: str) -> list:
    """Return {opp}/op_impl/ai_core/tbe paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        return [os.path.join(get_opp_paths("builtin")[0], _builtin_op_impl_rel(), "ai_core", "tbe")]
    if source == "vendor":
        return [os.path.join(p, "op_impl", "ai_core", "tbe") for p in get_opp_paths("vendor")]
    if source == "custom":
        return [os.path.join(p, "op_impl", "ai_core", "tbe") for p in get_opp_paths("custom")]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


@lru_cache(maxsize=None)
def get_ascend_scene_info() -> Tuple[str, str]:
    opp_path = get_opp_paths("builtin")[0]
    scene_file = os.path.abspath(os.path.join(opp_path, "scene.info"))
    scene_os, scene_arch = "", ""
    if os.path.isfile(scene_file):
        with open(scene_file) as f:
            scene_info = [line.strip() for line in f.readlines()]
            for item_info in scene_info:
                if "os=" in item_info:
                    scene_os = item_info.split("=")[-1]
                elif "arch=" in item_info:
                    scene_arch = item_info.split("=")[-1]
    return scene_os, scene_arch


@lru_cache(maxsize=None)
def get_op_info_paths(source: str, soc_lower: str) -> list:
    """Return {impl_base}/config/{soc_lower} paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        return [os.path.join(get_impl_base_paths("builtin")[0], "config", soc_lower)]
    if source == "vendor":
        return [os.path.join(p, "op_impl", "ai_core", "tbe", "config", soc_lower) for p in get_opp_paths("vendor")]
    if source == "custom":
        return [os.path.join(p, "op_impl", "ai_core", "tbe", "config", soc_lower) for p in get_opp_paths("custom")]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


@lru_cache(maxsize=None)
def get_ascend_lib64_path():
    opp_path = get_opp_paths("builtin")[0]
    scene_os, scene_arch = get_ascend_scene_info()
    return os.path.abspath(os.path.join(opp_path, "..", f"{scene_arch}-{scene_os}", "lib64"))


@lru_cache(maxsize=None)
def get_npu_hw_info(full_soc_version):
    """Parse hardware info from platform_config ini file.

    Args:
        full_soc_version: e.g. 'Ascend910B3'

    Returns:
        dict with keys: short_soc_version, ccec_aic_version, npu_arch,
        ai_core_cnt, cube_core_cnt, vector_core_cnt, cv_split,
        core_type_list, support_bf16, support_inf_nan,
        l0_a_size, l0_b_size, l0_c_size

    Raises:
        FileNotFoundError: if ini file not found
        RuntimeError: if required fields missing
    """
    opp_path = get_opp_paths("builtin")[0]
    scene_os, scene_arch = get_ascend_scene_info()
    config_dir = os.path.join(opp_path, "..", f"{scene_arch}-{scene_os}", "data", "platform_config")
    ini_path = os.path.join(config_dir, f"{full_soc_version}.ini")
    if not os.path.isfile(ini_path):
        raise FileNotFoundError(f"Platform config not found: {ini_path}")

    cfg = configparser.ConfigParser()
    cfg.read(ini_path)

    raw_fields = {
        "short_soc_version": ("version", "Short_SoC_version"),
        "ccec_aic_version": ("version", "CCEC_AIC_version"),
        "npu_arch": ("version", "NpuArch"),
        "ai_core_cnt": ("SoCInfo", "ai_core_cnt"),
        "cube_core_cnt": ("SoCInfo", "cube_core_cnt"),
        "vector_core_cnt": ("SoCInfo", "vector_core_cnt"),
        "cv_split": ("SoCInfo", "cube_vector_combine"),
        "core_type_list": ("SoCInfo", "core_type_list"),
        "support_bf16": ("SoCInfo", "support_bf16"),
        "support_inf_nan": ("SoCInfo", "support_inf_nan"),
        "l0_a_size": ("AICoreSpec", "l0_a_size"),
        "l0_b_size": ("AICoreSpec", "l0_b_size"),
        "l0_c_size": ("AICoreSpec", "l0_c_size"),
    }

    int_fields = {"ai_core_cnt", "cube_core_cnt", "vector_core_cnt", "l0_a_size", "l0_b_size", "l0_c_size"}
    bool_fields = {"support_bf16", "support_inf_nan"}
    int_defaults = {"cube_core_cnt": 0, "vector_core_cnt": 0, "l0_a_size": 0, "l0_b_size": 0, "l0_c_size": 0}

    result = {}
    missing = []
    for key, (section, option) in raw_fields.items():
        if cfg.has_option(section, option):
            val = cfg.get(section, option)
            if key in bool_fields:
                val = bool(int(val))
            elif key in int_fields:
                val = int(val)
            result[key] = val
        elif key in int_defaults:
            result[key] = int_defaults[key]
        elif key in bool_fields:
            result[key] = False
        else:
            missing.append(option)
    if missing:
        raise RuntimeError(f"[{full_soc_version}] missing fields: {missing} in {ini_path}")

    ccec = result.get("ccec_aic_version", "")
    if ccec.endswith("cube"):
        result["ccec_aic_version"] = ccec[:-4] + "$core_type"

    cv_split_val = result.pop("cv_split", None)
    if cv_split_val is None:
        cv_split_val = cfg.get("SoCInfo", "CUBE_VECTOR_SPLIT", fallback=None)
    if cv_split_val is not None:
        if not isinstance(cv_split_val, (list, tuple)):
            cv_split_val = str(cv_split_val).split(",")
        cv_split = bool("split" in cv_split_val)
    else:
        cv_split = False
    short_soc = result.get("short_soc_version", "")
    if short_soc in ("Ascend310B",):
        cv_split = False
    result["cv_split"] = cv_split

    result["full_soc_version"] = full_soc_version

    return result


def validate_core_limit(core_limit, hw_info: dict):
    """Validate --core-limit against physical SoC core counts from platform info.

    core_limit: (ai_core_limit, vector_core_limit) tuple from SWITCHES, either side
    optionally None. Raises ValueError when a set side exceeds the physical count.
    Sides whose physical count is unavailable (0/None in hw_info) are skipped.
    """
    if not core_limit or not isinstance(core_limit, tuple):
        return
    ai_limit, vec_limit = core_limit
    soc = hw_info.get("full_soc_version", "")
    ai_cnt = int(hw_info.get("ai_core_cnt") or 0)
    vec_cnt = int(hw_info.get("vector_core_cnt") or 0)
    if ai_limit and ai_cnt and ai_limit > ai_cnt:
        raise ValueError(f"--core-limit AI core value {ai_limit} exceeds physical ai_core_cnt {ai_cnt} of {soc}")
    if vec_limit and vec_cnt and vec_limit > vec_cnt:
        raise ValueError(
            f"--core-limit vector core value {vec_limit} exceeds physical vector_core_cnt {vec_cnt} of {soc}"
        )


@lru_cache(maxsize=None)
def get_l0_clear_tile_params(full_soc_version: str) -> Tuple[int, int, int, str]:
    """Compute M/K/N tile dimensions and npu_arch for the clear_l0 Ascend C helper kernel.

    The kernel performs one mmad pass (fp16 A x fp16 B -> fp32 C) that covers the
    full L0A/L0B and as much of L0C as a single pass allows:

        L0A tile: M * K * 2B  (fp16, full L0A)
        L0B tile: K * N * 2B  (fp16, full L0B)
        L0C tile: M * N * 4B  (fp32, <= L0C)

    K is chosen as the smallest multiple of 16 that:
      1. Divides both L0A_SIZE / dtype_bytes and L0B_SIZE / dtype_bytes evenly.
      2. Satisfies M * N * 4 <= L0C_SIZE.

    Args:
        full_soc_version: e.g. 'Ascend950DT_9596'

    Returns:
        (M_DIM, K_DIM, N_DIM, npu_arch) where npu_arch is the ccec --npu-arch value
        (e.g. 'dav-3510').

    Raises:
        RuntimeError: if L0 sizes are zero or no valid K can be found.
    """
    hw_info = get_npu_hw_info(full_soc_version)
    l0a_size = hw_info.get("l0_a_size", 0)
    l0b_size = hw_info.get("l0_b_size", 0)
    l0c_size = hw_info.get("l0_c_size", 0)

    if l0a_size == 0 or l0b_size == 0 or l0c_size == 0:
        raise RuntimeError(
            f"[{full_soc_version}] L0 sizes are zero (l0a={l0a_size}, l0b={l0b_size}, l0c={l0c_size}), "
            "clear_l0 Ascend C kernel not supported"
        )

    dtype_bytes = 2  # fp16
    l0c_dtype_bytes = 4  # fp32

    l0a_elems = l0a_size // dtype_bytes
    l0b_elems = l0b_size // dtype_bytes

    # Minimum K so that M * N * l0c_dtype_bytes <= l0c_size
    # M = l0a_elems / K, N = l0b_elems / K
    # => l0a_elems * l0b_elems * l0c_dtype_bytes / K^2 <= l0c_size
    # => K >= sqrt(l0a_elems * l0b_elems * l0c_dtype_bytes / l0c_size)
    min_k_sq = math.ceil(l0a_elems * l0b_elems * l0c_dtype_bytes / l0c_size)
    min_k = max(16, math.isqrt(min_k_sq - 1) + 1)
    # Round up to multiple of 16 (cube instruction alignment)
    k_dim = ((min_k + 15) // 16) * 16
    # K must divide both element counts evenly
    while k_dim <= min(l0a_elems, l0b_elems):
        if l0a_elems % k_dim == 0 and l0b_elems % k_dim == 0:
            break
        k_dim += 16
    else:
        raise RuntimeError(
            f"[{full_soc_version}] cannot find K (multiple of 16, divisor of "
            f"l0a_elems={l0a_elems} and l0b_elems={l0b_elems}) satisfying L0C constraint"
        )

    m_dim = l0a_elems // k_dim
    n_dim = l0b_elems // k_dim
    npu_arch = f"dav-{hw_info.get('npu_arch', '')}"

    return m_dim, k_dim, n_dim, npu_arch
