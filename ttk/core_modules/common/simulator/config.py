#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared constants for the NPUSim simulator backend."""

import os
import subprocess
import sys
from pathlib import Path

# NPUSim SoC version (``npusim record -s``) -> CANN platform ini name (the
# ``full_soc`` TTK uses for get_npu_hw_info / compilation). The ini lives under
# ``<opp>/../aarch64-linux/data/platform_config/<name>.ini`` and mirrors the
# same chip (NpuArch etc.) as the camodel NPUSim simulates.
SIM_PLATFORM_BY_SOC = {
    "Ascend950": "Ascend950PR_9589",
    "Ascend950DT": "Ascend950PR_9599",
}


def resolve_platform_soc(soc_version: str) -> str:
    """Map an NPUSim SoC version to the TTK platform name; passthrough if unknown."""
    return SIM_PLATFORM_BY_SOC.get(soc_version, soc_version)


def _cannsim_model_name(soc_version: str) -> str:
    """Read the camodel ``model_name`` for a SoC from the installed cannsim.

    The mapping lives in the CANN-bundled cannsim package
    (``cannsim.core.public.soc_info.SOC_INFO``) and differs across CANN
    versions (9.1.0: ``Ascend950 -> Ascend950PR_9599``, 9.2.0: ->
    ``Ascend950PR_9589``), so TTK reads it at runtime instead of hard-coding.
    Returns ``""`` when the cannsim package / SoC key is unavailable.
    """
    asc_home = os.getenv("ASCEND_TOOLKIT_HOME", "")
    site_packages = os.path.join(asc_home, "python", "site-packages")
    if not os.path.isdir(site_packages):
        return ""
    # ``soc_version`` / ``site_packages`` are passed as argv, never interpolated
    # into the code string, so a crafted value cannot inject Python code. Using
    # ``sys.executable`` keeps the cannsim import resolving to the same
    # interpreter as TTK (mirrors npusim_runner._cannsim_cmd).
    code = (
        "import sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from cannsim.core.public.soc_info import SOC_INFO; "
        "print(SOC_INFO.get(sys.argv[2], {}).get('model_name', ''))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, site_packages, soc_version],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        name = result.stdout.strip()
        if result.returncode == 0 and name:
            return name
    except (subprocess.TimeoutExpired, OSError):
        pass
    return ""


def resolve_camodel_lib_dir(soc_version: str) -> Path:
    """Return the camodel directory cannsim would use for ``soc_version``.

    Layout: ``$ASCEND_TOOLKIT_HOME/tools/simulator/<model_name>/camodel`` (the
    same ``ca_model_path`` cannsim's record computes). Used by the E2E npusim
    backend to inject the camodel runtime into LD_LIBRARY_PATH. Raises
    RuntimeError when the directory cannot be resolved.
    """
    asc_home = os.getenv("ASCEND_TOOLKIT_HOME", "")
    if not asc_home:
        raise RuntimeError("ASCEND_TOOLKIT_HOME is not set; install a CANN toolkit that ships cannsim.")
    model = _cannsim_model_name(soc_version) or SIM_PLATFORM_BY_SOC.get(soc_version, "")
    if not model:
        raise RuntimeError(
            f"Unknown NPUSim SoC version: {soc_version!r}. "
            "Use a SoC registered in the installed cannsim's SOC_INFO "
            "(e.g. Ascend950 / Ascend950DT / Ascend960)."
        )
    camodel = Path(asc_home) / "tools" / "simulator" / model / "camodel"
    if not camodel.is_dir():
        raise RuntimeError(
            f"camodel directory not found: {camodel}. Check that {asc_home} provides tools/simulator/{model}."
        )
    return camodel
