#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Serialize an in-memory testcase into the per-case simulation directory.

Layout (one case):
    <sim_output>/<safe_case>/
        switches.pkl            # pickled SWITCHES (kernel wrapper)
        dyn|cst|bin/param.pkl   # pickled RTSProfilingParam per enabled mode
        record_out/{cannsim,npusim}_*/   # NPUSim record archive (CANN cannsim)
        output_*.bin / result.json   # wrapper-written results
"""

import contextlib
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

from ..dump import case_directory_name

# Enabled kernel modes and their directory names.
KERNEL_MODES = ("dyn", "cst", "bin")


def sim_root(sw) -> Path:
    return Path(sw.sim_output_dir)


def case_dir(sw, testcase_name: str) -> Path:
    return sim_root(sw) / case_directory_name(testcase_name)


def ensure_case_dir(sw, testcase_name: str) -> Path:
    d = case_dir(sw, testcase_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_switches(sw, case_path: Path) -> Path:
    pkl = case_path / "switches.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(sw, f, protocol=2)
    return pkl


def dump_kernel_param(param, case_path: Path, mode: str) -> Optional[Path]:
    """Pickle a RTSProfilingParam for one kernel mode; returns the pkl path."""
    mdir = case_path / mode
    mdir.mkdir(parents=True, exist_ok=True)
    pkl = mdir / "param.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(param, f, protocol=2)
    logging.info("serialized %s param -> %s", mode, pkl)
    return pkl


def enabled_kernel_modes(sw) -> tuple:
    """The kernel modes that should actually run (enabled + profiling on)."""
    out = []
    for mode, switch in (("dyn", sw.dyn_switches), ("cst", sw.cst_switches), ("bin", sw.bin_switches)):
        if switch.enabled and switch.prof:
            out.append(mode)
    return tuple(out)


def clear_case_dir(sw, testcase_name: str) -> None:
    d = case_dir(sw, testcase_name)
    if d.exists():
        for child in d.iterdir():
            if child.name == "record_out":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    child.unlink()
    os.makedirs(d, exist_ok=True)
