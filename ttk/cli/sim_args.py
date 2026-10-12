#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""NPUSim simulator backend: CLI arguments and SWITCHES wiring.

``--backend npusim`` switches the kernel / aclnn execution from a real Ascend
device to the NPUSim (SoC-level simulator) backend. Simulation data flows via a
Python user_app (wrapper) that NPUSim ``record`` launches; results are written
back as ``output_*.bin`` and compared through the existing comparison pipeline.
"""

import os

_BACKEND_DESCRIPTIONS = {
    "npu": "real",
    "npusim": "simulation",
    "cpu": "framework API on CPU",
}


def add_sim_args(parser, *, backends=("npu", "npusim")):
    """Add ``--backend`` and simulator options to a mode parser.

    ``cpu`` is e2e-only (framework APIs can execute on CPU; kernel/aclnn main
    execution needs a device or npusim, CPU reference data is the golden
    system's job) — e2e passes it in ``backends``. Modes without ``npusim``
    (geir: NPU graph only) get ``--backend`` alone — the ``--sim-*`` options
    would be dead surface there, and the group is titled "execution backend"
    (no simulator to name). The ``--backend`` help lists only the actually
    registered values.

    Third-party collection (--golden-mode XPU / --xpu-perf) is remote-only
    (ttk.conf.yaml remote.endpoints); there is no local-execution backend.
    """
    has_sim = "npusim" in backends
    group = parser.add_argument_group("NPUSim simulator backend" if has_sim else "execution backend")
    group.add_argument(
        "--backend",
        choices=list(backends),
        default="npu",
        help="Execution backend: "
        + " / ".join(f"{name} ({_BACKEND_DESCRIPTIONS[name]})" for name in backends)
        + ". Default: npu",
    )
    if not has_sim:
        return
    group.add_argument(
        "--sim-cores",
        dest="sim_cores",
        default=None,
        help="Core selection for record/report, e.g. '0-2,12-14', 'all', '5'",
    )
    group.add_argument(
        "--sim-obj",
        dest="sim_object_file",
        default=None,
        help="Device object file for richer report content (passed to record -f)",
    )
    group.add_argument(
        "--sim-output",
        dest="sim_output_dir",
        default=None,
        help="Directory for simulation intermediate outputs (default: <root>/sim_output)",
    )
    group.add_argument(
        "--sim-report",
        dest="sim_report",
        action="store_true",
        help="Generate trace_core*.json and HTML performance report after simulation",
    )
    group.add_argument(
        "--sim-soc",
        dest="sim_soc_version",
        default=None,
        help="SoC version for simulation (default: Ascend950)",
    )


def apply_sim_args(sw, args):
    """Copy simulator CLI values into SWITCHES, then normalize npusim semantics."""
    if getattr(args, "backend", None):
        sw.backend = args.backend
        if args.backend == "cpu":
            sw.force_cpu = True  # --backend cpu → force_cpu
    if getattr(args, "sim_soc_version", None):
        sw.sim_soc_version = args.sim_soc_version
    if getattr(args, "sim_output_dir", None):
        sw.sim_output_dir = args.sim_output_dir
    if getattr(args, "sim_report", False):
        sw.sim_report = True
    if getattr(args, "sim_cores", None):
        sw.sim_cores = args.sim_cores
    if getattr(args, "sim_object_file", None):
        sw.sim_object_file = args.sim_object_file
    if sw.backend == "npusim":
        _apply_npusim_semantics(sw, args)
    return sw


def _apply_npusim_semantics(sw, args):
    """--backend npusim 归一化 + 与 --no-prof 互斥校验。"""
    _normalize_sim_switches(sw)
    # --no-prof (no device execution) must not be combined with the
    # simulator backend: the backend prepares input/golden itself.
    if getattr(args, "no_prof", False):
        raise ValueError(
            "--no-prof cannot be combined with --backend npusim; "
            "the simulator backend prepares input/golden automatically."
        )


def _normalize_sim_switches(sw):
    """Reuse model semantics so the existing pipeline skips device-only steps.

    - mode=ASCEND_CAMODEL: use_device()=False, run_time=1, rts_stream=None,
      clear_l1/clear_ub skipped (see MODE / RTSInterface).
    - dev_plat = sim_soc_version: avoids get_device_platform() calling
      get_npu_hw_info("Ascend950") which has no platform ini.
    - warmup/task_profiling/deterministic_level off: not applicable to sim.
    - process_per_device=1: avoid multiple camodel instances saturating the CPU.
    """
    from ttk.core_modules.common.simulator.config import resolve_platform_soc
    from ttk.switches import MODE

    sw.mode = MODE.ASCEND_CAMODEL
    # dev_plat is the CANN platform ini name (e.g. Ascend950PR_9589); NPUSim's
    # own SoC name stays in sw.sim_soc_version for ``record -s``.
    sw.dev_plat = resolve_platform_soc(sw.sim_soc_version)
    sw.warmup = False
    sw.task_profiling = False
    sw.deterministic_level = 0
    if not sw.process_per_device:
        sw.process_per_device = 1
    if not sw.sim_output_dir:
        sw.sim_output_dir = os.path.join(sw.root_path, "sim_output")
