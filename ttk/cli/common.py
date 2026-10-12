#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import logging

from ttk.cli.bridge import parse_cli_bool
from ttk.remote import is_remote_configured


def _add_io_args(parser):
    group = parser.add_argument_group("input & output")
    output_group = group.add_mutually_exclusive_group()
    output_group.add_argument(
        "-a",
        "--append",
        dest="append_file",
        metavar="FILE",
        help="Append results to existing CSV file; overwrites if file header does not match",
    )
    group.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to ttk config YAML (overrides ~/.config/ttk/ and ./ttk.conf.yaml)",
    )
    group.add_argument("-i", "--input", metavar="FILE", required=True, help="Test case file (csv/xlsx)")
    output_group.add_argument("-o", "--output", metavar="FILE", help="Output CSV file (overwrite existing)")
    group.add_argument("--po", "--progress-output", dest="progress_output", metavar="FILE", help="Progress output file")
    group.add_argument("--sheet", default=None, help="Excel worksheet name (default: first worksheet); ignored for csv")


def _add_case_filter_args(parser):
    group = parser.add_argument_group("case selection")
    group.add_argument(
        "--no-op", "--exclude-operator", dest="exclude_operator", metavar="NAMES", help="Exclude operator name(s)"
    )
    group.add_argument("--op", "--operator", dest="operator", metavar="NAMES", help="Filter by operator name(s)")
    group.add_argument("--priority", metavar="LIST", help="Filter by priority, e.g. --priority=1,3 or --priority=1-3")
    group.add_argument("-t", "--testcase", metavar="NAMES", help="Specify testcase name(s), comma-separated")
    group.add_argument(
        "--tc", "--testcase-count", type=int, dest="testcase_count", metavar="N", help="Randomly pick N test cases"
    )
    group.add_argument(
        "--ti",
        "--testcase-index",
        dest="testcase_index",
        metavar="LIST",
        help="Specify testcase indexes (0-based), e.g. --ti=0,2,6 or --ti=0-9",
    )


def _add_precision_args(parser):
    group = parser.add_argument_group("precision & data")
    group.add_argument(
        "--compare",
        default=None,
        choices=["close", "stat_rel_err", "mixed", "cosine", "binary", "requant", "cross_check"],
        help="Comparison method. Default: per-output routing via Spec.tolerance "
        "(requires --plugin), else mixed (mix_tolerance).",
    )
    group.add_argument(
        "--dump",
        nargs="?",
        const="full",
        metavar="TOKENS",
        help="Dump data: full, in, out, golden, xpu (comma-separated; full includes xpu)",
    )
    group.add_argument(
        "--dump-format",
        dest="dump_format",
        default="bin",
        choices=["bin", "npy", "pt", "print"],
        help="Dump file format (default: bin)",
    )
    group.add_argument(
        "--dump-on-fail", dest="dump_on_fail", action="store_true", help="Dump all data when precision check fails"
    )
    group.add_argument(
        "--golden-mode",
        dest="golden_mode",
        default="Enable",
        choices=["Enable", "Disable", "Promote", "XPU"],
        help="Golden mode: Enable/Disable/Promote (CPU high-precision) / XPU (GPU output as golden)",
    )
    group.add_argument(
        "--input-dist",
        dest="input_dist",
        choices=["uniform", "normal"],
        default="uniform",
        help="Random input data distribution (default: uniform)",
    )
    group.add_argument(
        "--manual-data-dirs",
        dest="manual_data_dirs",
        metavar="DIR",
        nargs="+",
        help="Manual data directories. Load matching testcase data when present; "
        "with --no-prof they are the dump output target (one directory)",
    )
    group.add_argument("--seed", "--random-seed", type=int, dest="random_seed", metavar="N", help="Random seed")


def _add_run_args(parser):
    group = parser.add_argument_group("execution control")
    group.add_argument(
        "--no-memory-check",
        dest="no_memory_check",
        action="store_true",
        help="Skip the host memory gate before per-case workloads (default: gate on; trips only at "
        "critically low free memory, waits at most 5 min then proceeds)",
    )
    group.add_argument(
        "--no-prof",
        dest="no_prof",
        action="store_true",
        help="Generate input/golden and dump manual data without running the "
        "main API; with --dump in,golden writes a restorable dataset "
        "(manual-data dump direction)",
    )
    group.add_argument("--plugin", metavar="PATH", help="External plugin path for customized golden/inputs")
    group.add_argument(
        "--proc-no-reuse", dest="proc_no_reuse", action="store_true", help="Create new process for each case"
    )
    group.add_argument(
        "--provider",
        metavar="PROVIDERS",
        default=None,
        help="Which providers to TEST (e.g. torch,tf). "
        "Test filter — narrows dispatch, does NOT override remote config. "
        "If not set, uses the first available key from spec's third_party.",
    )
    group.add_argument("--rerun", metavar="COLUMNS", help="Rerun failed cases, e.g. --rerun=precision_status")
    group.add_argument(
        "--task-prof",
        dest="task_prof",
        default=True,
        type=parse_cli_bool,
        help="Task-level profiling (msprof) switch (default: true)",
    )
    group.add_argument(
        "--validate", dest="validate_only", action="store_true", help="Validate CSV cases only, skip device execution"
    )


def add_common_args(parser):
    """Common args for the four subcommands (kernel/aclnn/e2e/geir)."""
    _add_io_args(parser)
    _add_case_filter_args(parser)
    _add_precision_args(parser)
    _add_run_args(parser)


def add_perf_args(parser, *, with_core_limit=True):
    """跨通路性能/编译资源参数组（不归属任何 mode 专属组）。

    --xpu-perf 四通路共享（三方性能采集，需远端 XPU 配置）；
    --core-limit 三通路共享（kernel=tiling 上限 / geir=GE 图编译 /
    e2e=GE 图与 aclgraph，eager 不支持）——aclnn 直调 API 无编译面，
    不注册（with_core_limit=False）。解析统一在 args_to_switches
    （四通路必经，收口于此）。
    """
    group = parser.add_argument_group("performance & compilation")
    if with_core_limit:
        group.add_argument(
            "--core-limit",
            dest="core_limit",
            metavar="N|A,V",
            default=None,
            help="Cap the core count visible to compilation. Single int caps AI "
            "cores, 'a,v' caps AI and vector cores separately; unset side falls "
            "back to physical count. Pathway mapping: kernel -> op tiling upper "
            "limit, geir/e2e -> GE graph compile and e2e aclgraph mode "
            "(e2e eager not supported)",
        )
    group.add_argument(
        "--xpu-perf",
        dest="xpu_perf",
        action="store_true",
        help="Collect 3rd-party (XPU) performance per case. "
        "Requires remote XPU config (ttk.conf.yaml or --config). PERF-only.",
    )


def normalize_xpu_options(sw):
    """XPU collection options vs remote config availability.

    --xpu-perf and --golden-mode XPU are explicit requests that depend on
    third-party collection: missing remote config (ttk.conf.yaml / --config)
    is a configuration error and raises (fail fast instead of degrading every
    case to GOLDEN_FAILURE after a full run).
    --dump xpu (explicit or via full) follows the data: disabled with a
    warning when no remote is configured — nothing to dump, nothing written.
    """
    if getattr(sw, "xpu_perf", False) and not is_remote_configured():
        raise RuntimeError("--xpu-perf requires remote XPU config (ttk.conf.yaml or --config), but none is configured.")
    if getattr(sw, "golden_mode", None) == "XPU" and not is_remote_configured():
        raise RuntimeError(
            "--golden-mode XPU requires remote XPU config (ttk.conf.yaml or --config), but none is configured."
        )
    dump_config = getattr(sw, "dump_config", None)
    if getattr(dump_config, "dump_xpu", False) and not is_remote_configured():
        logging.warning("--dump xpu: no remote XPU config (ttk.conf.yaml or --config); XPU output dump skipped")
        dump_config.dump_xpu = False
