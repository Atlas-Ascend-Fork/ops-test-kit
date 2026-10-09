# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from ttk.cli.bridge import (
    apply_e2e_args,
    args_to_switches,
    configure_manual_data,
    run_with_switches,
)
from ttk.cli.common import add_common_args, validate_xpu_perf_precondition
from ttk.cli.device import add_device_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_e2e_command(subparsers):
    parser = subparsers.add_parser("e2e", help="Framework API mode: torch_npu / TensorFlow end-to-end test")
    add_common_args(parser)
    add_device_args(parser)
    _add_e2e_args(parser)
    parser.set_defaults(handler=_handle_e2e)


def _add_e2e_args(parser):
    parser.add_argument(
        "--no-prof", action="store_true", help="Prepare input and CPU golden data without running the main API"
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU backend")
    parser.add_argument(
        "-d",
        "--dynamic",
        nargs="?",
        const=True,
        default=None,
        help="Enable dynamic shape graph test (default: disabled); use -d=false to disable",
    )
    parser.add_argument(
        "-c",
        "--const",
        nargs="?",
        const=True,
        default=None,
        help="Enable static shape graph test (default: disabled); use -c=false to disable",
    )
    parser.add_argument(
        "--aclgraph",
        action="store_true",
        default=False,
        help="Enable aclgraph mode via torch.compile with online kernel compilation by default",
    )
    parser.add_argument(
        "-b",
        "--binary",
        nargs="?",
        const="release",
        default=None,
        choices=("release", "true", "false", "1", "0", "yes", "no", "on", "off"),
        help="Reuse binary kernels for --aclgraph without online kernel compilation; "
        "-b or -b=release enables reuse, -b=false restores online compilation; incompatible with --super-kernel",
    )
    parser.add_argument(
        "--fullgraph",
        dest="fullgraph",
        default=0,
        type=int,
        help="Capture full graph in torch.compile (0=off, 1=on; default: 0)",
    )
    parser.add_argument(
        "--xpu-perf",
        dest="xpu_perf",
        action="store_true",
        help="Collect 3rd-party (XPU) performance per case. "
        "Requires remote XPU config (ttk.conf.yaml or --config). PERF-only.",
    )
    parser.add_argument(
        "--xpu-zero-upload",
        dest="xpu_zero_upload",
        action="store_true",
        help="Send input generation recipes + per-tensor sha256 instead of the input data itself; "
        "the server regenerates and verifies each digest. Saves the upload over slow links. "
        "Falls back to full upload automatically when a digest does not match.",
    )
    parser.add_argument(
        "--core-limit",
        dest="core_limit",
        default=None,
        help="Cap graph core count. Single int caps AI cores, 'a,v' caps AI and vector cores separately; "
        "unset side falls back to physical count. Static/dynamic GE and ACLGraph modes",
    )
    parser.add_argument(
        "--super-kernel",
        dest="super_kernel",
        action="store_true",
        default=False,
        help="Run graph mode ops through SuperKernel compilation (scope marking in GE graph mode, "
        "super_kernel_optimize option in aclgraph mode); requires -c/-d/--aclgraph",
    )
    add_sim_args(parser)


def _handle_e2e(args):
    sw = args_to_switches(args)
    sw.test_mode = "framework-api"
    sw.dyn_switches.enabled = False
    apply_e2e_args(sw, args)
    # Must run before configure_manual_data: rejects --no-prof + --backend npusim.
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "e2e")
    validate_xpu_perf_precondition(sw)
    run_with_switches(sw)
