#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
from ttk.cli.common import add_common_args, add_perf_args, normalize_xpu_options
from ttk.cli.device import add_device_args, add_device_control_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_e2e_command(subparsers):
    parser = subparsers.add_parser(
        "e2e",
        usage="ttk e2e [OPTION]... -i FILE",
        help="Framework API mode: torch_npu / TensorFlow end-to-end test",
    )
    add_common_args(parser)
    add_device_args(parser)
    add_device_control_args(parser)
    add_perf_args(parser)
    _add_e2e_args(parser)
    add_sim_args(parser, backends=("npu", "npusim", "cpu"))
    parser.set_defaults(handler=_handle_e2e)
    return parser


def _add_e2e_args(parser):
    group = parser.add_argument_group("e2e mode")
    group.add_argument(
        "--aclgraph",
        action="store_true",
        default=False,
        help="Enable aclgraph mode via torch.compile (kernel compile mode follows -b; binary reuse by default)",
    )
    group.add_argument(
        "-b",
        "--binary",
        nargs="?",
        const=True,
        default=None,
        help="Reuse binary kernels by default for all e2e legs (eager: jitCompile=disable; "
        "-c/-d GE graph: ge.jit_compile=0; aclgraph: static_kernel_compile=False); "
        "-b=false restores online kernel compilation; cannot be combined with --super-kernel",
    )
    group.add_argument(
        "-c",
        "--const",
        nargs="?",
        const=True,
        default=None,
        help="Enable static shape graph test (default: disabled); use -c=false to disable",
    )
    group.add_argument(
        "-d",
        "--dynamic",
        nargs="?",
        const=True,
        default=None,
        help="Enable dynamic shape graph test (default: disabled); use -d=false to disable",
    )
    group.add_argument(
        "--eager",
        dest="eager",
        nargs="?",
        const=True,
        default=None,
        help="Enable the eager (single-API) execution leg (default: enabled); "
        "use --eager=false to skip it — precision then anchors on graph legs, "
        "requires a graph mode: -c/-d/--aclgraph",
    )
    group.add_argument(
        "--fullgraph",
        dest="fullgraph",
        default=0,
        type=int,
        choices=[0, 1],
        help="Capture full graph in torch.compile (0=off, 1=on; default: 0)",
    )
    group.add_argument(
        "--super-kernel",
        dest="super_kernel",
        action="store_true",
        default=False,
        help="Run graph mode ops through SuperKernel compilation (scope marking in GE graph mode, "
        "super_kernel_optimize option in aclgraph mode); requires -c/-d/--aclgraph",
    )


def _handle_e2e(args):
    sw = args_to_switches(args)
    sw.test_mode = "framework-api"
    sw.dyn_switches.enabled = False
    apply_e2e_args(sw, args)
    from ttk.cli.bridge import _apply_device_state_args

    _apply_device_state_args(sw, args)
    # Must run before configure_manual_data: rejects --no-prof + --backend npusim.
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "e2e")
    normalize_xpu_options(sw)
    run_with_switches(sw)
