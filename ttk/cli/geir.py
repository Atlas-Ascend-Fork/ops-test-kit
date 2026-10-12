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
    _apply_device_state_args,
    args_to_switches,
    configure_manual_data,
    parse_cli_bool,
    run_with_switches,
)
from ttk.cli.common import add_common_args, add_perf_args, normalize_xpu_options
from ttk.cli.device import add_device_args, add_device_control_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_geir_command(subparsers):
    parser = subparsers.add_parser(
        "geir",
        usage="ttk geir [OPTION]... -i FILE",
        help="GEIR mode: GE graph compile + execute + compare",
    )
    add_common_args(parser)
    add_device_args(parser)
    add_device_control_args(parser)
    add_perf_args(parser)
    _add_geir_args(parser)
    # geir 无 npusim/cpu 形态——--backend 仅声明执行后端
    add_sim_args(parser, backends=("npu",))
    parser.set_defaults(handler=_handle_geir)
    return parser


def _add_geir_args(parser):
    group = parser.add_argument_group("geir mode")
    group.add_argument(
        "-b",
        "--binary",
        nargs="?",
        const="release",
        default=None,
        help="Binary reuse: ge.jit_compile=0; use -b=release to reuse compiled kernel",
    )
    group.add_argument(
        "-c",
        "--const",
        nargs="?",
        const="true",
        default=None,
        help="Enable const (static) shape test. Default: enabled. -c=false to disable.",
    )
    group.add_argument(
        "-d",
        "--dynamic",
        nargs="?",
        const="true",
        default=None,
        help="Enable dynamic shape test. Default: disabled. -d or -d=true to enable.",
    )


def _handle_geir(args):
    sw = args_to_switches(args)
    sw.test_mode = "geir"
    # Default: const on, dynamic off
    sw.cst_switches.enabled = True
    sw.dyn_switches.enabled = False
    if args.const is not None:
        sw.cst_switches.enabled = parse_cli_bool(args.const)
    if args.dynamic is not None:
        sw.dyn_switches.enabled = parse_cli_bool(args.dynamic)
    if args.binary is not None:
        val = args.binary
        if isinstance(val, str):
            val = parse_cli_bool(val)
        sw.geir_binary = val
    else:
        sw.geir_binary = False
    _apply_device_state_args(sw, args)
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "geir")
    normalize_xpu_options(sw)
    run_with_switches(sw)
