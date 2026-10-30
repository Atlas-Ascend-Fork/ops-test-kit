# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from ttk.cli.bridge import (
    _apply_device_state_args,
    apply_kernel_args,
    args_to_switches,
    configure_manual_data,
    run_with_switches,
)
from ttk.cli.common import add_common_args, add_perf_args, normalize_xpu_options
from ttk.cli.device import add_device_args, add_device_control_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_kernel_command(subparsers):
    parser = subparsers.add_parser(
        "kernel",
        usage="ttk kernel [OPTION]... -i FILE",
        help="Op mode: TBE/Davinci kernel compile + execute + compare",
    )
    add_common_args(parser)
    add_device_args(parser)
    add_device_control_args(parser, kernel_only=True, hbm_simt=True)
    add_perf_args(parser)
    _add_kernel_args(parser)
    add_sim_args(parser)
    parser.set_defaults(handler=_handle_kernel)
    return parser


def _add_kernel_args(parser):
    group = parser.add_argument_group("kernel mode")
    group.add_argument(
        "-b",
        "--binary",
        nargs="?",
        const=True,
        default=None,
        help="Binary mode; use -b=release to reuse pre-compiled release kernel",
    )
    group.add_argument(
        "--co", "--compile-only", dest="compile_only", action="store_true", help="Compile only, skip profiling"
    )
    group.add_argument(
        "--compile-opts",
        dest="compile_opts",
        action="append",
        metavar="KEY=VALUE",
        help="Compile option as key=value. Can be specified multiple times. "
        "e.g. --compile-opts op_debug_config=oom,dump_cce "
        "--compile-opts enable_deterministic_mode=1 "
        '--compile-opts "macros=-DFOO=1 -DBAR=2"',
    )
    group.add_argument(
        "-c", "--const", nargs="?", const=True, default=None, help="Enable const shape test; use -c=false to disable"
    )
    group.add_argument(
        "-d",
        "--dynamic",
        nargs="?",
        const=True,
        default=None,
        help="Enable dynamic shape test (default: enabled); use -d=false to disable",
    )
    group.add_argument(
        "--expect-impl-source",
        dest="expect_impl_source",
        default=None,
        metavar="SOURCE",
        help="Fail the run unless every tested operator resolves from the expected "
        "implementation source: builtin | vendor | custom | vendor=<name> | custom=<name> "
        "(guards against silent fallback to builtin when a custom op package is ineffective)",
    )
    group.add_argument(
        "--force-block-dim", dest="force_block_dim", default=None, help="Force block dim, e.g. --force-block-dim=2"
    )
    group.add_argument(
        "--tr", "--tiling-run", dest="tiling_run", type=int, default=None, help="Tiling function run times (default: 3)"
    )


def _handle_kernel(args):
    sw = args_to_switches(args)
    sw.test_mode = "op"
    apply_kernel_args(sw, args)
    _apply_device_state_args(sw, args)
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "kernel")

    normalize_xpu_options(sw)
    run_with_switches(sw)
