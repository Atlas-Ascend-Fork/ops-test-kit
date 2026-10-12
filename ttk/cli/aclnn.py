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
    apply_aclnn_args,
    args_to_switches,
    configure_manual_data,
    run_with_switches,
)
from ttk.cli.common import add_common_args, add_perf_args, normalize_xpu_options
from ttk.cli.device import add_device_args, add_device_control_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_aclnn_command(subparsers):
    parser = subparsers.add_parser(
        "aclnn",
        usage="ttk aclnn [OPTION]... -i FILE",
        help="ACLNN API mode: aclnn* C API execute + compare",
    )
    add_common_args(parser)
    add_device_args(parser)
    add_device_control_args(parser, hbm_simt=True)
    add_perf_args(parser, with_core_limit=False)
    add_sim_args(parser)
    parser.set_defaults(handler=_handle_aclnn)
    return parser


def _handle_aclnn(args):
    sw = args_to_switches(args)
    sw.test_mode = "aclnn"
    apply_aclnn_args(sw, args)
    from ttk.cli.bridge import _apply_device_state_args

    _apply_device_state_args(sw, args)
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "aclnn")
    normalize_xpu_options(sw)
    run_with_switches(sw)
