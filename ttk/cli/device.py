#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ttk.cli.bridge import parse_cli_bool


def add_device_args(parser):
    group = parser.add_argument_group("device")
    group.add_argument(
        "--deterministic-level",
        "--dl",
        dest="deterministic_level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Deterministic level: 0=off (default), 1=deterministic "
        "compute (MD5 consistent across NPU runs), "
        "2=strong consistency, "
        "3=deterministic execution; complete batch fields enable phase-two slice compare",
    )
    group.add_argument(
        "--dev",
        "--device",
        dest="device",
        type=int,
        metavar="N",
        help="Number of devices to use (default: all)",
    )
    group.add_argument(
        "--devices",
        "--device-whitelist",
        dest="devices",
        metavar="EXPR",
        help=(
            "Device ID expression: '3' / '0,1,2' / '0-7' (positive pick) with '^' exclusion "
            "(e.g. '0-7,^3' — exclude applies to positive picks; '^' alone = pick all then exclude). "
            "Combine with --dev N: N caps the pool, picks intersect inside it. "
            "(--device-whitelist is the legacy spelling of the same argument)"
        ),
    )
    group.add_argument(
        "--plat", "--platform", dest="platform", metavar="SOC", help="SoC version, e.g. --plat=Ascend910B2"
    )
    group.add_argument(
        "--pc", "--process-count", dest="process_count", type=int, metavar="N", help="Process count per device"
    )
    group.add_argument(
        "--proc-timeout",
        dest="proc_timeout",
        type=int,
        metavar="SEC",
        default=3600,
        help="Per-testcase timeout in seconds (default: 3600)",
    )
    group.add_argument("--run", type=int, metavar="N", help="Execution count (default: 3)")
    group.add_argument(
        "--warmup",
        default=True,
        type=parse_cli_bool,
        help="Warmup before profiling (default: true; false/0/no/off to disable)",
    )


def add_device_control_args(parser, *, kernel_only=False, hbm_simt=False):
    """设备控制组（执行期设备资源/状态控制），三档注册面：

    - --clear-ub/l1/l0 四子命令共享消费：裸 aclrt 与 torch_npu 同进程共存
      （E2E 场景）——UB/L1/L0 是物理资源，谁启 helper kernel 清的都是同一份。
      kernel/e2e/geir 经 kernel/device_state 门面；aclnn 经
      AclInterface→RTSInterface 委托（execute 每轮直调，helper kernel 由
      aclnn instance 预编译）。
    - hbm_simt=True 追加 kernel/aclnn 共享项：--limit 输入张量总量守门
      （testcase.validate）；--simt-stack-* 运行期 rtDeviceSetLimit
      （rts_interface._set_simt_stack_size，Ascend950），对本设备后续任意
      kernel 生效（kernel execute / aclnn AclOpExecutor.rts_context）。
      e2e/geir 不消费不注册。
    - kernel_only=True 追加 kernel 专属项：二进制执行超时/HBM 复用预留/
      执行前清输出/编译期 SIMT UB 尺寸——仅 kernel 通路消费（worker-execute
      与编译选项注入；aclnn 用预编译算子无编译面，不注册）。"""
    group = parser.add_argument_group("device control")
    if kernel_only:
        group.add_argument(
            "--clear-atomic",
            dest="clear_atomic",
            nargs="?",
            const=True,
            help="Force clear output and workspace memory before execution",
        )
    group.add_argument(
        "--clear-l0",
        dest="clear_l0",
        metavar="VALUE",
        default=None,
        help="Clear L0A/L0B/L0C to specified value before each run (default: 0)",
    )
    group.add_argument(
        "--clear-l1",
        dest="clear_l1",
        metavar="VALUE",
        default=None,
        help="Clear L1 to specified value before each run (default: 0)",
    )
    group.add_argument(
        "--clear-ub",
        dest="clear_ub",
        metavar="VALUE",
        default=None,
        help="Clear UB to specified value before each run (default: 0)",
    )
    if hbm_simt:
        group.add_argument(
            "-l",
            "--limit",
            type=int,
            metavar="GB",
            default=None,
            help="HBM memory limit per testcase in GB (default: 30)",
        )
    if kernel_only:
        group.add_argument(
            "--npu-timeout",
            dest="npu_timeout",
            type=int,
            default=0,
            help="NPU execution timeout in ms; 0 = no timeout, wait until done (default: 0). "
            "Host-side per-testcase timeout: see --proc-timeout",
        )
        group.add_argument("--reserve-hbm", dest="reserve_hbm", type=int, default=None, help="Reserve N MB of HBM")
        group.add_argument("--reuse-hbm", dest="reuse_hbm", action="store_true", help="Reuse input/output HBM")
    if hbm_simt:
        group.add_argument(
            "--simt-stack-dcu",
            dest="simt_stack_dcu",
            type=int,
            metavar="BYTES",
            default=None,
            help="SIMT DCU stack size",
        )
        group.add_argument(
            "--simt-stack-dvg",
            dest="simt_stack_dvg",
            type=int,
            metavar="BYTES",
            default=None,
            help="SIMT DVG stack size",
        )
    if kernel_only:
        group.add_argument(
            "--simt-ub", dest="simt_ub", type=int, metavar="BYTES", default=None, help="Force SIMT UB size in bytes"
        )
