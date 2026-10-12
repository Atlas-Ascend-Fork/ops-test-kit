#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
def register_info_command(subparsers):
    parser = subparsers.add_parser("info", help="Show device/environment information")
    parser.set_defaults(handler=_handle_info)
    return parser


def _handle_info(args):
    import logging

    _prev_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    try:
        from ttk.core_modules.common.device import DSMIInterface

        dsmi = DSMIInterface()
        count = dsmi.get_device_count()
        if count == 0:
            print("No Ascend devices found.")
            return
        print(f"Found {count} device(s):\n")
        print(f"  {'ID':>3}  {'Platform':<16s} {'Ver':<4s}  {'Temp':>5}  {'Util':>5}")
        print("-" * 45)
        for i in range(count):
            try:
                chip = dsmi.get_chip_info(i)
                platform = chip.get_complete_platform()
                ver = chip.get_ver()
            except Exception:
                platform = "N/A"
                ver = ""
            try:
                temp = dsmi.get_device_temperature(i)
                temp = f"{temp}C"
            except Exception:
                temp = "N/A"
            try:
                util = dsmi.get_device_util(i, 2)
                util = f"{util}%" if util is not None else "N/A"
            except Exception:
                util = "N/A"
            print(f"  {i:>3}  {platform:<16s} {ver:<4s}  {temp:>5}  {util:>5}")
    except ImportError:
        print("DSMI module not available. Is CANN installed?")
    except Exception as e:
        print(f"Error querying devices: {e}")
    finally:
        logging.getLogger().setLevel(_prev_level)
