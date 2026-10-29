#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""NPUSim simulator backend for the kernel/aclnn test modes.

See ``docs/npusim/npusim_design.md`` for the design.
"""

from .case_writer import case_dir
from .config import resolve_camodel_lib_dir, resolve_platform_soc
from .report import maybe_generate_sim_report
from .sim_profiling import run_aclnn_sim, run_kernel_sim

__all__ = [
    "run_kernel_sim",
    "run_aclnn_sim",
    "case_dir",
    "maybe_generate_sim_report",
    "resolve_platform_soc",
    "resolve_camodel_lib_dir",
]
