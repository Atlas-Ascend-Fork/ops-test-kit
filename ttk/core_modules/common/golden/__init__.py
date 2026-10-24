# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""common/golden：golden-mode NPU/XPU 的共享执行与性能承载（四通路共用）。

公共 API：npu_golden_runner（msprof 采集窗 + CPU 回落）、has_npu_device
（设备能力探测）、xpu_golden_perf（三方设备时间 → ProfileResult）、
golden_perf_columns（ProfileResult → 结果四列）。
"""

from .npu_golden import golden_perf_columns, has_npu_device, npu_golden_runner, xpu_golden_perf

__all__ = ["golden_perf_columns", "has_npu_device", "npu_golden_runner", "xpu_golden_perf"]
