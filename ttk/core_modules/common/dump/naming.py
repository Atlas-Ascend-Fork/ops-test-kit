# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""落盘命名协议：case 目录安全名与 bin 文件名 shape 段编解码。

dumper（调试 dump）与 manual_data（可回放数据集）共用，放底层避免双向借调。
"""

import hashlib
import re
from typing import Optional, Tuple

_SAFE_CASE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def case_directory_name(testcase_name: str) -> str:
    """Return the on-disk directory name for a manual-data testcase.

    Public because the simulator backend's ``case_writer`` reuses the same
    naming so manual-data replay and simulation share one per-case directory.
    """
    safe = _SAFE_CASE_NAME.sub("_", testcase_name).strip("._") or "case"
    if safe == testcase_name and len(safe) <= 120:
        return safe
    digest = hashlib.sha256(testcase_name.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:96]}-{digest}"


def encode_shape(shape) -> str:
    """Encode the storage shape into the bin filename segment.

    Inverse of :func:`decode_shape`: an empty shape encodes as ``scalar``,
    dimensions are joined with ``x`` (``(2, 3)`` -> ``2x3``).
    """
    shape = tuple(int(dimension) for dimension in shape)
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def decode_shape(encoded_shape: Optional[str]) -> Optional[Tuple[int, ...]]:
    """bin 文件名 shape 段解析（``scalar`` -> ``()``，``2x3`` -> ``(2, 3)``）。"""
    if encoded_shape is None:
        return None
    if encoded_shape == "scalar":
        return ()
    return tuple(int(dimension) for dimension in encoded_shape.split("x"))
