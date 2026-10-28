# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""mx_quant 锚点：算法行为 + 双 import 路径同源 + 两种入口序无环。

算法本体在 ttk/utilities/mx_quant.py；ttk.utilities.dtypes 尾部保留 PEP 562
惰性转发兼容历史 import 路径——「多路径解析到同一对象且两种入口序均不触发
半初始化环」由子进程干净解释器用例一次钉住；转发层未知名字走正常
AttributeError。
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ttk.utilities.mx_quant import MX_ELEMENT_DTYPES, mx_quantize


def _missing_en_dtypes():
    try:
        import en_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


# ── 算法行为 ─────────────────────────────────────────────────────────


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_mx_quantize_block_extremes():
    """块内极值两端：全零块 share_exp=-inf 被 clamp 到 -127 → 最小 scale、ele
    全零；全 e2m1 max(6.0) 块 share_exp=0 → scale=1.0、ele 精确表示。"""
    scale, ele = mx_quantize(np.zeros((1, 32), dtype=np.float16), "float4_e2m1")
    assert scale.shape == (1, 1, 2)
    assert float(scale.ravel()[0]) == 2.0**-127
    assert not ele.any()

    scale, ele = mx_quantize(np.full((1, 32), 6.0, dtype=np.float16), "float4_e2m1")
    assert float(scale.ravel()[0]) == 1.0
    assert np.all(ele.astype(np.float32) == 6.0)


def test_mx_element_dtypes_gate():
    """MX_ELEMENT_DTYPES 与 mx_quantize 支持集一致，非成员直接拒绝。"""
    assert "float4_e2m1" in MX_ELEMENT_DTYPES
    with pytest.raises(NotImplementedError):
        mx_quantize(np.zeros((2, 32), dtype=np.float16), mx_ele_dtype="float16")


# ── 双 import 路径同源 ──────────────────────────────────────────────────────
# 三路径解析到同一对象的契约由 test_subprocess_import_order_no_cycle 在干净
# 解释器内钉住（两种入口序均断言身份一致）；未知名字走正常 AttributeError。


def test_dtypes_lazy_forward():
    """PEP 562 惰性转发：两个 MX 名字解析成功，其余走正常 AttributeError
    （身份一致性在子进程用例钉住，此处补进程内成功分支）。"""
    from ttk.utilities import dtypes

    assert callable(dtypes.mx_quantize)
    assert "float4_e2m1" in dtypes.MX_ELEMENT_DTYPES
    with pytest.raises(AttributeError, match="mx_quantize_typo"):
        _ = dtypes.mx_quantize_typo


@pytest.mark.parametrize("first", ["ttk.utilities.mx_quant", "ttk.utilities.dtypes"])
def test_subprocess_import_order_no_cycle(first):
    """干净解释器内两种入口序均无半初始化环，双路径仍解析到同一对象。"""
    root = str(Path(__file__).resolve().parents[2])
    snippet = (
        f"import sys; sys.path.insert(0, {root!r}); "
        f"import {first}; "
        "import ttk.utilities.mx_quant as m; "
        "import ttk.utilities.dtypes as d; "
        "from ttk.utilities import mx_quantize, MX_ELEMENT_DTYPES; "
        "assert m.mx_quantize is d.mx_quantize is mx_quantize; "
        "assert m.MX_ELEMENT_DTYPES is d.MX_ELEMENT_DTYPES is MX_ELEMENT_DTYPES"
    )
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
