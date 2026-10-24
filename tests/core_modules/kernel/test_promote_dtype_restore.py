# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""_promote_dtype must restore the context even when the wrapped block raises.

Regression for: golden_mode=Promote (auto-enabled by the cross_check standard)
lifts context.input_dtypes fp32->fp64 / fp16->fp32 for the golden computation.
The restore used to sit plainly after `yield`, so a golden that raised -- which
is what an invalid testcase does -- left the promoted dtypes on the context.
Everything built afterwards inherited them; the GEIR graph then declared
DT_DOUBLE for a float32 testcase and the operator rejected it, masking the real
error (an out-of-range `dim`) behind "data type DT_DOUBLE is not supported".
"""

import numpy
import pytest

from ttk.core_modules.kernel import golden as output_generation

golden_mode = output_generation.__dict__["_golden_mode"]


class _Ctx:
    """Minimal stand-in exposing only what _promote_dtype touches."""

    def __init__(self):
        self.input_dtypes = ("float32", "float32")
        self.output_dtypes = ("float32",)
        self.input_arrays = (numpy.zeros(2, numpy.float32), numpy.zeros(2, numpy.float32))
        self.original_input_arrays = None
        self.input_distribution = None
        self.output_distribution = None
        self.flat_input_dtypes = ["float32", "float32"]
        self.flat_output_dtypes = ["float32"]

    def invalidate_flat_cache(self, *_names):
        pass


@pytest.mark.parametrize("raises", [False, True])
def test_promote_restores_dtypes(monkeypatch, raises):
    ctx = _Ctx()
    before_in, before_out = ctx.input_dtypes, ctx.output_dtypes
    monkeypatch.setattr(output_generation, "nest_by_distribution", lambda values, _dist: list(values), raising=False)

    if raises:
        with pytest.raises(RuntimeError), golden_mode("Promote", ctx):  # noqa: PT012 上下文管理器组合断言
            # Inside the block the promotion is expected to be in effect.
            assert list(ctx.input_dtypes) == ["float64", "float64"]
            raise RuntimeError("golden failed on an invalid testcase")
    else:
        with golden_mode("Promote", ctx):
            assert list(ctx.input_dtypes) == ["float64", "float64"]

    assert ctx.input_dtypes == before_in, "promoted input dtypes leaked out of the context"
    assert ctx.output_dtypes == before_out, "promoted output dtypes leaked out of the context"
