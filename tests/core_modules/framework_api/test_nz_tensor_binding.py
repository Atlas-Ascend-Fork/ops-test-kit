#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ScaleNZ keeps E8M0 logical order without changing other NZ inputs."""

import sys
from types import SimpleNamespace

import pytest
import torch

from ttk.core_modules.framework_api.torch.tensor_binding import apply_format_cast


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.uint8])
@pytest.mark.parametrize("with_e8m0", [False, True])
def test_non_scale_nz_keeps_original_cast_path(monkeypatch, dtype, with_e8m0):
    tensor = torch.arange(12).reshape(3, 4).to(dtype).T
    seen = []

    def cast(value, fmt):
        seen.append((value, fmt))
        return value.contiguous()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=cast))
    companion = SimpleNamespace(dtype="float8_e8m0") if with_e8m0 else None
    outputs = apply_format_cast([tensor, companion], ["FRACTAL_NZ", "ND"])
    result = outputs[0]
    assert outputs[1] is companion
    assert len(seen) == 1
    assert seen[0][0] is tensor
    assert seen[0][1] == 29
    assert result.is_contiguous()
    torch.testing.assert_close(result, tensor)


def test_canonical_scale_is_not_transposed(monkeypatch):
    seen = []

    def cast(tensor, fmt):
        seen.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=cast))
    scale = torch.ones(3, 17, 2)
    assert apply_format_cast([scale], ["FRACTAL_NZ"])[0] is scale
    assert seen == [(3, 17, 2)]


@pytest.mark.parametrize("transposed", [False, True])
def test_e8m0_pair_axis_keeps_logical_order(monkeypatch, transposed):
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        pytest.skip("torch does not expose E8M0")
    raw = torch.arange(3 * 17 * 2, dtype=torch.uint8).reshape(3, 17, 2)
    if transposed:
        raw = raw.reshape(3, 2, 17).transpose(-1, -2)
    scale = raw.view(dtype)
    seen = []
    contiguous = torch.Tensor.contiguous

    def copy_bytes(value, *args, **kwargs):
        assert value.dtype != dtype, "device contiguous must not run an E8M0 transpose"
        return contiguous(value, *args, **kwargs)

    def cast(tensor, fmt):
        assert tensor.is_contiguous()
        seen.append((tuple(tensor.shape), fmt))
        return tensor.clone()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=cast))
    monkeypatch.setattr(torch.Tensor, "contiguous", copy_bytes)
    result = apply_format_cast([scale], ["FRACTAL_NZ"])[0]
    assert seen == [((3, 17, 2), 29)]
    torch.testing.assert_close(result.view(torch.uint8), raw.contiguous())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8, torch.uint8])
def test_nd_and_contiguous_nz_are_unchanged(monkeypatch, dtype):
    tensor = torch.arange(12).reshape(3, 4).to(dtype)
    seen = []

    def cast(value, fmt):
        seen.append((value, fmt))
        return value

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=cast))
    result = apply_format_cast([tensor, tensor, None], ["ND", "NZ", "NZ"])
    assert result == [tensor, tensor, None]
    assert len(seen) == 1
    assert seen[0][0] is tensor
    assert seen[0][1] == 29
