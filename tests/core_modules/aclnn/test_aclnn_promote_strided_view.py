#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""ACLNN _promote_dtype must not materialize strided views (storage > view).

Regression for: golden_mode=Promote (forced by mix_tolerance / mixed /
cross_check standards) promoted torch inputs via bare .to(dtype). For a
strided view over a larger physical storage — e.g. a FRACTAL_NZ logical
view whose ceil-aligned fractal storage carries tail-block padding — .to()
materializes the view: the new tensor only keeps the logical-view elements
and every byte beyond the view is lost. Plugin goldens that decode NZ
layouts by walking untyped_storage() then see fewer elements than the CSV
tensor_storage_shapes declare and fail ("WeightNZ storage has N elements,
expected at least M"). Only K-block-aligned cases (view == storage) escaped.

Additional edge cases below probe the promote context further: TensorList
nesting, mixed dtypes / None slots, bfloat16, transposed views, tail-offset
views, empty views, 0-dim tensors, scalar promotion and mode gating.
"""

import types

import numpy
import torch

from ttk.core_modules.aclnn.golden import GoldenGenerator


class _Ctx:
    """Minimal stand-in exposing only what _promote_dtype touches."""

    def __init__(self, tensors, dtypes, scalars=None, scalar_dtypes=()):
        self.tensors = tensors
        self.tensor_dtypes = dtypes
        self.scalars = scalars
        self.scalar_dtypes = scalar_dtypes
        self.tensor_list_dist = ()
        self.scalar_list_dist = ()
        self.flat_tensor_dtypes = tuple(dtypes)
        self.flat_scalar_dtypes = tuple(scalar_dtypes)
        self.pure_output_indexes = ()

    def invalidate_flat_cache(self, *_names):
        pass


def _generator(ctx, golden_mode="Promote"):
    gen = GoldenGenerator.__new__(GoldenGenerator)
    gen._ctx = ctx
    gen._switch = types.SimpleNamespace(golden_mode=golden_mode)
    return gen


def _storage_numel(t):
    return t.untyped_storage().nbytes() // t.element_size()


def test_promote_preserves_strided_view_storage():
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    ctx = _Ctx((view,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert tuple(promoted.shape) == (3, 2)
        assert tuple(promoted.stride()) == (2, 1)
        assert promoted.storage_offset() == 0
        # whole physical storage survives the cast (tail padding included)
        assert _storage_numel(promoted) == 8
        assert torch.equal(torch.as_strided(promoted, (8,), (1,), 0), storage.to(torch.float32))
        assert torch.equal(promoted, view.to(torch.float32))
    assert ctx.tensors[0] is view, "promoted tensors leaked out of the context"


def test_promote_preserves_view_offset_and_stride():
    storage = torch.arange(12, dtype=torch.float16)
    view = torch.as_strided(storage, (2, 3), (4, 1), 5)
    ctx = _Ctx((view,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert tuple(promoted.stride()) == (4, 1)
        assert promoted.storage_offset() == 5
        assert _storage_numel(promoted) == 12
        assert torch.equal(promoted, view.to(torch.float32))


def test_promote_contiguous_tensor_stays_plain():
    t = torch.ones((2, 3), dtype=torch.float16)
    ctx = _Ctx((t,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert promoted.is_contiguous()
        assert _storage_numel(promoted) == 6
        assert torch.equal(promoted, torch.ones((2, 3), dtype=torch.float32))


def test_promote_tensor_list_nested_strided_view():
    """TensorList (dist=(0,2)) containing a strided view: nesting and per-tensor geometry survive."""
    storage = torch.arange(10, dtype=torch.float16)
    view = torch.as_strided(storage, (4, 2), (2, 1), 0)
    plain = torch.ones((2, 2), dtype=torch.float16)
    ctx = _Ctx((plain, (view, plain)), ("float16", "float16", "float16"))
    ctx.tensor_list_dist = (0, 2)
    with _generator(ctx)._golden_mode():
        top = ctx.tensors
        assert isinstance(top, (list, tuple))
        assert len(top) == 2
        promoted_list = top[1]
        assert isinstance(promoted_list, (list, tuple))
        assert len(promoted_list) == 2
        promoted = promoted_list[0]
        assert promoted.dtype == torch.float32
        assert tuple(promoted.shape) == (4, 2)
        assert tuple(promoted.stride()) == (2, 1)
        assert _storage_numel(promoted) == 10
        assert torch.equal(promoted, view.to(torch.float32))
        assert torch.equal(promoted_list[1], plain.to(torch.float32))
        assert torch.equal(top[0], plain.to(torch.float32))
    assert ctx.tensors[1][0] is view, "promoted tensors leaked out of the context"


def test_promote_mixed_dtypes_none_and_int_untouched():
    """int32 (not in DTYPE_PROMOTE_MAP) and None slots must pass through untouched."""
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    int_t = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    ctx = _Ctx((view, int_t, None), ("float16", "int32", "float16"))
    with _generator(ctx)._golden_mode():
        promoted_view, kept_int, kept_none = ctx.tensors
        assert promoted_view.dtype == torch.float32
        assert _storage_numel(promoted_view) == 8
        assert kept_int is int_t, "int32 tensor must keep object identity (no promote)"
        assert kept_int.dtype == torch.int32
        assert kept_none is None


def test_promote_bfloat16_strided_view():
    """bfloat16 (DTYPE_PROMOTE_MAP -> float32) strided view keeps whole storage."""
    storage = torch.arange(8, dtype=torch.bfloat16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    ctx = _Ctx((view,), ("bfloat16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert tuple(promoted.stride()) == (2, 1)
        assert _storage_numel(promoted) == 8
        assert torch.equal(promoted, view.to(torch.float32))


def test_promote_float32_to_float64_strided_view():
    """float32 -> float64 lane (element size doubles): whole storage rebuilt in the wider dtype."""
    storage = torch.arange(8, dtype=torch.float32)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    ctx = _Ctx((view,), ("float32",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float64
        assert tuple(promoted.stride()) == (2, 1)
        assert _storage_numel(promoted) == 8
        assert torch.equal(promoted, view.to(torch.float64))


def test_promote_float8_e5m2_strided_view():
    """float8_e5m2 -> float32 lane: 1-byte elements, whole-storage cast on the view."""
    storage = torch.arange(8, dtype=torch.float32).to(torch.float8_e5m2)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    ctx = _Ctx((view,), ("float8_e5m2",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert tuple(promoted.stride()) == (2, 1)
        assert _storage_numel(promoted) == 8
        assert torch.equal(promoted, view.to(torch.float32))


def test_promote_complex64_strided_view():
    """complex64 -> complex128 lane: complex storage survives the elementwise cast."""
    storage = torch.arange(8, dtype=torch.float32).to(torch.complex64)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    ctx = _Ctx((view,), ("complex64",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.complex128
        assert tuple(promoted.stride()) == (2, 1)
        assert _storage_numel(promoted) == 8
        assert torch.equal(promoted, view.to(torch.complex128))


def test_promote_transposed_view_keeps_stride():
    """Non-contiguous view whose storage == numel (transpose): stride geometry must survive."""
    storage = torch.arange(6, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (1, 3), 0)
    assert not view.is_contiguous()
    ctx = _Ctx((view,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert tuple(promoted.stride()) == (1, 3)
        assert _storage_numel(promoted) == 6
        assert torch.equal(promoted, view.to(torch.float32))


def test_promote_tail_offset_view():
    """View living at a non-zero offset (tail slice): rebuild at the same offset."""
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (4,), (1,), 4)
    ctx = _Ctx((view,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.storage_offset() == 4
        assert _storage_numel(promoted) == 8
        assert torch.equal(promoted, view.to(torch.float32))


def test_promote_empty_view_keeps_storage():
    """Empty logical view (numel=0) over a padded storage: storage must survive."""
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (0,), (1,), 0)
    ctx = _Ctx((view,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.numel() == 0
        assert _storage_numel(promoted) == 8


def test_promote_zero_dim_tensor():
    t = torch.tensor(1.5, dtype=torch.float16)
    ctx = _Ctx((t,), ("float16",))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert promoted.dtype == torch.float32
        assert promoted.dim() == 0
        assert promoted.item() == 1.5


def test_promote_scalars_numpy_alongside_tensors():
    """Scalar promotion (numpy astype path) keeps working alongside the tensor fix."""
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    scalars = (numpy.float16(1.5), numpy.float16(2.5))
    ctx = _Ctx((view,), ("float16",), scalars=scalars, scalar_dtypes=("float16", "float16"))
    with _generator(ctx)._golden_mode():
        promoted = ctx.tensors[0]
        assert _storage_numel(promoted) == 8
        s0, s1 = ctx.scalars
        assert numpy.float32(s0) == numpy.float32(1.5)
        assert s0.dtype == numpy.float32
        assert s1.dtype == numpy.float32
    assert ctx.scalars[0] is scalars[0], "scalars must be restored after the context"


def test_restore_full_backup_after_context():
    """All promoted state (tensors / dtypes / scalars) is restored on context exit."""
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    scalars = (numpy.float16(1.5),)
    ctx = _Ctx((view,), ("float16",), scalars=scalars, scalar_dtypes=("float16",))
    with _generator(ctx)._golden_mode():
        assert ctx.tensors[0].dtype == torch.float32
        assert ctx.tensor_dtypes == ("float32",)
    assert ctx.tensors[0] is view
    assert ctx.tensor_dtypes == ("float16",)
    assert ctx.scalars[0] is scalars[0]
    assert ctx.scalar_dtypes == ("float16",)


def test_golden_mode_enable_skips_promote():
    """golden_mode != Promote: the context must not touch tensors at all."""
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    ctx = _Ctx((view,), ("float16",))
    with _generator(ctx, golden_mode="Enable")._golden_mode():
        assert ctx.tensors[0] is view
        assert ctx.tensors[0].dtype == torch.float16
    assert ctx.tensors[0] is view


def test_promote_skips_pure_output_slots():
    """Pure output placeholder slots must not be promoted.

    Whole-case Promote (set when one output uses cross_check/mixed) promotes only
    input/inplace slots: hifloat8/e8m0 pure output templates keep their declared
    dtype and object identity while the fp16 input promotes to fp32.
    """
    storage = torch.arange(8, dtype=torch.float16)
    view = torch.as_strided(storage, (3, 2), (2, 1), 0)
    tpl_hif8 = torch.zeros((4,), dtype=torch.float16)
    tpl_e8m0 = torch.zeros((4,), dtype=torch.float16)
    ctx = _Ctx((view, tpl_hif8, tpl_e8m0), ("float16", "hifloat8", "float8_e8m0"))
    ctx.pure_output_indexes = (1, 2)
    with _generator(ctx)._golden_mode():
        promoted_view, kept_hif8, kept_e8m0 = ctx.tensors
        assert promoted_view.dtype == torch.float32
        assert _storage_numel(promoted_view) == 8
        assert kept_hif8 is tpl_hif8, "pure output template must keep object identity (no promote)"
        assert kept_e8m0 is tpl_e8m0, "pure output template must keep object identity (no promote)"
        assert ctx.tensor_dtypes == ("float32", "hifloat8", "float8_e8m0")
    assert ctx.tensors[1] is tpl_hif8
    assert ctx.tensors[2] is tpl_e8m0
    assert ctx.tensor_dtypes == ("float16", "hifloat8", "float8_e8m0")
