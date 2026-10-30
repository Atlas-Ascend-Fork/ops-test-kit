#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLNN golden-mode NPU: the device upload must not materialize a strided view.

The upload path used ``value.to(device)``. For a strided view over a larger
physical storage — e.g. a FRACTAL_NZ logical view whose ceil-aligned fractal
storage carries tail-block padding — ``.to(device)`` materializes the view:
the result keeps only the view's own elements and every byte beyond it is
dropped. Plugin goldens that decode NZ layouts by walking
``untyped_storage()`` then see fewer elements than the CSV
``tensor_storage_shapes`` declare and fail with ``WeightNZ storage has N
elements, expected at least M``.

This is the same class of defect the Promote path already handles through
``_promote_tensor``; the upload path never reused it, so ``--golden-mode NPU``
kept failing for operators whose inputs carry fractal tail blocks. An NZ
logical view can be *contiguous* while its storage still exceeds ``numel()``,
so the gate cannot be written in terms of ``is_contiguous()`` alone.

Cases that must distinguish the two behaviours move between devices (``cpu``
and ``meta``), which is the only way to force the copy the upload performs —
on a same-device ``.to()`` no copy happens and the defect stays invisible. The
geometry/storage assertions are device independent and therefore assert the
same guarantee the NPU target needs. No NPU is required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ttk.core_modules.aclnn.golden import GoldenGenerator, _golden_value_to_device

# An NZ logical view is 4355 elements while its ceil-aligned fractal storage
# holds 7680 — the shape quoted in the operator report.
_NZ_VIEW_SHAPE = (67, 65)
_NZ_VIEW_NUMEL = 4355
_NZ_STORAGE_NUMEL = 7680


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "NPU"
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


def _storage_numel(t):
    return t.untyped_storage().nbytes() // t.element_size()


def _nz_view(device="cpu"):
    base = torch.arange(_NZ_STORAGE_NUMEL, dtype=torch.float32).to(device)
    return torch.as_strided(base, _NZ_VIEW_SHAPE, (_NZ_VIEW_SHAPE[1], 1), 0)


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
    monkeypatch.setattr("ttk.core_modules.aclnn.golden.get_global_storage", lambda: _mock_switches())
    monkeypatch.setattr("ttk.core_modules.aclnn.golden.OpApiInfoKeeper", MagicMock())


# --------------------------------------------------------------------------- #
# Upload across devices: whole physical storage must survive
# --------------------------------------------------------------------------- #
def test_upload_cpu_to_meta_preserves_whole_storage_of_contiguous_view():
    """The reported shape: a contiguous logical view whose storage exceeds numel.

    ``.to("meta")`` forces the transfer. The result must still own the
    ceil-aligned storage, with the logical geometry untouched.
    """
    x2 = _nz_view("cpu")
    assert x2.is_contiguous(), "an NZ logical view can be contiguous; that is the trap"
    assert x2.numel() == _NZ_VIEW_NUMEL
    assert _storage_numel(x2) == _NZ_STORAGE_NUMEL

    moved = _golden_value_to_device(x2, "meta")

    assert _storage_numel(moved) == _NZ_STORAGE_NUMEL, "the upload must not drop the fractal tail blocks"
    assert moved.numel() == _NZ_VIEW_NUMEL
    assert tuple(moved.shape) == _NZ_VIEW_SHAPE
    assert tuple(moved.stride()) == (_NZ_VIEW_SHAPE[1], 1)
    assert moved.storage_offset() == 0


def test_upload_preserves_storage_of_offset_view():
    """Non-zero offset view: the rebuilt storage keeps the leading elements and
    the same offset, so a decoder still finds the padding it reads."""
    base = torch.arange(_NZ_STORAGE_NUMEL, dtype=torch.float32)
    view = torch.as_strided(base, (2, 3), (4, 1), 5)

    moved = _golden_value_to_device(view, "meta")

    assert _storage_numel(moved) == _NZ_STORAGE_NUMEL
    assert moved.storage_offset() == 5
    assert tuple(moved.stride()) == (4, 1)


def test_upload_preserves_storage_of_non_contiguous_view():
    """Transposed view (storage == numel): the stride geometry must survive."""
    base = torch.arange(6, dtype=torch.float32)
    view = torch.as_strided(base, (3, 2), (1, 3), 0)
    assert not view.is_contiguous()

    moved = _golden_value_to_device(view, "meta")

    assert tuple(moved.stride()) == (1, 3)
    assert _storage_numel(moved) == 6


def test_upload_preserves_storage_for_float8_nz_view():
    """The reported dtype (1-byte float8 element): whole storage survives."""
    base = torch.arange(_NZ_STORAGE_NUMEL, dtype=torch.float32).to(torch.float8_e4m3fn)
    view = torch.as_strided(base, _NZ_VIEW_SHAPE, (_NZ_VIEW_SHAPE[1], 1), 0)

    moved = _golden_value_to_device(view, "meta")

    assert _storage_numel(moved) == _NZ_STORAGE_NUMEL
    assert tuple(moved.stride()) == (_NZ_VIEW_SHAPE[1], 1)


def test_upload_preserves_storage_of_empty_view():
    """Empty logical view over a padded storage: the storage must survive."""
    base = torch.arange(8, dtype=torch.float32)
    view = torch.as_strided(base, (0,), (1,), 0)

    moved = _golden_value_to_device(view, "meta")

    assert moved.numel() == 0
    assert _storage_numel(moved) == 8


# --------------------------------------------------------------------------- #
# Non-regression: values, plain tensors, container recursion
# --------------------------------------------------------------------------- #
def test_upload_same_device_keeps_view_geometry():
    """A same-device upload performs no copy; the view is handed over as is."""
    view = _nz_view("cpu")

    moved = _golden_value_to_device(view, "cpu")

    assert tuple(moved.shape) == _NZ_VIEW_SHAPE
    assert tuple(moved.stride()) == (_NZ_VIEW_SHAPE[1], 1)
    assert moved.storage_offset() == 0


def test_upload_same_device_does_not_copy():
    """Same-device upload keeps ``Tensor.to`` semantics: the storage is reused."""
    x2 = _nz_view("cpu")

    moved = _golden_value_to_device(x2, "cpu")

    assert moved.untyped_storage().data_ptr() == x2.untyped_storage().data_ptr()


def test_upload_keeps_logical_values():
    """Same-device upload of a padded view keeps the values the golden reads."""
    x2 = _nz_view("cpu")

    moved = _golden_value_to_device(x2, "cpu")

    assert torch.equal(moved, x2)


def test_upload_plain_tensor_is_unchanged():
    """Ordinary tensors (storage == numel, offset 0, contiguous) behave as before."""
    t = torch.ones((2, 3), dtype=torch.float32)

    moved = _golden_value_to_device(t, "cpu")

    assert _storage_numel(moved) == 6
    assert moved.is_contiguous()
    assert tuple(moved.shape) == (2, 3)
    assert torch.equal(moved, t)


def test_upload_recurses_and_preserves_container_types():
    """Nested Sequence upload keeps container type and per-tensor storage."""
    view = _nz_view("cpu")
    np_leaf = np.ones(3, dtype="float32")
    nested = [view, (np_leaf, None, 5), "attr"]

    moved = _golden_value_to_device(nested, "meta")

    assert isinstance(moved, list)
    assert isinstance(moved[1], tuple)
    assert _storage_numel(moved[0]) == _NZ_STORAGE_NUMEL
    assert moved[1][0] is np_leaf, "non-torch values must pass through unchanged"
    assert moved[1][1] is None
    assert moved[1][2] == 5
    assert moved[2] == "attr"


# --------------------------------------------------------------------------- #
# Dispatch paths: all three upload call sites
# --------------------------------------------------------------------------- #
def _make_testcase(api_name, tensors):
    from ttk.core_modules.aclnn.testcase import TestcaseAclnn

    case = TestcaseAclnn()
    case.testcase_name = f"test_{api_name}_nz_upload"
    case.api_name = api_name
    case.tensors = tensors
    case.attributes = {}
    case.output_tensor_indexes = (0,)
    case._pure_output_indexes = ()
    case.manual_golden_binaries = None
    return case


def test_call_torch_api_upload_keeps_nz_storage():
    """``_call_torch_api``: the golden receives the tensor with its whole storage."""
    case = _make_testcase("aclnnRelu", [_nz_view("cpu")])
    gen = GoldenGenerator(case)

    received = {}

    def _native_gf(x):
        received["storage"] = _storage_numel(x)
        return [torch.ones((2,), dtype=torch.float32)]

    out = gen._call_torch_api(_native_gf, device="meta")

    assert received["storage"] == _NZ_STORAGE_NUMEL, "the golden must receive the full physical storage"
    assert isinstance(out[0], np.ndarray)


def test_call_custom_golden_upload_keeps_nz_storage(monkeypatch):
    """Same guarantee on the custom-golden path (the reported plugin path)."""
    from ttk.core_modules.aclnn.testcase import TestcaseAclnn

    case = _make_testcase("aclnnCustom", [_nz_view("cpu")])
    monkeypatch.setattr(TestcaseAclnn, "is_torch_dtype_support", lambda self: True)

    received = {}

    def _golden(self_t, alpha=None, device=None, **kwargs):
        received["storage"] = _storage_numel(self_t)
        return [torch.ones((2,), dtype=torch.float32)]

    fake_plan = SimpleNamespace(build_args=lambda tensors, scalars, attributes: ((tensors[0],), {}))
    case._param_plan_cache = fake_plan

    gen = GoldenGenerator(case)
    gen._call_custom_golden(_golden, device="meta")

    assert received["storage"] == _NZ_STORAGE_NUMEL, "custom goldens decode NZ from the uploaded storage"


def test_class_golden_upload_keeps_nz_storage(monkeypatch):
    """Class-form golden: ``_invoke_class`` uploads the name→value pool."""
    from collections import OrderedDict

    from ttk.core_modules.aclnn.info_keeper import OpApiInfo
    from ttk.core_modules.aclnn.testcase import TestcaseAclnn

    case = _make_testcase("aclnnClassGolden", [_nz_view("cpu")])
    monkeypatch.setattr(TestcaseAclnn, "is_torch_dtype_support", lambda self: True)
    params = OrderedDict()
    params["x"] = {"type": "aclTensor*"}
    monkeypatch.setattr(
        "ttk.core_modules.aclnn.golden.OpApiInfoKeeper",
        lambda: SimpleNamespace(info_of=lambda api: OpApiInfo(params=params)),
    )

    received = {}

    class _Golden:
        def __call__(self, x):  # noqa: N803
            received["storage"] = _storage_numel(x)
            return [torch.ones((2,), dtype=torch.float32)]

    gen = GoldenGenerator(case)
    gen._invoke_class(_Golden, device="meta")

    assert received["storage"] == _NZ_STORAGE_NUMEL, "class goldens must decode NZ from the full storage"


# --------------------------------------------------------------------------- #
# The reported failure shape
# --------------------------------------------------------------------------- #
def test_nz_upload_survives_decoder_storage_walk():
    """A decoder that walks ``untyped_storage()`` and refuses when it finds
    fewer elements than the declared storage shape (``weight_nz_to_nd``-style)
    reads the fractal tail blocks past the logical view; the upload must leave
    them in place."""
    view = _nz_view("cpu")

    moved = _golden_value_to_device(view, "meta")

    def _decode_like_weight_nz_to_nd(t):
        found = t.untyped_storage().nbytes() // t.element_size()
        if found < _NZ_STORAGE_NUMEL:
            raise ValueError(f"WeightNZ storage has {found} elements, expected at least {_NZ_STORAGE_NUMEL}")
        return found

    assert _decode_like_weight_nz_to_nd(moved) == _NZ_STORAGE_NUMEL
