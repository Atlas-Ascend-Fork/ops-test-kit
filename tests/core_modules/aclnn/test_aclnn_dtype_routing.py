# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_acl_overrides 与 Phase1ParamBuilder 装配链路的 dtype 路由。"""

import pytest
import torch

from ttk.core_modules.aclnn.execute import Phase1ParamBuilder, _acl_overrides


@pytest.mark.parametrize(
    ("declared", "expected_dtype"),
    [
        ("int4", 29),
        ("s4", 29),
        ("float4_e2m1", 40),
        ("float4_e1m2", 41),
        ("hifloat4", 42),
        ("fp4_e2m1", 40),
        ("f4_e1m2", 41),
    ],
)
def test_packed_4bit_overrides_view_and_dtype(declared, expected_dtype):
    view, acl_dtype = _acl_overrides(declared, (8, 4))
    assert view == (8, 4)
    assert acl_dtype == expected_dtype


@pytest.mark.parametrize(
    "declared",
    ["torch.float4_e2m1", "torch.float4_e1m2", "torch.hifloat4", "torch.hifloat4_scale", "torch.int4"],
)
def test_noncanonical_names_have_no_override(declared):
    assert _acl_overrides(declared, (8, 4)) == (None, None)


def test_hifloat4_scale_overrides_acl_dtype_only():
    view, acl_dtype = _acl_overrides("hifloat4_scale", (12, 1, 32, 2, 16))
    assert view is None
    assert acl_dtype == 43


@pytest.mark.parametrize("declared", ["float32", "float8_e8m0", "bfloat16", None])
def test_plain_dtype_has_no_override(declared):
    assert _acl_overrides(declared, (3, 4)) == (None, None)


def test_builder_passes_overrides_to_acl_tensor():
    class _RecordingDevice:
        def __init__(self):
            self.calls = []

        def create_acl_tensor(self, tt, fmt, ss, view_shape_override=None, acl_dtype_override=None):
            self.calls.append(
                {
                    "dtype": str(tt.dtype).split(".")[-1],
                    "view_override": tuple(view_shape_override) if view_shape_override is not None else None,
                    "acl_dtype": acl_dtype_override,
                }
            )
            return len(self.calls) - 1

        def create_acl_tensor_list(self, maybe_lst):
            return maybe_lst

    class _FakeCtx:
        def __init__(self, dtypes, tensors):
            self.flat_tensors = tensors
            self.flat_tensor_dtypes = dtypes
            self.flat_tensor_view_shapes = [tuple(t.shape) for t in tensors]
            self.flat_tensor_formats = ["ND"] * len(tensors)
            self.dyn_input_slot_indexes = []
            self.tensor_list_dist = []

        def flat_storage_shape(self, idx):
            return tuple(self.flat_tensors[idx].shape)

    ctx = _FakeCtx(
        ["hifloat4_scale", "float4_e2m1", "float32"],
        [
            torch.zeros((12, 1, 32, 2, 16), dtype=torch.float32),
            torch.zeros((8, 4), dtype=torch.uint8),
            torch.zeros((3, 4), dtype=torch.float32),
        ],
    )
    dvc = _RecordingDevice()
    Phase1ParamBuilder(ctx, dvc)._create_acl_tensor()

    assert dvc.calls[0] == {"dtype": "float32", "view_override": None, "acl_dtype": 43}
    assert dvc.calls[1] == {"dtype": "uint8", "view_override": (8, 4), "acl_dtype": 40}
    assert dvc.calls[2] == {"dtype": "float32", "view_override": None, "acl_dtype": None}
