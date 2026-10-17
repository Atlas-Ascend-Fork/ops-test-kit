#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch CPU golden and input plugin for DynamicQuantUpdateScatterV2."""

import numpy as np
import torch
from ml_dtypes import int4 as _int4  # noqa: F401 - registers int4 dtype for TTK

__spec__ = {
    "dynamic_quant_update_scatter_v2": "DynamicQuantUpdateScatterV2KernelSpec",
    "DynamicQuantUpdateScatterV2": "DynamicQuantUpdateScatterV2KernelSpec",
}

__golden__ = {"kernel": {"dynamic_quant_update_scatter_v2": "__golden_dynamic_quant_update_scatter_v2"}}
__input__ = {"kernel": {"dynamic_quant_update_scatter_v2": "__input_dynamic_quant_update_scatter_v2"}}

INT4_SCALE_RANGE = 15.0
INT4_QUANT_MAX = 7.0
QUANT_EPSILON = 1.0e-12

_KERNEL_TOLERANCE = {
    "int4": {"standard": "quant"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _wrap_to_int4(q):
    return ((q + 8) & 15) - 8


def _pack_int4(values):
    vals = np.asarray(values).astype(np.int8).reshape(-1)
    if vals.size % 2:
        vals = np.pad(vals, (0, 1), mode="constant")
    pairs = vals.reshape(-1, 2).view(np.uint8) & np.uint8(0x0F)
    return (pairs[:, 0] | (pairs[:, 1] << np.uint8(4))).astype(np.uint8, copy=False)


def _unpack_int4(packed, size):
    raw = np.asarray(packed, dtype=np.uint8).reshape(-1)
    unpacked = np.empty(raw.size * 2, dtype=np.uint8)
    unpacked[0::2] = raw & np.uint8(0x0F)
    unpacked[1::2] = (raw >> np.uint8(4)) & np.uint8(0x0F)
    return unpacked[:size].view(_int4)


def _to_numpy_for_golden(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _to_supported_torch_tensor(x, device):
    arr = np.asarray(x)
    if "int4" in str(arr.dtype) or arr.dtype.kind == "V":
        arr = arr.astype(np.int8)
    return torch.as_tensor(arr, device=device)


def __golden_dynamic_quant_update_scatter_v2(x, indices, var, var_scale, var_offset, **kwargs):
    x_t = torch.as_tensor(np.asarray(x).astype(np.float32), dtype=torch.float32)
    hidden = x_t.shape[-1]
    rows = x_t.reshape(-1, hidden)
    batch = rows.shape[0]

    out_var = np.array(var, copy=True)
    out_scale = np.array(var_scale, dtype=np.float32, copy=True)
    out_offset = np.array(var_offset, dtype=np.float32, copy=True)
    seq_len = out_var.shape[1] if out_var.ndim >= 2 else 0
    var_bytes = _pack_int4(out_var)
    scale_flat = out_scale.reshape(-1)
    offset_flat = out_offset.reshape(-1)
    indices_t = torch.as_tensor(np.asarray(indices), dtype=torch.int64).reshape(-1)

    scale_range = torch.tensor(INT4_SCALE_RANGE, dtype=torch.float32)
    quant_max = torch.tensor(INT4_QUANT_MAX, dtype=torch.float32)
    quant_eps = torch.tensor(QUANT_EPSILON, dtype=torch.float32)
    for b in range(batch):
        valid_idx = int(indices_t[b].item())
        if valid_idx < 0 or valid_idx >= seq_len:
            continue

        row = rows[b]
        max_val = torch.max(row)
        min_val = torch.min(row)
        scale = torch.maximum((max_val - min_val) / scale_range, quant_eps)
        offset_quant = quant_max - max_val / scale
        back_scale = torch.tensor(1.0, dtype=torch.float32) / scale
        quantized = torch.round(row * back_scale + offset_quant)
        quantized_i64 = _wrap_to_int4(quantized.to(torch.int64))

        dst = b * seq_len + valid_idx
        packed = _pack_int4(quantized_i64.numpy())
        byte_base = dst * hidden // 2
        byte_end = min(byte_base + packed.size, var_bytes.size)
        if 0 <= byte_base < byte_end:
            var_bytes[byte_base:byte_end] = packed[: byte_end - byte_base]
        scale_flat[dst] = np.float32(scale.item())
        offset_flat[dst] = np.float32(-offset_quant.item())

    out_var = _unpack_int4(var_bytes, out_var.size).reshape(out_var.shape)
    return [
        out_var,
        out_scale,
        out_offset,
    ]


def __input_dynamic_quant_update_scatter_v2(x, indices, var, var_scale, var_offset, **kwargs):
    if "half_step" in kwargs.get("testcase_name", ""):
        x_arr = np.asarray(x)
        hidden = x_arr.shape[-1]
        pattern = np.empty(hidden, dtype=np.float32)
        pattern[0] = 0.0
        pattern[1] = 15.0
        if hidden > 2:
            pattern[2:] = np.arange(hidden - 2, dtype=np.float32) % 15.0 + 0.5
        x = np.broadcast_to(pattern, x_arr.shape).astype(x_arr.dtype, copy=True)
    return [x, indices, var, var_scale, var_offset]


_dynamic_quant_update_scatter_v2_spec_golden = __golden_dynamic_quant_update_scatter_v2
_dynamic_quant_update_scatter_v2_spec_input = __input_dynamic_quant_update_scatter_v2


class _DynamicQuantUpdateScatterV2Compose:
    """Third-party reference executed on the remote GPU server.

    Calls the same golden implementation; non-float outputs (int4 var) are
    returned as None — only float cross_check outputs consult third_party,
    and None keeps output-slot alignment.
    """

    def __call__(self, x, indices, var, var_scale, var_offset, **kwargs):
        outputs = _dynamic_quant_update_scatter_v2_spec_golden(
            _to_numpy_for_golden(x),
            _to_numpy_for_golden(indices),
            _to_numpy_for_golden(var),
            _to_numpy_for_golden(var_scale),
            _to_numpy_for_golden(var_offset),
            **kwargs,
        )
        device = x.device if torch.is_tensor(x) else "cpu"
        return [
            None,
            _to_supported_torch_tensor(outputs[1], device),
            _to_supported_torch_tensor(outputs[2], device),
        ]


class DynamicQuantUpdateScatterV2KernelSpec:
    golden = _dynamic_quant_update_scatter_v2_spec_golden
    input = _dynamic_quant_update_scatter_v2_spec_input
    third_party = {"torch": _DynamicQuantUpdateScatterV2Compose}
    tolerance = _KERNEL_TOLERANCE


# 【不存在】aclnn 通路: CMakeLists.txt 使用 ACLNNTYPE aclnn_exclude.
# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 DynamicQuantUpdateScatterV2.
