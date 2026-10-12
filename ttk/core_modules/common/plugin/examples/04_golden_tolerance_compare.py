#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Full example: golden + third_party + tolerance + compare"""

import numpy
import torch


class LayerNormFullSpec:
    """LayerNorm — demonstrates all optional attributes"""

    # -- golden — function form --
    def golden(x, gamma, beta, *, epsilon=1e-5, **kwargs):
        x_t = torch.from_numpy(x)
        g_t = torch.from_numpy(gamma)
        b_t = torch.from_numpy(beta)
        mean = x_t.mean(dim=-1, keepdim=True)
        var = x_t.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_t - mean) / torch.sqrt(var + epsilon)
        return [(x_norm * g_t + b_t).numpy()]

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(self, *, epsilon=1e-5, **kwargs):
            self.eps = epsilon

        def __call__(self, x, gamma, beta, **kwargs):
            return [torch.nn.functional.layer_norm(x, [x.shape[-1]], gamma, beta, self.eps)]

    third_party = {"torch": ThirdPartyImpl, "tf": "tf.raw_ops.LayerNorm"}

    # -- tolerance — per-dtype 精度标准（官方标准）--
    # mix_tolerance（生态算子开源精度标准）是默认，用 dtype 表内置阈值，可省略；
    # 如需覆盖阈值：{"standard": "mix_tolerance", "rtol": 0.002}。
    # cross_check（需配合 third_party）见 Precision_Comparison.md。
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
    }

    # -- compare — custom comparison (optional, default cosine_similarity) --
    def compare(*outputs, **kwargs):
        """Custom comparison: returns dict (single output) or list[dict] (multi output).
        Each dict: pass(bool, required), precision(str|float, required),
        error_info(str, optional), metrics(dict, optional), diff_indices(list, optional).
        float precision is percentage (99.98, not 0.9998)."""
        npu_out, golden_out = outputs[0], outputs[1]
        cos_sim = numpy.dot(npu_out.flatten(), golden_out.flatten()) / (
            numpy.linalg.norm(npu_out.flatten()) * numpy.linalg.norm(golden_out.flatten())
        )
        return {
            "pass": cos_sim > kwargs.get("threshold", 0.99),
            "precision": cos_sim * 100,
            "metrics": {"cosine_similarity": cos_sim},
        }


# Explicit registration: class name uses *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "layer_norm_full": "LayerNormFullSpec",
}
