#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel 示例用例（cases.csv）的算子测试规范。

覆盖三个无内置 golden 的算子——numpy 同名属性回退的参数序与算子输入序
对不上（如 split 的 split_dim 张量在前、数据在后），须显式编写。加载：

    python3 -m ttk kernel -i cases.csv --plugin examples/case_store/kernel/golden.py

约定：kernel 通路 golden 按位收 numpy.ndarray（输入序 = 算子定义）、
按名收算子属性；统一 *args/**kwargs 书写。与输入同名的属性（ValueDepend
张量的取值项）不会出现在 kwargs 中——其值以输入张量为准（内容由 CSV
attributes 同名项填充）。
"""

import numpy

__spec__ = {
    "concat_d": "ConcatDTestSpec",
    "split": "SplitTestSpec",
    "reduce_min": "ReduceMinTestSpec",
    "non_zero": "NonZeroTestSpec",
    "add": "AddTestSpec",
    "pad": "PadTestSpec",
    "ascend_quant_v2": "AscendQuantV2TestSpec",
}


class ConcatDTestSpec:
    """x 为 TensorList 动态输入（嵌套序列），沿 concat_dim 拼接为单输出。"""

    def golden(*args, **kwargs):
        return [numpy.concatenate(list(args[0]), axis=int(kwargs["concat_dim"]))]


class SplitTestSpec:
    """输入序：split_dim 张量（ValueDepend）在前、数据 x 在后；num_split 为属性。"""

    def golden(*args, **kwargs):
        split_dim, x = args
        axis = int(numpy.asarray(split_dim).reshape(-1)[0])
        return list(numpy.split(x, int(kwargs["num_split"]), axis=axis))


class ReduceMinTestSpec:
    """输入序：数据 x 在前、axes 张量（ValueDepend）在后；keep_dims 为属性。"""

    def golden(*args, **kwargs):
        x, axes = args
        keep = bool(kwargs.get("keep_dims", kwargs.get("keepdims", False)))
        axis = tuple(int(a) for a in numpy.asarray(axes).reshape(-1)) or None
        return [numpy.min(x, axis=axis, keepdims=keep)]


class NonZeroTestSpec:
    """输出 shape 由输入取值决定（output_shape_unknown_indexes），golden
    现算 shape 供输出 buffer 分配：transpose=True → (ndim, N)，否则 (N, ndim)。"""

    def golden(*args, **kwargs):
        indices = numpy.asarray(numpy.nonzero(args[0]))
        if not bool(kwargs.get("transpose", False)):
            indices = indices.T
        return [indices.astype("int32")]


class AddTestSpec:
    """无 golden 声明（走 numpy.add 回退）——仅演示 Spec.tolerance 判据声明：
    float32 输出走统计相对误差（stat_rel_err）。"""

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
    }


class PadTestSpec:
    """零填充（Pad 无 constant_values 属性）；paddings 为 ValueDepend 张量输入，
    与输入同名属性被剔除，取值以输入张量为准（shape (rank, 2) int64）。"""

    def golden(*args, **kwargs):
        x, paddings = args[0], args[1]
        pads = tuple(tuple(int(v) for v in row) for row in numpy.asarray(paddings).reshape(-1, 2))
        return [numpy.pad(x, pads, mode="constant")]


class AscendQuantV2TestSpec:
    """量化 y = round(x * scale + offset) 饱和到 dst_type；scale 定值生成。
    int8 输出默认按 dtype 轴路由 binary_equal（如需 |a-g|<=1 的 quant 判据，
    须在 TestSpec tolerance 声明 standard='quant'，仅 int4/int8/uint8 输出可用）；
    float8 输出走 mix 可过，收紧用 CLI `--compare requant`（仅 hifloat8 自动
    路由；requant 为 CLI 专属判据，Spec.tolerance 不收）。"""


    def golden(*args, **kwargs):
        import ml_dtypes

        x, scale = args[0], args[1]
        out_dtype = str(kwargs.get("output_dtypes", ("int8",))[0])
        y = numpy.rint(x.astype("float32") * scale.astype("float32"))
        if out_dtype == "int8":
            return [numpy.clip(y, -128, 127).astype("int8")]
        if out_dtype == "float8_e5m2":
            # fp8 是浮点目标：不走整数 rint，fp16 算术后单次 cast（与 CANN 一致）
            return [(x * scale).astype(ml_dtypes.float8_e5m2)]
        if out_dtype == "int4":
            return [numpy.clip(y, -8, 7).astype("int8")]
        return [y.astype(out_dtype)]
