#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GEIR 示例用例（cases.csv）的算子测试规范。

GEIR 与 Kernel 共用同一套 TestSpec 插件契约（按位收 numpy.ndarray、按名收
算子属性，`--plugin` 加载）：

    python3 -m ttk geir -i cases.csv --plugin examples/case_store/geir/golden.py

约定：与 Kernel 的 golden.py 相同——统一 *args/**kwargs 书写；ValueDepend
张量以输入序混在 args 中（其值由 CSV input_data_ranges 定值生成或经
attributes 同名项提取为 Const）；可选输入缺省时以 None 占位传入；输出
dtype 经 kwargs["output_dtypes"] 获取；TensorList 输入/输出为嵌套序列。
"""

import numpy

__spec__ = {
    "arg_max_v2": "ArgMaxV2TestSpec",
    "smooth_l1_loss": "SmoothL1LossTestSpec",
    "conv2d": "Conv2DTestSpec",
    "split": "SplitTestSpec",
    "foreach_div_list": "ForeachDivListTestSpec",
    "foreach_add_list_inplace": "ForeachAddListInplaceTestSpec",
    "avg_pool_update": "AvgPoolUpdateTestSpec",
}


class ArgMaxV2TestSpec:
    """输入序：数据 x 在前、dimension 张量（ValueDepend，值经 input_data_ranges
    定值生成）在后；dtype 属性决定输出整型（int32/int64）。"""

    def golden(*args, **kwargs):
        x, dimension = args[0], args[1]
        axis = int(numpy.asarray(dimension).reshape(-1)[0])
        out_dtype = str(kwargs.get("output_dtypes", ("int64",))[0])
        return [numpy.argmax(x, axis=axis).astype(out_dtype)]


class SmoothL1LossTestSpec:
    """y = 0.5·d²/sigma（d<sigma）否则 d - 0.5·sigma，d = |predict - label|；
    sigma 为 Float 属性（含非有限值：sigma=inf 时恒走二次分支，y ≡ 0）。"""

    def golden(*args, **kwargs):
        predict, label = args[0], args[1]
        sigma = float(kwargs.get("sigma", 1.0))
        d = numpy.abs(numpy.asarray(predict, dtype="float32") - numpy.asarray(label, dtype="float32"))
        loss = numpy.where(d < sigma, 0.5 * d * d / sigma, d - 0.5 * sigma)
        return [loss]


class Conv2DTestSpec:
    """输入序：x / filter / bias（可选 None）/ offset_w（可选 None，int8 量化
    场景不参与本例）；属性 strides / pads / dilations / groups / data_format。
    GE pads 语义为 [top, bottom, left, right]，经 F.pad 转换后按 NCHW 卷积。"""

    def golden(*args, **kwargs):
        x, filt = args[0], args[1]
        bias = args[2] if len(args) > 2 and args[2] is not None else None
        strides = list(kwargs.get("strides", [1, 1, 1, 1]))
        pads = list(kwargs.get("pads", [0, 0, 0, 0]))
        dilations = list(kwargs.get("dilations", [1, 1, 1, 1]))
        groups = int(kwargs.get("groups", 1))
        out_dtype = str(kwargs.get("output_dtypes", ("float16",))[0])

        import torch

        t_x = torch.from_numpy(numpy.ascontiguousarray(x)).float()
        t_w = torch.from_numpy(numpy.ascontiguousarray(filt)).float()
        t_b = torch.from_numpy(numpy.ascontiguousarray(bias)).float() if bias is not None else None
        # NCHW 4 元组 strides/dilations → torch 的 (H, W) 二元组
        stride = (strides[2], strides[3])
        dilation = (dilations[2], dilations[3])
        # GE pads [top, bottom, left, right] → F.pad 尾轴起的 (left, right, top, bottom)
        if any(pads):
            t_x = torch.nn.functional.pad(t_x, (pads[3], pads[2], pads[0], pads[1]))
        y = torch.nn.functional.conv2d(t_x, t_w, bias=t_b, stride=stride, padding=0, dilation=dilation, groups=groups)
        return [y.numpy().astype(out_dtype)]


class SplitTestSpec:
    """输入序：split_dim 张量（ValueDepend，值经 attributes 同名项提取）在前、
    数据 x 在后；num_split 为属性；输出为 TensorList（多张量列表）。"""

    def golden(*args, **kwargs):
        split_dim, x = args
        axis = int(numpy.asarray(split_dim).reshape(-1)[0])
        return list(numpy.split(x, int(kwargs["num_split"]), axis=axis))


class ForeachDivListTestSpec:
    """双 TensorList 输入（x1/x2 嵌套序列）逐元素相除，输出 TensorList。"""

    def golden(*args, **kwargs):
        return [a / b for a, b in zip(args[0], args[1])]


class ForeachAddListInplaceTestSpec:
    """输入序：x1 / x2 双 TensorList + 标量 alpha 张量（值经 input_data_ranges
    定值生成）；y_i = x1_i + alpha·x2_i，原地输出（DYNAMIC_OUTPUT 名与输入
    x1 同名，desc 逐元素继承输入）。"""

    def golden(*args, **kwargs):
        alpha = float(numpy.asarray(args[2]).reshape(-1)[0])
        return [a + alpha * b for a, b in zip(args[0], args[1])]


class AvgPoolUpdateTestSpec:
    """avgpool 反传的 update 步：y = x1 / (kh·kw)（每个池化窗口位置均分
    梯度）；x2 为 padding 掩码通道，pads=0 时窗口数恒为 kh·kw、x2 不影响
    输出（ksize 3×3/2×2 多组探针实证）。"""

    def golden(*args, **kwargs):
        ksize = list(kwargs.get("ksize", [1, 1, 1, 1]))
        y = numpy.asarray(args[0], dtype="float32") / float(ksize[1] * ksize[2])
        return [y]
