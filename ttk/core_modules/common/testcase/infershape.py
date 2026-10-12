#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Shape inference methods
"""
Development Tips for shape inference

Inside a method, a shape can be a list.
BUT DO NOT pass a shape as list between public methods!!! Always treat it as a tuple!!!
"""

__all__ = ["shape_inference"]


# Standard Packages
import math
from typing import Tuple, Union

from ....utilities import is_shape, promote_scalar_shapes


def _produce_by_broadcast_rule(*input_shapes) -> tuple:
    """
    Produce an output shape by using input_shapes, this complies to broadcast rule
    """

    def __get_axis(shape, minus_dim) -> int:
        return shape[len(shape) + minus_dim] if len(shape) + minus_dim >= 0 else 1

    maximum_shape = []
    rank = max([len(shape) for shape in input_shapes])
    for dim in range(-rank, 0):
        max_axis = max([__get_axis(shape, dim) for shape in input_shapes])
        maximum_shape.append(max_axis)
    return tuple(maximum_shape)


def _elewise_check_input_parameters(args, output_num, relations):
    # Perform param check
    if len(args) < 1:
        # Number of input shapes must be higher than 1
        raise ValueError("Cannot perform shape inference mode ELEWISE if there is no input shapes")
    if output_num < 1:
        # Number of output shapes must be higher than 1
        raise ValueError("Cannot perform shape inference mode ELEWISE if there is no output shapes")
    # Only shape<int> can be passed as args
    for arg in args:
        if not is_shape(arg):
            raise TypeError(f"Elewise infershape function received invalid shapes: {str(args)}")
    # If relations are specified, it must be a tuple and its length must be the same as output_num
    # Also, related relations must use in-range int or NoneType as an index to indicate the
    # related input shape
    if relations is not None:
        if not isinstance(relations, tuple):
            raise TypeError(
                f"Elewise infershape function received {str(type(relations))} instead of tuple as shape relationships"
            )
        for relation in relations:
            if not isinstance(relation, (tuple, type(None))):
                raise TypeError(
                    f"Elewise infershape function received {str(type(relation))} instead of tuple or NoneType as shape relationship"
                )
            if isinstance(relation, tuple):
                if not len(relation) > 0:
                    raise ValueError(
                        "Cannot perform shape inference mode ELEWISE if output shape isn't related with any input shape"
                    )
                for related_shape_idx in relation:
                    if not isinstance(related_shape_idx, int):
                        raise TypeError(
                            f"Elewise infershape function received {str(type(related_shape_idx))} instead of "
                            "int as related input shape index"
                        )
                    if related_shape_idx >= len(args):
                        raise ValueError(
                            "Cannot perform shape inference mode ELEWISE if "
                            "output shape is related with input shape index "
                            f"that is out of range {str(len(args))}"
                        )


def elewise(*args, output_num=1, relations=None) -> Tuple[Tuple[int], ...]:
    """
    Produce output shapes by input shapes, this method complies to the broadcast rule
    """
    _elewise_check_input_parameters(args, output_num, relations)
    # Perform broadcast rules
    output_shapes = [(-10000,)] * output_num
    for output_idx in range(output_num):
        if relations is None:
            output_shapes[output_idx] = _produce_by_broadcast_rule(*args)
            continue
        related_shapes = tuple(
            args[shape_idx]
            for shape_idx in range(len(args))
            if relations[output_idx] is None or shape_idx in relations[output_idx]
        )
        output_shapes[output_idx] = _produce_by_broadcast_rule(*related_shapes)
    return tuple(output_shapes)


def reduce(input_tensor: tuple, axes) -> tuple:
    """
    Produce output shapes by input shapes, this method complies to the reduce rule
    """
    if not is_shape(input_tensor):
        raise RuntimeError(f"{str(input_tensor)} is not a shapelike object")
    mutable_shape = list(input_tensor)
    # For dynamic reduce axes
    if axes is None:
        return tuple(-1 if ori_axis != 1 else 1 for ori_axis in mutable_shape)
    for axis in axes:
        if axis >= len(mutable_shape) or (axis < 0 and -axis > len(mutable_shape)):
            continue
        mutable_shape[axis] = 1
    return tuple(mutable_shape)


def range_inference(shape: tuple) -> Union[tuple, type(None)]:
    """
    Produce range by input shape, this method complies to simple range inference rule
    """
    if shape is None:
        return None
    if not is_shape(shape):
        raise TypeError(f"Range inference function received invalid shape: {str(shape)}")
    _range = []
    for dim in shape:
        if dim in [-1, -2]:
            _range.append((1, None))
        else:
            _range.append((dim, dim))
    return tuple(_range)


def shape_inference(shapes: tuple, args: tuple, mode: str) -> tuple:
    """
    Produce output shape by input shapes, this method distribute arguments to their shape inference function
    :param shapes:
    :param args:
    :param mode:
    :return:
    """
    shapes = promote_scalar_shapes(shapes)
    if mode == "ELEWISE":
        return elewise(*shapes, output_num=args[0], relations=args[1])
    if mode == "REDUCE":
        reduce_results = []
        for shape in elewise(*shapes, output_num=args[1], relations=args[2]):
            result_shape = reduce(shape, args[0])
            reduce_results.append(result_shape)
        return tuple(reduce_results)
    if mode == "RANGE":
        return tuple(range_inference(shape) for shape in shapes)
    raise RuntimeError(f"UNKNOWN MODE {mode}")


def transform(shape, cur_format, target_format):
    """
    Produce target format shape with original shape
    :param shape:
    :param cur_format:
    :param target_format:
    :return:
    """
    if cur_format in FORMAT_TRANSFORMS:
        if target_format in FORMAT_TRANSFORMS[cur_format]:
            return FORMAT_TRANSFORMS[cur_format][target_format](shape)
        raise NotImplementedError(f"Cannot transform format {cur_format} to {target_format}")
    raise NotImplementedError(f"Cannot transform format {cur_format} to any format including {target_format}")


def _25Dinfer(before_c, c, after_c):
    c1 = math.ceil(c / 16)
    return (*before_c, c1, *after_c, 16)


def _NHWC25Dinfer(shape: tuple):
    if not len(shape) == 4:
        raise RuntimeError(f"4D to 5D transformation only supports 4 dimension input, but received {shape}")
    n, h, w, c = shape
    return _25Dinfer((n,), c, (h, w))


def _NDHWC25Dinfer(shape: tuple):
    if len(shape) != 5:
        raise RuntimeError(f"NDHWC to 6D transformation only supports 5 dimension input, but received {shape}")
    n, d, h, w, c = shape
    return _25Dinfer((n, d), c, (h, w))


def _NCHW25Dinfer(shape: tuple):
    if not len(shape) == 4:
        raise RuntimeError(f"4D to 5D transformation only supports 4 dimension input, but received {shape}")
    n, c, h, w = shape
    return _25Dinfer((n,), c, (h, w))


def _NWC25Dinfer(shape: tuple):
    if not len(shape) == 3:
        raise RuntimeError(f"3D to 5D transformation only supports 3 dimension input, but received {shape}")
    n, w, c = shape
    return _25Dinfer((n,), c, (1, w))


def _NZinfer(shape: tuple):
    if len(shape) < 2:
        shape = (2 - len(shape)) * (1,) + shape
    a, b = shape[-2:]
    return shape[:-2] + (math.ceil(b / 16), math.ceil(a / 16), 16, 16)


FORMAT_TRANSFORMS = {
    "NHWC": {"NC1HWC0": _NHWC25Dinfer, "FRACTAL_NZ": _NZinfer},
    "NCHW": {"NC1HWC0": _NCHW25Dinfer, "FRACTAL_NZ": _NZinfer},
    "NWC": {"NC1HWC0": _NWC25Dinfer, "FRACTAL_NZ": _NZinfer},
    "NDHWC": {"NDC1HWC0": _NDHWC25Dinfer, "FRACTAL_NZ": _NZinfer},
}
