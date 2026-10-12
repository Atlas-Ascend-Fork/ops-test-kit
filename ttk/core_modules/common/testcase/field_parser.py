#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Parser for each csv field

字段值统一经 ``_eval`` 求值：语法白名单（AST 校验）+ 受限命名空间双层防线——
用例 CSV 来自算子仓库（含第三方贡献），不允许任意代码执行。白名单内为字面量
容器/一元二元算术/白名单函数调用（float/int/bool/str/tuple/list），属性访问、
下标、推导式等一律拒绝；正则改写产生的 ``float("inf")``/``-float(0)`` 仍可用。
"""

__all__ = [
    "process_bool",
    "process_string",
    "process_dynamic_shapelike",
    "process_dynamic_inferable_shapelike",
    "shapelike_stc",
    "shapelike_stc_ex",
    "shapelike_float",
    "shapelike_float_signed",
    "shape_stride",
    "shapelike_stc_nested",
    "shapelike_stc_ex_nested",
    "shapelike_float_signed_nested",
    "shapelike_float_nested",
    "shapelike_dyn_nested",
    "string_container",
    "scalar_nested",
    "int_container",
    "process_eval",
    "process_int",
    "process_dict",
]


# Standard Packages
import ast
import logging
import re
from typing import Optional, Union

# Third-Party Packages
from ....utilities import is_shape

_EVAL_NAMESPACE = {
    "__builtins__": {},
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "tuple": tuple,
    "list": list,
}

_ALLOWED_NAMES = frozenset(_EVAL_NAMESPACE) - {"__builtins__"}

_BANNED_NODES = (
    ast.Attribute,
    ast.Subscript,
    ast.Starred,
    ast.Lambda,
    ast.IfExp,
    ast.NamedExpr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.JoinedStr,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


def _validate_expression(tree):
    """AST 白名单：只允许字面量容器、算术与白名单函数调用，杜绝逃逸语法。"""
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            raise ValueError(f"disallowed syntax in field expression: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("only whitelisted function calls (float/int/bool/str/tuple/list) are allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed in field expressions")
        elif isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"name {node.id!r} is not allowed in field expressions")


def _eval(value: str):
    """eval guarded by an AST whitelist + restricted namespace (see module docstring)."""
    tree = ast.parse(value, mode="eval")
    _validate_expression(tree)
    return eval(value, _EVAL_NAMESPACE)  # noqa: S307 双层防线求值（见模块 docstring）


def _is_inference(value: str) -> bool:
    # Check if inference is needed
    if value in ("ELEWISE", "REDUCE"):
        return True
    # Check if limited inference is needed
    if value.startswith("ELEWISE") and len(value) > 7:
        try:
            parsed = ast.literal_eval(value[7:])
        except (ValueError, SyntaxError, MemoryError) as err:
            raise ValueError(f"{value} is not a valid limited inference value!") from err
        if isinstance(parsed, (tuple, list)):
            return True
        raise ValueError(f"{value} is not a valid limited inference value!")
    if value.startswith("REDUCE") and len(value) > 6:
        try:
            parsed = ast.literal_eval(value[6:])
        except (ValueError, SyntaxError, MemoryError) as err:
            raise ValueError(f"{value} is not a valid limited inference value!") from err
        if isinstance(parsed, (tuple, list)):
            return True
        raise ValueError(f"{value} is not a valid limited inference value!")
    return False


def process_bool(value: str) -> bool:
    value = value.upper()
    if value == "TRUE":
        value = "True"
    elif value == "FALSE":
        value = "False"
    try:
        parsed = _eval(value)
    except Exception:
        raise TypeError(f"{value} is not a valid boolean value!") from None
    return bool(parsed)


def process_string(value) -> str:
    # noinspection PyBroadException
    try:
        result = _eval(value)
    except Exception:
        return value
    else:
        if isinstance(result, type(None)):
            return result
        return value


def _shapelike(
    value: str,
    allow_inference=False,
    positive_only=False,
    allow_float=False,
    allow_sub_value_none=False,
    allow_none=False,
    allow_empty_tensor=False,
    allow_inf=False,
) -> Union[tuple, str, type(None)]:
    if allow_float:
        # convert -0.0 or -0 to -float(0) for powf & atan operator
        value = value.replace("'", "").replace('"', "")
        value = re.sub(r"(-0+\.0*|-0+\.?)(?!(\d|\.))", r"-float(0)", value)
        if allow_inf:
            # convert inf to float(inf)
            value = re.sub(
                r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))', lambda match: f'float("{match.group(1)}")', value
            )
    if allow_inference and _is_inference(value):
        return value
    parsed = _eval(value)
    # No inference needed, treat as normal
    if parsed is None and allow_none:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError(f"{value} is not a valid shapelike value!")
    # Make sure all shapelike container is a tuple
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed
    # Iterate through all sub_values, set allowed_type
    allowed_type = (int,)
    if allow_float:
        allowed_type += (float,)
    if allow_sub_value_none:
        allowed_type += (type(None),)
    if all(isinstance(sub_value, allowed_type) for sub_value in parsed):
        # sub_value is single value, convert to tuple and return
        if positive_only and not (
            all(i > 0 for i in parsed if i is not None)
            or (all(i >= 0 for i in parsed if i is not None) if allow_empty_tensor else False)
        ):
            raise ValueError(
                f"shapelike value should not have "
                f"{'non-positive' if not allow_empty_tensor else 'negative'} dim {value}"
            )
        return (parsed,)
    for sub_value in parsed:
        if isinstance(sub_value, (tuple, list)):
            if not is_shape(sub_value, allowed_type):
                raise TypeError(f"{str(sub_value)} is not a valid shape in type {str(allowed_type)}")
            if not all(isinstance(i, allowed_type) for i in sub_value):
                raise ValueError(f"shapelike value should not have invalid dim {value}")
            if positive_only and not (
                all(i > 0 for i in sub_value if i is not None)
                or (all(i >= 0 for i in sub_value if i is not None) if allow_empty_tensor else False)
            ):
                raise ValueError(
                    f"shapelike value should not have "
                    f"{'non-positive' if not allow_empty_tensor else 'negative'} dim {value}"
                )
        elif isinstance(sub_value, type(None)):
            pass
        else:
            raise TypeError(f"{str(sub_value)} of {value} is not a valid shapelike value for its corresponding field")
    new_parsed = tuple(tuple(element) if element is not None else None for element in parsed)
    return new_parsed


def process_dynamic_shapelike(value: str) -> tuple:
    """
    For shapelike (1, 7, -1)
    :param value:
    :return:
    """
    return _shapelike(value, allow_none=True)


def process_dynamic_inferable_shapelike(value: str):
    """
    For shapelike REDUCE or (1, 7, -1)
    :param value:
    :return:
    """
    return _shapelike(value, allow_inference=True, allow_sub_value_none=True, allow_none=True)


def shapelike_stc(value: str):
    """
    For shapelike (34, 16, 16)
    :param value:
    :return:
    """
    return _shapelike(value, positive_only=True, allow_empty_tensor=True)


def shapelike_stc_ex(value: str):
    """
    For shapelike ELEWISE or (34, 16, 16)
    :param value:
    :return:
    """
    return _shapelike(
        value, allow_inference=True, positive_only=True, allow_sub_value_none=True, allow_empty_tensor=True
    )


def shapelike_float(value: str):
    """
    For shapelike (1.1001, 3.263)
    :param value:
    :return:
    """
    return _shapelike(value, positive_only=True, allow_float=True, allow_none=True, allow_empty_tensor=True)


def shapelike_float_signed(value: str):
    """
    For shapelike (None, -1.129) or (1.236, None)
    :param value:
    :return:
    """
    return _shapelike(value, allow_float=True, allow_sub_value_none=True, allow_inf=True)


def shape_stride(value: str):
    return _shapelike(value, positive_only=True, allow_none=True, allow_empty_tensor=True)


def shapelike_stc_nested(value: str):
    """Parse shape with TensorList nesting support.

    Supports 3-level nesting where:
      - ((3,3),(3,5))         → flat: two single tensors
      - (((3,3),(3,2)),(3,5)) → nested: TensorList of 2 + single tensor
      - None elements are allowed

    Returns the nested structure as-is for later flattening.
    """
    parsed = _eval(value)
    if not isinstance(parsed, (tuple, list)):
        raise TypeError(f"{value} is not a valid shapelike value!")
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    if all(isinstance(e, int) for e in parsed):
        if not all(d >= 0 for d in parsed):
            raise ValueError(f"shapelike value should not have negative dim {str(parsed)}")
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_shapes = []
                for sub in element:
                    if sub is None:
                        sub_shapes.append(None)
                    elif isinstance(sub, (tuple, list)):
                        if not is_shape(sub, (int,)):
                            raise TypeError(f"{str(sub)} is not a valid shape in {value}")
                        if not all(d >= 0 for d in sub):
                            raise ValueError(f"shapelike value should not have negative dim {str(sub)}")
                        sub_shapes.append(tuple(sub))
                    else:
                        raise TypeError(f"{str(sub)} is not a valid shape in {value}")
                result.append(tuple(sub_shapes))
            else:
                if not is_shape(element, (int,)):
                    raise TypeError(f"{str(element)} is not a valid shape in {value}")
                if not all(d >= 0 for d in element):
                    raise ValueError(f"shapelike value should not have negative dim {str(element)}")
                result.append(tuple(element))
        else:
            raise TypeError(f"{str(element)} of {value} is not a valid shapelike value")
    return tuple(result)


def shapelike_stc_ex_nested(value: str):
    """shapelike_stc_nested + string inference support (ELEWISE/REDUCE)."""
    if _is_inference(value):
        return value
    return shapelike_stc_nested(value)


def shapelike_dyn_nested(value: str):
    """Parse dynamic shapelike with TensorList nesting support.

    Same structure as shapelike_stc_nested but allows -1 (unknown dim) and -2
    (unknown rank).  e.g. ((-1,-1,-1),) or ((-2,),)
    """
    if value is None or (isinstance(value, str) and value.strip() in ("", "None")):
        return None
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (tuple, list)):
        raise TypeError(f"{value} is not a valid shapelike value!")
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    def _check_dyn(sub, raw):
        for d in sub:
            if d < 0 and d not in (-1, -2):
                raise ValueError(f"dynamic shape dim {d} invalid in {raw} (only -1/-2 allowed)")

    if all(isinstance(e, int) for e in parsed):
        _check_dyn(parsed, value)
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_shapes = []
                for sub in element:
                    if sub is None:
                        sub_shapes.append(None)
                    elif isinstance(sub, (tuple, list)):
                        if not is_shape(sub, (int,)):
                            raise TypeError(f"{str(sub)} is not a valid shape in {value}")
                        _check_dyn(sub, value)
                        sub_shapes.append(tuple(sub))
                    else:
                        raise TypeError(f"{str(sub)} is not a valid shape in {value}")
                result.append(tuple(sub_shapes))
            else:
                if not is_shape(element, (int,)):
                    raise TypeError(f"{str(element)} is not a valid shape in {value}")
                _check_dyn(element, value)
                result.append(tuple(element))
        else:
            raise TypeError(f"{str(element)} of {value} is not a valid shapelike value")
    return tuple(result)


def shapelike_float_signed_nested(value: str):
    """Parse signed float shapelike with TensorList nesting support.

    Supports same nesting as shapelike_stc_nested but allows float and None values.
    e.g. ((None, 1.0), (-1.0, 1.0))                → flat
         (((None, 1.0), (None, 1.0)), (-1.0, 1.0))  → nested TensorList
    """
    value = re.sub(r"(-0+\.0*|-0+\.?)(?!(\d|\.))", r"-float(0)", value)
    value = re.sub(r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))', lambda match: f'float("{match.group(1)}")', value)
    allowed = (int, float, type(None))
    parsed = _eval(value)
    if parsed is None:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError(f"{value} is not a valid shapelike value!")
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    if all(isinstance(e, allowed) for e in parsed):
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_items = []
                for sub in element:
                    if sub is None:
                        sub_items.append(None)
                    elif isinstance(sub, (tuple, list)):
                        for v in sub:
                            if not isinstance(v, allowed):
                                raise TypeError(f"{str(v)} is not a valid float value in {value}")
                        sub_items.append(tuple(sub))
                    else:
                        raise TypeError(f"{str(sub)} is not a valid float range in {value}")
                result.append(tuple(sub_items))
            else:
                for v in element:
                    if not isinstance(v, allowed):
                        raise TypeError(f"{str(v)} is not a valid float value in {value}")
                result.append(tuple(element))
        else:
            raise TypeError(f"{str(element)} of {value} is not a valid float shapelike value")
    return tuple(result)


def shapelike_float_nested(value: str):
    """Parse float shapelike with TensorList nesting support (positive only).

    Same nesting as shapelike_float_signed_nested but enforces positive values.
    e.g. ((0.001, 0.001), (0.001, 0.001))               → flat
         (((0.001, 0.001), (0.001, 0.001)), (0.001, 0.001))  → nested TensorList
    """
    parsed = _eval(value)
    if parsed is None:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError(f"{value} is not a valid shapelike value!")
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    allowed = (int, float)
    if all(isinstance(e, allowed) for e in parsed):
        if not all(d >= 0 for d in parsed if isinstance(d, (int, float))):
            logging.warning(f"shapelike value should not have negative dim {str(parsed)}")
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_items = []
                for sub in element:
                    if sub is None:
                        sub_items.append(None)
                    elif isinstance(sub, (tuple, list)):
                        for v in sub:
                            if not isinstance(v, allowed):
                                raise TypeError(f"{str(v)} is not a valid float value in {value}")
                            if v < 0:
                                raise ValueError(f"shapelike value should not have negative dim {str(sub)}")
                        sub_items.append(tuple(sub))
                    else:
                        raise TypeError(f"{str(sub)} is not a valid float range in {value}")
                result.append(tuple(sub_items))
            else:
                for v in element:
                    if not isinstance(v, allowed):
                        raise TypeError(f"{str(v)} is not a valid float value in {value}")
                    if v < 0:
                        raise ValueError(f"shapelike value should not have negative dim {str(element)}")
                result.append(tuple(element))
        else:
            raise TypeError(f"{str(element)} of {value} is not a valid float shapelike value")
    return tuple(result)


def _container(value: str, allowed_type: Union[type, tuple, list]):
    result = _eval(value)
    for t in allowed_type:
        if isinstance(result, t):
            if result is None:
                return result
            return (result,)
    result = tuple(result)
    for element in result:
        if allowed_type and not isinstance(element, allowed_type):
            raise TypeError(
                f"Received type {str(type(element))} for element {str(element)} instead of {str(allowed_type)}"
            )
    return result


def string_container(value: str) -> tuple:
    """
    Container for multiple string
    :param value:
    :return:
    """
    # noinspection PyBroadException
    try:
        result = _container(value, (str, type(None)))
    except Exception:
        result = (value,)
    return result


def scalar_nested(value: str, allowed_type=None) -> tuple:
    """Scalar field with TensorList nesting support.

    Handles string, float, and int element types.
    'float32'                              → ('float32',)
    1e-08                                  → (1e-08,)
    ('float32', 'float32')                 → ('float32', 'float32')
    (1e-08, 1e-08)                         → (1e-08, 1e-08)
    (('float32','float32'), 'float32')     → (('float32','float32'), 'float32')
    ''                                     → ()

    :param allowed_type: type or tuple of types for element validation.
                         None means no validation.
    """
    if value == "":
        return ()
    try:
        parsed = _eval(value)
    except Exception:
        raise TypeError(f"{value} is not a valid scalar value!") from None
    if not isinstance(parsed, (tuple, list)):
        if allowed_type and not isinstance(parsed, allowed_type):
            raise TypeError(f"Value {parsed!r} (type {type(parsed).__name__}) is not {allowed_type}")
        return (parsed,)
    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if allowed_type:
                for e in element:
                    if not isinstance(e, allowed_type):
                        raise TypeError(
                            f"Element {e!r} (type {type(e).__name__}) in nested group is not {allowed_type}"
                        )
            result.append(tuple(element))
        else:
            if allowed_type and not isinstance(element, allowed_type):
                raise TypeError(f"Element {element!r} (type {type(element).__name__}) is not {allowed_type}")
            result.append(element)
    return tuple(result)


def process_eval(value: str):
    return _eval(value)


def int_container(value: str) -> tuple:
    """
    Container for multiple integer
    :param value:
    :return:
    """
    if value == "":
        return ()
    try:
        result = _container(value, (int, type(None)))
    except TypeError as terr:
        raise TypeError(f"Invalid value {value} for int_container: {terr.args}") from terr
    except Exception as e:
        raise ValueError(f"Invalid value {value} for int_container: {e.args}") from e
    else:
        return result


def process_int(value: str) -> Optional[int]:
    """
    Integer
    :param value:
    :return:
    """
    if value is None or value in {"None", ""}:
        return None
    try:
        result = _eval(value)
    except Exception:
        raise TypeError(f"Invalid value [{value}] for int") from None
    else:
        if isinstance(result, int):
            return result
        if result is None:
            return None
        raise TypeError(f"Invalid value [{value}] for int")


def process_dict(value: str) -> dict:
    """
    dictionary
    :param value:
    :return:
    """
    try:
        # convert -0.0 or -0 to -float(0) for powf & atan operator
        value = re.sub(r"(-0+\.0*|-0+\.?)(?!(\d|\.))", r"-float(0)", value)

        # convert inf to float(inf), nan to float(nan)
        # but skip when nan/inf is inside single/double quotes (dict key)
        def _replace_nan_inf(m):
            start = m.start()
            if start > 0 and value[start - 1] in ("'", '"'):
                return m.group(0)
            return f'float("{m.group(1)}")'

        value = re.sub(r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))', _replace_nan_inf, value)
        result = _eval(value)
    except Exception:
        raise ValueError(f"Invalid value {value} for dict") from None
    else:
        if isinstance(result, (dict, type(None))):
            return result
        raise TypeError(f"Value {value} is not a dict")
