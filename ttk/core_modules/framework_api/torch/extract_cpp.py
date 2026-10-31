#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""C++ builtin 试调用探测源：对无 Python 签名的内置算子构造探针调用，
从 TypeError 报错反推真实参数结构（参数名/最大位置边界/kw-only/剔除/类型纠偏）。
"""

import logging
import re
from typing import List, Optional

from ..api_model import ParamInfo


def _build_probes(params: List[ParamInfo]) -> Optional[list]:
    """按参数类型构造探测值列表；遇到未知类型返回 None。"""
    import torch as _torch

    _PROBE_MAP = {
        "int": 0,
        "float": 1.0,
        "Number": 1.0,
        "bool": False,
        "str": "mean",
        "torch.dtype": _torch.float32,
        "Dtype": _torch.float32,
        "torch.memory_format": _torch.contiguous_format,
    }

    dummy = _torch.zeros(2, 3)
    probes = []
    for p in params:
        if p.is_tensor_like:
            probes.append(dummy)
        else:
            val = _PROBE_MAP.get(p.type)
            if val is None:
                return None
            probes.append(val)
    return probes


def _extract_cpp_param_names(obj) -> Optional[set]:
    """从 "missing N required positional" TypeError 提取 C++ 实参名集。"""
    try:
        obj()
    except TypeError as e:
        msg = str(e)
        m = re.search(r"missing \d+ required positional arguments?: (.+)", msg)
        return {n.strip().strip("\"'") for n in m.group(1).split(",")} if m else None
    except Exception:
        return None


def _probe_max_positional(obj, probes, param_count: int) -> int:
    """渐进调用探测最大位置参数边界。

    已知权衡：任意异常（"error"）按"执行失败≠签名拒绝"计为有效边界（与
    _correct_type_by_probes 的探测协议一致），设备级错误会污染判定——
    探测仅发生在解析期，调用方须保证探测环境可用。"""

    def _classify(n):
        try:
            obj(*probes[:n])
            return "ok"
        except TypeError as e:
            msg = str(e).lower()
            if "too many" in msg or ("takes" in msg and "positional" in msg):
                return "too_many"
            if "invalid combination" in msg:
                return "too_many"
            if "missing" in msg:
                return "missing"
            return "error"
        except Exception:
            return "error"

    max_pos = 0
    found_valid = False
    for n in range(param_count + 1):
        result = _classify(n)
        if result in ("ok", "error"):
            max_pos = n
            found_valid = True
        elif result == "missing":
            continue
        elif result == "too_many" and found_valid:
            break
    return max_pos


def _first_optional_index(params: List[ParamInfo]) -> int:
    for i, p in enumerate(params):
        if p.is_optional:
            return i
    return len(params)


def _remove_deprecated_loss_params(params: List[ParamInfo], cpp_param_set: set) -> set:
    """剔除 C++ 签名中不存在的废弃 loss 参数（size_average/reduce），返回剔除名集。"""
    _DEPRECATED_LOSS_PARAMS = {"size_average": "bool", "reduce": "bool"}
    to_remove = []
    removed_names = set()
    for i, p in enumerate(params):
        if (
            p.name in _DEPRECATED_LOSS_PARAMS
            and _DEPRECATED_LOSS_PARAMS[p.name] == p.type
            and p.name not in cpp_param_set
        ):
            to_remove.append(i)
            removed_names.add(p.name)
    for idx in reversed(to_remove):
        params.pop(idx)
    return removed_names


def _correct_type_by_probes(obj, call_args, p: ParamInfo, alt_probes) -> bool:
    """按 alt 探测表尝试修正参数类型/默认值；返回是否修正成功。

    TypeError 视为该 alt 值不被接受（换下一个）；其他异常视为"接受但执行
    失败"，同样完成类型修正（与原始探测协议一致）。
    """
    fixed = False
    for alt_type, alt_val in alt_probes.get(p.type, []):
        try:
            obj(*call_args, **{p.name: alt_val})
            p.type = alt_type
            p.default = alt_val
            fixed = True
            break
        except TypeError:
            continue
        except Exception:
            p.type = alt_type
            p.default = alt_val
            fixed = True
            break
    return fixed


def _probe_keyword_only_params(obj, params: List[ParamInfo], probes, max_pos: int):
    """验证 kw-only 参数确实被 C++ 接受；类型不符尝试 alt 探测修正，仍失败则剔除。"""
    kw_only_params = [(i, params[i], probes[i]) for i in range(len(params)) if params[i].is_keyword_only]
    if not kw_only_params:
        return

    base = probes[: _first_optional_index(params)]

    to_remove = []
    for idx, p, val in kw_only_params:
        try:
            obj(*base, **{p.name: val})
        except TypeError as _te:
            _te_msg = str(_te).lower()
            if "unexpected keyword" not in _te_msg:
                continue
            if not _correct_type_by_probes(obj, base, p, _KW_ONLY_ALT_PROBES):
                to_remove.append(idx)
        except Exception as _e:
            logging.debug("param extract skipped: %s", _e)
    for idx in reversed(to_remove):
        params.pop(idx)


def _probe_optional_params(obj, params: List[ParamInfo], probes, max_pos: int):
    """kwarg 探测验证可选参数（Step 3 fallback）；剔除后经约定恢复（size_average→reduction）。"""
    first_optional = _first_optional_index(params)
    if first_optional >= len(params):
        return

    kw_params = [(i, params[i], probes[i]) for i in range(first_optional, len(params))]
    to_remove = []
    for idx, p, val in kw_params:
        test_pos = [probes[j] for j in range(max_pos) if j != idx]
        try:
            obj(*test_pos, **{p.name: val})
        except TypeError as _te:
            _te_msg = str(_te).lower()
            if "missing" in _te_msg and "required positional" in _te_msg:
                continue
            if "unexpected keyword" not in _te_msg and "got an unexpected keyword" not in _te_msg:
                continue
            if not _correct_type_by_probes(obj, test_pos, p, _OPTIONAL_ALT_PROBES):
                to_remove.append(idx)
        except Exception as _e:
            logging.debug("param extract skipped: %s", _e)
    removed_names = set()
    for idx in reversed(to_remove):
        removed_names.add(params[idx].name)
        params.pop(idx)

    if removed_names:
        _restore_deprecated_aliases(obj, probes[:first_optional], params, removed_names)


def _probe_cpp_signature(obj, params: List[ParamInfo]):
    """Probe a C++ builtin to detect keyword-only boundary and validate param names.

    Strategy:
    1. Extract actual C++ param names from "missing N required positional" TypeError
    2. Detect max positional boundary by progressive calling
    3. Remove params not in C++ signature, recover via convention (size_average→reduction)

    Only called for C++ builtins (no inspect.signature available).
    """
    if not params or any(p.is_keyword_only for p in params):
        return

    probes = _build_probes(params)
    if probes is None:
        return

    # Step 1: Extract actual C++ param names from TypeError
    cpp_param_set = _extract_cpp_param_names(obj)

    # Step 2: Detect max positional boundary
    max_pos = _probe_max_positional(obj, probes, len(params))
    for i in range(max_pos, len(params)):
        params[i].is_keyword_only = True

    # Step 3: Remove deprecated params not in C++ signature (if we have ground truth)
    if cpp_param_set is not None:
        removed_names = _remove_deprecated_loss_params(params, cpp_param_set)
        if removed_names:
            _restore_deprecated_aliases(obj, probes[: _first_optional_index(params)], params, removed_names)

    # Step 4: Verify keyword-only params are actually accepted by C++.
    # 无 kw-only 参数时提前返回——原逻辑此时整个探测收尾，fallback 不再执行。
    if not any(p.is_keyword_only for p in params):
        return
    _probe_keyword_only_params(obj, params, probes, max_pos)

    # Step 3 fallback: verify optional params by kwarg probing
    _probe_optional_params(obj, params, probes, max_pos)


def _restore_deprecated_aliases(obj, pos_probes, params, removed_names):
    """probing 剔除参数后的恢复：已废弃名（size_average/reduce）映射现代等价 reduction——
    已存在则改写其类型/默认；不存在则探调验证可接受后按 kw-only 补插。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return

    _DEPRECATED_TO_MODERN = {
        "size_average": ("reduction", "int", 1),
        "reduce": ("reduction", "int", 1),
    }

    for old_name in removed_names:
        if old_name not in _DEPRECATED_TO_MODERN:
            continue
        new_name, new_type, new_default = _DEPRECATED_TO_MODERN[old_name]
        existing = next((p for p in params if p.name == new_name), None)
        if existing is not None:
            existing.type = new_type
            existing.default = new_default
            existing.is_keyword_only = True
            continue
        try:
            obj(*pos_probes, **{new_name: new_default})
        except TypeError:
            continue
        except Exception as _e:
            logging.debug("param extract skipped: %s", _e)
        existing_kw = [p for p in params if p.is_keyword_only]
        insert_idx = len(params)
        if existing_kw:
            insert_idx = params.index(existing_kw[0])
        params.insert(
            insert_idx,
            ParamInfo(name=new_name, type=new_type, default=new_default, is_optional=True, is_keyword_only=True),
        )


_KW_ONLY_ALT_PROBES = {
    "str": [("int", 1)],
    "int": [("str", "mean")],
    "Dtype": [("int", 0)],
    "torch.dtype": [("int", 0)],
    "bool": [("str", "mean")],
}

# Step 3 fallback 可选参数校验的探测表（宽表：含 Tensor 类型名互转）
_OPTIONAL_ALT_PROBES = {
    "str": [("int", 1)],
    "int": [("str", "mean")],
    "Dtype": [("int", 0)],
    "torch.dtype": [("int", 0)],
    "bool": [("str", "mean")],
    "torch.Tensor": "Tensor",
    "torch.BoolTensor": "Tensor",
    "torch.IntTensor": "Tensor",
    "torch.LongTensor": "Tensor",
    "torch.FloatTensor": "Tensor",
    "torch.DoubleTensor": "Tensor",
}
