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
torch 参数解析器（torch/ 域）。

Multi-strategy fallback:
  1. TypeError fallback — call with wrong args, parse error message
  2. Manual override config — for APIs that can't be auto-parsed or have incorrect parsing

Returns APIParamInfo with typed parameter list including Tensor/List[Tensor]/Scalar/etc.
数据模型（ParamInfo/OverloadInfo/APIParamInfo/resolve_function）见 framework_api/api_model.py。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from ..api_model import (
    APIParamInfo,
    OverloadTensorLayout,
    ParamInfo,
    is_tensor_method_name,
    resolve_function,
)
from .extract_cpp import (
    _probe_cpp_signature,
)
from .extract_declaration import _extract_params_from_op_declaration
from .extract_docstring import (
    _extract_params_from_docstring,
    _parse_docstring_args_section,  # noqa: F401  # re-export（测试直引）
    _parse_params_from_signature,
    _parse_return_count_from_first_line,  # noqa: F401  # re-export（测试直引）
)
from .extract_pyi import _extract_params_from_pyi
from .extract_schema import (
    _extract_params_from_aten_schemas,
    _torch_dtype_type,
    _torch_tensor_type,
)
from .type_inference import (
    _coerce_default_value,  # noqa: F401  # re-export（测试直引）
    _enrich_types_from_annotations,
    _infer_type_from_name,  # noqa: F401  # re-export（测试直引）
    _infer_type_from_value,  # noqa: F401  # re-export（测试直引）
    _is_name_inferred_type,
    _normalize_args_type,  # noqa: F401  # re-export（测试直引）
)


def _extract_params_from_type_error(obj) -> Optional[Tuple[List[ParamInfo], str, List[List[ParamInfo]]]]:
    """Parse TypeError from calling obj() with wrong args.

    Returns (primary_params, source, all_overloads).
    For multi-overload APIs, all_overloads contains all parsed signatures.
    For single-signature APIs, all_overloads is empty (use primary_params).
    """
    try:
        obj()
    except TypeError as e:
        error_msg = str(e)
        if "expected one of:" in error_msg:
            all_overloads = []
            for line in error_msg.split("\n"):
                line_s = line.strip()
                if line_s.startswith("*"):
                    sig = line_s[1:].strip()
                    if sig.startswith("(") and sig.endswith(")"):
                        sig = sig[1:-1]
                    parsed = _parse_params_from_signature(sig)
                    if parsed:
                        all_overloads.append(parsed)
            if all_overloads:
                return all_overloads[0], f"TypeError(multi-overload, {len(all_overloads)} signatures)", all_overloads
        m = re.search(r"expected\s*\(([^)]+(?:\([^)]*\))*[^)]*)\)", error_msg)
        if m:
            sig = m.group(1)
            params = _parse_params_from_signature(sig)
            if params:
                return params, "TypeError(single-signature)", []
    except Exception as _e:
        logging.debug("param extract skipped: %s", _e)
    return None


_SEQUENCE_INNER_TYPES = {"Tensor": "tuple of Tensors", "int": "tuple of ints", "float": "tuple of floats"}
_SCALAR_ANNOTATION_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}


def _sequence_annotation_to_type(annotation) -> str:
    """list/tuple/Sequence[...] 按其元素类型归一成 tuple of xxx。"""
    args = getattr(annotation, "__args__", ())
    if not args:
        return "int"
    return _SEQUENCE_INNER_TYPES.get(_annotation_to_type(args[0]), "int")


def _annotation_to_type(annotation) -> str:
    """Convert a type annotation object to our type string.

    Returns empty string for None or unrecognized annotations.
    """
    if annotation is None:
        return ""
    import collections.abc
    import types as _types
    import typing

    # PEP 604 的 `str | None` 是 types.UnionType,没有 __origin__(torch>=2.6 的
    # .pyi/源码普遍改用这种写法),故 __origin__ 取不到时用 get_origin 兜底。
    origin = getattr(annotation, "__origin__", None) or typing.get_origin(annotation)
    # 不能写成 origin in (..., getattr(_types, 'UnionType', None)):Python < 3.10 无 UnionType,
    # getattr 返回 None,而无 origin 的注解(int/float/str/bool)其 origin 恰为 None,
    # 会整片落进 Union 分支、args 为空后误返回 "Tensor"。
    if origin in (typing.Optional, typing.Union) or (hasattr(_types, "UnionType") and origin is _types.UnionType):
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        return _annotation_to_type(non_none[0]) if non_none else "Tensor"
    # 序列注解识别:list/tuple 的 origin 即 list/tuple(typing.List/Tuple 亦然),
    # 而 typing.Sequence[T] 与 collections.abc.Sequence[T] 的 origin 是 collections.abc.Sequence。
    if origin in (list, tuple, collections.abc.Sequence):
        return _sequence_annotation_to_type(annotation)
    scalar = _SCALAR_ANNOTATION_TYPES.get(annotation)
    if scalar:
        return scalar
    try:
        if isinstance(annotation, type) and issubclass(annotation, int):
            return "int"
    except TypeError:
        pass  # noqa: S110 协议探测：失败跳过属预期
    if annotation is _torch_dtype_type():
        return "Dtype"
    if annotation is _torch_tensor_type():
        return "Tensor"
    return ""


def _extract_params_from_annotations(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Extract params from obj.__annotations__.

    Works for any Python object that has __annotations__ (functions, classes, etc).
    Does NOT require inspect.signature — safe to call on any object.
    Default values are obtained from inspect.signature if available, otherwise None.
    """
    ann = getattr(obj, "__annotations__", None)
    if not ann:
        return None

    param_names = [k for k in ann if k != "return"]
    if not param_names:
        return None

    import inspect as _inspect

    defaults_map = {}
    kw_only_set = set()
    try:
        sig = _inspect.signature(obj)
        for pname, param in sig.parameters.items():
            if param.default != _inspect.Parameter.empty:
                defaults_map[pname] = param.default
            if param.kind == _inspect.Parameter.KEYWORD_ONLY:
                kw_only_set.add(pname)
    except (ValueError, TypeError):
        pass  # noqa: S110 协议探测：失败跳过属预期

    params = []
    for pname in param_names:
        type_hint = _annotation_to_type(ann[pname])
        if not type_hint:
            type_hint = _infer_type_from_name(pname)
        default_raw = defaults_map.get(pname)
        is_optional = pname in defaults_map
        default = _coerce_default_value(str(default_raw), type_hint) if is_optional else None
        params.append(
            ParamInfo(
                name=pname,
                type=type_hint,
                default=default,
                is_optional=is_optional,
                is_keyword_only=pname in kw_only_set,
            )
        )

    return (params, "__annotations__") if params else None


def _extract_params_from_inspect(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Try inspect.signature for Python-callable functions."""
    import inspect as _inspect

    try:
        sig = _inspect.signature(obj)
    except (ValueError, TypeError):
        return None

    ann = getattr(obj, "__annotations__", None) or {}
    params = []
    for pname, param in sig.parameters.items():
        if param.kind == _inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == _inspect.Parameter.VAR_POSITIONAL:
            type_hint = _infer_type_from_name(pname)
            if type_hint in ("Tensor", "list of Tensors", "tuple of Tensors", "Tensor[]", "List[Tensor]"):
                params.append(
                    ParamInfo(name=pname, type="Tensor", default=None, is_optional=False, is_var_positional=True)
                )
            continue

        if pname in ann:
            type_hint = _annotation_to_type(ann[pname])
        elif param.annotation is not None and param.annotation != _inspect.Parameter.empty:
            type_hint = _annotation_to_type(param.annotation)
        else:
            type_hint = _infer_type_from_name(pname)
        if _is_name_inferred_type(type_hint) and param.default != _inspect.Parameter.empty:
            if isinstance(param.default, bool):
                type_hint = "bool"
            elif isinstance(param.default, int):
                type_hint = "int"
            elif isinstance(param.default, float):
                type_hint = "float"
            elif isinstance(param.default, str):
                type_hint = "str"
        is_optional = param.default != _inspect.Parameter.empty
        default = None if not is_optional else _coerce_default_value(str(param.default), type_hint)
        # In Python, parameters after *args are implicitly keyword-only
        # (e.g. meshgrid(*tensors, indexing=None)).  Mapping this correctly
        # ensures build_args() routes such params to kwargs instead of args.
        is_kw_only = param.kind == _inspect.Parameter.KEYWORD_ONLY
        params.append(
            ParamInfo(name=pname, type=type_hint, default=default, is_optional=is_optional, is_keyword_only=is_kw_only)
        )
    return (params, "inspect.signature") if params else None


def _strip_inplace_if_separate_exists(api_name: str, params: List[ParamInfo]) -> List[ParamInfo]:
    """torch.xxx_ 存在独立原地 API 时移除 inplace 参数（原地变体单独测试的设计约定）。"""
    if not api_name.startswith("torch.") or api_name.startswith("torch.nn."):  # noqa: PIE810 保持
        return params
    if api_name.endswith("_"):
        params[:] = [p for p in params if p.name != "inplace"]
        return params
    inplace_name = api_name + "_"
    inplace_obj = resolve_function(inplace_name)
    if inplace_obj is None:
        return params
    params[:] = [p for p in params if p.name != "inplace"]
    return params


# Step 4 kw-only 校验的类型修正探测表（窄表：仅标量类型互转）


def _strip_alias_extra_params(api_name: str, params: List[ParamInfo]) -> List[ParamInfo]:
    """Post-process parsed params for C++ builtins.

    1. Remove 'inplace' if a separate inplace API exists (design: test inplace separately).
    2. Remove params whose type resolved to 'name' (named-dimension artefact).
    3. For C++ builtins: probe to detect keyword-only boundary and remove
       params the C++ function does not accept at all.
    """
    _strip_inplace_if_separate_exists(api_name, params)

    if not api_name.startswith("torch."):
        return params

    params[:] = [p for p in params if p.type != "name"]

    obj = resolve_function(api_name)
    is_cpp = obj is not None and type(obj).__name__ == "builtin_function_or_method"

    if is_cpp:
        _probe_cpp_signature(obj, params)

    return params


def _extract_params_from_functional_alias(api_name: str) -> Optional[Tuple[List[ParamInfo], str]]:
    """矩阵第⑦级后备：本体失败时按别名候选解析（候选侧跑完整 D/T/A 裁决）。

    候选三路：torch.X → torch.nn.functional.X；npu_X → torch_npu.X；原地 X_ →
    去下划线的 torch.X / F.X。source 带 alias(候选, 内部胜者) 标记。
    """
    if not api_name.startswith("torch."):
        return None
    func_name = api_name.split(".")[-1]

    candidates = [f"torch.nn.functional.{func_name}"]

    if func_name.startswith("npu_"):
        candidates.append(f"torch_npu.{func_name}")

    if func_name.endswith("_") and not func_name.startswith("_"):
        base_name = func_name[:-1]
        candidates.append(f"torch.{base_name}")
        candidates.append(f"torch.nn.functional.{base_name}")

    for candidate in candidates:
        obj = resolve_function(candidate)
        if obj is None:
            continue
        doc_result = _extract_params_from_docstring(obj, candidate)
        te_result = _extract_params_from_type_error(obj)
        ann_result = _extract_params_from_annotations(obj)

        if te_result:
            params, source, overloads = te_result
            if overloads and len(overloads) > 1:
                _enrich_types_from_annotations(overloads[0], ann_result[0] if ann_result else [])
                _strip_alias_extra_params(api_name, overloads[0])
                return overloads[0], f"alias({candidate}, {source})"

        if doc_result and te_result:
            merged = _merge_docstring_and_type_error(doc_result[0], te_result[0])
            if merged:
                if ann_result:
                    _enrich_types_from_annotations(merged, ann_result[0])
                _strip_alias_extra_params(api_name, merged)
                return merged, f"alias({candidate}, merge({doc_result[1]}, {te_result[1]}))"

        if doc_result:
            params, doc_source, _ = doc_result
            if ann_result:
                _enrich_types_from_annotations(params, ann_result[0])
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, {doc_source})"

        if te_result:
            params, te_source, _ = te_result
            if ann_result:
                _enrich_types_from_annotations(params, ann_result[0])
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, {te_source})"

        if ann_result:
            params = ann_result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, __annotations__)"

        result = _extract_params_from_inspect(obj)
        if result:
            params = result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, inspect.signature)"
        result = _extract_params_from_op_declaration(candidate)
        if result:
            params = result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, torch_npu._op())"
    return None


def _prefer_inspect_var_args(obj, params, api_name, ann_result=None):
    """当前源缺 *args 张量位、inspect.signature 有 → 换 inspect 结果。

    docstring 把 ``*tensors`` 记成普通参数、__annotations__ 直接省略 *args
    条目，两类源都表达不了 VAR_POSITIONAL；inspect 能查到即更准确，返回
    换源后的 APIParamInfo（source="inspect.signature"），否则 None。
    ann_result 给定时顺带补类型。"""
    has_var_pos = any(getattr(p, "is_var_positional", False) for p in params)
    if has_var_pos:
        return None
    inspect_result = _extract_params_from_inspect(obj)
    if not inspect_result:
        return None
    inspect_has_var_pos = any(getattr(p, "is_var_positional", False) for p in inspect_result[0])
    if not inspect_has_var_pos:
        return None
    inspect_params = inspect_result[0]
    inspect_names = {p.name for p in inspect_params}
    if inspect_names <= {"args", "kwargs"}:
        return None
    if ann_result:
        _enrich_types_from_annotations(inspect_result[0], ann_result[0])
    _strip_alias_extra_params(api_name, inspect_result[0])
    return APIParamInfo(api_name=api_name, params=inspect_result[0], source="inspect.signature")


def _prefer_inspect_keyword_only(obj, params, api_name):
    """当前源无 kw-only 参数、inspect.signature 有 → 换 inspect 结果。

    docstring 保不住 ``*`` 分隔，kw-only 边界丢失；inspect 查到即更准确。
    返回换源后的 APIParamInfo（source="inspect.signature"），否则 None。"""
    has_kw = any(getattr(p, "is_keyword_only", False) for p in params)
    if has_kw:
        return None
    inspect_result = _extract_params_from_inspect(obj)
    if not inspect_result:
        return None
    inspect_has_kw = any(getattr(p, "is_keyword_only", False) for p in inspect_result[0])
    if not inspect_has_kw:
        return None
    inspect_names = {p.name for p in inspect_result[0]}
    if inspect_names <= {"args", "kwargs"}:
        return None
    _strip_alias_extra_params(api_name, inspect_result[0])
    return APIParamInfo(api_name=api_name, params=inspect_result[0], source="inspect.signature")


def extract_api_params(api_name: str) -> Optional[APIParamInfo]:
    """解析入口（异常收敛）：链上任一源抛异常 → warning 日志 + None。"""
    try:
        return _extract_api_params_impl(api_name)
    except Exception as e:
        logging.warning(f"extract_api_params({api_name}) raised {type(e).__name__}: {e}")
        return None


def _extract_api_params_impl(api_name: str) -> Optional[APIParamInfo]:
    """多源解析链编排：纯路由（对象解析归各族函数），矩阵见 _extract_params_from_generic。

    A schema 注册表 —— _extract_params_from_aten_schemas：resolved 对象带
                      _schemas（torch.ops.*、torch_npu.* ≥2.9）即 FunctionSchema
                      权威直出，任意 api_name 形态均先试；
    B npu 族        —— _extract_params_from_npu：Declaration → docstring → inspect
                      族内闭合链。torch_npu.* 三级穷尽即终止（不出族）；
                      torch.npu_*（deprecated 别名）仅 Declaration 一级；
    C 通用          —— 其余全部（含 B 出族的 torch.npu_* 别名）：
                      D/T/A 组合裁决 + 后备链（inspect → 别名 → pyi）。
    """
    # The extension package is imported by FrameworkApiInfoKeeper.get before
    # this is reached; no import here.
    info = _extract_params_from_aten_schemas(api_name)
    if info:
        return info

    info = _extract_params_from_npu(api_name)
    if info is not None:
        return info

    if api_name.startswith("torch_npu."):
        return None  # npu 族内穷尽（Declaration/docstring/inspect 全败），不出族

    return _extract_params_from_generic(api_name)


def _extract_params_from_npu(api_name: str) -> Optional[APIParamInfo]:
    """npu 族内闭合链：Declaration → docstring → inspect（首到先得）。

    域：torch_npu.*（三级全走，穷尽返回 None，编排层终止）+ torch.npu_*
    （deprecated 别名，仅 Declaration 一级——None 后由编排层路由至通用链）。
    Declaration 入口自门控（extract_declaration.py，内部自行 resolve）；本函数
    只做族内编排与包装。
    """
    result = _extract_params_from_op_declaration(api_name)
    if result:
        params, source, return_count = result
        return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=[return_count])

    if not api_name.startswith("torch_npu."):
        return None
    obj = resolve_function(api_name)
    if obj is None:
        return None
    doc_result = _extract_params_from_docstring(obj, api_name)
    if doc_result:
        params, source, doc_rc = doc_result
        return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=[doc_rc] if doc_rc else [])
    result = _extract_params_from_inspect(obj)
    if result:
        params, source = result
        return APIParamInfo(api_name=api_name, params=params, source=source)
    return None


def _apply_inspect_preference(obj, params, api_name, ann, kw_first: bool = False):
    """inspect 更准时换用：docstring 缺 kw-only 或 *args 张量位时。

    kw_first：先探 kw-only（docstring 源特有），再探 *args；
    不换源返回 None，由调用方 finalize。
    """
    if kw_first:
        upgraded = _prefer_inspect_keyword_only(obj, params, api_name)
        if upgraded:
            return upgraded
    upgraded = _prefer_inspect_var_args(obj, params, api_name, ann)
    return upgraded


def _merge_docstring_and_type_error(
    doc_params: List[ParamInfo], te_params: List[ParamInfo]
) -> Optional[List[ParamInfo]]:
    """Merge docstring and TypeError results as equal-priority sources.

    Rules:
    - Use the longer param list as base (more info is better).
    - For each param present in both, take the more authoritative type:
      - If one is clearly inferred (generic Tensor/int/float) and the other
        is specific (Number, torch.dtype, tuple of ints, etc.), prefer the specific one.
      - Otherwise keep the base param's type.
    - Preserve defaults from whichever source has them.
    """
    if not doc_params or not te_params:
        return doc_params or te_params

    # Build name→ParamInfo lookup for the secondary source
    base = doc_params if len(doc_params) >= len(te_params) else te_params
    other = te_params if base is doc_params else doc_params
    other_map = {p.name: p for p in other}

    merged = []
    for bp in base:
        op = other_map.get(bp.name)
        if op is None:
            merged.append(bp)
            continue

        name = bp.name
        # Pick type: prefer authoritative over inferred
        if bp.type == op.type:
            ptype = bp.type
        elif _is_name_inferred_type(bp.type) and not _is_name_inferred_type(op.type):
            ptype = op.type
        elif not _is_name_inferred_type(bp.type) and _is_name_inferred_type(op.type):
            ptype = bp.type
        else:
            # Both are specific but different — prefer TypeError (C++ ground truth)
            ptype = op.type if other is te_params else bp.type

        # Pick default: prefer non-empty
        default = bp.default if bp.default is not None else op.default

        # Pick optional: either source says optional
        is_optional = bp.is_optional or op.is_optional

        # Pick keyword_only: TypeError is authoritative for positional/keyword boundary
        is_kw = op.is_keyword_only if other is te_params else bp.is_keyword_only

        merged.append(ParamInfo(name=name, type=ptype, default=default, is_optional=is_optional, is_keyword_only=is_kw))

    # Add params only in 'other' that aren't in 'base'
    base_names = {p.name for p in base}
    for op in other:
        if op.name not in base_names:
            merged.append(op)

    return merged if merged else None


def _extract_params_from_generic(api_name: str) -> Optional[APIParamInfo]:
    """通用家族组合裁决矩阵（D=docstring，T=TypeError，A=annotations，三者先并行采集）。

    首个命中即返回：
      ① T 多重载(>1)    → T 直出——重载集唯一来源（A 补类型）
      ② D ∧ T 单签名    → merge(D,T)：参数更长者作基座、另一侧补类型（A 再补）
      ③ D              → D（kw 键不齐/缺 *args 时升级 inspect 换源重解；A 补类型）
      ④ T 单签名        → T（A 补类型）
      ⑤ A              → A（零张量参数时升级 inspect 补 *args）
      ⑥ inspect         → 纯 Python 函数直解（C++ builtin 上为 ValueError）
      ⑦ functional 别名 → 原地/函数式变体的签名挂在别名上（已按别名侧收尾，不再 strip）
      ⑧ pyi stub        → Tensor 方法/nn.functional 的 stub 签名（多重载/单签名）

    对象解析自包含（resolve 失败即 None——编排层不再预解析传递）。
    """
    obj = resolve_function(api_name)
    if obj is None:
        return None
    doc = _extract_params_from_docstring(obj, api_name)
    te = _extract_params_from_type_error(obj)
    ann = _extract_params_from_annotations(obj)

    if te and len(te[2]) > 1:  # ①
        params, source, overloads = te
        return _finalize(api_name, overloads[0], source, ann, overloads=overloads)

    if doc and te:  # ②
        (doc_params, doc_source, doc_rc), (te_params, te_source, _) = doc, te
        merged = _merge_docstring_and_type_error(doc_params, te_params)
        if merged:
            return _finalize(
                api_name, merged, f"merge({doc_source}, {te_source})", ann, return_counts=[doc_rc] if doc_rc else []
            )

    if doc:  # ③
        params, source, doc_rc = doc
        upgraded = _apply_inspect_preference(obj, params, api_name, ann, kw_first=True)
        return upgraded or _finalize(api_name, params, source, ann, return_counts=[doc_rc] if doc_rc else [])

    if te:  # ④
        params, source, _overloads = te
        return _finalize(api_name, params, source, ann, overloads=[params])

    if ann:  # ⑤
        params, source = ann
        upgraded = _apply_inspect_preference(obj, params, api_name, None)
        return upgraded or _finalize(api_name, params, source)

    result = _extract_params_from_inspect(obj)  # ⑥
    if result:
        params, source = result
        return _finalize(api_name, params, source)

    result = _extract_params_from_functional_alias(api_name)  # ⑦
    if result:
        params, source = result
        return APIParamInfo(api_name=api_name, params=params, source=source)

    result = _extract_params_from_pyi(api_name)  # ⑧
    if result:
        params, source, overloads = result
        if len(overloads) > 1:
            return _finalize(api_name, overloads[0], source, None, overloads=overloads)
        return _finalize(api_name, params, source, None, overloads=[params])

    return None


def _finalize(
    api_name: str,
    params: List[ParamInfo],
    source: str,
    ann: Optional[Tuple[List[ParamInfo], str]] = None,
    return_counts: Optional[List[int]] = None,
    overloads: Optional[List[List[ParamInfo]]] = None,
) -> APIParamInfo:
    """裁决后统一收尾：A 侧补类型 → 别名后处理（probing 剔除）→ 构造 APIParamInfo。

    overloads 非 None 时走多重载构造（params 须为 overloads[0]）；
    否则按 return_counts（可为空）构造单签名。
    """
    if ann:
        _enrich_types_from_annotations(params, ann[0])
    _strip_alias_extra_params(api_name, params)
    if overloads is not None:
        return APIParamInfo(api_name=api_name, params=params, source=source, overloads=overloads)
    return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=return_counts or [])


_MANUAL_OVERRIDES: Dict[str, APIParamInfo] = {}

# hccl 手工表惰性装载标记（get_api_params 首查触发，进程内一次）
_HCCL_TABLE_READY = False


def _load_hccl_manual_table():
    """hccl 手工表惰性装载（幂等）。

    torch_npu 系不登记（npu._schemas 权威解析），表仅存 mindspeed 项。
    直接存全量 APIParamInfo（含 overloads/return_count）——register_api_params
    的平参登记会重建 overloads 且丢 return_count（out 位期望数）。
    """
    global _HCCL_TABLE_READY
    if _HCCL_TABLE_READY:
        return
    _HCCL_TABLE_READY = True
    from .hccl_apis import hccl_apis as _hccl_table

    _MANUAL_OVERRIDES.update(_hccl_table)


def register_api_params(api_name: str, params: List[ParamInfo], source: str = "manual"):
    """平参登记进 _MANUAL_OVERRIDES（overloads 重建且 return_count=0——需保真
    return_count 的全量登记走 _load_hccl_manual_table 直存 APIParamInfo）。
    """
    _MANUAL_OVERRIDES[api_name] = APIParamInfo(api_name=api_name, params=params, source=source)


def get_api_params(api_name: str) -> Optional[APIParamInfo]:
    """keeper 面解析入口：manual 表首查（惰性装载 hccl 表）→ 六源链 → Tensor 方法
    补 self 位（别名结果已含 self 等价张量的除外）→ 置解析标志（is_tensor_method /
    is_mutable 命名投影，schema 源已在 _parse_aten_schema_arguments 落位）。
    """
    _load_hccl_manual_table()
    if api_name in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[api_name]
    result = extract_api_params(api_name)
    if result is not None:
        result.is_tensor_method = is_tensor_method_name(api_name)
        if result.is_tensor_method:
            _inject_self_param(result)
        result.mark_inplace_mutable()
    return result


def _inject_self_param(info: APIParamInfo) -> None:
    """Tensor 方法首参补 self 位。

    跳过两类：functional 别名结果已含等价张量（source 带 alias(，
    如 torch.nn.functional.relu_ 的 input）；首参已是 self（重复调用/
    inspect 源自带，防叠加）。多重重载逐表插入并重建 layout，params
    取可选参数最多的重载。
    """
    if "alias(" in (info.source or ""):
        return
    if info.params and info.params[0].name == "self":
        return
    self_param = ParamInfo(name="self", type="Tensor")
    if info.overloads:
        for ov in info.overloads:
            ov.params.insert(0, self_param)
            ov.layout = OverloadTensorLayout.build(ov.params, ov.return_count)
        best = max(info.overloads, key=lambda oi: sum(1 for p in oi.params if p.is_optional))
        info.params = best.params
    else:
        info.params.insert(0, self_param)
