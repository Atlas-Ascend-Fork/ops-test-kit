#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Convert Ascend profiling reports (kernel_details.csv, torch_npu profiler / msprof export)
into TTK test case CSVs for one of the four test modes.

Target mode (--mode, decided by the user's request when the skill is invoked):
  * aclnn  — ACLNN 模式：按 aclnn 头文件签名对齐张量，生成引擎 API 用例（默认）
  * kernel — Kernel 模式：kernel I/O 原样转换（GE 内部常量保留为算子输入），
             op_name 由 op Type 驼峰转下划线派生（有算子信息库时用 opInterface 名）
  * geir   — GEIR 模式：同 kernel + dyn_input_shapes 列（留空，profiling 无动态 shape 信息）
  * e2e    — E2E 模式：aclnn API 经映射表转 torch API，按 torch 签名对齐张量；
             无映射的 API 跳过并计入报告（映射表 ACLNN_TO_TORCH 可扩展）

Operator libraries (--op-libs, colon-separated checkouts such as ops-nn /
ops-transformer):
  * aclnn/e2e: every ``**/op_api`` directory holding aclnn headers is appended to
    ``TTK_OP_API_HEADER_PATH`` so custom operators resolve without an installed
    op package (source headers win over installed ones, mirroring TTK).
  * e2e: ``torch_extension`` wrappers are scanned for registered torch.ops names
    and schemas; aclnn APIs declared in the same op directory get a derived
    ``torch.ops.<namespace>.<op>`` mapping and the parsed schemas are registered
    into FrameworkApiInfoKeeper, so alignment works without the package built.

Design principles (agreed with repo owner):
  * Signature layer (op/api name / shapes / dtypes / formats)  -> filled from profiling, verbatim.
  * Structure layer (output indexes / scalar dtypes / None placeholders) -> derived from
    op definition / API signatures (aclnn headers, torch signatures) when importable.
  * Value layer (attributes values / data ranges / strides) -> ALWAYS LEFT EMPTY; the
    conversion report lists every case that needs manual values.
  * Management fields -> auto generated (testcase_name, remark).
  * NO dedup by default: one profile row -> one testcase. Rows are never filtered by
    shape size; shapes are kept verbatim.
  * Communication kernels (COMMUNICATION core / Hccl / hcom / hccl / HCCL name prefix)
    and rows without signature info (N/A shapes) are skipped and reported. E2E mode
    additionally skips aclnn APIs without a torch mapping (reported; extend
    ACLNN_TO_TORCH to cover them).

Usage:
    python3 prof2case.py -i kernel_details.csv -o outdir/ [--mode {aclnn,kernel,geir,e2e}] [--dedup] [--no-selfcheck]
                          [--op-libs /path/to/ops-nn:/path/to/ops-transformer]
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict


def _ensure_ttk_importable():
    """The signature libs (ApiLib/E2eApiLib/KernelOpLib) import the ttk package
    from THIS repository; invoked by absolute path from another CWD it is not
    on sys.path and every lib silently degrades to 'unavailable'. Walk up from
    this file and put the repo root (the ttk package's parent) on sys.path."""
    parent = os.path.dirname(os.path.abspath(__file__))
    while parent and os.path.dirname(parent) != parent:
        if os.path.isfile(os.path.join(parent, "ttk", "__init__.py")):
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
        parent = os.path.dirname(parent)


_ensure_ttk_importable()

# --------------------------------------------------------------------------- #
# Mapping tables
# --------------------------------------------------------------------------- #

DTYPE_MAP = {
    "INT8": "int8",
    "DT_INT8": "int8",
    "UINT8": "uint8",
    "DT_UINT8": "uint8",
    "INT16": "int16",
    "DT_INT16": "int16",
    "UINT16": "uint16",
    "DT_UINT16": "uint16",
    "INT32": "int32",
    "DT_INT32": "int32",
    "UINT32": "uint32",
    "DT_UINT32": "uint32",
    "INT64": "int64",
    "DT_INT64": "int64",
    "UINT64": "uint64",
    "DT_UINT64": "uint64",
    "FLOAT": "float32",
    "DT_FLOAT": "float32",
    "FLOAT16": "float16",
    "DT_FLOAT16": "float16",
    "HFLOAT16": "float16",
    "DOUBLE": "float64",
    "DT_DOUBLE": "float64",
    "BF16": "bfloat16",
    "DT_BF16": "bfloat16",
    "BOOL": "bool",
    "DT_BOOL": "bool",
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "DT_FLOAT8_E4M3FN": "float8_e4m3fn",
    "FLOAT8_E5M2": "float8_e5m2",
    "DT_FLOAT8_E5M2": "float8_e5m2",
    "FLOAT8_E8M0": "float8_e8m0",
    "DT_FLOAT8_E8M0": "float8_e8m0",
    "COMPLEX64": "complex64",
    "DT_COMPLEX64": "complex64",
    "COMPLEX128": "complex128",
    "DT_COMPLEX128": "complex128",
}
UNDEFINED_DTYPES = {"DT_UNDEFINED", "UNDEFINED", "N/A", "NULL", "", "NONE"}
NULL_FORMATS = {"NULL", "N/A", ""}

# output params that do not follow the *Out/*Output/*Ref naming convention
EXTRA_OUTPUT_NAMES = {"softmaxLse", "softmaxLseOut"}
# Backward/Grad APIs: params ending with Out that are actually inputs (mirrors TTK rules)
BACKWARD_OUTPUT_EXCLUDE = frozenset({"gradOutput", "gradOut", "grad_output", "attentionOut"})

# scalar dtype heuristics (aclScalar* params carry no dtype in headers)
SCALAR_FLOAT_NAMES = {
    "alpha",
    "beta",
    "eps",
    "epsilon",
    "scale",
    "scaleValue",
    "value",
    "fillValue",
    "gamma",
    "lambda",
    "momentum",
    "dampening",
    "exponent",
    "base",
    "factor",
    "routedScalingFactor",
    "weightThreshold",
    "ratio",
    "p",
    "temperature",
}
SCALAR_INT_NAMES = {
    "dim",
    "axis",
    "k",
    "n",
    "num",
    "count",
    "groups",
    "start",
    "end",
    "step",
    "numHeads",
    "numKeyValueHeads",
    "preTokens",
    "nextTokens",
    "sparseMode",
    "innerPrecise",
    "blockSize",
    "groupSize",
    "splitItem",
    "groupType",
    "groupListType",
    "actType",
    "quantGroupSize",
    "dstType",
    "roundMode",
    "kGroup",
    "groupCount",
    "groupSelectMode",
    "renorm",
    "normType",
    "seed",
    "offset",
    "mode",
    "sortedStrides",
    "cubeMathType",
    "reduceType",
    "normTypeBefore",
}
SCALAR_BOOL_NAMES = {"descending", "sorted", "outFlag", "rounding", "biasFlag", "ceilMode"}
SCALAR_INT_SUFFIXES = ("Size", "Num", "Count", "Mode", "Type", "Index", "Len", "Dim", "Axis", "Tokens", "Heads")

INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}


def guess_scalar_dtype(name: str):
    low = name
    if low in SCALAR_FLOAT_NAMES:
        return "float32"
    if low in SCALAR_BOOL_NAMES:
        return "bool"
    if low in SCALAR_INT_NAMES or low.endswith(SCALAR_INT_SUFFIXES):
        return "int64"
    if any(low.endswith(s) for s in ("Factor", "Scale", "Rate", "Weight", "Eps")):
        return "float32"
    return None  # unknown -> None placeholder + warning


# --------------------------------------------------------------------------- #
# Operator library scanning (--op-libs: ops-nn / ops-transformer checkouts)
# --------------------------------------------------------------------------- #

# same function pattern as ttk OpApiInfoKeeper (single source of truth for parsing)
ACLNN_FUNC_PATTERN = re.compile(r"\b([A-Za-z0-9_]+)\s+\b([A-Za-z0-9_]+)\s*\(([^){]*?)\)(?=\s*\;)")
# directories never holding op_api headers / torch extensions (examples/
# holds sample operators like ops-transformer's mc2/all_gather_add — they are
# not shippable APIs and would inflate the --audit inventory)
_PRUNE_DIRS = frozenset(
    {
        ".git",
        ".arts",
        ".claude",
        ".gitcode",
        ".opencode",
        "__pycache__",
        "build",
        "docs",
        "examples",
        "node_modules",
        "tests",
    }
)


def collect_aclnn_header_dirs(roots):
    """Collect directories that directly hold aclnn op_api headers.

    Both library layouts are covered: ``<category>/<op>/op_api`` (ops-nn) and
    ``<module>/<op>/op_host/op_api`` (ops-transformer); any other directory
    directly containing aclnn*.h files is accepted too."""
    dirs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
            if not any(f.endswith((".h", ".hpp")) for f in filenames):
                continue
            if os.path.basename(dirpath) == "op_api" or any(
                f.startswith("aclnn") and f.endswith((".h", ".hpp")) for f in filenames
            ):
                dirs.append(dirpath)
    return dirs


def aclnn_apis_in_header_dir(hdr_dir):
    """aclnn API names declared by the header files directly inside a directory."""
    apis = set()
    if not os.path.isdir(hdr_dir):
        return apis
    for f in sorted(os.listdir(hdr_dir)):
        if not f.endswith((".h", ".hpp")):
            continue
        try:
            with open(os.path.join(hdr_dir, f), encoding="utf-8", errors="replace") as hf:
                content = hf.read()
        except OSError:
            continue
        for match in ACLNN_FUNC_PATTERN.finditer(content):
            func = match.group(2).strip()
            if func.startswith("aclnn") and func.endswith("GetWorkspaceSize"):
                apis.add(func[: -len("GetWorkspaceSize")])
    return apis


def aclnn_apis_in_dir(op_dir):
    """aclnn API names declared by an op directory's own op_api headers."""
    apis = set()
    for sub in ("op_api", os.path.join("op_host", "op_api")):
        apis |= aclnn_apis_in_header_dir(os.path.join(op_dir, sub))
    return apis


def _literal_return(func_node):
    """Literal (string / list of strings) returned by a function, or None."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                return None
            if isinstance(value, str):
                return [value]
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return list(value)
            return None
    return None


def _builder_name(init_node, constants):
    """Registered op name from a ``super().__init__("name")`` style call.

    Handles both ``super().__init__("literal")`` and the two-argument
    ``super(Cls, self).__init__(OP_NAME)`` form with a module-level constant."""
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_super_init = (
            isinstance(func, ast.Attribute)
            and func.attr == "__init__"
            and isinstance(func.value, ast.Call)
            and getattr(func.value.func, "id", "") == "super"
        ) or (isinstance(func, ast.Attribute) and func.attr == "__init__" and isinstance(func.value, ast.Name))
        if not is_super_init or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return constants.get(arg.id)
    return None


def parse_torch_ext_wrapper(path):
    """(registered_name, [schemas]) from an OpBuilder wrapper, or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as pf:
            tree = ast.parse(pf.read())
    except (OSError, SyntaxError, ValueError):
        return None
    # module-level string constants (OP_NAME = "..."): builders may register
    # via a name constant instead of a literal
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [getattr(b, "id", "") for b in node.bases]
        if "OpBuilder" not in bases:
            continue
        name, schemas = None, None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                name = _builder_name(item, constants)
            elif isinstance(item, ast.FunctionDef) and item.name == "schema":
                schemas = _literal_return(item)
        if name:
            return name, schemas or []
    return None


def _detect_torch_ops_namespace(root):
    """torch.ops namespace from ``<root>/torch_extension/<pkg>/op_builder/builder.py``."""
    pkg_root = os.path.join(root, "torch_extension")
    if not os.path.isdir(pkg_root):
        return None
    for entry in sorted(os.listdir(pkg_root)):
        if os.path.isfile(os.path.join(pkg_root, entry, "op_builder", "builder.py")):
            return entry
    return "cann_ops_transformer"


def _match_aclnn_to_torch(apis, torch_names, notes, op_dir):
    """Pair aclnn APIs with torch op names registered inside one op directory.

    Exact snake-name match wins, then mutual prefix; a directory with a single
    torch op attributes every non-grad aclnn API to it. Grad/Backward APIs are
    never attributed by fallback (their torch wrappers live in separate dirs)."""
    pairs = {}
    for api in sorted(apis):
        snake = camel_to_snake(api[len("aclnn") :]) if api.startswith("aclnn") else camel_to_snake(api)
        exact = [t for t in torch_names if t == snake]
        if len(exact) == 1:
            pairs[api] = exact[0]
            continue
        partial = [t for t in torch_names if snake.startswith(t + "_") or t.startswith(snake + "_")]
        if len(partial) == 1:
            pairs[api] = partial[0]
            continue
        if len(torch_names) == 1 and not snake.endswith(("_grad", "_backward")):
            pairs[api] = torch_names[0]
        elif len(torch_names) > 1:
            notes.append(f"{op_dir}: aclnn API {api} 与多个 torch 算子共存，未派生映射")
    return pairs


class OpLibScan:
    """Result of scanning --op-libs roots: header dirs + torch.ops signatures."""

    def __init__(self):
        self.header_dirs = []
        self.all_apis = {}  # aclnn api -> one header dir declaring it (inventory)
        self.torch_schemas = {}  # torch api (torch.ops.<ns>.<op>) -> [schema strings]
        self.aclnn_to_torch = {}  # aclnn api -> torch api
        self.notes = []


def scan_op_libs(roots):
    """Scan operator library checkouts for aclnn headers and torch extensions."""
    scan = OpLibScan()
    # normalize (trailing slash / surrounding whitespace would break the
    # dirname-based torch_extension package-root detection downstream)
    roots = [os.path.normpath(r.strip()) for r in roots if r and r.strip()]
    scan.header_dirs = collect_aclnn_header_dirs(roots)
    for hdr_dir in scan.header_dirs:
        for api in aclnn_apis_in_header_dir(hdr_dir):
            scan.all_apis.setdefault(api, hdr_dir)
    for root in roots:
        if not os.path.isdir(root):
            scan.notes.append(f"算子库根目录不存在: {root}")
            continue
        namespace = _detect_torch_ops_namespace(root)
        if namespace is None:
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
            if os.path.basename(dirpath) != "torch_extension" or os.path.dirname(dirpath) == root:
                continue
            op_dir = os.path.dirname(dirpath)
            torch_ops = OrderedDict()  # registered name -> schemas
            for py in sorted(filenames):
                if not py.endswith(".py") or py.startswith(("graph_convert", "__")):
                    continue
                parsed = parse_torch_ext_wrapper(os.path.join(dirpath, py))
                if parsed:
                    torch_ops.setdefault(parsed[0], []).extend(parsed[1])
            if not torch_ops:
                continue
            for name, schemas in torch_ops.items():
                scan.torch_schemas.setdefault(f"torch.ops.{namespace}.{name}", []).extend(schemas)
            for api, name in _match_aclnn_to_torch(
                aclnn_apis_in_dir(op_dir), list(torch_ops), scan.notes, op_dir
            ).items():
                if api in scan.aclnn_to_torch:
                    scan.notes.append(f"{api} 在多个算子目录重复声明，保留首个映射 {scan.aclnn_to_torch[api]}")
                    continue
                scan.aclnn_to_torch[api] = f"torch.ops.{namespace}.{name}"
    # global fallback: torch ops whose op directory carries no aclnn header pair
    # with repo-wide APIs by exact snake name (experimental/ module relocations,
    # header/torch-extension split dirs, ...)
    paired = set(scan.aclnn_to_torch.values())
    for torch_api in sorted(scan.torch_schemas):
        if torch_api in paired:
            continue
        torch_name = torch_api.rsplit(".", 1)[-1]
        for api in sorted(scan.all_apis):
            if api in scan.aclnn_to_torch or not api.startswith("aclnn"):
                continue
            if camel_to_snake(api[len("aclnn") :]) == torch_name:
                scan.aclnn_to_torch[api] = torch_api
                paired.add(torch_api)
                break
    return scan


# torch native schema type -> ParamInfo.type member recognized by TTK
_SCHEMA_SCALARS = {
    "int": "int",
    "int64": "int",
    "SymInt": "int",
    "float": "float",
    "double": "float",
    "bool": "bool",
    "str": "str",
    "string": "str",
    "Scalar": "Scalar",
    "Number": "Number",
}


def parse_torch_schema(schema):
    """'op(Tensor x, Tensor? b=None, *, int k=1) -> Tensor' -> (op_name, [param dicts]).

    Param dicts carry the keys of ttk ParamInfo; alias annotations (a!) are
    stripped, '?'/defaults mark optionality, '*' starts keyword-only params."""
    head = re.match(r"\s*([A-Za-z0-9_.]+)\s*\(", schema)
    if not head:
        return None, None
    op_name = head.group(1)
    depth, close = 0, -1
    for i in range(head.end() - 1, len(schema)):
        if schema[i] == "(":
            depth += 1
        elif schema[i] == ")":
            depth -= 1
            if depth == 0:
                close = i
                break
    if close < 0:
        return op_name, None
    body, params, keyword_only = schema[head.end() : close], [], False
    seg, seg_depth = "", 0
    segments = []
    for ch in body:
        if ch == "(":
            seg_depth += 1
        elif ch == ")":
            seg_depth -= 1
        if ch == "," and seg_depth == 0:
            segments.append(seg)
            seg = ""
        else:
            seg += ch
    if seg.strip():
        segments.append(seg)
    for raw in segments:
        piece = raw.strip()
        if not piece:
            continue
        if piece == "*":
            keyword_only = True
            continue
        has_default = "=" in piece
        decl = piece.split("=", 1)[0].strip()
        toks = decl.split()
        if len(toks) < 2:
            continue  # unnamed schema argument — cannot align
        ptype, pname = toks[-2], toks[-1]
        cleaned = re.sub(r"\([^)]*\)", "", ptype)
        is_opt = cleaned.endswith("?")
        is_list = cleaned.endswith("[]")
        base = cleaned.replace("?", "").replace("[]", "")
        if base == "Tensor":
            mapped = "tuple of Tensors" if is_list else "Tensor"
        elif base in _SCHEMA_SCALARS:
            mapped = "list of Numbers" if is_list else _SCHEMA_SCALARS[base]
        else:
            mapped = ""  # Generator / Dict / ... — neither tensor nor scalar
        default = None
        if has_default:
            try:
                default = ast.literal_eval(piece.split("=", 1)[1].strip())
            except (ValueError, SyntaxError):
                default = piece.split("=", 1)[1].strip()
        params.append(
            {
                "name": pname,
                "type": mapped,
                "default": default,
                "is_optional": is_opt or has_default,
                "is_keyword_only": keyword_only,
                "is_var_positional": False,
            }
        )
    return op_name, params


def extend_env_path(env_name, dirs):
    """Append dirs to a colon-separated path env var (no duplicates)."""
    have = [d for d in os.environ.get(env_name, "").split(":") if d]
    for d in dirs:
        if d not in have:
            have.append(d)
    os.environ[env_name] = ":".join(have)


# --------------------------------------------------------------------------- #
# Profile row parsing (tolerant)
# --------------------------------------------------------------------------- #


def clean_cell(s: str) -> str:
    return (s or "").strip().strip('"').strip()


def split_multi(s: str):
    """Split a profiling multi-value cell ('a;b;c'). Empty segments are kept
    ('a;;c' -> ['a','','c']); an empty segment means a scalar/placeholder."""
    s = clean_cell(s)
    if s in ("", "N/A", "NULL", "None"):
        return []
    return [p.strip().strip('"').strip() for p in s.split(";")]


def parse_shape(seg: str):
    """'1,3,1024,1024' -> (1,3,1024,1024); '' -> (); 'None' -> None."""
    seg = (seg or "").strip()
    if seg in ("", "()"):
        return ()
    if seg == "None":
        return None
    try:
        return tuple(int(x) for x in seg.split(",") if x.strip() != "")
    except ValueError:
        return None


def numel(shape) -> int:
    if not shape:
        return 0
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


class KTensor:
    __slots__ = ("shape", "dtype", "fmt")

    def __init__(self, shape, dtype, fmt):
        self.shape, self.dtype, self.fmt = shape, dtype, fmt

    def is_aux(self) -> bool:
        """GE-internal const tensors (perm/begin/end/pad lists/shape descriptors):
        small int tensors that are NOT primary data."""
        return self.dtype in INT_DTYPES and 0 < numel(self.shape) <= 8

    def is_scalar_shaped(self) -> bool:
        return self.shape == ()

    def __repr__(self):
        return f"K({self.shape},{self.dtype},{self.fmt})"


def parse_tensor_triplet(shapes_raw, dtypes_raw, fmts_raw, warns, ctx, keep_no_dtype_scalars=False):
    """Zip the three profiling columns into KTensor list, dropping DT_UNDEFINED
    placeholders (their shape/format segments are dropped in lockstep).
    Unknown dtypes drop the tensor with a LOUD warning (case-level, review
    needed). With keep_no_dtype_scalars (aclnn/e2e INPUT side), an empty
    shape + empty dtype segment pair is a scalar whose dtype the export lacks
    — kept as a 0-d tensor with dtype None (case-level warning) instead of
    being silently deleted; kernel/geir keep the drop (a materialized const
    would break the op input count). Missing/NULL format segments default to
    ND — the tensor is KEPT (dropping is keyed on dtype only: a missing
    Formats column must not drop every tensor). Unparseable shape segments
    are kept as None with a case-level warning."""
    shapes = split_multi(shapes_raw)
    dtypes = split_multi(dtypes_raw)
    fmts = split_multi(fmts_raw)
    n = max(len(shapes), len(dtypes), len(fmts))
    if len({len(shapes), len(dtypes), len(fmts)} - {0}) > 1:
        warns.append(f"{ctx}: field length mismatch shapes={len(shapes)} dtypes={len(dtypes)} formats={len(fmts)}")
    tensors = []
    unknown = []
    bad_shapes = []
    scalar_no_dtype = []
    for i in range(n):
        shp = shapes[i] if i < len(shapes) else ""
        dt = (dtypes[i] if i < len(dtypes) else "").upper()
        fm = fmts[i] if i < len(fmts) else ""
        if dt in UNDEFINED_DTYPES:
            if keep_no_dtype_scalars and dt == "" and shp in ("", "()"):
                # an empty shape segment marks a scalar slot; an empty dtype
                # segment only means the export lacks that value — keep the
                # scalar (dtype None, warned below), do not delete the input
                if fm in NULL_FORMATS:
                    fm = "ND"
                tensors.append(KTensor((), None, fm))
                scalar_no_dtype.append(i)
            continue  # undefined placeholder
        if dt not in DTYPE_MAP:
            unknown.append(f"'{dt}'@seg{i}")
            continue
        if fm in NULL_FORMATS:
            fm = "ND"  # column absent / empty segment -> default, tensor kept
        shape = parse_shape(shp)
        if shape is None:
            bad_shapes.append(f"'{shp}'@seg{i}")
        tensors.append(KTensor(shape, DTYPE_MAP[dt], fm))
    if unknown:
        warns.append(f"{ctx}: tensor dropped for unknown dtype {','.join(unknown)} — case needs review")
    if bad_shapes:
        warns.append(f"{ctx}: unparseable shape segment {','.join(bad_shapes)} kept as None — case needs review")
    if scalar_no_dtype:
        warns.append(
            f"{ctx}: scalar kept but dtype segment empty (unknown dtype) at seg {','.join(map(str, scalar_no_dtype))} — case needs review"
        )
    if tensors and not fmts:
        raw_fmt = clean_cell(fmts_raw or "").upper()
        if raw_fmt in ("NULL", "N/A", "NONE"):
            # column exists but holds a null value (msprof writes NULL when
            # formats were not captured) — distinct from a missing column
            warns.append(f"{ctx}: formats value is {raw_fmt}, defaulted to ND")
        else:
            warns.append(f"{ctx}: formats column missing/empty, defaulted to ND")
    return tensors


# --------------------------------------------------------------------------- #
# aclnn API resolution & signature info
# --------------------------------------------------------------------------- #


class ApiLib:
    """Thin wrapper over ttk OpApiInfoKeeper; None-safe when ttk/CANN absent."""

    def __init__(self):
        self.available = False
        self.infos = {}
        try:
            logging.disable(logging.ERROR)
            from ttk.core_modules.aclnn.info_keeper import OpApiInfoKeeper  # noqa: PLC0415

            keeper = OpApiInfoKeeper()
            self.infos = keeper.api_info
            self.available = True
        except Exception as err:  # noqa: BLE001
            # ttk or CANN headers unavailable -> degrade to heuristic mode
            logging.debug(f"OpApiInfoKeeper unavailable, fallback mode: {err}")
        finally:
            logging.disable(logging.NOTSET)

    def info_of(self, api):
        return self.infos.get(api)

    def resolve(self, candidates):
        """Pick the longest candidate present in headers. Returns (api, hit_bool)."""
        if not self.available:
            return candidates[0], False
        hits = [c for c in candidates if c in self.infos]
        if hits:
            return max(hits, key=len), True
        return candidates[0], False


def api_candidates(name: str, op_type: str):
    """Profile Name is '{aclnnApi}_{kernelName}_{opType}'. Build resolution candidates."""
    name = clean_cell(name)
    if name.startswith("aclnn"):
        base = name
        if op_type and base.endswith("_" + op_type):
            base = base[: -len(op_type) - 1]
        segs = base.split("_")
        # candidate prefixes: aclnnFoo, aclnnFoo_Bar, ... (longest header hit wins)
        return ["_".join(segs[: i + 1]) for i in range(len(segs))], "aclnn-name"
    t = clean_cell(op_type)
    # only CamelCase op types can map to an aclnn API; snake_case names
    # (_collect_welford_kernel, bf16_hybrid_attention, ...) are torch/custom kernels
    if not t or not t[0].isupper() or "_" in t:
        return [], "non-aclnn-kernel"
    return ["aclnn" + t], "type-guess"


def is_output_param(name: str, api_name: str) -> bool:
    if name in EXTRA_OUTPUT_NAMES or name.endswith("Ref"):
        return True
    is_output_name = name in ("output", "out") or name.endswith(("Out", "OutOptional", "Output", "OutputOptional"))
    if not is_output_name:
        return False
    is_backward = "Backward" in api_name or "Grad" in api_name
    return not (is_backward and name in BACKWARD_OUTPUT_EXCLUDE)


# --------------------------------------------------------------------------- #
# Signature alignment
# --------------------------------------------------------------------------- #


class CaseDraft:
    __slots__ = ("api", "tensors", "out_indexes", "scalar_dtypes", "warns", "remark_bits")

    def __init__(self, api):
        self.api = api
        self.tensors = []  # list of (shape|None, dtype|None, fmt|None, is_list)
        self.out_indexes = []
        self.scalar_dtypes = []
        self.warns = []
        self.remark_bits = []


def align_case(api, info, kins, kouts, row_ctx):
    """Map profile kernel tensors onto the aclnn API signature.

    Strategy: left-align inputs onto non-output tensor params (signature order),
    left-align outputs onto output params; when no param follows the *Out/*Ref
    naming the last tensor param is the output (TTK's (-1,) fallback rule, e.g.
    aclnnGeluV2's y), and surplus kernel outputs right-align onto trailing free
    params (BatchNorm saveMean/saveInvstd, ...WeightNzV2 outputScale). Kernel
    inputs materialize in signature order with a TensorList expanded in place,
    so when a single mid-signature TensorList param is followed by fixed params
    (index_put_'s values) those are reserved from the TAIL of the kernel inputs
    (right-aligned) and the list absorbs the middle; a leading *Ref param whose
    self buffer GE also reports as first kernel input is dropped first (it is
    already covered by the output alignment). Under input surplus, GE-internal
    aux consts (small int tensors) are skipped for non-TensorList slots so they
    cannot steal a slot from a real operand and shift the alignment
    (FastLayerNorm's axis); TensorList slots keep them (aclnnIndex indices).
    Skipped aux backfill still-free params afterwards, surplus ones are dropped
    with an aux-drop remark. Deficit params become None placeholders.
    Scalar-shaped kernel inputs feed aclScalar params (dtype from profiling).
    TensorList params wrap their single matched tensor into a 1-element list.
    """
    d = CaseDraft(api)
    params = info.params
    out_param_pos = [i for i, n in enumerate(info.tensors) if is_output_param(n, api)]
    if not out_param_pos and kouts and info.tensors:
        # no param follows the *Out/*Ref naming -> the last tensor is the output
        # (mirrors TTK's (-1,) auto-fill fallback, e.g. aclnnGeluV2's y)
        out_param_pos = [len(info.tensors) - 1]
    in_param_pos = [i for i, n in enumerate(info.tensors) if i not in out_param_pos]

    # split scalar-shaped inputs: they preferentially feed aclScalar params
    scalar_kins = [t for t in kins if t.is_scalar_shaped()]
    real_kins = [t for t in kins if not t.is_scalar_shaped()]
    # GE-internal aux consts (perm/begin/end/axis lists): under input surplus
    # they are skipped for non-TensorList slots (taking one there shifts every
    # following operand — FastLayerNorm's axis const used to land on
    # weightOptional); they backfill still-free params after alignment
    aux_kins = [t for t in real_kins if t.is_aux()]

    assign = {}  # position in info.tensors -> (list[KTensor], from_output: bool)
    selfbuf_dropped = []  # inplace self buffers dropped from the input pool (Ref out params)

    # outputs -> output params (left align)
    for pos, kt in zip(out_param_pos, kouts):
        assign[pos] = ([kt], True)
    extra_outs = kouts[len(out_param_pos) :]  # placed after input alignment
    if len(out_param_pos) > len(kouts):
        d.warns.append(f"{len(out_param_pos) - len(kouts)} output params unmatched (None)")

    # inputs -> non-output params
    # 'self'-leading accumulation APIs (addmm/baddbmm style) often omit the
    # accumulator in the kernel; a count deficit there means leading params are
    # the missing ones -> right align. Otherwise left align (signature order).
    first_in_name = info.tensors[in_param_pos[0]] if in_param_pos else ""
    self_deficit = False
    if first_in_name == "self" and len(real_kins) == len(in_param_pos):
        # addmm/baddbmm bias-broadcast pattern: a single 1-D kernel input is the
        # accumulation vector -> it belongs to 'self', not to a mat operand
        # (shape may be None for an unparseable profile segment — guard it)
        one_d = [t for t in real_kins if t.shape is not None and len(t.shape) == 1]
        if len(one_d) == 1:
            od = one_d[0]
            rest = [t for t in real_kins if t is not od]
            assign[in_param_pos[0]] = ([od], False)
            for pos, kt in zip(in_param_pos[1:], rest):
                assign[pos] = ([kt], False)
        else:
            for pos, kt in zip(in_param_pos, real_kins):
                assign[pos] = ([kt], False)
    elif len(real_kins) < len(in_param_pos) and first_in_name == "self":
        self_deficit = True
        offset = len(in_param_pos) - len(real_kins)
        for pos, kt in zip(in_param_pos[offset:], real_kins):
            assign[pos] = ([kt], False)
    else:
        # pass 1: positional for plain tensors; bool tensors prefer *mask* params.
        # Under input surplus GE-internal aux consts are skipped for non-
        # TensorList slots — they are the likeliest surplus and must not shift
        # real operands (FastLayerNorm's axis const). TensorList slots keep
        # them: list elements are often small int tensors (aclnnIndex indices).
        # With matching/deficit counts kernel order is trusted as-is.
        bool_kins = [t for t in real_kins if t.dtype == "bool"]
        plain_kins = [t for t in real_kins if t.dtype != "bool"]
        surplus = len(real_kins) > len(in_param_pos)
        tl_in_pos = [p for p in in_param_pos if params[info.tensors[p]]["type"] == "aclTensorList*"]
        pre_fixed = [p for p in in_param_pos if tl_in_pos and p < tl_in_pos[0]]
        post_fixed = [p for p in in_param_pos if tl_in_pos and p > tl_in_pos[0] and p not in tl_in_pos]
        # kernel inputs materialize in signature order with a TensorList
        # expanded in place, so fixed params AFTER a mid-signature list param
        # (index_put_'s values) sit beyond the list members: reserve them from
        # the TAIL of the kernel inputs (right-aligned) — plain left alignment
        # feeds list members to those slots. Single list only: with several
        # lists the middle cannot be split unambiguously.
        tail_align = len(tl_in_pos) == 1 and post_fixed and len(plain_kins) - len(pre_fixed) >= len(post_fixed)
        ki = 0
        if tail_align:
            # inplace *Ref leading param: GE materializes the self buffer as
            # the first kernel input as well (aclnnIndexPutImpl's self) even
            # though the signature routes it through the output side — drop
            # the matching input so it does not pollute the list; only while
            # the remaining inputs still cover the fixed params (a sole
            # matching operand is a real input, not the self buffer)
            if (
                info.tensors[0].endswith("Ref")
                and 0 in out_param_pos
                and kouts
                and plain_kins
                and len(plain_kins) - 1 >= len(pre_fixed) + len(post_fixed)
                and (plain_kins[0].shape, plain_kins[0].dtype, plain_kins[0].fmt)
                == (kouts[0].shape, kouts[0].dtype, kouts[0].fmt)
            ):
                d.remark_bits.append(f"selfbuf-drop:{plain_kins[0]!r}")
                selfbuf_dropped.append(plain_kins[0])
                plain_kins = plain_kins[1:]
            for pos in pre_fixed:
                if ki >= len(plain_kins):
                    break
                if surplus:
                    while ki < len(plain_kins) and plain_kins[ki].is_aux():
                        ki += 1
                    if ki >= len(plain_kins):
                        break
                assign[pos] = ([plain_kins[ki]], False)
                ki += 1
            for pos, kt in zip(post_fixed, plain_kins[len(plain_kins) - len(post_fixed) :]):
                assign[pos] = ([kt], False)
            middle = plain_kins[ki : len(plain_kins) - len(post_fixed)]
            if middle:
                assign[tl_in_pos[0]] = (middle, False)
            d.remark_bits.append(f"tail-align:{','.join(info.tensors[p] for p in post_fixed)}")
        else:
            if post_fixed:
                d.warns.append("post-list fixed params left-aligned, needs spot check")
            for pos in in_param_pos:
                if ki >= len(plain_kins):
                    break
                if surplus and params[info.tensors[pos]]["type"] != "aclTensorList*":
                    while ki < len(plain_kins) and plain_kins[ki].is_aux():
                        ki += 1
                    if ki >= len(plain_kins):
                        break
                assign[pos] = ([plain_kins[ki]], False)
                ki += 1
        for kt in bool_kins:
            mask_pos = next((p for p in in_param_pos if "mask" in info.tensors[p].lower() and p not in assign), None)
            if mask_pos is not None:
                assign[mask_pos] = ([kt], False)
            else:
                free = next((p for p in in_param_pos if p not in assign), None)
                if free is not None:
                    assign[free] = ([kt], False)

    def assigned_kts():
        for kts, _ in assign.values():
            yield from kts

    # surplus kernel outputs -> trailing free params, right-aligned: outputs of
    # APIs that don't follow the *Out naming used to be dropped entirely
    # (BatchNorm saveMean/saveInvstd, GroupedMatmulSwigluQuantWeightNzV2 outputScale)
    if extra_outs:
        free_tail = []
        for pos in reversed(range(len(info.tensors))):
            if pos in assign:
                break
            free_tail.append(pos)
        for pos, kt in zip(free_tail, reversed(extra_outs)):
            assign[pos] = ([kt], True)
        dropped_outs = extra_outs[: max(0, len(extra_outs) - len(free_tail))]
        if dropped_outs:
            d.warns.append(f"extra kernel outputs dropped: {dropped_outs}")

    # aux consts skipped by the surplus-aware fill backfill still-free input
    # params; surplus ones keep no slot and drop quietly in the leftover pass
    for kt in aux_kins:
        if any(kt is a for a in assigned_kts()):
            continue
        free = next((p for p in in_param_pos if p not in assign), None)
        if free is not None:
            assign[free] = ([kt], False)

    # leftover kernel inputs: TensorList params absorb the non-aux ones (concat/
    # grouped inputs), aux const tensors (perm/begin/end/...) are dropped quietly
    leftover_ins = [
        t for t in real_kins if not any(t is a for a in assigned_kts()) and not any(t is x for x in selfbuf_dropped)
    ]
    if leftover_ins:
        tl_pos = next((p for p in in_param_pos if params[info.tensors[p]]["type"] == "aclTensorList*"), None)
        for kt in leftover_ins:
            if kt.is_aux():
                d.remark_bits.append(f"aux-drop:{kt!r}")
            elif tl_pos is not None:
                assign.setdefault(tl_pos, ([], False))[0].append(kt)
            else:
                d.warns.append(f"extra kernel input dropped: {kt!r}")

    # scalar params <- scalar-shaped kernel inputs (positional, dtype from profiling)
    scalar_names = info.scalars
    scalar_hint_dtypes = []
    for i, sname in enumerate(scalar_names):
        if i < len(scalar_kins):
            if scalar_kins[i].dtype is None:
                # scalar survived parsing but the export lacked its dtype segment
                d.warns.append(f"scalar dtype unknown for param '{sname}' (None placeholder)")
            scalar_hint_dtypes.append(scalar_kins[i].dtype)
        else:
            g = guess_scalar_dtype(sname)
            if g is None:
                d.warns.append(f"scalar dtype unknown for param '{sname}' (None placeholder)")
            scalar_hint_dtypes.append(g)
    if len(scalar_kins) > len(scalar_names):
        # leftover scalar-shaped inputs fall back to unfilled tensor params
        rest = scalar_kins[len(scalar_names) :]
        for pos in in_param_pos:
            if pos not in assign and rest:
                assign[pos] = ([rest.pop(0)], False)
        for kt in rest:
            d.remark_bits.append(f"aux-drop:{kt!r}")

    # unmatched in-params -> None; counted after every backfill pass (aux /
    # scalar overflow) so refilled slots are not reported as holes, and the
    # 'self'-style deficit is reported once instead of twice
    n_in_assigned = sum(1 for p in in_param_pos if p in assign)
    if in_param_pos and n_in_assigned < len(in_param_pos):
        if self_deficit:
            d.warns.append(
                f"{len(in_param_pos) - n_in_assigned} leading input params unmatched (None) — 'self' style right-aligned"
            )
        elif in_param_pos[0] not in assign:
            d.warns.append("first input param unmatched (None) — alignment may be shifted")
        else:
            d.warns.append(f"{len(in_param_pos) - n_in_assigned} optional input params unmatched (None)")

    # outputs whose profile shape segment was empty stay 0-d — flag for review
    for kts, from_out in assign.values():
        if from_out and kts and kts[0].is_scalar_shaped():
            d.warns.append(f"empty output shape in profile (kept 0-d): {kts[0]!r}")

    # build ordered tensor list following signature order
    out_idx_set = set()
    for pos, pname in enumerate(info.tensors):
        ptype = params[pname]["type"]
        is_list = ptype == "aclTensorList*"
        hit = assign.get(pos)
        if hit is None:
            d.tensors.append((None, None, None, is_list))
            continue
        kts, from_out = hit
        if is_list:
            d.tensors.append(
                (tuple(t.shape for t in kts), tuple(t.dtype for t in kts), tuple(t.fmt for t in kts), is_list)
            )
        else:
            kt = kts[0]
            d.tensors.append((kt.shape, kt.dtype, kt.fmt, is_list))
        if from_out:
            out_idx_set.add(pos)
    # *Ref params must be in output indexes (inplace semantics)
    for pos, pname in enumerate(info.tensors):
        if pname.endswith("Ref") and d.tensors[pos][0] is not None:
            out_idx_set.add(pos)
    d.out_indexes = sorted(out_idx_set)
    d.scalar_dtypes = scalar_hint_dtypes
    return d


def draft_unresolved(api, kins, kouts):
    """No header signature: keep profile order (inputs then outputs), no indexes."""
    d = CaseDraft(api)
    for kt in kins:
        d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))
    for kt in kouts:
        d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))
        d.out_indexes.append(len(d.tensors) - 1)
    d.warns.append("API_NOT_RESOLVED: signature unavailable, tensors kept in kernel order")
    return d


# --------------------------------------------------------------------------- #
# Kernel / GEIR mode conversion
# --------------------------------------------------------------------------- #


def camel_to_snake(name: str) -> str:
    """CamelCase GE op type -> snake op name (same rule as ttk.utilities.string)."""
    try:
        from ttk.utilities.string import camel_to_snake as _c2s  # noqa: PLC0415

        return _c2s(name)
    except Exception:
        snake = ""
        sub_head = False
        for idx, ch in enumerate(name):
            if ch.islower():
                sub_head = False
            if ch.isdigit():
                sub_head = True
            if ch.isupper() and idx != 0:
                if not sub_head:
                    snake += "_"
                    sub_head = True
                elif idx + 1 < len(name) and name[idx + 1].islower():
                    snake += "_"
            snake += ch
        return snake.lower()


def is_camel_op_type(op_type: str) -> bool:
    t = clean_cell(op_type)
    return bool(t) and t[0].isupper() and "_" not in t


class KernelOpLib:
    """OpInfoKeeper wrapper; None-safe without ttk / NPU / ops-info (op names
    are then derived via camel_to_snake and left for selfcheck to flag)."""

    def __init__(self):
        self.available = False
        self.keeper = None
        try:
            logging.disable(logging.ERROR)
            from ttk.core_modules.kernel.info_keeper import OpInfoKeeper  # noqa: PLC0415

            keeper = OpInfoKeeper()
            keeper.ops_info  # noqa: B018  force load; raises without SOC/opp
            self.keeper = keeper
            self.available = True
        except Exception as err:  # noqa: BLE001
            logging.debug(f"OpInfoKeeper unavailable, op names derived only: {err}")
        finally:
            logging.disable(logging.NOTSET)

    def op_name_of(self, op_type: str):
        """Returns (op_name, resolved_bool); prefers ops-info opInterface name.

        Both camel_to_snake and the acronym-aware variant are candidates:
        ttk's camel_to_snake splits leading acronym runs differently from the
        official op build-tool naming rule (asc-devkit naming_conversion_table:
        FFNToAttention -> ffn_to_attention, GRUCell -> gru_cell), and real
        custom ops register the acronym-kept name (ops-transformer
        mc2/ffn_to_attention). With ops-info, either variant resolves via the
        keeper (alias index tolerates underscore differences); without it the
        acronym variant follows the official rule for the derived name."""
        t = clean_cell(op_type)
        variants = list(dict.fromkeys([camel_to_snake(t), _acronym_snake(t)]))
        if self.available:
            for snake in variants:
                info = self.keeper.info_of(snake)
                if info:
                    return info.get("opInterface.value") or snake, True
        return variants[-1], False

    def value_depend_inputs(self, op_name: str):
        """Op inputs whose values are needed at compile time (valueDepend)."""
        if not self.available:
            return None
        info = self.keeper.info_of(op_name)
        if not info:
            return None
        return [i["name"] for i in info.get("inputs", []) if str(i.get("valueDepend", "ignore")).lower() != "ignore"]

    def required_attrs(self, op_name: str):
        if not self.available:
            return None
        info = self.keeper.info_of(op_name)
        if not info:
            return None
        return {a.get("name"): a.get("type") for a in info.get("attr", []) if a.get("defaultValue") is None}


def emit_kernel_row(op_name, kins, kouts, name, remark, geir=False):
    """Kernel/GEIR row: kernel I/O verbatim — GE-internal aux consts (perm/
    begin/end/axis) are real op inputs here and are KEPT, unlike aclnn mode."""
    in_fmts = tuple(t.fmt for t in kins)
    out_fmts = tuple(t.fmt for t in kouts)
    row = {
        "testcase_name": name,
        "op_name": op_name,
        "input_shapes": pyobj(tuple(t.shape for t in kins)),
        "input_dtypes": pyobj(tuple(t.dtype for t in kins)),
        "input_formats": pyobj(in_fmts) if any(f not in (None, "ND") for f in in_fmts) else "",
        "output_shapes": pyobj(tuple(t.shape for t in kouts)),
        "output_dtypes": pyobj(tuple(t.dtype for t in kouts)),
        "output_formats": pyobj(out_fmts) if any(f not in (None, "ND") for f in out_fmts) else "",
        "attributes": "",
    }
    if geir:
        row["dyn_input_shapes"] = ""
    row["remark"] = remark
    return row


# --------------------------------------------------------------------------- #
# E2E mode conversion (aclnn API -> torch API)
# --------------------------------------------------------------------------- #

# aclnn API -> (torch API, note). note == "" means faithful mapping; a note
# marks approximate mappings (semantics/defaults may differ — user reviews).
# Extend this table to cover more APIs; unmapped ones are skipped and reported.
ACLNN_TO_TORCH = {
    "aclnnAdd": ("torch.add", ""),
    "aclnnAdds": ("torch.add", "kernel 标量 other，值需补 attributes"),
    "aclnnSub": ("torch.sub", ""),
    "aclnnSubs": ("torch.sub", "kernel 标量 other，值需补 attributes"),
    "aclnnMul": ("torch.mul", ""),
    "aclnnMuls": ("torch.mul", "kernel 标量 other，值需补 attributes"),
    "aclnnDiv": ("torch.div", ""),
    "aclnnAbs": ("torch.abs", ""),
    "aclnnNeg": ("torch.neg", ""),
    "aclnnExp": ("torch.exp", ""),
    "aclnnLog": ("torch.log", ""),
    "aclnnSqrt": ("torch.sqrt", ""),
    "aclnnRsqrt": ("torch.rsqrt", ""),
    "aclnnSigmoid": ("torch.sigmoid", ""),
    "aclnnSilu": ("torch.nn.functional.silu", ""),
    "aclnnSin": ("torch.sin", ""),
    "aclnnCos": ("torch.cos", ""),
    "aclnnPowTensorScalar": ("torch.pow", "exponent 标量，值需补 attributes"),
    "aclnnReciprocal": ("torch.reciprocal", ""),
    "aclnnMaximum": ("torch.maximum", ""),
    "aclnnMinimum": ("torch.minimum", ""),
    "aclnnBitwiseAnd": ("torch.bitwise_and", ""),
    "aclnnBitwiseAndTensor": ("torch.bitwise_and", ""),
    "aclnnBitwiseAndScalar": ("torch.bitwise_and", "kernel 标量 other，值需补 attributes"),
    "aclnnBitwiseOr": ("torch.bitwise_or", ""),
    "aclnnBitwiseXor": ("torch.bitwise_xor", ""),
    "aclnnBitwiseNot": ("torch.bitwise_not", ""),
    "aclnnLeftShift": ("torch.bitwise_left_shift", ""),
    "aclnnRightShift": ("torch.bitwise_right_shift", ""),
    "aclnnEq": ("torch.eq", ""),
    "aclnnEqScalar": ("torch.eq", "kernel 标量 other，值需补 attributes"),
    "aclnnNe": ("torch.ne", ""),
    "aclnnNeScalar": ("torch.ne", "kernel 标量 other，值需补 attributes"),
    "aclnnLt": ("torch.lt", ""),
    "aclnnLtScalar": ("torch.lt", "kernel 标量 other，值需补 attributes"),
    "aclnnLe": ("torch.le", ""),
    "aclnnLeScalar": ("torch.le", "kernel 标量 other，值需补 attributes"),
    "aclnnGt": ("torch.gt", ""),
    "aclnnGtScalar": ("torch.gt", "kernel 标量 other，值需补 attributes"),
    "aclnnGe": ("torch.ge", ""),
    "aclnnGeScalar": ("torch.ge", "kernel 标量 other，值需补 attributes"),
    "aclnnZerosLike": ("torch.zeros_like", ""),
    "aclnnOnesLike": ("torch.ones_like", ""),
    "aclnnInplaceCopy": ("torch.Tensor.copy_", ""),
    "aclnnInplaceZero": (
        "torch.ops.aten.zero_.default",
        "aten 重载（torch.Tensor.zero_ 签名无显式 self，keeper 不可解析）",
    ),
    "aclnnInplaceFillScalar": ("torch.Tensor.fill_", "value 必填"),
    "aclnnInplaceMaskedFillScalar": ("torch.Tensor.masked_fill_", "value 必填"),
    "aclnnInplaceMaskedScatter": ("torch.Tensor.masked_scatter_", ""),
    "aclnnCat": ("torch.cat", "dim 必填"),
    "aclnnIndexSelect": ("torch.index_select", "dim 必填"),
    "aclnnScatterAdd": ("torch.scatter_add", "dim/idx 语义需核对"),
    "aclnnSort": ("torch.sort", "dim/descending 默认值可能与业务不同"),
    "aclnnNonzeroV2": ("torch.nonzero", ""),
    "aclnnArange": ("torch.arange", "start/end 必填"),
    "aclnnMean": ("torch.mean", "dim/keepdim 需核对"),
    "aclnnTopK": ("torch.topk", "k 必填"),
    "aclnnEmbedding": ("torch.embedding", ""),
    "aclnnScatterNdUpdate": ("torch_npu.npu_scatter_nd_update_", ""),
    "aclnnTransposeBatchMatMul": ("torch_npu.npu_transpose_batchmatmul", ""),
    "aclnnTransposeBatchMatMulWeightNz": ("torch_npu.npu_transpose_batchmatmul", "权重为 NZ 布局，torch 侧常规 ND"),
    "aclnnTransposeQuantBatchMatMul": ("torch_npu.npu_transpose_quant_batchmatmul", ""),
    "aclnnConvertWeightToINT4Pack": ("torch_npu.npu_convert_weight_to_int4pack", ""),
    "aclnnGeGlu": ("torch_npu.npu_geglu", ""),
    "aclnnGeGluV3": ("torch_npu.npu_geglu", ""),
    # ---- 常规算子的标准 torch 对应（GE 名与 torch 名差异较大，经签名核实）----
    "aclnnMaxPool": ("torch.nn.functional.max_pool2d", "kernel_size/stride/padding 必填"),
    "aclnnMaxPoolV3": ("torch.nn.functional.max_pool2d", "kernel_size/stride/padding 必填"),
    "aclnnUnique2": ("torch.unique", ""),
    "aclnnUniqueDim": ("torch.unique", "dim/sorted/return_inverse 语义需核对"),
    "aclnnLogSigmoid": ("torch.nn.functional.logsigmoid", ""),
    "aclnnLogSigmoidForward": ("torch.nn.functional.logsigmoid", "GE 前向变体，语义一致"),
    "aclnnSwish": ("torch.nn.functional.silu", "swish 即 silu（x·sigmoid(x)）"),
    "aclnnShrink": ("torch.nn.functional.softshrink", "lambd 必填"),
    "aclnnBatchMatMulWeightNz": ("torch.bmm", "权重为 NZ 布局，torch 侧常规 ND"),
    "aclnnGemm": ("torch.mm", "trans_a/trans_b/alpha/beta 需核对"),
    "aclnnIndexPutImpl": ("torch.Tensor.index_put_", ""),
    "aclnnIndexFillTensor": ("torch.Tensor.index_fill_", ""),
    "aclnnRepeatInterleaveTensor": ("torch.repeat_interleave", ""),
    "aclnnRepeatInterleaveInt": ("torch.repeat_interleave", "repeats 标量值需补 attributes"),
    "aclnnRepeatInterleaveWithDim": ("torch.repeat_interleave", "dim 需核对"),
    "aclnnRepeatInterleaveIntWithDim": ("torch.repeat_interleave", "repeats 标量值需补 attributes"),
    "aclnnLinalgVectorNorm": ("torch.linalg.vector_norm", "ord/dim 需核对"),
    "aclnnMseLossOut": ("torch.nn.functional.mse_loss", "reduction 默认值需核对"),
    "aclnnNLLLoss2d": ("torch.nn.functional.nll_loss", "2D 特例语义一致，参数需核对"),
    "aclnnMedianDim": ("torch.median", "dim/keepdim 需核对"),
    "aclnnNanMedian": ("torch.nanmedian", ""),
    "aclnnNanMedianDim": ("torch.nanmedian", "dim/keepdim 需核对"),
    "aclnnRReluWithNoise": ("torch.nn.functional.rrelu", "lower/upper 必填，训练态噪声不可复现"),
    "aclnnKvRmsNormRopeCache": ("torch_npu.npu_kv_rmsnorm_rope_cache", ""),
    "aclnnKvRmsNormRopeCacheV2": ("torch_npu.npu_kv_rmsnorm_rope_cache_v2", ""),
    "aclnnGroupedMatmulFinalizeRoutingWeightNzV2": (
        "torch_npu.npu_grouped_matmul_finalize_routing",
        "权重 NZ 布局 + V2 版本，参数落位需抽查",
    ),
    "aclnnAddmm": ("torch.addmm", ""),
    "aclnnMm": ("torch.mm", ""),
    "aclnnMatmul": ("torch.matmul", ""),
    "aclnnBatchMatMul": ("torch.bmm", ""),
    "aclnnGeluV2": ("torch.nn.functional.gelu", "approximate 默认值可能与业务不同"),
    "aclnnFastLayerNorm": ("torch.nn.functional.layer_norm", "normalized_shape 必填"),
    "aclnnRmsNorm": ("torch_npu.npu_rms_norm", ""),
    "aclnnBatchNorm": ("torch.nn.functional.batch_norm", "training/running stats 语义需核对"),
    "aclnnGroupNorm": ("torch.nn.functional.group_norm", "num_groups 必填"),
    "aclnnConvolution": ("torch.nn.functional.conv2d", "stride/padding/dilation/groups 必填"),
    "aclnnUpsampleBilinear2d": ("torch.nn.functional.interpolate", "size/scale_factor/mode='bilinear' 必填"),
    "aclnnConstantPadNd": ("torch.nn.functional.pad", "pad 列表/mode='constant' 必填"),
    "aclnnSplit": ("torch.split", "split_size_or_sections 必填"),
    "aclnnSoftmax": ("torch.softmax", "dim 必填"),
    # ---- torch_npu engine ops (mapping verified via torch_npu ge-converters / plugin docs) ----
    "aclnnDynamicMxQuant": ("torch_npu.npu_dynamic_mx_quant", ""),
    "aclnnRmsNormDynamicMxQuant": ("torch_npu.npu_rms_norm_dynamic_mx_quant", ""),
    "aclnnSwigluMxQuant": ("torch_npu.npu_swiglu_mx_quant", ""),
    "aclnnAddRmsNorm": ("torch_npu.npu_add_rms_norm", ""),
    "aclnnAddRmsNormCast": ("torch_npu.npu_add_rms_norm_cast", ""),
    "aclnnSwiGlu": ("torch_npu.npu_swiglu", ""),
    "aclnnMoeGatingTopK": ("torch_npu.npu_moe_gating_top_k", "k 必填"),
    "aclnnMoeTokenUnpermute": ("torch_npu.npu_moe_token_unpermute", ""),
    "aclnnMoeFinalizeRoutingV2": ("torch_npu.npu_moe_finalize_routing", ""),
    "aclnnFusedInferAttentionScore": ("torch_npu.npu_fused_infer_attention_score", "num_heads 等必填参数需补"),
    "aclnnSimThreadExponential": ("torch_npu.npu_sim_exponential_", ""),
    "aclnnIndex": ("torch.ops.aten.index.Tensor", "indices 为 TensorList，吸收全部索引张量"),
    "aclnnMoeInitRoutingV3": ("torch_npu.npu_moe_init_routing_v2", "torch API 名为 v2（GE 算子为 V3），语义对应需核对"),
    "aclnnQkvRmsNormRopeCacheV2": ("torch_npu.npu_qkv_rms_norm_rope_cache", "V2 与 torch 版本对应关系需核对"),
    "aclnnFlashAttentionScoreV4": ("torch_npu.npu_fusion_attention", "FA 版本由入参分派，默认值可能与业务不同"),
    "aclnnTopKTopPSampleV2": ("torch_npu.npu_top_k_top_p_sample", "V2 映射到基础版 API，语义需核对"),
    # ---- FlashAttention 家族（torch 侧名为 fusion_attention，经 torch_npu ge-converter 核实） ----
    "aclnnFlashAttentionScore": ("torch_npu.npu_fusion_attention", "head_num/input_layout 等必填参数需补"),
    "aclnnFlashAttentionScoreV2": ("torch_npu.npu_fusion_attention_v2", "必填参数需补"),
    "aclnnFlashAttentionScoreV3": ("torch_npu.npu_fusion_attention_v3", "必填参数需补"),
    "aclnnFlashAttentionScoreGrad": ("torch_npu.npu_fusion_attention_grad", ""),
    "aclnnFlashAttentionScoreGradV2": ("torch_npu.npu_fusion_attention_grad_v2", ""),
    "aclnnFlashAttentionScoreGradV3": ("torch_npu.npu_fusion_attention_grad_v3", ""),
    "aclnnFlashAttentionScoreGradV4": ("torch_npu.npu_fusion_attention_grad", "V4 映射到基础版 API，语义需核对"),
    "aclnnNsaSelectedAttention": (
        "torch_npu.npu_nsa_select_attention",
        "torch API 名为 select（无 -ed 后缀），语义一致",
    ),
    "aclnnNsaSelectedAttentionGrad": (
        "torch_npu.npu_nsa_select_attention_grad",
        "torch API 名为 select（无 -ed 后缀），语义一致",
    ),
    "aclnnNsaSelectedAttentionInfer": (
        "torch_npu.npu_nsa_select_attention_infer",
        "torch API 名为 select（无 -ed 后缀），语义一致",
    ),
    # ---- WeightNz 量化矩阵乘族（torch 侧同一 API，NZ 布局由权重张量携带；用法经 cann-recipes 网络核实） ----
    "aclnnQuantMatmulWeightNz": (
        "torch_npu.npu_quant_matmul",
        "权重为 NZ 布局（4-D shape），torch 侧常规 ND；scale/pertoken_scale 落位需抽查",
    ),
    "aclnnGroupedMatmulWeightNz": (
        "torch_npu.npu_grouped_matmul",
        "权重为 NZ 布局；列表参数按 kernel 序单元素填充，需抽查",
    ),
    "aclnnGroupedMatmulSwigluQuantWeightNzV2": (
        "torch_npu.npu_grouped_matmul_swiglu_quant_v2",
        "权重为 NZ 布局；参数落位需抽查",
    ),
    # ---- TP 通信融合 matmul 族（torch 侧 base API，用法经 cann-recipes qwen3_5/hunyuan-image 核实）----
    "aclnnMatmulAllReduce": ("torch_npu.npu_mm_all_reduce_base", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulAllReduceV2": ("torch_npu.npu_mm_all_reduce_base", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulAllReduceV3": ("torch_npu.npu_mm_all_reduce_base", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAllGatherMatmul": ("torch_npu.npu_all_gather_base_mm", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAllGatherMatmulV2": ("torch_npu.npu_all_gather_base_mm", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulReduceScatter": ("torch_npu.npu_mm_reduce_scatter_base", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulReduceScatterV2": ("torch_npu.npu_mm_reduce_scatter_base", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAlltoAllMatmul": ("torch_npu.npu_all_to_all_matmul", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAlltoAllMatmulBase": ("torch_npu.npu_all_to_all_matmul", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAlltoAllMatmulV2": ("torch_npu.npu_all_to_all_matmul", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulAlltoAll": ("torch_npu.npu_matmul_all_to_all", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulAlltoAllBase": ("torch_npu.npu_matmul_all_to_all", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnMatmulAlltoAllV2": ("torch_npu.npu_matmul_all_to_all", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnQuantMatmulAlltoAll": ("torch_npu.npu_quant_matmul_all_to_all", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnQuantMatmulAlltoAllBase": (
        "torch_npu.npu_quant_matmul_all_to_all",
        "TP 通信融合；hcom 等分布式参数需人工补值",
    ),
    "aclnnQuantMatmulAlltoAllV2": ("torch_npu.npu_quant_matmul_all_to_all", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAlltoAllQuantMatmul": ("torch_npu.npu_all_to_all_quant_matmul", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAlltoAllQuantMatmulBase": (
        "torch_npu.npu_all_to_all_quant_matmul",
        "TP 通信融合；hcom 等分布式参数需人工补值",
    ),
    "aclnnAlltoAllQuantMatmulV2": ("torch_npu.npu_all_to_all_quant_matmul", "TP 通信融合；hcom 等分布式参数需人工补值"),
    "aclnnAlltoAllvQuantGroupedMatMul": (
        "torch_npu.npu_alltoallv_quant_gmm",
        "TP 通信融合；hcom 等分布式参数需人工补值",
    ),
    "aclnnAlltoAllvQuantGroupedMatMulV2": (
        "torch_npu.npu_alltoallv_quant_gmm",
        "TP 通信融合；hcom 等分布式参数需人工补值",
    ),
    "aclnnQuantGroupedMatMulAlltoAllv": (
        "torch_npu.npu_quant_gmm_alltoallv",
        "TP 通信融合；hcom 等分布式参数需人工补值",
    ),
    "aclnnQuantGroupedMatMulAlltoAllvV2": (
        "torch_npu.npu_quant_gmm_alltoallv",
        "TP 通信融合；hcom 等分布式参数需人工补值",
    ),
    # ---- 量化路由族（同一 torch API，量化/NZ 布局由输入携带，值层参数需人工补）----
    "aclnnAscendQuant": ("torch_npu.npu_quantize", "量化路由变体；scale/offset 等值需人工补"),
    "aclnnAscendQuantV3": ("torch_npu.npu_quantize", "量化路由变体；scale/offset 等值需人工补"),
    "aclnnAscendAntiQuant": ("torch_npu.npu_anti_quant", "反量化路由变体；scale/offset 等值需人工补"),
    "aclnnWeightQuantBatchMatmul": ("torch_npu.npu_weight_quant_batchmatmul", "量化路由变体；参数落位需抽查"),
    "aclnnWeightQuantBatchMatmulV2": ("torch_npu.npu_weight_quant_batchmatmul", "量化路由变体；参数落位需抽查"),
    "aclnnWeightQuantBatchMatmulV3": ("torch_npu.npu_weight_quant_batchmatmul", "量化路由变体；参数落位需抽查"),
    "aclnnWeightQuantBatchMatmulNz": (
        "torch_npu.npu_weight_quant_batchmatmul",
        "量化 + NZ 布局路由变体；参数落位需抽查",
    ),
    "aclnnMatmulCompress": ("torch_npu.npu_matmul_compress_dequant", "压缩路由变体；参数落位需抽查"),
    "aclnnQuantConvolution": ("torch_npu.npu_quant_conv2d", "量化路由变体；stride/padding 等值需人工补"),
    "aclnnQuantConvolutionWeightNz": ("torch_npu.npu_quant_conv2d", "量化 + NZ 布局路由变体；参数落位需抽查"),
    "aclnnFusedQuantMatmul": ("torch_npu.npu_quant_matmul", "量化路由变体；参数落位需抽查"),
    "aclnnFusedQuantMatmulWeightNz": ("torch_npu.npu_quant_matmul", "量化 + NZ 布局路由变体；参数落位需抽查"),
    "aclnnBatchMatmulQuant": ("torch_npu.npu_quant_matmul", "量化路由变体；参数落位需抽查"),
    "aclnnInnerQuantFlashAttn": ("torch_npu.npu_quant_fusion_attention", "引擎内部实现变体，同上层 API"),
    "aclnnQuantFlashAttentionScoreGrad": (
        "torch_npu.npu_quant_fusion_attention_backward",
        "量化路由变体；参数落位需抽查",
    ),
    # ---- Inner/Unpadding 内部路由变体（同一上层 API 的引擎实现/布局路由）----
    "aclnnInnerFusedInferAttentionScore": ("torch_npu.npu_fused_infer_attention_score", "引擎内部实现变体，同上层 API"),
    "aclnnInnerFusedInferAttentionScoreTensor": (
        "torch_npu.npu_fused_infer_attention_score",
        "引擎内部实现变体，同上层 API",
    ),
    "aclnnFlashAttentionUnpaddingScoreGrad": (
        "torch_npu.npu_fusion_attention_grad",
        "非 padding 布局路由变体，同上层 API",
    ),
    "aclnnFlashAttentionUnpaddingScoreGradV2": (
        "torch_npu.npu_fusion_attention_grad",
        "非 padding 布局路由变体，同上层 API",
    ),
    "aclnnFlashAttentionUnpaddingScoreGradV3": (
        "torch_npu.npu_fusion_attention_grad",
        "非 padding 布局路由变体，同上层 API",
    ),
    "aclnnFlashAttentionUnpaddingScoreGradV4": (
        "torch_npu.npu_fusion_attention_grad",
        "非 padding 布局路由变体，同上层 API",
    ),
    "aclnnFlashAttentionUnpaddingScoreGradV5": (
        "torch_npu.npu_fusion_attention_grad",
        "非 padding 布局路由变体，同上层 API",
    ),
}

# torch API -> kernel 物化序的参数填充顺序。这些 API 的 torch 签名序与 kernel I/O 物化序
# 不一致（量化/GMM 流，填充序按 cann-recipes-infer 网络调用用法核实）：real kernel 输入
# 按此顺序落到「参数名」上，列表参数包装为单元素列表；签名序输出。不在表内的 API 走
# 位置左对齐。
E2E_PARAM_FILL_ORDER = {
    "torch_npu.npu_quant_matmul": ["x1", "x2", "scale", "pertoken_scale", "bias"],
    "torch_npu.npu_grouped_matmul": ["x", "weight", "scale", "group_list", "per_token_scale", "bias"],
    "torch_npu.npu_grouped_matmul_swiglu_quant_v2": [
        "x",
        "x_scale",
        "group_list",
        "weight",
        "weight_scale",
        "smooth_scale",
        "bias",
    ],
}


class E2eApiLib:
    """FrameworkApiInfoKeeper wrapper; None-safe without ttk/torch."""

    def __init__(self):
        self.available = False
        self.keeper = None
        try:
            logging.disable(logging.ERROR)
            from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper  # noqa: PLC0415

            self.keeper = FrameworkApiInfoKeeper()
            self.available = True
        except Exception as err:  # noqa: BLE001
            logging.debug(f"FrameworkApiInfoKeeper unavailable: {err}")
        finally:
            logging.disable(logging.NOTSET)

    def info_of(self, api):
        return self.keeper.get(api) if self.available else None

    def register_schemas(self, torch_schemas):
        """Register torch.ops schemas parsed from operator library wrappers.

        Keeps alignment and selfcheck working without the extension package
        built/installed; the keeper cache serves these api_names directly and
        no package import is attempted."""
        if not self.available:
            return 0
        try:
            from ttk.core_modules.framework_api.api_model import ParamInfo  # noqa: PLC0415
        except Exception as err:  # noqa: BLE001
            logging.debug(f"ParamInfo unavailable, op-lib schemas not registered: {err}")
            return 0
        registered = 0
        for api, schemas in torch_schemas.items():
            overloads = []
            for schema in schemas:
                _name, params = parse_torch_schema(schema)
                if params:
                    overloads.append([ParamInfo(**p) for p in params])
            if not overloads:
                continue
            try:
                self.keeper.register(api, overloads, source="op-lib schema")
                registered += 1
            except Exception as err:  # noqa: BLE001
                logging.debug(f"register schema for {api} failed: {err}")
        return registered


def draft_e2e(torch_api, info, kins, kouts, note):
    """Align kernel tensors onto the torch API signature.

    Real kernel inputs left-align onto non-out tensor params; under input
    surplus GE-internal aux consts are skipped for fixed params and backfill
    the still-free ones (leftovers drop with an aux-drop remark). List /
    var-positional params (torch.cat tensors, index_put_ indices) absorb ALL
    leftover inputs — aux kept: list members are often small int tensors —
    and the slot is emitted at its SIGNATURE position (TTK maps CSV slots
    onto tensor params positionally). Kernel inputs materialize in signature
    order with a TensorList expanded in place, so when a single
    mid-signature list param is followed by fixed params (index_put_'s
    values) those are reserved from the TAIL of the kernel inputs
    (right-aligned) and the list absorbs the middle. Inplace Tensor methods
    take the kernel output as `self` when the real inputs do not cover the
    FIXED params (a list-param deficit means the kernel did report self).
    Outputs are compared via the API return value (no out= variant
    generated); the exception is a REQUIRED `out` param — TTK validation
    demands output_tensor_indexes there, so kernel outputs fill out= slots.
    Scalar-shaped kernel inputs map to Number params -> value layer (empty).
    APIs listed in E2E_PARAM_FILL_ORDER bypass positional alignment: kernel
    inputs land on params by name in kernel-materialization order (quant/GMM
    flows whose signature order differs, verified against cann-recipes).
    """
    d = CaseDraft(torch_api)
    if note:
        d.remark_bits.append(f"approx-map:{note}")
    if info is None:
        for kt in kins:
            d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))
        d.warns.append("E2E_SIGNATURE_UNAVAILABLE: torch API 签名不可用，张量按 kernel 顺序保留")
        return d
    real_kins = [t for t in kins if not t.is_scalar_shaped()]
    for t in kins:
        if t.is_scalar_shaped():
            # scalar kernel inputs map to Number params whose values come from
            # attributes (never fabricated) — they cannot become tensor slots,
            # so record the drop in remark + warn for traceability
            d.remark_bits.append(f"scalar-drop:{t!r}")
            d.warns.append(f"scalar-shaped input dropped (Number param, value unrecoverable): {t!r}")
    dist = list(getattr(info, "tensor_distribution", None) or ())
    in_typed = [(i, t) for i, t in enumerate(info.tensors) if t.name != "out"]

    def is_list_param(idx, t):
        # var-positional params (torch.cat tensors) — distribution < 0, or a
        # non-Tensor param type when the distribution is unavailable
        if idx < len(dist):
            return dist[idx] < 0
        return t.is_var_positional or t.type != "Tensor"

    var_idx = var_pos = None
    for i, t in in_typed:
        if is_list_param(i, t):
            var_idx, var_pos = i, t
            break
    fixed = [t for i, t in in_typed if not is_list_param(i, t)]
    first_fixed_idx = next((i for i, t in in_typed if not is_list_param(i, t)), None)

    fill_order = E2E_PARAM_FILL_ORDER.get(torch_api)
    assigned = {}  # in_typed index -> KTensor | [KTensor] (list slot payload)
    if fill_order:
        # curated path for APIs whose torch signature order differs from the
        # kernel I/O materialization order (quant/GMM flows, usage verified
        # against cann-recipes networks): real inputs land on params BY NAME in
        # kernel order, list params wrap their single tensor. Slots are emitted
        # in signature order with None placeholders for the gaps in between —
        # TTK maps CSV slots onto tensor params POSITIONALLY at call time.
        # The aux heuristic does NOT apply on this curated path: the kernel
        # order is manually verified and small int tensors are business
        # inputs here (a group_list with <= 7 experts is int64, numel <= 8)
        by_name = OrderedDict((t.name, (i, t)) for i, t in in_typed)
        queue = list(real_kins)
        for pname in fill_order:
            if pname in by_name and queue:
                assigned[pname] = queue.pop(0)
        for t in queue:
            d.warns.append(f"extra kernel input dropped: {t!r}")
        filled_idx = [i for i, t in in_typed if t.name in assigned]
        if filled_idx:
            last = max(filled_idx)
            for i, t in in_typed:
                if i > last:
                    break
                kt = assigned.get(t.name)
                if kt is None:
                    d.tensors.append((None, None, None, False))
                elif is_list_param(i, t):
                    d.tensors.append(((kt.shape,), (kt.dtype,), (kt.fmt,), True))
                else:
                    d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))
    elif info.is_inplace and kouts:
        # Tensor method (xxx_): GE may or may not report `self` as a kernel
        # input (copy_ reports only src; masked_fill_ reports self too). The
        # `self` slot takes the kernel output when the real (non-const)
        # inputs do not cover the FIXED params — a deficit coming from list
        # params means the kernel DID report self (index_put_'s indices).
        n_real = sum(1 for t in real_kins if not t.is_aux())
        if n_real < len(fixed) and fixed and fixed[0].name == "self" and first_fixed_idx is not None:
            assigned[first_fixed_idx] = kouts[0]
            fixed = fixed[1:]
            if len(kouts) > 1:
                d.warns.append(f"extra kernel outputs dropped (return value used): {kouts[1:]}")

    if fill_order is None:
        surplus = len(real_kins) > len(fixed) and var_pos is None
        aux_skipped = []
        ki = 0
        # fixed (non-list) fill targets in signature order; list params and
        # slots already taken by the inplace branch above are excluded
        fill_targets = [i for i, t in in_typed if not is_list_param(i, t) and i not in assigned]
        n_list_params = sum(1 for i, t in in_typed if is_list_param(i, t))
        post_fixed = [i for i in fill_targets if var_pos is not None and i > var_idx]
        pre_fixed = [i for i in fill_targets if var_pos is None or i < var_idx]
        # kernel inputs materialize in signature order with a TensorList
        # expanded in place, so fixed params AFTER a mid-signature list param
        # (index_put_'s values) sit beyond the list members: reserve them from
        # the TAIL of the kernel inputs (right-aligned) — plain left alignment
        # feeds list members to those slots. Single list only: with several
        # lists the middle cannot be split unambiguously.
        tail_align = (
            var_pos is not None
            and n_list_params == 1
            and post_fixed
            and len(real_kins) - len(pre_fixed) >= len(post_fixed)
        )
        for i, t in in_typed:
            if is_list_param(i, t) or i in assigned:
                continue
            if tail_align and i > var_idx:
                continue  # filled from the reserved tail below
            if ki >= len(real_kins):
                break
            if surplus:
                while ki < len(real_kins) and real_kins[ki].is_aux():
                    aux_skipped.append(real_kins[ki])
                    ki += 1
                if ki >= len(real_kins):
                    break
            assigned[i] = real_kins[ki]
            ki += 1
        if tail_align:
            for i, kt in zip(post_fixed, real_kins[len(real_kins) - len(post_fixed) :]):
                assigned[i] = kt
            d.remark_bits.append(f"tail-align:{','.join(dict(in_typed)[i].name for i in post_fixed)}")
        elif post_fixed:
            d.warns.append("post-list fixed params left-aligned, needs spot check")
        # skipped aux backfill remaining unfilled fixed params (e.g. the (1,) int32
        # broadcast-scalar src of a copy_); leftovers drop with an aux-drop remark
        unfilled = [i for i, t in in_typed if not is_list_param(i, t) and i not in assigned]
        for pos, kt in zip(unfilled, aux_skipped):
            assigned[pos] = kt
        for kt in aux_skipped[len(unfilled) :]:
            d.remark_bits.append(f"aux-drop:{kt!r}")
        rest_end = len(real_kins) - len(post_fixed) if tail_align else len(real_kins)
        rest = real_kins[ki:rest_end]
        if var_pos is not None:
            # TensorList/var-positional slot absorbs ALL leftover inputs at its
            # SIGNATURE position — aux kept: list members are often small int
            # tensors (index_put_'s indices), mirroring the aclnn-mode rule
            if rest:
                assigned[var_idx] = rest
        else:
            for t in rest:
                if t.is_aux():
                    d.remark_bits.append(f"aux-drop:{t!r}")
                else:
                    d.warns.append(f"extra kernel input dropped: {t!r}")
        # emit in signature order: TTK maps CSV slots onto tensor params
        # POSITIONALLY, so a mid-signature list param (index_put_'s indices
        # before values) must not be appended after the fixed slots; gaps
        # become None placeholders, trailing unfilled params are not emitted
        if assigned:
            last = max(assigned)
            for i, t in in_typed:
                if i > last:
                    break
                kt = assigned.get(i)
                if kt is None:
                    d.tensors.append((None, None, None, is_list_param(i, t)))
                elif isinstance(kt, list):
                    d.tensors.append(
                        (tuple(s.shape for s in kt), tuple(s.dtype for s in kt), tuple(s.fmt for s in kt), True)
                    )
                else:
                    d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))

    n_required = sum(
        1 for i, t in enumerate(info.tensors) if t.name != "out" and not t.is_optional and not is_list_param(i, t)
    )
    # required-param coverage from FILLED slots only: mid-signature gaps are
    # emitted as None placeholders (dtype None) and optional filled slots must
    # not cover required ones either — emitted slots align with in_typed
    # positions here (required-`out` slots are appended after this check)
    n_filled_req = sum(
        1
        for j, s in enumerate(d.tensors)
        if s[1] is not None and not s[3] and j < len(in_typed) and not in_typed[j][1].is_optional
    )
    if n_filled_req < n_required:
        d.warns.append(f"{n_required - n_filled_req} required tensor params unmatched")

    # required `out` param: TTK demands output_tensor_indexes and routes those
    # slots into kwargs['out'] (matched by name), so kernel outputs become out=
    # slots appended after the inputs; optional out stays unfilled (return-value
    # comparison, no out= variant generated)
    out_param = next((t for t in info.tensors if t.name == "out"), None)
    if out_param is not None and not out_param.is_optional and kouts:
        if getattr(out_param, "is_tensor_list", False):
            # TensorList out: one flat slot per kernel output — TTK routes each
            # output_tensor_indexes entry into kwargs['out'] as a list
            for kt in kouts:
                d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))
                d.out_indexes.append(len(d.tensors) - 1)
        else:
            kt = kouts[0]
            d.tensors.append((kt.shape, kt.dtype, kt.fmt, False))
            if len(kouts) > 1:
                d.warns.append(f"extra kernel outputs dropped (return value used): {kouts[1:]}")
            d.out_indexes.append(len(d.tensors) - 1)
    return d


def e2e_required_scalars(info):
    """Required (no-default) Number params — attributes must be filled manually."""
    if info is None:
        return []
    return [s.name for s in info.scalars if not s.is_optional]


def torch_signature_compatible(aclnn_info, torch_info):
    """Required torch tensor params must be coverable by the aclnn signature.

    Counted against ALL aclnn tensor params (inputs, *Out and *Ref slots):
    inout-style APIs (AttentionUpdate's localOut is an aclnn out-param but a
    required torch input) feed those slots from kernel I/O too."""
    torch_req = sum(1 for t in torch_info.tensors if t.name != "out" and not t.is_optional)
    return torch_req <= len(aclnn_info.tensors)


def _acronym_snake(name: str) -> str:
    """Acronym-aware snake case: FFNToAttention -> ffn_to_attention,
    KVQuantSparseFlashAttention -> kv_quant_sparse_flash_attention,
    FFNV2 -> ffn_v2 (camel_to_snake splits leading uppercase runs and glues
    version digits: f_fnv2, k_v_...)."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            prev = name[i - 1]
            nxt = name[i + 1] if i + 1 < len(name) else ""
            if prev.islower() or (prev.isupper() and (nxt.islower() or nxt.isdigit())):
                out.append("_")
        out.append(ch.lower())
    return "".join(out)


_VERSION_SUFFIX = re.compile(r"^(.*)_v(\d+)$")


def _stem_variants(snake):
    """(stem, note) name candidates for one snake name, decreasing fidelity.

    Exact name first, then _grad->_backward aliasing and WeightNz / version
    suffix stripping in any combination (FooWeightNzV2 -> FooWeightNz ->
    Foo). Each stripping layer appends its review note; the first variant
    reaching a stem wins (exact names carry no note)."""
    entries = [(snake, "")]
    if snake.endswith("_grad"):
        entries.append((snake[: -len("_grad")] + "_backward", ""))
    seen = set()
    variants = []
    frontier = list(entries)
    while frontier:
        stem, note = frontier.pop(0)
        if stem in seen:
            continue
        seen.add(stem)
        variants.append((stem, note))
        succ = []
        if stem.endswith("_weight_nz"):
            wn_note = "WeightNz→同 API（权重 NZ 布局，torch 侧常规 ND）"
            succ.append((stem[: -len("_weight_nz")], (note + "；" if note else "") + wn_note))
        m = _VERSION_SUFFIX.match(stem)
        if m:
            base, n = m.group(1), int(m.group(2))
            for v in range(n - 1, 0, -1):
                succ.append((f"{base}_v{v}", (note + "；" if note else "") + f"版本后缀剥离（v{n}→v{v}），语义需核对"))
            succ.append((base, (note + "；" if note else "") + f"版本后缀剥离（v{n}→基础版），语义需核对"))
        frontier.extend(succ)
    return variants


def auto_resolve_torch_api(aclnn_api, api_lib, e2e_lib, cache):
    """Name-based e2e fallback mapping.

    Candidate stems per aclnn API, in decreasing fidelity (each still passes
    the signature check):
      1. exact snake name (both camel_to_snake and acronym-aware variant;
         acronyms like FFN/KV and version digits split differently) ->
         torch_npu.npu_*, torch.*, torch.nn.functional.* (+ _grad->_backward)
      2. WeightNz suffix stripped (same torch op, weight carried in NZ
         layout) -> e.g. aclnnAddmmWeightNz -> torch.addmm
      3. version suffix decremented towards the base (engine iteration of
         the same op) -> e.g. aclnnEluV2 -> torch.nn.functional.elu
    WeightNz and version stripping combine (WeightNzV2 -> WeightNz -> base).
    Stripped tiers carry a review note (listed under 近似映射/自动解析映射).
    Unverifiable pairs (either signature missing) are never mapped; accepted
    mappings carry an auto-map remark bit and are listed in the report."""
    if aclnn_api in cache:
        return cache[aclnn_api]
    result = None
    if (
        api_lib is not None
        and api_lib.available
        and e2e_lib is not None
        and e2e_lib.available
        and aclnn_api.startswith("aclnn")
    ):
        ainfo = api_lib.info_of(aclnn_api)
        if ainfo is not None:
            stems = []  # (stem, note) — exact stems first (no note)
            for snake in dict.fromkeys(
                [camel_to_snake(aclnn_api[len("aclnn") :]), _acronym_snake(aclnn_api[len("aclnn") :])]
            ):
                stems.extend(_stem_variants(snake))
            stems = list(dict.fromkeys(stems))
            candidates = []
            for stem, note in stems:
                candidates.append((f"torch_npu.npu_{stem}", note))
                # standard torch targets for every tier; stripped tiers carry a
                # review note (e.g. aclnnAddmmWeightNz -> torch.addmm,
                # aclnnEluV2 -> torch.nn.functional.elu)
                candidates.append((f"torch.{stem}", note))
                candidates.append((f"torch.nn.functional.{stem}", note))
            logging.disable(logging.WARNING)
            try:
                for cand, note in candidates:
                    tinfo = e2e_lib.info_of(cand)
                    if tinfo is not None and torch_signature_compatible(ainfo, tinfo):
                        result = (cand, note)
                        break
            finally:
                logging.disable(logging.NOTSET)
    cache[aclnn_api] = result
    return result


def audit_op_libs(op_scan, mode, api_lib, e2e_lib, lib_mapping, auto_cache):
    """Full-inventory convertibility audit for the --op-libs libraries.

    For every aclnn API declared by the library headers: aclnn-mode signature
    resolvability plus (e2e mode) the mapping source — static table /
    library-derived / name-based auto-resolution / none."""
    audit = {
        "total": len(op_scan.all_apis),
        "aclnn_resolved": 0,
        "aclnn_unresolved": [],
        "e2e_static": 0,
        "e2e_lib": 0,
        "e2e_auto": 0,
        "e2e_none": 0,
        "unmapped": [],
        "auto_apis": [],
    }
    for api in sorted(op_scan.all_apis):
        if api_lib is not None and api_lib.available and api in api_lib.infos:
            audit["aclnn_resolved"] += 1
        else:
            audit["aclnn_unresolved"].append(api)
        if api in ACLNN_TO_TORCH:
            audit["e2e_static"] += 1
        elif api in lib_mapping:
            audit["e2e_lib"] += 1
        else:
            auto = auto_resolve_torch_api(api, api_lib, e2e_lib, auto_cache)
            if auto is not None:
                audit["e2e_auto"] += 1
                audit["auto_apis"].append((api, auto[0], auto[1]))
            else:
                audit["e2e_none"] += 1
                audit["unmapped"].append(api)
    return audit


# --------------------------------------------------------------------------- #
# CSV emission
# --------------------------------------------------------------------------- #


def pyobj(v) -> str:
    return str(v)


def emit_row(d: CaseDraft, name: str, remark: str):
    shapes = tuple(t[0] for t in d.tensors)
    dtypes = tuple(t[1] for t in d.tensors)
    fmts = tuple(t[2] for t in d.tensors)
    any_fmt = any(f not in (None, "ND") for f in fmts)
    scalars = tuple(d.scalar_dtypes) if d.scalar_dtypes else ""
    out_idx = tuple(d.out_indexes) if d.out_indexes else ""
    row = {
        "testcase_name": name,
        "api_name": d.api,
        "tensor_view_shapes": pyobj(shapes) if d.tensors else "",
        "tensor_dtypes": pyobj(dtypes) if d.tensors else "",
        "tensor_formats": pyobj(fmts) if any_fmt else "",
        "scalar_dtypes": pyobj(scalars) if scalars else "",
        "output_tensor_indexes": pyobj(out_idx) if out_idx else "",
        "output_inplace_indexes": "",
        "remark": remark,
    }
    return row


ACLNN_HEADERS = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "scalar_dtypes",
    "output_tensor_indexes",
    "output_inplace_indexes",
    "remark",
]

KERNEL_HEADERS = [
    "testcase_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "attributes",
    "remark",
]

GEIR_HEADERS = KERNEL_HEADERS[:-1] + ["dyn_input_shapes", "remark"]

E2E_HEADERS = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "remark",
]

MODE_HEADERS = {"aclnn": ACLNN_HEADERS, "kernel": KERNEL_HEADERS, "geir": GEIR_HEADERS, "e2e": E2E_HEADERS}

# TTK testcase class per mode (import path, class name) — used by the selfcheck
MODE_TESTCASES = {
    "aclnn": ("ttk.core_modules.aclnn.testcase", "TestcaseAclnn"),
    "kernel": ("ttk.core_modules.kernel.testcase", "TestcaseKernel"),
    "geir": ("ttk.core_modules.geir.testcase", "GeirTestcase"),
    "e2e": ("ttk.core_modules.framework_api.testcase", "TestcaseE2e"),
}


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def iter_kernel_rows(path):
    # utf-8-sig: msprof exports may carry a BOM; header keys are stripped so a
    # trailing tab/space in the export (e.g. "Duration(us)\t") does not
    # silently blank out columns — clean_cell normalizes cell values only
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        yield from reader


def emit_e2e_row(d: CaseDraft, name: str, remark: str):
    shapes = tuple(t[0] for t in d.tensors)
    dtypes = tuple(t[1] for t in d.tensors)
    fmts = tuple(t[2] for t in d.tensors)
    any_fmt = any(f not in (None, "ND") for f in fmts)
    return {
        "testcase_name": name,
        "api_name": d.api,
        "tensor_view_shapes": pyobj(shapes) if d.tensors else "",
        "tensor_dtypes": pyobj(dtypes) if d.tensors else "",
        "tensor_formats": pyobj(fmts) if any_fmt else "",
        "attributes": "",
        "output_tensor_indexes": pyobj(tuple(d.out_indexes)) if d.out_indexes else "",
        "remark": remark,
    }


def convert(in_path, out_dir, mode="aclnn", dedup=False, selfcheck=True, op_libs=None, audit=False):
    os.makedirs(out_dir, exist_ok=True)
    headers = MODE_HEADERS[mode]
    op_scan = scan_op_libs(op_libs) if op_libs else None
    if op_scan is not None and op_scan.header_dirs:
        # before ApiLib(): the keeper reads the env var at construction and
        # source headers win over installed ones (TTK semantics)
        extend_env_path("TTK_OP_API_HEADER_PATH", op_scan.header_dirs)
    api_lib = ApiLib() if mode in ("aclnn", "e2e") else None
    op_lib = KernelOpLib() if mode in ("kernel", "geir") else None
    e2e_lib = E2eApiLib() if mode == "e2e" else None
    lib_mapping = {}  # e2e: aclnn api -> (torch api, note) derived from --op-libs
    lib_schemas_registered = 0
    if op_scan is not None:
        lib_mapping = {api: (tapi, "") for api, tapi in op_scan.aclnn_to_torch.items() if api not in ACLNN_TO_TORCH}
    if mode == "e2e" and op_scan is not None and op_scan.torch_schemas and e2e_lib is not None:
        lib_schemas_registered = e2e_lib.register_schemas(op_scan.torch_schemas)
    auto_cache = {}  # aclnn api -> auto-resolved torch api (or None)
    auto_mapped = {}  # e2e: aclnn api -> (torch api, row count) via name-based auto mapping
    stats = Counter()
    warn_agg = defaultdict(Counter)  # api/op -> warn-pattern -> count
    skipped = Counter()  # reason -> count
    needs_attr_apis = {}  # api/op -> {param: type} (params needing values)
    unresolved = Counter()  # aclnn: header miss
    unmapped = Counter()  # e2e: aclnn api without torch mapping
    lib_mapped = Counter()  # e2e: rows converted via an --op-libs derived mapping
    approx_apis = {}  # e2e: torch api -> approximate-mapping note
    req_scalar_apis = {}  # e2e: torch api -> [required scalar param names]
    vd_inputs = {}  # kernel/geir: op -> {"names": [...], "count": n} value-depend inputs
    rows_out = []
    api_seq = Counter()
    seen_sig = {}
    parse_warns = []

    for row_no, row in enumerate(iter_kernel_rows(in_path) if in_path else [], start=2):
        name = clean_cell(row.get("Name", ""))
        op_type = clean_cell(row.get("Type", ""))
        core = clean_cell(row.get("Accelerator Core", ""))
        # ---- skip rules (cannot become test cases in any mode) ----
        # hccl helper kernels may run on AI cores (not tagged COMMUNICATION),
        # so match the name prefix case variants too (Hccl/hcom/hccl/HCCL)
        if core == "COMMUNICATION" or name.startswith(("Hccl", "hcom", "hccl", "HCCL")):
            skipped["communication kernel"] += 1
            continue
        in_raw = row.get("Input Shapes", "")
        dt_raw = row.get("Input Data Types", "")
        if clean_cell(in_raw) in ("", "N/A") and clean_cell(row.get("Output Shapes", "")) in ("", "N/A"):
            skipped["no signature info (N/A)"] += 1
            continue

        ctx = f"row{row_no}"
        case_parse_warns = []
        # aclnn/e2e: keep empty-shape+empty-dtype scalars (aclScalar/Number
        # inputs the export lacks the dtype for); kernel/geir keep the drop —
        # there the 0-d entry is a GE-materialized const and emitting it as a
        # kernel input would break the op input count
        kins = parse_tensor_triplet(
            in_raw,
            dt_raw,
            row.get("Input Formats", ""),
            case_parse_warns,
            ctx,
            keep_no_dtype_scalars=mode in ("aclnn", "e2e"),
        )
        kouts = parse_tensor_triplet(
            row.get("Output Shapes", ""),
            row.get("Output Data Types", ""),
            row.get("Output Formats", ""),
            case_parse_warns,
            ctx,
        )
        if not kins and not kouts:
            skipped["all tensors undefined"] += 1
            continue

        # ---- mode-specific conversion ----
        draft = None
        op_name = torch_api = api = None
        if mode in ("kernel", "geir"):
            # op Type must be a CamelCase GE op type
            if not is_camel_op_type(op_type):
                skipped["non-aclnn kernel (snake_case/unrecognized type)"] += 1
                continue
            op_name, op_resolved = op_lib.op_name_of(op_type)
            draft = CaseDraft(op_name)  # warns/remark_bits carrier only
            # value-depend inputs: real ops-info entry, else aux-shape heuristic
            vd_names = op_lib.value_depend_inputs(op_name)
            if vd_names is None:
                vd_names = [f"{t!r}" for t in kins if t.is_aux()]
            if vd_names:
                entry = vd_inputs.setdefault(op_name, {"names": vd_names, "count": 0})
                entry["count"] += 1
            kattrs = op_lib.required_attrs(op_name)
            if kattrs:
                needs_attr_apis.setdefault(op_name, kattrs)
        elif mode == "e2e":
            cands, method = api_candidates(name, op_type)
            if not cands:
                skipped["non-aclnn kernel (snake_case/unrecognized type)"] += 1
                continue
            aclnn_api, _hit = api_lib.resolve(cands)
            mapped = ACLNN_TO_TORCH.get(aclnn_api) or lib_mapping.get(aclnn_api)
            auto_used = False
            if mapped is None:
                # name-based fallback (tiered stems, signature-verified): static
                # table and library-derived mappings always take precedence
                auto = auto_resolve_torch_api(aclnn_api, api_lib, e2e_lib, auto_cache)
                if auto is not None:
                    mapped = auto
                    auto_used = True
            if mapped is None:
                unmapped[aclnn_api] += 1
                skipped["e2e: no torch mapping"] += 1
                continue
            if aclnn_api in lib_mapping:
                lib_mapped[aclnn_api] += 1
            torch_api, note = mapped
            if auto_used:
                cnt = auto_mapped.get(aclnn_api, (torch_api, 0))[1]
                auto_mapped[aclnn_api] = (torch_api, cnt + 1)
            if note:
                approx_apis.setdefault(torch_api, note)
            info = e2e_lib.info_of(torch_api)
            reqs = e2e_required_scalars(info)
            if reqs:
                req_scalar_apis.setdefault(torch_api, reqs)
            draft = draft_e2e(torch_api, info, kins, kouts, note)
            if auto_used:
                draft.remark_bits.append(f"auto-map:{torch_api}")
            if not draft.tensors:
                # APIs whose params are all Numbers (torch.arange) or rows whose
                # inputs are all scalar/const cannot become tensor cases
                skipped["e2e: no tensor params after alignment"] += 1
                continue
        else:  # aclnn
            cands, method = api_candidates(name, op_type)
            if not cands:
                skipped["non-aclnn kernel (snake_case/unrecognized type)"] += 1
                continue
            api, hit = api_lib.resolve(cands)
            info = api_lib.info_of(api) if hit else None
            if info is None:
                unresolved[api] += 1
                draft = draft_unresolved(api, kins, kouts)
            else:
                draft = align_case(api, info, kins, kouts, ctx)
            # C-type params (attributes) that need manual values — collect per api
            if info is not None:
                cparams = {
                    n: p["type"] for n, p in info.params.items() if n not in info.tensors and n not in info.scalars
                }
                if cparams:
                    needs_attr_apis.setdefault(api, cparams)

        # unknown-dtype / unparseable-shape drops are case-level review items;
        # other parse notes go global
        for w in case_parse_warns:
            if "unknown dtype" in w or "unparseable shape" in w:
                draft.warns.append(w.split(":", 1)[1].strip())
            else:
                parse_warns.append(w)

        try:
            dur = float(clean_cell(row.get("Duration(us)", "")) or 0)
        except ValueError:
            dur = 0.0

        # dedup (optional): identical signature collapses into one case
        if mode in ("kernel", "geir"):
            sig_key = (
                op_name,
                tuple((t.shape, t.dtype, t.fmt) for t in kins),
                tuple((t.shape, t.dtype, t.fmt) for t in kouts),
            )
            agg_key = op_name
            prefix = op_name
        else:
            sig_key = (
                draft.api,
                tuple((t[0], t[1], t[2]) for t in draft.tensors),
                tuple(draft.scalar_dtypes),
                tuple(draft.out_indexes),
            )
            agg_key = draft.api
            # explicit parens: binds as (A or "api") if e2e else draft.api
            prefix = (draft.api.rsplit(".", 1)[-1].rstrip("_") or "api") if mode == "e2e" else draft.api
        if dedup and sig_key in seen_sig:
            seen_sig[sig_key]["count"] += 1
            stats["rows_deduped"] += 1
            continue

        api_seq[prefix] += 1
        tname = f"{prefix}_{api_seq[prefix]:05d}"
        remark = ";".join(
            [
                f"src=row{row_no}",
                f"name={name}",
                f"type={op_type}",
                f"dur={dur:g}us",
                f"model={clean_cell(row.get('Model ID', ''))}",
                f"task={clean_cell(row.get('Task ID', ''))}",
            ]
            + draft.remark_bits
        )
        if mode in ("kernel", "geir"):
            csv_row = emit_kernel_row(op_name, kins, kouts, tname, remark, geir=(mode == "geir"))
        elif mode == "e2e":
            csv_row = emit_e2e_row(draft, tname, remark)
        else:
            csv_row = emit_row(draft, tname, remark)
        rows_out.append(csv_row)
        if dedup:
            seen_sig[sig_key] = {"name": tname, "count": 1}
        for w in draft.warns:
            key = w.split(":")[0][:60]
            warn_agg[agg_key][key] += 1
        stats["cases"] += 1

    # dedup: annotate occurrence counts into remark
    if dedup:
        by_name = {r["testcase_name"]: r for r in rows_out}
        for meta in seen_sig.values():
            if meta["count"] > 1:
                by_name[meta["name"]]["remark"] += f";xN={meta['count']}"

    # ---- write CSV ----
    csv_path = os.path.join(out_dir, f"cases_{mode}.csv")
    if rows_out or in_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
    else:
        csv_path = None  # audit-only run: no profiling input, no cases

    # ---- self check (validate one representative per unique signature) ----
    lib_available = {
        "aclnn": api_lib.available if api_lib else False,
        "kernel": op_lib.available if op_lib else False,
        "geir": op_lib.available if op_lib else False,
        "e2e": e2e_lib.available if e2e_lib else False,
    }[mode]
    selfcheck_summary = []
    if selfcheck and lib_available:
        selfcheck_summary = run_selfcheck(rows_out, warn_agg, mode)
        if not selfcheck_summary and rows_out:
            selfcheck_summary = [("SELFCHECK_PASSED: 全部签名通过 TTK 校验", 0, [])]
    elif selfcheck:
        selfcheck_summary = [("SELFCHECK_SKIPPED: 签名/算子信息库不可用（无 CANN/torch 环境或 SOC）", 0, [])]

    # ---- full-inventory audit (--audit): convertibility of every library API ----
    audit_summary = None
    if audit:
        if mode not in ("aclnn", "e2e"):
            print(f"[prof2case] --audit 仅支持 aclnn/e2e 模式，当前 --mode {mode}，已忽略")
        elif op_scan is None:
            print("[prof2case] --audit 需配合 --op-libs 指定算子库根目录，已忽略")
        else:
            if e2e_lib is None:
                e2e_lib = E2eApiLib()
            audit_summary = audit_op_libs(op_scan, mode, api_lib, e2e_lib, lib_mapping, auto_cache)

    # ---- report ----
    report_path = os.path.join(out_dir, "conversion_report.md")
    op_libs_info = None
    if op_scan is not None:
        op_libs_info = {
            "roots": list(op_libs or []),
            "header_dirs": len(op_scan.header_dirs),
            "torch_apis": len(op_scan.torch_schemas),
            "schemas_registered": lib_schemas_registered,
            "derived": len(lib_mapping),
            "derived_rows": sum(lib_mapped.values()),
            "notes": op_scan.notes,
        }
    write_report(
        report_path,
        in_path,
        stats,
        skipped,
        unresolved,
        warn_agg,
        needs_attr_apis,
        rows_out,
        lib_available,
        selfcheck_summary,
        parse_warns,
        dedup,
        mode,
        op_lib.available if op_lib else False,
        unmapped,
        approx_apis,
        req_scalar_apis,
        vd_inputs,
        op_libs_info,
        auto_mapped,
        audit_summary,
    )
    if csv_path:
        print(f"[prof2case] mode={mode} cases: {stats['cases']} -> {csv_path}")
    print(f"[prof2case] report -> {report_path}")
    return csv_path, report_path


def run_selfcheck(rows_out, warn_agg, mode="aclnn"):
    """Parse generated rows through TTK's real parser (one representative per
    signature: validation is deterministic per signature). Returns
    [(fail_reason, count, example_names)]."""
    sig_groups = OrderedDict()
    headers = MODE_HEADERS[mode]
    key_field = "op_name" if mode in ("kernel", "geir") else "api_name"
    for r in rows_out:
        key = (
            r[key_field],
            r.get("tensor_view_shapes", r.get("input_shapes", "")),
            r.get("tensor_dtypes", r.get("input_dtypes", "")),
            r.get("scalar_dtypes", ""),
            r.get("output_tensor_indexes", r.get("output_shapes", "")),
        )
        sig_groups.setdefault(key, []).append(r["testcase_name"])
    by_name = {r["testcase_name"]: r for r in rows_out}
    reps = []
    name_of = {}
    for names in sig_groups.values():
        rep = by_name[names[0]]
        reps.append(rep)
        name_of[rep["testcase_name"]] = names
    failures = Counter()
    examples = defaultdict(list)
    logging.disable(logging.CRITICAL)
    try:
        import importlib  # noqa: PLC0415

        from ttk.core_modules.common.testcase.factory import UniversalTestcaseFactory  # noqa: PLC0415

        mod_name, cls_name = MODE_TESTCASES[mode]
        testcase_class = getattr(importlib.import_module(mod_name), cls_name)
        table = (headers, [[r.get(h, "") for h in headers] for r in reps])
        factory = UniversalTestcaseFactory(table, testcase_class, skip_validate=True)
        for tc in factory.get():
            try:
                tc.validate()
            except Exception as err:  # noqa: BLE001
                reason = f"EXC:{type(err).__name__}"
                tc.is_valid = False
                tc.fail_reason = reason
            if not getattr(tc, "is_valid", True):
                reason = getattr(tc, "fail_reason", "UNKNOWN")
                failures[reason] += len(name_of.get(tc.testcase_name, [tc.testcase_name]))
                if len(examples[reason]) < 3:
                    examples[reason].append(tc.testcase_name)
    except Exception as err:  # noqa: BLE001
        return [(f"SELFCHECK_FAILED: {err}", 0, [])]
    finally:
        logging.disable(logging.NOTSET)
    return [(k, v, examples[k]) for k, v in failures.most_common()]


def write_report(
    path,
    in_path,
    stats,
    skipped,
    unresolved,
    warn_agg,
    needs_attr_apis,
    rows_out,
    api_available,
    selfcheck_summary,
    parse_warns,
    dedup,
    mode="aclnn",
    op_lib_available=False,
    unmapped=None,
    approx_apis=None,
    req_scalar_apis=None,
    vd_inputs=None,
    op_libs_info=None,
    auto_mapped=None,
    audit_summary=None,
):
    unmapped = unmapped or Counter()
    approx_apis = approx_apis or {}
    req_scalar_apis = req_scalar_apis or {}
    vd_inputs = vd_inputs or {}
    key_field = "op_name" if mode in ("kernel", "geir") else "api_name"
    api_counts = Counter(r[key_field] for r in rows_out)
    mode_desc = {
        "aclnn": "ACLNN（引擎 API 链，按 aclnn 头文件签名对齐）",
        "kernel": "Kernel（算子编译执行链，kernel I/O 原样转换）",
        "geir": "GEIR（GE 图编译执行链，kernel I/O 原样转换 + dyn_input_shapes 留空）",
        "e2e": "E2E（框架 API 链，aclnn→torch 映射 + torch 签名对齐）",
    }
    lines = []
    ap = lines.append
    ap("# Profiling → TTK 用例转换报告")
    ap("")
    if in_path:
        ap(f"- 源文件: `{in_path}`")
    else:
        ap("- 源文件: （纯审计，无 profiling 输入）")
    ap(f"- 目标模式: **{mode}** — {mode_desc[mode]}")
    ap(f"- 生成用例数: **{stats['cases']}**（{'按唯一签名去重' if dedup else '逐行转换，未去重'}）")
    ap(f"- 覆盖{'算子' if mode in ('kernel', 'geir') else 'API'}数: {len(api_counts)}")
    if mode in ("kernel", "geir"):
        ap(
            f"- 算子信息库: {'可用（op_name 取 opInterface 名）' if op_lib_available else '**不可用**（无 SOC/ops-info 环境，op_name 由驼峰转下划线派生，输入输出数量未核对）'}"
        )
    elif mode == "e2e":
        ap(
            f"- torch 签名库: {'可用（ttk FrameworkApiInfoKeeper）' if api_available else '**不可用**（未做签名对齐与自检，全部用例需人工核对）'}"
        )
    else:
        ap(
            f"- 头文件签名库: {'可用（ttk OpApiInfoKeeper）' if api_available else '**不可用**（未做签名对齐与自检，全部用例需人工核对）'}"
        )
    if op_libs_info is not None:
        ap(
            f"- 算子库（--op-libs）: {len(op_libs_info['roots'])} 个根，aclnn 头文件目录 {op_libs_info['header_dirs']} 个"
            f"（已并入 TTK_OP_API_HEADER_PATH，源码头文件优先于已安装版本）"
        )
        if mode == "e2e":
            ap(
                f"  - torch.ops 签名 {op_libs_info['torch_apis']} 个（注册 {op_libs_info['schemas_registered']} 个，"
                f"离线对齐可用）；aclnn→torch 派生映射 {op_libs_info['derived']} 个，覆盖 {op_libs_info['derived_rows']} 行"
            )
        if op_libs_info["notes"]:
            for note in op_libs_info["notes"][:10]:
                ap(f"  - 注意: {note}")
    ap("")
    ap("## 值层空缺提醒（重要）")
    ap("")
    if in_path is None:
        # pure audit (no -i): no cases were generated, the gaps below apply to
        # a follow-up conversion run
        ap("纯审计模式不生成用例；以下值层空缺适用于后续带 `-i` 的转换结果，需要人工补齐：")
    else:
        ap("profiling 不含任何值信息，以下字段**全部留空**，需要人工补齐：")
    ap("")
    if mode in ("kernel", "geir"):
        ap("- `attributes`（算子 ATTR 值：axis/perm/padding/strides 等）—— 缺失可能导致编译失败或语义偏离")
        ap(
            "- ValueDepend 输入值（perm/begin/end/axes 等）—— 张量保留在输入中，但**值需在 attributes 指定**，否则随机生成"
        )
        if mode == "geir":
            ap(
                "- `dyn_input_shapes` —— 留空（profiling 无动态 shape 信息），GE 图按静态 shape 编译；需动态编译请人工补 -1 维度"
            )
    elif mode == "e2e":
        ap("- `attributes`（API 必选非张量参数：dim/normalized_shape/padding 等）—— 必选参数不补值**无法通过校验**")
        ap(
            "- kernel 标量输入（shape 为 `()` 的张量）对应 Number 参数，值不可恢复，未填入 attributes（每处记 `scalar-drop` remark 并列入对齐警告）"
        )
    else:
        ap("- `attributes`（API 的 C 参数值：dim/stride/padding/numHeads 等）—— 含 C 参数的 API 不补值**无法通过校验**")
    ap("- `scalar_data_ranges` / `input_data_ranges`（数据范围）—— 空时 TTK 默认 [-2,2] 随机")
    if mode != "e2e":
        ap(
            "- `tensor_view_strides` / `offsets` / `storage_shapes`（非连续布局）—— 空时按连续张量生成，与真实业务布局可能不同"
        )
    ap("- 标量值本身由 TTK 随机生成，与真实业务取值无关")
    ap("")
    ap("## 跳过统计")
    ap("")
    if skipped:
        for reason, c in skipped.most_common():
            ap(f"- {reason}: {c} 行")
    else:
        ap("- 无")
    ap("")
    if mode == "e2e":
        ap("## 未映射 API（无 torch 映射，已跳过；可扩展脚本内 `ACLNN_TO_TORCH` 表）")
        ap("")
        if unmapped:
            for api, c in unmapped.most_common():
                ap(f"- `{api}`: {c} 行")
            ap("")
            ap(
                "说明：Metadata 预处理、内部变体（Inner/Inplace）、版本滞后于 torch_npu 的 API 等"
                "没有 torch 等价接口，属预期跳过；需要覆盖时优先转 **aclnn** 模式（`--mode aclnn`）。"
            )
        else:
            ap("- 无")
        ap("")
        if auto_mapped:
            ap("## 同名自动解析映射（exact 名匹配 + 签名校验，语义请抽查）")
            ap("")
            for api, (target, cnt) in sorted(auto_mapped.items(), key=lambda kv: -kv[1][1]):
                ap(f"- `{api}`（{cnt} 行）-> `{target}`")
            ap("")
        if approx_apis:
            ap("## 近似映射（语义/默认值可能不同，抽查核对）")
            ap("")
            for api, note in sorted(approx_apis.items(), key=lambda kv: -api_counts.get(kv[0], 0)):
                n = api_counts.get(api, 0)
                ap(f"- `{api}`（{n} 例）: {note}")
            ap("")
    if mode == "aclnn":
        ap("## 未解析 API（头文件中未找到，用例按 kernel 张量顺序生成，无法通过校验）")
        ap("")
        if unresolved:
            for api, c in unresolved.most_common():
                ap(f"- `{api}`: {c} 行")
            ap("")
            ap(
                "处理建议：确认算子包已安装，或通过 `--op-libs` 指向算子库源码仓（自动收集 op_api 头文件），"
                "或通过 `TTK_OP_API_HEADER_PATH` 指向自定义 op_api 头文件目录，或手工修正 api_name。"
            )
        else:
            ap("- 无")
        ap("")
    if mode in ("kernel", "geir") and vd_inputs:
        ap("## 疑似 ValueDepend 输入（值需在 attributes 指定，否则随机生成）")
        ap("")
        for op, entry in sorted(vd_inputs.items(), key=lambda kv: -kv[1]["count"]):
            names = ", ".join(f"`{n}`" for n in entry["names"])
            ap(f"- `{op}`（{entry['count']} 例）: {names}")
        ap("")
    if needs_attr_apis:
        ap("## 需人工补 attributes 的 API/算子")
        ap("")
        if mode == "aclnn":
            ap(
                "以下 API 存在 C 类型参数（int64_t/double/bool/char*/aclIntArray* 等），`attributes` 为空会导致 PARAM_COUNT_MISMATCH："
            )
        elif mode in ("kernel", "geir"):
            ap("以下算子存在无默认值的 ATTR（来自算子信息库），`attributes` 为空可能导致编译失败：")
        else:
            ap("以下 API 存在必选非张量参数（来自 torch 签名），`attributes` 为空无法通过校验：")
        ap("")
        for api, cparams in sorted(needs_attr_apis.items(), key=lambda kv: -api_counts.get(kv[0], 0)):
            n = api_counts.get(api, 0)
            plist = ", ".join(f"`{k}`({v})" for k, v in cparams.items())
            ap(f"- `{api}`（{n} 例）: {plist}")
        ap("")
    if mode == "e2e" and req_scalar_apis:
        ap("## 需人工补 attributes 的必选标量参数（torch 签名）")
        ap("")
        for api, scalars in sorted(req_scalar_apis.items(), key=lambda kv: -api_counts.get(kv[0], 0)):
            n = api_counts.get(api, 0)
            ap(f"- `{api}`（{n} 例）: {', '.join(f'`{s}`' for s in scalars)}")
        ap("")
    if audit_summary is not None:
        ap("## 算子库全量算子审计（--audit）")
        ap("")
        total = audit_summary["total"]
        pct = (audit_summary["aclnn_resolved"] * 100 // total) if total else 0
        aclnn_unresolved = audit_summary.get("aclnn_unresolved") or []
        ap(f"- 库内 aclnn API 总数: **{total}**")
        # self-contained: the「未解析 API」section lists conversion-row misses
        # only and is always empty in a pure audit run (no -i)
        ap(
            f"- aclnn 模式可解析（头文件签名对齐）: {audit_summary['aclnn_resolved']}（{pct}%）"
            + ("" if not aclnn_unresolved else f" —— 未解析 {len(aclnn_unresolved)} 项（清单见本节末尾）")
        )
        covered = audit_summary["e2e_static"] + audit_summary["e2e_lib"] + audit_summary["e2e_auto"]
        epct = (covered * 100 // total) if total else 0
        ap(
            f"- e2e 映射覆盖: {covered}/{total}（{epct}%）= 内置表 {audit_summary['e2e_static']}"
            f" + 算子库派生 {audit_summary['e2e_lib']} + 同名自动解析 {audit_summary['e2e_auto']}"
        )
        ap(f"- e2e 无映射: {audit_summary['e2e_none']}（清单如下，Metadata/内部变体/版本滞后/引擎级通路等）")
        ap("")
        for api in audit_summary["unmapped"]:
            ap(f"  - `{api}`")
        if audit_summary["auto_apis"]:
            ap("")
            ap("自动解析映射清单（同名/后缀剥离 + 签名校验，语义请抽查）:")
            for api, target, note in audit_summary["auto_apis"]:
                ap(f"  - `{api}` -> `{target}`" + (f"（{note}）" if note else ""))
        if aclnn_unresolved:
            ap("")
            ap("aclnn 未解析清单（头文件/签名库均未命中）:")
            for api in aclnn_unresolved:
                ap(f"  - `{api}`")
        ap("")
    ap("## 对齐警告（按 API 聚合）")
    ap("")
    if warn_agg:
        for api, wcnt in sorted(warn_agg.items(), key=lambda kv: -sum(kv[1].values())):
            ap(f"- `{api}`:")
            for w, c in wcnt.most_common():
                ap(f"  - {w}: {c} 例")
    else:
        ap("- 无")
    if parse_warns:
        ap("")
        ap("## 解析警告")
        ap("")
        for w in parse_warns[:50]:
            ap(f"- {w}")
    if selfcheck_summary:
        ap("")
        ap("## TTK 校验预检（每签名取 1 例代表校验，结果按签名外推）")
        ap("")
        has_failures = any(names for _, _, names in selfcheck_summary)
        for reason, c, names in selfcheck_summary:
            if names:
                ap(f"- `{reason}`: {c} 例（如 {', '.join(names)}）")
            else:
                ap(f"- `{reason}`")
        if has_failures:
            ap("")
            if mode == "aclnn":
                ap(
                    "其中 `PARAM_COUNT_MISMATCH`/`OP_API_NOT_FOUND` 与上方「需人工补 attributes」/「未解析 API」对应；补值后可复检："
                )
            else:
                ap("预期失败（缺值/无映射/派生名）与上方清单对应；补值后可复检：")
            ap("")
            ap("```shell")
            ap(f"python3 -m ttk {mode} -i cases_{mode}.csv --validate")
            ap("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=None, help="kernel_details.csv 路径（--audit 全量审计时可省略）")
    ap.add_argument("-o", "--output", default="prof2case_out", help="输出目录（默认 prof2case_out）")
    ap.add_argument(
        "--mode",
        choices=("aclnn", "kernel", "geir", "e2e"),
        default="aclnn",
        help="目标测试模式（默认 aclnn；由 skill 使用时的用户需求决定）",
    )
    ap.add_argument("--dedup", action="store_true", help="按唯一签名去重（默认逐行转换不去重）")
    ap.add_argument("--no-selfcheck", action="store_true", help="跳过 TTK 校验预检")
    ap.add_argument(
        "--op-libs",
        default="",
        help="算子库源码仓根目录，冒号分隔（如 /path/to/ops-nn:/path/to/ops-transformer）；"
        "自动收集 op_api 头文件解析自定义算子签名（aclnn/e2e），并扫描 torch_extension 派生 e2e 映射",
    )
    ap.add_argument(
        "--audit",
        action="store_true",
        help="审计 --op-libs 算子库全量算子的可转换性（aclnn 头文件解析 / e2e 映射覆盖三路来源），写入报告；"
        "可不带 -i（纯审计）",
    )
    args = ap.parse_args()
    if not args.input and not (args.audit and args.op_libs):
        ap.error("需要 -i 输入 profiling 文件；或 --audit 配合 --op-libs 做算子库全量审计")
    if args.input and not os.path.isfile(args.input):
        ap.error(f"输入文件不存在: {args.input}")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    op_libs = [r for r in args.op_libs.split(":") if r.strip()]
    convert(
        args.input,
        args.output,
        mode=args.mode,
        dedup=args.dedup,
        selfcheck=not args.no_selfcheck,
        op_libs=op_libs,
        audit=args.audit,
    )


if __name__ == "__main__":
    main()
