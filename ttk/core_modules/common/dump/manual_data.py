#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Restorable input and CPU-golden data shared by E2E, ACLNN, and Kernel."""

import filecmp
import hashlib
import logging
import math
import os
import pathlib
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy

from ....utilities import is_4bit_dtype, resolve_numpy_dtypes
from ....utilities.data import constant_array
from ....utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor
from .formatter import _CUSTOM_NUMPY_DTYPES, DumpLoadError, get_formatter

SUPPORTED_MANUAL_DATA_FORMATS = ("bin", "npy", "pt")
_SAFE_CASE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_DATA_FILE_NAME = re.compile(
    r"^(input|scalar|golden)_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)"
    r"(?:__shape_(scalar|[0-9]+(?:x[0-9]+)*))?\.(bin|pt|npy)$"
)
# XPU 三方输出槽位（--dump xpu 落盘协议）：xpu_{provider}_{idx}_{dtype}.{ext}
# ——provider 占 segment 位（DataDumper.dump_xpu 同款命名），回放按 idx 对位、
# 多 provider 同 idx 取字典序首个（回放取单一真值源）。
_XPU_FILE_NAME = re.compile(
    r"^xpu_([A-Za-z0-9][A-Za-z0-9_.-]*?)_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)\.(bin|pt|npy)$"
)
_NONE_DTYPE = "none"
_UNKNOWN_SHAPE = object()
# 调试 dump（{root}/dump/{case}/）产物读回放行协议：
# - output[_{segment}]_{idx}_{dtype} / tiling_data[_{segment}]：执行期派生产物，
#   读回侧会重新执行/编译产出——扫描时忽略，但把目录标记为"含调试产物"；
# - input_{segment}_{idx}_{dtype}：按段分流——dyn（kernel 执行前）/const、dynamic
#   （geir 数据集与模式无关）去段映射 input 槽；eager/graph_*（e2e dump-on-fail
#   执行后补漏，in-place API 可能已改写）与未知段一律拒绝。
_DEBUG_OUTPUT_FILE_NAME = re.compile(
    r"^output(?:_[A-Za-z0-9][A-Za-z0-9_.-]*?)?_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)"
    r"\.(bin|pt|npy)$"
)
_DEBUG_TILING_FILE_NAME = re.compile(r"^tiling_data(?:_[A-Za-z0-9][A-Za-z0-9_.-]*?)?\.(bin|pt|npy)$")
_DEBUG_INPUT_FILE_NAME = re.compile(
    r"^input_([A-Za-z0-9][A-Za-z0-9_.-]*?)_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)\.(bin|pt|npy)$"
)
_DEBUG_INPUT_ACCEPTED_SEGMENTS = ("dyn", "const", "dynamic")
_DEBUG_INPUT_REJECTED_SEGMENTS = ("eager", "graph_cst", "graph_dyn", "graph_aclgraph")

_RAW_UINT8_COMPATIBLE_DTYPES = {
    "int4",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e8m0",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat8",
    "hifloat4",
}


class ManualDataError(RuntimeError):
    """Manual data is missing, corrupt, or incompatible with the CSV case."""


@dataclass(frozen=True)
class _ManualDataFile:
    role: str
    index: int
    dtype: str
    file_format: str
    path: pathlib.Path
    saved_shape: Optional[Tuple[int, ...]] = None
    provider: Optional[str] = None  # xpu 槽位专属（其余 role 恒 None）


@dataclass
class ManualDataCase:
    inputs: List[Any]
    scalars: List[Any]
    case_dir: pathlib.Path
    file_format: str
    _golden_files: Tuple[_ManualDataFile, ...]
    _xpu_files: Tuple[_ManualDataFile, ...] = ()  # 按 (index, provider) 排序

    @property
    def has_goldens(self) -> bool:
        return bool(self._golden_files)

    @property
    def has_third_parties(self) -> bool:
        """case 目录含 xpu_{provider}_{idx}_{dtype} 槽位文件（--dump xpu 回放）。"""
        return bool(self._xpu_files)

    def third_party_paths(self) -> tuple:
        """三方输出文件路径，按槽位 idx 对位（0..max 连续，缺位补 None）。

        多 provider 同 idx 取字典序首个（_xpu_files 已按 (index, provider)
        排序）——回放取单一真值源。"""
        if not self._xpu_files:
            return ()
        by_index: Dict[int, str] = {}
        for entry in self._xpu_files:
            by_index.setdefault(entry.index, str(entry.path))
        count = max(by_index) + 1
        return tuple(by_index.get(index) for index in range(count))

    def load_goldens(self, references=None, shapes=None, dtypes=None) -> List[Any]:
        """Load saved goldens after device output shapes are available.

        Raw ``bin`` payloads do not carry shape metadata, so non-None bin Golden
        filenames encode it. E2E passes device outputs as references, while ACLNN
        passes returned output shapes and the CSV output dtypes. A cache may carry
        additional operator-specific Goldens; those are restored from their own
        file metadata and left for a custom compare hook to interpret.
        """
        if references is not None and shapes is not None:
            raise ManualDataError("golden references and explicit shapes are mutually exclusive")

        expected_shapes = []
        storage_shapes = []
        if references is not None:
            for index, value in enumerate(references):
                if value is None:
                    expected_shapes.append(None)
                    storage_shapes.append(None)
                elif isinstance(value, str):
                    expected_shapes.append(_UNKNOWN_SHAPE)
                    storage_shapes.append(_UNKNOWN_SHAPE)
                else:
                    label = f"output[{index}]"
                    logical_shape = _value_shape(value, label)
                    storage = _as_numpy(value, label)
                    storage_shape = tuple(storage.shape)
                    expected_shape = logical_shape if not logical_shape and storage.size == 1 else storage_shape
                    expected_shapes.append(expected_shape)
                    storage_shapes.append(storage_shape)
        elif shapes is not None:
            shape_values = list(shapes)
            dtype_values = list(dtypes or ())
            if dtypes is not None and len(dtype_values) != len(shape_values):
                raise ManualDataError(
                    f"golden dtype count {len(dtype_values)} != output shape count {len(shape_values)}"
                )
            for index, shape in enumerate(shape_values):
                if shape is None:
                    expected_shapes.append(None)
                    storage_shapes.append(None)
                    continue
                logical_shape = tuple(int(dimension) for dimension in shape)
                if dtypes is None:
                    storage_shape = _saved_array_shape(shape)
                else:
                    storage_shape = physical_array_spec(dtype_values[index], shape)[1]
                expected_shape = logical_shape if not logical_shape and math.prod(storage_shape) == 1 else storage_shape
                expected_shapes.append(expected_shape)
                storage_shapes.append(storage_shape)
        else:
            expected_shapes = [_UNKNOWN_SHAPE] * len(self._golden_files)
            storage_shapes = [_UNKNOWN_SHAPE] * len(self._golden_files)

        golden_count = len(self._golden_files)
        output_count = len(expected_shapes)
        if golden_count < output_count:
            raise ManualDataError(f"golden slot count {golden_count} < device output count {output_count}")
        if golden_count > output_count:
            # Different APIs of one operator may expose different output subsets,
            # so no positional shape mapping is valid when their counts differ.
            expected_shapes = [_UNKNOWN_SHAPE] * golden_count
            storage_shapes = [_UNKNOWN_SHAPE] * golden_count

        values = []
        for entry, expected_shape, storage_shape in zip(self._golden_files, expected_shapes, storage_shapes):
            # A None golden suppresses comparison for an optional output. The
            # device API may still materialize that output during replay.
            if entry.dtype == _NONE_DTYPE:
                values.append(_load_none(entry, "golden"))
                continue
            if expected_shape is None:
                values.append(_load_none(entry, "golden"))
                continue
            logging.info("Reading manual golden file: %s ...", entry.path)
            if (
                entry.saved_shape is not None
                and expected_shape is not _UNKNOWN_SHAPE
                and tuple(expected_shape) != entry.saved_shape
            ):
                raise ManualDataError(
                    f"golden[{entry.index}] saved shape {entry.saved_shape} "
                    f"!= device output shape {tuple(expected_shape)}"
                )
            resolved_storage_shape = entry.saved_shape if storage_shape is _UNKNOWN_SHAPE else storage_shape
            if resolved_storage_shape is None and entry.file_format == "bin" and entry.dtype != _NONE_DTYPE:
                # 当前协议：读侧无 shape 映射（额外 golden 槽 / golden 数与 output 数
                # 不一致）且文件名不带存量 shape 段 → 读一维流，交 custom compare 解释
                logging.warning(
                    "golden[%s] bin payload %s loaded as a flat stream "
                    "(no reader-side shape mapping); custom compare hook interprets",
                    entry.index,
                    entry.path.name,
                )
            values.append(_load_array(entry, resolved_storage_shape))
        return values


def resolve_manual_data_directories(switches, required: bool = True) -> Tuple[pathlib.Path, ...]:
    """Resolve manual data directories from CLI --manual-data-dirs (deduplicated)."""
    values = list(getattr(switches, "manual_data_dirs", ()) or ())
    result = []
    seen = set()
    for value in values:
        path = pathlib.Path(value).expanduser().resolve()
        if path not in seen:
            result.append(path)
            seen.add(path)
    if not result and required:
        raise ManualDataError("no manual data directory was resolved; pass --manual-data-dirs")
    return tuple(result)


def manual_data_store(switches) -> "ManualDataStore":
    return ManualDataStore(resolve_manual_data_directories(switches))


def manual_data_dump_store(testcase, switches) -> Optional["ManualDataStore"]:
    """Dump-direction store: one output directory, old same-name case invalidated first.

    Returns None when no directory is configured. The caller decides when the
    write direction applies (--no-prof runs, or a full-flow run whose load
    directories turned out empty).
    """
    directories = resolve_manual_data_directories(switches, required=False)
    if not directories:
        return None
    if len(directories) > 1:
        raise ManualDataError("manual data output requires exactly one --manual-data-dirs path")
    store = ManualDataStore(directories)
    store.invalidate_case(testcase.testcase_name)
    return store


def _all_roots_empty(directories) -> bool:
    return all(not (root.exists() and any(root.iterdir())) for root in directories)


def load_manual_data_case(
    testcase, case_type: str, switches, before_load: Optional[Callable[[], None]] = None, require_goldens: bool = True
) -> Optional[ManualDataCase]:
    """Load a manual-data case from the CLI directory list.

    None (normal generation) when --no-prof runs (write direction regenerates),
    no directory is given, or every configured directory is empty. A case miss
    in a non-empty directory raises instead of silently regenerating.
    """
    if getattr(switches, "no_prof", False):
        return None
    directories = resolve_manual_data_directories(switches, required=False)
    if not directories or _all_roots_empty(directories):
        return None
    if getattr(switches, "golden_mode", "Enable") != "Enable":
        raise ManualDataError("manual-data loading requires --golden-mode Enable")
    if getattr(switches, "validate_only", False):
        raise ManualDataError("manual-data loading cannot be combined with --validate")
    if case_type == "kernel" and getattr(switches, "compile_only", False):
        raise ManualDataError("manual-data loading cannot be combined with Kernel --compile-only")
    if case_type == "e2e" and getattr(switches, "force_cpu", False):
        raise ManualDataError("manual-data loading is the device stage and cannot use --cpu")
    store = ManualDataStore(directories)
    if before_load is not None:
        before_load()
    return store.load_case(testcase, case_type, require_goldens=require_goldens)


def snapshot_manual_values(values: Sequence[Any], label: str) -> List[Any]:
    """Copy values before golden callbacks can mutate their backing storage."""
    snapshots = []
    for index, value in enumerate(values or ()):
        if value is None:
            snapshots.append(None)
        else:
            snapshots.append(_as_numpy(value, f"{label}[{index}]").copy())
    return snapshots


def case_directory_name(testcase_name: str) -> str:
    """Return the on-disk directory name for a manual-data testcase.

    Public because the simulator backend's ``case_writer`` reuses the same
    naming so manual-data replay and simulation share one per-case directory.
    """
    safe = _SAFE_CASE_NAME.sub("_", testcase_name).strip("._") or "case"
    if safe == testcase_name and len(safe) <= 120:
        return safe
    digest = hashlib.sha256(testcase_name.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:96]}-{digest}"


def _decode_shape(encoded_shape: Optional[str]) -> Optional[Tuple[int, ...]]:
    """存量数据 shape 段解析（早期版本落盘的 golden 文件名兼容）。"""
    if encoded_shape is None:
        return None
    if encoded_shape == "scalar":
        return ()
    return tuple(int(dimension) for dimension in encoded_shape.split("x"))


def _as_numpy(value, label: str) -> numpy.ndarray:
    if isinstance(value, str):
        raise ManualDataError(f"cannot save {label} sentinel {value!r}")
    if isinstance(value, numpy.ndarray):
        array = value
    elif isinstance(value, numpy.generic):
        array = numpy.asarray(value)
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        array = torch_to_numpy_tensor(value.detach().cpu())
    else:
        try:
            array = numpy.asarray(value)
        except Exception as exc:
            raise ManualDataError(f"unsupported {label} value type: {type(value)!r}") from exc
    if array.dtype.hasobject or array.dtype.fields:
        raise ManualDataError(f"cannot save {label} with dtype {array.dtype}")
    return numpy.ascontiguousarray(array)


def _value_shape(value, label: str) -> Tuple[int, ...]:
    """Return logical shape before contiguous conversion promotes scalars to (1,)."""
    if isinstance(value, numpy.ndarray):
        return tuple(value.shape)
    if hasattr(value, "shape"):
        return tuple(int(dimension) for dimension in value.shape)
    try:
        return tuple(numpy.asarray(value).shape)
    except Exception as exc:
        raise ManualDataError(f"cannot determine {label} shape: {type(value)!r}") from exc


def _saved_array_shape(shape) -> Tuple[int, ...]:
    result = tuple(int(dimension) for dimension in shape)
    return result if result else (1,)


def physical_array_spec(logical_dtype, logical_shape) -> Tuple[str, Tuple[int, ...]]:
    """Return the dtype/shape of the numpy backing array saved by TTK."""
    dtype_name = getattr(logical_dtype, "name", str(logical_dtype))
    shape = tuple(int(dimension) for dimension in logical_shape)
    if dtype_name == "complex32":
        return "float16", shape + (2,)
    if dtype_name in ("uint1", "int1"):
        element_count = math.prod(shape) if shape else 1
        return "uint8", ((element_count + 7) // 8,)
    try:
        resolved = resolve_numpy_dtypes((dtype_name,))[0]
        return numpy.dtype(resolved).name, _saved_array_shape(shape)
    except Exception as exc:
        raise ManualDataError(f"cannot resolve manual-data dtype {dtype_name!r}") from exc


def _input_specs(testcase, case_type: str) -> List[Optional[Tuple[str, Tuple[int, ...]]]]:
    # kernel/geir 共用 flat_input_* 槽协议（GeirTestcase 继承 TestcaseKernel）
    if case_type in ("kernel", "geir"):
        shapes = list(getattr(testcase, "flat_input_shapes", ()) or ())
        dtypes = list(getattr(testcase, "flat_input_dtypes", ()) or ())
    else:
        shapes = list(getattr(testcase, "flat_tensor_view_shapes", ()) or ())
        dtypes = list(getattr(testcase, "flat_tensor_dtypes", ()) or ())
    if len(dtypes) != len(shapes):
        raise ManualDataError(f"CSV tensor dtype count {len(dtypes)} != tensor slot count {len(shapes)}")
    result = []
    for index, shape in enumerate(shapes):
        if shape is None:
            result.append(None)
        elif case_type in ("e2e", "aclnn") and any(int(dimension) == -1 for dimension in shape):
            # Dynamic API slots are materialized by npu_preprocess, not restored
            # from a CPU input file.  A replay store may carry a None marker.
            result.append(None)
        elif case_type in ("kernel", "geir"):
            result.append(physical_array_spec(dtypes[index], shape))
        else:
            result.append(physical_array_spec(dtypes[index], testcase.flat_storage_shape(index)))
    return result


def _saved_input_indexes(testcase, case_type: str, input_count: int) -> Tuple[int, ...]:
    """Return input slots whose contents are required to replay the API."""
    if case_type != "aclnn":
        return tuple(range(input_count))
    pure_outputs = set(getattr(testcase, "pure_output_indexes", ()) or ())
    return tuple(index for index in range(input_count) if index not in pure_outputs)


def _restore_pure_outputs(testcase, values, full_specs, saved_indexes):
    """Expand compact ACLNN data with fresh storage for pure output slots."""
    saved_values = dict(zip(saved_indexes, values))
    init_value = 0 if testcase.api_name == "aclnnInplaceOne" else 1
    restored = []
    for index, spec in enumerate(full_specs):
        if index in saved_values:
            restored.append(saved_values[index])
            continue
        if spec is None:
            restored.append(None)
            continue
        dtype_name, storage_shape = spec
        dtype = resolve_numpy_dtypes((dtype_name,))[0]
        restored.append(constant_array(dtype, storage_shape, init_value=init_value))
    return restored


def _scalar_specs(testcase, case_type: str) -> List[Optional[Tuple[str, Tuple[int, ...]]]]:
    if case_type != "aclnn":
        return []
    result = []
    for dtype in getattr(testcase, "flat_scalar_dtypes", ()) or ():
        if dtype is None:
            result.append(None)
        else:
            result.append(physical_array_spec(dtype, ()))
    return result


def _expected_golden_count(testcase, case_type: str) -> Optional[int]:
    if case_type in ("aclnn", "kernel", "geir"):
        # 通路协议：两阶段 golden 槽数由 testcase 声明（kernel/geir 的 shape-unknown
        # 元数据槽等读写侧差异封装在通路侧）；未声明协议则回退 flat 输出列数
        declared = getattr(testcase, "manual_data_golden_dtypes", None)
        if declared is not None:
            return len(declared)
        return len(getattr(testcase, "flat_output_dtypes", ()) or ())
    return None


def _resolve_file_dtype(dtype_name: str):
    try:
        return resolve_numpy_dtypes((dtype_name,))[0]
    except Exception as exc:
        raise ManualDataError(f"cannot resolve dtype {dtype_name!r} from manual-data filename") from exc


def _expected_bin_size(dtype_name: str, shape: Tuple[int, ...]) -> int:
    count = math.prod(shape)
    if is_4bit_dtype(dtype_name):
        return (count + 1) // 2
    try:
        return count * numpy.dtype(_resolve_file_dtype(dtype_name)).itemsize
    except Exception as exc:
        if isinstance(exc, ManualDataError):
            raise
        raise ManualDataError(f"cannot determine byte size for dtype {dtype_name!r}") from exc


def _same_bytes(left: pathlib.Path, right: pathlib.Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    if left.samefile(right):
        return True
    return filecmp.cmp(str(left), str(right), shallow=False)


def _load_array(entry: _ManualDataFile, shape: Optional[Tuple[int, ...]]) -> numpy.ndarray:
    if entry.dtype == _NONE_DTYPE:
        raise ManualDataError(f"{entry.role}[{entry.index}] is None, not an array")
    dtype = _resolve_file_dtype(entry.dtype)
    try:
        if entry.file_format == "bin" and shape is not None:
            # bin 无 shape 元数据：按文件名协议（dtype 名 + 期望 shape）预检字节数
            expected_size = _expected_bin_size(entry.dtype, shape)
            if entry.path.stat().st_size != expected_size:
                raise ManualDataError(f"{entry.path} byte size {entry.path.stat().st_size} != expected {expected_size}")
        array = get_formatter(entry.file_format).load(entry.path, dtype, shape)
    except DumpLoadError as exc:
        raise ManualDataError(str(exc)) from None
    except ManualDataError:
        raise
    except Exception as exc:
        raise ManualDataError(f"cannot load {entry.path}: {exc}") from exc

    if not array.flags.c_contiguous:
        array = numpy.ascontiguousarray(array)
    expected_dtype = numpy.dtype(dtype).name
    if array.dtype.name != expected_dtype:
        raise ManualDataError(f"{entry.path} dtype {array.dtype.name!r} != filename dtype {expected_dtype!r}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ManualDataError(f"{entry.path} shape {tuple(array.shape)} != expected {tuple(shape)}")
    return array


def _load_none(entry: _ManualDataFile, role: str):
    if entry.dtype != _NONE_DTYPE:
        raise ManualDataError(f"{role}[{entry.index}] should be None but file dtype is {entry.dtype!r}")
    if entry.path.stat().st_size != 0:
        raise ManualDataError(f"None marker must be empty: {entry.path}")


class ManualDataStore:
    """Write one complete per-case dataset and replay it by testcase name."""

    def __init__(self, roots: Iterable[Any]):
        if isinstance(roots, (str, os.PathLike)):
            roots = (roots,)
        self.roots = tuple(pathlib.Path(root).expanduser().resolve() for root in roots)
        if not self.roots:
            raise ManualDataError("at least one manual data directory is required")

    def case_dir(self, testcase_name: str, root_index: int = 0) -> pathlib.Path:
        return self.roots[root_index] / case_directory_name(testcase_name)

    def invalidate_case(self, testcase_name: str):
        """Remove an older same-name case before a new prepare attempt."""
        if len(self.roots) != 1:
            raise ManualDataError("manual data preparation requires exactly one output directory")
        self._remove_existing(self.case_dir(testcase_name))

    def write_case(
        self,
        testcase,
        case_type: str,
        inputs,
        goldens,
        scalars=(),
        file_format: str = "bin",
        write_goldens: bool = True,
    ) -> pathlib.Path:
        self._validate_case_type(case_type)
        if file_format not in SUPPORTED_MANUAL_DATA_FORMATS:
            raise ManualDataError(
                f"manual data format {file_format!r} is not supported; "
                f"choose {', '.join(SUPPORTED_MANUAL_DATA_FORMATS)}"
            )
        if len(self.roots) != 1:
            raise ManualDataError("manual data preparation requires exactly one output directory")

        all_input_values = list(inputs or ())
        scalar_values = list(scalars or ())
        golden_values = list(goldens or ()) if write_goldens else []
        full_input_specs = _input_specs(testcase, case_type)
        scalar_specs = _scalar_specs(testcase, case_type)
        self._validate_value_count("input", all_input_values, len(full_input_specs))
        saved_input_indexes = _saved_input_indexes(testcase, case_type, len(full_input_specs))
        input_values = [all_input_values[index] for index in saved_input_indexes]
        input_specs = [full_input_specs[index] for index in saved_input_indexes]
        self._validate_value_count("scalar", scalar_values, len(scalar_specs))
        if write_goldens:
            expected_goldens = _expected_golden_count(testcase, case_type)
            if expected_goldens is not None:
                self._validate_value_count("golden", golden_values, expected_goldens)

        root = self.roots[0]
        root.mkdir(parents=True, exist_ok=True)
        target = self.case_dir(testcase.testcase_name)
        self._remove_existing(target)
        temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(root)))
        try:
            self._write_values(temporary, input_values, "input", file_format, input_specs)
            self._write_values(temporary, scalar_values, "scalar", file_format, scalar_specs)
            if write_goldens:
                self._write_values(temporary, golden_values, "golden", file_format)
            self._scan_case(temporary)
            os.replace(str(temporary), str(target))
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise
        roles = "input/golden" if write_goldens else "input"
        logging.info("[%s] dumped restorable %s manual data at %s", testcase.testcase_name, roles, target)
        return target

    @staticmethod
    def _validate_case_type(case_type: str):
        if case_type not in ("e2e", "aclnn", "kernel", "geir"):
            raise ManualDataError(f"unsupported manual data case type: {case_type}")

    @staticmethod
    def _validate_value_count(role: str, values: Sequence[Any], expected: int):
        if len(values) != expected:
            raise ManualDataError(f"{role} slot count {len(values)} != CSV {expected}")

    @staticmethod
    def _remove_existing(path: pathlib.Path):
        # A symlink must be unlinked before directory handling can follow it.
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(str(path))

    def _write_values(
        self,
        case_dir: pathlib.Path,
        values: Sequence[Any],
        role: str,
        file_format: str,
        specs: Optional[Sequence[Optional[Tuple[str, Tuple[int, ...]]]]] = None,
    ):
        for index, value in enumerate(values):
            expected = specs[index] if specs is not None else None
            self._write_single(case_dir, index, value, expected, role, file_format, specs is not None)

    def _write_single(self, case_dir, index, value, expected, role, file_format, has_specs):
        if value is None:
            if has_specs and expected is not None:
                raise ManualDataError(f"{role}[{index}] is None but CSV declares a tensor")
            path = case_dir / f"{role}_{index}_{_NONE_DTYPE}.{file_format}"
            path.touch()
            return
        if has_specs and expected is None:
            raise ManualDataError(f"{role}[{index}] has data but CSV declares None")

        label = f"{role}[{index}]"
        array = _as_numpy(value, label)
        if expected is not None:
            expected_dtype, expected_shape = expected
            if array.dtype.name != expected_dtype:
                raise ManualDataError(
                    f"{role}[{index}] dtype {array.dtype.name!r} != CSV storage dtype {expected_dtype!r}"
                )
            if tuple(array.shape) != expected_shape:
                raise ManualDataError(
                    f"{role}[{index}] shape {tuple(array.shape)} != CSV storage shape {expected_shape}"
                )

        dtype_name = array.dtype.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dtype_name):
            raise ManualDataError(f"dtype {dtype_name!r} cannot be encoded in a data filename")
        stem = f"{role}_{index}_{dtype_name}"
        path = case_dir / f"{stem}.{file_format}"
        self._write_array(path, array, stem, file_format)
        # golden 不编码 shape（当前协议）：reshape 真值 = 读侧（load_goldens 的
        # references/shapes）传入的设备输出 shape；写侧自校验按实际 shape 还原
        restored = _load_array(
            _ManualDataFile(role, index, dtype_name, file_format, path, None),
            tuple(array.shape),
        )
        is_packed_4bit = file_format == "bin" and is_4bit_dtype(dtype_name)
        if is_packed_4bit:
            verified = numpy.array_equal(restored, array)
        else:
            verified = restored.tobytes(order="C") == array.tobytes(order="C")
        if not verified:
            raise ManualDataError(f"{path} did not pass write/read verification")

    @staticmethod
    def _write_array(path: pathlib.Path, array: numpy.ndarray, stem: str, file_format: str):
        if file_format == "bin":
            get_formatter(file_format).dump(array, str(path.parent), stem)
            return
        if file_format == "npy":
            payload = array
            if array.dtype.name in _CUSTOM_NUMPY_DTYPES:
                payload = array.view(numpy.dtype(f"V{array.dtype.itemsize}"))
            get_formatter(file_format).dump(payload, str(path.parent), stem)
            return

        try:
            import torch

            try:
                payload = numpy_to_torch_tensor(array)
            except Exception:
                raw = numpy.frombuffer(array.tobytes(order="C"), dtype=numpy.uint8).copy()
                payload = {
                    "ttk_raw_bytes": torch.from_numpy(raw),
                    "ttk_shape": tuple(array.shape),
                }
            torch.save(payload, str(path))
        except Exception as exc:
            raise ManualDataError(f"cannot write pt file {path}: {exc}") from exc

    def _find_case(self, testcase_name: str) -> pathlib.Path:
        checked = []
        for root in self.roots:
            case_dir = root / case_directory_name(testcase_name)
            checked.append(str(case_dir))
            if case_dir.exists():
                if case_dir.is_symlink() or not case_dir.is_dir():
                    raise ManualDataError(f"manual-data case path is not a regular directory: {case_dir}")
                return case_dir
        raise ManualDataError(f"manual-data case {testcase_name!r} was not found; checked: " + ", ".join(checked))

    @staticmethod
    def _scan_case(case_dir: pathlib.Path):
        entries_by_format = {file_format: {} for file_format in SUPPORTED_MANUAL_DATA_FORMATS}
        xpu_by_format = {file_format: {} for file_format in SUPPORTED_MANUAL_DATA_FORMATS}
        ignored_debug_files = []
        try:
            paths = sorted(case_dir.iterdir(), key=lambda item: item.name)
        except Exception as exc:
            raise ManualDataError(f"cannot list manual-data case directory {case_dir}: {exc}") from exc
        if not paths:
            raise ManualDataError(f"manual-data case directory is empty: {case_dir}")

        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ManualDataError(f"manual-data case contains a non-regular file: {path}")
            xpu_match = _XPU_FILE_NAME.fullmatch(path.name)
            match = _DATA_FILE_NAME.fullmatch(path.name)
            if xpu_match is not None and match is None:
                provider, index_text, dtype_name, current_format = xpu_match.groups()
                key = (provider, int(index_text))
                format_entries = xpu_by_format[current_format]
                if key in format_entries:
                    raise ManualDataError(f"duplicate {current_format} manual-data slot xpu[{provider}][{index_text}]")
                if dtype_name != _NONE_DTYPE:
                    _resolve_file_dtype(dtype_name)
                format_entries[key] = _ManualDataFile(
                    "xpu", int(index_text), dtype_name, current_format, path, None, provider
                )
                continue
            if match is not None:
                role, index_text, dtype_name, shape_token, current_format = match.groups()
                # shape 段仅存量数据携带（早期版本落盘）：兼容解析用于 load 侧
                # 核对/reshape 兜底，新数据无此段；出现于其他位置仍是协议违规
                if shape_token is not None and (
                    role != "golden" or current_format != "bin" or dtype_name == _NONE_DTYPE
                ):
                    raise ManualDataError(f"shape suffix is only valid for non-None bin Golden files: {path.name}")
                key = (role, int(index_text))
                format_entries = entries_by_format[current_format]
                if key in format_entries:
                    raise ManualDataError(f"duplicate {current_format} manual-data slot {role}[{index_text}]")
                if dtype_name != _NONE_DTYPE:
                    _resolve_file_dtype(dtype_name)
                format_entries[key] = _ManualDataFile(
                    role,
                    int(index_text),
                    dtype_name,
                    current_format,
                    path,
                    _decode_shape(shape_token),
                )
                continue
            debug_input_match = _DEBUG_INPUT_FILE_NAME.fullmatch(path.name)
            if debug_input_match is not None:
                segment, index_text, dtype_name, current_format = debug_input_match.groups()
                if segment in _DEBUG_INPUT_REJECTED_SEGMENTS:
                    raise ManualDataError(
                        f"post-execution dump-on-fail input {path.name} cannot be replayed "
                        "(in-place APIs may have overwritten it); re-dump with --no-prof --dump in,golden"
                    )
                if segment not in _DEBUG_INPUT_ACCEPTED_SEGMENTS:
                    raise ManualDataError(f"unexpected file in manual-data case: {path.name}")
                key = ("input", int(index_text))
                format_entries = entries_by_format[current_format]
                existing = format_entries.get(key)
                if existing is not None:
                    if not _same_bytes(existing.path, path):
                        raise ManualDataError(
                            f"conflicting input slot {index_text}: {existing.path.name} != {path.name}"
                        )
                    continue
                if dtype_name != _NONE_DTYPE:
                    _resolve_file_dtype(dtype_name)
                format_entries[key] = _ManualDataFile("input", int(index_text), dtype_name, current_format, path, None)
                continue
            if _DEBUG_OUTPUT_FILE_NAME.fullmatch(path.name) or _DEBUG_TILING_FILE_NAME.fullmatch(path.name):
                ignored_debug_files.append(path.name)
                continue
            raise ManualDataError(f"unexpected file in manual-data case: {path.name}")

        if ignored_debug_files:
            logging.info(
                "manual-data case %s: ignoring debug-dump artifacts %s (replayed by re-execution)",
                case_dir,
                sorted(ignored_debug_files),
            )

        available_formats = [
            file_format
            for file_format in SUPPORTED_MANUAL_DATA_FORMATS
            if entries_by_format[file_format] or xpu_by_format[file_format]
        ]
        selected_format = available_formats[0]
        if len(available_formats) > 1:
            logging.info(
                "manual-data case %s contains formats %s; selected %s by priority",
                case_dir,
                available_formats,
                selected_format,
            )
        xpu_entries = tuple(
            sorted(xpu_by_format[selected_format].values(), key=lambda entry: (entry.index, entry.provider))
        )
        return entries_by_format[selected_format], xpu_entries, selected_format

    @staticmethod
    def _ordered_files(entries, role: str, expected_count: Optional[int], required: bool = True):
        files = sorted(
            (entry for (entry_role, _), entry in entries.items() if entry_role == role),
            key=lambda entry: entry.index,
        )
        indexes = [entry.index for entry in files]
        if indexes != list(range(len(files))):
            raise ManualDataError(f"{role} file indexes must be contiguous from zero, got {indexes}")
        count_mismatch = expected_count is not None and len(files) != expected_count
        if count_mismatch and (required or files):
            raise ManualDataError(f"{role} slot count {len(files)} != CSV {expected_count}")
        return files

    @staticmethod
    def _load_expected_values(files, specs, role: str) -> List[Any]:
        values = []
        for entry, expected in zip(files, specs):
            if expected is None:
                values.append(_load_none(entry, role))
                continue
            expected_dtype, expected_shape = expected
            if entry.dtype == _NONE_DTYPE:
                raise ManualDataError(f"{role}[{entry.index}] is None but CSV declares a tensor")
            logging.info("Reading manual %s file: %s ...", role, entry.path)
            load_entry = entry
            if entry.dtype != expected_dtype:
                raw_bin_compatible = (
                    entry.file_format == "bin"
                    and "uint8" in (entry.dtype, expected_dtype)
                    and ({entry.dtype, expected_dtype} - {"uint8"}) <= _RAW_UINT8_COMPATIBLE_DTYPES
                )
                if not raw_bin_compatible:
                    raise ManualDataError(
                        f"{role}[{entry.index}] filename dtype {entry.dtype!r} != CSV storage dtype {expected_dtype!r}"
                    )
                logging.info(
                    "Reinterpreting raw bin %s[%s] from %s to %s",
                    role,
                    entry.index,
                    entry.dtype,
                    expected_dtype,
                )
                load_entry = replace(entry, dtype=expected_dtype)
            values.append(_load_array(load_entry, expected_shape))
        return values

    def load_case(self, testcase, case_type: str, require_goldens: bool = True) -> ManualDataCase:
        self._validate_case_type(case_type)
        case_dir = self._find_case(testcase.testcase_name)
        entries, xpu_entries, file_format = self._scan_case(case_dir)
        full_input_specs = _input_specs(testcase, case_type)
        saved_input_indexes = _saved_input_indexes(testcase, case_type, len(full_input_specs))
        compact_input_specs = [full_input_specs[index] for index in saved_input_indexes]
        input_files = self._ordered_files(entries, "input", None)
        if len(input_files) == len(full_input_specs):
            input_specs = full_input_specs
            compact_inputs = False
        elif len(input_files) == len(compact_input_specs):
            input_specs = compact_input_specs
            compact_inputs = True
        else:
            raise ManualDataError(
                f"input slot count {len(input_files)} != CSV replay inputs "
                f"{len(compact_input_specs)} or legacy tensors {len(full_input_specs)}"
            )
        scalar_specs = _scalar_specs(testcase, case_type)
        scalar_files = self._ordered_files(entries, "scalar", len(scalar_specs))

        golden_count = _expected_golden_count(testcase, case_type)
        golden_files = self._ordered_files(entries, "golden", golden_count, required=require_goldens)
        inputs = self._load_expected_values(input_files, input_specs, "input")
        if compact_inputs and len(compact_input_specs) != len(full_input_specs):
            inputs = _restore_pure_outputs(testcase, inputs, full_input_specs, saved_input_indexes)
        scalars = self._load_expected_values(scalar_files, scalar_specs, "scalar")
        logging.info(
            "[%s] loaded manual data (%d input, %d scalar, %d golden, %d xpu) from %s",
            testcase.testcase_name,
            len(inputs),
            len(scalars),
            len(golden_files),
            len(xpu_entries),
            case_dir,
        )
        return ManualDataCase(inputs, scalars, case_dir, file_format, tuple(golden_files), xpu_entries)
