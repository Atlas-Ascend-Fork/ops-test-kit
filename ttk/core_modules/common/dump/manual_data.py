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
import logging
import os
import pathlib
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy

from ....utilities import resolve_numpy_dtypes
from ....utilities.data import constant_array
from ....utilities.dtypes import torch_to_numpy_tensor
from .formatter import (
    _NONE_DTYPE,
    _RAW_UINT8_COMPATIBLE_DTYPES,
    ManualDataError,
    _expected_bin_size,
    _resolve_file_dtype,
    _saved_array_shape,
    get_formatter,
    storage_layout,
)
from .loader import DataLoader, DataLoadError
from .naming import case_directory_name, decode_shape, encode_shape

SUPPORTED_MANUAL_DATA_FORMATS = ("bin", "npy", "pt")


_DATA_FILE_NAME = re.compile(
    r"^(input|scalar|golden)_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)"
    r"(?:_+shape_(scalar|[0-9]+(?:x[0-9]+)*))?\.(bin|pt|npy)$"
)
# XPU 三方输出槽位（--dump xpu 落盘协议）：xpu_{provider}_{idx}_{dtype}[_shape_{dims}].{ext}
# provider 占 segment 位（与 DataDumper.dump_xpu 命名一致），回放按 idx 对齐；
# 多个 provider 同一 idx 时取字典序第一个。bin 靠文件名 shape 段自描述，
# 没有声明或 device_outputs shape 时按该段还原。


_XPU_FILE_NAME = re.compile(
    r"^xpu_([A-Za-z0-9][A-Za-z0-9_.-]*?)_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)"
    r"(?:_+shape_(scalar|[0-9]+(?:x[0-9]+)*))?\.(bin|pt|npy)$"
)
# 调试 dump（{root}/dump/{case}/）产物的读回放规则：
# - output[_{segment}]_{idx}_{dtype} / tiling_data[_{segment}]：执行时派生产物，
#   回放会重新执行/编译产出，扫描时忽略，只把目录标记为"含调试产物"；
# - input_{segment}_{idx}_{dtype}：按段分流。dyn（kernel 执行前）/const、dynamic
#   （geir 数据集）映射到 input 槽；eager/graph_*（e2e dump-on-fail 执行后补写，
#   in-place API 可能已改写）和未知段直接拒绝。
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
    """manual-data 数据集对外句柄：读视图 + 写委托（IO 由内部 store 完成）。

    worker 只通过本类使用数据集；ManualDataStore 是内部持久层。
    """

    inputs: List[Any]
    scalars: List[Any]
    case_dir: Optional[pathlib.Path]  # dump 方向句柄（store.dump_case）不落盘、无目录
    file_format: str
    golden_files: Tuple[_ManualDataFile, ...]
    xpu_files: Tuple[_ManualDataFile, ...] = ()  # 按 (index, provider) 排序
    store: Optional["ManualDataStore"] = field(default=None, repr=False, compare=False)
    testcase_name: str = field(default="", repr=False, compare=False)

    def dump(self, testcase, case_type: str, inputs, goldens, switches, scalars=()) -> pathlib.Path:
        """数据集落盘（判定条件与 dump_manual_dataset 一致），委托内部 store。"""
        if self.store is None:
            raise ManualDataError("ManualDataCase has no backing store for write")
        return self.store.dump(testcase, case_type, inputs, goldens, switches, scalars)

    def dump_xpu(self, third_parties: dict, switches) -> None:
        """三方输出槽位写入数据集，委托内部 store。"""
        if self.store is None:
            raise ManualDataError("ManualDataCase has no backing store for write")
        self.store.dump_xpu(self.testcase_name, third_parties, file_format=switches.dump_config.file_format)

    @property
    def has_goldens(self) -> bool:
        return bool(self.golden_files)

    @property
    def has_third_parties(self) -> bool:
        """case 目录含 xpu_{provider}_{idx}_{dtype} 槽位文件（--dump xpu 回放）。"""
        return bool(self.xpu_files)

    def third_party_paths(self) -> tuple:
        """三方输出文件路径，按槽位 idx 对齐（0..max 连续，缺位补 None）。

        多个 provider 同一 idx 时取字典序第一个（xpu_files 已按
        (index, provider) 排序）。"""
        if not self.xpu_files:
            return ()
        by_index: Dict[int, str] = {}
        for entry in self.xpu_files:
            by_index.setdefault(entry.index, str(entry.path))
        count = max(by_index) + 1
        return tuple(by_index.get(index) for index in range(count))

    def third_party_saved_shapes(self) -> tuple:
        """三方输出文件名 shape 段（无段槽为 None），顺序与 :meth:`third_party_paths` 一致。"""
        if not self.xpu_files:
            return ()
        by_index: Dict[int, Tuple[int, ...]] = {}
        for entry in self.xpu_files:
            by_index.setdefault(entry.index, entry.saved_shape)
        count = max(by_index) + 1
        return tuple(by_index.get(index) for index in range(count))

    def third_party_dtypes(self) -> tuple:
        """三方输出文件名 dtype（none 槽为 None），顺序与 :meth:`third_party_paths` 一致。"""
        if not self.xpu_files:
            return ()
        by_index: Dict[int, Optional[str]] = {}
        for entry in self.xpu_files:
            by_index.setdefault(entry.index, None if entry.dtype == _NONE_DTYPE else entry.dtype)
        count = max(by_index) + 1
        return tuple(by_index.get(index) for index in range(count))

    def load_xpu(self, testcase, device_outputs=None) -> Optional[list]:
        """装载三方输出槽位：按 third_party_* 视图委托内部 store。"""
        if self.store is None:
            raise ManualDataError("ManualDataCase has no backing store for xpu loading")
        return self.store.load_xpu(
            testcase,
            self.third_party_paths(),
            self.third_party_saved_shapes(),
            self.third_party_dtypes(),
            device_outputs=device_outputs,
        )

    def load_goldens(self, device_outputs=None, shapes=None, dtypes=None) -> List[Any]:
        """Load saved goldens after device output shapes are available.

        Raw ``bin`` payloads do not carry shape metadata, so non-None bin Golden
        filenames encode the storage shape (``_shape_<dims>``; legacy files may
        use ``__shape_``). The self-describing shape is the reshape truth and
        is NOT cross-checked against reader-side shapes (written from the same
        array, trustworthy; byte-size/dtype checks in ``_load_array`` catch
        corruption). Files without a shape segment fall back to reader-side
        shapes. E2E passes the executed device outputs (``device_outputs``) as
        shape truth, while ACLNN passes declared output shapes and the CSV
        output dtypes. A cache may carry additional operator-specific Goldens;
        those are restored from their own file metadata and left for a custom
        compare hook to interpret.
        """
        if self.store is None:
            raise ManualDataError("ManualDataCase has no backing store for golden loading")
        storage_shapes = self._resolve_golden_shapes(device_outputs, shapes, dtypes)
        return self.store.load_goldens(self.golden_files, storage_shapes)

    def _resolve_golden_shapes(self, device_outputs, shapes, dtypes) -> List[Any]:
        """解析 golden 存储 shape：来自设备输出或声明槽；None 表示抑制或未映射。"""
        if device_outputs is not None and shapes is not None:
            raise ManualDataError("device outputs and explicit shapes are mutually exclusive")

        storage_shapes = []
        if device_outputs is not None:
            for index, value in enumerate(device_outputs):
                if value is None or isinstance(value, str):
                    storage_shapes.append(None)
                else:
                    storage_shapes.append(tuple(_as_numpy(value, f"output[{index}]").shape))
        elif shapes is not None:
            shape_values = list(shapes)
            dtype_values = list(dtypes or ())
            if dtypes is not None and len(dtype_values) != len(shape_values):
                raise ManualDataError(
                    f"golden dtype count {len(dtype_values)} != output shape count {len(shape_values)}"
                )
            for index, shape in enumerate(shape_values):
                if shape is None:
                    storage_shapes.append(None)
                    continue
                if dtypes is None:
                    storage_shapes.append(_saved_array_shape(shape))
                else:
                    storage_shapes.append(storage_layout(dtype_values[index], shape)[1])
        else:
            storage_shapes = [None] * len(self.golden_files)

        golden_count = len(self.golden_files)
        output_count = len(storage_shapes)
        if golden_count < output_count:
            raise ManualDataError(f"golden slot count {golden_count} < device output count {output_count}")
        if golden_count > output_count:
            # Different APIs of one operator may expose different output subsets,
            # so no positional shape mapping is valid when their counts differ.
            storage_shapes = [None] * golden_count
        return storage_shapes


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


def _all_roots_empty(directories) -> bool:
    return all(not (root.exists() and any(root.iterdir())) for root in directories)


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
            result.append(storage_layout(dtypes[index], shape))
        else:
            result.append(storage_layout(dtypes[index], testcase.flat_storage_shape(index)))
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
            result.append(storage_layout(dtype, ()))
    return result


def _expected_golden_count(testcase, case_type: str) -> Optional[int]:
    # golden 槽数由 testcase 声明（kernel/geir 的 shape-unknown 元数据槽等
    # 读写侧差异在通路侧处理）；没声明时回退到 flat 输出列数
    if case_type in ("aclnn", "kernel", "geir"):
        declared = getattr(testcase, "manual_data_golden_dtypes", None)
        if declared is not None:
            return len(declared)
        return len(getattr(testcase, "flat_output_dtypes", ()) or ())
    return None


def _expected_write_golden_count(testcase, case_type: str) -> Optional[int]:
    """写侧 golden 槽数核对基准。

    aclnn/kernel/geir 与回放侧同源；e2e 输出为执行派生，回放侧不校验
    （额外 operator-specific golden 合法），仅写侧按 flat 输出声明核——
    声明为空（纯副作用算子，golden 为被写参数回读）不核。
    """
    if case_type == "e2e":
        return len(getattr(testcase, "flat_output_dtypes", ()) or ()) or None
    return _expected_golden_count(testcase, case_type)


def _same_bytes(left: pathlib.Path, right: pathlib.Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    if left.samefile(right):
        return True
    return filecmp.cmp(str(left), str(right), shallow=False)


class ManualDataStore:
    """Write one complete per-case dataset and replay it by testcase name."""

    @classmethod
    def dump_case(cls, testcase, switches) -> Optional[ManualDataCase]:
        """Dump 方向数据集句柄：单输出目录，旧同名 case 先失效；未配目录返回 None。"""
        directories = resolve_manual_data_directories(switches, required=False)
        if not directories:
            return None
        if len(directories) > 1:
            raise ManualDataError("manual data output requires exactly one --manual-data-dirs path")
        store = cls(directories)
        store.invalidate_case(testcase.testcase_name)
        return ManualDataCase([], [], None, "", (), store=store, testcase_name=testcase.testcase_name)

    @classmethod
    def load(
        cls,
        testcase,
        case_type: str,
        switches,
        notify_before_load: Optional[Callable[[], None]] = None,
        require_goldens: bool = True,
    ) -> Optional[ManualDataCase]:
        """CLI 目录装载：no-prof / 未配目录 / 全空目录返回 None（正常生成）。"""
        if getattr(switches, "no_prof", False):
            return None
        directories = resolve_manual_data_directories(switches, required=False)
        if not directories or _all_roots_empty(directories):
            return None
        # 回放就是复用已存 golden，所以只禁止 Disable：Disable 的 SUPPRESSED 分支
        # 在 manual 装载之后才判断，放行会让本应跳过的生成结果参与比对。
        # XPU / NPU / Promote 放行。
        if getattr(switches, "golden_mode", "Enable") == "Disable":
            raise ManualDataError("manual-data loading cannot be combined with --golden-mode Disable")
        if getattr(switches, "validate_only", False):
            raise ManualDataError("manual-data loading cannot be combined with --validate")
        if case_type == "kernel" and getattr(switches, "compile_only", False):
            raise ManualDataError("manual-data loading cannot be combined with Kernel --compile-only")
        if case_type == "e2e" and getattr(switches, "force_cpu", False):
            raise ManualDataError("manual-data loading is the device stage and cannot use --cpu")
        store = cls(directories)
        if notify_before_load is not None:
            notify_before_load()
        return store.load_case(testcase, case_type, require_goldens=require_goldens)

    def dump(self, testcase, case_type: str, inputs, goldens, switches, scalars=()) -> pathlib.Path:
        """数据集落盘：判定 --dump golden 且无占位串后委托 write_case。

        golden 槽数写侧核对（回放侧仅 aclnn/kernel/geir 兜底、e2e 须宽松，
        错位真值只能在制备侧拦——写侧漏检会让回放比对静默错）。
        """
        golden_values = tuple(goldens or ())
        write_goldens = bool(getattr(switches.dump_config, "dump_golden", False)) and not any(
            isinstance(g, str) for g in golden_values
        )
        # 空 golden 合法（golden-XPU 首跑回落 input-only：真值设备后回填），
        # 仅非空时核数——错位真值落盘会让回放比对静默错
        if write_goldens and golden_values:
            expected_goldens = _expected_write_golden_count(testcase, case_type)
            if expected_goldens is not None and len(golden_values) != expected_goldens:
                raise ManualDataError(f"golden slot count {len(golden_values)} != CSV {expected_goldens}")
        return self.write_case(
            testcase,
            case_type,
            inputs,
            golden_values if write_goldens else (),
            scalars=scalars,
            file_format=switches.dump_config.file_format,
            write_goldens=write_goldens,
        )

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
        all_input_values = list(inputs or ())
        scalar_values = list(scalars or ())
        golden_values = list(goldens or ()) if write_goldens else []
        full_input_specs = _input_specs(testcase, case_type)
        scalar_specs = _scalar_specs(testcase, case_type)
        saved_input_indexes = _saved_input_indexes(testcase, case_type, len(full_input_specs))
        input_values = [all_input_values[index] for index in saved_input_indexes]
        input_specs = [full_input_specs[index] for index in saved_input_indexes]

        root = self.roots[0]
        root.mkdir(parents=True, exist_ok=True)
        target = self.case_dir(testcase.testcase_name)
        self._remove_existing(target)
        temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(root)))
        try:
            self.dump_inputs(temporary, input_values, input_specs, file_format)
            self.dump_inputs(temporary, scalar_values, scalar_specs, file_format, role="scalar")
            if write_goldens:
                self.dump_goldens(temporary, golden_values, file_format)
            self._scan_case(temporary)
            os.replace(str(temporary), str(target))
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise
        roles = "input/golden" if write_goldens else "input"
        logging.info("[%s] dumped restorable %s manual data at %s", testcase.testcase_name, roles, target)
        return target

    @staticmethod
    def _remove_existing(path: pathlib.Path):
        # A symlink must be unlinked before directory handling can follow it.
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(str(path))

    def dump_inputs(
        self,
        case_dir: pathlib.Path,
        inputs: Sequence[Any],
        specs: Sequence[Optional[Tuple[str, Tuple[int, ...]]]],
        file_format: str,
        role: str = "input",
    ) -> None:
        """数据集 input/scalar 槽写入（含 CSV 声明校验与 None 标记）。"""
        for index, value in enumerate(inputs):
            expected = specs[index]
            if value is None:
                if expected is not None:
                    raise ManualDataError(f"{role}[{index}] is None but CSV declares a tensor")
                self._write_single(case_dir, index, None, None, role, file_format)
                continue
            if expected is None:
                raise ManualDataError(f"{role}[{index}] has data but CSV declares None")
            self._write_single(case_dir, index, value, expected, role, file_format)

    def dump_goldens(self, case_dir: pathlib.Path, goldens: Sequence[Any], file_format: str) -> None:
        """数据集 golden 槽写入（含 shape 段与 None 标记）。"""
        for index, value in enumerate(goldens):
            self._write_single(case_dir, index, value, None, "golden", file_format)

    def _write_single(self, case_dir, index, value, expected, role, file_format):
        if value is None:
            (case_dir / f"{role}_{index}_{_NONE_DTYPE}.{file_format}").touch()
            return

        label = f"{role}[{index}]"
        logical_shape = _value_shape(value, label)
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

        dtype_name = _encode_dtype_name(array.dtype.name)
        stem = f"{role}_{index}_{dtype_name}"
        saved_shape = None
        if role == "golden":
            saved_shape = logical_shape if not logical_shape and array.size == 1 else tuple(array.shape)
            stem = _stem_with_bin_shape(stem, file_format, saved_shape)
        path = case_dir / f"{stem}.{file_format}"
        get_formatter(file_format).dump(array, str(path.parent), stem)

    def dump_xpu(self, testcase_name: str, third_parties: dict, file_format: str = "bin") -> None:
        """三方输出槽位追加写入已有 case 目录（采集发生在 input/golden 之后）。

        命名与 --dump xpu 协议一致：xpu_{provider}_{idx}_{dtype}[_shape_{token}].{ext}，
        bin 靠文件名 shape 段自描述；None/str 槽跳过但保留 idx；回放时
        _scan_case 按文件名对齐读取。provider 名校验同 DataDumper.dump_xpu。
        """
        from .dumper import _validate_xpu_provider

        case_dir = self._find_case(testcase_name)
        for provider, outputs in (third_parties or {}).items():
            _validate_xpu_provider(provider)
            for idx, data in enumerate(outputs):
                if data is None or isinstance(data, str):
                    continue
                label = f"xpu[{provider}][{idx}]"
                array = _as_numpy(data, label)
                dtype_name = _encode_dtype_name(array.dtype.name)
                stem = _stem_with_bin_shape(f"xpu_{provider}_{idx}_{dtype_name}", file_format, tuple(array.shape))
                get_formatter(file_format).dump(array, str(case_dir), stem)

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
                provider, index_text, dtype_name, shape_token, current_format = xpu_match.groups()
                if shape_token is not None and current_format != "bin":
                    raise ManualDataError(f"shape suffix is only valid for bin XPU files: {path.name}")
                key = (provider, int(index_text))
                format_entries = xpu_by_format[current_format]
                if key in format_entries:
                    raise ManualDataError(f"duplicate {current_format} manual-data slot xpu[{provider}][{index_text}]")
                if dtype_name != _NONE_DTYPE:
                    _resolve_file_dtype(dtype_name)
                format_entries[key] = _ManualDataFile(
                    "xpu", int(index_text), dtype_name, current_format, path, decode_shape(shape_token), provider
                )
                continue
            if match is not None:
                role, index_text, dtype_name, shape_token, current_format = match.groups()
                # shape 段由写侧写入文件名（`_shape_`，存量数据是 `__shape_`）：
                # 解析后供 load 侧自描述还原；出现在其他位置仍是协议违规
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
                    decode_shape(shape_token),
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

    def load_case(self, testcase, case_type: str, require_goldens: bool = True) -> ManualDataCase:
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
        inputs = self.load_inputs(input_files, input_specs, "input")
        if compact_inputs:
            inputs = _restore_pure_outputs(testcase, inputs, full_input_specs, saved_input_indexes)
        scalars = self.load_inputs(scalar_files, scalar_specs, "scalar")
        logging.info(
            "[%s] loaded manual data (%d input, %d scalar, %d golden, %d xpu) from %s",
            testcase.testcase_name,
            len(inputs),
            len(scalars),
            len(golden_files),
            len(xpu_entries),
            case_dir,
        )
        return ManualDataCase(
            inputs,
            scalars,
            case_dir,
            file_format,
            tuple(golden_files),
            xpu_entries,
            store=self,
            testcase_name=testcase.testcase_name,
        )

    @staticmethod
    def load_goldens(golden_files, storage_shapes):
        """装载 golden 槽：saved shape 段优先，委托 DataLoader.load_goldens。"""
        paths, dtypes, shapes = [], [], []
        for entry, storage_shape in zip(golden_files, storage_shapes):
            if entry.dtype == _NONE_DTYPE:
                if entry.path.stat().st_size != 0:
                    raise ManualDataError(f"None marker must be empty: {entry.path}")
                paths.append(None)
                dtypes.append(None)
                shapes.append(None)
                continue
            resolved_storage_shape = (entry.saved_shape or (1,)) if entry.saved_shape is not None else storage_shape
            if resolved_storage_shape is None and entry.file_format == "bin" and entry.dtype != _NONE_DTYPE:
                logging.warning(
                    "golden[%s] bin payload %s loaded as a flat stream "
                    "(no reader-side shape mapping); custom compare hook interprets",
                    entry.index,
                    entry.path.name,
                )
            if entry.file_format == "bin" and resolved_storage_shape is not None:
                expected_size = _expected_bin_size(entry.dtype, resolved_storage_shape)
                if entry.path.stat().st_size != expected_size:
                    raise ManualDataError(
                        f"{entry.path} byte size {entry.path.stat().st_size} != expected {expected_size}"
                    )
            paths.append(entry.path)
            dtypes.append(_resolve_file_dtype(entry.dtype))
            shapes.append(resolved_storage_shape)
        arrays = DataLoader.load_goldens(paths, dtypes, shapes)
        for entry, array, resolved in zip(golden_files, arrays, shapes):
            if array is None or entry.dtype == _NONE_DTYPE:
                continue
            _check_loaded(entry, array, entry.dtype, resolved)
        return arrays

    @staticmethod
    def load_inputs(files, specs, role: str):
        """装载 input/scalar 槽：校验 dtype 与 none 标记后委托 DataLoader.load_inputs。"""
        paths, dtypes, shapes = [], [], []
        for entry, expected in zip(files, specs):
            if expected is None:
                if entry.dtype != _NONE_DTYPE:
                    raise ManualDataError(f"{role}[{entry.index}] should be None but file dtype is {entry.dtype!r}")
                if entry.path.stat().st_size != 0:
                    raise ManualDataError(f"None marker must be empty: {entry.path}")
                paths.append(None)
                dtypes.append(None)
                shapes.append(None)
                continue
            expected_dtype, expected_shape = expected
            if entry.dtype == _NONE_DTYPE:
                raise ManualDataError(f"{role}[{entry.index}] is None but CSV declares a tensor")
            load_dtype = entry.dtype
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
                    "Reinterpreting raw bin %s[%s] from %s to %s", role, entry.index, entry.dtype, expected_dtype
                )
                load_dtype = expected_dtype
            if entry.file_format == "bin" and expected_shape is not None:
                expected_size = _expected_bin_size(load_dtype, expected_shape)
                if entry.path.stat().st_size != expected_size:
                    raise ManualDataError(
                        f"{entry.path} byte size {entry.path.stat().st_size} != expected {expected_size}"
                    )
            paths.append(entry.path)
            dtypes.append(_resolve_file_dtype(load_dtype))
            shapes.append(expected_shape)
        arrays = DataLoader.load_inputs(paths, dtypes, shapes)
        for array, entry, expected in zip(arrays, files, specs):
            if array is None or expected is None:
                continue
            expected_dtype, expected_shape = expected
            _check_loaded(entry, array, expected_dtype, expected_shape)
        return arrays

    @staticmethod
    def load_xpu(testcase, paths, saved_shapes, filename_dtypes, device_outputs=None) -> list:
        """装载 xpu 槽位，委托 DataLoader.load_xpu。

        shape/dtype 优先取声明槽（flat_output_*）；e2e 没有声明槽时用
        device_outputs；都没有时用文件名 shape 段和 dtype。缺文件或 shape
        未知的槽补 None。
        """
        shapes = getattr(testcase, "flat_output_shapes", None)
        if shapes is None:
            shapes = getattr(testcase, "flat_output_view_shapes", None)
        if not shapes and device_outputs:
            shapes = tuple(tuple(r.shape) if r is not None and hasattr(r, "shape") else None for r in device_outputs)
        shapes = shapes or ()
        dtypes = resolve_numpy_dtypes(getattr(testcase, "flat_output_dtypes", ()) or ())
        if not dtypes and device_outputs:
            dtypes = tuple(
                getattr(r, "dtype", None) if r is not None and hasattr(r, "dtype") else None for r in device_outputs
            )
        name = getattr(testcase, "testcase_name", None) or getattr(testcase, "op_name", None)
        files, load_dtypes, load_shapes, expected_storages = [], [], [], []
        for idx in range(max(len(paths), len(shapes))):
            fp = paths[idx] if idx < len(paths) else None
            declared_shape = shapes[idx] if idx < len(shapes) else None
            saved_shape = saved_shapes[idx] if idx < len(saved_shapes) else None
            if fp is None or (declared_shape is None and saved_shape is None):
                files.append(None)
                load_dtypes.append(None)
                load_shapes.append(None)
                expected_storages.append(None)
                continue
            dtype = dtypes[idx] if idx < len(dtypes) else None
            filename_dtype = filename_dtypes[idx] if idx < len(filename_dtypes) else None
            if dtype is None and declared_shape is None:
                dtype = filename_dtype
            shape = declared_shape if declared_shape is not None else (saved_shape or (1,))
            expected_storage = storage_layout(dtype, declared_shape)[1] if declared_shape is not None else shape
            files.append(fp)
            load_dtypes.append(dtype)
            load_shapes.append(shape)
            expected_storages.append(expected_storage)
        arrays = DataLoader.load_xpu(files, load_dtypes, load_shapes)
        for idx, (array, expected) in enumerate(zip(arrays, expected_storages)):
            if array is None or expected is None:
                continue
            if tuple(array.shape) != expected:
                raise DataLoadError(
                    f"[{name}] manual XPU output[{idx}] shape {tuple(array.shape)} does not match "
                    f"expected storage {expected}"
                )
        return arrays


def snapshot_manual_values(values: Sequence[Any], label: str) -> List[Any]:
    """Copy values before golden callbacks can mutate their backing storage."""
    snapshots = []
    for index, value in enumerate(values or ()):
        if value is None:
            snapshots.append(None)
        else:
            snapshots.append(_as_numpy(value, f"{label}[{index}]").copy())
    return snapshots


# ══ --dump xpu 三方输出装载/探测 ══

# manual 装载在结果里的伪 provider 名，与远端结果 dict 形态一致，屏显/报告不用改
MANUAL_XPU_PROVIDER = "manual_binary"


def manual_xpu_results() -> dict:
    """manual 装载的 xpu_results 伪条目（status=PASS，api 标记来源）。"""
    return {MANUAL_XPU_PROVIDER: {"status": "PASS", "api": "manual_xpu_binaries"}}


def has_manual_xpu_source(testcase, manual_case=None) -> bool:
    """是否声明了 manual 三方输出源（不读文件）。

    与 load_third_parties 的双来源一致：CSV ``flat_manual_xpu_binaries``
    非空，或 manual-data case 里有 xpu 槽位文件。
    """
    if getattr(testcase, "flat_manual_xpu_binaries", None):
        return True
    return manual_case is not None and bool(getattr(manual_case, "has_third_parties", False))


def _check_loaded(entry, array, expected_dtype_name, expected_shape):
    """读后复核：dtype 与 shape 要和文件名一致。"""
    expected_dtype = numpy.dtype(_resolve_file_dtype(expected_dtype_name)).name
    if array.dtype.name != expected_dtype:
        raise ManualDataError(f"{entry.path} dtype {array.dtype.name!r} != filename dtype {expected_dtype_name!r}")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ManualDataError(f"{entry.path} shape {tuple(array.shape)} != expected {tuple(expected_shape)}")


def _encode_dtype_name(dtype_name: str) -> str:
    """校验 dtype 名可以放进文件名（回放按文件名解析）。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dtype_name):
        raise ManualDataError(f"dtype {dtype_name!r} cannot be encoded in a data filename")
    return dtype_name


def _stem_with_bin_shape(stem: str, file_format: str, shape) -> str:
    """bin 不带元数据，把存储 shape 写进文件名。"""
    if file_format == "bin":
        stem += f"_shape_{encode_shape(shape)}"
    return stem


def load_csv_slots(files, dtypes, shapes) -> list:
    """逐槽装载 CSV 手工文件：三个列表同序，缺文件的槽补 None；shape 为 None
    的槽不按声明还原（bin 读回一维流，交调用方解释）。"""
    return DataLoader.load_xpu(files, dtypes, shapes)


def load_third_parties(testcase, manual_case=None, device_outputs=None) -> Optional[list]:
    """装载 manual 三方输出（--dump xpu 的对侧），两个来源：

    ① CSV ``manual_xpu_binaries`` 显式路径（kernel/aclnn/geir）；
    ② manual-data case 目录槽位，走 ManualDataCase.load_xpu。
    都没有时返回 None，调用方走远端采集。
    """
    paths = getattr(testcase, "flat_manual_xpu_binaries", None)
    if paths:
        return ManualDataStore.load_xpu(testcase, paths, (), (), device_outputs)
    if manual_case is not None and getattr(manual_case, "has_third_parties", False):
        return manual_case.load_xpu(testcase, device_outputs)
    return None
