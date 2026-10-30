#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare FA batch relations from a TTK phase-one result CSV and NPU dumps.

The script deliberately has no plugin import.  It consumes the result CSV emitted
by phase one and the raw files in ``NPU_DUMP_PATH`` so operator assets remain
responsible only for constructing equivalent inputs.
"""

import argparse
import ast
import csv
import hashlib
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPORT_HEADERS = (
    "testcase_name",
    "api_name",
    "result",
    "SHA-256",
    "errors",
    "batch_consistency_id",
    "batch_axis",
    "batch_slice_info",
    "batch_seed",
    "case_number",
    "tensor_view_shapes",
    "tensor_dtypes",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "inplace_input_indexes",
)

# Phase two needs the original case description, not a display-only --title CSV.
REQUIRED_RESULT_HEADERS = (
    "testcase_name",
    "api_name",
    "precision_status",
    "tensor_view_shapes",
    "tensor_dtypes",
    "attributes",
    "input_data_ranges",
    "batch_consistency_id",
    "batch_axis",
    "batch_slice_info",
    "batch_seed",
)

BYTE_WIDTHS = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}


@dataclass(frozen=True)
class Relation:
    """One logical Q-output slice selected by a batch relation."""

    axes: Tuple[int, ...]
    slices: Tuple[Tuple[int, int, int], ...]
    seed: int


@dataclass
class Sample:
    """A raw output slice plus the context needed to prove comparability."""

    row: Dict[str, str]
    profile: "BatchProfile"
    relation: Relation
    value: bytes
    shape: Tuple[int, ...]
    dtype: str
    context: Dict[str, Any]

    @property
    def group_key(self):
        # Slice offsets and lengths are intentionally not group identity.  The
        # established contract is seed + logical axis; context validates shape.
        return self.profile.name, self.relation.axes, self.relation.seed


def parse_cell(row: Dict[str, str], name: str, default=None):
    """Parse a TTK FREE_EVAL/DICT result field with an actionable error."""
    value = row.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        testcase_name = row.get("testcase_name", "<unknown>")
        raise ValueError(f"{testcase_name}: invalid {name}: {error}") from error


def is_enabled(row: Dict[str, str]) -> bool:
    value = row.get("is_enabled")
    if value is None or not value.strip():
        return True
    normalized = value.strip().title()
    try:
        return bool(ast.literal_eval(normalized))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"{row.get('testcase_name', '<unknown>')}: invalid is_enabled") from error


def normalize_value(value):
    """Create a deterministic, JSON-like comparison value from nested fields."""
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [normalize_value(item) for item in value]
    return value


def product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"invalid output shape {tuple(shape)!r}")
        result *= dimension
    return result


def byte_width(dtype: str) -> int:
    normalized = str(dtype).lower()
    if normalized.startswith("torch."):
        normalized = normalized[len("torch.") :]
    width = BYTE_WIDTHS.get(normalized)
    if width is None:
        raise ValueError(f"unsupported raw output dtype {dtype!r}")
    return width


def find_output_dump(dump_dir: Path, dump_names: Sequence[str], output_index: int = 0) -> Optional[Path]:
    """Locate an output .bin dump across flat and per-testcase subdirectory layouts.

    Supports the legacy flat ``{name}_output_{i}.bin`` layout and the eager-dump
    ``{name}/output_*_{i}_*.bin`` layout (e.g. ``output_eager_0_int32.bin``).
    Reject ambiguous nested matches instead of choosing an arbitrary output.
    """
    for name in dump_names:
        flat_path = dump_dir / f"{name}_output_{output_index}.bin"
        if flat_path.is_file():
            return flat_path
        case_dir = dump_dir / name
        if case_dir.is_dir():
            candidates = sorted(path for path in case_dir.glob(f"output_*_{output_index}_*.bin") if path.is_file())
            if len(candidates) > 1:
                paths = ", ".join(str(path) for path in candidates)
                raise ValueError(f"{name}: ambiguous output dump for output {output_index}; found {paths}")
            if candidates:
                return candidates[0]
    return None


def complete_batch_metadata(row: Dict[str, str]):
    """Return parsed metadata only when all three batch fields are provided.

    A deterministic-only case may set neither field.  Treat a partial triple in
    the same way instead of turning phase two into an accidental execution gate.
    """
    names = ("batch_axis", "batch_slice_info", "batch_seed")
    values = tuple(parse_cell(row, name) for name in names)
    if any(value is None for value in values):
        return None
    return values


def slice_indices(value: Tuple[int, int, int]) -> range:
    """Return the logical positions selected by one positive-step slice."""
    return range(*value)


def slices_overlap(left: Tuple[int, int, int], right: Tuple[int, int, int]) -> bool:
    """Check overlap using selected positions, not contiguous span bounds."""
    left_indices, right_indices = slice_indices(left), slice_indices(right)
    if len(left_indices) > len(right_indices):
        left_indices, right_indices = right_indices, left_indices
    return any(index in right_indices for index in left_indices)


def parse_slice(testcase_name: str, axis: int, value) -> Tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{testcase_name}: invalid logical axis {axis} slice {value!r}")
    if not all(type(item) is int for item in value):
        raise ValueError(f"{testcase_name}: logical slices must contain integers")
    start, stop, step = (int(item) for item in value)
    result = (start, stop, step)
    if step <= 0 or start < 0 or not slice_indices(result):
        raise ValueError(f"{testcase_name}: logical slice must have a non-negative start and non-empty positive step")
    return result


def parse_relations(row: Dict[str, str]) -> Optional[List[Relation]]:
    """Parse the five FA operators' Q-only batch contract from one result row."""
    metadata = complete_batch_metadata(row)
    if metadata is None:
        return None
    batch_axis, batch_slices, batch_seed = metadata
    testcase_name = row["testcase_name"]
    if not (
        isinstance(batch_axis, (tuple, list))
        and isinstance(batch_slices, (tuple, list))
        and isinstance(batch_seed, (tuple, list))
    ):
        raise ValueError(f"{testcase_name}: batch metadata must use nested sequences")
    if not (len(batch_axis) == len(batch_slices) == len(batch_seed)):
        raise ValueError(f"{testcase_name}: batch metadata top-level counts differ")
    if (
        not batch_axis
        or not isinstance(batch_axis[0], (tuple, list))
        or any(type(axis) is not int for axis in batch_axis[0])
        or tuple(batch_axis[0]) not in ((0,), (0, 1))
    ):
        raise ValueError(f"{testcase_name}: Q batch relation must use logical axes (0,) or (0, 1)")
    if batch_slices[0] is None or batch_seed[0] is None:
        raise ValueError(f"{testcase_name}: Q batch relation requires slices and seeds")
    if any(value is not None for value in batch_slices[1:]) or any(value is not None for value in batch_seed[1:]):
        raise ValueError(f"{testcase_name}: only Q batch relations are supported")

    axes = tuple(int(axis) for axis in batch_axis[0])
    slice_groups = batch_slices[0]
    seed_groups = batch_seed[0]
    if not (isinstance(slice_groups, (tuple, list)) and isinstance(seed_groups, (tuple, list))):
        raise ValueError(f"{testcase_name}: Q batch groups must be nested sequences")
    if len(slice_groups) != len(axes) or len(seed_groups) != len(axes):
        raise ValueError(f"{testcase_name}: Q batch groups do not match the declared axes")
    if not slice_groups or not isinstance(slice_groups[0], (tuple, list)):
        raise ValueError(f"{testcase_name}: Q batch relation is empty")
    sample_count = len(slice_groups[0])
    if not sample_count or any(
        not isinstance(group, (tuple, list)) or len(group) != sample_count for group in (*slice_groups, *seed_groups)
    ):
        raise ValueError(f"{testcase_name}: Q batch samples have inconsistent counts")

    relations = []
    for sample_index in range(sample_count):
        slices = []
        seed = None
        for group_index, axis in enumerate(axes):
            slices.append(parse_slice(testcase_name, axis, slice_groups[group_index][sample_index]))
            seed_value = seed_groups[group_index][sample_index]
            if type(seed_value) is not int:
                raise ValueError(f"{testcase_name}: Q relation seed must be an integer")
            if seed is not None and seed != seed_value:
                raise ValueError(f"{testcase_name}: B and S relation seeds must be identical")
            seed = int(seed_value)
        if axes == (0, 1) and len(slice_indices(slices[0])) != 1:
            raise ValueError(f"{testcase_name}: logical (B, S) needs exactly one B per sample")
        relations.append(Relation(axes, tuple(slices), seed))

    for index, relation in enumerate(relations):
        for candidate in relations[index + 1 :]:
            if relation.axes != candidate.axes:
                continue
            first_b, second_b = relation.slices[0], candidate.slices[0]
            if not slices_overlap(first_b, second_b):
                continue
            if relation.axes == (0,):
                raise ValueError(f"{testcase_name}: Q relation samples overlap logical B positions")
            first_s, second_s = relation.slices[1], candidate.slices[1]
            if slices_overlap(first_s, second_s):
                raise ValueError(f"{testcase_name}: Q relation samples overlap logical S positions")
    return relations


def validate_batch_consistency_id(row: Dict[str, str], relations: Sequence[Relation]):
    """Validate full IDs; accept checked length IDs from historical result CSVs."""
    value = parse_cell(row, "batch_consistency_id")
    if value is None:
        raise ValueError(f"{row['testcase_name']}: complete batch metadata needs batch_consistency_id")
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise ValueError(f"{row['testcase_name']}: batch_consistency_id must contain one Q relation group")
    id_groups = value[0]
    axes = relations[0].axes
    if not isinstance(id_groups, (tuple, list)) or len(id_groups) != len(axes):
        raise ValueError(f"{row['testcase_name']}: batch_consistency_id axis groups do not match Q axes")
    legacy_id = False
    for group_index, axis in enumerate(axes):
        ids = id_groups[group_index]
        if not isinstance(ids, (tuple, list)) or len(ids) != len(relations):
            raise ValueError(f"{row['testcase_name']}: batch_consistency_id sample count does not match Q relations")
        for relation_id, relation in zip(ids, relations):
            parts = str(relation_id).split("_")
            if len(parts) not in (4, 5):
                raise ValueError(f"{row['testcase_name']}: invalid batch_consistency_id relation {relation_id!r}")
            try:
                actual = tuple(int(part) for part in parts)
            except ValueError as error:
                raise ValueError(
                    f"{row['testcase_name']}: invalid batch_consistency_id relation {relation_id!r}"
                ) from error
            expected = (relation.seed, axis, *relation.slices[group_index])
            if len(parts) == 4:
                selection = slice_indices(relation.slices[group_index])
                expected = (relation.seed, axis, len(selection), selection.step)
                legacy_id = True
            if actual != expected:
                raise ValueError(
                    f"{row['testcase_name']}: batch_consistency_id relation {relation_id!r} does not match {expected!r}"
                )
    if legacy_id:
        logging.warning(
            "%s: reading legacy length-based batch_consistency_id; new phase-one results use "
            "seed_axis_start_stop_step. Comparison groups still use seed + axis.",
            row["testcase_name"],
        )


class BatchProfile:
    """Operator-family geometry and relation-context adapter."""

    def __init__(self, name: str, api_suffixes: Iterable[str], input_count: int, output_dtype=None):
        self.name = name
        self.api_suffixes = frozenset(api_suffixes)
        self.input_count = input_count
        self.output_dtype = output_dtype

    @staticmethod
    def _attributes(row):
        attributes = parse_cell(row, "attributes", {})
        if not isinstance(attributes, dict):
            raise ValueError(f"{row['testcase_name']}: attributes must be a dictionary")
        attributes = dict(attributes)
        aliases = {
            "layoutQOptional": "layout_q",
            "layoutKOptional": "layout_k",
            "layoutKvOptional": "layout_kv",
            "layout_query": "layout_q",
            "layout_key": "layout_k",
            "maxSeqlenQ": "max_seqlen_q",
            "quantMode": "quant_mode",
            "maskMode": "mask_mode",
            "cmpRatio": "cmp_ratio",
            "returnValue": "return_value",
            "softmaxScale": "softmax_scale",
            "oriMaskMode": "ori_mask_mode",
            "cmpMaskMode": "cmp_mask_mode",
            "oriWinLeft": "ori_win_left",
            "oriWinRight": "ori_win_right",
            "topkValueMode": "topk_value_mode",
            "returnSoftmaxLse": "return_softmax_lse",
            "ropeHeadDim": "rope_head_dim",
        }
        for source, target in aliases.items():
            if source in attributes:
                if target in attributes and attributes[target] != attributes[source]:
                    raise ValueError(f"{row['testcase_name']}: conflicting {source}/{target} attributes")
                attributes[target] = attributes.pop(source)
        return attributes

    def _input_row(self, row):
        """Keep ACLNN output slots out of the input-relation context."""
        if not row.get("api_name", "").startswith("aclnn"):
            return row
        shapes = parse_cell(row, "tensor_view_shapes", ())
        dtypes = parse_cell(row, "tensor_dtypes", ())
        output_indexes = parse_cell(row, "output_tensor_indexes", ())
        expected_indexes = (self.input_count, self.input_count + 1)
        if (
            not all(isinstance(value, (tuple, list)) for value in (shapes, dtypes, output_indexes))
            or tuple(output_indexes) != expected_indexes
            or len(shapes) != self.input_count + 2
            or len(dtypes) != len(shapes)
        ):
            raise ValueError(f"{row['testcase_name']}: ACLNN requires output_tensor_indexes={expected_indexes}")
        normalized = dict(row)
        normalized["tensor_view_shapes"] = repr(shapes[: self.input_count])
        normalized["tensor_dtypes"] = repr(dtypes[: self.input_count])
        return normalized

    def _validate_output_metadata(self, row, shape, dtype):
        if not row.get("api_name", "").startswith("aclnn"):
            return
        shapes = parse_cell(row, "tensor_view_shapes")
        dtypes = parse_cell(row, "tensor_dtypes")
        output_shape = shapes[self.input_count]
        if (
            not isinstance(output_shape, (tuple, list))
            or tuple(output_shape) != tuple(shape)
            or str(dtypes[self.input_count]) != dtype
        ):
            raise ValueError(f"{row['testcase_name']}: ACLNN output 0 shape/dtype does not match Q geometry")

    def supports(self, api_name: str) -> bool:
        return api_name.rsplit(".", 1)[-1] in self.api_suffixes

    def make_samples(self, row: Dict[str, str], dump_dir: Path) -> List[Sample]:
        relations = parse_relations(row)
        if relations is None:
            return []
        validate_batch_consistency_id(row, relations)
        input_row = self._input_row(row)
        geometry = self.geometry(input_row)
        output_shape, output_dtype = geometry["output_shape"], geometry["output_dtype"]
        self._validate_output_metadata(row, output_shape, output_dtype)
        output_bytes = self._read_output(row, dump_dir, output_shape, output_dtype)
        samples = []
        for relation in relations:
            value, shape = self._extract(input_row, relation, geometry, output_bytes)
            samples.append(
                Sample(row, self, relation, value, shape, output_dtype, self._context(input_row, relation, geometry))
            )
        return samples

    def geometry(self, row):
        raise NotImplementedError

    def _context(self, row, relation, geometry):
        raise NotImplementedError

    @staticmethod
    def _extract(row, relation, geometry, output_bytes):
        batch_slice = relation.slices[0]
        batch_indices = slice_indices(batch_slice)
        if batch_slice[1] > geometry["batch_size"]:
            raise ValueError(f"{row['testcase_name']}: logical B slice exceeds output batch")
        sequence_slice = relation.slices[1] if relation.axes == (0, 1) else None
        output_shape = geometry["output_shape"]
        bsnd = geometry["layout_q"] == "BSND"
        feature_shape = output_shape[2:] if bsnd else output_shape[1:]
        token_bytes = product(feature_shape) * byte_width(geometry["output_dtype"])
        chunks, lengths = [], []
        for batch_index in batch_indices:
            storage_length = geometry["q_storage_lengths"][batch_index]
            effective_length = geometry["q_effective_lengths"][batch_index]
            if effective_length == 0:
                raise ValueError(f"{row['testcase_name']}: relation selects no effective Q output")
            if sequence_slice is not None:
                if sequence_slice[1] > effective_length:
                    raise ValueError(f"{row['testcase_name']}: logical S slice exceeds Q effective length")
                positions = slice_indices(sequence_slice)
            else:
                # BSND indexers omit padding, while TND output is physically packed
                # by cu_seqlens.  Keep the complete TND span so relation geometry
                # remains comparable across cases; seqused is validated separately.
                positions = range(effective_length if bsnd else storage_length)
            lengths.append(len(positions))
            offset = batch_index * output_shape[1] if bsnd else geometry["q_prefix"][batch_index]
            if positions.step == 1:
                chunks.append(
                    output_bytes[(offset + positions.start) * token_bytes : (offset + positions.stop) * token_bytes]
                )
            else:
                chunks.extend(
                    output_bytes[(offset + index) * token_bytes : (offset + index + 1) * token_bytes]
                    for index in positions
                )
        if not all(lengths) or (bsnd and len(set(lengths)) != 1):
            raise ValueError(f"{row['testcase_name']}: B relation needs positive, equal BSND Q effective lengths")
        shape = (len(lengths), lengths[0], *feature_shape) if bsnd else (sum(lengths), *feature_shape)
        return b"".join(chunks), shape

    @staticmethod
    def _read_output(row: Dict[str, str], dump_dir: Path, shape: Sequence[int], dtype: str) -> bytes:
        testcase_name = row["testcase_name"]
        if row.get("precision_status", "").upper() != "PASS":
            raise ValueError(f"{testcase_name}: precision_status is not PASS")
        eager_precision = row.get("eager_precision") or ""
        # Covers both NO_OUTPUT and SIM_NO_OUTPUT phase-one markers.
        if "NO_OUTPUT" in eager_precision:
            raise ValueError(f"{testcase_name}: phase one did not produce a device output")
        # ACLNN honors dump_file_prefix while the E2E dumper retains the testcase
        # name. Prefer the explicit prefix, then retain E2E/cache compatibility.
        prefix = str(row.get("dump_file_prefix") or "").strip()
        dump_names = (prefix, testcase_name) if prefix and prefix != testcase_name else (testcase_name,)
        output_path = find_output_dump(dump_dir, dump_names)
        if output_path is None:
            flat_paths = ", ".join(str(dump_dir / f"{name}_output_0.bin") for name in dump_names)
            nested_paths = ", ".join(str(dump_dir / name / "output_*_0_*.bin") for name in dump_names)
            expected_paths = f"{flat_paths}, or {nested_paths}"
            raise ValueError(f"{testcase_name}: missing output dump; expected {expected_paths}")
        output_bytes = output_path.read_bytes()
        expected_bytes = product(shape) * byte_width(dtype)
        if len(output_bytes) != expected_bytes:
            raise ValueError(
                f"{testcase_name}: output bytes={len(output_bytes)}, expected={expected_bytes} for {tuple(shape)!r}/{dtype}"
            )
        return output_bytes


class MlaProfile(BatchProfile):
    """SMLA/QSMLA/MQSMLA output-0 relation handling."""

    PREFIX_ATTRIBUTE_NAMES = (
        "cu_seqlens_q_values",
        "cu_seqlens_ori_kv_values",
        "cu_seqlens_cmp_kv_values",
    )
    BATCH_ATTRIBUTE_NAMES = frozenset(
        ("seqused_q_values", "seqused_ori_kv_values", "seqused_cmp_kv_values", "cmp_residual_kv_values")
    )
    BLOCK_COUNT_ATTRIBUTE_NAMES = frozenset(("block_num1", "block_num2"))
    PRIVATE_ATTRIBUTE_NAMES = frozenset(("batch_deterministic_level",))

    def geometry(self, row):
        shapes = parse_cell(row, "tensor_view_shapes")
        dtypes = parse_cell(row, "tensor_dtypes")
        if not shapes or not dtypes or shapes[0] is None:
            raise ValueError(f"{row['testcase_name']}: Q tensor metadata is required")
        output_shape = tuple(shapes[0])
        attributes = self._attributes(row)
        layout_q = attributes.get("layout_q", "BSND")
        if layout_q not in ("BSND", "TND"):
            raise ValueError(f"{row['testcase_name']}: unsupported layout_q={layout_q!r}")
        if len(output_shape) != (4 if layout_q == "BSND" else 3):
            raise ValueError(f"{row['testcase_name']}: Q shape does not match layout_q={layout_q}")
        prefixes = {}
        for name in self.PREFIX_ATTRIBUTE_NAMES:
            value = attributes.get(name)
            if value is not None:
                prefixes[name] = [int(item) for item in value]
        if layout_q == "TND":
            q_prefix = prefixes.get("cu_seqlens_q_values")
            if not q_prefix or q_prefix[0] != 0 or q_prefix[-1] != output_shape[0]:
                raise ValueError(f"{row['testcase_name']}: TND Q prefix must span output T")
            batch_size = len(q_prefix) - 1
        else:
            q_prefix = None
            batch_size = output_shape[0]
        for name, value in prefixes.items():
            if len(value) != batch_size + 1 or value[0] != 0:
                raise ValueError(f"{row['testcase_name']}: {name} must contain B + 1 prefix values")
            is_q_prefix = name == "cu_seqlens_q_values"
            invalid = (
                any(right <= left for left, right in zip(value, value[1:]))
                if is_q_prefix
                else any(right < left for left, right in zip(value, value[1:]))
            )
            if invalid:
                order = "strictly increasing" if is_q_prefix else "non-decreasing"
                raise ValueError(f"{row['testcase_name']}: {name} must be {order}")
        q_lengths = (
            [right - left for left, right in zip(q_prefix, q_prefix[1:])]
            if q_prefix is not None
            else [output_shape[1]] * batch_size
        )
        return {
            "attributes": attributes,
            "input_dtypes": dtypes,
            "batch_size": batch_size,
            "layout_q": layout_q,
            "q_prefix": q_prefix,
            "q_storage_lengths": q_lengths,
            "q_effective_lengths": q_lengths,
            "output_shape": output_shape,
            "output_dtype": self.output_dtype or str(dtypes[0]),
        }

    @staticmethod
    def _normalize_prefix(value: Sequence[int], batch_indices: Sequence[int]):
        normalized = [0]
        for batch_index in batch_indices:
            normalized.append(normalized[-1] + int(value[batch_index + 1]) - int(value[batch_index]))
        return normalized

    def _context(self, row, relation, geometry):
        attributes, batch_size = geometry["attributes"], geometry["batch_size"]
        batch_indices = slice_indices(relation.slices[0])
        sequence_slice = relation.slices[1] if relation.axes == (0, 1) else None
        normalized = {}
        for key, value in attributes.items():
            if key in self.PRIVATE_ATTRIBUTE_NAMES or key in ("B", "T1", "T2", "T3"):
                continue
            if key in self.PREFIX_ATTRIBUTE_NAMES and isinstance(value, (list, tuple)):
                normalized[key] = self._normalize_prefix(value, batch_indices)
            elif key in self.BLOCK_COUNT_ATTRIBUTE_NAMES and attributes.get("layout_kv") != "PA_BBND":
                continue
            elif key in self.BATCH_ATTRIBUTE_NAMES and value is not None:
                if not isinstance(value, (list, tuple)) or len(value) != batch_size:
                    raise ValueError(f"{row['testcase_name']}: {key} must contain B={batch_size} values")
                normalized[key] = normalize_value([value[index] for index in batch_indices])
            else:
                normalized[key] = normalize_value(value)
        masks = [attributes.get("ori_mask_mode")]
        if attributes.get("has_cmp_kv", True):
            masks.append(attributes.get("cmp_mask_mode"))
        sequence_context = None
        if sequence_slice is not None:
            sequence_indices = slice_indices(sequence_slice)
            sequence_context = (
                [0, len(sequence_indices)] if all(value in (None, 0) for value in masks) else list(sequence_indices)
            )
        return {
            "relation_batch_count": len(batch_indices),
            "relation_sequence_slice": sequence_context,
            "input_dtypes": normalize_value(geometry["input_dtypes"]),
            "attributes": normalized,
        }


class IndexerProfile(BatchProfile):
    """LI_V2/QLI_V2 TopK-index output-0 relation handling."""

    @staticmethod
    def _vector(attributes, name, batch_size, default):
        value = attributes.get(f"{name}_values")
        values = [int(item) for item in value] if value is not None else list(default)
        if len(values) != batch_size:
            raise ValueError(f"{name}_values length does not equal B={batch_size}")
        return values

    @staticmethod
    def _prefix_lengths(attributes, name: str, batch_size: int):
        value = attributes.get(f"{name}_values")
        if value is None:
            return None
        value = [int(item) for item in value]
        if len(value) != batch_size + 1 or value[0] != 0:
            raise ValueError(f"{name}_values must contain B + 1 prefix values")
        if any(right < left for left, right in zip(value, value[1:])):
            raise ValueError(f"{name}_values must be non-decreasing")
        return [right - left for left, right in zip(value, value[1:])]

    @classmethod
    def _effective_lengths(cls, attributes, name: str, batch_size: int, storage_lengths=None):
        """Resolve optional seqused values without changing physical storage geometry."""
        default = storage_lengths if storage_lengths is not None else [0] * batch_size
        lengths = cls._vector(attributes, name, batch_size, default)
        if any(length < 0 for length in lengths):
            raise ValueError(f"{name}_values must not contain negative lengths")
        if storage_lengths is not None and any(length > capacity for length, capacity in zip(lengths, storage_lengths)):
            raise ValueError(f"{name}_values exceeds its physical storage length")
        return lengths

    def geometry(self, row):
        shapes = parse_cell(row, "tensor_view_shapes")
        dtypes = parse_cell(row, "tensor_dtypes")
        attributes = self._attributes(row)
        if len(shapes) != self.input_count:
            raise ValueError(f"{row['testcase_name']}: expected LI_V2/QLI_V2 direct input slots")
        q_shape = tuple(shapes[0])
        k_shape = tuple(shapes[1])
        layout_q = attributes.get("layout_q", "BSND")
        layout_k = attributes.get("layout_k", "BSND")
        for name, shape, layout in (("Q", q_shape, layout_q), ("K", k_shape, layout_k)):
            if layout in ("BSND", "TND", "PA_BBND") and len(shape) != (3 if layout == "TND" else 4):
                raise ValueError(f"{row['testcase_name']}: {name} shape {shape} does not match layout={layout}")
        if layout_q == "BSND":
            batch_size, q_extent, q_heads, head_dim = q_shape
            q_prefix = None
            q_storage_lengths = [q_extent] * batch_size
        elif layout_q == "TND":
            q_extent, q_heads, head_dim = q_shape
            q_prefix = [int(item) for item in attributes.get("cu_seqlens_q_values", ())]
            if len(q_prefix) < 2 or q_prefix[0] != 0 or q_prefix[-1] != q_extent:
                raise ValueError(f"{row['testcase_name']}: TND Q prefix must span the Q tensor")
            batch_size = len(q_prefix) - 1
            q_storage_lengths = self._prefix_lengths(attributes, "cu_seqlens_q", batch_size)
            if q_storage_lengths is None or sum(q_storage_lengths) != q_extent:
                raise ValueError(f"{row['testcase_name']}: invalid TND Q prefix")
        else:
            raise ValueError(f"{row['testcase_name']}: unsupported layout_q={layout_q}")
        if layout_k == "BSND":
            if int(k_shape[0]) != batch_size:
                raise ValueError(f"{row['testcase_name']}: key B does not match Q B")
            key_heads = int(k_shape[2])
            k_storage_lengths = [int(k_shape[1])] * batch_size
            block_size = None
        elif layout_k == "TND":
            key_heads = int(k_shape[1])
            k_storage_lengths = self._prefix_lengths(attributes, "cu_seqlens_k", batch_size)
            if k_storage_lengths is None or sum(k_storage_lengths) != int(k_shape[0]):
                raise ValueError(f"{row['testcase_name']}: invalid TND K prefix")
            block_size = None
        elif layout_k == "PA_BBND":
            block_size = int(k_shape[1])
            key_heads = int(k_shape[2])
            k_storage_lengths = None
        else:
            raise ValueError(f"{row['testcase_name']}: unsupported layout_k={layout_k}")
        topk = attributes.get("topk", attributes.get("sparse_count"))
        if topk is None:
            raise ValueError(f"{row['testcase_name']}: topk or sparse_count is required")
        topk = int(topk)
        output_shape = (batch_size, q_extent, key_heads, topk) if layout_q == "BSND" else (q_extent, key_heads, topk)
        return {
            "attributes": attributes,
            "input_dtypes": tuple(dtypes),
            "batch_size": int(batch_size),
            "q_heads": int(q_heads),
            "key_heads": key_heads,
            "head_dim": int(head_dim),
            "q_storage_lengths": q_storage_lengths,
            "q_effective_lengths": self._effective_lengths(attributes, "seqused_q", batch_size, q_storage_lengths),
            "k_storage_lengths": k_storage_lengths,
            "k_effective_lengths": self._effective_lengths(attributes, "seqused_k", batch_size, k_storage_lengths),
            "residual": self._vector(attributes, "cmp_residual_k", batch_size, [0] * batch_size),
            "layout_q": layout_q,
            "layout_k": layout_k,
            "block_size": block_size,
            "q_prefix": q_prefix,
            "output_shape": tuple(output_shape),
            "output_dtype": "int32",
            "topk": topk,
        }

    def _context(self, row, relation, geometry):
        batch_indices = slice_indices(relation.slices[0])
        context = {
            key: geometry[key]
            for key in ("layout_q", "layout_k", "q_heads", "key_heads", "head_dim", "topk", "block_size")
        }
        # K padding changes storage offsets, not the effective keys being compared.
        for key in ("q_storage_lengths", "q_effective_lengths", "k_effective_lengths", "residual"):
            values = geometry[key]
            context[key] = tuple(values[index] for index in batch_indices) if values is not None else None
        if relation.axes == (0, 1):
            sequence_length = len(slice_indices(relation.slices[1]))
            context["q_storage_lengths"] = (sequence_length,)
            context["q_effective_lengths"] = (sequence_length,)
        return {
            **context,
            "input_dtypes": normalize_value(geometry["input_dtypes"]),
            "attributes": {
                key: value
                for key, value in geometry["attributes"].items()
                if key
                not in (
                    "batch_deterministic_level",
                    "seqused_q_values",
                    "seqused_k_values",
                    "cu_seqlens_q_values",
                    "cu_seqlens_k_values",
                    "cmp_residual_k_values",
                )
                and not isinstance(value, (list, tuple, dict))
            },
        }


class CompressorProfile(BatchProfile):
    name = "Compressor"
    API_SUFFIXES = frozenset(
        (
            "compressor",
            "aclnnCompressor",
            "quant_compressor",
            "aclnnQuantCompressor",
        )
    )
    OUTPUT_DTYPE = "bfloat16"

    _ATTR_ALIASES = {
        "cmp_ratio": "cmpRatio",
        "cache_mode": "cacheMode",
        "coff": "coff",
        "start_pos": "startPos",
        "seqused": "seqused",
        "cu_seqlens": "cuSeqlens",
        "state_cache_stride_dim0": "stateCacheStrideDim0",
    }

    def __init__(self):
        super().__init__("Compressor", self.API_SUFFIXES, 5, self.OUTPUT_DTYPE)

    def supports(self, api_name: str) -> bool:
        return api_name.rsplit(".", 1)[-1] in self.API_SUFFIXES

    # ------------------------------------------------------------------
    # attribute helpers (snake_case + camelCase compatible)
    # ------------------------------------------------------------------

    def _attr(self, attributes, key, default=None):
        if attributes is None:
            return default
        if key in attributes:
            return attributes[key]
        camel = self._ATTR_ALIASES.get(key, key)
        if camel in attributes:
            return attributes[camel]
        snake = next((k for k, v in self._ATTR_ALIASES.items() if v == key), key)
        if snake in attributes:
            return attributes[snake]
        return default

    def _get_start_pos_list(self, attrs):
        raw = self._attr(attrs, "start_pos_values", self._attr(attrs, "start_pos"))
        if raw is None:
            return [0]
        if isinstance(raw, (list, tuple)):
            return [int(v) for v in raw]
        try:
            return [int(raw)]
        except (ValueError, TypeError):
            return [0]

    def _get_seqused_list(self, attrs, batch_size):
        raw = self._attr(attrs, "seqused_values", self._attr(attrs, "seqused"))
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            return [int(v) for v in raw]
        try:
            return [int(raw)] * batch_size
        except (ValueError, TypeError):
            return None

    def _get_cuseqlens_list(self, attrs):
        raw = self._attr(attrs, "cu_seqlens_values", self._attr(attrs, "cu_seqlens"))
        if raw is None:
            return None
        return [int(v) for v in raw]

    # ------------------------------------------------------------------
    # batch_consistency_id parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_slice_id(slice_id):
        parts = str(slice_id).split("_")
        if len(parts) < 5:
            return None
        try:
            return {
                "seed": int(parts[0]),
                "axis": int(parts[1]),
                "start": int(parts[2]),
                "end": int(parts[3]),
                "step": int(parts[4]),
            }
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _flatten_consistency_id(bcid):
        result = []
        for tensor_idx, tensor_id in enumerate(bcid or ()):
            if tensor_id is None:
                continue
            if isinstance(tensor_id, str):
                if tensor_id and tensor_id != "None":
                    result.append(
                        (
                            tensor_id,
                            {
                                "tensor_idx": tensor_idx,
                                "axis_idx": 0,
                                "slice_idx": 0,
                            },
                        )
                    )
                continue
            for axis_idx, axis_id in enumerate(tensor_id):
                if axis_id is None:
                    continue
                if isinstance(axis_id, str):
                    if axis_id and axis_id != "None":
                        result.append(
                            (
                                axis_id,
                                {
                                    "tensor_idx": tensor_idx,
                                    "axis_idx": axis_idx,
                                    "slice_idx": 0,
                                },
                            )
                        )
                    continue
                for slice_idx, sid in enumerate(axis_id):
                    if sid is None or not isinstance(sid, str) or sid == "None":
                        continue
                    result.append(
                        (
                            sid,
                            {
                                "tensor_idx": tensor_idx,
                                "axis_idx": axis_idx,
                                "slice_idx": slice_idx,
                            },
                        )
                    )
        return result

    # ------------------------------------------------------------------
    # geometry / region calculation
    # ------------------------------------------------------------------

    def _infer_layout(self, row):
        shapes = parse_cell(row, "tensor_view_shapes")
        if shapes and shapes[0] is not None:
            return len(tuple(shapes[0])) == 2
        return False

    def _compute_cmpkv_shape(self, row, attrs):
        shapes = parse_cell(row, "tensor_view_shapes")
        if not shapes or shapes[0] is None or len(shapes) < 2 or shapes[1] is None:
            return None
        x_shape = tuple(shapes[0])
        wkv_shape = tuple(shapes[1])
        cmp_ratio = int(self._attr(attrs, "cmp_ratio", 4))
        coff = int(self._attr(attrs, "coff", 1))
        d = wkv_shape[0] // coff
        if len(x_shape) == 2:  # TH
            t = x_shape[0]
            cu_seqlens = self._get_cuseqlens_list(attrs)
            b = len(cu_seqlens) - 1 if cu_seqlens and len(cu_seqlens) > 1 else 1
            sr = min(t, t // cmp_ratio + b)
            return (sr, d)
        b, s = x_shape[0], x_shape[1]
        sr = (s + cmp_ratio - 1) // cmp_ratio
        return (b, sr, d)

    def _get_slice_info(self, row, info, parsed):
        batch_slice_info = parse_cell(row, "batch_slice_info")
        if batch_slice_info is not None and info["tensor_idx"] < len(batch_slice_info):
            tensor_slices = batch_slice_info[info["tensor_idx"]]
            if tensor_slices is not None and info["axis_idx"] < len(tensor_slices):
                axis_slices = tensor_slices[info["axis_idx"]]
                if axis_slices is not None and info["slice_idx"] < len(axis_slices):
                    sl = axis_slices[info["slice_idx"]]
                    if isinstance(sl, (tuple, list)) and len(sl) >= 2:
                        step = int(sl[2]) if len(sl) >= 3 else 1
                        return {"start": int(sl[0]), "stop": int(sl[1]), "step": step}
        return {"start": parsed["start"], "stop": parsed["end"], "step": parsed["step"]}

    def _get_companion_bidx(self, row, tensor_idx, slice_idx):
        """BSH combined slice: get the batch index from the axis-0 slice."""
        batch_slice_info = parse_cell(row, "batch_slice_info")
        if batch_slice_info is None or tensor_idx >= len(batch_slice_info):
            return 0
        tensor_slices = batch_slice_info[tensor_idx]
        if tensor_slices is None or len(tensor_slices) < 1:
            return 0
        axis_0_slices = tensor_slices[0]
        if axis_0_slices is None or slice_idx >= len(axis_0_slices):
            return 0
        sl = axis_0_slices[slice_idx]
        if sl is None or len(sl) < 1:
            return 0
        return int(sl[0])

    @staticmethod
    def _compute_token_region(start, stop, start_pos, cmp_ratio):
        head_size = cmp_ratio - (start + start_pos) % cmp_ratio
        compare_len = (stop - start - head_size % cmp_ratio) // cmp_ratio
        cache_len = cmp_ratio - start_pos % cmp_ratio
        if head_size == cmp_ratio:
            start_idx = (start - cache_len) // cmp_ratio + 1
        else:
            start_idx = (start + head_size + cmp_ratio - 1) // cmp_ratio
        compare_len = max(compare_len, 0)
        return start_idx, compare_len

    def _find_th_batch(self, start, length, cu_seqlens, seqused_list, cmp_ratio):
        base = cu_seqlens[0] if cu_seqlens else 0
        batch_size = len(cu_seqlens) - 1
        tc_idx = 0
        b_idx = 0
        for b_idx in range(batch_size):
            if start >= cu_seqlens[b_idx + 1] - base:
                if seqused_list and b_idx < len(seqused_list):
                    tc_idx += seqused_list[b_idx] // cmp_ratio
                continue
            adjusted_start = start - cu_seqlens[b_idx]
            return b_idx, tc_idx, adjusted_start, adjusted_start + length
        # Fallback: last batch
        last = max(batch_size - 1, 0)
        adjusted_start = start - (cu_seqlens[last] if last < len(cu_seqlens) else 0)
        return last, tc_idx, adjusted_start, adjusted_start + length

    # ------------------------------------------------------------------
    # output loading
    # ------------------------------------------------------------------

    def _load_output_array(self, row, dump_dir):
        testcase_name = row["testcase_name"]
        if row.get("precision_status", "").upper() != "PASS":
            raise ValueError(f"{testcase_name}: precision_status is not PASS")
        prefix = str(row.get("dump_file_prefix") or "").strip()
        dump_names = (prefix, testcase_name) if prefix and prefix != testcase_name else (testcase_name,)
        output_path = find_output_dump(dump_dir, dump_names)
        if output_path is None:
            flat_paths = ", ".join(str(dump_dir / f"{name}_output_0.bin") for name in dump_names)
            nested_paths = ", ".join(str(dump_dir / name / "output_*_0_*.bin") for name in dump_names)
            raise ValueError(f"{testcase_name}: missing cmp_kv output dump; expected {flat_paths}, or {nested_paths}")
        raw_bytes = output_path.read_bytes()
        raw = np.frombuffer(raw_bytes, dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)

    @staticmethod
    def _reshape_output(array, expected_shape):
        if expected_shape is None:
            return array
        expected_size = product(expected_shape) if expected_shape else 0
        if expected_size > 0 and array.size == expected_size:
            return array.reshape(expected_shape)
        if len(expected_shape) == 2:
            last = expected_shape[-1]
            if last and array.size % last == 0:
                return array.reshape((array.size // last, last))
        elif len(expected_shape) == 3:
            first, last = expected_shape[0], expected_shape[-1]
            if first and last and array.size % (first * last) == 0:
                return array.reshape((first, array.size // (first * last), last))
        return array

    # ------------------------------------------------------------------
    # sample construction
    # ------------------------------------------------------------------

    def make_samples(self, row, dump_dir):
        bcid = parse_cell(row, "batch_consistency_id")
        if bcid is None:
            return []
        flat = self._flatten_consistency_id(bcid)
        if not flat:
            return []
        attrs = dict(parse_cell(row, "attributes", {}))
        is_th = self._infer_layout(row)
        cmp_ratio = int(self._attr(attrs, "cmp_ratio", 4))
        cache_mode = int(self._attr(attrs, "cache_mode", 1))
        coff = int(self._attr(attrs, "coff", 1))
        start_pos_list = self._get_start_pos_list(attrs)
        cu_seqlens = self._get_cuseqlens_list(attrs)
        batch_size = (
            len(cu_seqlens) - 1
            if cu_seqlens and len(cu_seqlens) > 1
            else (len(start_pos_list) if start_pos_list else 1)
        )
        seqused_list = self._get_seqused_list(attrs, batch_size)
        cmpkv_shape = self._compute_cmpkv_shape(row, attrs)
        array = self._load_output_array(row, dump_dir)
        array = self._reshape_output(array, cmpkv_shape)

        shapes = parse_cell(row, "tensor_view_shapes")
        wkv_shape = tuple(shapes[1]) if shapes and len(shapes) > 1 and shapes[1] is not None else None
        wgate_shape = tuple(shapes[2]) if shapes and len(shapes) > 2 and shapes[2] is not None else None
        ape_shape = tuple(shapes[4]) if shapes and len(shapes) > 4 and shapes[4] is not None else None

        samples: List[Sample] = []
        for slice_id, info in flat:
            parsed = self._parse_slice_id(slice_id)
            if parsed is None:
                continue
            slice_info = self._get_slice_info(row, info, parsed)
            if slice_info is None:
                continue

            start = slice_info["start"]
            stop = slice_info["stop"]
            length = stop - start
            axis = parsed["axis"]
            seed = parsed["seed"]

            value = b""
            shape: Tuple[int, ...] = (0,)

            if is_th:
                # TH layout: find owning batch via cu_seqlens
                if cu_seqlens and len(cu_seqlens) > 1:
                    b_idx, tc_idx, adj_start, adj_stop = self._find_th_batch(
                        start, length, cu_seqlens, seqused_list, cmp_ratio
                    )
                else:
                    b_idx, tc_idx, adj_start, adj_stop = 0, 0, start, stop
                sp = start_pos_list[b_idx] if b_idx < len(start_pos_list) else 0
                start_idx, compare_len = self._compute_token_region(adj_start, adj_stop, sp, cmp_ratio)
                start_idx += tc_idx
                if compare_len > 0 and array.ndim >= 2:
                    region = array[start_idx : start_idx + compare_len, :]
                    value = np.ascontiguousarray(region).tobytes()
                    shape = (compare_len, array.shape[-1])
            elif axis == 0 and array.ndim == 3:
                if stop > start:
                    if seqused_list and start < len(seqused_list):
                        valid_blocks = seqused_list[start] // cmp_ratio
                    else:
                        valid_blocks = array.shape[1]
                    end_block = min(valid_blocks, array.shape[1])
                    region = array[start:stop, :end_block, :]
                    value = np.ascontiguousarray(region).tobytes()
                    shape = (stop - start, end_block, array.shape[-1])
            elif axis == 1 and array.ndim == 3:
                # BSH token axis: use companion batch slice for bidx
                bidx = self._get_companion_bidx(row, info["tensor_idx"], info["slice_idx"])
                sp = start_pos_list[bidx] if bidx < len(start_pos_list) else 0
                start_idx, compare_len = self._compute_token_region(start, stop, sp, cmp_ratio)
                if compare_len > 0:
                    region = array[bidx, start_idx : start_idx + compare_len, :]
                    value = np.ascontiguousarray(region).tobytes()
                    shape = (compare_len, array.shape[-1])
            else:
                continue

            relation = Relation(
                axes=(axis,),
                slices=((start, stop, slice_info.get("step", 1)),),
                seed=seed,
            )
            context = {
                "api_name": row.get("api_name"),
                "layout": "TH" if is_th else "BSH",
                "cmp_ratio": cmp_ratio,
                "cache_mode": cache_mode,
                "coff": coff,
                "wkv_shape": wkv_shape,
                "wgate_shape": wgate_shape,
                "ape_shape": ape_shape,
                "output_dtype": self.OUTPUT_DTYPE,
            }
            samples.append(Sample(row, self, relation, value, shape, self.OUTPUT_DTYPE, context))

        return samples


PROFILES = (
    MlaProfile("SMLA", ("sparse_flash_mla", "sparse_flash_mla_ttk", "aclnnSparseFlashMla"), 18),
    MlaProfile(
        "QSMLA",
        ("quant_sparse_flash_mla", "quant_sparse_flash_mla_ttk", "aclnnQuantSparseFlashMla"),
        21,
        "bfloat16",
    ),
    MlaProfile(
        "MQSMLA",
        ("mixed_quant_sparse_flash_mla", "mixed_quant_sparse_flash_mla_ttk", "aclnnMixedQuantSparseFlashMla"),
        18,
    ),
    IndexerProfile("LI_V2", ("lightning_indexer", "lightning_indexer_v2", "aclnnLightningIndexerV2"), 11),
    IndexerProfile(
        "QLI_V2", ("quant_lightning_indexer", "quant_lightning_indexer_v2", "aclnnQuantLightningIndexerV2"), 13
    ),
    CompressorProfile(),
)


def find_profile(api_name: str) -> Optional[BatchProfile]:
    return next((profile for profile in PROFILES if profile.supports(api_name)), None)


def report_record(row: Dict[str, str], status: str, digest: str, errors: Sequence[str], case_number: int):
    return {
        **{name: row.get(name, "") for name in REPORT_HEADERS},
        "result": status,
        "SHA-256": digest,
        "errors": "; ".join(errors),
        "case_number": case_number,
    }


def write_report(path: Path, records: Sequence[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.stem}.", suffix=".csv", delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=REPORT_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_result_headers(fieldnames: Optional[Sequence[str]]):
    """Reject display-only phase-one CSVs before replacing a final report."""
    if fieldnames is None:
        raise ValueError("phase-one result CSV has no header")
    missing = [name for name in REQUIRED_RESULT_HEADERS if name not in fieldnames]
    if missing:
        raise ValueError(
            "phase-one result CSV is missing required columns: "
            f"{', '.join(missing)}; do not use a --title-trimmed CSV for phase two"
        )


def compare_result_csv(result_path: Path, final_path: Path, dump_dir: Path):
    with result_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        validate_result_headers(reader.fieldnames)
        rows = list(reader)

    testcase_names = set()
    for row in rows:
        testcase_name = row.get("testcase_name")
        if not testcase_name:
            continue
        if testcase_name in testcase_names:
            raise ValueError(
                "phase-one result CSV has duplicate testcase_name "
                f"{testcase_name!r}; rerun phase one without duplicate or appended rows"
            )
        testcase_names.add(testcase_name)

    samples = []
    records = []
    for row in rows:
        if not row.get("testcase_name"):
            continue
        try:
            if not is_enabled(row) or complete_batch_metadata(row) is None:
                continue
            profile = find_profile(row.get("api_name", ""))
            if profile is None:
                raise ValueError("unsupported FA API for unified batch comparison")
            samples.extend(profile.make_samples(row, dump_dir))
        except (OSError, ValueError, TypeError, IndexError) as error:
            message = str(error)
            if not message.startswith(f"{row['testcase_name']}:"):
                message = f"{row['testcase_name']}: {message}"
            records.append(report_record(row, "FAIL", "", (message,), 1))

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.group_key].append(sample)
    for group_samples in grouped.values():
        group_samples.sort(key=lambda sample: (sample.row["testcase_name"], sample.relation.slices))
        reference = group_samples[0]
        errors = []
        for sample in group_samples[1:]:
            if sample.shape != reference.shape or sample.dtype != reference.dtype:
                errors.append(
                    f"{sample.row['testcase_name']}: relation shape/dtype differs from {reference.row['testcase_name']}"
                )
            if sample.context != reference.context:
                errors.append(
                    f"{sample.row['testcase_name']}: relation context differs from {reference.row['testcase_name']}"
                )
            if sample.value != reference.value:
                errors.append(
                    f"{sample.row['testcase_name']}: raw output bytes differ from {reference.row['testcase_name']}"
                )
        case_number = len({sample.row["testcase_name"] for sample in group_samples})
        status = "FAIL" if errors else ("PASS" if len(group_samples) > 1 else "NOT_APPLICABLE")
        if len(group_samples) == 1:
            errors.append("one relation sample: no same-case or cross-case peer")
        for sample in group_samples:
            records.append(
                report_record(sample.row, status, hashlib.sha256(sample.value).hexdigest(), errors, case_number)
            )

    records.sort(key=lambda record: (record["testcase_name"], record["batch_seed"], record["batch_axis"]))
    write_report(final_path, records)
    failed = sum(record["result"] == "FAIL" for record in records)
    passed = sum(record["result"] == "PASS" for record in records)
    not_applicable = sum(record["result"] == "NOT_APPLICABLE" for record in records)
    return failed == 0, {"pass": passed, "fail": failed, "not_applicable": not_applicable, "records": len(records)}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare FA TTK batch-consistency output dumps.")
    parser.add_argument("--result", required=True, help="Phase-one TTK result CSV")
    parser.add_argument("--final_result", required=True, help="Final relation report CSV")
    parser.add_argument(
        "--dump_dir",
        default=os.environ.get("NPU_DUMP_PATH"),
        help="Directory holding the output .bin dumps (default: %(default)s)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    dump_path = args.dump_dir
    if not dump_path:
        print("batch consistency comparison failed: NPU_DUMP_PATH is not set and --dump_dir was not provided")
        return 1
    try:
        passed, summary = compare_result_csv(Path(args.result), Path(args.final_result), Path(dump_path))
    except (OSError, ValueError, csv.Error) as error:
        print(f"batch consistency comparison failed: {error}")
        return 1
    print(
        "batch consistency report: "
        f"PASS={summary['pass']} FAIL={summary['fail']} NOT_APPLICABLE={summary['not_applicable']} "
        f"records={summary['records']} final_result={args.final_result}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
