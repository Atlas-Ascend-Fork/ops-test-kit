# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ManualDataStore 读写往返、完整性校验与 prepare/replay 机制的单元测试。"""

import logging
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.aclnn.input_generate import InputGenerator
from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.common.dump import (
    ManualDataError,
    ManualDataStore,
    load_manual_data_case,
    manual_data_dump_store,
)
from ttk.core_modules.framework_api.input_generate import generate_inputs
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.utilities import resolve_numpy_dtypes


def _e2e_case(name="case/with spaces"):
    """构造一个填好默认字段的 E2E 测试用例。"""
    case = TestcaseE2e()
    case.testcase_name = name
    case.api_name = "torch.add"
    case.tensor_view_shapes = ((2, 2), (2, 2))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((3, 3), (2, 2))
    case.tensor_view_offsets = (1, 0)
    case.tensor_view_strides = ((3, 1), (2, 1))
    case.output_tensor_indexes = (1,)
    case.inplace_input_indexes = ()
    case.attributes = {"alpha": 1}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.golden_api = ""
    return case


def _aclnn_case():
    """构造一个填好默认字段的 ACLNN 测试用例。"""
    case = TestcaseAclnn()
    case.testcase_name = "aclnn_case"
    case.api_name = "aclnnAdd"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.output_inplace_indexes = ()
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.scalar_dtypes = ("float32",)
    case.scalar_data_ranges = ((0, 1),)
    return case


def _kernel_case(name="kernel_case"):
    """构造一个填好默认字段的 kernel 测试用例（SimpleNamespace）。"""
    return SimpleNamespace(
        testcase_name=name,
        flat_input_shapes=((2,), None),
        flat_input_dtypes=("float32", "float32"),
        flat_output_shapes=((2,),),
        flat_output_dtypes=("float32",),
    )


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_round_trip_uses_only_typed_data_files(tmp_path, file_format):
    """aclnn 用例以 bin/npy/pt 三种格式写入再加载，数据与文件名约定一致。"""
    case = _aclnn_case()
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalars = [np.array(0.25, dtype=np.float32)]
    goldens = [np.array([3.0, 4.0], dtype=np.float32)]
    store = ManualDataStore(tmp_path)

    case_dir = store.write_case(
        case,
        "aclnn",
        inputs,
        goldens,
        scalars=scalars,
        file_format=file_format,
    )
    loaded = store.load_case(case, "aclnn")
    loaded_goldens = loaded.load_goldens(references=goldens)

    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(loaded.inputs[1], np.ones(2, dtype=np.float32))
    np.testing.assert_array_equal(loaded.scalars[0], scalars[0])
    np.testing.assert_array_equal(loaded_goldens[0], goldens[0])
    assert {path.name for path in case_dir.iterdir()} == {
        f"input_0_float32.{file_format}",
        f"scalar_0_float32.{file_format}",
        f"golden_0_float32.{file_format}",
    }


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_e2e_and_aclnn_share_compact_inputs(tmp_path, file_format):
    """E2E and ACLNN can exchange their compatible compact tensor inputs."""
    e2e = _e2e_case("shared_input")
    e2e.tensor_view_shapes = ((2,),)
    e2e.tensor_dtypes = ("float32",)
    e2e.tensor_formats = ("ND",)
    e2e.tensor_storage_shapes = ((2,),)
    e2e.tensor_view_offsets = (0,)
    e2e.tensor_view_strides = ((1,),)
    e2e.output_tensor_indexes = ()

    aclnn = _aclnn_case()
    aclnn.testcase_name = e2e.testcase_name
    aclnn.scalar_dtypes = ()
    aclnn.scalar_data_ranges = ()
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]

    e2e_store = ManualDataStore(tmp_path / "e2e")
    e2e_store.write_case(e2e, "e2e", [inputs[0]], golden, file_format=file_format)
    restored_aclnn = e2e_store.load_case(aclnn, "aclnn")
    np.testing.assert_array_equal(restored_aclnn.inputs[0], inputs[0])
    np.testing.assert_array_equal(restored_aclnn.inputs[1], np.ones(2, np.float32))

    aclnn_store = ManualDataStore(tmp_path / "aclnn")
    case_dir = aclnn_store.write_case(aclnn, "aclnn", inputs, golden, file_format=file_format)
    assert not any(path.name.startswith("input_1_") for path in case_dir.iterdir())
    np.testing.assert_array_equal(aclnn_store.load_case(e2e, "e2e").inputs[0], inputs[0])

    # A pre-0824 ACLNN directory carrying every tensor slot remains valid.
    legacy_path = case_dir / f"input_1_float32.{file_format}"
    source_path = case_dir / f"input_0_float32.{file_format}"
    shutil.copy2(source_path, legacy_path)
    loaded = aclnn_store.load_case(aclnn, "aclnn")
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(loaded.inputs[1], inputs[0])


@pytest.mark.parametrize(
    ("custom_dtype", "e2e_shape", "aclnn_shape"),
    [("float8_e4m3fn", (4,), (4,)), ("float4_e2m1", (4,), (8,)), ("hifloat8", (4,), (4,))],
)
def test_raw_bin_uint8_reinterprets_portable_storage(tmp_path, custom_dtype, e2e_shape, aclnn_shape):
    """Raw bins may exchange uint8 storage with ACLNN custom dtypes."""
    try:
        dtype = resolve_numpy_dtypes((custom_dtype,))[0]
    except RuntimeError as exc:
        pytest.skip(str(exc))

    e2e = _e2e_case(f"raw_{custom_dtype}")
    e2e.tensor_view_shapes = (e2e_shape,)
    e2e.tensor_dtypes = ("uint8",)
    e2e.tensor_formats = ("ND",)
    e2e.tensor_storage_shapes = (e2e_shape,)
    e2e.tensor_view_offsets = (0,)
    e2e.tensor_view_strides = ((1,),)
    e2e.output_tensor_indexes = ()

    aclnn = _aclnn_case()
    aclnn.testcase_name = e2e.testcase_name
    aclnn.tensor_view_shapes = (aclnn_shape, aclnn_shape)
    aclnn.tensor_dtypes = (custom_dtype, "float32")
    aclnn.tensor_formats = ("ND", "ND")
    aclnn.tensor_storage_shapes = (aclnn_shape, aclnn_shape)
    aclnn.tensor_view_offsets = (0, 0)
    aclnn.tensor_view_strides = ((1,), (1,))
    aclnn.scalar_dtypes = ()
    aclnn.scalar_data_ranges = ()

    raw = np.array([0x10, 0x32, 0x54, 0x76], dtype=np.uint8).reshape(e2e_shape)
    golden = [np.zeros(aclnn_shape, dtype=np.float32)]
    e2e_store = ManualDataStore(tmp_path / "e2e")
    e2e_store.write_case(e2e, "e2e", [raw], golden)

    loaded_aclnn = e2e_store.load_case(aclnn, "aclnn")
    assert loaded_aclnn.inputs[0].dtype == np.dtype(dtype)
    assert loaded_aclnn.inputs[0].shape == aclnn_shape

    aclnn_store = ManualDataStore(tmp_path / "aclnn")
    aclnn_store.write_case(
        aclnn,
        "aclnn",
        [loaded_aclnn.inputs[0], np.zeros(aclnn_shape, dtype=np.float32)],
        golden,
    )
    loaded_e2e = aclnn_store.load_case(e2e, "e2e")
    np.testing.assert_array_equal(loaded_e2e.inputs[0], raw)


def test_input_only_dataset_replays_without_requiring_golden(tmp_path):
    case = _aclnn_case()
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalars = [np.array(0.25, dtype=np.float32)]
    store = ManualDataStore(tmp_path)

    case_dir = store.write_case(
        case,
        "aclnn",
        inputs,
        (),
        scalars=scalars,
        write_goldens=False,
    )
    loaded = store.load_case(case, "aclnn", require_goldens=False)

    assert not loaded.has_goldens
    assert not any(path.name.startswith("golden_") for path in case_dir.iterdir())
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    with pytest.raises(ManualDataError, match="golden slot count"):
        store.load_case(case, "aclnn")


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_complete_dataset_remains_loadable_after_directory_move(tmp_path, file_format):
    """写入后整体拷贝到新目录仍可正常加载。"""
    case = _aclnn_case()
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalars = [np.array(0.25, dtype=np.float32)]
    goldens = [np.array([3.0, 4.0], dtype=np.float32)]
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    ManualDataStore(source).write_case(case, "aclnn", inputs, goldens, scalars=scalars, file_format=file_format)
    shutil.copytree(source, destination)
    loaded = ManualDataStore(destination).load_case(case, "aclnn")
    loaded_goldens = loaded.load_goldens(references=goldens)

    assert loaded.case_dir.parent == destination.resolve()
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(loaded.scalars[0], scalars[0])
    np.testing.assert_array_equal(loaded_goldens[0], goldens[0])


def test_npy_round_trip_restores_custom_bfloat16_dtype(tmp_path):
    """npy 格式往返后自定义 bfloat16 dtype 仍可还原。"""
    case = _e2e_case("bfloat16_npy")
    case.tensor_dtypes = ("bfloat16", "bfloat16")
    dtype = resolve_numpy_dtypes(("bfloat16",))[0]
    inputs = [np.arange(9, dtype=np.float32).astype(dtype).reshape(3, 3), np.zeros((2, 2), dtype=dtype)]
    golden = [np.ones((2, 2), dtype=dtype)]
    store = ManualDataStore(tmp_path)

    store.write_case(case, "e2e", inputs, golden, file_format="npy")
    loaded = store.load_case(case, "e2e")
    loaded_golden = loaded.load_goldens(references=golden)

    assert loaded.inputs[0].dtype.name == "bfloat16"
    assert loaded_golden[0].dtype.name == "bfloat16"
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
@pytest.mark.parametrize(
    "logical_dtype",
    ["complex32", "uint1", "int4", "float8_e4m3fn", "float8_e5m2"],
)
def test_special_physical_storage_round_trip(tmp_path, file_format, logical_dtype):
    """complex32/uint1/int4/float8 等特殊物理存储 dtype 三格式往返保持字节一致。"""
    case = _e2e_case(f"{logical_dtype}_{file_format}")
    case.tensor_dtypes = (logical_dtype, "float32")
    if logical_dtype == "complex32":
        storage = np.arange(18, dtype=np.float16).reshape(3, 3, 2)
    elif logical_dtype == "uint1":
        storage = np.array([0x55, 0x01], dtype=np.uint8)
    else:
        dtype = resolve_numpy_dtypes((logical_dtype,))[0]
        storage = np.arange(-4, 5, dtype=np.int8).astype(dtype).reshape(3, 3)
    store = ManualDataStore(tmp_path)

    store.write_case(
        case,
        "e2e",
        [storage, np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
        file_format=file_format,
    )
    loaded = store.load_case(case, "e2e")

    assert loaded.inputs[0].dtype.name == storage.dtype.name
    assert loaded.inputs[0].shape == storage.shape
    if logical_dtype == "int4" and file_format == "bin":
        np.testing.assert_array_equal(loaded.inputs[0], storage)
    else:
        assert loaded.inputs[0].tobytes() == storage.tobytes()


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_kernel_round_trip_uses_kernel_csv_shapes(tmp_path, file_format):
    """kernel 用例按 CSV 声明的 shape/dtype 往返，None 槽位写空文件。"""
    case = _kernel_case(f"kernel_{file_format}")
    inputs = [np.array([1.0, 2.0], np.float32), None]
    goldens = [np.array([3.0, 4.0], np.float32)]
    store = ManualDataStore(tmp_path)

    case_dir = store.write_case(case, "kernel", inputs, goldens, file_format=file_format)
    loaded = store.load_case(case, "kernel")
    loaded_goldens = loaded.load_goldens(
        shapes=case.flat_output_shapes,
        dtypes=case.flat_output_dtypes,
    )

    assert loaded.file_format == file_format
    assert (case_dir / f"input_1_none.{file_format}").stat().st_size == 0
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    assert loaded.inputs[1] is None
    np.testing.assert_array_equal(loaded_goldens[0], goldens[0])


def test_e2e_noncontiguous_backing_storage_and_none_slot_round_trip(tmp_path):
    """非连续 backing storage 与 None golden 槽位往返一致。"""
    case = _e2e_case()
    first = np.arange(9, dtype=np.float32).reshape(3, 3)
    second = np.ones((2, 2), dtype=np.float32)
    golden = [np.ones((2, 2), dtype=np.float32), None]
    store = ManualDataStore(tmp_path)

    store.write_case(case, "e2e", [first, second], golden)
    loaded = store.load_case(case, "e2e")
    loaded_golden = loaded.load_goldens(references=golden)

    np.testing.assert_array_equal(loaded.inputs[0], first)
    assert loaded_golden[1] is None
    assert (loaded.case_dir / "golden_1_none.bin").stat().st_size == 0


def test_e2e_none_golden_suppresses_optional_device_output(tmp_path):
    """None golden 抑制可选设备输出槽位。"""
    case = _e2e_case("none_golden_optional_device_output")
    store = ManualDataStore(tmp_path)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.ones((2, 2), np.float32), None],
    )

    loaded = store.load_case(case, "e2e")
    goldens = loaded.load_goldens(
        references=[
            np.zeros((2, 2), np.float32),
            np.zeros((1, 2), np.float32),
        ]
    )

    np.testing.assert_array_equal(goldens[0], np.ones((2, 2), np.float32))
    assert goldens[1] is None


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_extra_goldens_are_loaded_for_operator_specific_compare(tmp_path, file_format):
    """A cross-API cache keeps extra Goldens in file order for the custom compare."""
    case = _e2e_case(f"extra_goldens_{file_format}")
    goldens = [
        np.full((2, 2), 1, np.float32),
        np.full((3,), 2, np.float32),
        np.full((1, 4), 3, np.float32),
        np.full((2, 3), 4, np.float32),
    ]
    store = ManualDataStore(tmp_path)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        goldens,
        file_format=file_format,
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(
        references=[
            np.zeros_like(goldens[0]),
            np.zeros_like(goldens[3]),
        ]
    )

    assert len(restored) == len(goldens)
    for actual, expected in zip(restored, goldens):
        if file_format == "bin":
            # 当前协议：golden 数 > 设备 output 数时无位置 shape 映射，bin 无
            # 文件内 shape → 读一维流，交 custom compare hook 解释
            np.testing.assert_array_equal(actual, expected.reshape(-1))
        else:
            np.testing.assert_array_equal(actual, expected)


def test_tensor_list_grouping_is_rebuilt_from_csv_structure(tmp_path):
    """TensorList 分组在加载后由 CSV 结构重建。"""
    case = _e2e_case("tensor_list")
    case.tensor_view_shapes = (((2,), (3,)), (2,))
    case.tensor_dtypes = (("float32", "int32"), "float32")
    case.tensor_formats = (("ND", "ND"), "ND")
    case.tensor_storage_shapes = (((2,), (3,)), (2,))
    case.tensor_view_offsets = ((0, 0), 0)
    case.tensor_view_strides = (((1,), (1,)), (1,))
    case.output_tensor_indexes = (1,)
    inputs = [
        np.array([1.0, 2.0], np.float32),
        np.array([3, 4, 5], np.int32),
        np.zeros(2, np.float32),
    ]
    store = ManualDataStore(tmp_path)
    store.write_case(case, "e2e", inputs, [np.ones(2, np.float32)])

    loaded = store.load_case(case, "e2e")
    switches = SimpleNamespace(plugin_path=("must-not-be-scanned",))
    backend = SimpleNamespace(device_type=lambda: "npu", inputs_from_numpy=lambda tc, ri: ri)
    generate_inputs(case, switches, backend, object(), stored_inputs=loaded.inputs)

    assert isinstance(case.tensors[0], list)
    assert len(case.tensors[0]) == 2
    assert tuple(case.tensors[0][0].shape) == (2,)
    assert tuple(case.tensors[0][1].shape) == (3,)
    assert tuple(case.tensors[1].shape) == (2,)


@pytest.mark.parametrize(
    ("scenario", "expected_match"),
    [
        ("changed_dtype", "filename dtype"),
        ("corrupt_file", "byte size"),
        ("unknown_sidecar", "unexpected file"),
        ("extra_slot", "slot count"),
    ],
)
def test_invalid_manual_data_is_rejected(tmp_path, scenario, expected_match):
    """各类数据完整性违规（dtype 改写/文件损坏/未知 sidecar/多余槽位）均在 load_case 阶段被拒绝。"""
    case = _e2e_case(f"reject_{scenario}")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    if scenario == "changed_dtype":
        case.tensor_dtypes = ("float16", "float32")
        case.invalidate_flat_cache("tensor_dtypes")
    elif scenario == "corrupt_file":
        with (case_dir / "input_0_float32.bin").open("ab") as stream:
            stream.write(b"corrupt")
    elif scenario == "unknown_sidecar":
        (case_dir / "manifest.json").write_text("{}", encoding="utf-8")
    elif scenario == "extra_slot":
        (case_dir / "input_2_float32.bin").write_bytes(b"")

    with pytest.raises(ManualDataError, match=expected_match):
        store.load_case(case, "e2e")


def test_precision_fields_can_change_without_invalidating_prepared_data(tmp_path):
    """精度字段（tolerance/absolute_precision）变更不会使已准备数据失效。"""
    case = _e2e_case("precision_case")
    case.precision_tolerances = ((0.001, 0.001),)
    case.absolute_precision = (1e-5,)
    store = ManualDataStore(tmp_path)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )

    case.precision_tolerances = ((0.005, 0.005),)
    case.absolute_precision = (1e-4,)

    assert store.load_case(case, "e2e").case_dir.name == "precision_case"


@pytest.mark.parametrize("file_format", ["npy", "pt"])
def test_self_describing_format_rejects_same_size_wrong_shape(tmp_path, file_format):
    """npy/pt 自描述格式在同字节数但 shape 不符时被拒绝。"""
    case = _e2e_case(f"wrong_{file_format}_shape")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
        file_format=file_format,
    )
    path = case_dir / f"input_0_float32.{file_format}"
    wrong_shape = np.zeros((1, 9), np.float32)
    if file_format == "npy":
        np.save(path, wrong_shape)
    else:
        import torch

        torch.save(torch.from_numpy(wrong_shape), path)

    with pytest.raises(ManualDataError, match="stored shape"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize("file_format", ["npy", "pt"])
def test_self_describing_format_rejects_same_size_wrong_dtype(tmp_path, file_format):
    """npy/pt 自描述格式在同字节数但 dtype 不符时被拒绝。"""
    case = _e2e_case(f"wrong_{file_format}_dtype")
    case.tensor_dtypes = ("bfloat16", "bfloat16")
    dtype = resolve_numpy_dtypes(("bfloat16",))[0]
    inputs = [
        np.zeros((3, 3), dtype=dtype),
        np.zeros((2, 2), dtype=dtype),
    ]
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        inputs,
        [np.zeros((2, 2), dtype=dtype)],
        file_format=file_format,
    )
    path = case_dir / f"input_0_bfloat16.{file_format}"
    wrong_dtype = np.zeros((3, 3), np.int16)
    if file_format == "npy":
        np.save(path, wrong_dtype)
    else:
        import torch

        torch.save(torch.from_numpy(wrong_dtype), path)

    with pytest.raises(ManualDataError, match="dtype .* != filename dtype"):
        store.load_case(case, "e2e")


def test_pt_raw_byte_payload_keeps_shape_inside_data_file(tmp_path):
    """pt 原始字节负载将 shape 存入数据文件；篡改 shape 后被拒绝。"""
    import torch

    case = _e2e_case("raw_byte_pt_shape")
    case.tensor_dtypes = ("float128", "float32")
    inputs = [
        np.arange(9, dtype=np.float128).reshape(3, 3),
        np.zeros((2, 2), np.float32),
    ]
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        inputs,
        [np.zeros((2, 2), np.float32)],
        file_format="pt",
    )
    path = case_dir / "input_0_float128.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {"ttk_raw_bytes", "ttk_shape"}
    assert tuple(payload["ttk_shape"]) == (3, 3)

    payload["ttk_shape"] = (1, 9)
    torch.save(payload, path)
    with pytest.raises(ManualDataError, match="stored shape"):
        store.load_case(case, "e2e")


def test_pt_legacy_torch_load_warns_about_pickle_fallback(tmp_path, monkeypatch, caplog):
    """旧版 torch.load 不支持 weights_only 时回退 pickle 并发出告警。"""
    import torch

    case = _e2e_case("legacy_pt_load")
    store = ManualDataStore(tmp_path)
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    store.write_case(case, "e2e", inputs, [np.zeros((2, 2), np.float32)], file_format="pt")
    original_load = torch.load

    def legacy_load(*args, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("weights_only is unsupported")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", legacy_load)
    with caplog.at_level(logging.WARNING):
        loaded = store.load_case(case, "e2e")

    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    assert "falling back to pickle-based loading" in caplog.text


def test_pt_load_type_error_does_not_fall_back_to_pickle(tmp_path, monkeypatch, caplog):
    """torch.load 抛 TypeError（非 weights_only 不支持）时不回退 pickle，直接报错。"""
    import torch

    case = _e2e_case("pt_load_type_error")
    store = ManualDataStore(tmp_path)
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    store.write_case(case, "e2e", inputs, [np.zeros((2, 2), np.float32)], file_format="pt")

    def broken_load(path, map_location, weights_only):
        raise TypeError("payload decoder failed")

    monkeypatch.setattr(torch, "load", broken_load)
    with caplog.at_level(logging.WARNING), pytest.raises(ManualDataError, match="payload decoder failed"):
        store.load_case(case, "e2e")

    assert "falling back to pickle-based loading" not in caplog.text


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_none_slots_are_explicit_empty_data_files(tmp_path, file_format):
    """None 槽位写入显式空文件，加载后还原为 None。"""
    case = _e2e_case("none_input")
    case.tensor_view_shapes = (None, (2, 2))
    case.tensor_storage_shapes = (None, (2, 2))
    case.tensor_view_offsets = (None, 0)
    case.tensor_view_strides = (None, (2, 1))
    case._flat_tensor_view_shapes = None
    case._flat_tensor_storage_shapes = None
    case._flat_tensor_view_offsets = None
    case._flat_tensor_view_strides = None
    case_dir = ManualDataStore(tmp_path).write_case(
        case,
        "e2e",
        [None, np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format=file_format,
    )

    loaded = ManualDataStore(tmp_path).load_case(case, "e2e")

    assert loaded.inputs[0] is None
    assert (case_dir / f"input_0_none.{file_format}").stat().st_size == 0


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_missing_none_marker_is_rejected(tmp_path, file_format):
    """None 标记文件被删除后加载报错。"""
    case = _e2e_case("missing_none_marker")
    case.tensor_view_shapes = (None, (2, 2))
    case.tensor_storage_shapes = (None, (2, 2))
    case.tensor_view_offsets = (None, 0)
    case.tensor_view_strides = (None, (2, 1))
    case._flat_tensor_view_shapes = None
    case._flat_tensor_storage_shapes = None
    case._flat_tensor_view_offsets = None
    case._flat_tensor_view_strides = None
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [None, np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format=file_format,
    )
    (case_dir / f"input_0_none.{file_format}").unlink()

    with pytest.raises(ManualDataError, match="slot count|contiguous"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_zero_element_tensor_is_not_confused_with_none(tmp_path, file_format):
    """零元素张量不被误判为 None 槽位。"""
    case = _e2e_case("zero_element_tensor")
    case.tensor_view_shapes = ((0, 3), (2, 2))
    case.tensor_storage_shapes = ((0, 3), (2, 2))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((3, 1), (2, 1))
    case._flat_tensor_view_shapes = None
    case._flat_tensor_storage_shapes = None
    case._flat_tensor_view_offsets = None
    case._flat_tensor_view_strides = None
    empty = np.empty((0, 3), np.float32)
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [empty, np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format=file_format,
    )

    loaded = store.load_case(case, "e2e")

    assert loaded.inputs[0].shape == (0, 3)
    assert loaded.inputs[0].dtype == np.dtype("float32")
    assert (case_dir / f"input_0_float32.{file_format}").is_file()


@pytest.mark.parametrize(
    ("formats", "expected_format", "expected_value"),
    [
        (("pt", "npy"), "npy", 2.0),
        (("pt", "npy", "bin"), "bin", 3.0),
    ],
)
def test_mixed_data_formats_use_whole_dataset_priority(tmp_path, formats, expected_format, expected_value):
    """混合格式数据集按整体优先级（bin > npy > pt）选择加载格式。"""
    case = _e2e_case("mixed_formats")
    values = {"pt": 1.0, "npy": 2.0, "bin": 3.0}
    target_store = ManualDataStore(tmp_path / "mixed")
    target_dir = target_store.case_dir(case.testcase_name)
    target_dir.mkdir(parents=True)
    for file_format in formats:
        source_store = ManualDataStore(tmp_path / f"source-{file_format}")
        source_dir = source_store.write_case(
            case,
            "e2e",
            [
                np.full((3, 3), values[file_format], np.float32),
                np.zeros((2, 2), np.float32),
            ],
            [np.full((2, 2), values[file_format], np.float32)],
            file_format=file_format,
        )
        for path in source_dir.iterdir():
            shutil.copy2(path, target_dir / path.name)

    loaded = target_store.load_case(case, "e2e")
    loaded_golden = loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])

    assert loaded.file_format == expected_format
    np.testing.assert_array_equal(loaded.inputs[0], np.full((3, 3), expected_value, np.float32))
    np.testing.assert_array_equal(loaded_golden[0], np.full((2, 2), expected_value, np.float32))


def test_incomplete_high_priority_format_does_not_fall_back(tmp_path):
    """高优先级格式不完整时不回退到低优先级格式。"""
    case = _e2e_case("incomplete_priority")
    target_store = ManualDataStore(tmp_path / "mixed")
    bin_dir = ManualDataStore(tmp_path / "bin").write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
        file_format="bin",
    )
    npy_dir = ManualDataStore(tmp_path / "npy").write_case(
        case,
        "e2e",
        [np.ones((3, 3), np.float32), np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format="npy",
    )
    target_dir = target_store.case_dir(case.testcase_name)
    target_dir.mkdir(parents=True)
    for source_dir in (bin_dir, npy_dir):
        for path in source_dir.iterdir():
            shutil.copy2(path, target_dir / path.name)
    (target_dir / "golden_0_float32.bin").unlink()

    loaded = target_store.load_case(case, "e2e")
    assert loaded.file_format == "bin"
    with pytest.raises(ManualDataError, match="golden slot count"):
        loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])


def test_long_case_name_maps_stably_within_directory_limit(tmp_path):
    """超长用例名稳定映射到 ≤120 字符的目录名。"""
    name = "kernel command / " + "very-long-case-" * 20
    case = _e2e_case(name)
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )

    assert len(case_dir.name) <= 120
    assert store.case_dir(name) == case_dir
    assert store.load_case(case, "e2e").case_dir == case_dir
    assert store.case_dir(name + "changed") != case_dir


@pytest.mark.parametrize("filename", ["input_1_float32.bin", "golden_0_float32.bin"])
def test_missing_data_slot_is_rejected(tmp_path, filename):
    """缺少任一数据槽位文件时加载报错。"""
    case = _e2e_case("missing_slot")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    (case_dir / filename).unlink()

    with pytest.raises(ManualDataError, match="slot count|contiguous"):
        store.load_case(case, "e2e").load_goldens(references=[np.zeros((2, 2), np.float32)])


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_nonempty_none_marker_is_rejected(tmp_path, file_format):
    """None 标记文件非空时加载 golden 报错。"""
    case = _e2e_case("none_golden")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [None],
        file_format=file_format,
    )
    (case_dir / f"golden_0_none.{file_format}").write_bytes(b"not-empty")

    loaded = store.load_case(case, "e2e")
    with pytest.raises(ManualDataError, match="must be empty"):
        loaded.load_goldens(references=[None])


def test_bin_golden_filename_shape_matches_device_output(tmp_path):
    """bin golden 文件名 shape 后缀与设备输出 shape 匹配后正确还原。"""
    case = _e2e_case("deferred_bin_shape")
    store = ManualDataStore(tmp_path)
    golden = np.arange(4, dtype=np.float32).reshape(2, 2)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [golden],
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])

    assert restored[0].shape == (2, 2)
    assert (case_dir / "golden_0_float32.bin").is_file()
    np.testing.assert_array_equal(restored[0], golden)


def test_bin_golden_same_numel_wrong_shape_legacy_suffix_rejected(tmp_path):
    """存量数据（文件名带 shape 段）同 numel 但设备输出 shape 不符时仍被拒绝。"""
    case = _e2e_case("same_numel_wrong_shape_legacy")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.arange(6, dtype=np.float32).reshape(2, 3)],
    )
    (case_dir / "golden_0_float32.bin").rename(case_dir / "golden_0_float32__shape_2x3.bin")

    loaded = store.load_case(case, "e2e")
    with pytest.raises(ManualDataError, match="saved shape .*device output shape"):
        loaded.load_goldens(references=[np.zeros((3, 2), np.float32)])


def test_bin_golden_same_numel_reshapes_by_reader_shape(tmp_path):
    """新数据无 shape 段：reshape 真值 = 读侧传入的设备输出 shape。"""
    case = _e2e_case("same_numel_reader_truth")
    store = ManualDataStore(tmp_path)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.arange(6, dtype=np.float32).reshape(2, 3)],
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.zeros((3, 2), np.float32)])

    assert restored[0].shape == (3, 2)


@pytest.mark.parametrize(
    ("shape", "token"),
    [
        ((), "scalar"),
        ((0,), "0"),
        ((2, 0, 3), "2x0x3"),
    ],
)
def test_bin_golden_restores_scalar_and_zero_dimensions_by_reader_shape(tmp_path, shape, token):
    """bin golden 标量与含 0 维度 shape 由读侧 references 还原（文件名无 shape 段）。"""
    case = _e2e_case(f"shape_{token}")
    store = ManualDataStore(tmp_path)
    golden = np.zeros(shape, dtype=np.float32)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [golden],
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.zeros(shape, np.float32)])

    assert (case_dir / "golden_0_float32.bin").is_file()
    expected_storage_shape = (1,) if shape == () else shape
    assert restored[0].shape == expected_storage_shape


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_scalar_golden_round_trip_preserves_shape(tmp_path, file_format):
    """标量 golden 三格式往返后 shape 保持为 (1,)。"""
    case = _e2e_case(f"scalar_{file_format}")
    store = ManualDataStore(tmp_path)
    golden = np.array(1.0, dtype=np.float32)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [golden],
        file_format=file_format,
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.array(0.0, np.float32)])

    assert restored[0].shape == (1,)
    np.testing.assert_array_equal(restored[0], golden)


def test_golden_sentinel_never_publishes_case_directory(tmp_path):
    """GOLDEN_FAILURE 哨兵使 write_case 报错且不发布用例目录。"""
    case = _e2e_case("failed_prepare")
    store = ManualDataStore(tmp_path)

    with pytest.raises(ManualDataError, match="sentinel"):
        store.write_case(
            case,
            "e2e",
            [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
            ["GOLDEN_FAILURE"],
        )

    assert not store.case_dir(case.testcase_name).exists()


def test_failed_reprepare_invalidates_previous_case(tmp_path):
    """重准备失败时哨兵报错并清除已有用例目录。"""
    case = _e2e_case("failed_reprepare")
    store = ManualDataStore(tmp_path)
    inputs = [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    old_golden = np.ones((2, 2), np.float32)
    store.write_case(case, "e2e", inputs, [old_golden])

    with pytest.raises(ManualDataError, match="sentinel"):
        store.write_case(case, "e2e", inputs, ["GOLDEN_FAILURE"])

    assert not store.case_dir(case.testcase_name).exists()


def test_invalidate_case_unlinks_directory_symlink_without_touching_target(tmp_path):
    """invalidate_case 解除符号链接但不影响链接目标内容。"""
    case = _e2e_case("symlinked_case")
    store = ManualDataStore(tmp_path / "prepared")
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    case_dir = store.case_dir(case.testcase_name)
    case_dir.parent.mkdir(parents=True)
    case_dir.symlink_to(external, target_is_directory=True)

    store.invalidate_case(case.testcase_name)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not case_dir.is_symlink()


def test_dump_and_load_helpers_share_store_policy(tmp_path):
    """dump/load 辅助函数共享同一 store 策略与目录。"""
    dump_case = _e2e_case("dump_helper")
    dump_switches = SimpleNamespace(no_prof=True, manual_data_dirs=(str(tmp_path),))

    dump_store = manual_data_dump_store(dump_case, dump_switches)

    assert isinstance(dump_store, ManualDataStore)
    assert not dump_store.case_dir(dump_case.testcase_name).exists()

    load_case = _e2e_case("load_helper")
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    ManualDataStore(tmp_path).write_case(load_case, "e2e", inputs, [np.zeros((2, 2), np.float32)])
    load_switches = SimpleNamespace(
        no_prof=False,
        manual_data_dirs=(str(tmp_path),),
        golden_mode="Enable",
        validate_only=False,
        force_cpu=False,
    )
    loaded_status = []

    manual_case = load_manual_data_case(load_case, "e2e", load_switches, before_load=lambda: loaded_status.append(True))

    assert loaded_status == [True]
    np.testing.assert_array_equal(manual_case.inputs[0], inputs[0])


def test_load_returns_none_for_no_prof_empty_and_missing_dirs(tmp_path):
    """no-prof（写方向）/ 无目录 / 目录全空 三种情况 load 返回 None。"""
    case = _e2e_case("load_none")
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    ManualDataStore(tmp_path).write_case(case, "e2e", inputs, [np.zeros((2, 2), np.float32)] * 2)
    base = {"golden_mode": "Enable", "validate_only": False, "force_cpu": False}

    assert (
        load_manual_data_case(case, "e2e", SimpleNamespace(no_prof=True, manual_data_dirs=(str(tmp_path),), **base))
        is None
    )
    assert load_manual_data_case(case, "e2e", SimpleNamespace(no_prof=False, manual_data_dirs=(), **base)) is None
    empty = tmp_path / "empty_root"
    empty.mkdir()
    assert (
        load_manual_data_case(case, "e2e", SimpleNamespace(no_prof=False, manual_data_dirs=(str(empty),), **base))
        is None
    )


def test_load_raises_when_case_missing_in_nonempty_directory(tmp_path):
    """目录非空但用例未命中 → raise（不静默现场生成）。"""
    other = _e2e_case("other_case")
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    ManualDataStore(tmp_path).write_case(other, "e2e", inputs, [np.zeros((2, 2), np.float32)] * 2)
    case = _e2e_case("missing_case")
    switches = SimpleNamespace(
        no_prof=False,
        manual_data_dirs=(str(tmp_path),),
        golden_mode="Enable",
        validate_only=False,
        force_cpu=False,
    )

    with pytest.raises(ManualDataError, match="was not found"):
        load_manual_data_case(case, "e2e", switches)


def test_e2e_restore_rebuilds_view_without_running_input_plugin():
    """e2e 回放按 stride/offset 重建视图且不运行输入插件。"""
    from ttk.core_modules.common.backend.torch_backend import np_to_torch_inputs

    case = _e2e_case("restore_e2e")
    storage = np.arange(9, dtype=np.float32).reshape(3, 3)
    output = np.zeros((2, 2), dtype=np.float32)
    switches = SimpleNamespace(plugin_path=("must-not-be-scanned",))
    backend = SimpleNamespace(device_type=lambda: "npu", inputs_from_numpy=np_to_torch_inputs)

    raw = generate_inputs(case, switches, backend, object(), stored_inputs=[storage, output])

    assert tuple(case.tensors[0].stride()) == (3, 1)
    assert float(case.tensors[0][0, 0]) == float(storage.ravel()[1])
    assert np.shares_memory(raw[0], storage)


def test_aclnn_restore_rebuilds_tensor_and_scalar_without_plugins(monkeypatch):
    """aclnn 回放重建张量与标量且不运行插件。"""
    case = _aclnn_case()
    switches = SimpleNamespace(plugin_path=("must-not-be-scanned",))
    monkeypatch.setattr(
        "ttk.core_modules.aclnn.input_generate.get_global_storage",
        lambda: switches,
    )
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalar = np.array(0.5, dtype=np.float32)

    InputGenerator(case).gen(stored_inputs=inputs, stored_scalars=[scalar])

    assert tuple(case.tensors[0].stride()) == (1,)
    assert float(case.scalars[0]) == pytest.approx(0.5)
