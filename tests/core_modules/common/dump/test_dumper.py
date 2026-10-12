# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""formatter / DataDumper / DumpConfig 测试。

- formatter：注册制（bin/npy/pt/print 四实现 + 未知格式 ValueError）；
  数据三形态 × 格式语义矩阵字节级校验
- DataDumper：目录布局 {root}/dump/{case}/（NPU_DUMP_PATH 覆盖、惰性创建）
  + 命名协议 {role}[_{segment}]_{idx}_{dtype}.{ext}（dump_xpu dtype 自推）
  + 开关语义 + 占位槽位跳过保 idx + dump_on_fail 补漏（list 形态 segment
  双作用 input/output）
- DumpConfig：bool 字段 + from_cli（含 xpu 位；full 四类全开）
"""

import numpy
import pytest

from ttk.core_modules.common.dump import DataDumper, DumpConfig
from ttk.core_modules.common.dump.formatter import DumpLoadError, get_formatter


class _Sw:
    """DataDumper 最小 switches 桩。"""

    def __init__(self, root, cfg=None):
        self.root_path = str(root)
        self.dump_config = cfg or DumpConfig()


# ── formatter 注册制 ───────────────────────────────────────────────────────


class TestFormatterRegistry:
    def test_four_formats_registered(self):
        for name in ("bin", "npy", "pt", "print"):
            assert get_formatter(name).ext == ("print" if name == "print" else name)

    def test_unknown_format_raises(self):
        # 未知格式报错，不静默兜底 bin
        with pytest.raises(ValueError, match="Unknown dump format"):
            get_formatter("weird")


class TestFormatterSemantics:
    def test_ndarray_bin_raw_bytes(self, tmp_path):
        arr = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
        get_formatter("bin").dump(arr, str(tmp_path), "pin_input_0")
        assert (tmp_path / "pin_input_0.bin").read_bytes() == arr.tobytes()

    def test_ndarray_npy_roundtrip(self, tmp_path):
        arr = numpy.arange(6, dtype=numpy.int32).reshape(2, 3)
        get_formatter("npy").dump(arr, str(tmp_path), "pin_output_1")
        numpy.testing.assert_array_equal(numpy.load(tmp_path / "pin_output_1.npy"), arr)

    def test_ndarray_npy_e5m2_stored_as_uint8(self, tmp_path):
        # ml_dtypes float8_e5m2 的 .npy descr 为非法 '<f1'（写出即不可读回），
        # 存储约定：降为 uint8 view 落盘，读侧 load_array._postprocess_special 还原
        ml_dtypes = pytest.importorskip("ml_dtypes")
        arr = numpy.array([1, 2, 4, 8], dtype=ml_dtypes.float8_e5m2)
        get_formatter("npy").dump(arr, str(tmp_path), "pin_fp8_0")
        back = numpy.load(tmp_path / "pin_fp8_0.npy")
        assert back.dtype == numpy.uint8
        numpy.testing.assert_array_equal(back.view(ml_dtypes.float8_e5m2), arr)

    def test_bytes_npy_e5m2_stored_as_uint8(self, tmp_path):
        ml_dtypes = pytest.importorskip("ml_dtypes")
        raw = numpy.array([1, 2, 4, 8], dtype=ml_dtypes.float8_e5m2).view(numpy.uint8).tobytes()
        get_formatter("npy").dump(raw, str(tmp_path), "pin_fp8b_0", dtype="float8_e5m2")
        back = numpy.load(tmp_path / "pin_fp8b_0.npy")
        assert back.dtype == numpy.uint8
        assert back.tobytes() == raw

    def test_ndarray_print_writes_no_file(self, tmp_path, capsys):
        arr = numpy.ones((2, 2), dtype=numpy.float16)
        get_formatter("print").dump(arr, str(tmp_path), "pin_any")
        assert list(tmp_path.iterdir()) == []
        assert capsys.readouterr().out

    def test_bytes_bin_raw_write(self, tmp_path):
        payload = b"\x01\x02\x03\x04"
        get_formatter("bin").dump(payload, str(tmp_path), "pin_bytes_0", dtype="int32")
        assert (tmp_path / "pin_bytes_0.bin").read_bytes() == payload

    def test_bytes_npy_decodes_by_dtype(self, tmp_path):
        arr = numpy.array([1.5, -2.5], dtype=numpy.float16)
        get_formatter("npy").dump(arr.tobytes(), str(tmp_path), "pin_golden_0", dtype="float16")
        back = numpy.load(tmp_path / "pin_golden_0.npy")
        assert back.dtype == numpy.float16
        numpy.testing.assert_array_equal(back, arr)

    def test_bytes_npy_no_dtype_raw_write(self, tmp_path):
        payload = b"\xff\xfe"
        get_formatter("npy").dump(payload, str(tmp_path), "p", dtype=None)
        assert (tmp_path / "p.npy").read_bytes() == payload

    def test_torch_pt_roundtrip_if_available(self, tmp_path):
        torch = pytest.importorskip("torch")
        t = torch.arange(4, dtype=torch.float32)
        get_formatter("pt").dump(t, str(tmp_path), "pin_pt_0")
        back = torch.load(tmp_path / "pin_pt_0.pt")
        numpy.testing.assert_array_equal(back.numpy(), t.numpy())

    def test_bin_load_roundtrip(self, tmp_path):
        arr = numpy.arange(6, dtype=numpy.int32).reshape(2, 3)
        get_formatter("bin").dump(arr, str(tmp_path), "pin_ld_0")
        back = get_formatter("bin").load(tmp_path / "pin_ld_0.bin", numpy.int32, (2, 3))
        numpy.testing.assert_array_equal(back, arr)

    def test_npy_load_restores_custom_dtype_view(self, tmp_path):
        # 与写侧 _as_npy_savable 同契约：自定义 dtype 以等宽 void 落盘，读侧 view 还原
        ml_dtypes = pytest.importorskip("ml_dtypes")
        arr = numpy.array([1, 2, 4, 8], dtype=ml_dtypes.bfloat16)
        get_formatter("npy").dump(arr, str(tmp_path), "pin_bf16_0")
        back = get_formatter("npy").load(tmp_path / "pin_bf16_0.npy", arr.dtype)
        assert back.dtype == arr.dtype
        numpy.testing.assert_array_equal(back.view(numpy.uint16), arr.view(numpy.uint16))

    def test_load_shape_mismatch_raises_dump_load_error(self, tmp_path):
        arr = numpy.ones((2, 3), dtype=numpy.float32)
        get_formatter("npy").dump(arr, str(tmp_path), "pin_sh_0")
        with pytest.raises(DumpLoadError, match="stored shape"):
            get_formatter("npy").load(tmp_path / "pin_sh_0.npy", numpy.float32, (3, 2))

    def test_pt_load_raw_bytes_payload(self, tmp_path):
        torch = pytest.importorskip("torch")
        raw = numpy.arange(4, dtype=numpy.float32).tobytes()
        payload = {
            "ttk_raw_bytes": torch.from_numpy(numpy.frombuffer(raw, dtype=numpy.uint8).copy()),
            "ttk_shape": (4,),
        }
        torch.save(payload, str(tmp_path / "pin_raw_0.pt"))
        back = get_formatter("pt").load(tmp_path / "pin_raw_0.pt", numpy.float32, (4,))
        numpy.testing.assert_array_equal(back, numpy.arange(4, dtype=numpy.float32))


# ── DumpConfig ─────────────────────────────────────────────────────────────


class TestDumpConfig:
    def test_from_cli_tokens(self):
        cfg = DumpConfig.from_cli("in,out,golden")
        assert cfg.dump_input
        assert cfg.dump_output
        assert cfg.dump_golden

    def test_from_cli_full_includes_xpu(self):
        cfg = DumpConfig.from_cli("full")
        assert cfg.dump_input
        assert cfg.dump_output
        assert cfg.dump_golden
        assert cfg.dump_xpu is True

    def test_from_cli_xpu(self):
        cfg = DumpConfig.from_cli("xpu")
        assert cfg.dump_xpu
        assert not cfg.dump_input

    def test_from_cli_none(self):
        cfg = DumpConfig.from_cli(None)
        assert not any((cfg.dump_input, cfg.dump_output, cfg.dump_golden, cfg.dump_xpu))


# ── DataDumper：目录 + 命名 + 开关 ─────────────────────────────────────────


class TestDataDumper:
    def test_layout_and_naming(self, tmp_path):
        cfg = DumpConfig(dump_input=True, dump_output=True, dump_golden=True)
        dumper = DataDumper(_Sw(tmp_path, cfg), "softmax_v2")
        arr = numpy.ones((2,), dtype=numpy.float32)
        dumper.dump_inputs([arr], segment="dyn")
        dumper.dump_outputs([arr.tobytes()], dtypes=["float32"], segment="cst")
        dumper.dump_goldens([arr])
        case_dir = tmp_path / "dump" / "softmax_v2"
        assert (case_dir / "input_dyn_0_float32.bin").exists()
        assert (case_dir / "output_cst_0_float32.bin").exists()
        assert (case_dir / "golden_0_float32.bin").exists()

    def test_npu_dump_path_env_override(self, tmp_path, monkeypatch):
        override = tmp_path / "external"
        monkeypatch.setenv("NPU_DUMP_PATH", str(override))
        dumper = DataDumper(_Sw(tmp_path, DumpConfig(dump_input=True)), "case")
        assert str(dumper.dump_dir).startswith(str(override))

    def test_gates_respect_config(self, tmp_path):
        cfg = DumpConfig()  # 全关
        dumper = DataDumper(_Sw(tmp_path, cfg), "case")
        dumper.dump_inputs([numpy.ones(1)])
        dumper.dump_outputs([numpy.ones(1)])
        dumper.dump_goldens([numpy.ones(1)])
        # 开关全关：不落任何文件，也不产生空用例目录（惰性建目录）
        assert not (tmp_path / "dump" / "case").exists()

    def test_force_bypasses_gate(self, tmp_path):
        cfg = DumpConfig()
        dumper = DataDumper(_Sw(tmp_path, cfg), "case")
        dumper.dump_inputs([numpy.ones(1, dtype=numpy.float32)], segment="dyn", force=True)
        assert (tmp_path / "dump" / "case" / "input_dyn_0_float32.bin").exists()

    def test_placeholder_slots_skip_but_keep_index(self, tmp_path):
        # None/str 占位槽位（可选输出未落地、SUPPRESSED 等）跳过但保留原 idx
        # ——文件名索引与 CSV 槽位对齐，不因占位前移
        dumper = DataDumper(_Sw(tmp_path, DumpConfig(dump_input=True, dump_output=True, dump_golden=True)), "case")
        arr = numpy.ones(1, dtype=numpy.float32)
        dumper.dump_inputs([None, arr])
        dumper.dump_outputs(["SUPPRESSED", arr])
        dumper.dump_goldens(["GOLDEN_FAILURE", None, arr])
        case_dir = tmp_path / "dump" / "case"
        assert sorted(p.name for p in case_dir.iterdir()) == [
            "golden_2_float32.bin",
            "input_1_float32.bin",
            "output_1_float32.bin",
        ]

    def test_dump_xpu_naming_embeds_dtype(self, tmp_path):
        # dtype 从数据自推进文件名（xpu_torch_0_float16.bin 契约）
        dumper = DataDumper(_Sw(tmp_path, DumpConfig(dump_xpu=True)), "case")
        arr = numpy.ones((2,), dtype=numpy.float16)
        dumper.dump_xpu({"torch": [arr, None, b"\x01"]})
        names = sorted(p.name for p in (tmp_path / "dump" / "case").iterdir())
        assert names == ["xpu_torch_0_float16.bin", "xpu_torch_2_none.bin"]

    def test_dump_on_fail_segment_applies_to_input_and_output(self, tmp_path):
        # list 形态 outputs：segment 双作用于 input/output（GEIR 调用方式）
        dumper = DataDumper(_Sw(tmp_path, DumpConfig(dump_on_fail=True)), "case")
        arr = numpy.ones(1, dtype=numpy.int32)
        dumper.dump_on_fail([arr], [arr], [arr], segment="const")
        case_dir = tmp_path / "dump" / "case"
        assert (case_dir / "input_const_0_int32.bin").exists()
        assert (case_dir / "output_const_0_int32.bin").exists()
        assert (case_dir / "golden_0_int32.bin").exists()

    def test_dump_on_fail_fills_disabled_slots_only(self, tmp_path):
        cfg = DumpConfig(dump_input=True, dump_on_fail=True)
        dumper = DataDumper(_Sw(tmp_path, cfg), "case")
        arr = numpy.ones(1, dtype=numpy.int32)
        dumper.dump_on_fail([arr], [arr], [arr])
        # input 已开：不重复写；output/golden 未开：force 补漏
        assert not (tmp_path / "dump" / "case" / "input_0_int32.bin").exists()
        assert (tmp_path / "dump" / "case" / "output_0_int32.bin").exists()
        assert (tmp_path / "dump" / "case" / "golden_0_int32.bin").exists()

    def test_dump_on_fail_respects_switch(self, tmp_path):
        cfg = DumpConfig()  # dump_on_fail 关
        dumper = DataDumper(_Sw(tmp_path, cfg), "case")
        dumper.dump_on_fail([numpy.ones(1)], [numpy.ones(1)], [numpy.ones(1)])
        assert not (tmp_path / "dump" / "case").exists()

    def test_dump_named_tiling(self, tmp_path):
        dumper = DataDumper(_Sw(tmp_path, DumpConfig()), "case")
        dumper.dump_named("tiling_data_dyn", b"\x00\x01")
        assert (tmp_path / "dump" / "case" / "tiling_data_dyn.bin").read_bytes() == b"\x00\x01"

    def test_none_dtype_placeholder(self, tmp_path):
        dumper = DataDumper(_Sw(tmp_path, DumpConfig(dump_output=True)), "case")
        dumper.dump_outputs([b"\x01"])  # 无 dtype
        assert (tmp_path / "dump" / "case" / "output_0_none.bin").exists()

    def test_case_directory_sanitized(self, tmp_path):
        dumper = DataDumper(_Sw(tmp_path, DumpConfig()), "a/b c")
        assert "/" not in dumper.dump_dir.split("dump/")[-1]
