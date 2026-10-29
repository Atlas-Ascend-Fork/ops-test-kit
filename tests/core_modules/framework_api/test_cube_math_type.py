# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""CubeMathType 精度模式支持（worker._setup_cube_math_type / _cube_math_type_value）
与 4bit golden dtype 归一化（worker._normalize_output_dtype_for_4bit_golden）。

- 门控：非 NPU 后端 / 非 torch 框架直接返回（不导入 torch_npu）；
- 按需复位：未携带 CubeMathType 且本进程未改写过时不触碰 torch_npu 默认；
  携带该键、或前序改写过时先回 KEEP_DTYPE + allow_hf32=False 基线再按值覆盖；
- 值 3（USE_HF32）额外置 matmul/conv allow_hf32=True；非法值告警并保持基线；
- 4bit 归一化：golden 为 4bit dtype 时输出 uint8/宽 dtype 统一 unpack 对齐。
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.framework_api import worker as e2e_worker


def _cpu_backend():
    return SimpleNamespace(is_npu=lambda: False)


def _npu_backend():
    return SimpleNamespace(is_npu=lambda: True)


def _ctx(storage=None):
    return SimpleNamespace(storage=storage if storage is not None else {})


def _case(api_name="torch.matmul", attributes=None):
    return SimpleNamespace(api_name=api_name, attributes=attributes)


def _has_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


def _import_torch_npu():
    import torch_npu

    return torch_npu


def _missing_ml_dtypes() -> bool:
    try:
        import ml_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


class TestCubeMathTypeValue:
    def test_torch_case_reads_attribute(self):
        assert e2e_worker._cube_math_type_value(_case(attributes={"CubeMathType": 3})) == 3

    def test_missing_attribute_is_none(self):
        assert e2e_worker._cube_math_type_value(_case(attributes={})) is None
        assert e2e_worker._cube_math_type_value(_case(attributes=None)) is None

    def test_tf_case_is_none(self):
        # TF 通路无 cube 精度模式概念，即使误填属性也不取值
        assert (
            e2e_worker._cube_math_type_value(_case(api_name="tf.linalg.matmul", attributes={"CubeMathType": 3})) is None
        )


class TestSetupCubeMathType:
    def test_cpu_backend_skips_without_importing_torch_npu(self, monkeypatch):
        # sys.modules 占位 None：import torch_npu 会抛 ImportError，借此断言早退不触发导入
        monkeypatch.setitem(sys.modules, "torch_npu", None)
        e2e_worker._setup_cube_math_type(_cpu_backend(), _case(attributes={"CubeMathType": 3}), _ctx())

    def test_tf_framework_skips_without_importing_torch_npu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_npu", None)
        e2e_worker._setup_cube_math_type(
            _npu_backend(), _case(api_name="tf.linalg.matmul", attributes={"CubeMathType": 3}), _ctx()
        )

    def test_clean_process_skips_without_importing_torch_npu(self, monkeypatch):
        # 未携带该键且前序用例未改写过：不触碰 torch_npu 进程默认（不导入）
        monkeypatch.setitem(sys.modules, "torch_npu", None)
        e2e_worker._setup_cube_math_type(_npu_backend(), _case(attributes={}), _ctx())

    @pytest.mark.skipif(not _has_torch_npu(), reason="torch_npu is not installed")
    def test_clean_process_keeps_torch_npu_state_without_key(self):
        torch_npu = _import_torch_npu()
        before = (
            torch_npu.npu.matmul.cube_math_type,
            torch_npu.npu.matmul.allow_hf32,
            torch_npu.npu.conv.allow_hf32,
        )
        e2e_worker._setup_cube_math_type(_npu_backend(), _case(attributes={}), _ctx())
        after = (
            torch_npu.npu.matmul.cube_math_type,
            torch_npu.npu.matmul.allow_hf32,
            torch_npu.npu.conv.allow_hf32,
        )
        assert after == before

    @pytest.mark.skipif(not _has_torch_npu(), reason="torch_npu is not installed")
    def test_dirty_process_resets_baseline_without_key(self):
        torch_npu = _import_torch_npu()
        npu = _npu_backend()
        ctx = _ctx()
        e2e_worker._setup_cube_math_type(npu, _case(attributes={"CubeMathType": 3}), ctx)
        assert torch_npu.npu.matmul.cube_math_type == torch_npu.npu.CubeMathType.USE_HF32
        # 前序改写过：无键用例须复位基线，防跨用例状态泄漏
        e2e_worker._setup_cube_math_type(npu, _case(attributes={}), ctx)
        assert torch_npu.npu.matmul.cube_math_type == torch_npu.npu.CubeMathType.KEEP_DTYPE
        assert torch_npu.npu.matmul.allow_hf32 is False
        assert torch_npu.npu.conv.allow_hf32 is False

    @pytest.mark.skipif(not _has_torch_npu(), reason="torch_npu is not installed")
    def test_apply_and_reset(self):
        torch_npu = _import_torch_npu()
        npu = _npu_backend()
        ctx = _ctx()

        e2e_worker._setup_cube_math_type(npu, _case(attributes={"CubeMathType": 3}), ctx)
        assert torch_npu.npu.matmul.cube_math_type == torch_npu.npu.CubeMathType.USE_HF32
        assert torch_npu.npu.matmul.allow_hf32 is True
        assert torch_npu.npu.conv.allow_hf32 is True

        # 值 2：USE_FP16，allow_hf32 不额外置位
        e2e_worker._setup_cube_math_type(npu, _case(attributes={"CubeMathType": "2"}), ctx)
        assert torch_npu.npu.matmul.cube_math_type == torch_npu.npu.CubeMathType.USE_FP16
        assert torch_npu.npu.matmul.allow_hf32 is False

        # 同进程前序已改写（ctx 标记）：无属性用例复位基线，无跨用例泄漏
        e2e_worker._setup_cube_math_type(npu, _case(attributes={}), ctx)
        assert torch_npu.npu.matmul.cube_math_type == torch_npu.npu.CubeMathType.KEEP_DTYPE
        assert torch_npu.npu.matmul.allow_hf32 is False
        assert torch_npu.npu.conv.allow_hf32 is False

    @pytest.mark.skipif(not _has_torch_npu(), reason="torch_npu is not installed")
    def test_invalid_value_warns_and_keeps_baseline(self, caplog):
        torch_npu = _import_torch_npu()
        with caplog.at_level("WARNING"):
            e2e_worker._setup_cube_math_type(_npu_backend(), _case(attributes={"CubeMathType": 99}), _ctx())
        assert torch_npu.npu.matmul.cube_math_type == torch_npu.npu.CubeMathType.KEEP_DTYPE
        assert "CubeMathType" in caplog.text


@pytest.mark.skipif(_missing_ml_dtypes(), reason="ml_dtypes is not installed")
class TestNormalize4BitGolden:
    def _int4(self):
        from ml_dtypes import int4

        return int4

    def test_uint8_output_aligned_to_int4_golden(self):
        int4 = self._int4()
        golden = np.array([1, 2, 3, 4], dtype=int4)
        packed = np.array([0x21, 0x43], dtype=np.uint8)  # 低半字节在前
        outputs = [packed]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].dtype == int4
        assert outputs[0].tolist() == golden.tolist()

    def test_wide_dtype_output_aligned_to_4bit_golden(self):
        # int32 输出（每 32bit 元素 8 个 4bit 值）：view uint8 后 unpack
        int4 = self._int4()
        golden = np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=int4)
        outputs = [np.array([0x21], dtype=np.int32)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].dtype == int4
        assert outputs[0].tolist() == golden.tolist()

    def test_non_4bit_golden_untouched(self):
        golden = np.zeros((2, 2), dtype=np.float32)
        outputs = [np.ones((2, 2), dtype=np.int32)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].dtype == np.int32

    def test_same_dtype_and_none_slots_skipped(self):
        int4 = self._int4()
        golden = np.array([1, 2], dtype=int4)
        outputs = [np.array([1, 2], dtype=int4), None]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden, None])
        assert outputs[0].tolist() == [1, 2]
        assert outputs[1] is None

    def test_length_mismatch_no_crash(self):
        int4 = self._int4()
        golden = [np.array([1, 2], dtype=int4)]
        outputs = [np.array([0x21], dtype=np.uint8), np.array([9], dtype=np.int32)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, golden)
        assert outputs[0].dtype == int4
        assert outputs[1].dtype == np.int32

    def test_odd_size_packed_output_trims_padding_nibble(self):
        # pack_4bits 奇数补零约定：末字节高 nibble 为填充 0，裁齐 golden 元素数
        int4 = self._int4()
        golden = np.array([1, 2, 3, 4, 5], dtype=int4)
        outputs = [np.array([0x21, 0x43, 0x05], dtype=np.uint8)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].dtype == int4
        assert outputs[0].tolist() == golden.tolist()

    def test_odd_size_wide_dtype_trims_padding(self):
        # 1 个 int32（8 个 4bit 槽）装 5 值 + 3 填充：粒度内全零尾部裁齐 golden
        int4 = self._int4()
        golden = np.array([1, 2, 3, 4, 5], dtype=int4)
        outputs = [np.array([0x00054321], dtype=np.int32)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].tolist() == golden.tolist()

    def test_size_anomaly_beyond_pack_granularity_not_trimmed(self):
        # 5 值只需 3 字节，第 4 字节超出打包粒度：尺寸异常不裁，长度不齐暴露给比对
        int4 = self._int4()
        golden = np.array([1, 2, 3, 4, 5], dtype=int4)
        outputs = [np.array([0x21, 0x43, 0x05, 0x00], dtype=np.uint8)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].size == 8

    def test_nonzero_tail_not_trimmed(self):
        # 尾部 nibble 非零（非填充）：不裁，避免把超尺寸输出截断成假 PASS
        int4 = self._int4()
        golden = np.array([1, 2, 3, 4, 5], dtype=int4)
        outputs = [np.array([0x21, 0x43, 0x65], dtype=np.uint8)]
        e2e_worker._normalize_output_dtype_for_4bit_golden(outputs, [golden])
        assert outputs[0].size == 6
        assert outputs[0].tolist() == [1, 2, 3, 4, 5, 6]
