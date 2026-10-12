# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""validate_outputs + 懒加载注册 + batch 一致性单测。"""

import numpy as np
import pytest

from ttk.core_modules.common.comparison import compare_batch_consistency, get_comparison
from ttk.core_modules.common.comparison.batch_consistency import (
    BatchSummary,
    assemble_batch_summary,
    slice_batch_hashes,
)
from ttk.core_modules.common.comparison.comparison import PLACEHOLDER_PASS, validate_outputs


class TestValidateOutputs:
    def test_empty_outputs(self):
        verdict = validate_outputs([], [])
        assert verdict.needs_compare is False
        assert verdict.status == "EMPTY"

    def test_all_valid(self):
        verdict = validate_outputs([np.ones(2), np.ones(2)], [np.ones(2), np.ones(2)])
        assert verdict.needs_compare is True
        assert verdict.per_output == ["VALID", "VALID"]
        assert verdict.status is None

    def test_golden_placeholder_fake_pass(self):
        # 模式开关/显式豁免 → fake-pass
        for token in ("SUPPRESSED", "DYN_OFF"):
            verdict = validate_outputs([np.ones(2)], [token])
            assert verdict.needs_compare is False
            assert verdict.per_output == [(token, True)], token

    def test_golden_failure_placeholder_fails(self):
        # golden 生成失败类 → FAIL（真值缺失必须暴露）
        for token in ("UNSUPPORTED", "GOLDEN_TIMEOUT", "GOLDEN_FAILURE"):
            verdict = validate_outputs([np.ones(2)], [token])
            assert verdict.needs_compare is False
            assert verdict.per_output == [(token, False)], token
            assert token not in PLACEHOLDER_PASS, token

    def test_unknown_string_fails(self):
        verdict = validate_outputs([np.ones(2)], ["SOMETHING_ELSE"])
        assert verdict.per_output == [("SOMETHING_ELSE", False)]

    def test_output_none_fails(self):
        verdict = validate_outputs([None], [np.ones(2)])
        assert verdict.per_output == [("NO_OUTPUT", False)]

    def test_golden_none_suppressed(self):
        verdict = validate_outputs([np.ones(2)], [None])
        assert verdict.per_output == [("SUPPRESSED", True)]

    def test_third_party_count_mismatch(self):
        verdict = validate_outputs([np.ones(2), np.ones(2)], [np.ones(2), np.ones(2)], third_parties=[np.ones(2)])
        assert verdict.status == "THIRD_PARTY_COUNT_MISMATCH"
        assert verdict.needs_compare is False

    def test_mixed_placeholder_valid(self):
        verdict = validate_outputs(["DYN_OFF", np.ones(2)], [np.ones(2), np.ones(2)])
        assert verdict.needs_compare is True
        assert verdict.per_output[0] == ("DYN_OFF", True)
        assert verdict.per_output[1] == "VALID"


class TestPlaceholderPassTable:
    def test_kernel_perf_tokens_included(self):
        # perf 判定与精度列共用单表
        for token in (
            "DYN_OFF",
            "CST_OFF",
            "BIN_OFF",
            "DYN_UNSUPPORTED",
            "BIN_OPERATOR_NOT_FOUND",
            "DYN_INPUT_MISSING",
        ):
            assert token in PLACEHOLDER_PASS, token

    def test_fail_tokens_excluded(self):
        # golden 生成失败类不进 fake-pass 表（判 FAIL）
        for token in ("NO_OUTPUT", "COMPARE_FAILURE", "SOMETHING", "UNSUPPORTED", "GOLDEN_TIMEOUT", "GOLDEN_FAILURE"):
            assert token not in PLACEHOLDER_PASS, token


class TestLazyRegistry:
    def test_get_comparison_known_tokens(self):
        for token, cls_name in [
            ("binary_equal", "BinaryComparison"),
            ("bin", "BinaryComparison"),
            ("mix_tolerance", "MixToleranceComparison"),
            ("mixed", "MixToleranceComparison"),
            ("close", "CloseComparison"),
            ("isclose", "CloseComparison"),
            ("stat_rel_err", "StatRelErrComparison"),
            ("cosine", "CosineSimilarityComparison"),
            ("cross_check", "CrossCheckComparison"),
            ("quant", "QuantComparison"),
            ("requant", "ReQuantizeComparison"),
        ]:
            cls = get_comparison(token)
            assert cls.__name__ == cls_name, token

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            get_comparison("nonexistent_standard")


class _Tc:
    testcase_name = "t"
    api_name = "a"


class TestSliceBatchHashes:
    def test_slice_and_hash(self):
        output = np.arange(12, dtype=np.float32).reshape(3, 4)
        summary = slice_batch_hashes(output, "case", "bcid-1", batch_axis=0, batch_slice_info="0:3")
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3
        assert all(len(h) == 32 for h in summary.sample_hashes)

    def test_identical_rows_same_hash(self):
        output = np.ones((3, 4), dtype=np.float32)
        summary = slice_batch_hashes(output, "case", "bcid-1", 0, "0:3")
        assert len(set(summary.sample_hashes)) == 1

    def test_missing_meta(self):
        summary = slice_batch_hashes(np.ones((2, 2)), "case", "bcid", None, None)
        assert summary.sample_hashes is None
        assert summary.unsupported_reason == "BATCH_META_MISSING"

    def test_axis_too_large(self):
        summary = slice_batch_hashes(np.ones((2, 2)), "case", "bcid", 5, "0:2")
        assert summary.unsupported_reason == "UNSLICEABLE"

    def test_non_ndarray(self):
        summary = slice_batch_hashes(b"\x00\x01", "case", "bcid", 0, "0:2")
        assert summary.unsupported_reason == "UNSLICEABLE"

    def test_torch_tensor_coerced(self):
        """aclnn 比对后 output_bytes 是 torch 视图：asarray 归一后可切。"""
        torch = pytest.importorskip("torch")
        output = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        summary = slice_batch_hashes(output, "case", "bcid-1", 0, "0:3")
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3


class _BatchCase:
    """assemble 消费的最小 testcase 契约。"""

    def __init__(self, batch_axis, batch_slice_info, name="t"):
        self.testcase_name = name
        self.batch_axis = batch_axis
        self.batch_slice_info = batch_slice_info


class TestAssembleBatchSummary:
    def test_nested_batch_axis_derives_scalar(self):
        """嵌套 batch_axis（slot→轴组）派生标量输出轴（Q-only 契约）。"""
        case = _BatchCase(((0,),), ((((0, 3, 1),),),))
        output = np.arange(12, dtype=np.float32).reshape(3, 4)
        summary = assemble_batch_summary([output], case, "bcid-1")
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3
        assert summary.batch_axis == 0

    def test_b_s_axes_pick_batch_axis(self):
        """(0, 1) 双轴组：取首组（B 轴）切输出。"""
        case = _BatchCase(((0, 1),), (((((0, 3, 1),), ((0, 4, 1),))),))
        output = np.arange(12, dtype=np.float32).reshape(3, 4)
        summary = assemble_batch_summary([output], case, "bcid-1")
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3

    def test_nderivable_axis_unsupported(self):
        case = _BatchCase((), None)
        output = np.ones((2, 2))
        summary = assemble_batch_summary([output], case, "bcid-1")
        assert summary.sample_hashes is None
        assert summary.unsupported_reason == "BATCH_META_MISSING"

    def test_none_id_or_empty_outputs(self):
        case = _BatchCase(((0,),), ((((0, 3, 1),),),))
        assert assemble_batch_summary([], case, "bcid") is None
        assert assemble_batch_summary([np.ones(2)], case, None) is None
        assert assemble_batch_summary([None], case, "bcid") is None
        assert assemble_batch_summary(["DYN_OFF"], case, "bcid") is None


class TestCompareBatchConsistency:
    def _summary(self, name, bcid, hashes, reason=None):
        return BatchSummary(
            testcase_name=name,
            batch_consistency_id=bcid,
            sample_hashes=hashes,
            unsupported_reason=reason,
        )

    def test_consistent_group_passes(self):
        h = ["a" * 32, "b" * 32]
        summaries = {
            "c1": self._summary("c1", "g1", list(h)),
            "c2": self._summary("c2", "g1", list(h)),
        }
        ok, failures = compare_batch_consistency(summaries)
        assert ok is True
        assert failures == {}

    def test_inconsistent_position_fails(self):
        summaries = {
            "c1": self._summary("c1", "g1", ["a" * 32, "b" * 32]),
            "c2": self._summary("c2", "g1", ["a" * 32, "c" * 32]),
        }
        ok, failures = compare_batch_consistency(summaries)
        assert ok is False
        assert "sample[1] differs" in failures["g1"][0]

    def test_sample_count_mismatch_fails(self):
        summaries = {
            "c1": self._summary("c1", "g1", ["a" * 32]),
            "c2": self._summary("c2", "g1", ["a" * 32, "b" * 32]),
        }
        ok, failures = compare_batch_consistency(summaries)
        assert ok is False
        assert "sample count mismatch" in failures["g1"][0]

    def test_unsupported_skipped(self):
        summaries = {
            "c1": self._summary("c1", "g1", None, reason="UNSLICEABLE"),
            "c2": self._summary("c2", "g1", ["a" * 32]),
        }
        ok, failures = compare_batch_consistency(summaries)
        assert ok is True

    def test_single_member_group_skipped(self):
        summaries = {"c1": self._summary("c1", "g1", ["a" * 32])}
        ok, _ = compare_batch_consistency(summaries)
        assert ok is True
