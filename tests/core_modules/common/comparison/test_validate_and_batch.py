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
    SampleHash,
    assemble_batch_summary,
    slice_batch_hashes,
)
from ttk.core_modules.common.comparison.comparison import PLACEHOLDER_PASS, _fold_pass, compare, validate_outputs


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
        # 模式开关占位（golden 侧）→ fake-pass
        for token in ("DYN_OFF",):
            verdict = validate_outputs([np.ones(2)], [token])
            assert verdict.needs_compare is False
            assert verdict.per_output == [(token, True)], token

    def test_golden_disabled_suppressed_skips(self):
        # golden 侧 "SUPPRESSED"（--golden-mode Disable）→ SKIP；
        # output 为模式占位哨兵时仍 SKIP（Disable 优先）
        verdict = validate_outputs([np.ones(2)], ["SUPPRESSED"])
        assert verdict.needs_compare is False
        assert verdict.per_output == [("SUPPRESSED", None)]
        verdict = validate_outputs(["DYN_OFF"], ["SUPPRESSED"])
        assert verdict.per_output == [("SUPPRESSED", None)]

    def test_operator_not_found_fails(self):
        # 算子未找到是确定失败，不属"主动不比对"
        for token in ("DYN_OPERATOR_NOT_FOUND", "CST_OPERATOR_NOT_FOUND", "BIN_OPERATOR_NOT_FOUND"):
            verdict = validate_outputs([token], [np.ones(2)])
            assert verdict.per_output == [(token, False)], token
            assert token not in PLACEHOLDER_PASS, token

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
            "SUPPRESSED",
        ):
            assert token in PLACEHOLDER_PASS, token

    def test_fail_tokens_excluded(self):
        # golden 生成失败类 + 算子未找到/输入缺失类不进 fake-pass 表（判 FAIL）
        for token in (
            "NO_OUTPUT",
            "COMPARE_FAILURE",
            "SOMETHING",
            "UNSUPPORTED",
            "GOLDEN_TIMEOUT",
            "GOLDEN_FAILURE",
            "DYN_OPERATOR_NOT_FOUND",
            "CST_OPERATOR_NOT_FOUND",
            "BIN_OPERATOR_NOT_FOUND",
            "STC_OPERATOR_NOT_FOUND",
            "DYN_INPUT_MISSING",
        ):
            assert token not in PLACEHOLDER_PASS, token


class TestTriStateVerdict:
    """compare() 三态返回：True=PASS / False=FAIL / None=SKIP。"""

    def test_golden_disabled_returns_skip(self):
        precision, _, passed, metrics = compare([np.ones(2)], ["SUPPRESSED"], ("float32",), standards=[None])
        assert precision == "SUPPRESSED"
        assert passed is None
        assert metrics == {}

    def test_per_output_exemption_still_passes(self):
        # golden=None（shape 0 输出等单输出豁免）保持 PASS
        precision, _, passed, _ = compare([np.ones(2)], [None], ("float32",), standards=[None])
        assert precision == "SUPPRESSED"
        assert passed is True

    def test_fold_pass_precedence(self):
        assert _fold_pass([True, True]) is True
        assert _fold_pass([False, None]) is False  # FAIL 压过 SKIP
        assert _fold_pass([True, None]) is None  # 有未比对位且无失败 → SKIP
        assert _fold_pass([None]) is None
        assert _fold_pass([]) is True

    def test_output_sentinel_not_found_fails(self):
        precision, _, passed, _ = compare(["DYN_OPERATOR_NOT_FOUND"], [np.ones(2)], ("float32",), standards=[None])
        assert passed is False
        assert precision == "DYN_OPERATOR_NOT_FOUND"


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


# batch 三字段嵌套样例：q 槽声明轴 0，3 个样本切片 + 逐样本种子。
_AXIS = ((0,),)
_SLICES = ([[(0, 1, 1), (1, 2, 1), (2, 3, 1)]],)
_SEEDS = ([[7, 8, 9]],)


class TestSliceBatchHashes:
    def test_slice_and_hash(self):
        output = np.arange(12, dtype=np.float32).reshape(3, 4)
        summary = slice_batch_hashes(output, "case", "bcid-1", (_AXIS, _SLICES, _SEEDS))
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3
        assert all(len(sample.digest) == 32 for sample in summary.sample_hashes)
        assert [sample.seeds for sample in summary.sample_hashes] == [(7,), (8,), (9,)]
        assert [sample.lengths for sample in summary.sample_hashes] == [(1,), (1,), (1,)]

    def test_identical_rows_same_hash(self):
        output = np.ones((3, 4), dtype=np.float32)
        summary = slice_batch_hashes(output, "case", "bcid-1", (_AXIS, _SLICES, _SEEDS))
        assert len({sample.digest for sample in summary.sample_hashes}) == 1

    def test_undeclared_positions_not_hashed(self):
        """只 hash 声明的样本切片位置；未声明行不参与。"""
        output = np.arange(16, dtype=np.float32).reshape(4, 4)
        slices = ([[(0, 1, 1), (2, 3, 1)]],)
        seeds = ([[7, 8]],)
        summary = slice_batch_hashes(output, "case", "bcid-1", (_AXIS, slices, seeds))
        assert len(summary.sample_hashes) == 2

    def test_multi_axis_sample_slice(self):
        """(0, 1) 双轴：每样本在每轴各带切片，长度按轴聚合。"""
        output = np.arange(8, dtype=np.float32).reshape(2, 4)
        slices = ([[(0, 1, 1), (1, 2, 1)], [(0, 2, 1), (2, 4, 1)]],)
        seeds = ([[7, 8], [7, 8]],)
        summary = slice_batch_hashes(output, "case", "bcid-1", (((0, 1),), slices, seeds))
        assert len(summary.sample_hashes) == 2
        assert summary.sample_hashes[0].key == ((7, 7), (1, 2))
        assert summary.sample_hashes[1].key == ((8, 8), (1, 2))

    def test_missing_meta(self):
        summary = slice_batch_hashes(np.ones((2, 2)), "case", "bcid", (None, None, None))
        assert summary.sample_hashes is None
        assert summary.unsupported_reason == "BATCH_META_MISSING"

    def test_axis_too_large(self):
        summary = slice_batch_hashes(np.ones((2, 2)), "case", "bcid", (((5,),), ([[(0, 1, 1)]],), ([[7]],)))
        assert summary.sample_hashes is None
        assert summary.unsupported_reason == "UNSLICEABLE"

    def test_non_ndarray(self):
        summary = slice_batch_hashes(b"\x00\x01", "case", "bcid", (_AXIS, _SLICES, _SEEDS))
        assert summary.unsupported_reason == "UNSLICEABLE"

    def test_torch_tensor_coerced(self):
        """aclnn 比对后 output_bytes 是 torch 视图：asarray 归一后可切。"""
        torch = pytest.importorskip("torch")
        output = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        summary = slice_batch_hashes(output, "case", "bcid-1", (_AXIS, _SLICES, _SEEDS))
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3


class _BatchCase:
    """assemble 消费的最小 testcase 契约。"""

    def __init__(self, batch_axis, batch_slice_info, batch_seed, name="t"):
        self.testcase_name = name
        self.batch_axis = batch_axis
        self.batch_slice_info = batch_slice_info
        self.batch_seed = batch_seed


class TestAssembleBatchSummary:
    def test_nested_batch_fields_hashed(self):
        """嵌套 batch 三字段（slot→轴组→样本）→ 按声明样本切输出。"""
        case = _BatchCase(_AXIS, _SLICES, _SEEDS)
        output = np.arange(12, dtype=np.float32).reshape(3, 4)
        summary = assemble_batch_summary([output], case, "bcid-1")
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 3
        assert summary.batch_axis == _AXIS

    def test_b_s_axes_sample_hash(self):
        """(0, 1) 双轴组样例：样本在两轴上各取切片。"""
        case = _BatchCase(
            ((0, 1),),
            ([[(0, 1, 1), (1, 2, 1)], [(0, 2, 1), (2, 4, 1)]],),
            ([[7, 8], [7, 8]],),
        )
        output = np.arange(8, dtype=np.float32).reshape(2, 4)
        summary = assemble_batch_summary([output], case, "bcid-1")
        assert summary.sample_hashes is not None
        assert len(summary.sample_hashes) == 2

    def test_nderivable_axis_unsupported(self):
        case = _BatchCase((), None, None)
        output = np.ones((2, 2))
        summary = assemble_batch_summary([output], case, "bcid-1")
        assert summary.sample_hashes is None
        assert summary.unsupported_reason == "BATCH_META_MISSING"

    def test_none_id_or_empty_outputs(self):
        case = _BatchCase(_AXIS, _SLICES, _SEEDS)
        assert assemble_batch_summary([], case, "bcid") is None
        assert assemble_batch_summary([np.ones(2)], case, None) is None
        assert assemble_batch_summary([None], case, "bcid") is None
        assert assemble_batch_summary(["DYN_OFF"], case, "bcid") is None


def _summary(name, bcid, keyed_hashes, reason=None, api="torch.ops.demo.op"):
    """keyed_hashes: [( (seeds,), (lengths,), digest ), ...]；None = 不支持切片。"""
    samples = (
        None
        if keyed_hashes is None
        else [SampleHash(seeds=seeds, lengths=lengths, digest=digest) for seeds, lengths, digest in keyed_hashes]
    )
    return BatchSummary(
        testcase_name=name,
        batch_consistency_id=bcid,
        sample_hashes=samples,
        unsupported_reason=reason,
        api_name=api,
    )


class TestCompareBatchConsistency:
    def test_consistent_group_passes(self):
        hashes = [((7,), (1,), "a" * 32), ((8,), (1,), "b" * 32)]
        summaries = {
            "c1": _summary("c1", "g1", hashes),
            "c2": _summary("c2", "g1", hashes),
        }
        ok, failures, groups, failed_cases = compare_batch_consistency(summaries)
        assert ok is True
        assert failures == {}
        assert groups == 2
        assert failed_cases == set()

    def test_reorder_seed_pairs_by_key(self):
        """同 seed 样本跨用例换序：按 api+seed+长度分组对齐，而非声明顺序。"""
        summaries = {
            "c1": _summary("c1", "g1", [((7,), (1,), "a" * 32), ((8,), (1,), "b" * 32)]),
            "c2": _summary("c2", "g2", [((8,), (1,), "b" * 32), ((7,), (1,), "a" * 32)]),
            # 其他 API 的样本同 seed 同长度：被 api_name 隔离，不参与 c1/c2 比对
            "c3": _summary("c3", "g3", [((7,), (1,), "x" * 32), ((8,), (1,), "y" * 32)], api="other_op"),
        }
        ok, failures, groups, failed_cases = compare_batch_consistency(summaries)
        assert ok is True
        assert groups == 2
        assert failed_cases == set()

    def test_inconsistent_key_fails(self):
        summaries = {
            "c1": _summary("c1", "g1", [((7,), (1,), "a" * 32), ((8,), (1,), "b" * 32)]),
            "c2": _summary("c2", "g1", [((7,), (1,), "a" * 32), ((8,), (1,), "c" * 32)]),
        }
        ok, failures, _, failed_cases = compare_batch_consistency(summaries)
        assert ok is False
        assert any("seed=(8,)" in key for key in failures)
        assert failed_cases == {"c1", "c2"}

    def test_single_member_key_skipped(self):
        """单成员 key 组跳过；组内成员数不对齐不影响其他组判定。"""
        summaries = {
            "c1": _summary("c1", "g1", [((7,), (1,), "a" * 32), ((8,), (1,), "b" * 32)]),
            "c2": _summary("c2", "g1", [((7,), (1,), "a" * 32)]),
        }
        ok, failures, groups, failed_cases = compare_batch_consistency(summaries)
        assert ok is True
        assert failures == {}
        assert groups == 1
        assert failed_cases == set()

    def test_unsupported_skipped(self):
        summaries = {
            "c1": _summary("c1", "g1", None, reason="UNSLICEABLE"),
            "c2": _summary("c2", "g1", [((7,), (1,), "a" * 32)]),
        }
        ok, failures, groups, _ = compare_batch_consistency(summaries)
        assert ok is True
        assert groups == 0

    def test_single_member_group_skipped(self):
        summaries = {"c1": _summary("c1", "g1", [((7,), (1,), "a" * 32)])}
        ok, failures, groups, failed_cases = compare_batch_consistency(summaries)
        assert ok is True
        assert failures == {}
        assert groups == 0
        assert failed_cases == set()
