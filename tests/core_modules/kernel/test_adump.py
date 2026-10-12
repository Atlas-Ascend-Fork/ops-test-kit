# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.adump: dump structures & AdxInterface static logic."""

import ctypes

import pytest

from ttk.core_modules.kernel.runtime.adx_interface import (
    ADX_FORMATTERS,
    ADX_TENSOR_POSITIONS,
    AdxInterface,
)
from ttk.core_modules.kernel.runtime.adx_structure import (
    AdxBlockInfo,
    AdxDumpMeta,
    AscendDumpType,
)


def test_ascend_dump_type_values():
    assert AscendDumpType.DEFAULT.value == 0
    assert AscendDumpType.TENSOR.value == 2
    assert AscendDumpType.SIMT.value == 7


def test_tensor_positions_map():
    assert ADX_TENSOR_POSITIONS[0] == "GM"
    assert ADX_TENSOR_POSITIONS[2] == "L1"
    assert ADX_TENSOR_POSITIONS[7] == "FIXBUF"


def test_formatters_cover_common_formats():
    for spec in ("%d", "%u", "%f", "%x", "%X", "%s", "%p"):
        assert spec in ADX_FORMATTERS


def test_block_info_size_and_data_ptr():
    block = AdxBlockInfo()
    assert AdxBlockInfo.size() == ctypes.sizeof(AdxBlockInfo)
    ptr = block.data_ptr()
    assert ptr.value == ctypes.addressof(block) + AdxBlockInfo.size()


def test_dump_meta_data_ptr_offset():
    meta = AdxDumpMeta()
    base = meta.data_ptr().value
    assert meta.data_ptr_offset(8).value == base + 8


@pytest.mark.parametrize(("platform", "expected"), [("Ascend910", 75), ("Ascend950", 108)])
def test_get_max_dump_core_num(platform, expected):
    assert AdxInterface._get_max_dump_core_num(platform) == expected


@pytest.mark.parametrize(("platform", "expected"), [("Ascend910", 50), ("Ascend950", 72)])
def test_get_aic_idx_offset(platform, expected):
    assert AdxInterface._get_aic_idx_offset(platform) == expected


def test_has_simt_dump_info_false_for_before_david():
    assert AdxInterface._has_simt_dump_info(10 * 1024 * 1024, "Ascend910") is False


def test_has_simt_dump_info_true_for_david_when_large():
    threshold = 108 * AdxInterface.ADX_MAX_STR_LEN
    assert AdxInterface._has_simt_dump_info(threshold + 1, "Ascend950") is True
    assert AdxInterface._has_simt_dump_info(threshold, "Ascend950") is False


def test_match_fmt_pattern_finds_all_specs():
    matches = AdxInterface._match_fmt_pattern("val=%d hex=%x str=%s")
    specs = [m[0] for m in matches]
    assert specs == ["%d", "%x", "%s"]


def test_match_fmt_pattern_empty_when_no_spec():
    assert AdxInterface._match_fmt_pattern("plain text no format") == []


def test_adx_constants():
    assert AdxInterface.ADX_PRINT_MAGIC == 0x5AA5BCCD
    assert AdxInterface.ADX_MAX_STR_LEN == 1024 * 1024
