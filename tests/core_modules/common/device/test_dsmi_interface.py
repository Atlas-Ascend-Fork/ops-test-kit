# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.common.device.dsmi: device info enums, supported-type maps & structures."""

import pytest

from ttk.core_modules.common.device.dsmi.info import (
    DSMI_ECC_DEVICE_TYPE,
    DSMI_ERROR_CODE,
    DSMI_HEALTH_STATE,
    DsmiFreqSupportedType,
    DsmiUtilSupportedType,
)
from ttk.core_modules.common.device.dsmi.interface import DSMIInterface
from ttk.core_modules.common.device.dsmi.structures import (
    dsmi_aicpu_info_stru,
    dsmi_chip_info_stru,
    dsmi_ecc_info_stru,
)


def test_error_code_enum():
    assert DSMI_ERROR_CODE.DSMI_ERROR_NONE.value == 0
    assert DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value == 0xFFFE


def test_health_state_enum():
    assert DSMI_HEALTH_STATE.OK.value == 0
    assert DSMI_HEALTH_STATE.OFF.value == 0xFFFFFFFF


def test_ecc_device_type_enum():
    assert DSMI_ECC_DEVICE_TYPE.HBM.value == 2
    assert DSMI_ECC_DEVICE_TYPE.NONE.value == 0xFF


@pytest.mark.parametrize("chip", ["Ascend910", "Ascend910B", "Ascend950"])
def test_freq_supported_type_nonempty(chip):
    assert len(DsmiFreqSupportedType[chip]) > 0


@pytest.mark.parametrize("chip", ["Ascend910", "Ascend910B", "Ascend950"])
def test_util_supported_type_nonempty(chip):
    assert len(DsmiUtilSupportedType[chip]) > 0


def test_freq_supported_type_specific_chip():
    assert DsmiFreqSupportedType["Ascend910"] == (1, 2, 6, 7, 9)
    assert DsmiFreqSupportedType["Ascend910B"] == (2, 6, 7, 9)


def test_chip_info_complete_platform():
    info = dsmi_chip_info_stru()
    info.chip_type = b"Ascend"
    info.chip_name = b"910B"
    info.chip_ver = b"v1"
    assert info.get_complete_platform() == "Ascend910B"
    assert info.get_ver() == "v1"


def test_aicpu_avg_util_rate():
    info = dsmi_aicpu_info_stru()
    info.aicpuNum = 16
    for i in range(16):
        info.utilRate[i] = 50
    assert info.get_avg_util_rate() == 50.0


def test_ecc_info_enabled_flags():
    info = dsmi_ecc_info_stru()
    info.enable_flag = 0
    assert info.get_enabled() is False
    info.enable_flag = 1
    assert info.get_enabled() is True
    info.enable_flag = 2
    assert info.get_enabled() is None


def test_ecc_info_error_counts():
    info = dsmi_ecc_info_stru()
    info.single_bit_error_count = 3
    info.double_bit_error_count = 5
    assert info.get_single_bit_error_count() == 3
    assert info.get_double_bit_error_count() == 5


class TestGetDeviceHealthStates:
    def test_batch_maps_per_device(self):
        dsmi = DSMIInterface()

        class _FakeHealthFunc:
            restype = None

            def __call__(self, dev_id, out):
                out[0] = 0 if dev_id.value == 0 else 3
                return 0

        class _FakeDll:
            dsmi_get_device_health = _FakeHealthFunc()

        dsmi._dsmidll = _FakeDll()
        try:
            states = dsmi.get_device_health_states([0, 1])
            assert states == {0: DSMI_HEALTH_STATE.OK, 1: DSMI_HEALTH_STATE.CRITICAL}
        finally:
            dsmi._dsmidll = None

    def test_capability_missing_returns_none(self, caplog):
        dsmi = DSMIInterface()
        dsmi._dsmidll = 1
        try:
            assert dsmi.get_device_health_states([0]) is None
            assert any("DSMI health query unavailable" in r.message for r in caplog.records)
        finally:
            dsmi._dsmidll = None
