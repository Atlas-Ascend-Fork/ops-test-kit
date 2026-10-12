#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
DRV Interface
"""

# Standard Packages
import ctypes
import logging
from typing import Optional, Tuple, Union

# Third-Party Packages
from .....utilities import Singleton, get_loaded_so_path, longest_match
from .info import (
    DSMI_ECC_DEVICE_TYPE,
    DSMI_ERROR_CODE,
    DSMI_FREQ_DEVICE_TYPE,
    DSMI_HEALTH_STATE,
    DSMI_UTIL_DEVICE_TYPE,
    DsmiFreqSupportedType,
    DsmiUtilSupportedType,
)
from .structures import (
    dsmi_aicpu_info_stru,
    dsmi_chip_info_stru,
    dsmi_ecc_info_stru,
    dsmi_hbm_info_stru,
    dsmi_memory_info_stru,
)


class DSMIInterface(metaclass=Singleton):
    """
    DRV Function Wrappers
    """

    dsmi_util_cache_short_soc: dict = {}
    dsmi_freq_cache_short_soc: dict = {}
    device_temperature_invoke_ok: Optional[bool] = None
    device_utilization_invoke_ok: Optional[bool] = None
    device_frequency_invoke_ok: Optional[bool] = None

    def __init__(self):
        self._dsmidll = None

    @property
    def dsmidll(self):
        if self._dsmidll is None:
            self._dsmidll = ctypes.CDLL("libdrvdsmi_host.so")
        return self._dsmidll

    def print_so_path(self):
        """
        Print a debug message for libdrvdsmi_host.so path
        """
        logging.debug(f"Using libdrvdsmi_host.so from {get_loaded_so_path(self.dsmidll)}")

    def get_device_count(self) -> int:
        device_count = (ctypes.c_int * 1)()
        self.dsmidll.dsmi_get_device_count.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_count(device_count)
        if self._parse_error(error_code, "dsmi_get_device_count"):
            return 0
        return device_count[0]

    def list_logical_device_id(self) -> Tuple[int, ...]:
        device_count = self.get_device_count()
        device_ids = (ctypes.c_int * device_count)()
        self.dsmidll.dsmi_list_device.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_list_device(device_ids, ctypes.c_int(device_count))
        if self._parse_error(error_code, "dsmi_list_device"):
            return ()
        return tuple(device_ids)

    def get_physical_id_from_logical_id(self, logical_id: int) -> int:
        device_logicid = ctypes.c_int(logical_id)
        device_phyid = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_phyid_from_logicid.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_phyid_from_logicid(device_logicid, device_phyid)
        if self._parse_error(error_code, "dsmi_get_phyid_from_logicid"):
            return -1
        return device_phyid[0]

    def get_device_health_state(self, device_id: int) -> Union[None, DSMI_HEALTH_STATE, int]:
        device_id = ctypes.c_int(device_id)
        device_phealth = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_health.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_health(device_id, device_phealth)
        if self._parse_error(error_code, "dsmi_get_device_health"):
            return None
        try:
            return DSMI_HEALTH_STATE(device_phealth[0])
        except ValueError:
            return device_phealth[0]

    def get_device_health_states(self, device_ids) -> Optional[dict]:
        """Batch health query. Returns {device_id: DSMI_HEALTH_STATE | int | None}
        (per-device value None = that device's query failed); returns None when the
        query capability itself is missing (driver / libdrvdsmi_host.so unavailable)."""
        try:
            return {d: self.get_device_health_state(d) for d in device_ids}
        except Exception as exc:
            logging.warning(f"DSMI health query unavailable: {exc}")
            return None

    def get_device_error(self, device_id: int) -> Optional[Tuple[int, Tuple[int, ...]]]:
        device_id = ctypes.c_int(device_id)
        device_errorcount = (ctypes.c_uint * 1)()
        device_perrorcode = (ctypes.c_uint * 128)()
        self.dsmidll.dsmi_get_device_errorcode.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_errorcode(device_id, device_errorcount, device_perrorcode)
        if self._parse_error(error_code, "dsmi_get_device_errorcode"):
            return None
        return device_errorcount[0], tuple(device_perrorcode)

    def get_device_error_description(self, device_id: int, error_code: int) -> Optional[bytes]:
        device_id = ctypes.c_int(device_id)
        device_errorcode = ctypes.c_uint(error_code)
        device_perrorinfo = (ctypes.c_char * 256)()
        self.dsmidll.dsmi_query_errorstring.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_query_errorstring(
            device_id, device_errorcode, device_perrorinfo, ctypes.c_int(256)
        )
        if self._parse_error(error_code, "dsmi_query_errorstring"):
            return None
        return bytes(device_perrorinfo)

    def get_chip_info(self, device_id: int) -> Optional[dsmi_chip_info_stru]:
        device_id = ctypes.c_int(device_id)
        result_struct = dsmi_chip_info_stru()
        self.dsmidll.dsmi_get_chip_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_chip_info(device_id, ctypes.c_void_p(ctypes.addressof(result_struct)))
        if self._parse_error(error_code, "dsmi_get_chip_info"):
            return None
        return result_struct

    def get_aicpu_info(self, device_id: int, soc: str) -> Optional[dsmi_aicpu_info_stru]:
        if soc == "Ascend310":
            return None
        device_id = ctypes.c_int(device_id)
        result_struct = dsmi_aicpu_info_stru()
        self.dsmidll.dsmi_get_aicpu_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_aicpu_info(device_id, ctypes.c_void_p(ctypes.addressof(result_struct)))
        if self._parse_error(error_code, "dsmi_get_aicpu_info"):
            return None
        return result_struct

    def get_memory_info(self, device_id: int) -> Optional[dsmi_memory_info_stru]:
        device_id = ctypes.c_int(device_id)
        result_struct = dsmi_memory_info_stru()
        self.dsmidll.dsmi_get_memory_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_memory_info(device_id, ctypes.c_void_p(ctypes.addressof(result_struct)))
        if self._parse_error(error_code, "dsmi_get_memory_info"):
            return None
        return result_struct

    def get_hbm_info(self, device_id: int) -> Optional[dsmi_hbm_info_stru]:
        device_id = ctypes.c_int(device_id)
        result_struct = dsmi_hbm_info_stru()
        self.dsmidll.dsmi_get_hbm_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_hbm_info(device_id, ctypes.c_void_p(ctypes.addressof(result_struct)))
        if self._parse_error(error_code, "dsmi_get_hbm_info"):
            return None
        return result_struct

    def get_ecc_info(
        self, device_id: int, device_type: Union[int, DSMI_ECC_DEVICE_TYPE]
    ) -> Optional[dsmi_ecc_info_stru]:
        if isinstance(device_type, int):
            device_type = DSMI_ECC_DEVICE_TYPE(device_type)
        device_id = ctypes.c_int(device_id)
        result_struct = dsmi_ecc_info_stru()
        self.dsmidll.dsmi_get_ecc_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_ecc_info(
            device_id, device_type.value, ctypes.c_void_p(ctypes.addressof(result_struct))
        )
        if self._parse_error(error_code, "dsmi_get_ecc_info"):
            return None
        return result_struct

    def get_device_frequency(
        self, device_id: int, device_type: Union[int, DSMI_FREQ_DEVICE_TYPE], soc: Optional[str] = None
    ) -> Optional[int]:
        if not self._supported(soc, self.dsmi_freq_cache_short_soc, DsmiFreqSupportedType, device_type):
            return None
        if self.device_frequency_invoke_ok is not None and not self.device_frequency_invoke_ok:
            return None
        if isinstance(device_type, int):
            device_type = DSMI_FREQ_DEVICE_TYPE(device_type)
        device_id = ctypes.c_int(device_id)
        pfrequency = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_frequency.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_frequency(device_id, device_type.value, pfrequency)
        if error_code == DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value:
            self.device_frequency_invoke_ok = False
            return None
        if self._parse_error(error_code, "dsmi_get_device_frequency"):
            self.device_frequency_invoke_ok = False
            return None
        return pfrequency[0]

    def get_device_temperature(self, device_id: int) -> Optional[int]:
        if self.device_temperature_invoke_ok is not None and not self.device_temperature_invoke_ok:
            return None
        device_id = ctypes.c_int(device_id)
        ptemperature = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_temperature.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_temperature(device_id, ptemperature)
        if error_code == DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value:
            logging.debug("DSMI API [dsmi_get_device_temperature] is not supported")
            self.device_temperature_invoke_ok = False
            return None
        if self._parse_error(error_code, "dsmi_get_device_temperature"):
            self.device_temperature_invoke_ok = False
            return None
        return ptemperature[0]

    def get_device_util(
        self, device_id: int, device_type: Union[int, DSMI_UTIL_DEVICE_TYPE], soc: Optional[str] = None
    ) -> Optional[int]:
        if not self._supported(soc, self.dsmi_util_cache_short_soc, DsmiUtilSupportedType, device_type):
            return None
        if self.device_utilization_invoke_ok is not None and not self.device_utilization_invoke_ok:
            return None
        if isinstance(device_type, int):
            device_type = DSMI_UTIL_DEVICE_TYPE(device_type)
        device_id = ctypes.c_int(device_id)
        putil = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_utilization_rate.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_utilization_rate(device_id, device_type.value, putil)
        if error_code == DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value:
            self.device_utilization_invoke_ok = False
            return None
        if self._parse_error(error_code, "dsmi_get_device_utilization_rate"):
            self.device_utilization_invoke_ok = False
            return None
        return putil[0]

    @staticmethod
    def _supported(soc, detected_soc_map, soc_support_map: dict, device_type):
        if not soc:
            return True
        if soc == "ERR":
            return False
        if soc in detected_soc_map:
            short_soc = detected_soc_map[soc]
        else:
            short_soc = longest_match(soc, list(soc_support_map.keys()))
            detected_soc_map[soc] = short_soc
        if not short_soc:
            return False
        return int(device_type) in soc_support_map[short_soc]

    @staticmethod
    def _parse_error(error_code: int, function_name: str) -> bool:
        if error_code != 0:
            try:
                logging.error(f"DSMI API Call {function_name} failed: {DSMI_ERROR_CODE(error_code).name}")
                return True
            except ValueError:
                pass
            logging.error(f"DSMI API Call {function_name} failed with unknown code: {error_code}")
            return True
        logging.debug(f"DSMI API Call {function_name}() Success")
        return False
