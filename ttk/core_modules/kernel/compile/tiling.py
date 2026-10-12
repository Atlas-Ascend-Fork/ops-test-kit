#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""tiling 解析。"""

import logging
from typing import Any, Optional, Tuple

import numpy


def parse_tiling_data(tiling_data: Any) -> Tuple[Optional[bytes], Optional[tuple]]:
    """
    Parse tiling data
    :param tiling_data:
    :return:
    """
    # little-endian int32 Tiling data
    le_int32 = numpy.dtype(numpy.int32)
    le_int32 = le_int32.newbyteorder("<")
    if isinstance(tiling_data, tuple):
        tiling_data_np_array = numpy.array(tiling_data, dtype=le_int32)
        tiling_data_bytes = tiling_data_np_array.tobytes()
        tuple_tiling_data = tuple(tiling_data_np_array)
    elif isinstance(tiling_data, bytes):
        tiling_data_bytes = tiling_data
        if len(tiling_data) % 4 == 0:
            tiling_data_np_array = numpy.frombuffer(tiling_data, dtype=le_int32)
            tuple_tiling_data = tuple(tiling_data_np_array)
        else:
            tuple_tiling_data = ("TILING_NOT_4BYTE_ALIGNED",)
    elif tiling_data is None:
        return None, None
    else:
        raise TypeError("Tiling data parsing error, received unknown object: " + str(tiling_data))
    return tiling_data_bytes, tuple_tiling_data


def render_tiling_data(tuple_tiling_data: tuple, compile_info: dict, tiling_key: int):
    """Render parsed tiling data into a human-readable string (display only).

    Uses compile_info["_vars"] (tiling_key → variable-name list) to map the
    int32 values to named fields, e.g. (1024, 16) → "{'block_dim': 1024, ...}".
    Feeds profiling logs and result summary tables only — execution paths
    consume the bytes from parse_tiling_data.
    """
    if tuple_tiling_data is None:
        return "Tiling data is None"
    if len(tuple_tiling_data) > 0:
        if "_vars" in compile_info:
            _vars = {int(k): v for k, v in compile_info["_vars"].items()}
            tiling_data_indexes = _vars.get(tiling_key)
            if tiling_data_indexes is not None:
                if len(tiling_data_indexes) == len(tuple_tiling_data):
                    dict_tiling_data = dict(zip(tiling_data_indexes, tuple_tiling_data))
                    dyn_str_tiling_data = str(dict_tiling_data)
                else:
                    even_is_zero = [data == 0 for idx, data in enumerate(tuple_tiling_data) if idx % 2 == 1]
                    if len(tiling_data_indexes) * 2 == len(tuple_tiling_data) and all(even_is_zero):
                        odd_data = [data for idx, data in enumerate(tuple_tiling_data) if idx % 2 == 0]
                        dict_tiling_data = dict(zip(tiling_data_indexes, odd_data))
                        dyn_str_tiling_data = str(dict_tiling_data)
                    else:
                        dyn_str_tiling_data = f"Tiling data not match with compile_info _vars: {tuple_tiling_data}"
                        logging.warning(
                            f"Tiling data not match with sub-keys in compile_info _vars: "
                            f"{tuple_tiling_data} VS {tiling_data_indexes}"
                        )
            else:
                dyn_str_tiling_data = f"Tiling_key {str(tiling_key)} not found in _vars key {str(tuple(_vars.keys()))}, treat as Tik operator"
        else:
            dyn_str_tiling_data = f"{tuple_tiling_data}"
    else:
        dyn_str_tiling_data = "Tiling data is empty"
    return dyn_str_tiling_data
