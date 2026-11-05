# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""读文件原语：按后缀格式读单个数组文件（DataDumper 的读侧对等）。

读写都经 formatter 注册制：DataDumper 写走 DumpFormatter.dump，
DataLoader 读走 DumpFormatter.load（每格式一实现；print 不支持读）。
"""

import logging
import os
from typing import Optional, Tuple

from .formatter import (
    DumpLoadError,
    ManualDataError,
    get_formatter,
)


class DataLoadError(ValueError):
    """manual xpu 三方装载失败（shape 与声明不符等）。"""


class DataLoader:
    @staticmethod
    def load_xpu(files, dtypes, shapes) -> list:
        """xpu 槽批量读（与 dump_xpu 对称）。"""
        return DataLoader._load_files(files, dtypes, shapes)

    @staticmethod
    def load_inputs(files, dtypes, shapes) -> list:
        """input/scalar 槽批量读（与 dump_inputs 对称）。"""
        return DataLoader._load_files(files, dtypes, shapes)

    @staticmethod
    def load_goldens(files, dtypes, shapes) -> list:
        """golden 槽批量读（与 dump_goldens 对称）。"""
        return DataLoader._load_files(files, dtypes, shapes)

    @staticmethod
    def _load_files(files, dtypes, shapes) -> list:
        """批量逐槽读：文件 None 槽补 None，其余 read_array。

        读后复核（dtype/shape）与 bin 预检在各域上层按其口径完成。
        """
        values = []
        for idx, fp in enumerate(files):
            if fp is None:
                values.append(None)
                continue
            shape = shapes[idx] if idx < len(shapes) else None
            logging.info("Reading file: %s ...", fp)
            try:
                values.append(DataLoader.read_array(fp, dtypes[idx] if idx < len(dtypes) else None, shape))
            except DumpLoadError as exc:
                raise ManualDataError(str(exc)) from None
        return values

    @staticmethod
    def read_array(path, dtype: Optional[str] = None, shape: Optional[Tuple[int, ...]] = None):
        ext = os.path.splitext(str(path))[1].lstrip(".")
        return get_formatter(ext).load(path, dtype, shape)
