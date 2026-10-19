#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""DataDumper：worker 阶段产物的统一落盘器。

只管"写文件"，不管"什么时候该写"（时序由编排层定）。通路只传数据源 +
预处理列表（Kernel 三模式展开/tiling 两条目、ACLNN pure_output 滤除、
E2E 多模式分调），本类做"按配置判断 + 遍历 + 落盘"。

落盘目录与命名：
    目录：NPU_DUMP_PATH env > {root}/dump/{case}/（与 store 目录同构）；
          惰性创建——首次实际写文件才 makedirs，开关全关不产生空目录
    文件：{role}[_{segment}]_{idx}_{dtype}.{ext}
      role    ∈ input / output / golden / xpu / tiling_data
      segment 可选（Kernel: dyn/cst/bin——input 仅 dyn；E2E: eager/graph_cst/
               graph_dyn/graph_aclgraph；GEIR: const/dynamic；ACLNN: 无）
      xpu     segment 位 = provider（xpu_torch_0_float16.bin）
      dtype   numpy dtype.name；None 占位 "none"（store 约定）
      tiling_data 等无 dtype 特殊 role：{role}_{segment}.{ext}
    idx 与 CSV 槽位对齐：None/str 占位槽位跳过不落盘、不移位后续索引
"""

__all__ = ["DataDumper"]

# Standard Packages
import logging
import os
import re
from typing import Optional, Sequence, Union

from ....utilities.dtypes import dtype_to_name
from .config import DumpConfig
from .formatter import get_formatter
from .manual_data import _encode_shape, case_directory_name

_NONE_DTYPE = "none"

# xpu 槽位文件名回放口径：provider 含 _<纯数字> 段（如 a_12_b / cuda_0）时，
# 读侧 _XPU_FILE_NAME 的非贪婪切分会把数字段解析为 idx、真 idx+dtype 拼入
# dtype 位。落盘前校验拒绝此类 provider 名。
_AMBIGUOUS_XPU_PROVIDER = re.compile(r"_\d+(?:_|$)")


def _validate_xpu_provider(provider) -> None:
    name = str(provider)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or _AMBIGUOUS_XPU_PROVIDER.search(name):
        raise ValueError(
            f"XPU provider name {name!r} cannot be encoded in dump slot filenames; "
            "rename the provider (alphanumerics with no digit-only underscore segments)"
        )


def _dtype_name(dtype) -> str:
    if dtype is None:
        return _NONE_DTYPE
    return dtype_to_name(dtype)


def _slot_dtype(dtypes, idx, data):
    """槽位 dtype：显式 dtypes 优先（bytes 等无自描述形态），缺省从数据自推。"""
    if idx < len(dtypes) and dtypes[idx] is not None:
        return dtypes[idx]
    return getattr(data, "dtype", None)


def _shape_token(data) -> Optional[str]:
    """数据物理 shape → 文件名 shape 段 token（无 shape 自描述则 None）。

    bytes/占位串无 shape；ndarray/Tensor 按实际维度编码（bin 才追加，npy/pt
    文件自带 shape）。"""
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    return _encode_shape(tuple(int(dimension) for dimension in shape))


class DataDumper:
    """worker 内按 DumpConfig 落盘。一次构造，四方法反复调。"""

    def __init__(self, switches, name_prefix: str):
        """switches: 取 dump_config + root_path；另查 NPU_DUMP_PATH env
        （用户手动 export 的目录覆盖通道，优先于默认目录）。
        name_prefix: dump_file_prefix or testcase_name（调用方算好传入，
        兼作子目录名 = case_directory_name 同款安全化）。
        目录惰性创建：首次实际写文件时才 makedirs，开关全关不产生空目录。"""
        self.cfg: DumpConfig = switches.dump_config
        base = os.getenv("NPU_DUMP_PATH") or os.path.join(switches.root_path, "dump")
        self.dump_dir = os.path.join(base, case_directory_name(name_prefix))
        self._dir_created = False

    def _formatter(self):
        return get_formatter(self.cfg.file_format)

    def _ensure_dir(self):
        if not self._dir_created:
            os.makedirs(self.dump_dir, exist_ok=True)
            self._dir_created = True

    def _write(
        self, role: str, data, idx: Optional[int], dtype, segment: Optional[str], shape_token: Optional[str] = None
    ):
        self._ensure_dir()
        seg = f"_{segment}" if segment else ""
        # tiling_data 等无 dtype/无 idx 的特殊 role：idx=None 直用 {role}{segment}
        name = f"{role}{seg}" if idx is None else f"{role}{seg}_{idx}_{_dtype_name(dtype)}"
        if shape_token is not None:
            name += f"_shape_{shape_token}"
        self._formatter().dump(data, self.dump_dir, name, dtype=dtype)

    # ── 四槽位 ──

    def dump_inputs(
        self, inputs: Sequence, dtypes: Sequence = (), segment: Optional[str] = None, force: bool = False
    ) -> None:
        """force or cfg.dump_input → input[_{segment}]_{idx}_{dtype}.{ext}。

        None/str 槽位（可选输出未落地、SUPPRESSED 等占位串）跳过但保留原 idx
        ——文件名索引与 CSV 槽位对齐，不因占位前移。"""
        if not (force or self.cfg.dump_input):
            return
        logging.info("Dump Input data....")
        for idx, data in enumerate(inputs):
            if data is None or isinstance(data, str):
                continue
            self._write("input", data, idx, _slot_dtype(dtypes, idx, data), segment)

    def dump_outputs(
        self, outputs: Sequence, dtypes: Sequence = (), segment: Optional[str] = None, force: bool = False
    ) -> None:
        """force or cfg.dump_output → output[_{segment}]_{idx}_{dtype}.{ext}
        （None/str 占位跳过保 idx，同 dump_inputs）。"""
        if not (force or self.cfg.dump_output):
            return
        logging.info(f"Dump Output data{' [' + segment + ']' if segment else ''}....")
        for idx, data in enumerate(outputs):
            if data is None or isinstance(data, str):
                continue
            self._write("output", data, idx, _slot_dtype(dtypes, idx, data), segment)

    def dump_goldens(self, goldens: Sequence, dtypes: Sequence = (), force: bool = False) -> None:
        """force or cfg.dump_golden → golden_{idx}_{dtype}.{ext}（无 segment；
        None/str 占位跳过保 idx，同 dump_inputs）。"""
        if not (force or self.cfg.dump_golden):
            return
        logging.info("Dump Golden data....")
        for idx, data in enumerate(goldens):
            if data is None or isinstance(data, str):
                continue
            self._write("golden", data, idx, _slot_dtype(dtypes, idx, data), None)

    def dump_xpu(self, third_parties: dict, force: bool = False) -> None:
        """force or cfg.dump_xpu → xpu_{provider}_{idx}_{dtype}.{ext}
        （third_parties 按 provider 分组遍历，provider 占 segment 位；dtype
        缺省从数据自推——ndarray/Tensor 进文件名，bytes 无自描述则 none；
        bin 格式追加 `_shape_<token>` 段，回放按文件名自描述还原）。"""
        if not (force or self.cfg.dump_xpu):
            return
        logging.info("Dump XPU third-party data....")
        bin_format = self._formatter().ext == "bin"
        for provider, outputs in (third_parties or {}).items():
            _validate_xpu_provider(provider)
            for idx, data in enumerate(outputs):
                if data is None or isinstance(data, str):
                    continue
                shape_token = _shape_token(data) if bin_format else None
                self._write("xpu", data, idx, _slot_dtype((), idx, data), str(provider), shape_token=shape_token)

    def dump_named(self, name: str, data, dtype=None) -> None:
        """特殊条目（tiling_data 等）：按调用方组合名直写 {name}.{ext}，不走开关。"""
        self._ensure_dir()
        self._formatter().dump(data, self.dump_dir, name, dtype=dtype)

    def dump_on_fail(
        self,
        inputs: Sequence,
        outputs: Union[Sequence, dict],
        goldens: Sequence,
        dtypes: Sequence = (),
        segment: Optional[str] = None,
    ) -> None:
        """比对失败补漏：dump_on_fail 开关控制，未开启的槽位 force 落盘
        （只补未开的部分，已开的不重复写；xpu 槽不参与补漏）。

        outputs：单 segment 传列表（segment 同时作用于 input/output）；多
        segment（Kernel 三模式/E2E 多形态各自触发）传 {segment: outputs}——
        此时 segment 仅作用于 input。"""
        if not self.cfg.dump_on_fail:
            return
        if not self.cfg.dump_input:
            self.dump_inputs(inputs, segment=segment, force=True)
        if not self.cfg.dump_output:
            if isinstance(outputs, dict):
                for seg, seg_outputs in outputs.items():
                    self.dump_outputs(seg_outputs, dtypes=dtypes, segment=seg, force=True)
            else:
                self.dump_outputs(outputs, dtypes=dtypes, segment=segment, force=True)
        if not self.cfg.dump_golden:
            self.dump_goldens(goldens, dtypes=dtypes, force=True)
