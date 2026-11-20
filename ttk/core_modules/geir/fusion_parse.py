#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir/fusion_parse.py — fusion_result.json 采集（融合 pass 命中/生效计数 → 结果列）。

GE 编译管线在 ``ge.exportCompileStat`` 非 NONE 时落盘（fusion_statistic_writer.cc），
路径取 serve 进程 CWD 回退；仅采集 graph_fusion 组，ub_fusion 不进结果列。

schema：
  {"session_and_graph_id_{N}_{M}":                  会话层（per-AddGraph 追加）
     {"graph_fusion": {"PassName": {"match_times": "1", "effect_times": "1"}}}}

- ``parse_fusion_sessions(path)``：读文件 → {session_key: {pass: (match, effect)}}；
  缺失/不可解析返回 None（列置 NA，不影响结果行）
- ``format_pass_stats(passes)``：单会话计数 → ``"PassA:hit=N,effect=M;..."``；
  计数兼容 int/整数字符串编码，键缺失保持 None
"""

__all__ = ["FUSION_STAT_FILENAME", "parse_fusion_sessions", "format_pass_stats"]

import json
import logging
import os
from typing import Dict, Optional, Tuple

FUSION_STAT_FILENAME = "fusion_result.json"

_FUSION_GROUP = "graph_fusion"
_MATCH_KEY = "match_times"
_EFFECT_KEY = "effect_times"


def parse_fusion_sessions(path: str) -> Optional[Dict[str, Dict[str, Tuple[Optional[int], Optional[int]]]]]:
    """读 fusion_result.json → {session_key: {pass_name: (match, effect)}}。

    文件缺失/非 dict 根节点返回 None；schema 未知形态按会话解析失败跳过（debug 记录）。
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logging.debug("[GEIR] fusion result unreadable (%s): %s", path, exc)
        return None
    if not isinstance(data, dict):
        logging.warning("[GEIR] fusion result schema unparsable (%s): root is %s", path, type(data).__name__)
        return None

    sessions: Dict[str, Dict[str, Tuple[Optional[int], Optional[int]]]] = {}
    for session_key, session_val in data.items():
        if not isinstance(session_val, dict):
            logging.debug("[GEIR] fusion session %r is not a dict, skipped", session_key)
            continue
        passes: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        group_val = session_val.get(_FUSION_GROUP)
        if isinstance(group_val, dict):
            for pass_name, pass_val in group_val.items():
                if not isinstance(pass_val, dict):
                    continue
                passes[pass_name] = (_to_int(pass_val.get(_MATCH_KEY)), _to_int(pass_val.get(_EFFECT_KEY)))
        sessions[session_key] = passes
    return sessions


def format_pass_stats(passes: Dict[str, Tuple[Optional[int], Optional[int]]]) -> Optional[str]:
    """单会话计数 → "PassA:hit=N,effect=M;..."；空会话返回 None。"""
    if not passes:
        return None
    return ";".join(f"{name}:hit={hit},effect={eff}" for name, (hit, eff) in passes.items())


def _to_int(val):
    """计数宽容转换：int/整数字符串 → int；bool/浮点/其余 → None。"""
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return None
    return None
