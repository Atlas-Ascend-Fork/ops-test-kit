# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""config loader 测试：yaml 解析。"""

import pytest

from ttk.config.loader import load_config
from ttk.remote.config import RemoteConfig


def _parse_from_yaml(yaml_path):
    cfg = load_config(yaml_path)
    remote = cfg.get("remote")
    return RemoteConfig.from_dict(remote) if remote else None


# -- yaml 解析 --------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_text", "check"),
    [
        # 正常解析 2 endpoint + hardware 字段
        (
            "remote:\n  endpoints:\n"
            "    - host: 10.0.0.100\n      port: 9091\n"
            "    - host: 10.0.0.101\n      port: 9090\n      hardware: gpu\n",
            lambda c: len(c.endpoints) == 2 and c.endpoints[1].hardware == "gpu",
        ),
        # 空 endpoints
        ("remote:\n  endpoints: []\n", lambda c: c.endpoints == []),
    ],
    ids=["two-endpoints", "empty-endpoints"],
)
def test_yaml_parsing(tmp_path, yaml_text, check):
    """yaml 解析：多 endpoint + hardware 字段 / 空 endpoints。"""
    yaml_file = tmp_path / "ttk.conf.yaml"
    yaml_file.write_text(yaml_text)
    config = _parse_from_yaml(str(yaml_file))
    assert config is not None
    assert check(config)
