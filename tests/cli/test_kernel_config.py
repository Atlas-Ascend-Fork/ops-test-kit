# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""config loader + args_to_switches 测试：yaml endpoint 解析、--provider 映射、
config=None 时无条件加载（lazy fallback 已删）。"""

import types
from unittest.mock import MagicMock


def _make_args(**kwargs):
    """构造 args mock，只填 args_to_switches 实际读取的字段。"""
    args = MagicMock()
    args.provider = kwargs.get("provider")
    args.config = kwargs.get("config")
    args.input = kwargs.get("input", "x.csv")
    args.output = kwargs.get("output")
    args.run = kwargs.get("run", 1)
    return args


def test_yaml_endpoints_parsed_into_remote_config(tmp_path, monkeypatch):
    """yaml 中 endpoint 的 host/port/providers 字段被 loader 正确解析进 RemoteConfig。"""
    user_config = tmp_path / "ttk.conf.yaml"
    user_config.write_text("""
remote:
  endpoints:
    - host: "10.0.0.1"
      port: 9090
      providers: ["tf"]
""")
    monkeypatch.chdir(tmp_path)

    from ttk.config import loader

    loader._config = None

    from ttk.config.loader import get_remote_config, load_config

    load_config()

    config = get_remote_config()
    assert config is not None
    assert len(config.endpoints) == 1
    assert config.endpoints[0].host == "10.0.0.1"
    assert config.endpoints[0].port == 9090
    assert config.endpoints[0].providers == ["tf"]


def test_yaml_without_endpoints_returns_none(tmp_path, monkeypatch):
    """yaml 无 endpoints 时 get_remote_config() 返回 None（无远端配置的默认语义）。"""
    user_config = tmp_path / "ttk.conf.yaml"
    user_config.write_text("""
remote:
  backoff_base_s: 1.0
""")
    monkeypatch.chdir(tmp_path)

    from ttk.config import loader

    loader._config = None

    from ttk.config.loader import get_remote_config, load_config

    load_config()

    assert get_remote_config() is None


def test_provider_cli_lands_on_provider_filter(tmp_path, monkeypatch):
    """--provider 落到 sw.provider_filter（worker 侧测试过滤器），不修改 RemoteConfig。"""
    user_config = tmp_path / "ttk.conf.yaml"
    user_config.write_text("""
remote:
  backoff_base_s: 1.0
""")
    monkeypatch.chdir(tmp_path)

    from ttk.config import loader

    loader._config = None

    args = _make_args(provider="torch")

    from ttk.cli.bridge import args_to_switches

    sw = args_to_switches(args)

    assert sw.provider_filter == "torch"


def test_args_to_switches_loads_config_unconditionally():
    """config=None 时 args_to_switches 仍无条件调 load_config（lazy fallback 已删的回归看护）。"""
    from ttk.cli.bridge import args_to_switches
    from ttk.config import loader

    saved = loader._config
    loader._config = None
    try:
        args = types.SimpleNamespace(config=None, provider=None, input="x.csv", output=None)
        sw = args_to_switches(args)
        assert loader._config is not None
        assert sw.config_path is None
        assert sw.provider_filter is None
    finally:
        loader._config = saved
