# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""worker_bootstrap 从 SWITCHES.config_path 加载 yaml 配置的单元测试。"""


def test_worker_bootstrap_loads_config_from_switches(tmp_path, monkeypatch):
    """worker_bootstrap 从 SWITCHES.config_path 加载 yaml（含 TLS）——根因测试。"""
    from ttk.config import loader

    # 桩掉 default_logging_config：它有全局副作用（清 root logger handler +
    # numpy.seterr('ignore') + warnings.filterwarnings），会污染同 session 其它
    # 测试。worker_bootstrap 的核心是 load_config，logging 非本测试关注点。
    from ttk.core_modules.common.process import simple_command as pool
    from ttk.switches import SWITCHES

    monkeypatch.setattr(pool, "default_logging_config", lambda **kw: None)

    yaml_path = tmp_path / "worker-test.yaml"
    yaml_path.write_text('remote:\n  tls_ca: "/fake/ca.crt"\n  endpoints:\n    - host: 10.0.0.1\n      port: 9090\n')
    saved = loader._config
    loader._config = None  # 模拟 forkserver worker：干净的 _config
    try:
        sw = SWITCHES()
        sw.config_path = str(yaml_path)
        sw.logging_to_file = False

        from ttk.core_modules.common.process.simple_command import worker_bootstrap

        worker_bootstrap(sw)

        rc = loader.get_remote_config()
        assert rc is not None
        assert rc.tls_ca == "/fake/ca.crt"
        assert len(rc.endpoints) == 1
    finally:
        loader._config = saved
        loader.load_config()  # 恢复默认 config，别污染后续测试


def test_worker_bootstrap_without_config_path_loads_defaults(monkeypatch):
    """config_path=None 时 worker_bootstrap 走标准路径（不 crash）。"""
    from ttk.config import loader
    from ttk.core_modules.common.process import simple_command as pool
    from ttk.switches import SWITCHES

    monkeypatch.setattr(pool, "default_logging_config", lambda **kw: None)
    saved = loader._config
    loader._config = None
    try:
        sw = SWITCHES()
        sw.config_path = None
        sw.logging_to_file = False
        from ttk.core_modules.common.process.simple_command import worker_bootstrap

        worker_bootstrap(sw)
        cfg = loader.get_config()
        assert "remote" in cfg  # default.yaml 有 remote 段
    finally:
        loader._config = saved
        loader.load_config()
