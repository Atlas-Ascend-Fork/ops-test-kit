# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.common.device.platform: OPP path resolution,
platform-config ini parsing & L0 tile math."""

import pytest

from ttk.core_modules.common.device import platform


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


class _OppCacheTestBase:
    """共享缓存清理：测试用 tmp 环境改写 ASCEND_OPP_PATH，须防 lru_cache 污染
    同会话其他测试。TestGetL0ClearTileParams 不继承（唯一 soc 名 + mock，无需清理）。"""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        for fn in (
            platform.get_opp_paths,
            platform.get_op_impl_paths,
            platform.get_impl_base_paths,
            platform.get_op_info_paths,
            platform.get_ascend_scene_info,
            platform.get_ascend_lib64_path,
            platform.get_npu_hw_info,
        ):
            fn.cache_clear()
        yield
        for fn in (
            platform.get_opp_paths,
            platform.get_op_impl_paths,
            platform.get_impl_base_paths,
            platform.get_op_info_paths,
            platform.get_ascend_scene_info,
            platform.get_ascend_lib64_path,
            platform.get_npu_hw_info,
        ):
            fn.cache_clear()


def _setup_opp(tmp_path, ini_map=None):
    """构造 {tmp}/opp（设为 ASCEND_OPP_PATH）+ scene.info + {tmp}/{arch}-{os}/
    data/platform_config/ 下的 ini 族；返回 opp 根。"""
    opp = _mk(tmp_path / "opp")
    (opp / "scene.info").write_text("os=linux\narch=x86_64\n")
    cfg_dir = _mk(tmp_path / "x86_64-linux" / "data" / "platform_config")
    for soc, text in (ini_map or {}).items():
        (cfg_dir / f"{soc}.ini").write_text(text)
    return opp


class TestGetOppPaths(_OppCacheTestBase):
    def test_builtin_requires_env(self, monkeypatch):
        monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
        with pytest.raises(RuntimeError, match="ASCEND_OPP_PATH"):
            platform.get_opp_paths("builtin")

    def test_builtin_returns_env_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path))
        assert platform.get_opp_paths("builtin") == [str(tmp_path)]

    def test_vendor_config_ini_priority(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path))
        _mk(tmp_path / "vendors" / "v1")
        _mk(tmp_path / "vendors" / "v2")
        (tmp_path / "vendors" / "config.ini").write_text("load_priority=v1,missing,v2\n")
        assert platform.get_opp_paths("vendor") == [str(tmp_path / "vendors" / "v1"), str(tmp_path / "vendors" / "v2")]

    def test_vendor_no_config_falls_back_to_custom_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path))
        _mk(tmp_path / "op_impl" / "custom")
        assert platform.get_opp_paths("vendor") == [str(tmp_path / "op_impl" / "custom")]

    def test_vendor_no_config_no_custom_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path))
        assert platform.get_opp_paths("vendor") == []

    def test_custom_filters_missing_and_empty(self, tmp_path, monkeypatch):
        a = _mk(tmp_path / "a")
        b = _mk(tmp_path / "b")
        monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"{a}:{tmp_path / 'missing'}:{b}::")
        assert platform.get_opp_paths("custom") == [str(a), str(b)]

    def test_custom_requires_env(self, monkeypatch):
        monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)
        assert platform.get_opp_paths("custom") == []

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            platform.get_opp_paths("bogus")


class TestGetOpImplPaths(_OppCacheTestBase):
    def test_builtin_new_layout_with_vendors_dir(self, tmp_path, monkeypatch):
        opp = _setup_opp(tmp_path)
        _mk(opp / "vendors")
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        assert platform.get_op_impl_paths("builtin") == [str(opp / "built-in" / "op_impl")]

    def test_builtin_old_layout_without_vendors_dir(self, tmp_path, monkeypatch):
        opp = _setup_opp(tmp_path)
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        assert platform.get_op_impl_paths("builtin") == [str(opp / "op_impl" / "built-in")]


class TestGetAscendSceneInfo(_OppCacheTestBase):
    def test_parses_os_and_arch(self, tmp_path, monkeypatch):
        opp = _setup_opp(tmp_path)
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        assert platform.get_ascend_scene_info() == ("linux", "x86_64")

    def test_missing_scene_file_returns_empty(self, tmp_path, monkeypatch):
        opp = _mk(tmp_path / "opp")
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        assert platform.get_ascend_scene_info() == ("", "")


_FULL_INI = """\
[version]
Short_SoC_version = Ascend910B
CCEC_AIC_version = ascend910b
NpuArch = 910b

[SoCInfo]
ai_core_cnt = 20
cube_core_cnt = 10
vector_core_cnt = 40
cube_vector_combine = split
core_type_list = VectorCube
support_bf16 = 1
support_inf_nan = 0

[AICoreSpec]
l0_a_size = 65536
l0_b_size = 65536
l0_c_size = 262144
"""

_REQUIRED_ONLY_INI = """\
[version]
Short_SoC_version = Ascend910B
CCEC_AIC_version = ascend910b
NpuArch = 910b

[SoCInfo]
ai_core_cnt = 20
cube_vector_combine = combine
core_type_list = AiCore
"""


class TestGetNpuHwInfo(_OppCacheTestBase):
    def test_full_parse_types_and_values(self, tmp_path, monkeypatch):
        opp = _setup_opp(tmp_path, {"Ascend910B": _FULL_INI})
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        info = platform.get_npu_hw_info("Ascend910B")
        assert info["short_soc_version"] == "Ascend910B"
        assert info["npu_arch"] == "910b"
        assert info["ai_core_cnt"] == 20
        assert info["l0_c_size"] == 262144
        assert info["support_bf16"] is True
        assert info["support_inf_nan"] is False
        assert info["cv_split"] is True
        assert info["full_soc_version"] == "Ascend910B"

    def test_ccec_cube_suffix_rewritten(self, tmp_path, monkeypatch):
        ini = _FULL_INI.replace("CCEC_AIC_version = ascend910b", "CCEC_AIC_version = ascend910b_cube")
        opp = _setup_opp(tmp_path, {"Ascend910B": ini})
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        assert platform.get_npu_hw_info("Ascend910B")["ccec_aic_version"] == "ascend910b_$core_type"

    def test_optional_fields_defaulted_when_missing(self, tmp_path, monkeypatch):
        opp = _setup_opp(tmp_path, {"Ascend910B": _REQUIRED_ONLY_INI})
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        info = platform.get_npu_hw_info("Ascend910B")
        assert info["cube_core_cnt"] == 0
        assert info["vector_core_cnt"] == 0
        assert info["l0_a_size"] == 0
        assert info["support_bf16"] is False
        assert info["cv_split"] is False

    def test_missing_required_field_raises(self, tmp_path, monkeypatch):
        ini = _FULL_INI.replace("NpuArch = 910b\n", "")
        opp = _setup_opp(tmp_path, {"Ascend910B": ini})
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        with pytest.raises(RuntimeError, match=r"\[Ascend910B\] missing fields: \['NpuArch'\]"):
            platform.get_npu_hw_info("Ascend910B")

    def test_ini_not_found_raises(self, tmp_path, monkeypatch):
        opp = _setup_opp(tmp_path)
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        with pytest.raises(FileNotFoundError, match="Platform config not found"):
            platform.get_npu_hw_info("Ascend910B")

    def test_ascend310b_forces_cv_split_off(self, tmp_path, monkeypatch):
        ini = _FULL_INI.replace("Short_SoC_version = Ascend910B", "Short_SoC_version = Ascend310B")
        opp = _setup_opp(tmp_path, {"Ascend310B1": ini})
        monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
        assert platform.get_npu_hw_info("Ascend310B1")["cv_split"] is False


def _hw_info(l0a, l0b, l0c, npu_arch="3510"):
    return {
        "l0_a_size": l0a,
        "l0_b_size": l0b,
        "l0_c_size": l0c,
        "npu_arch": npu_arch,
        "full_soc_version": "Test",
    }


class TestGetL0ClearTileParams:
    def test_perfect_square_hardware_sizes(self, monkeypatch):
        monkeypatch.setattr(platform, "get_npu_hw_info", lambda soc: _hw_info(65536, 65536, 65536))
        m, k, n, arch = platform.get_l0_clear_tile_params("TestSocP2")
        assert (m, k, n, arch) == (128, 256, 128, "dav-3510")
        assert m * n * 4 <= 65536

    def test_non_square_sizes_respect_l0c_bound(self, monkeypatch):
        l0a, l0b, l0c = 4096, 2048, 8000
        monkeypatch.setattr(platform, "get_npu_hw_info", lambda soc: _hw_info(l0a, l0b, l0c))
        m, k, n, _ = platform.get_l0_clear_tile_params("TestSocOdd")
        assert l0a // 2 % k == 0
        assert l0b // 2 % k == 0
        assert k % 16 == 0
        assert m * k * 2 == l0a
        assert n * k * 2 == l0b
        assert m * n * 4 <= l0c

    def test_zero_l0_raises(self, monkeypatch):
        monkeypatch.setattr(platform, "get_npu_hw_info", lambda soc: _hw_info(65536, 65536, 0))
        with pytest.raises(RuntimeError, match="L0 sizes are zero"):
            platform.get_l0_clear_tile_params("TestSocZero")
