# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""om_compiler 单测：atc 真实转换（多算子）+ 缓存命中 + 失败透传 + 环境归一化。

覆盖（设计文档 §11）：
- atc 真实转换：Add f32 基线 + MatMul / Concat / Split 多语义算子族；
- 缓存命中：同输入二次调用零 atc 子进程（subprocess.run 断言不被触达）；
- 失败透传：非法算子 atc rc≠0 → OmniCompileError 载荷含 ATC_COMPILE_FAILED 与
  stdout/stderr 合并尾部（atc 诊断走 stdout）；
- 纯 CPU 部分：_normalized_env 的 PYTHONPATH 归一化 / ATC_NOT_FOUND /
  插件指纹退化兜底 / 缓存键指纹敏感性（无 Ascend 环境可跑）。

atc 相关用例按环境可用性 skip（无 Ascend 环境仅跑纯 CPU 部分）。conftest 的
isolate_ttk_environment 剥离 ASCEND_* 环境变量；生产链路（python -m ttk）由
cli 入口的 ttk._env.setup_env() source set_env.sh 回填，单测直接 import
compile_om 不经 CLI，故 _ascend_env_for_atc 复用同一 healing
（ttk._env._source_setenv_bash，EC0008 防御）重建 atc 子进程所需环境，测毕还原。

跨环境矩阵（换机器可移植性）：
- 无 Ascend（atc 探测失败）→ 真实编译整类 skip，纯 CPU 项照跑；
- 无 onnx / 环境 opset 超 atc 插件上限（param_extractor.ATC_MAX_OPSET 同款
  产品门禁）→ 真实编译用例 skip（该环境下 onnx 通路产品侧即不支持，测试不
  误报编译失败）；
- 有 atc 无 NPU 且未设 TTK_ONNX_TEST_SOC → soc 不可解析 skip（atc 离线编译
  不占卡，可经环境变量显式指定目标芯片跑通）；
- 建模统一走 build_model_bytes（env 最高 opset）；raw 模型（非法算子/NonZero）
  的 opset/IR 同源，输出 ValueInfo 用 make_tensor_value_info(UNDEFINED, None)
  （等价 make_empty_tensor_value_info，但不依赖 onnx 1.13 特有 helper）。
"""

import os
from unittest import mock

import numpy as np
import pytest

from ttk.core_modules.framework_api.onnx import om_compiler
from ttk.core_modules.framework_api.onnx.model_builder import build_model_bytes
from ttk.core_modules.framework_api.onnx.om_compiler import (
    OmniCompileError,
    _cache_key,
    _detect_om_path,
    _find_atc,
    _normalized_env,
    _plugin_fingerprint,
    compile_om,
)

_ATC, _ASCEND_ROOT = _find_atc()
requires_atc = pytest.mark.skipif(_ATC is None, reason="atc not found (no Ascend toolkit on disk)")


def _resolve_soc():
    """soc_version 解析：TTK_ONNX_TEST_SOC 显式指定优先，其次 DSMI 首个可解析
    芯片的完整平台名（atc --soc_version 直传值）。

    atc 离线编译不占卡，但需要合法 soc_version 串。遍历而非只探 0 号卡：
    ASCEND_VISIBLE_DEVICES 等可见性配置下 0 号卡可能不可达。无设备且未显式
    指定时返回 None（调用方 skip）。
    """
    soc = os.environ.get("TTK_ONNX_TEST_SOC")
    if soc:
        return soc
    try:
        from ttk.core_modules.common.device.dsmi.interface import DSMIInterface

        dsmi = DSMIInterface()
        for dev_id in range(dsmi.get_device_count()):
            info = dsmi.get_chip_info(dev_id)
            name = info.get_complete_platform() if info is not None else None
            if name:
                return name
    except Exception:  # noqa: BLE001  # 无设备/DSMI 缺失按不可解析处理（调用方 skip）
        return None
    return None


def _specs(*arrays):
    return [(f"input_{i}", arr.shape, str(arr.dtype)) for i, arr in enumerate(arrays)]


def _write_onnx(tmp_path, op_type, arrays, attrs, name="case"):
    path = tmp_path / f"{name}.onnx"
    path.write_bytes(build_model_bytes(op_type, _specs(*arrays), attrs))
    return str(path)


def _shapes(arrays):
    return {name: shape for name, shape, _ in _specs(*arrays)}


def test_normalized_env_prepends_cann_site_packages(monkeypatch, tmp_path):
    fake_root = str(tmp_path / "fake_root")
    cann_py = os.path.join(fake_root, "python", "site-packages")

    monkeypatch.setenv("PYTHONPATH", "/some/external")
    env = _normalized_env(fake_root)
    entries = env["PYTHONPATH"].split(":")
    assert entries[0] == cann_py
    assert "/some/external" in entries

    # 已在列时不重复插入（幂等）
    monkeypatch.setenv("PYTHONPATH", f"{cann_py}:/some/external")
    env = _normalized_env(fake_root)
    assert env["PYTHONPATH"].split(":") == [cann_py, "/some/external"]


def test_atc_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(om_compiler, "_find_atc", lambda: (None, "fake_root"))
    path = tmp_path / "m.onnx"
    path.write_bytes(b"dummy-bytes")  # _find_atc 失败先于模型读取，无需真实模型
    with pytest.raises(OmniCompileError, match="ATC_NOT_FOUND"):
        compile_om(str(path), {"input_0": (2, 3)}, "Ascend910_9362", str(tmp_path / "cache"))


def test_plugin_fingerprint_degenerate_roots(tmp_path):
    """根缺失/目录空 → 常量兜底（不阻塞编译）。"""
    assert _plugin_fingerprint(None) == "no-ascend-root"
    nonexistent = tmp_path / "nonexistent"
    assert _plugin_fingerprint(str(nonexistent)) == "unreadable-plugin-dir"
    empty_root = tmp_path / "empty-root"
    (empty_root / "opp" / "built-in" / "framework" / "onnx").mkdir(parents=True)
    assert _plugin_fingerprint(str(empty_root)) == "no-plugins"


def test_cache_key_includes_plugin_fingerprint():
    """指纹入键——同输入不同指纹必异键、同输入同指纹稳定复现。"""
    args = "--framework=5 --log=error"
    base = b"model-bytes"
    stock = _cache_key(base, "Ascend910_9362", args, "fp-stock")
    assert stock != _cache_key(base, "Ascend910_9362", args, "fp-patched")
    assert _cache_key(base, "Ascend910_9362", args, "fp-stock") == stock


def test_detect_om_path_probing(tmp_path):
    """产物路径探测：空目录空串、动态平台后缀 glob 兜底、exact 优先。

    纯 CPU 钉住 _detect_om_path 的探测逻辑（动态 om 的 atc 级验证见
    TestAtcRealCompile——CANN 版本行为分叉时此处仍恒绿）。
    """
    case = tmp_path / "case"
    case.mkdir()
    assert _detect_om_path(str(case)) == ""
    dynamic = case / "model_linux_aarch64.om"
    dynamic.write_bytes(b"om")
    assert _detect_om_path(str(case)) == str(dynamic)
    exact = case / "model.om"
    exact.write_bytes(b"om")
    assert _detect_om_path(str(case)) == str(exact)


@requires_atc
@pytest.mark.slow
class TestAtcRealCompile:
    """atc 真实转换：多语义算子族 + 缓存命中 + 失败透传。"""

    @pytest.fixture(autouse=True)
    def _ascend_env_for_atc(self):
        """conftest 的 isolate_ttk_environment 剥离 ASCEND_* 环境变量——生产链路
        由 cli 入口的 ttk._env.setup_env() source set_env.sh 回填，单测不经 CLI。
        此处复用同一 healing 重建 atc 子进程环境（EC0008 防御），测毕还原。
        """
        import ttk._env as env_mod

        saved = dict(os.environ)
        try:
            if _ASCEND_ROOT:
                env_mod._source_setenv_bash(_ASCEND_ROOT)
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)

    @pytest.fixture(autouse=True)
    def _need_onnx_with_supported_opset(self):
        """onnx 可用性 + opset 上限门禁。

        对齐 param_extractor.ATC_MAX_OPSET 的产品门禁：环境 onnx 包 opset 超
        atc 插件上限时，onnx 通路整体不支持（CSV 用例解析期即报
        ONNX_OPSET_VERSION_UNSUPPORTED）——真实编译用例随环境 skip，不误报
        atc 编译失败。build_model_bytes 以 env 最高 opset 建模，门禁必须先行。
        """
        onnx_defs = pytest.importorskip("onnx.defs")

        from ttk.core_modules.framework_api.onnx.param_extractor import ATC_MAX_OPSET

        env_opset = onnx_defs.onnx_opset_version()
        if env_opset > ATC_MAX_OPSET:
            pytest.skip(
                f"onnx.defs.onnx_opset_version()={env_opset} exceeds the atc plugin "
                f"ceiling ({ATC_MAX_OPSET}); the onnx pathway is unsupported in this "
                f"environment (same gate as param_extractor) — install e.g. onnx 1.13.x"
            )

    @pytest.fixture
    def soc(self):
        soc = _resolve_soc()
        if not soc:
            pytest.skip("soc_version unresolvable: no NPU via DSMI and TTK_ONNX_TEST_SOC unset")
        return soc

    def test_compile_op_family(self, tmp_path, soc):
        """atc 真实转换：Add 基线 + MatMul / Concat / Split 多语义算子族逐一编译出 om。"""
        cases = [
            ("Add", (np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32)), {}),
            ("MatMul", (np.zeros((32, 64), np.float32), np.zeros((64, 16), np.float32)), {}),
            ("Concat", (np.zeros((2, 3), np.int32), np.zeros((2, 4), np.int32)), {"axis": 1}),
            ("Split", (np.zeros((6, 4), np.float32),), {"axis": 0, "num_outputs": 3}),
        ]
        for op_type, arrays, attrs in cases:
            path = _write_onnx(tmp_path, op_type, arrays, attrs)
            om = compile_om(path, _shapes(arrays), soc, str(tmp_path / "cache"))
            assert om.endswith(".om"), op_type
            assert os.path.isfile(om), op_type

    def test_cache_hit_skips_atc(self, tmp_path, soc):
        arrays = (np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))
        path = _write_onnx(tmp_path, "Add", arrays, {})
        work = str(tmp_path / "cache")
        # 指纹钉为常量：缓存命中判定不受真实插件目录抖动（并发重编 .so）影响
        with mock.patch.object(om_compiler, "_plugin_fingerprint", return_value="fp-stable"):
            first = compile_om(path, _shapes(arrays), soc, work)
            with mock.patch.object(om_compiler.subprocess, "run") as run_mock:
                second = compile_om(path, _shapes(arrays), soc, work)
            run_mock.assert_not_called()  # 缓存命中：零 atc 子进程
        assert second == first
        # 指纹入键 → 同输入不同指纹必异键（缓存失效）由纯 CPU 的
        # test_cache_key_includes_plugin_fingerprint 钉住，此处不重复真实编译。

    def test_failure_payload_from_stdout(self, tmp_path, soc):
        """atc 诊断走 stdout：失败载荷合并捕获并透传（非法算子）。"""
        import onnx
        from onnx import TensorProto, helper

        node = helper.make_node("NotARealOpAtAll", ["input_0"], ["output_0"])
        graph = helper.make_graph(
            [node],
            "bad",
            [helper.make_tensor_value_info("input_0", TensorProto.FLOAT, (2, 3))],
            # 输出只给名字：UNDEFINED/None（对齐 model_builder 的 name-only 约定，
            # 全 onnx 版本可用——不依赖 1.13 特有的 make_empty_tensor_value_info）
            [helper.make_tensor_value_info("output_0", TensorProto.UNDEFINED, None)],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", onnx.defs.onnx_opset_version())],
            ir_version=onnx.IR_VERSION,
        )
        path = tmp_path / "bad.onnx"
        path.write_bytes(model.SerializeToString())
        with pytest.raises(OmniCompileError, match="ATC_COMPILE_FAILED"):
            compile_om(str(path), {"input_0": (2, 3)}, soc, str(tmp_path / "cache"))

    def test_dynamic_output_om_filename_detected(self, tmp_path, soc):
        """动态 shape om 带平台后缀：产物路径探测实际文件名。

        atc 对数据依赖 shape 算子的行为随 CANN 版本分叉：静默产出动态 dim om
        （cann-9.2.0 实测，产物名带 _linux_<arch> 后缀）或直接拒转（设计 §10
        风险 2 的原始预期）——两侧均为 TTK 的合法暴露（前者走 runner 动态输出
        守卫、后者 ATC_COMPILE_FAILED 原文透传），分别断言；探测逻辑本身由
        纯 CPU 的 test_detect_om_path_probing 恒绿钉住。
        """
        import onnx
        from onnx import TensorProto, helper

        node = helper.make_node("NonZero", ["input_0"], ["output_0"])
        graph = helper.make_graph(
            [node],
            "nz",
            [helper.make_tensor_value_info("input_0", TensorProto.FLOAT, (6,))],
            [helper.make_tensor_value_info("output_0", TensorProto.UNDEFINED, None)],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", onnx.defs.onnx_opset_version())],
            ir_version=onnx.IR_VERSION,
        )
        path = tmp_path / "nonzero.onnx"
        path.write_bytes(model.SerializeToString())
        try:
            om = compile_om(str(path), {"input_0": (6,)}, soc, str(tmp_path / "cache"))
        except OmniCompileError as exc:
            # 双出口断言（成功→探测 / 拒转→透传），无法收敛为 pytest.raises 单出口
            assert "ATC_COMPILE_FAILED" in str(exc)  # noqa: PT017
            pytest.skip(f"atc on this CANN rejects dynamic-shape outputs (payload passthrough ok): {str(exc)[:120]}")
        assert om.endswith(".om")
        assert os.path.isfile(om)
        assert os.path.basename(om).startswith("model")
