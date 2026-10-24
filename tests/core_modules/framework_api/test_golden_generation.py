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
测试 ttk.core_modules.framework_api.golden 模块。

覆盖内容：
- generate_golden: 优先级链路（golden_api → 自定义插件 → CPU 回退）。
- exec_backend 设备选择：None = CPU 生成；NPU 后端 = 插件收 device 串、
  golden_api 与同 API 在 NPU 上执行。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ttk.core_modules.framework_api.api_model import ParamInfo
from ttk.core_modules.framework_api.golden import generate_golden
from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper


@pytest.fixture(autouse=True)
def _clean_keeper():
    FrameworkApiInfoKeeper().clear_cache()
    yield
    FrameworkApiInfoKeeper().clear_cache()


def _register(api_name, params, source="test"):
    """向 FrameworkApiInfoKeeper 注册 API 参数信息。"""
    FrameworkApiInfoKeeper().register(api_name, params, source=source)


def _make_testcase(api_name, shapes, dtypes, attrs=None, output_tensor_indexes=(), golden_api=None):
    """构造一个 TestcaseE2e 测试用例。"""
    from ttk.core_modules.framework_api.testcase import TestcaseE2e

    case = TestcaseE2e()
    case.testcase_name = f"test_{api_name}"
    case.api_name = api_name
    case.is_valid = True
    case.fail_reason = None
    case.tensor_view_shapes = shapes
    case.tensor_dtypes = dtypes
    case.attributes = attrs or {}
    case.output_tensor_indexes = output_tensor_indexes
    case.golden_api = golden_api
    return case


class TestGenerateGolden:
    """generate_golden: 优先级链路 golden_api → 自定义插件 → CPU 回退。"""

    def test_priority1_golden_api(self):
        """优先级 1：指定 golden_api 时走 golden_api 路径。"""
        _register(
            "torch.golden_target",
            [
                ParamInfo(name="input", type="Tensor"),
            ],
        )
        case = _make_testcase("torch.base_api", ((3,),), ("float32",), golden_api="torch.golden_target")
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        with patch("ttk.core_modules.framework_api.golden.resolve_api") as mock_resolve:
            mock_fn = MagicMock(return_value=torch.tensor([10.0, 20.0, 30.0]))
            mock_resolve.return_value = (mock_fn, False)
            result = generate_golden(case, raw)
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], [10.0, 20.0, 30.0])

    def test_priority2_custom_plugin(self):
        """优先级 2：有自定义插件时走插件路径。"""
        _register(
            "torch.relu",
            [
                ParamInfo(name="input", type="Tensor"),
            ],
        )
        case = _make_testcase("torch.relu", ((3,),), ("float32",))
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        case.tensors = raw

        fake_func = MagicMock(return_value=np.array([4.0, 5.0, 6.0]))
        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get:
            mock_get.return_value = fake_func
            result = generate_golden(case, raw)
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], [4.0, 5.0, 6.0])

    def test_priority3_cpu_fallback(self):
        """优先级 3：无插件时回退到 CPU 执行。"""
        _register(
            "torch.relu",
            [
                ParamInfo(name="input", type="Tensor"),
            ],
        )
        case = _make_testcase("torch.relu", ((4,),), ("float32",))
        raw = [np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)]

        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get:
            mock_get.return_value = None
            result = generate_golden(case, raw)
            assert len(result) == 1
            np.testing.assert_array_almost_equal(result[0], [0.0, 0.0, 1.0, 2.0])

    def test_priority3_cpu_fails_raises(self):
        """CPU 回退失败时抛出 RuntimeError。"""
        _register(
            "torch.noexist_api",
            [
                ParamInfo(name="input", type="Tensor"),
            ],
        )
        case = _make_testcase("torch.noexist_api", ((3,),), ("float32",))
        raw = [np.array([1.0], dtype=np.float32)]

        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get, patch(
            "ttk.core_modules.framework_api.golden._run_api"
        ) as mock_cpu:
            mock_get.return_value = None
            mock_cpu.side_effect = RuntimeError("no CPU impl")
            with pytest.raises(RuntimeError, match="cannot run on CPU"):
                generate_golden(case, raw)

    def test_priority3_npu_only_fast_fail(self):
        """优先级 3：npu_device 自定义算子无 CPU kernel → is_npu_only 直接
        返回 UNSUPPORTED，不执行后续链路。"""
        case = _make_testcase(
            "npu_device.compat.v1.tbe.npu_vector_ops.lamb_apply_optimizer_assign",
            ((3,),),
            ("float32",),
        )
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get, patch(
            "ttk.core_modules.framework_api.golden._run_api"
        ) as mock_cpu:
            mock_get.return_value = None
            result = generate_golden(case, raw)
            assert result == ["UNSUPPORTED"]
            mock_cpu.assert_not_called()

    def test_priority1_golden_api_npu_only(self):
        """优先级 1：golden_api 指向 NPU-only API → UNSUPPORTED 快速失败。"""
        case = _make_testcase(
            "tf.raw_ops.Add",
            ((3,),),
            ("float32",),
            golden_api="npu_device.compat.v1.tbe.npu_vector_ops.fake_op",
        )
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        with patch("ttk.core_modules.framework_api.golden._run_api") as mock_cpu:
            result = generate_golden(case, raw)
            assert result == ["UNSUPPORTED"]
            mock_cpu.assert_not_called()

    def test_with_nested_distribution(self):
        """嵌套分布（tuple of Tensors）正确传递。"""
        _register(
            "torch.stack_test",
            [
                ParamInfo(name="tensors", type="tuple of Tensors"),
            ],
        )
        case = _make_testcase("torch.stack_test", (((3,), (3,)),), (("float32", "float32"),))
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        raw = [a, b]

        with patch("ttk.core_modules.framework_api.golden.resolve_api") as mock_resolve:
            mock_fn = MagicMock(return_value=torch.zeros((2, 3)))
            mock_resolve.return_value = (mock_fn, False)
            generate_golden(case, raw)
            call_args = mock_fn.call_args[0]
            assert isinstance(call_args[0], (list, tuple))

    def test_mutable_ref_variables_reset_before_golden(self):
        """TF mutable-ref 变量在 golden 生成前按 raw_inputs 复位（防脏状态复算）。

        子进程运行：本文件 import torch 后同进程再 import tensorflow 会触发
        torch_npu 与 TF C 扩展冲突（段错误），参照 test_kernel_xpu_e2e 的
        子进程隔离模式。
        """
        import os
        import subprocess
        import sys
        import textwrap

        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import tensorflow"], capture_output=True, timeout=90, check=False
            )
        except Exception:
            probe = None
        if probe is None or probe.returncode != 0:
            pytest.skip("tensorflow not importable")

        script = textwrap.dedent(
            """
            import numpy as np
            import tensorflow as tf

            from ttk.core_modules.framework_api.golden import generate_golden
            from ttk.core_modules.framework_api.testcase import TestcaseE2e

            case = TestcaseE2e()
            case.testcase_name = "tc"
            case.api_name = "tf.raw_ops.ResourceApplyAdam"
            case.is_valid = True
            case.fail_reason = None
            case.tensor_view_shapes = ((3,),)
            case.tensor_dtypes = ("float32",)
            case.attributes = {}
            case.golden_api = "disable"
            var = tf.Variable(np.full((3,), 7.0, dtype=np.float32))
            case.tensors = [var]
            raw = [np.zeros((3,), dtype=np.float32)]
            assert generate_golden(case, raw) == ["SUPPRESSED"]
            np.testing.assert_array_equal(var.numpy(), np.zeros((3,), dtype=np.float32))
            print("RESET_OK")
            """
        )
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, env.get("PYTHONPATH", "")) if p)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env=env,
        )
        assert proc.returncode == 0, f"mutable-ref reset failed:\n{proc.stderr[-2000:]}"
        assert "RESET_OK" in proc.stdout


class TestExecBackendRouting:
    """exec_backend 决定设备：None = CPU 生成；NPU 后端 = 插件收 device 串、
    golden_api 与同 API 在 NPU 上执行。"""

    def test_npu_mode_without_exec_backend_runs_cpu_fallback(self):
        """golden_mode=NPU 且未传 exec_backend → 同 API 在 CPU 上生成。"""
        _register("torch.relu", [ParamInfo(name="input", type="Tensor")])
        case = _make_testcase("torch.relu", ((4,),), ("float32",))
        raw = [np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)]
        switches = SimpleNamespace(golden_mode="NPU")

        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get:
            mock_get.return_value = None
            result = generate_golden(case, raw, None, switches, "npu")
            assert len(result) == 1
            np.testing.assert_array_almost_equal(result[0], [0.0, 0.0, 1.0, 2.0])

    def test_exec_backend_routes_same_api_with_numpy_inputs(self):
        """exec_backend 为 NPU 后端且无插件/golden_api → 同 API 经 _run_api
        上板：to_device 收 numpy 数组（内部 from_numpy 后上板），不做先转
        torch 再传入的双重转换。"""
        from contextlib import nullcontext

        received = {}

        class _FakeNpuBackend:
            def is_npu(self):
                return True

            def set_device(self, dev_id):
                received["set_device"] = dev_id

            def device_scope(self, dev_id):
                return nullcontext()

            def to_device(self, tensor, dev_id=0, preserve_stride=False):
                received["input_type"] = type(tensor).__name__
                return tensor

        _register("torch.relu", [ParamInfo(name="input", type="Tensor")])
        case = _make_testcase("torch.relu", ((2, 3),), ("float32",))
        raw = [np.ones((2, 3), dtype=np.float32)]
        switches = SimpleNamespace(golden_mode="NPU")

        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get, patch(
            "ttk.core_modules.framework_api.golden._exec_and_convert", return_value=[np.ones(2)]
        ):
            mock_get.return_value = None
            result = generate_golden(case, raw, None, switches, "npu", exec_backend=_FakeNpuBackend(), dev_id=5)

        assert received["input_type"] == "ndarray", "to_device must receive numpy arrays"
        assert received["set_device"] == 5
        assert len(result) == 1

    def test_exec_backend_gives_plugin_device_string(self):
        """exec_backend 为 NPU 后端 → 插件收 device="npu:{dev_id}"；未传 → None。"""
        _register("torch.relu", [ParamInfo(name="input", type="Tensor")])
        case = _make_testcase("torch.relu", ((3,),), ("float32",))
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        switches = SimpleNamespace(golden_mode="NPU")
        fake_backend = SimpleNamespace(is_npu=lambda: True)

        with patch("ttk.core_modules.common.plugin.resolver.get_plugin_function") as mock_get, patch(
            "ttk.core_modules.framework_api.golden._call_plugin_with_plan"
        ) as mock_plan:
            mock_get.return_value = object()
            mock_plan.return_value = [np.array([9.0])]

            generate_golden(case, raw, None, switches, "npu", exec_backend=fake_backend, dev_id=2)
            assert mock_plan.call_args.kwargs["device"] == "npu:2"

            generate_golden(case, raw, None, switches, "npu")
            assert mock_plan.call_args.kwargs["device"] is None

    def test_exec_backend_routes_golden_api_to_npu(self):
        """exec_backend 为 NPU 后端且声明 golden_api → 以 golden_api 为目标经
        _run_api 在 NPU 上执行（api_info 绑定、backend/dev_id 透传），跳过
        is_npu_only / is_dtype_support 检查（NPU-only 的 API 在 NPU 上可执行）。"""
        npu_only_api = "npu_device.compat.v1.tbe.npu_vector_ops.fake_op"
        _register(npu_only_api, [ParamInfo(name="input", type="Tensor")])
        case = _make_testcase("torch.base_api", ((3,),), ("float32",), golden_api=npu_only_api)
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        switches = SimpleNamespace(golden_mode="NPU")
        fake_backend = SimpleNamespace(is_npu=lambda: True)

        with patch("ttk.core_modules.framework_api.golden._run_api") as mock_npu:
            mock_npu.return_value = [np.array([1.0, 1.0, 1.0])]
            result = generate_golden(case, raw, None, switches, "npu", exec_backend=fake_backend, dev_id=1)

            mock_npu.assert_called_once()
            assert mock_npu.call_args[0][0] == npu_only_api, "must target the golden_api, not the DUT api"
            assert mock_npu.call_args[0][3] is fake_backend, "exec_backend must be passed through"
            assert mock_npu.call_args[0][4] == 1, "dev_id must be passed through"
            assert mock_npu.call_args.kwargs["api_info"] is not None, "golden_api overload info must be bound"
            np.testing.assert_array_equal(result[0], [1.0, 1.0, 1.0])
