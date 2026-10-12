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
TTK backward 专项测试：后缀协议 / 带参拒绝 / 梯度 dtype 锚点 / CPU golden 链。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _make_case(api_name, **kw):
    from ttk.core_modules.framework_api.testcase import TestcaseE2e

    case = TestcaseE2e()
    case.testcase_name = kw.pop("testcase_name", "bw_t")
    case.api_name = api_name
    case.tensor_view_shapes = kw.pop("tensor_view_shapes", ((2, 3), (2, 3)))
    case.tensor_dtypes = kw.pop("tensor_dtypes", ("float32", "float32"))
    for k, v in kw.items():
        setattr(case, k, v)
    case.validate()
    return case


class TestResultToNumpyFlatBackwardGate:
    """_result_to_numpy_flat 的元组启发式必须带 backward 门控：原生返回
    (Tensor, list[Tensor]) 的 API 不得被误拆（list 项走 np.array 堆叠语义）。"""

    class _KeepListBackend:
        """模拟 base.result_to_numpy 语义：list 项 np.array 堆叠、不递归展开。"""

        @staticmethod
        def result_to_numpy(result, copy=False):
            if result is None:
                return [None]
            results = result if isinstance(result, (tuple, list)) else [result]
            out = []
            for r in results:
                out.append(r.numpy() if hasattr(r, "numpy") else np.asarray(r))
            return out

    def test_native_tuple_list_not_split_without_backward(self):
        from ttk.core_modules.framework_api.execute import _result_to_numpy_flat

        t0, t1, t2 = torch.ones(2), torch.full((2,), 2.0), torch.full((2,), 3.0)
        result = (t0, [t1, t2])  # API 原生返回形态
        nps = _result_to_numpy_flat(self._KeepListBackend(), result)
        assert len(nps) == 2
        assert nps[1].shape == (2, 2)  # list 项整体堆叠，非逐项展开

    def test_backward_wrapper_split_with_gate(self):
        from ttk.core_modules.framework_api.execute import _result_to_numpy_flat

        t0, t1, t2 = torch.ones(2), torch.full((2,), 2.0), torch.full((2,), 3.0)
        result = (t0, [t1, t2])  # backward 包装 (forward, grads) 同形态
        nps = _result_to_numpy_flat(self._KeepListBackend(), result, backward=True)
        assert len(nps) == 3
        assert nps[1].tolist() == [2.0, 2.0]
        assert nps[2].tolist() == [3.0, 3.0]


class TestBackwardSuffixProtocol:
    def test_strip_plain_suffix(self):
        case = _make_case("torch.add.backward")
        assert case.backward_mode is True
        assert case.api_name == "torch.add"

    def test_no_suffix_untouched(self):
        case = _make_case("torch.add")
        assert case.backward_mode is False
        assert case.api_name == "torch.add"

    def test_parameterized_suffix_rejected(self):
        case = _make_case("torch.add.backward(grad_tensors=...)")
        assert case.is_valid is False
        assert "parameterized" in case.fail_reason

    def test_empty_forward_rejected(self):
        case = _make_case(".backward")
        assert case.is_valid is False
        assert "unsupported backward suffix" in case.fail_reason

    def test_trailing_dot_forward_rejected(self):
        case = _make_case("torch..backward")
        assert case.is_valid is False
        assert "unsupported backward suffix" in case.fail_reason


class TestBackwardDtypeAnchor:
    def _ext(self, dtypes=("float32", "float32")):
        from ttk.core_modules.framework_api.compare import _extend_backward_dtypes

        case = _make_case("torch.add.backward", tensor_dtypes=dtypes)
        return _extend_backward_dtypes(case, ("float32",))

    def test_appends_grad_dtypes_for_floating_inputs(self):
        assert self._ext() == ("float32", "float32", "float32")

    def test_mixed_dtypes_only_floating_grads(self):
        assert self._ext(("float32", "int64")) == ("float32", "float32")

    def test_noop_without_backward_mode(self):
        from ttk.core_modules.framework_api.compare import _extend_backward_dtypes

        case = _make_case("torch.add")
        assert _extend_backward_dtypes(case, ("float32",)) == ("float32",)


class TestBackwardExecutionHelpers:
    def test_enable_requires_grad_only_floating(self):
        from ttk.core_modules.framework_api.torch.eager_runner import enable_requires_grad as _enable_requires_grad

        a = torch.randn(2, 3)
        b = torch.randn(2, 3, requires_grad=True)
        i = torch.ones(2, 3, dtype=torch.int64)
        _enable_requires_grad(None, [a, b, i, "x", None])
        assert a.requires_grad
        assert b.requires_grad
        assert not i.requires_grad

    def test_run_backward_non_scalar_output(self):
        from ttk.core_modules.framework_api.torch.eager_runner import run_backward as _run_backward

        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3, requires_grad=True)
        out = torch.add(a, b)
        grads = _run_backward(None, out, [a, b])
        assert len(grads) == 2
        assert torch.equal(grads[0], torch.ones(2, 3))
        assert torch.equal(grads[1], torch.ones(2, 3))

    def test_run_backward_unused_input(self):
        from ttk.core_modules.framework_api.torch.eager_runner import run_backward as _run_backward

        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3, requires_grad=True)
        out = a * 2
        grads = _run_backward(None, out, [a, b])
        assert len(grads) == 1
        assert torch.equal(grads[0], torch.full((2, 3), 2.0))


class TestCpuGoldenBackwardChain:
    def test_run_backward_on_cpu_manual_path(self):
        from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend
        from ttk.core_modules.framework_api.golden import _run_backward_on_cpu

        raw = [np.random.rand(2, 3).astype(np.float32) for _ in range(2)]
        case = _make_case("torch.add.backward")
        backend = CpuTorchBackend()
        out = _run_backward_on_cpu("torch.add", raw, case, None, backend)
        assert len(out) == 3
        np.testing.assert_allclose(out[0], raw[0] + raw[1], rtol=1e-6)
        np.testing.assert_allclose(out[1], np.ones((2, 3), dtype=np.float32), rtol=1e-6)
        np.testing.assert_allclose(out[2], np.ones((2, 3), dtype=np.float32), rtol=1e-6)
