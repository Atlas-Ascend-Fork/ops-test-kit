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
XPU 三方能力专项测试：collect 门面（remote-only）/ ExecutionSpec / 模式位。
"""

import pytest

torch = pytest.importorskip("torch")


def _switches(**kw):
    from argparse import Namespace

    kw.setdefault("backend", "npu")
    return Namespace(**kw)


class TestExecutionSpecFields:
    def test_data_fields_default_none(self):
        from ttk.remote import ExecutionSpec

        spec = ExecutionSpec(provider="torch", api="torch.add")
        assert spec.inputs is None
        assert spec.attrs is None
        assert spec.testcase_name is None

    def test_data_fields_carry_payload(self):
        from ttk.remote import ExecutionSpec

        spec = ExecutionSpec(provider="torch", api="torch.add", inputs=[1], attrs={"a": 1})
        assert spec.inputs == [1]
        assert spec.attrs == {"a": 1}


class TestCollectFacadeRouting:
    def test_remote_delegates_to_remote_client(self, monkeypatch):
        import ttk.core_modules.common.xpu as xpu_mod
        import ttk.remote.client as client_mod

        def fake_remote_collect(**kw):
            calls.update(kw)
            return ("p", None, {})

        calls = {}
        monkeypatch.setattr(client_mod, "collect_third_party", fake_remote_collect)
        out = xpu_mod.collect_third_party(
            op_name="Add",
            inputs=[],
            input_names=[],
            op_type=None,
            attributes={},
            testcase_name="t",
            switches=_switches(),
            need_data=True,
        )
        assert out == ("p", None, {})
        assert calls["op_name"] == "Add"


class TestXpuModeOf:
    """xpu_mode_of：worker 三方采集编排的按位模式查询（DATA|PERF）。"""

    @staticmethod
    def _sw(**kw):
        from argparse import Namespace

        return Namespace(dump_config=Namespace(dump_xpu=kw.pop("dump_xpu", False)), xpu_perf=kw.pop("xpu_perf", False))

    def test_mode_zero_when_nothing_requested(self):
        from ttk.core_modules.common.xpu import XPU_DATA, XPU_PERF, xpu_mode_of

        assert xpu_mode_of(self._sw(), need_data=False) == 0
        assert (XPU_DATA, XPU_PERF) == (1, 2)  # 位值即公开契约

    def test_need_data_sets_data_bit(self):
        from ttk.core_modules.common.xpu import XPU_DATA, xpu_mode_of

        assert xpu_mode_of(self._sw(), need_data=True) == XPU_DATA

    def test_dump_xpu_sets_data_bit_without_cross_check(self):
        """--dump xpu 单独即触发 DATA 采集（不依赖 cross_check 判据）。"""
        from ttk.core_modules.common.xpu import XPU_DATA, xpu_mode_of

        assert xpu_mode_of(self._sw(dump_xpu=True), need_data=False) == XPU_DATA

    def test_xpu_perf_sets_perf_bit_and_combines(self):
        from ttk.core_modules.common.xpu import XPU_DATA, XPU_PERF, xpu_mode_of

        assert xpu_mode_of(self._sw(xpu_perf=True), need_data=False) == XPU_PERF
        assert xpu_mode_of(self._sw(xpu_perf=True, dump_xpu=True), need_data=False) == (XPU_DATA | XPU_PERF)
        assert xpu_mode_of(self._sw(xpu_perf=True), need_data=True) == (XPU_DATA | XPU_PERF)


class TestBindParams:
    """bind_params 按名绑定契约（自 ttk.core_modules.common.xpu 单份归位）。"""

    @staticmethod
    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            pytest.param({"reduction": "none"}, (2.0, 0.25, "none"), id="first_two_omitted"),
            pytest.param({"gamma": 2.0, "reduction": "none"}, (2.0, 0.25, "none"), id="middle_omitted"),
            pytest.param({"alpha": -1.6e38, "reduction": "none"}, (2.0, -1.6e38, "none"), id="head_omitted"),
            pytest.param({"gamma": 3.689, "alpha": 0.7, "reduction": "none"}, (3.689, 0.7, "none"), id="all_given"),
        ],
    )
    def test_omitted_earlier_optional_keeps_slot_alignment(attrs, expected):
        """省略前序可选属性时, 后面的值按名落位不得左移错位(类形式三方绑定场景)。"""
        from ttk.core_modules.common.xpu import bind_params

        class Impl:
            def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
                self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

        pool = dict(attrs)
        pool.update({"pred": "P", "target": "T", "weight": None})
        ia, ik = bind_params(Impl.__init__, pool, warn_leftover=False)
        inst = Impl(*ia, **ik)
        assert (inst.gamma, inst.alpha, inst.reduction) == expected

    @staticmethod
    def test_pre_star_default_holds_slot_for_varargs_and_kwonly():
        """缺省的前置可选参数占位后, *args 收集剩余、keyword-only 仍走 Python 默认。"""
        from ttk.core_modules.common.xpu import bind_params

        def f(a=1, b=2, *rest, key=None):
            return (a, b, rest, key)

        ia, ik = bind_params(f, {"b": 9, "zzz": "leftover", "key": "K"})
        assert f(*ia, **ik) == (1, 9, ("leftover",), "K")

        def g(x, *, axis=-1):
            return (x, axis)

        ia, ik = bind_params(g, {"x": 1})
        assert g(*ia, **ik) == (1, -1)
