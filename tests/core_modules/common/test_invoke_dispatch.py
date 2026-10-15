#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""func_dispatch 测试：framework_of 框架识别、bind_by_name 按名绑定参数、resolve_callable_str 字符串解析。"""

from types import SimpleNamespace

import numpy
import pytest

from ttk.core_modules.common.invoke import (
    Invoke,
    resolve_callable_str,
)
from ttk.core_modules.common.invoke import (
    _bind_by_name as bind_by_name,
)
from ttk.core_modules.common.invoke import (
    _framework_of as framework_of,
)


def test_framework_of_identifies_numpy_torch_and_custom():
    """framework_of 识别 numpy ufunc / numpy 函数 / torch builtin / torch OpOverloadPacket；
    自定义函数、type、instance 返回 None。"""
    import torch

    assert framework_of(numpy.add) == "numpy"  # ufunc
    assert framework_of(numpy.isposinf) == "numpy"  # 非 ufunc
    assert framework_of(numpy.linalg.norm) == "numpy"  # 子模块函数 (numpy.linalg)
    assert framework_of(torch.add) == "torch"  # 顶层 builtin
    assert framework_of(torch.ops.aten.amax) == "torch"  # OpOverloadPacket (torch._ops.aten)
    assert framework_of(torch.nn.functional.relu) == "torch"  # 子模块函数
    assert framework_of(lambda x: x) is None  # 自定义 lambda

    def g(x, **kw):
        return x

    assert framework_of(g) is None

    class C:
        def __call__(self, x):
            return x

    assert framework_of(C) is None  # type(class)
    assert framework_of(C()) is None  # instance


def _func_with_module(name):
    def f(x):
        return x

    f.__module__ = name
    return f


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        # 根包等值与根包加点子模块 → 对应框架（tf/tensorflow 均映射 tf）
        ("torch", "torch"),
        ("torch.nn.functional", "torch"),
        ("torch._ops.aten", "torch"),
        ("numpy", "numpy"),
        ("numpy.linalg", "numpy"),
        ("tf", "tf"),
        ("tf.raw_ops", "tf"),
        ("tensorflow", "tf"),
        ("tensorflow.python.ops", "tf"),
        # 相似名（裸前缀、无点）→ custom
        ("torchair", None),
        ("torch_npu", None),
        ("torchvision", None),
        ("tfrecord", None),
        ("tf_helpers", None),
        ("numpy_helpers", None),
        ("tensorflow_helpers", None),
        ("numpyx", None),
    ],
)
def test_framework_of_namespace_boundaries(module_name, expected):
    """命名空间边界：根包等值/根包加点子模块才属框架，裸前缀相似名归 custom。

    函数与 callable 实例两种形态断言（__module__ / type(obj).__module__）。
    """
    assert framework_of(_func_with_module(module_name)) == expected

    class C:
        def __call__(self, x):
            return x

    C.__module__ = module_name
    assert framework_of(C()) == expected


def test_bind_by_name_partial_split_init_and_call():
    """__init__ 和 __call__ 分阶段绑定：init 从 pool 取 x/axis，call 从 pool 取 y。"""

    class G:
        def __init__(self, x, *, axis):
            self.x, self.axis = x, axis

        def __call__(self, y):
            return [self.x + y + self.axis]

    pool = {"x": 1, "y": 2, "axis": 9}
    ia, ik = bind_by_name(G.__init__, pool)
    inst = G(*ia, **ik)
    ca, ck = bind_by_name(inst.__call__, pool)
    assert inst(*ca, **ck) == [12]
    assert inst.x == 1
    assert inst.axis == 9


def test_resolve_callable_str_from_dotted_path():
    """resolve_callable_str 把 'numpy.add' / 'torch.mm' 等字符串解析为 callable。"""
    import torch

    assert resolve_callable_str("numpy.add") is numpy.add
    assert resolve_callable_str("np.negative") is numpy.negative
    assert resolve_callable_str("torch.mm") is torch.mm
    assert callable(resolve_callable_str("torch.ops.aten.amax"))


# ── Invoke.golden：统一执行入口 ────────────────────────────────────────


def test_golden_unregistered_returns_unsupported():
    """func is None → ["UNSUPPORTED"]（未注册，不执行闭包）。"""
    from ttk.core_modules.common.invoke import Invoke

    def boom(func):
        raise AssertionError("must not be invoked")

    assert Invoke.golden(SimpleNamespace(), None, _switches(), boom) == ["UNSUPPORTED"]


def test_golden_disable_returns_suppressed():
    """golden_mode=Disable → ["SUPPRESSED"]；override 优先于全局。"""
    from ttk.core_modules.common.invoke import Invoke

    case = SimpleNamespace(golden_mode_override=None)
    assert Invoke.golden(case, numpy.abs, _switches(golden_mode="Disable")) == ["SUPPRESSED"]
    # override=Promote 叠加全局 Disable → 不抑制（通路侧另有 switch 级预检）
    seen = Invoke.golden(
        SimpleNamespace(golden_mode_override="Promote"),
        numpy.abs,
        _switches(golden_mode="Disable"),
        lambda func: ["ran"],
    )
    assert seen == ["ran"]


def test_golden_resolves_str_and_passes_to_closure():
    """str 形态经 resolve_callable_str 解析后传入通路闭包。"""
    from ttk.core_modules.common.invoke import Invoke

    seen = {}

    def invoke(func):
        seen["func"] = func
        return [1]

    result = Invoke.golden(SimpleNamespace(), "numpy.abs", _switches(), invoke)
    assert result == [1]
    assert seen["func"] is numpy.abs


def test_golden_direct_call_without_closure():
    """invoke 缺省 → 直接调用 func()（e2e runner 形态）。"""
    from ttk.core_modules.common.invoke import Invoke

    assert Invoke.golden(SimpleNamespace(), lambda: [2, 3], _switches()) == [2, 3]


def test_golden_placeholder_semantics():
    """Timeout → GOLDEN_TIMEOUT；Exception → GOLDEN_FAILURE（不外抛）。"""
    from ttk.core_modules.common.invoke import Invoke

    def timeout(func):
        raise TimeoutError

    def failure(func):
        raise ValueError("bad golden")

    case = SimpleNamespace(golden_mode_override=None)
    assert Invoke.golden(case, numpy.abs, _switches(), timeout) == ["GOLDEN_TIMEOUT"]
    assert Invoke.golden(case, numpy.abs, _switches(), failure) == ["GOLDEN_FAILURE"]


# ── Invoke.npu_preprocess：统一执行入口 ───────────────────────────────────


class _FakeAclnnPlan:
    """aclnn 形态 plan：param_layout + build_args(tensors, scalars, attributes)。"""

    param_layout = (("tensor", "x", "aclTensor*", None), ("other", "attr0", "int", 3))

    def build_args(self, tensors, scalars, attributes):
        args = [tensors[0], scalars[0], attributes.get("attr0")]
        extra = {name: value for name, value in attributes.items() if name != "attr0"}
        return args, extra


class _FakeAclnnCase:
    def __init__(self):
        self.api_name = "aclnnX"
        self.testcase_name = "case"
        self.tensors = ["t0"]
        self.scalars = [7]
        self.attributes = {"attr0": 1, "leftover": "lv"}
        self._plan = _FakeAclnnPlan()

    def get_param_plan(self):
        return self._plan


class _FakeE2eCase:
    def __init__(self):
        self.api_name = "torch.add"
        self.testcase_name = "e2e-case"
        self.attributes = {"alpha": 2, "declared": "d"}
        self._plan = SimpleNamespace(overload_params=(SimpleNamespace(name="alpha"), SimpleNamespace(name="x")))

    def get_param_plan(self):
        return self._plan


def _switches(**kwargs):
    return SimpleNamespace(short_soc_version="Ascend910B", **kwargs)


def test_npu_preprocess_unregistered_returns_none():
    """func is None → None（未注册跳过，不装配、不触设备；upstream !281 契约）。"""
    assert Invoke.npu_preprocess(_FakeAclnnCase(), None, _switches()) is None


def test_npu_preprocess_self_assembles_from_plan():
    """device_args 缺省 → ACLNN 形态经 plan.build_args 自装配 + 属性按关键字注入。"""
    seen = {}

    def hook(x, scalar, attr0, leftover, testcase_name):
        seen.update(x=x, scalar=scalar, attr0=attr0, leftover=leftover, testcase_name=testcase_name)

    assert Invoke.npu_preprocess(_FakeAclnnCase(), hook, _switches()) is None
    assert seen == {
        "x": "t0",
        "scalar": 7,
        "attr0": 1,
        "leftover": "lv",
        "testcase_name": "case",
    }


def test_npu_preprocess_device_args_passthrough_with_signature_trimming():
    """E2E 形态：device_args 直通；未声明的元信息/属性按签名裁剪丢弃。"""
    seen = {}

    def hook(x, alpha):
        seen.update(x=x, alpha=alpha)

    case = _FakeE2eCase()
    assert (
        Invoke.npu_preprocess(
            case,
            hook,
            _switches(),
            device_args=(["t0"], {"alpha": 2}),
        )
        is None
    )
    assert seen == {"x": "t0", "alpha": 2}  # declared/testcase_name 等未声明 → 裁剪


def test_npu_preprocess_kwargs_hook_receives_meta_extras():
    """声明 **kwargs 的 hook 自动收到元信息（含属性池中非参数名条目）。"""
    seen = {}

    def hook(**kw):
        seen.update(kw)

    case = _FakeE2eCase()
    Invoke.npu_preprocess(
        case,
        hook,
        _switches(),
        device_args=((), {"alpha": 2}),
    )
    assert seen["alpha"] == 2
    assert seen["declared"] == "d"  # 非参数名属性注入
    assert seen["testcase_name"] == "e2e-case"
    assert seen["short_soc_version"] == "Ascend910B"


def test_npu_preprocess_rejects_context_parameter():
    """hook 声明 context 参数 → NPU_PREPROCESS_FAILURE（框架无关契约）。"""

    def hook(context):
        pass

    with pytest.raises(RuntimeError, match="must not declare a framework context parameter"):
        Invoke.npu_preprocess(_FakeAclnnCase(), hook, _switches(), device_args=((), {}))


def test_npu_preprocess_must_return_none():
    """hook 返回非 None → NPU_PREPROCESS_FAILURE（就地修改，不产出数据）。"""

    def hook():
        return 1

    with pytest.raises(RuntimeError, match="must return None"):
        Invoke.npu_preprocess(_FakeAclnnCase(), hook, _switches(), device_args=((), {}))


def test_npu_preprocess_wraps_hook_exception():
    """hook 异常包装为 NPU_PREPROCESS_FAILURE；已带前缀的不双重包装。"""

    def boom():
        raise ValueError("bad state")

    def prefixed():
        raise RuntimeError("NPU_PREPROCESS_FAILURE: bad state")

    case = _FakeAclnnCase()
    with pytest.raises(RuntimeError, match="NPU_PREPROCESS_FAILURE: bad state"):
        Invoke.npu_preprocess(case, boom, _switches(), device_args=((), {}))
    with pytest.raises(RuntimeError, match="^NPU_PREPROCESS_FAILURE: bad state$"):
        Invoke.npu_preprocess(case, prefixed, _switches(), device_args=((), {}))


def test_npu_preprocess_runs_inside_device_scope():
    """device_scope 为工厂：hook 在其返回的上下文内执行。"""
    order = []

    class _Scope:
        def __enter__(self):
            order.append("enter")

        def __exit__(self, *exc):
            order.append("exit")

    def hook():
        order.append("hook")

    Invoke.npu_preprocess(
        _FakeAclnnCase(),
        hook,
        _switches(),
        device_args=((), {}),
        device_scope=_Scope,
    )
    assert order == ["enter", "hook", "exit"]


def test_bind_by_name_missing_param_raises_unknown_param_error():
    """缺值且无默认值的参数 → UnknownParamError（名拼错快速失败的公开契约）。"""
    from ttk.core_modules.common.xpu import UnknownParamError

    def hook(alpha, beta):
        return alpha + beta

    with pytest.raises(UnknownParamError, match="parameter 'beta' of .* has no value in pool"):
        bind_by_name(hook, {"alpha": 1})


@pytest.mark.parametrize(
    ("pool", "expected"),
    [
        ({"alpha": -1.6e38, "reduction": "none"}, (2.0, -1.6e38, "none")),
        ({"gamma": 3.0, "reduction": "none"}, (3.0, 0.25, "none")),
        ({"reduction": "none"}, (2.0, 0.25, "none")),
    ],
)
def test_bind_by_name_omitted_earlier_optional_keeps_slot_alignment(pool, expected):
    """省略前序可选属性时补默认值占位，后面的值按名落位不得左移错位。

    pre-star 可选参数缺失若直接跳过，后续参数左移一位（alpha 的值落到
    gamma 上）；post-star（keyword-only）缺失留给 Python 走默认。"""

    class G:
        def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
            self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    ia, ik = bind_by_name(G.__init__, pool)
    inst = G(*ia, **ik)
    assert (inst.gamma, inst.alpha, inst.reduction) == expected


def test_bind_by_name_pre_star_default_holds_slot_for_varargs_and_kwonly():
    """缺省的前置可选参数占位后，*args 收集剩余、keyword-only 仍走 Python 默认。"""

    def f(a=1, b=2, *rest, key=None):
        return a, b, rest, key

    ia, ik = bind_by_name(f, {"b": 20, "x": 9, "y": 8})
    assert f(*ia, **ik) == (1, 20, (9, 8), None)

    def g(x, *, axis=-1):
        return x, axis

    ia, ik = bind_by_name(g, {"x": 5})
    assert g(*ia, **ik) == (5, -1)
