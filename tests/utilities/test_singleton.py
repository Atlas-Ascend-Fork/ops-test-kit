# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.utilities.singleton: per-(class, args) instance caching semantics."""

import threading

import pytest

from ttk.utilities.singleton import Singleton


class Keeper(metaclass=Singleton):
    def __init__(self, name="default"):
        self.name = name


def test_same_args_same_instance():
    assert Keeper("a") is Keeper("a")
    assert Keeper() is Keeper()


def test_distinct_args_distinct_instances():
    assert Keeper("a") is not Keeper("b")
    assert Keeper("a") is not Keeper()


def test_kwargs_order_insensitive():
    class Kw(metaclass=Singleton):
        def __init__(self, a=0, b=0):
            self.value = (a, b)

    assert Kw(a=1, b=2) is Kw(b=2, a=1)


def test_same_name_classes_do_not_collide():
    # key is the class object itself, not cls.__name__
    cls_a = Singleton("SameName", (), {})
    cls_b = Singleton("SameName", (), {})
    assert cls_a() is not cls_b()
    assert cls_a() is cls_a()


def test_unhashable_args_raise():
    with pytest.raises(TypeError):
        Keeper(["not-hashable"])


def test_concurrent_first_construction_returns_one_instance():
    class Fresh(metaclass=Singleton):
        def __init__(self):
            threading.Event().wait(0.01)  # widen the race window

    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(Fresh())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(r is results[0] for r in results)


def test_nested_singleton_construction_does_not_self_deadlock():
    """单例 __init__ 内构造另一单例（OperatorInterface→Opc 嵌套，曾引入过
    非重入 Lock 自死锁）：RLock 下同线程重入必须可行。"""

    class Inner(metaclass=Singleton):
        pass

    class Outer(metaclass=Singleton):
        def __init__(self):
            self.inner = Inner()

    outer = Outer()  # 非重入 Lock 时此处永久阻塞
    assert isinstance(outer.inner, Inner)
    assert Outer() is outer
