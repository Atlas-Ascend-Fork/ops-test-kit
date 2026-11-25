# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""原地 TensorList API（torch._foreach_add_ 族）的 mutable 槽生命周期回归。

历史缺陷：``Tensor(a!)[] self`` 槽位在 eager 执行时是 list[Tensor]，
prepare_mutable_rounds/clone/to_numpy 全链路只接受单 Tensor，预置阶段即抛
``AttributeError: 'list' object has no attribute 'is_contiguous'``。
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _backend():
    from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend

    return CpuTorchBackend()


def test_clone_tensorlist_recursive_and_type_kept():
    backend = _backend()
    tensors = [torch.ones(3), torch.arange(4, dtype=torch.float32)]
    cloned = backend.clone(tensors)
    assert isinstance(cloned, list)
    cloned[0][0] = 42
    assert tensors[0][0] == 1, "逐元素克隆须为深拷贝"
    tup = backend.clone((torch.zeros(2),))
    assert isinstance(tup, tuple)
    assert tup[0].shape == (2,)
    assert backend.clone([None]) == [None]


def test_to_numpy_tensorlist_recursive():
    backend = _backend()
    tensors = [torch.full((3,), 6.0), torch.full((2,), 3.0)]
    nps = backend.to_numpy(tensors, safe=True)
    assert isinstance(nps, list)
    assert len(nps) == 2
    assert nps[0].tolist() == [6.0, 6.0, 6.0]


def test_prepare_mutable_rounds_tensorlist_carriers_clean_per_round():
    backend = _backend()
    args = [[torch.zeros(3)], [torch.ones(3)]]
    load = backend.prepare_mutable_rounds(args, {}, {0}, 2)
    load(0)
    args[0][0].add_(5)
    load(1)
    assert args[0][0].eq(0).all(), "逐轮载体须从初值克隆，不得携带上轮原地改写"


def test_output_nps_tensorlist_readback_flattened():
    from ttk.core_modules.framework_api.utils import output_nps

    backend = _backend()
    xs = [torch.full((3,), 6.0), torch.full((2,), 3.0)]
    result_nps = output_nps(backend, xs, [xs], {}, ())
    assert [a.tolist() for a in result_nps] == [[6.0, 6.0, 6.0], [3.0, 3.0]]
    declared = output_nps(
        backend,
        None,
        [xs],
        {},
        (0,),
        api_name="torch._foreach_add_",
        adapter=SimpleNamespace(mutable_readback=lambda *a: []),
    )
    assert [a.tolist() for a in declared] == [[6.0, 6.0, 6.0], [3.0, 3.0]], "声明被写位的 TensorList 须逐元素展平"


def test_foreach_add_inplace_eager_chain():
    """复现链：schema/命名识别槽 0 → 逐轮载体 → 原地执行 → 取回。"""
    from ttk.core_modules.framework_api.adapter import get_adapter
    from ttk.core_modules.framework_api.utils import _eager_mutable_keys, output_nps, round_nps

    backend = _backend()
    adapter = get_adapter("torch")
    args = [[torch.zeros(3), torch.zeros(2)], [torch.full((3,), 3.0), torch.full((2,), 3.0)]]
    kwargs = {"alpha": 2.0}
    keys = _eager_mutable_keys(adapter, "torch._foreach_add_", args, kwargs, True)
    assert keys == {0}

    total = 3
    load = backend.prepare_mutable_rounds(args, kwargs, keys, total)
    result = None
    for i in range(total):
        load(i)
        result = torch._foreach_add_(*args, **kwargs)
    nps = output_nps(backend, result, args, kwargs, ())
    assert [a.tolist() for a in nps] == [[6.0, 6.0, 6.0], [6.0, 6.0]]
    md5_nps = round_nps(backend, result, args, (), api_name="torch._foreach_add_", kwargs=kwargs, adapter=adapter)
    assert len(md5_nps) == 2


def test_graph_round_md5_tensorlist_inplace_backups():
    from ttk.core_modules.framework_api.torch.graph_runner import _round_md5

    backend = _backend()
    args = [[torch.full((3,), 6.0)]]
    first = _round_md5(args, {}, backend, args[0], {0: args[0]}, None)
    args[0][0].add_(1)
    args[0] = [torch.full((3,), 6.0)]
    second = _round_md5(args, {}, backend, args[0], {0: args[0]}, None)
    assert first == second, "TensorList 被写位回读须展平后参与 MD5"
