# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ParamInfo.is_mutable / is_resource 参数级可变性回归（共享字段 + 双框架填充）。

设计契约：可变性是接口确定即可知的静态事实，落在参数级模型上——
  - torch：aten schema alias_info.is_write（schema 源）/ 尾下划线命名投影
    （docstring/pyi 源，_resolve_api_flags 兜底）；
  - tf：OpDef is_ref / DT_RESOURCE 按参数名落 flag（mutable_input_flags）。
torch 侧 in-process；TF 侧子进程运行——torch 进程后加载 TF 的 C 扩展会段错误
（torch_npu 与 TF 运行时冲突，参照 test_golden_generation 的子进程隔离模式）。
"""

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 不可导入")


def test_paraminfo_mutable_defaults():
    """共享字段默认 False（框架无关的纯模型行为）。"""
    from ttk.core_modules.framework_api.api_model import ParamInfo

    p = ParamInfo(name="x", type="Tensor")
    assert p.is_mutable is False
    assert p.is_resource is False


@requires_torch
def test_torch_schema_fills_is_mutable():
    """torch.ops.*（schema 源）：写别名逐参数落 is_mutable。"""
    from ttk.core_modules.framework_api.torch.param_extractor import get_api_params

    info = get_api_params("torch.ops.aten.add_")
    flags = {p.name: p.is_mutable for p in info.params}
    assert flags["self"] is True
    assert flags["other"] is False
    assert flags["alpha"] is False

    plain = get_api_params("torch.ops.aten.add")
    assert not any(p.is_mutable for p in plain.params)


@requires_torch
def test_torch_naming_projection_fills_first_tensor_param():
    """torch.Tensor.xxx_（docstring 源，无别名元数据）：命名投影到首张量参数。"""
    from ttk.core_modules.framework_api.torch.param_extractor import get_api_params

    info = get_api_params("torch.Tensor.add_")
    assert info.params[0].name == "self"
    assert info.params[0].is_mutable is True
    assert not any(p.is_mutable for p in info.params[1:])


def test_eager_mutable_keys_derivation():
    """execute 的克隆轮换键派生：mutable_slots 模型派生（位置+名字），解析降级回退槽 0。"""
    from ttk.core_modules.framework_api.execute import _eager_mutable_keys

    args = [object(), object(), object()]

    def _adapter(positions, names):
        return SimpleNamespace(mutable_slots=lambda _api: (frozenset(positions), frozenset(names)))

    # 模型被写位：位置下标 + kwargs 名字（运行期非 None 过滤）
    adapter = _adapter([0, 2], ["acc"])
    assert _eager_mutable_keys(adapter, "torch.fake", args, {"acc": object()}, False) == {0, 2, "acc"}
    assert _eager_mutable_keys(adapter, "torch.fake", args, {}, False) == {0, 2}
    # 位次越界 / 值为 None → 剔除
    assert _eager_mutable_keys(_adapter([5], []), "torch.fake", args, {}, False) == set()
    assert _eager_mutable_keys(_adapter([0], []), "torch.fake", [None, object()], {}, False) == set()

    # 模型空 + 命名为原地（torch 解析降级）→ 回退槽 0；args[0] 为 None 则不回退
    empty = _adapter([], [])
    assert _eager_mutable_keys(empty, "torch.fake_", args, {}, True) == {0}
    assert _eager_mutable_keys(empty, "torch.fake_", args, {}, False) == set()
    assert _eager_mutable_keys(empty, "torch.fake_", [None, object()], {}, True) == set()


def test_mutable_model_reads_short_circuit_non_tf_names():
    """非 TF 前缀短路（不 import tensorflow）——mutability 钩子的进程隔离守卫。"""
    from ttk.core_modules.framework_api.adapter import get_adapter
    from ttk.core_modules.framework_api.tf.param_extractor import (
        mutable_flat_indexes,
        resource_param_names,
    )

    tf_adapter = get_adapter("tf")
    assert tf_adapter.is_mutable_api("torch.add") is False
    assert tf_adapter.is_mutable_api("") is False
    assert tf_adapter.mutable_slots("torch.ops.aten.add_") == (frozenset(), frozenset())
    assert mutable_flat_indexes("torch.add") == ()
    assert resource_param_names("torch.nn.functional.relu_") == frozenset()


def _tf_available():
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import tensorflow"], capture_output=True, timeout=90, check=False
        )
    except Exception:
        return False
    return probe.returncode == 0


def test_tf_param_flags_filled_from_opdef():
    """TF 侧 OpDef → ParamInfo flag + 模型读取（子进程：TF C 扩展进程隔离）。"""
    if not _tf_available():
        pytest.skip("tensorflow not importable")

    script = textwrap.dedent(
        """
        import json

        import tensorflow as tf

        from ttk.core_modules.framework_api.adapter import get_adapter
        from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper
        from ttk.core_modules.framework_api.tf.param_extractor import (
            mutable_flat_indexes,
        )

        keeper = FrameworkApiInfoKeeper()

        def flags(api):
            info = keeper.get(api)
            return None if info is None else [[p.name, p.is_mutable, p.is_resource] for p in info.params]

        adapter = get_adapter("tf")
        v = tf.Variable([1.0, 2.0])
        out = {
            "resource_scatter_add": flags("tf.raw_ops.ResourceScatterAdd"),
            "scatter_div_raw": flags("tf.raw_ops.ScatterDiv"),
            "scatter_div_v1": flags("tf.compat.v1.scatter_div"),
            "relu": flags("tf.nn.relu"),
            "mutable_rsa": adapter.is_mutable_api("tf.raw_ops.ResourceScatterAdd"),
            "mutable_relu": adapter.is_mutable_api("tf.nn.relu"),
            "slots_v1": [sorted(list(x)) for x in adapter.mutable_slots("tf.compat.v1.scatter_div")],
            "flat_rsa": list(mutable_flat_indexes("tf.raw_ops.ResourceScatterAdd")),
            "flat_relu": list(mutable_flat_indexes("tf.nn.relu")),
            "readback_rsa": [
                [float(x) for x in row] for row in adapter.mutable_readback("tf.raw_ops.ResourceScatterAdd", [v], {})
            ],
            "readback_relu": adapter.mutable_readback("tf.nn.relu", [v], {}),
        }
        print("RESULT:" + json.dumps(out))
        """
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=300, check=False, text=True)
    assert run.returncode == 0, f"subprocess failed: {run.stderr[-2000:]}"
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT:"))
    out = json.loads(line[len("RESULT:") :])

    # Resource* raw op：DT_RESOURCE 位 mutable + resource
    assert out["resource_scatter_add"][0] == ["resource", True, True]
    assert out["resource_scatter_add"][1] == ["indices", False, False]
    assert out["resource_scatter_add"][2] == ["updates", False, False]
    # TF1 is_ref 位（raw）：mutable 非 resource
    assert out["scatter_div_raw"][0] == ["ref", True, False]
    # snake_case 封装（CamelCase 归一化代理查询）：mutable 非 resource
    assert out["scatter_div_v1"][0] == ["ref", True, False]
    # 普通 API：零可变位
    assert all(not f[1] for f in out["relu"])
    # 统一钩子：谓词 / 槽位 / 回读
    assert out["mutable_rsa"] is True
    assert out["mutable_relu"] is False
    assert out["slots_v1"] == [[0], ["ref"]]
    assert out["flat_rsa"] == [0]
    assert out["flat_relu"] == []
    assert out["readback_rsa"] == [[1.0, 2.0]]
    assert out["readback_relu"] == []
