# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""E2E 通路结构断言：框架子目录隔离 + 依赖方向 + 前缀分支收敛。

- torch/ 与 tf/ 互不 import（双向断言，含函数内惰性 import）；
- tf/ 顶层零 torch/torch_npu import（torch_npu 与 TF C 扩展同进程冲突）；
- onnx/ 全树零 torch/torch_npu/tensorflow import（numpy 直通域）；
- onnx_backend 零 ACL（无 ctypes import / CDLL 调用 / 通路层 import——so
  加载属通路层职责，设备探测走 DSMI）；
- common/ 不得 import 通路（framework_hooks 注册表为唯一合法反向通道，且其
  自身不含通路 import——由通路侧主动注册）；
- 通路共享层（worker/execute/compare/utils/golden/input_generate/instance/
  testcase/result/adapter 之外的共享文件）不得出现裸 tf 前缀判断——框架路由
  单源于 detect_framework / adapter；
- e2e 通路不消费 invoke._framework_of/_FRAMEWORK_ROOTS（onnx 名不在分类器
  注册表内，路由经 adapter 单源，grep 收口钉住）。
"""

import ast
from pathlib import Path

FRAMEWORK_API = Path(__file__).resolve().parents[3] / "ttk" / "core_modules" / "framework_api"
COMMON = Path(__file__).resolve().parents[3] / "ttk" / "core_modules" / "common"


def _py_files(root):
    return sorted(p for p in Path(root).rglob("*.py"))


def _scan_imports(path):
    """全文（含函数体内惰性 import）收集 import 依赖。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    abs_mods, rel_imports = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                abs_mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                abs_mods.add(node.module.split(".")[0])
            elif node.level:
                rel_imports.add((node.level, (node.module or "").split(".")[0] if node.module else ""))
    return abs_mods, rel_imports


def test_torch_and_tf_packages_do_not_import_each_other():
    """torch/ 与 tf/ 互不 import（含方法内惰性 import）。

    相对导入：level=1 只能是本包兄弟模块，level≥2 须避开对方包；
    绝对导入：不得出现 framework_api.{对方包} 路径（相对检查的补全——
    历史上此处曾是一段恒真断言死循环，绝对形态从未被真正检查过）。
    """
    torch_dir = FRAMEWORK_API / "torch"
    tf_dir = FRAMEWORK_API / "tf"
    for src, other in ((torch_dir, "tf"), (tf_dir, "torch")):
        for path in _py_files(src):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert f"framework_api.{other}" not in node.module, f"{path}: 绝对导入对方包 {node.module}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert f"framework_api.{other}" not in alias.name, f"{path}: 绝对导入对方包 {alias.name}"
            _, rels = _scan_imports(path)
            for level, module in rels:
                if level >= 2 and module == other:
                    raise AssertionError(f"{path} imports sibling framework package: {'.' * level}{module}")


def test_tf_package_top_level_avoids_torch():
    """tf/ 顶层（模块级）零 torch / torch_npu import；惰性 import 亦不得触达。"""
    for path in _py_files(FRAMEWORK_API / "tf"):
        abs_mods, _ = _scan_imports(path)
        assert "torch" not in abs_mods, f"{path.name} imports torch"
        assert "torch_npu" not in abs_mods, f"{path.name} imports torch_npu"


def test_onnx_package_avoids_torch_and_tf():
    """onnx/ 全树零 torch / torch_npu / tensorflow import（含方法内惰性）。

    onnx 域是 numpy 直通域：eager/golden 均不触框架 tensor；torch_npu 与
    onnx worker 内 libascendcl 的 ACL 生命周期互斥，torch 生态设施不得经
    onnx 链路进入 worker 进程。
    """
    for path in _py_files(FRAMEWORK_API / "onnx"):
        abs_mods, _ = _scan_imports(path)
        assert "torch" not in abs_mods, f"{path.name} imports torch"
        assert "torch_npu" not in abs_mods, f"{path.name} imports torch_npu"
        assert "tensorflow" not in abs_mods, f"{path.name} imports tensorflow"


def test_onnx_backend_zero_acl():
    """common/backend/onnx_backend.py 零 ACL：so 加载属通路层职责。

    ast 扫描代码（不受 docstring 说明文字误报）：无 ctypes import、无 CDLL
    调用、无通路层 import。设备探测走 DSMI（common/device 既有设施）。
    """
    path = COMMON / "backend" / "onnx_backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "ctypes", f"{path} imports ctypes"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "ctypes", f"{path} imports ctypes"
            assert "framework_api" not in node.module, f"{path} imports pathway module: {node.module}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "CDLL":
            raise AssertionError(f"{path} calls CDLL directly")


def test_common_does_not_import_pathway():
    """common/ 不得 import framework_api（依赖方向约束）。

    TF 的 L1 张量构造经 framework_hooks 注册表注入（通路侧主动注册），
    common 内不允许出现任何通路 import（含函数内惰性）。
    """
    for path in _py_files(COMMON):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "framework_api" in node.module:
                raise AssertionError(f"{path} imports pathway module: {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "framework_api" not in alias.name, f"{path} imports pathway module: {alias.name}"


def test_shared_layer_prefix_branches_confined():
    """通路共享层不得出现裸 tf 前缀判断（startswith(("tf." 等)）。

    允许清单：adapter.py（adapter 选择说明性引用）与 torch//tf/ 子包内的
    api_loader（加载器前缀语义）。共享文件一律走 detect_framework / adapter。
    """
    allowed = {"adapter.py", "api_model.py"}
    shared_files = [
        p
        for p in _py_files(FRAMEWORK_API)
        if p.parent == FRAMEWORK_API and p.name not in allowed and p.name != "__init__.py"
    ]
    for path in shared_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "startswith":
                if not node.args:
                    continue
                arg = node.args[0]
                vals = []
                if isinstance(arg, ast.Constant):
                    vals = [arg.value]
                elif isinstance(arg, (ast.Tuple, ast.List)):
                    vals = [e.value for e in arg.elts if isinstance(e, ast.Constant)]
                if any(isinstance(v, str) and v in ("tf.", "tensorflow.") for v in vals):
                    raise AssertionError(
                        f"{path.name}: raw framework-prefix branch — route via detect_framework/adapter"
                    )


def test_adapter_registry_covers_both_frameworks():
    """adapter 注册表覆盖 torch/tf 且 get_adapter 惰性隔离。"""
    from ttk.core_modules.framework_api.adapter import FrameworkAdapter, get_adapter

    torch_adapter = get_adapter("torch")
    tf_adapter = get_adapter("tf")
    assert isinstance(torch_adapter, FrameworkAdapter)
    assert isinstance(tf_adapter, FrameworkAdapter)
    assert torch_adapter is not tf_adapter
    assert torch_adapter.framework == "torch"
    assert tf_adapter.framework == "tf"


def test_adapter_mutable_hooks():
    """mutability 钩子的框架守卫：异框架名短路，不触达对方框架的解析器。"""
    from ttk.core_modules.framework_api.adapter import get_adapter

    torch_adapter = get_adapter("torch")
    # torch 适配器收到 TF 名 → 短路（不 import tensorflow）
    assert torch_adapter.is_mutable_api("tf.raw_ops.ResourceScatterAdd") is False
    assert torch_adapter.mutable_slots("tf.add") == (frozenset(), frozenset())
    assert torch_adapter.mutable_readback("torch.add", [], {}) == []
    torch_adapter.reset_mutable_inputs(object(), [object()], [None])  # no-op 不抛

    tf_adapter = get_adapter("tf")
    # tf 适配器收到 torch 名 → 短路（不 import torch 侧解析路径之外的任何 TF 触达）
    assert tf_adapter.is_mutable_api("torch.add") is False
    assert tf_adapter.mutable_slots("torch.ops.aten.add_") == (frozenset(), frozenset())


def test_relative_imports_resolve():
    """framework_api 全树相对导入目标可解析（防改名残留断链）。

    tf/graph_runner 曾残留 `from .tf_stateful import ...`（tf_stateful →
    tf/stateful 更名时漏改，函数内惰性 import 无人触达即静默潜伏，TF 图模式
    全部退化为 GRAPH_EXEC_FAILURE）。本断言按 ast 全量扫描（含函数体内惰性
    import）相对导入并核对目标模块文件存在。
    """
    for path in _py_files(FRAMEWORK_API):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level):
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                target = base.joinpath(*node.module.split("."))
                assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (
                    f"{path}: stale relative import {'.' * node.level}{node.module}"
                )
            else:
                # `from . import x`：x 须为本包子模块（属性再导出场景验包存在兜底）
                for alias in node.names:
                    sub = base / alias.name
                    if not (sub.with_suffix(".py").is_file() or (sub / "__init__.py").is_file()):
                        assert (base / "__init__.py").is_file(), (
                            f"{path}: stale relative import {'.' * node.level}{alias.name}"
                        )


def test_framework_api_does_not_consume_framework_of():
    """e2e 通路（含 onnx 链路）不消费 invoke._framework_of / _FRAMEWORK_ROOTS。

    框架归属分类器按调用方 root package 前缀匹配（numpy/torch/tf），无 onnx
    条目——onnx api（onnx.<OpType>）的执行/比对/装配走 adapter 路由，不经
    该分类器（grep 收口的结构化钉住；_framework_of 的消费方仅
    kernel/aclnn 的 golden 层，按各自通路 api_name 消费，onnx 名不可达）。
    """
    for path in _py_files(FRAMEWORK_API):
        text = path.read_text(encoding="utf-8")
        assert "_framework_of" not in text, f"{path}: e2e shared layer must not consume invoke._framework_of"
        assert "_FRAMEWORK_ROOTS" not in text, f"{path}: e2e shared layer must not reference _FRAMEWORK_ROOTS"


def test_onnx_om_is_runtime_output():
    """onnx_om 是运行产物目录（root_path 下、按用例归组）：源码树不追踪。"""
    from pathlib import Path as _Path

    gitignore = _Path(__file__).resolve().parents[3] / ".gitignore"
    assert "/onnx_om/" in gitignore.read_text(encoding="utf-8")


_RETIRED_RANK_STAGING_OP_MARKERS = (
    "MatmulReduceScatter",
    "AllGatherMatmul",
    "MatmulAllReduce",
    "MatmulAlltoAll",
    "AlltoAllMatmul",
    "QuantMatmulAllReduce",
    "GroupedMatMul",
    "AlltoAllvGrouped",
    "BatchMatMul",
    "npu_moe_distribute_combine",
    "bmm_reduce_scatter",
    "all_to_all_all_gather",
)


def test_worker_carries_no_rank_op_names():
    """多卡编排层（worker.py）零算子名：rank 输入算子特定语义在
    torch/rank_inputs.py，编排层不得再出现算子名清单。"""
    source = (FRAMEWORK_API / "worker.py").read_text(encoding="utf-8")
    leaked = [marker for marker in _RETIRED_RANK_STAGING_OP_MARKERS if marker in source]
    assert not leaked, f"operator names leaked into the multi-device orchestration path: {leaked}"
