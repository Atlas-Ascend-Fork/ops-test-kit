#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/plugin 接口测试。

覆盖公开面：get_plugin_function（双系统优先级 / 旧系统 __golden__·__input__
约定 / level 门控 / 路径形态 / miss）、SpecResolver（8 个 resolve）、
get_spec_class_meta、InvalidSpecError（索引期 + 校验期）、clear_plugin_cache。

fixture：examples/ 真实 spec 文件（golden 三形态、third_party、tolerance、
compare/pre_compare/customize_inputs）；无 example 的属性（torch_graph /
npu_preprocess）与旧系统 __input__ 用 tmp_path 合成。
"""

from pathlib import Path

import pytest

from ttk.core_modules.common.plugin import (
    InvalidSpecError,
    SpecResolver,
    get_plugin_function,
    get_spec_class_meta,
)
from ttk.core_modules.common.plugin.custom_plugin_scanner import PluginScanner
from ttk.core_modules.common.plugin.loader import clear_plugin_cache

_EXAMPLES = Path(__file__).resolve().parents[4] / "ttk" / "core_modules" / "common" / "plugin" / "examples"
_EXAMPLES_STR = str(_EXAMPLES)


class TestGetPluginFunction:
    def test_spec_priority_over_custom(self, tmp_path):
        """同 op 双系统并存：TestSpec 优先，旧 __golden__ 约定不接管。"""
        (tmp_path / "prio.py").write_text(
            "def custom_prio_golden(x, y, **kwargs):\n"
            "    return x + y\n"
            "\n"
            '__golden__ = {"kernel": {"prio_op": "custom_prio_golden"}}\n'
            "\n"
            "class PrioSpec:\n"
            "    def golden(x, y, **kwargs):\n"
            "        return x + y\n"
            "\n"
            '__spec__ = {"prio_op": "PrioSpec"}\n'
        )
        func = get_plugin_function("prio_op", "golden", "kernel", str(tmp_path))
        assert func is not None
        assert func.__qualname__.split(".")[0] == "PrioSpec"

    def test_custom_input_convention(self, tmp_path):
        """旧系统 __input__ 约定：plugin_type="input" 按 level 加载。"""
        (tmp_path / "inp.py").write_text(
            'def gen_inputs(x, **kwargs):\n    return x\n\n__input__ = {"kernel": {"custom_in_op": "gen_inputs"}}\n'
        )
        func = get_plugin_function("custom_in_op", "input", "kernel", str(tmp_path))
        assert func is not None
        assert func.__name__ == "gen_inputs"
        assert get_plugin_function("custom_in_op", "golden", "kernel", str(tmp_path)) is None

    def test_custom_golden_level_gating(self, tmp_path):
        """旧系统注册按 level 隔离：aclnn 注册不被 kernel 查询命中（回归）。"""
        (tmp_path / "levels.py").write_text(
            "def shared_golden(x, **kwargs):\n"
            "    return x\n"
            "\n"
            '__golden__ = {"aclnn": {"leak_op": "shared_golden"}, "kernel": {"kernel_op": "shared_golden"}}\n'
        )
        assert get_plugin_function("leak_op", "golden", "kernel", str(tmp_path)) is None
        assert get_plugin_function("kernel_op", "golden", "kernel", str(tmp_path)) is not None
        assert get_plugin_function("leak_op", "golden", "aclnn", str(tmp_path)) is not None

    def test_invalid_level_raises(self, tmp_path):
        """旧系统 level 域为 kernel/aclnn/e2e；越界 level 抛 ValueError。"""
        with pytest.raises(ValueError, match="Invalid level type"):
            get_plugin_function("any_op", "golden", "geir", str(tmp_path))

    @pytest.mark.parametrize(
        ("decl", "func_name"),
        [
            ("__golden__ = {REG}", "plain"),
            ("__golden__: dict = {REG}", "annotated"),
            ("alias = __golden__ = {REG}", "chained_not_first"),
            ("__golden__ = alias = {REG}", "chained_first"),
        ],
        ids=["plain-assign", "annotated-assign", "chained-not-first-target", "chained-first-target"],
    )
    def test_declaration_assignment_forms(self, tmp_path, decl, func_name):
        """注册声明的合法赋值形态均被发现：普通 / 类型注解 / 链式赋值（首/非首目标）。

        IDE 类型检查器会给注册字典自动加注解（AnnAssign），漏识别会使插件
        静默失效并回退默认 golden。
        """
        body = (
            "def plain(x, **kwargs):\n    return x\n\n"
            "def annotated(x, **kwargs):\n    return x\n\n"
            "def chained_not_first(x, **kwargs):\n    return x\n\n"
            "def chained_first(x, **kwargs):\n    return x\n\n"
        )
        reg = f'{{"kernel": {{"decl_op": "{func_name}"}}}}'
        (tmp_path / "decls.py").write_text(body + decl.replace("{REG}", reg) + "\n")
        func = get_plugin_function("decl_op", "golden", "kernel", str(tmp_path))
        assert func is not None
        assert func.__name__ == func_name

    @pytest.mark.parametrize(
        ("decl", "plugin_type", "field"),
        [
            ("__golden__: dict", "golden", "__golden__"),
            ('__golden__ = "not-a-dict"', "golden", "__golden__"),
            ("def helper():\n    __golden__ = {'kernel': {'decl_op': 'plain'}}\n", "golden", "__golden__"),
            ("async def helper():\n    __golden__ = {'kernel': {'decl_op': 'plain'}}\n", "golden", "__golden__"),
            ("class Holder:\n    __golden__ = {'kernel': {'decl_op': 'plain'}}\n", "golden", "__golden__"),
            ("def helper():\n    __input__ = {'kernel': {'decl_op': 'inp'}}\n", "input", "__input__"),
            ("async def helper():\n    __input__ = {'kernel': {'decl_op': 'inp'}}\n", "input", "__input__"),
            ("class Holder:\n    __input__ = {'kernel': {'decl_op': 'inp'}}\n", "input", "__input__"),
        ],
        ids=[
            "annotation-only",
            "non-dict-value",
            "local-golden",
            "async-local-golden",
            "class-golden",
            "local-input",
            "async-local-input",
            "class-input",
        ],
    )
    def test_declaration_ignored_forms(self, tmp_path, decl, plugin_type, field):
        """仅注解未赋值、非字典值与函数/异步函数/类作用域（新命名空间）声明均不注册。"""
        (tmp_path / "ignored.py").write_text(decl + "\n")
        scanner = PluginScanner(str(tmp_path), plugin_type, field, "kernel")
        assert scanner.scan() == {}

    def test_declaration_scan_does_not_execute_module(self, tmp_path):
        """AST 扫描阶段不执行插件模块（模块级副作用不触发，注册表仍命中）；
        模块级 if 分支内的声明同样命中。"""
        (tmp_path / "side_effect.py").write_text(
            'raise RuntimeError("must not execute")\n\n__golden__ = {"kernel": {"side_op": "plain"}}\n'
        )
        (tmp_path / "in_if.py").write_text('if True:\n    __golden__ = {"kernel": {"if_op": "plain"}}\n')
        scanner = PluginScanner(str(tmp_path), "golden", "__golden__", "kernel")
        result = scanner.scan()
        assert "side_op" in result
        assert "if_op" in result

    def test_plugin_path_forms_equivalent(self):
        """str / (Path,) / [str] 同目录输入等价且共享缓存条目。"""
        f1 = get_plugin_function("add", "golden", "kernel", _EXAMPLES_STR)
        f2 = get_plugin_function("add", "golden", "kernel", (_EXAMPLES,))
        f3 = get_plugin_function("add", "golden", "kernel", [str(_EXAMPLES)])
        assert f1 is not None
        assert f1 is f2 is f3

    def test_miss_returns_none(self):
        """无 spec 的 op 返回 None（builtin 兜底在流程层，不在本层）；无路径同。"""
        assert get_plugin_function("relu", "golden", "kernel", _EXAMPLES_STR) is None
        assert get_plugin_function("add", "golden", "kernel", None) is None

    def test_discovery_rules(self, tmp_path):
        """文件发现规则：__spec__ 优先于命名约定；_ 前缀文件跳过；语法错误文件跳过。"""
        (tmp_path / "conv.py").write_text(
            "class ConvOpTestSpec:\n"
            "    def golden(x, **kwargs):\n"
            "        return x\n"
            "\n"
            "class PrioConvTestSpec:\n"
            "    def golden(x, **kwargs):\n"
            "        return x\n"
        )
        (tmp_path / "indexed.py").write_text(
            "class IndexedSpec:\n"
            "    def golden(x, **kwargs):\n"
            "        return x\n"
            "\n"
            '__spec__ = {"prio_conv": "IndexedSpec"}\n'
        )
        (tmp_path / "_hidden.py").write_text(
            "class HiddenSpec:\n"
            "    def golden(x, **kwargs):\n"
            "        return x\n"
            "\n"
            '__spec__ = {"hidden_op": "HiddenSpec"}\n'
        )
        (tmp_path / "broken.py").write_text("def broken(:\n")

        # 命名约定兜底：无 __spec__ 时按 PascalCase+TestSpec 查类
        func = get_plugin_function("conv_op", "golden", "kernel", str(tmp_path))
        assert func is not None
        assert func.__qualname__.split(".")[0] == "ConvOpTestSpec"
        # __spec__ 索引优先于命名约定
        func = get_plugin_function("prio_conv", "golden", "kernel", str(tmp_path))
        assert func is not None
        assert func.__qualname__.split(".")[0] == "IndexedSpec"
        # _ 前缀文件不参与发现；语法错误文件被跳过且不影响兄弟文件
        assert get_plugin_function("hidden_op", "golden", "kernel", str(tmp_path)) is None


class TestSpecResolver:
    def test_resolve_golden_str_to_callable(self):
        """golden 字符串形态解析为 callable（"torch.abs" → torch.abs）。"""
        func = SpecResolver("abs_str", "kernel", _EXAMPLES_STR).resolve_golden()
        assert callable(func)
        assert func.__name__ == "abs"

    def test_resolve_golden_func_and_class_forms(self):
        """函数形态返回 callable；类形态原样返回类（由 invoke 层实例化）。"""
        func = SpecResolver("abs_func", "kernel", _EXAMPLES_STR).resolve_golden()
        assert callable(func)
        assert not isinstance(func, type)
        cls = SpecResolver("abs_class", "kernel", _EXAMPLES_STR).resolve_golden()
        assert isinstance(cls, type)
        assert cls.__name__ == "_AbsGolden"

    def test_resolve_golden_fallback(self):
        """未命中 spec：有 fallback 调 fallback()；无 fallback 返回 None。"""
        sentinel = object()
        resolver = SpecResolver("relu", "kernel", _EXAMPLES_STR)
        assert resolver.resolve_golden(fallback=lambda: sentinel) is sentinel
        assert resolver.resolve_golden() is None

    def test_resolve_pre_inputs(self):
        """input 查询名桥接到 customize_inputs 属性。"""
        func = SpecResolver("histogram_input", "kernel", _EXAMPLES_STR).resolve_pre_inputs()
        assert callable(func)

    def test_resolve_pre_compare_and_compare(self):
        resolver = SpecResolver("topk", "kernel", _EXAMPLES_STR)
        assert callable(resolver.resolve_pre_compare())
        assert callable(resolver.resolve_compare())

    def test_resolve_third_party_forms(self):
        """str 原样透传（不转 callable，provider 推导在消费方）；dict 透传。"""
        assert SpecResolver("softmax_simple", "kernel", _EXAMPLES_STR).resolve_third_party() == (
            "torch.nn.functional.softmax"
        )
        tp = SpecResolver("softmax_multi_vendor", "kernel", _EXAMPLES_STR).resolve_third_party()
        assert tp == {"torch": "torch.nn.functional.softmax", "tf": "tf.raw_ops.Softmax"}

    def test_resolve_tolerance(self):
        tolerance = SpecResolver("layer_norm_full", "kernel", _EXAMPLES_STR).resolve_tolerance()
        assert isinstance(tolerance, dict)
        assert "float32" in tolerance

    def test_resolve_torch_graph_and_npu_preprocess(self, tmp_path):
        """无 example 的属性用合成 spec：torch_graph 须 nn.Module 子类。"""
        from torch import nn

        (tmp_path / "extra.py").write_text(
            "import torch.nn as nn\n"
            "\n"
            "class GraphOpSpec:\n"
            "    class _Module(nn.Module):\n"
            "        def forward(self, x):\n"
            "            return x\n"
            "    torch_graph = _Module\n"
            "\n"
            "class PreprocOpSpec:\n"
            "    def npu_preprocess(*args, **kwargs):\n"
            "        return None\n"
            "\n"
            '__spec__ = {"graph_op": "GraphOpSpec", "preproc_op": "PreprocOpSpec"}\n'
        )
        graph = SpecResolver("graph_op", "e2e", str(tmp_path)).resolve_torch_graph()
        assert isinstance(graph, type)
        assert issubclass(graph, nn.Module)
        preproc = SpecResolver("preproc_op", "aclnn", str(tmp_path)).resolve_npu_preprocess()
        assert callable(preproc)


class TestSpecClassMeta:
    def test_meta_for_spec(self):
        meta = get_spec_class_meta("add", _EXAMPLES_STR)
        assert meta == {"spec_file": str(_EXAMPLES / "01_minimal_golden.py"), "class_name": "AddTestSpec"}

    def test_meta_miss(self):
        assert get_spec_class_meta("relu", _EXAMPLES_STR) is None
        assert get_spec_class_meta("add", None) is None


class TestValidation:
    def test_non_string_spec_value_raises(self, tmp_path):
        """__spec__ 值非字符串类名：索引期 InvalidSpecError（fail-fast）。"""
        (tmp_path / "bad_value.py").write_text('__spec__ = {"bad_value": 123}\n')
        with pytest.raises(InvalidSpecError):
            get_plugin_function("bad_value", "golden", "kernel", str(tmp_path))

    def test_bad_attr_type_raises(self, tmp_path):
        """spec 属性类型不符：首载校验抛 InvalidSpecError。"""
        (tmp_path / "bad_attr.py").write_text(
            'class BadAttrSpec:\n    golden = 42\n\n__spec__ = {"bad_attr": "BadAttrSpec"}\n'
        )
        with pytest.raises(InvalidSpecError, match="golden"):
            get_plugin_function("bad_attr", "golden", "kernel", str(tmp_path))

    def test_framework_tolerance_token_rejected(self, tmp_path):
        """框架增强 token（cosine）不得写入 Spec.tolerance，报错指路 --compare。"""
        (tmp_path / "bad_tol.py").write_text(
            "class BadTolSpec:\n"
            '    tolerance = {"float32": {"standard": "cosine"}}\n'
            "\n"
            '__spec__ = {"bad_tol": "BadTolSpec"}\n'
        )
        with pytest.raises(InvalidSpecError, match="--compare"):
            get_plugin_function("bad_tol", "tolerance", "kernel", str(tmp_path))


class TestQueryCache:
    def test_cache_and_clear(self, monkeypatch, tmp_path):
        """查询结果缓存；clear_plugin_cache 后重查反映新的底层查询。"""
        import ttk.core_modules.common.plugin.spec_manager as sm

        monkeypatch.setattr(sm, "get_spec_attr", lambda *a, **k: "v1")
        assert get_plugin_function("cache_probe", "third_party", "kernel", str(tmp_path)) == "v1"
        monkeypatch.setattr(sm, "get_spec_attr", lambda *a, **k: "v2")
        assert get_plugin_function("cache_probe", "third_party", "kernel", str(tmp_path)) == "v1"  # 命中缓存
        clear_plugin_cache()
        assert get_plugin_function("cache_probe", "third_party", "kernel", str(tmp_path)) == "v2"
