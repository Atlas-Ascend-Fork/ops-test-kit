#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""--sig 签名源解析测试：py stub ast 合成（torch 无关）与 parse_schema 后端（torch 门控）。"""

import ast
import textwrap

import pytest

from ttk.core_modules.framework_api.torch.signature_parser import (
    _synthesize_func_line,
    parse_signature_source,
)


def _first_func(source):
    return [n for n in ast.parse(textwrap.dedent(source)).body if isinstance(n, ast.FunctionDef)][0]


# ── ast → aten func 行合成（无 torch 依赖） ──────────────────────────────────


def test_synthesize_kwonly_defaults_and_return_tuple():
    node = _first_func(
        """
        def myadd(x: torch.Tensor, y: torch.Tensor, *, bias: Optional[torch.Tensor] = None,
                  alpha: float = 1.0, layout: str = "BSND", flag: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            ...
        """
    )
    assert _synthesize_func_line(node) == (
        "myadd(Tensor x, Tensor y, *, Tensor? bias=None, float alpha=1.0, "
        'str layout="BSND", bool flag=False) -> (Tensor, Tensor)'
    )


def test_synthesize_annotation_forms():
    node = _first_func(
        """
        def forms(a: Tensor, b: Optional[int] = None, c: List[torch.Tensor] = None,
                  d: Union[int, float] = 1, e: "torch.Tensor" = None) -> torch.Tensor:
            ...
        """
    )
    line = _synthesize_func_line(node)
    assert "Tensor a" in line
    assert "int? b=None" in line
    assert "Tensor? e=None" in line
    assert "Scalar d=1" in line


def test_synthesize_optional_without_default_gets_none():
    node = _first_func(
        """
        def f(x: Optional[torch.Tensor]) -> torch.Tensor:
            ...
        """
    )
    assert _synthesize_func_line(node) == "f(Tensor? x=None) -> Tensor"


def test_synthesize_rejects_unannotated_param():
    """flashmla 类无注解 dispatcher：直接喂 op 源码报错，指明缺注解的参数。"""
    node = _first_func(
        """
        def sparse_flash_mla(q, ori_kv=None, softmax_scale=1.0):
            ...
        """
    )
    with pytest.raises(ValueError, match="parameter 'q' has no annotation"):
        _synthesize_func_line(node)


def test_synthesize_rejects_var_args_and_missing_return():
    with pytest.raises(ValueError, match="var-args"):
        _synthesize_func_line(_first_func("def f(x: Tensor, *args) -> Tensor: ..."))
    with pytest.raises(ValueError, match="missing return annotation"):
        _synthesize_func_line(_first_func("def f(x: Tensor): ..."))


def test_synthesize_rejects_unknown_annotation():
    with pytest.raises(ValueError, match="unknown annotation type"):
        _synthesize_func_line(_first_func("def f(x: torch.nn.Module) -> Tensor: ..."))


def test_synthesize_rejects_variadic_return_tuple():
    with pytest.raises(ValueError, match="variadic return tuple"):
        _synthesize_func_line(_first_func("def f(x: Tensor) -> Tuple[Tensor, ...]: ..."))


def test_parse_signature_source_rejects_non_stub_suffix():
    with pytest.raises(ValueError, match="expected .py/.pyi"):
        parse_signature_source("signatures.yaml")


# ── parse_schema 后端（torch 门控：无 torch 环境逐用例跳过） ──────────────────


def test_parse_signature_line_full_name_and_types():
    pytest.importorskip("torch")
    from ttk.core_modules.framework_api.torch.extract_schema import parse_signature_line

    name, params, return_count = parse_signature_line(
        "torch.ops.ttktest.myadd(Tensor x, Tensor y, *, float alpha=1.0) -> (Tensor, Tensor)"
    )
    assert name == "torch.ops.ttktest.myadd"
    assert return_count == 2
    assert [p.name for p in params] == ["x", "y", "alpha"]
    assert params[0].type == "Tensor"
    assert not params[0].is_optional
    assert params[2].type == "float"
    assert params[2].is_keyword_only
    assert params[2].default == 1.0


def test_parse_signature_file_registers_by_function_name(tmp_path):
    pytest.importorskip("torch")
    from ttk.core_modules.framework_api.torch.signature_parser import parse_signature_file

    stub = tmp_path / "stub.py"
    stub.write_text(
        "from typing import Optional, Tuple\n"
        "import torch\n"
        "\n"
        "\n"
        "def sparse_flash_mla(\n"
        "    q: torch.Tensor,\n"
        "    *,\n"
        "    ori_kv: Optional[torch.Tensor] = None,\n"
        '    layout_q: str = "BSND",\n'
        "    return_softmax_lse: bool = False,\n"
        ") -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    ...\n"
    )
    entries = parse_signature_file(stub)
    (info,) = entries.values()
    assert "sparse_flash_mla" in entries
    assert sum(1 for p in info.params if p.is_tensor) == 2
    assert info.overloads[0].return_count == 2
    assert info.source == f"signature:{stub}"


def test_parse_signature_file_skips_broken_defs(tmp_path, caplog):
    pytest.importorskip("torch")
    from ttk.core_modules.framework_api.torch.signature_parser import parse_signature_file

    stub = tmp_path / "mixed.py"
    stub.write_text(
        "import torch\n\n\ndef broken(q, kv=None):\n    ...\n\n\ndef good(q: torch.Tensor) -> torch.Tensor:\n    ...\n"
    )
    with caplog.at_level("WARNING"):
        entries = parse_signature_file(stub)
    assert list(entries) == ["good"]
    assert "parameter 'q' has no annotation" in caplog.text


def test_parse_signature_file_extracts_opbuilder_schema_strings(tmp_path, caplog):
    """算子 torch_extension 实际 py 可直接喂：无注解 dispatcher 跳过（告警），
    OpBuilder.schema() 列表里的 aten func 行字面量（相邻拼接）照常登记。
    """
    pytest.importorskip("torch")
    from ttk.core_modules.framework_api.torch.signature_parser import parse_signature_file

    stub = tmp_path / "op_extension.py"
    stub.write_text(
        "from typing import Optional\n"
        "import torch\n"
        "\n"
        "\n"
        "class SparseMlaOpBuilder:\n"
        "    def schema(self) -> list:\n"
        "        return [\n"
        '            "sparse_flash_mla(Tensor q, *, "\n'
        '            "Tensor? ori_kv=None, float softmax_scale=1.0) -> (Tensor, Tensor)",\n'
        '            "sparse_mla_meta(int n, *, Tensor? x=None) -> Tensor",\n'
        "        ]\n"
        "\n"
        "\n"
        "def sparse_flash_mla(q, ori_kv=None, softmax_scale=1.0):  # 无注解 dispatcher\n"
        "    ...\n"
        "\n"
        "\n"
        'PLAIN = "not a schema (x) but no arrow"\n'
        'DOC = """looks like f(x) -> y\n'
        'but multiline docstring, whole-constant 不匹配单行文法"""\n'
    )
    with caplog.at_level("WARNING"):
        entries = parse_signature_file(stub)

    assert set(entries) == {"sparse_flash_mla", "sparse_mla_meta"}
    info = entries["sparse_flash_mla"]
    assert [p.name for p in info.params] == ["q", "ori_kv", "softmax_scale"]
    assert info.overloads[0].return_count == 2
    assert "parameter 'q' has no annotation" in caplog.text
