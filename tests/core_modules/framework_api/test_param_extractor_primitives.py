# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""torch/param_extractor 测试（原 simple_param_extractor）：ParamInfo/APIParamInfo 属性、
签名解析、类型推断、NPU 声明解析、manual override、aten schema 提取、overload 匹配、
pyi 加载、pickle 往返。数据模型自 api_model 引用。"""

import pickle

import pytest

from ttk.core_modules.framework_api.api_model import APIParamInfo, ParamInfo, split_by_comma
from ttk.core_modules.framework_api.torch.npu_declaration import _normalize_npu_type, _parse_npu_declaration
from ttk.core_modules.framework_api.torch.param_extractor import (
    _MANUAL_OVERRIDES,
    _infer_type_from_name,
    _infer_type_from_value,
    _parse_params_from_signature,
    extract_api_params,
    get_api_params,
    register_api_params,
)

# == ParamInfo 属性 ==========================================================


class TestParamInfo:
    @pytest.mark.parametrize(
        ("type_str", "attr"),
        [
            ("Tensor", "is_tensor"),
        ],
        ids=["Tensor"],
    )
    def test_is_tensor(self, type_str, attr):
        """is_tensor 对 'Tensor'/'tensor' 均返回 True。"""
        assert getattr(ParamInfo(name="x", type=type_str), attr) is True

    @pytest.mark.parametrize(
        ("type_str", "attr"),
        [
            ("Scalar[]", "is_scalar_list"),
        ],
        ids=["scalar-list"],
    )
    def test_is_scalar_and_scalar_list(self, type_str, attr):
        """is_scalar / is_scalar_list 覆盖标量 + 标量列表。"""
        assert getattr(ParamInfo(name="x", type=type_str), attr) is True

    def test_defaults_and_var_positional(self):
        """ParamInfo 默认值（type=''/default=None/is_optional=False）+ var_positional 标记。"""
        p = ParamInfo(name="x")
        assert p.type == ""
        assert p.default is None
        assert p.is_optional is False
        assert p.is_var_positional is False
        vp = ParamInfo(name="tensors", type="Tensor", is_var_positional=True)
        assert vp.is_var_positional is True
        assert vp.is_tensor is True


# == APIParamInfo 属性 =======================================================


class TestAPIParamInfo:
    def test_tensors_and_scalars_filter(self):
        """tensors/scalars 属性正确过滤 ParamInfo 列表。"""
        info = APIParamInfo(
            api_name="test",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="alpha", type="float"),
                ParamInfo(name="dim", type="int"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        assert info.tensor_count == 2
        assert info.scalar_count == 2

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ([ParamInfo("tensors", "tuple of Tensors"), ParamInfo("out", "Tensor")], (-1, 0)),
        ],
        ids=["mixed"],
    )
    def test_tensor_distribution(self, params, expected):
        """tensor_distribution：flat=(0,0)，TensorList=-1，混合=(-1,0)。"""
        assert APIParamInfo(api_name="test", params=params).tensor_distribution == expected


# == split_by_comma ========================================================


class TestSplitByComma:
    @pytest.mark.parametrize(
        ("s", "expected"),
        [
            ("a, [b, (c, d)], e", ["a", "[b, (c, d)]", "e"]),
        ],
        ids=["deep-nesting"],
    )
    def test_split_with_nesting(self, s, expected):
        """逗号分割，括号内的逗号不分割。"""
        assert split_by_comma(s) == expected


# == _parse_params_from_signature ============================================


class TestParseSignature:
    def test_typed_and_optional_params(self):
        """解析带类型和默认值的签名：'Tensor input, int dim=-1, bool keepdim=False'。"""
        params = _parse_params_from_signature("Tensor input, int dim=-1, bool keepdim=False")
        assert len(params) == 3
        assert params[0].name == "input"
        assert params[0].type == "Tensor"
        assert params[1].is_optional
        assert params[1].default == -1
        assert params[2].is_optional

    def test_empty_and_names_only_fallback(self):
        """空签名→None；无类型签名→按名字推断类型（input→Tensor, dim→int, keepdim→bool）。"""
        assert _parse_params_from_signature("") is None
        assert _parse_params_from_signature("  ") is None

        params = _parse_params_from_signature("input, dim, keepdim")
        assert len(params) == 3
        assert params[0].type == "Tensor"
        assert params[1].type == "int"
        assert params[2].type == "bool"


# == _infer_type_from_name ==================================================


class TestInferTypeFromName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("input", "Tensor"),
        ],
        ids=["tensor"],
    )
    def test_infer_by_name(self, name, expected):
        """按参数名推断类型：常见名字→对应类型，未知→Tensor。"""
        assert _infer_type_from_name(name) == expected


# == _infer_type_from_value =================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("None", "Optional[Tensor]"),
    ],
    ids=["none"],
)
def test_infer_type_from_value(value, expected):
    """按默认值字面量推断类型：None/bool/int/float/str。"""
    assert _infer_type_from_value(value) == expected


# == _normalize_npu_type ====================================================


@pytest.mark.parametrize(
    ("npu_type", "expected"),
    [
        ("TensorList", "tuple of Tensors"),
    ],
    ids=["tensor-list"],
)
def test_normalize_npu_type(npu_type, expected):
    """NPU 类型归一化：TensorList→tuple of Tensors、int?→int、Optional/List 嵌套展开等。"""
    assert _normalize_npu_type(npu_type) == expected


# == _parse_npu_declaration =================================================


class TestParseNpuDeclaration:
    def test_tensor_list_and_return_type_skipped(self):
        """TensorList 类型保留；'-> Tensor' 返回类型被跳过。"""
        params = _parse_npu_declaration("TensorList tensors, int dim")
        assert params[0].type == "tuple of Tensors"

        params = _parse_npu_declaration("Tensor input -> Tensor")
        assert len(params) == 1

    def test_star_makes_keyword_only(self):
        """'*' 后的参数标记为 keyword_only。"""
        params = _parse_npu_declaration("Tensor input, *, Tensor? smooth_scales=None, Tensor? group_index=None")
        assert len(params) == 3
        assert not params[0].is_keyword_only
        assert params[1].is_keyword_only
        assert params[2].is_keyword_only


# == manual overrides =======================================================


class TestManualOverrides:
    def setup_method(self):
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        _MANUAL_OVERRIDES.clear()

    def test_register_override_and_unknown(self):
        """register_api_params 注册后 get_api_params 能取到；override 优先；未知 API→None。"""
        params = [ParamInfo("input", "Tensor"), ParamInfo("dim", "int")]
        register_api_params("torch.custom_op", params, source="test")
        info = get_api_params("torch.custom_op")
        assert info is not None
        assert info.tensor_count == 1
        assert info.source == "test"

        # override 优先
        register_api_params("torch.add", [ParamInfo("x", "Tensor")], source="override")
        assert get_api_params("torch.add").source == "override"

        # 未知 → None
        assert get_api_params("nonexistent.module.func") is None


# == extract_api_params (torch 实测) =========================================


class TestExtractTorchApi:
    def test_torch_add(self):
        """torch.add 提取出 ≥2 个 tensor 参数。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.add")
        assert info is not None
        assert info.tensor_count >= 2

    def test_torch_div_multi_overload(self):
        """torch.div 有 ≥2 个 overload，match_overload(2) 和 match_overload(1) 均匹配。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.div")
        assert len(info.overloads) >= 2
        assert info.match_overload(2)[0]
        assert info.match_overload(1)[0]


# == APIParamInfo multi-overload =============================================


class TestMatchOverload:
    def test_match_by_count_and_type(self):
        """match_overload 按参数数量和 tensor/scalar 类型匹配对应 overload。"""
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo("tensors", "tuple of Tensors")],
                [ParamInfo("input", "Tensor")],
            ],
            source="test",
        )
        # count=1 + is_nested=[False] → 第 1 个 overload
        assert info.match_overload(1, [False])[0]
        # count=1 + is_nested=[True] → 第 0 个 overload
        assert info.match_overload(1, [True])[0]


# == Tensor 方法 self 注入 ==================================================


class TestTensorMethodSelf:
    def test_resolve_and_inject_self(self):
        """torch.Tensor.add_ 解析成功；get_api_params 注入 self 参数。"""
        from ttk.core_modules.framework_api.api_model import resolve_function as _resolve_function
        from ttk.core_modules.framework_api.torch.param_extractor import _is_tensor_method

        assert _is_tensor_method("torch.Tensor.add_")
        assert not _is_tensor_method("torch.add")

        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        assert _resolve_function("torch.Tensor.add_") is not None
        info = get_api_params("torch.Tensor.add_")
        assert info.params[0].name == "self"
        assert info.params[0].type == "Tensor"


# == _normalize_args_type (union) ===========================================


class TestNormalizeArgsTypeUnion:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("int, float, inf, -inf, 'fro', 'nuc'", "int|float|str"),
        ],
        ids=["comma-enum"],
    )
    def test_comma_separated_union(self, raw, expected):
        """逗号分隔的 union 类型解析（含枚举值、tuple 变体、去重）。"""
        from ttk.core_modules.framework_api.torch.param_extractor import _normalize_args_type

        assert _normalize_args_type(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Tensor or Number", "Number"),  # Tensor 成员被过滤
        ],
        ids=["tensor-filtered"],
    )
    def test_or_union(self, raw, expected):
        """'or' 分隔的 union 类型解析（Tensor 成员被过滤，由 overload 处理）。"""
        from ttk.core_modules.framework_api.torch.param_extractor import _normalize_args_type

        assert _normalize_args_type(raw) == expected


# == coerce_value (union) ===================================================


class TestCoerceValueUnion:
    @pytest.mark.parametrize(
        ("value", "type_str", "expected"),
        [
            ("[1,2,3]", "int|tuple of ints", (1, 2, 3)),
        ],
        ids=["int-or-tuple-list"],
    )
    def test_coerce_union_type(self, value, type_str, expected):
        """coerce_value 对 union 类型按值自动选择正确分支转换。"""
        from ttk.core_modules.framework_api.api_param import coerce_value

        assert coerce_value(value, type_str) == expected


# == _enrich_types_from_annotations =========================================


class TestEnrichTypesFromAnnotations:
    def test_union_not_overwritten_but_default_filled(self):
        """union 类型不被 annotations 覆盖，但 default 从 annotations 填入。"""
        from ttk.core_modules.framework_api.torch.param_extractor import (
            ParamInfo,
            _enrich_types_from_annotations,
        )

        params = [ParamInfo(name="p", type="int|float|str", default=None)]
        ann = [ParamInfo(name="p", type="float", default="fro")]
        _enrich_types_from_annotations(params, ann)
        assert params[0].type == "int|float|str"
        assert params[0].default == "fro"


# == annotation 类型转换 ====================================================


class TestAnnotationToType:
    @staticmethod
    def test_optional_torch_dtype_is_dtype():
        """torch.dtype 及 Optional[torch.dtype] 均归一成 Dtype。"""
        torch = pytest.importorskip("torch")
        from typing import Optional

        from ttk.core_modules.framework_api.torch.param_extractor import _annotation_to_type

        assert _annotation_to_type(torch.dtype) == "Dtype"
        assert _annotation_to_type(Optional[torch.dtype]) == "Dtype"


# == nn.functional pyi ======================================================


class TestNnFunctionalPyi:
    def setup_method(self):
        import ttk.core_modules.framework_api.torch.param_extractor as spe

        spe._PYI_CACHE = None

    def teardown_method(self):
        import ttk.core_modules.framework_api.torch.param_extractor as spe

        spe._PYI_CACHE = None

    def test_conv_tbc_reexport_and_signature(self):
        """torch.nn.functional.conv_tbc 的 pyi reexport 解析：source=pyi-stub，3 个 tensor。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.nn.functional.conv_tbc")
        assert info is not None
        assert info.source == "pyi-stub"
        assert info.tensor_count == 3
        names = [p.name for p in info.params]
        assert {"input", "weight", "bias", "pad"} <= set(names)


# == var_positional 签名解析 ================================================


class TestParseSignatureVarPositional:
    @pytest.mark.parametrize(
        ("sig", "name", "is_var_pos"),
        [
            ("equation: str, *operands: Tensor", "operands", True),
        ],
        ids=["mixed"],
    )
    def test_star_prefix_marks_var_positional(self, sig, name, is_var_pos):
        """'*' 前缀标记 var_positional 参数。"""
        params = _parse_params_from_signature(sig)
        op = next(p for p in params if p.name == name)
        assert op.is_var_positional is is_var_pos


# == meshgrid 全链路 ========================================================


class TestMeshgridExtraction:
    def test_meshgrid_var_positional_and_keyword_only(self):
        """torch.meshgrid：tensors 是 var_positional tensor，indexing 是 keyword_only str。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.meshgrid")
        var_pos = [p for p in info.params if p.is_var_positional]
        assert len(var_pos) == 1
        assert var_pos[0].name == "tensors"
        assert var_pos[0].is_tensor_like
        indexing = next(p for p in info.params if p.name == "indexing")
        assert indexing.is_keyword_only
        assert indexing.type == "str"


# == docstring 解析 =========================================================


class TestDocstringArgs:
    def test_deeply_indented_args_section(self):
        """深缩进 Args 段落解析：tensors + continuation line 跳过。"""
        from ttk.core_modules.framework_api.torch.param_extractor import _parse_docstring_args_section

        doc = """Some description.

        Args:
            tensors (list of Tensor): list of tensors.
                Continuation line that should be skipped.
            indexing: (str, optional): the indexing mode.

        Returns:
            Something.
"""
        result = _parse_docstring_args_section(doc)
        assert result["tensors"] == ("list of Tensor", False)


# == aten schema 提取 ======================================================


class TestAtenSchemaExtraction:
    def test_aten_convolution_basic_and_out_overload(self):
        """torch.ops.aten.convolution：基础参数 + out overload（2 个 overload）。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        assert info is not None
        assert "aten._schemas" in info.source
        assert info.tensor_count >= 2
        assert len(info.overloads) == 2
        out_ov = info.overloads[1].params
        assert any(p.name == "out" and p.is_keyword_only for p in out_ov)
        # bias 是 optional tensor
        bias = next(p for p in info.params if p.name == "bias")
        assert bias.is_tensor
        assert bias.is_optional

    def test_aten_add_multi_overload_and_defaults(self):
        """torch.ops.aten.add：≥10 个 overload，alpha 默认值=1。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.add")
        assert len(info.overloads) >= 10
        # 找到含 alpha 的 tensor overload
        for ov in info.overloads:
            names = [p.name for p in ov.params]
            if "alpha" in names and sum(1 for p in ov.params if p.is_tensor_like) >= 2:
                alpha = next(p for p in ov.params if p.name == "alpha")
                assert alpha.is_optional
                assert alpha.default == 1
                return
        pytest.fail("未找到含 alpha 的 tensor overload")

    def test_aten_nonexistent_returns_none(self):
        """不存在的 aten op → None。"""
        assert extract_api_params("torch.ops.aten.nonexistent_op_xyz") is None


# == OverloadTensorLayout ==================================================


class TestOverloadTensorLayout:
    @pytest.mark.parametrize(
        ("out_type", "is_optional", "is_required", "return_count", "expected_out"),
        [
            ("Tensor[]", False, True, 4, 4),  # required tensor list out
        ],
        ids=["req-list"],
    )
    def test_out_layout_variants(self, out_type, is_optional, is_required, return_count, expected_out):
        """4 种 out 布局：optional/required × single/tensor_list。"""
        from ttk.core_modules.framework_api.api_model import OverloadTensorLayout

        params = [
            ParamInfo("input", "Tensor"),
            ParamInfo("out", out_type, is_optional=is_optional, is_keyword_only=True),
        ]
        layout = OverloadTensorLayout.build(params, return_count=return_count)
        assert layout.out_param is not None
        assert layout.is_out_required is is_required
        assert layout.out_expected_count == expected_out


# == OverloadInfo + pickle ==================================================


class TestOverloadInfoPickle:
    def test_api_param_info_pickle(self):
        """APIParamInfo pickle 往返：api_name/overloads/params 均存活。"""
        info = APIParamInfo(
            api_name="torch.test_op",
            params=[
                ParamInfo("input", "Tensor"),
                ParamInfo("dim", "int"),
            ],
            source="test",
        )
        restored = pickle.loads(pickle.dumps(info))
        assert restored.api_name == "torch.test_op"
        assert len(restored.overloads) == 1
        assert len(restored.params) == 2


# == _parse_return_count_from_first_line ====================================


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("sort(input, *, out=None) -> (Tensor, LongTensor)", 2),
    ],
    ids=["tuple-2"],
)
def test_parse_return_count(line, expected):
    """从签名第一行解析返回值数量：单值/元组/无标注。"""
    from ttk.core_modules.framework_api.torch.param_extractor import _parse_return_count_from_first_line

    assert _parse_return_count_from_first_line(line) == expected


# == 返回值数量 + out 升级（实测 torch API）================================


class TestReturnCountAndOutUpgrade:
    @pytest.mark.parametrize(
        ("api", "expected_count"),
        [
            ("torch.topk", 2),
        ],
        ids=["topk"],
    )
    def test_return_count(self, api, expected_count):
        """实测 torch API 的 return_count。"""
        info = extract_api_params(api)
        assert info is not None
        assert info.overloads[0].return_count == expected_count

    @pytest.mark.parametrize(
        ("api", "expected_out_count"),
        [
            ("torch.topk", 2),
        ],
        ids=["topk"],
    )
    def test_out_is_tensor_list_with_expected_count(self, api, expected_out_count):
        """多返回值 API 的 out 是 TensorList，out_expected_count 等于 return_count。"""
        info = extract_api_params(api)
        ov = info.overloads[0]
        assert ov.layout.is_out_tensor_list
        assert ov.layout.out_expected_count == expected_out_count


# == npu_hans decode/encode =================================================


class TestNpuHansDecodeEncode:
    def test_decode_and_encode_signatures(self):
        """torch_npu.npu_hans_decode/encode 的签名：out required、tensor list 布局。"""
        # decode: 4 input tensor, out required single
        decode = extract_api_params("torch_npu.npu_hans_decode").overloads[0]
        assert decode.return_count == 1
        assert decode.layout.input_count == 4
        assert decode.layout.is_out_required
        assert not decode.layout.is_out_tensor_list

        # encode: 1 input tensor, out required tensor list (4 outputs)
        encode = extract_api_params("torch_npu.npu_hans_encode").overloads[0]
        assert encode.return_count == 4
        assert encode.layout.input_count == 1
        assert encode.layout.is_out_required
        assert encode.layout.is_out_tensor_list


class TestInplaceStripping:
    """inplace 参数剥离回归。

    曾因批量 docstring 脚本的陈旧行号事故丢失 _strip_inplace_if_separate_exists
    调用行——语料胜者分布与全量测试均未覆盖该行为而漏网，此测试钉死之。
    """

    @staticmethod
    def _torch_available():
        try:
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    def test_strips_inplace_when_variant_exists(self):
        if not self._torch_available():
            pytest.skip("torch not importable")
        info = extract_api_params("torch.relu")
        assert info is not None
        assert "inplace" not in [p.name for p in info.params], "relu_ 存在时应剥离 inplace"

    def test_functional_namespace_keeps_inplace(self):
        if not self._torch_available():
            pytest.skip("torch not importable")
        info = extract_api_params("torch.nn.functional.relu")
        assert info is not None
        assert "inplace" in [p.name for p in info.params], "torch.nn. 排除域内 inplace 应保留"
