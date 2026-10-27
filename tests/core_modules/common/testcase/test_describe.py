# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""spec.describe 参数化用例：未知列收集 + {shapes, dtypes} 契约 + 四通路映射。"""

import pytest

from ttk.core_modules.common.testcase.describe import collect_describe_params
from ttk.core_modules.framework_api.testcase import TestcaseE2e
from ttk.switches import SWITCHES, get_global_storage, set_global_storage

_SPEC_BODY = """
__spec__ = {"describe_op": "DescribeSpec"}


class DescribeSpec:
    @staticmethod
    def describe(params):
        b, s, n, d = params["b"], params["s"], params["n"], params["d"]
        return {
            "shapes": ((b, s, n, d), (b, s, n, d)),
            "dtypes": ("bfloat16", "bfloat16"),
        }
"""


@pytest.fixture(autouse=True)
def _restore_global_storage():
    prev = get_global_storage()
    yield
    set_global_storage(prev)


def _switches(plugin_path):
    sw = SWITCHES()
    sw.plugin_path = str(plugin_path)
    return sw


def _write_spec(tmp_path, body=_SPEC_BODY):
    (tmp_path / "dspec.py").write_text(body)
    set_global_storage(_switches(tmp_path))


def _e2e_case(original_dict=None, **kwargs):
    case = TestcaseE2e()
    case.testcase_name = "tc"
    case.api_name = "describe_op"
    case.is_valid = True
    case.original_dict = original_dict or {
        "testcase_name": "tc",
        "api_name": "describe_op",
        "b": "2",
        "s": "4096",
        "n": "32",
        "d": "128",
    }
    for key, value in kwargs.items():
        setattr(case, key, value)
    return case


class TestCollectDescribeParams:
    def test_types_and_exclusions(self):
        case = TestcaseE2e()
        case.original_dict = {
            "testcase_name": "tc",
            "api_name": "torch.ops.x",
            "b": "2",
            "s": "4096",
            "ratio": "1.5",
            "block": "(128, 64)",
            "layout": "BSND",
            "quoted": "'BSND'",
            "empty": "",
            "none_cell": None,
        }
        params = collect_describe_params(case)
        assert params["b"] == 2
        assert params["s"] == 4096
        assert params["ratio"] == 1.5
        assert params["block"] == (128, 64)
        assert params["layout"] == "BSND"
        assert params["quoted"] == "BSND"
        assert "empty" not in params
        assert "none_cell" not in params
        assert "api_name" not in params
        assert "testcase_name" not in params

    def test_no_original_dict(self):
        case = TestcaseE2e()
        assert collect_describe_params(case) == {}


class TestApplyDescribeE2e:
    def test_describe_fills_shapes_dtypes(self, tmp_path):
        _write_spec(tmp_path)
        case = _e2e_case()
        case._apply_describe_shapes()
        assert case.tensor_view_shapes == ((2, 4096, 32, 128), (2, 4096, 32, 128))
        assert case.tensor_dtypes == ("bfloat16", "bfloat16")

    def test_csv_fields_not_overwritten(self, tmp_path):
        _write_spec(tmp_path)
        case = _e2e_case(
            original_dict={
                "testcase_name": "tc",
                "api_name": "describe_op",
                "b": "2",
                "s": "4096",
                "n": "32",
                "d": "128",
                "tensor_dtypes": "('float16', 'float16')",
                "attributes": '{"csv_wins": 1}',
            },
            tensor_dtypes=("float16", "float16"),
            attributes={"csv_wins": 1},
        )
        case._apply_describe_shapes()
        assert case.tensor_dtypes == ("float16", "float16")
        assert case.attributes == {"csv_wins": 1}
        assert case.tensor_view_shapes == ((2, 4096, 32, 128), (2, 4096, 32, 128))

    def test_extraneous_keys_warned_and_ignored(self, tmp_path, caplog):
        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "ExtraSpec"}\n\nclass ExtraSpec:\n    @staticmethod\n    def describe(params):\n        return {"shapes": ((2, 3),), "dtypes": ("float32",), "attributes": {"x": 1}}\n',
        )
        case = _e2e_case()
        case._apply_describe_shapes()
        assert case.tensor_view_shapes == ((2, 3),)
        assert case.attributes == {}
        assert "out of the {shapes, dtypes} contract" in caplog.text

    def test_existing_shapes_skip_describe(self, tmp_path, monkeypatch):
        from ttk.core_modules.common.plugin.resolver import SpecResolver

        _write_spec(tmp_path)
        called = {"hit": False}
        real = SpecResolver.resolve_describe

        def fake(self):
            called["hit"] = True
            return real(self)

        monkeypatch.setattr("ttk.core_modules.common.plugin.resolver.SpecResolver.resolve_describe", fake)
        case = _e2e_case(tensor_view_shapes=((4, 4),), tensor_dtypes=("float32",))
        case._apply_describe_shapes()
        assert called["hit"] is False
        assert case.tensor_view_shapes == ((4, 4),)
        assert case.tensor_dtypes == ("float32",)

    def test_shapes_given_dtypes_missing_fills_dtypes_only(self, tmp_path):
        """shapes/dtypes 不强绑定：CSV 给了 shapes 没给 dtypes → 只回填 dtypes。"""
        _write_spec(tmp_path)
        case = _e2e_case(
            original_dict={
                "testcase_name": "tc",
                "api_name": "describe_op",
                "b": "2",
                "s": "4096",
                "n": "32",
                "d": "128",
                "tensor_view_shapes": "((4, 4), (4, 4))",
            },
            tensor_view_shapes=((4, 4), (4, 4)),
        )
        case._apply_describe_shapes()
        assert case.tensor_view_shapes == ((4, 4), (4, 4))
        assert case.tensor_dtypes == ("bfloat16", "bfloat16")

    def test_dtypes_given_shapes_missing_fills_shapes_only(self, tmp_path):
        """CSV 给了 dtypes 没给 shapes → 只回填 shapes，dtypes 保持 CSV 值。"""
        _write_spec(tmp_path)
        case = _e2e_case(
            original_dict={
                "testcase_name": "tc",
                "api_name": "describe_op",
                "b": "2",
                "s": "4096",
                "n": "32",
                "d": "128",
                "tensor_dtypes": "('float16', 'float16')",
            },
            tensor_dtypes=("float16", "float16"),
        )
        case._apply_describe_shapes()
        assert case.tensor_view_shapes == ((2, 4096, 32, 128), (2, 4096, 32, 128))
        assert case.tensor_dtypes == ("float16", "float16")

    def test_no_unknown_columns_skip_describe(self, tmp_path, monkeypatch):
        from ttk.core_modules.common.plugin.resolver import SpecResolver

        _write_spec(tmp_path)
        called = {"hit": False}
        real = SpecResolver.resolve_describe

        def fake(self):
            called["hit"] = True
            return real(self)

        monkeypatch.setattr("ttk.core_modules.common.plugin.resolver.SpecResolver.resolve_describe", fake)
        case = _e2e_case(original_dict={"testcase_name": "tc", "api_name": "describe_op"})
        case._apply_describe_shapes()
        assert called["hit"] is False

    def test_no_describe_function_keeps_semantics(self, tmp_path):
        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "NoDescribeSpec"}\n\nclass NoDescribeSpec:\n    pass\n',
        )
        case = _e2e_case()
        case._apply_describe_shapes()
        assert case.is_valid is True
        assert case.tensor_view_shapes is None

    def test_describe_exception_invalidates(self, tmp_path):
        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "BoomSpec"}\n\nclass BoomSpec:\n    @staticmethod\n    def describe(params):\n        raise ValueError("bad params")\n',
        )
        case = _e2e_case()
        case._apply_describe_shapes()
        assert case.is_valid is False
        assert case.fail_reason.startswith("DESCRIBE_FAILED")

    def test_describe_non_dict_invalidates(self, tmp_path):
        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "ListSpec"}\n\nclass ListSpec:\n    @staticmethod\n    def describe(params):\n        return [1, 2]\n',
        )
        case = _e2e_case()
        case._apply_describe_shapes()
        assert case.is_valid is False
        assert case.fail_reason == "DESCRIBE_RESULT_NOT_DICT"

    def test_missing_shapes_key_invalidates(self, tmp_path):
        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "NoShapesSpec"}\n\nclass NoShapesSpec:\n    @staticmethod\n    def describe(params):\n        return {"dtypes": ("float32",)}\n',
        )
        case = _e2e_case()
        case._apply_describe_shapes()
        assert case.is_valid is False
        assert case.fail_reason == "DESCRIBE_MISSING_SHAPES"


class TestApplyDescribeKernel:
    def _kernel_case(self, original_dict=None, **kwargs):
        from ttk.core_modules.kernel.testcase import TestcaseKernel

        case = TestcaseKernel()
        case.testcase_name = "tc"
        case.op_name = "describe_op"
        case.is_valid = True
        case.original_dict = original_dict or {
            "testcase_name": "tc",
            "op_name": "describe_op",
            "b": "2",
            "s": "4096",
            "n": "32",
            "d": "128",
        }
        for key, value in kwargs.items():
            setattr(case, key, value)
        return case

    def test_kernel_maps_input_fields(self, tmp_path):
        _write_spec(tmp_path)
        case = self._kernel_case()
        case._apply_describe_shapes()
        assert case.input_shapes == ((2, 4096, 32, 128), (2, 4096, 32, 128))
        assert case.input_dtypes == ("bfloat16", "bfloat16")

    def test_geir_uses_geir_pathway(self, tmp_path, monkeypatch):
        from ttk.core_modules.common.plugin.resolver import SpecResolver
        from ttk.core_modules.geir.testcase import GeirTestcase

        _write_spec(tmp_path)
        captured = {}
        real = SpecResolver.resolve_describe

        def fake(self):
            captured["pathway"] = self._pathway
            return real(self)

        monkeypatch.setattr("ttk.core_modules.common.plugin.resolver.SpecResolver.resolve_describe", fake)
        case = GeirTestcase()
        case.testcase_name = "tc"
        case.op_name = "describe_op"
        case.is_valid = True
        case.original_dict = {"testcase_name": "tc", "op_name": "describe_op", "b": "2", "s": "8", "n": "2", "d": "4"}
        case._apply_describe_shapes()
        assert captured["pathway"] == "geir"
        assert case.input_shapes == ((2, 8, 2, 4), (2, 8, 2, 4))

    def test_kernel_existing_shapes_skip(self, tmp_path, monkeypatch):
        from ttk.core_modules.common.plugin.resolver import SpecResolver

        _write_spec(tmp_path)
        called = {"hit": False}
        real = SpecResolver.resolve_describe

        def fake(self):
            called["hit"] = True
            return real(self)

        monkeypatch.setattr("ttk.core_modules.common.plugin.resolver.SpecResolver.resolve_describe", fake)
        case = self._kernel_case(input_shapes=((4, 4),), input_dtypes=("float32",))
        case._apply_describe_shapes()
        assert called["hit"] is False

    def test_kernel_shapes_given_dtypes_filled(self, tmp_path):
        """kernel：CSV 给了 input_shapes 没给 input_dtypes → 只回填 dtypes。"""
        _write_spec(tmp_path)
        case = self._kernel_case(
            original_dict={
                "testcase_name": "tc",
                "op_name": "describe_op",
                "b": "2",
                "s": "8",
                "n": "2",
                "d": "4",
                "input_shapes": "((4, 4), (4, 4))",
            },
            input_shapes=((4, 4), (4, 4)),
        )
        case._apply_describe_shapes()
        assert case.input_shapes == ((4, 4), (4, 4))
        assert case.input_dtypes == ("bfloat16", "bfloat16")


class TestApplyDescribeAclnn:
    def test_aclnn_maps_full_param_list(self, tmp_path, monkeypatch):
        from ttk.core_modules.aclnn.testcase import TestcaseAclnn
        from ttk.core_modules.common.plugin.resolver import SpecResolver

        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "AclnnSpec"}\n\nclass AclnnSpec:\n    @staticmethod\n    def describe(params):\n        return {"shapes": ((2, 3), (3, 5)), "dtypes": ("float32", "float32")}\n',
        )
        captured = {}
        real = SpecResolver.resolve_describe

        def fake(self):
            captured["pathway"] = self._pathway
            return real(self)

        monkeypatch.setattr("ttk.core_modules.common.plugin.resolver.SpecResolver.resolve_describe", fake)
        case = TestcaseAclnn()
        case.testcase_name = "tc"
        case.api_name = "describe_op"
        case.is_valid = True
        case.original_dict = {"testcase_name": "tc", "api_name": "describe_op", "b": "2"}
        case._apply_describe_shapes()
        assert captured["pathway"] == "aclnn"
        assert case.tensor_view_shapes == ((2, 3), (3, 5))
        assert case.tensor_dtypes == ("float32", "float32")


class TestDescribeFactoryWiring:
    """CSV 无输入形状列 → factory 解析成行 → validate 期 describe 回填。"""

    def test_parameterized_e2e_row_fills_shapes(self, tmp_path):
        from ttk.core_modules.common.testcase.factory import UniversalTestcaseFactory

        _write_spec(tmp_path)
        header = ["testcase_name", "api_name", "b", "s", "n", "d", "attributes"]
        rows = [["tc1", "describe_op", "2", "4096", "32", "128", "{'alpha': 1.0}"]]
        case = UniversalTestcaseFactory((header, rows), TestcaseE2e).get()[0]
        assert case.tensor_view_shapes == ((2, 4096, 32, 128), (2, 4096, 32, 128))
        assert case.tensor_dtypes == ("bfloat16", "bfloat16")
        assert case.attributes == {"alpha": 1.0}
        assert case.fail_reason != "tensor_view_shapes is empty"

    def test_parameterized_kernel_row_fills_inputs(self, tmp_path):
        from ttk.core_modules.common.testcase.factory import UniversalTestcaseFactory
        from ttk.core_modules.kernel.testcase import TestcaseKernel

        _write_spec(tmp_path)
        header = ["testcase_name", "op_name", "b", "s", "n", "d", "output_shapes", "output_dtypes"]
        rows = [
            [
                "tc1",
                "describe_op",
                "2",
                "4096",
                "32",
                "128",
                "((2, 4096, 32, 128), (2, 4096, 32, 128))",
                "('float32', 'float32')",
            ]
        ]
        factory = UniversalTestcaseFactory((header, rows), TestcaseKernel, skip_validate=True)
        case = factory.get()[0]
        case._apply_describe_shapes()
        assert case.input_shapes == ((2, 4096, 32, 128), (2, 4096, 32, 128))
        assert case.input_dtypes == ("bfloat16", "bfloat16")

    def test_parameterized_row_without_describe_stays_invalid(self, tmp_path):
        from ttk.core_modules.common.testcase.factory import UniversalTestcaseFactory

        _write_spec(
            tmp_path,
            body='__spec__ = {"describe_op": "NoDescribeSpec"}\n\nclass NoDescribeSpec:\n    pass\n',
        )
        header = ["testcase_name", "api_name", "b", "s"]
        rows = [["tc1", "describe_op", "2", "8"]]
        case = UniversalTestcaseFactory((header, rows), TestcaseE2e).get()[0]
        assert case.tensor_view_shapes == ()
        assert case.fail_reason == "tensor_view_shapes is empty"
