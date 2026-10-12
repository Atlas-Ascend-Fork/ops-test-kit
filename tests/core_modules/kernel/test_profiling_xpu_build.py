# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Unit tests for ttk.remote.client app-layer spec resolution.

Covers client.build_spec (Task 7). The extract_spec_providers test is covered by
test_profiling_config.py. No server fixture.
"""

from ttk.remote.client import build_spec

# ---- build_spec (signature: provider, tp, spec_file, spec_class) ----


def test_build_spec_dict_api_string():
    """third_party 为 dict 且值为 API 字符串时，spec.type 标记为 api。"""
    spec = build_spec(
        "torch",
        {"torch": "torch.add", "tf": "tf.raw_ops.Add"},
        spec_file=None,
        spec_class=None,
    )
    assert spec.provider == "torch"
    assert spec.type == "api"
    assert spec.api == "torch.add"


def test_build_spec_dict_impl_class_marks_spec_mode():
    """third_party dict 值为实现类 + spec_file/spec_class 非空时，spec.type 标记为 spec。"""

    class _Dummy:
        pass

    spec = build_spec("torch", {"torch": _Dummy}, spec_file="/tmp/s.py", spec_class="_SpecCls")
    assert spec.provider == "torch"
    assert spec.type == "spec"
    assert spec.spec_file == "/tmp/s.py"
    assert spec.spec_class == "_SpecCls"
