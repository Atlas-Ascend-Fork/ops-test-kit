# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""compute_output_md5 边界与 None 位（#287）：异形同字节 / None 位移位不得产生相同摘要。"""

import numpy as np

from ttk.core_modules.framework_api.utils import compute_output_md5, finalize_det_status


def test_reshaped_outputs_digest_differ():
    """#287 最小复现：([1,2],[3]) 与 ([1],[2,3]) 拼接字节相同，摘要须不同并判 FAIL。"""
    first = [np.array([1, 2], dtype=np.int64), np.array([3], dtype=np.int64)]
    second = [np.array([1], dtype=np.int64), np.array([2, 3], dtype=np.int64)]
    assert compute_output_md5(first) != compute_output_md5(second)
    assert finalize_det_status([compute_output_md5(first), compute_output_md5(second)], "t") == "FAIL"


def test_none_position_recorded():
    """#287：None 位参与摘要——[None, x] 与 [x]、[None, x] 与 [x, None] 不得同摘要。"""
    x = np.array([1, 2], dtype=np.float32)
    assert compute_output_md5([None, x]) != compute_output_md5([x])
    assert compute_output_md5([None, x]) != compute_output_md5([x, None])


def test_same_outputs_digest_stable():
    """同型同值摘要稳定（确定性语义不变）；同字节异 shape / 异 dtype 即变化。"""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert compute_output_md5([a]) == compute_output_md5([a.copy()])
    assert compute_output_md5([a]) != compute_output_md5([a.reshape(4)])
    assert compute_output_md5([a]) != compute_output_md5([a.astype(np.float64)])
    assert finalize_det_status([compute_output_md5([a]), compute_output_md5([a.copy()])], "t") == "PASS"
