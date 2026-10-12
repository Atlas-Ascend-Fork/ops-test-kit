#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
mindspeed hccl 通信算子手工签名表。

torch_npu 系不登记：torch_npu ≥2.9 起 npu._schemas（aten 权威源，解析链第一
优先级）可完整解析（参数/类型/可选性齐全）；手工快照易与权威 schema 脱节
（参数名演化后按旧表对账会误判用例）。hcom/world_size 注入位语义由
testcase.RUNTIME_INJECTED_PARAMS 通用豁免承担。防腐化断言见
tests/core_modules/framework_api/test_hccl_apis.py。

装载方式：**惰性**——param_extractor.get_api_params 首查时经
_load_hccl_manual_table 装入 _MANUAL_OVERRIDES（全量 APIParamInfo，含
overloads/return_count——register_api_params 的平参登记会丢 return_count）。
无 instance.prepare_env eager 注册；forkserver worker 亦可随处重解析。
mindspeed 系 3 项保留：纯 Python 包（inspect 大概率可直解），仅 torch 域
（tf 无对应物）。
"""

from ..api_model import APIParamInfo, OverloadInfo, ParamInfo

hccl_apis = {
    "mindspeed.npu_bmm_reducescatter_alltoall": APIParamInfo(
        api_name="mindspeed.npu_bmm_reducescatter_alltoall",
        source="manual-hccl",
        overloads=[
            OverloadInfo(
                params=[
                    ParamInfo(name="x", type="Tensor"),
                    ParamInfo(name="weight", type="Tensor"),
                    ParamInfo(name="group_ep", type="str", is_optional=True),
                    ParamInfo(name="group_ep_worldsize", type="int", is_optional=True),
                    ParamInfo(name="group_tp", type="str", is_optional=True),
                    ParamInfo(name="group_tp_worldsize", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="shard_type", type="int", is_optional=True, default=0, is_keyword_only=True),
                ],
                return_count=1,
            )
        ],
    ),
    "mindspeed.npu_alltoall_allgather_bmm": APIParamInfo(
        api_name="mindspeed.npu_alltoall_allgather_bmm",
        source="manual-hccl",
        overloads=[
            OverloadInfo(
                params=[
                    ParamInfo(name="x", type="Tensor"),
                    ParamInfo(name="weight", type="Tensor"),
                    ParamInfo(name="group_ep", type="str", is_optional=True),
                    ParamInfo(name="group_ep_worldsize", type="int", is_optional=True),
                    ParamInfo(name="group_tp", type="str", is_optional=True),
                    ParamInfo(name="group_tp_worldsize", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="shard_type", type="int", is_optional=True, default=0, is_keyword_only=True),
                    ParamInfo(name="act_type", type="str", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="need_allgather_out",
                        type="bool",
                        is_optional=True,
                        default=False,
                        is_keyword_only=True,
                    ),
                    ParamInfo(
                        name="need_activation_feature",
                        type="bool",
                        is_optional=True,
                        default=False,
                        is_keyword_only=True,
                    ),
                ],
                return_count=3,
            )
        ],
    ),
    "mindspeed.npu_mm_all_reduce_add_rms_norm": APIParamInfo(
        api_name="mindspeed.npu_mm_all_reduce_add_rms_norm",
        source="manual-hccl",
        overloads=[
            OverloadInfo(
                params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="residual", type="Tensor"),
                    ParamInfo(name="gamma", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="reduce_op", type="str", is_optional=True, default="sum", is_keyword_only=True),
                    ParamInfo(name="epsilon", type="float", is_optional=True, default=1e-6, is_keyword_only=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="dequant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="antiquant_group_size",
                        type="int",
                        is_optional=True,
                        default=0,
                        is_keyword_only=True,
                    ),
                    ParamInfo(name="comm_turn", type="int", is_optional=True, default=0, is_keyword_only=True),
                ],
                return_count=2,
            )
        ],
    ),
}
