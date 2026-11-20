#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
GeirTestcase — GEIR 模式的用例结构，继承自 TestcaseKernel（共享 Golden 生成逻辑）

图模式（融合 Pass 测试）：CSV ``graph_ops`` 列非空即图模式，多算子 DAG 建图。
路由分界：

- **校验期**以原始声明 ``graph_ops`` 门控：8 个 override 使单算子检查在图模式
  空返回（口/数量/属性对账由 ``geir/graph_validate.py`` 的图校验接管），validate
  末尾 ``validate_graph_case`` 把声明固化/转写为 ``tc._resolved_graph``
- **执行期**只判 ``_resolved_graph`` 是否非空（worker/builder/compiler 一处判定，
  避免多处判定漂移）；``_resolved_graph`` 只存可 pickle 纯数据，经 pickle 跨
  worker 进程边界传递
"""

__all__ = ["GeirTestcase"]

import logging
from typing import Dict, Optional

from ttk.core_modules.common.testcase.field_types import FIELD_TYPES
from ttk.core_modules.kernel.testcase import TestcaseKernel


class GeirTestcase(TestcaseKernel):
    __slots__ = (
        "dyn_input_shapes",
        "graph_ops",
        "graph_edges",
        "graph_outputs",
        "_resolved_graph",
        "_graph_raw_out_str",
    )

    describe_pathway = "geir"

    non_platform_static_property_headers: Dict[str, tuple] = {
        **TestcaseKernel.non_platform_static_property_headers,
        "dyn_input_shapes": (FIELD_TYPES.SHAPELIKE_DYN_NESTED, None, None),
        # ---- 图模式 3 新列（全部可选；graph_ops 非空 = 图模式）----
        # 节点列表（拓扑序）：STRING_CONTAINER 解析 "('mul','add_n')"
        "graph_ops": (FIELD_TYPES.STRING_CONTAINER, None, ()),
        # 连线（(src_ref, dst_ref) 二元组嵌套结构）：FREE_EVAL 自由求值
        # （与 manual_input_binaries 同款），结构合法性由 graph_validate 把关
        "graph_edges": (FIELD_TYPES.FREE_EVAL, None, ()),
        # 图输出引用；缺省 None = 末节点全部输出
        "graph_outputs": (FIELD_TYPES.FREE_EVAL, None, None),
        # attributes 列不加新注册：图模式复用其承载嵌套格式（顶层键 oN），
        # FIELD_TYPES.DICT 既有注册即可解析嵌套 dict（process_dict 只做顶层校验）
    }
    property_headers: Dict[str, tuple] = {
        **non_platform_static_property_headers,
        **TestcaseKernel.static_property_headers,
        **TestcaseKernel.special_property_headers,
    }
    complete_headers: Dict[str, tuple] = {
        **TestcaseKernel.identity_headers,
        **property_headers,
        **TestcaseKernel.option_headers,
    }

    def __init__(self):
        super().__init__()
        self.dyn_input_shapes: Optional[tuple] = None
        self.graph_ops: tuple = ()
        self.graph_edges: tuple = ()
        self.graph_outputs: Optional[tuple] = None
        # 图模式唯一工作形态：validate 末尾由 validate_graph_case 固化
        # （规范化纯数据：节点表、自动直链展开后的显式边、推导后的输出引用）
        self._resolved_graph: Optional[dict] = None
        # output_shapes/output_ori_shapes 原始 str 快照（validate 入口截获，
        # 见 validate 注释）
        self._graph_raw_out_str: bool = False

    @staticmethod
    def supported_rerun_title() -> tuple:
        # GEIR 结果列不含 kernel 专属的 bin_perf_us/perf_status（二进制复用走
        # cst_bin_perf_us/dyn_bin_perf_us 列名）。fusion_pass_details 取值为按模式
        # 组装的计数串或 NA，rerun 筛选按数值/PASS 判定成功，不作为 rerun 目标列。
        return "dyn_perf_us", "cst_perf_us", "precision_status"

    # ══════════════════════════════════════════════════════════════════
    # 校验·早：attributesN 平铺合并跳过（图模式 attributes 为嵌套节点格式，
    # 合并即破坏结构；attributesN 填写报错在 graph_validate._check_mode_restrictions）
    # ══════════════════════════════════════════════════════════════════
    def _merge_extended_attributes(self):
        if self.graph_ops:
            return
        super()._merge_extended_attributes()

    # ══════════════════════════════════════════════════════════════════
    # 校验·中：单算子检查 override（graph_ops 非空即空返回，
    # 口/数量/属性对账由 graph_validate 接管）
    # ══════════════════════════════════════════════════════════════════
    def _check_op_name(self):
        # 图模式：锚点是 golden/容差经 SpecResolver 的绑定锚而非被测算子，
        # 融合目标（如 fused_mul_add_n）常既无部署信息也无 REG_OP，SOC/proto
        # 可达性检查都会误杀（图的算子集合是 graph_ops，平台 gating 由 soc_series
        # 列 + pass 的 MeetRequirements 承担）。空值 OP_NAME_MISSING 保留。
        if self.graph_ops:
            if not self.op_name:
                self.is_valid = False
                self.fail_reason = "OP_NAME_MISSING"
            return
        # 单算子：保留 proto 回落（纯图级 IR 算子无 ops-info 条目属正常）
        was_valid = self.is_valid
        super()._check_op_name()
        if not was_valid or self.is_valid or self.fail_reason != "SOC_NOT_SUPPORT":
            return
        from ttk.core_modules.geir.proto_loader import ProtoLoader

        if ProtoLoader().get_op_info(self.op_name):
            logging.warning(
                f"Operator [{self.op_name}] has no kernel ops-info entry on this SOC; "
                "allowed by IR registration only. If no fusion pass eliminates it, "
                "graph compile will fail loudly."
            )
            self.is_valid = True
            self.fail_reason = None

    def _check_attributes(self):
        # --op-impl-mode 注入会给嵌套 attributes 加伪顶层键 impl_mode（防嵌套污染）
        if self.graph_ops:
            return
        super()._check_attributes()

    def _check_input_count(self):
        # 图级输入数 ≠ 锚点算子输入数（input_shapes 为 dN 扁平序）
        if self.graph_ops:
            return
        super()._check_input_count()

    def _check_output_count(self):
        # 图级输出数 ≠ 锚点算子输出数（output_shapes 按 graph_outputs 序）
        if self.graph_ops:
            return
        super()._check_output_count()

    def _apply_describe_shapes(self):
        # describe 是单算子契约（两 key 描述锚点算子口）
        if self.graph_ops:
            return
        super()._apply_describe_shapes()

    def _auto_set_inplace_indexes(self):
        # 图模式跳过自动推断（inplace 首版整体禁用）
        if self.graph_ops:
            return
        super()._auto_set_inplace_indexes()

    def _cache_split_attributes(self):
        # 按锚点输入名把 attributes 切成 spec_tensors/spec_attrs 是单算子语义；
        # 图模式 attributes 为嵌套节点格式，顶层无与图输入名（dN）同名的键，
        # 此处切分结果恒为空，同名常量定值由 validate_graph_case 解析后按 dN 并入
        if self.graph_ops:
            d_names = {f"d{i}" for i in range(len(self.input_shapes or ()))}
            attrs = self.attributes or {}
            self._spec_tensors_cache = {k: v for k, v in attrs.items() if k in d_names}
            self._spec_attrs_cache = {k: v for k, v in attrs.items() if k not in d_names}
            return
        super()._cache_split_attributes()

    # ══════════════════════════════════════════════════════════════════
    # 校验·末：图校验编排 → 固化 _resolved_graph
    # ══════════════════════════════════════════════════════════════════
    def validate(self):
        # 门控快照：output_shapes/output_ori_shapes 的推理字符串必须在 super()
        # 之前截获：_parse_output_shapes（kernel/testcase.py）会按锚点 + flat
        # 输入做单算子推理，图模式下 dN 扁平输入会让 same_as_input 类推理
        # "假成功"，尾部图校验再见不到 str（output_ori_shapes 经等价头回落与
        # output_shapes 同源，一并快照）
        if self.graph_ops:
            self._graph_raw_out_str = isinstance(self.output_shapes, str) or isinstance(
                getattr(self, "output_ori_shapes", None), str
            )
        super().validate()
        if not self.graph_ops or not self.is_valid:
            return
        from ttk.core_modules.geir.graph_validate import validate_graph_case

        errors = validate_graph_case(self)
        if errors:
            self.is_valid = False
            self.fail_reason = "GRAPH_DESC_INVALID"
            for e in errors:
                logging.error("[%s] %s", self.testcase_name, e)
