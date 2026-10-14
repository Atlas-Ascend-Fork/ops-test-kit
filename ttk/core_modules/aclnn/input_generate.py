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
Input generation method for Universal testcases
"""

__all__ = ["InputGenerator"]


# Standard Packages
import logging

import numpy

from ttk.switches import get_global_storage

from ...utilities import RandomData, get, nest_by_distribution, numpy_to_torch_tensor, resolve_numpy_dtypes
from ...utilities.container import deep_flatten
from ..common.plugin import get_plugin_function
from ..common.testcase.batch import batch_relation_kwargs, has_complete_batch_relation
from ..common.testcase.input_data import assign_tensor_value, broadcast_tensorlist_values, generate_np_storages
from ..common.testcase.tensor import make_storage_view
from . import OpApiInfo, OpApiInfoKeeper

# Third-party Packages
from .testcase import TestcaseAclnn


class InputGenerator:
    def __init__(self, context: TestcaseAclnn):
        self._ctx = context
        self._switch = get_global_storage()

    def gen(self, stored_inputs=None, stored_scalars=None):
        """generate tensor & scalar."""
        if stored_inputs is not None:
            self._restore(stored_inputs, stored_scalars or ())
            return
        # 张量数据源：manual_tensor_binaries 声明 → 按文件路径装载，否则随机生成
        if self._ctx.manual_tensor_binaries:
            self._use_manual_input()
        else:
            self._realtime_random_tensors()
            # attributes 张量值覆写（manual 路径以文件声明为准）
            self._override_tensors_from_attributes()
        # Automatic random input scalars
        self._realtime_random_scalars()

        if self._ctx.is_torch_dtype_support():
            # torch 原生 dtype：numpy.ndarray -> torch.Tensor
            self._convert_np_to_torch_tensor()
            self._package_scalars()
        else:
            # 非 torch 原生 dtype：保持 numpy.ndarray，使用 numpy as_strided
            self._convert_np_to_numpy_view()
            self._package_scalars_numpy()

        # Check special input operators
        input_func = get_plugin_function(self._ctx.api_name, "input", "aclnn", self._switch.plugin_path)
        if input_func:
            self._call_custom_input(input_func)

    def _restore(self, stored_inputs, stored_scalars):
        """Restore final generated state without rerunning random/input plugins."""
        self._ctx.np_storages = list(stored_inputs)
        use_numpy = not self._ctx.is_torch_dtype_support()
        if use_numpy:
            self._convert_np_to_numpy_view()
        else:
            self._convert_np_to_torch_tensor()

        scalar_values = []
        flat_scalar_dtypes = self._ctx.flat_scalar_dtypes or ()
        for index, value in enumerate(stored_scalars):
            if value is None or use_numpy:
                scalar_values.append(value)
                continue
            dtype = get(flat_scalar_dtypes, index)
            scalar_values.append(numpy_to_torch_tensor(value, is_complex32="complex32" in str(dtype)).squeeze())
        self._ctx.scalars = tuple(nest_by_distribution(scalar_values, self._ctx.scalar_list_dist))

    def _call_custom_input(self, input_func):
        plan = self._ctx.get_param_plan()
        args, extra_attrs = plan.build_args(self._ctx.tensors, self._ctx.scalars, self._ctx.attributes)
        kwargs = {
            "short_soc_version": self._switch.short_soc_version,
            "testcase_name": self._ctx.testcase_name,
            "tensor_dtypes": self._ctx.tensor_dtypes,
            "tensor_formats": self._ctx.tensor_formats,
            "scalar_dtypes": self._ctx.scalar_dtypes,
            "input_ranges": self._ctx.input_data_ranges,
            "use_numpy": not self._ctx.is_torch_dtype_support(),
        }
        kwargs.update(batch_relation_kwargs(self._ctx))
        # 框架已完成片段替换的标记（relation 完整即已在生成期重生成，插件可省略该段）
        kwargs["batch_applied"] = has_complete_batch_relation(self._ctx)
        kwargs.update(extra_attrs)
        # testcase 上下文（框架保留键，不被 attributes 覆盖）：多卡 rank 输入
        # 重生成等插件逻辑经此触达 rank 信息（my_rank/device_ids）与槽表
        # 写入口（set_flat_tensor/set_flat_storage/attributes）。
        kwargs["testcase"] = self._ctx
        result = input_func(*args, **kwargs)
        self._apply_custom_input_result(result)

    def _apply_custom_input_result(self, result):
        """消费 customize_inputs 返回值，按输入张量槽位回写。

        返回值为与输入张量形参同序的新张量序列（纯输出槽除外，TensorList
        可嵌套可平铺）；None 槽位保持原值；整体返回 None 表示原地写法（插件
        经共享内存视图改写，无需回写）。回写经既有视图 copy 写穿 storage
        （np_storages/dump/golden 同源可见）；shape 不匹配显式报错，防止
        返回值被静默丢弃。
        """
        if result is None:
            return
        flat = deep_flatten(result) if isinstance(result, (tuple, list, set)) else (result,)
        flat_tensors = self._ctx.flat_tensors or ()
        pure_outputs = set(self._ctx.pure_output_indexes)
        input_slots = [idx for idx in range(len(flat_tensors)) if idx not in pure_outputs]
        if len(flat) != len(input_slots):
            raise RuntimeError(
                f"[{self._ctx.testcase_name}] customize_inputs returned {len(flat)} tensors, "
                f"expected {len(input_slots)} input tensor slots (pure outputs excluded); "
                "return None when customizing in-place"
            )
        for slot, val in zip(input_slots, flat):
            if val is None:
                continue
            view = flat_tensors[slot]
            if view is None:
                raise RuntimeError(
                    f"[{self._ctx.testcase_name}] customize_inputs returned a value for flat slot "
                    f"{slot} which has no generated tensor (dynamic/optional); use the `testcase` "
                    "kwarg writers (set_flat_tensor/set_flat_storage) instead"
                )
            self._copy_into_view(view, val, slot)

    @staticmethod
    def _copy_into_view(view, val, slot):
        import torch

        try:
            if isinstance(view, torch.Tensor):
                view.copy_(val)
            else:
                view[...] = val
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"customize_inputs returned an incompatible value for flat tensor slot {slot}: {e}"
            ) from e

    def _use_manual_input(self):
        """manual_tensor_binaries：按 CSV 声明的文件路径装载张量数据。

        精确对齐语义（kernel _check_manual_binaries 同款）：
        - 非动态输入槽（含 inplace 输出）必须声明路径，缺一即报错——
          防止"少给"静默回退随机数造成数据来源错觉
        - 动态槽（shape 未知/-1）必须 None（物化靠 npu_preprocess）
        - 纯输出槽可缺省（保持常量初始化语义），声明则按文件装载

        声明与 flat 张量槽一一对齐（嵌套结构随 TensorList 分布展开）。
        文件经 load_array 读取（.pt/.npy/裸 bin），dtype 按 CSV 声明、
        shape 按 storage shape 推导。
        """
        logging.info("Using manually configured input data")
        flat_binaries = self._ctx.flat_manual_tensor_binaries
        view_shapes = self._ctx.flat_tensor_view_shapes
        if flat_binaries and len(flat_binaries) > len(view_shapes):
            raise RuntimeError(
                f"manual_tensor_binaries declares {len(flat_binaries)} paths, "
                f"more than flat tensor slots ({len(view_shapes)})"
            )
        pure_outputs = set(self._ctx.pure_output_indexes)
        missing = [
            idx
            for idx, view_shape in enumerate(view_shapes)
            if idx not in pure_outputs
            and (flat_binaries[idx] if idx < len(flat_binaries) else None) is None
            and not (view_shape is None or -1 in view_shape)
        ]
        if missing:
            raise RuntimeError(
                f"manual_tensor_binaries missing paths for non-dynamic input slots {missing} "
                f"(declared {len(flat_binaries)} entries for {len(view_shapes)} slots); "
                "every non-dynamic input slot must carry a file path"
            )
        overrides = {idx: fp for idx, fp in enumerate(flat_binaries) if fp is not None}
        self._ctx.np_storages, self._ctx.actual_input_data_ranges = generate_np_storages(
            self._ctx, self._switch, zero_init_apis=("aclnnInplaceOne",), overrides=overrides
        )

    def _realtime_random_tensors(self):
        self._ctx.np_storages, self._ctx.actual_input_data_ranges = generate_np_storages(
            self._ctx, self._switch, zero_init_apis=("aclnnInplaceOne",)
        )

    def _override_tensors_from_attributes(self):
        """attributes 中按张量形参名声明的值 → 覆写对应张量槽。

        值经 make_storage_view 落盘（与执行侧 as_strided 视图同构，stride/offset
        语义一致）；TensorList 支持标量广播与逐子张量对位。已消费的键从
        attributes 移除——残键会经 golden 的 _package_golden_torch_kwargs 深拷贝
        流入自动 golden 调用，触发 unrecognized keyword arguments 误报
        GOLDEN_FAILURE。
        """
        op_api_info = OpApiInfoKeeper().info_of(self._ctx.api_name)
        attrs = self._ctx.attributes
        if not op_api_info or not attrs:
            return
        consumed = [name for name in op_api_info.tensors if name in attrs]
        if not consumed:
            return
        dist = self._ctx.tensor_list_dist
        storages = self._ctx.np_storages or ()
        flat_shapes = self._ctx.flat_tensor_view_shapes
        flat_idx = 0
        for pos, name in enumerate(op_api_info.tensors):
            sub_count = dist[pos] if pos < len(dist) and dist[pos] > 0 else 0
            first_slot = flat_idx
            flat_idx += max(sub_count, 1)
            if name not in attrs:
                continue
            val = attrs[name]
            if val is None or (isinstance(val, str) and val in ("None", "")):
                raise ValueError(
                    f"[{self._ctx.testcase_name}] Invalid attributes: tensor param '{name}' "
                    f"is assigned {val!r}. Give a concrete value, or declare the tensor "
                    "in tensor_view_shapes only."
                )
            per_tensor_val = broadcast_tensorlist_values(val, sub_count, name) if sub_count else [val]
            for j, sub_val in enumerate(per_tensor_val):
                slot = first_slot + j
                if slot >= len(storages) or storages[slot] is None:
                    logging.warning(
                        "[%s] attributes value of tensor '%s' (flat slot %d) not applied: "
                        "slot has no generated storage (dynamic/optional)",
                        self._ctx.testcase_name,
                        name,
                        slot,
                    )
                    continue
                view = make_storage_view(
                    storages[slot],
                    flat_shapes[slot],
                    self._ctx.flat_view_stride(slot),
                    self._ctx.flat_view_offset(slot),
                )
                assign_tensor_value(view, sub_val, f"{name}[{j}]" if sub_count else name)
        for name in consumed:
            attrs.pop(name, None)

    def _realtime_random_scalars(self):
        flat_scalar_dtypes = self._ctx.flat_scalar_dtypes
        dtypes = resolve_numpy_dtypes(flat_scalar_dtypes)
        scalars = []

        for idx, sd in enumerate(dtypes):
            if sd is None:
                scalars.append(None)
                continue
            data_range = get(self._ctx.flat_scalar_data_ranges or (), idx, out_of_range=(None, None))
            dtype = get(dtypes, idx)
            rd = RandomData(dtype, [1], data_range)
            np_arr = rd.generate(self._switch.input_distribution)
            t_scalar = numpy_to_torch_tensor(np_arr, is_complex32="complex32" in str(dtype))
            scalars.append(t_scalar.squeeze())
        self._ctx.scalars = nest_by_distribution(scalars, self._ctx.scalar_list_dist)

    def _convert_np_to_torch_tensor(self):
        import torch

        torch_tensors = []
        flat_shapes = self._ctx.flat_tensor_view_shapes
        for idx, np_arr in enumerate(self._ctx.np_storages):
            if np_arr is None:
                torch_tensors.append(None)
            else:
                dtype = get(self._ctx.flat_tensor_dtypes, idx)
                v_shape = flat_shapes[idx]
                v_stride = self._ctx.flat_view_stride(idx)
                v_offset = self._ctx.flat_view_offset(idx)
                t_storage = numpy_to_torch_tensor(np_arr, is_complex32="complex32" in str(dtype))
                try:
                    if t_storage.dim() == 0 and v_shape == ():
                        t_view = t_storage
                    else:
                        t_view = torch.as_strided(t_storage, v_shape, v_stride, v_offset)
                except RuntimeError:
                    logging.error(
                        f"torch.as_strided failed. storage_shape={t_storage.shape} "
                        f"view_shape={v_shape}, view_stride={v_stride}, view_offset={v_offset}"
                    )
                    raise
                torch_tensors.append(t_view)
        self._ctx.tensors = nest_by_distribution(torch_tensors, self._ctx.tensor_list_dist)

    def _package_scalars(self):
        """
        1.package scalar-list.
        2.pick scalars defined in context.attributes.
        """
        import torch

        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self._ctx.api_name)
        scalars = list(self._ctx.scalars)
        scalars = scalars[: len(op_api_info.scalars)]
        for idx, s_name in enumerate(op_api_info.scalars):
            if idx >= len(scalars):
                raise RuntimeError(f"Some Scalar/ScalarList is not configured: {op_api_info.scalars[idx:]}")
            if s_name in self._ctx.attributes:
                val = self._ctx.attributes[s_name]
                if isinstance(val, (list, tuple)):
                    if not isinstance(scalars[idx], list):
                        raise RuntimeError(
                            f"[{s_name}] is a list/tuple configured in attributes. "
                            f"But got a scalar rather than ScalarList [{scalars[idx]}]. "
                            f"Check scalar_dtypes nesting."
                        )
                    if len(val) != len(scalars[idx]):
                        raise RuntimeError(f"Value count of [{s_name}] mismatch: expected [{len(scalars[idx])}].")
                    for j in range(len(scalars[idx])):
                        scalars[idx][j] = torch.tensor(val[j], dtype=scalars[idx][j].dtype)
                else:
                    scalars[idx] = torch.tensor(val, dtype=scalars[idx].dtype)
        self._ctx.scalars = tuple(scalars)

    def _convert_np_to_numpy_view(self):
        """非 torch 原生 dtype 时，使用 numpy as_strided 创建 view（替代 torch.as_strided）"""
        np_views = []
        flat_shapes = self._ctx.flat_tensor_view_shapes
        for idx, np_arr in enumerate(self._ctx.np_storages):
            if np_arr is None:
                np_views.append(None)
            else:
                v_shape = flat_shapes[idx]
                v_stride = self._ctx.flat_view_stride(idx)
                v_offset = self._ctx.flat_view_offset(idx)
                try:
                    np_views.append(make_storage_view(np_arr, v_shape, v_stride, v_offset))
                except Exception:
                    logging.error(
                        f"numpy.as_strided failed. storage_shape={np_arr.shape} "
                        f"view_shape={v_shape}, view_stride={v_stride}, view_offset={v_offset}"
                    )
                    raise
        self._ctx.tensors = nest_by_distribution(np_views, self._ctx.tensor_list_dist)

    def _package_scalars_numpy(self):
        """非 torch 原生 dtype 时，scalar 保持为 numpy scalar/ndarray"""
        scalars = self._ctx.scalars
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self._ctx.api_name)
        scalars = scalars[: len(op_api_info.scalars)]
        for idx, s_name in enumerate(op_api_info.scalars):
            if idx >= len(scalars):
                raise RuntimeError(f"Some Scalar/ScalarList is not configured: {op_api_info.scalars[idx:]}")
            if s_name in self._ctx.attributes:
                val = self._ctx.attributes[s_name]
                if isinstance(val, (list, tuple)):
                    if not isinstance(scalars[idx], list):
                        raise RuntimeError(
                            f"[{s_name}] is a list/tuple configured in attributes. "
                            f"But got a scalar rather than ScalarList [{scalars[idx]}]. "
                            f"Check scalar_dtypes nesting."
                        )
                    if len(val) != len(scalars[idx]):
                        raise RuntimeError(f"Value count of [{s_name}] mismatch: expected [{len(scalars[idx])}].")
                    for j in range(len(scalars[idx])):
                        scalars[idx][j] = numpy.array(val[j], dtype=scalars[idx][j].dtype)
                else:
                    scalars[idx] = numpy.array(val, dtype=scalars[idx].dtype)
        self._ctx.scalars = tuple(scalars)

    @staticmethod
    def is_broadcast(tensor) -> bool:
        """
        判断 tensor 是否是广播视图（stride 中有 0 且对应 size > 1）

        Args:
            tensor: torch.Tensor 或 None

        Returns:
            bool: True 表示是广播视图，False 表示非广播
        """
        if tensor is None:
            return False
        return any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride()))
