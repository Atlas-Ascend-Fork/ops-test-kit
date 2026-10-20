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
output generation method for Universal testcases
"""

__all__ = ["GoldenGenerator"]


# Standard Packages
import contextlib
import copy
import logging
from collections.abc import Callable

# Third-party Packages
from ttk.switches import get_global_storage

from ...utilities import (
    DTYPE_PROMOTE_MAP,
    acl_to_torch_dtype,
    camel_to_snake,
    get,
    is_torch_native_dtype,
    resolve_numpy_dtypes,
    str_to_torch_dtype,
)
from ...utilities.container import deep_flatten, nest_by_distribution
from ...utilities.file_utils import load_array
from ..common.invoke import Invoke, golden_flatten, golden_mode, instantiate_and_call
from ..common.invoke import _framework_of as framework_of
from ..common.plugin import SpecResolver
from ..common.testcase.batch import batch_relation_kwargs
from . import OpApiInfo, OpApiInfoKeeper
from .testcase import TestcaseAclnn

ACLNN_GOLDEN: dict = {
    "aclnnInplaceOne": "torch.ops.aten.ones_like",
    "aclnnInplaceZero": "torch.ops.aten.zeros_like",
    "aclnnMaxDim": "torch.ops.aten.max",
    "aclnnMax": "torch.ops.aten.amax",
    "aclnnVarCorrection": "torch.ops.aten.var",
    "aclnnStdMeanCorrection": "torch.ops.aten.std_mean",
    "aclnnIsFinite": "torch.ops.aten.isfinite",
    "aclnnIsInf": "torch.ops.aten.isinf",
    "aclnnIsClose": "torch.ops.aten.isclose",
    "aclnnMatmul": "torch.mm",
    "aclnnBatchMatMul": "torch.bmm",
    "aclnnArgMax": "torch.argmax",
    "aclnnAminmaxAll": "torch.aminmax",
    "aclnnAminmaxDim": "torch.aminmax",
    "aclnnLogSumExp": "torch.logsumexp",
}


def _promote_tensor(t, tgt_dt):
    """torch 张量提升：strided 视图（存储>视图/非零offset/非连续，如 FRACTAL_NZ 逻辑视图）
    整存储元素级提升并按原视图几何重建，视图外的分形尾块数据在 cast 后存活；
    普通 contiguous 张量直接 .to()。"""
    storage_numel = t.untyped_storage().nbytes() // t.element_size()
    if storage_numel > t.numel() or t.storage_offset() != 0 or not t.is_contiguous():
        whole = t.as_strided((storage_numel,), (1,), 0)
        promoted = whole.to(tgt_dt)
        return promoted.as_strided(tuple(t.shape), t.stride(), t.storage_offset())
    return t.to(tgt_dt)


class GoldenGenerator:
    def __init__(self, context: TestcaseAclnn):
        self._ctx = context
        self._switch = get_global_storage()

    def gen(self):
        if golden_mode(self._ctx, self._switch) == "Disable":
            golden_arrays = ("SUPPRESSED",)
        elif self._ctx.manual_golden_binaries:
            golden_arrays = self._load_golden_from_file()
        else:
            golden_arrays = self._generate_golden()

        # golden_arrays now is flatten.
        # maybe torch or numpy arrays.
        self._ctx.golden_tensors = golden_arrays[: len(self._ctx.output_tensor_indexes)]

    def _load_golden_from_file(self):
        """manual_golden_binaries：按 CSV 声明的文件路径装载 golden。

        声明与输出槽（output_tensor_indexes 序）一一对齐；未声明/None 的
        槽位回退自动生成（golden 与输入分开提供——未给的槽由已装载的输入
        推导，比对不缺席；整字段空声明即走纯自动路径）。dtype 取该槽首个
        flat 输出的声明 dtype，shape 取其 view shape（load_array 按 shape
        重排，4-bit 做打包后处理）。TensorList 输出槽按整槽单文件（存储连续）。
        """
        binaries = self._ctx.manual_golden_binaries or ()
        dtypes = resolve_numpy_dtypes(self._ctx.flat_output_dtypes)
        auto_goldens = None
        goldens = []
        for slot, nested_idx in enumerate(self._ctx.output_tensor_indexes):
            fp = binaries[slot] if slot < len(binaries) else None
            if fp is None:
                if auto_goldens is None:
                    auto_goldens = self._generate_golden()
                goldens.append(auto_goldens[slot] if slot < len(auto_goldens) else None)
                continue
            flat_idx = self._ctx.flat_index_of(nested_idx)
            view_shape = get(self._ctx.flat_output_view_shapes, flat_idx)
            goldens.append(load_array(fp, get(dtypes, flat_idx), view_shape))
        return goldens

    def _call_torch_api(self, gf):
        """ACLNN torch API golden:packaged tensors + torch-convention kwargs → gf(*tensors, **kwargs)。

        torch 标量重载（如 _foreach_* 的 scalar 分支）要求 0-dim 张量，而 ACLNN 侧
        "scalar" aclTensor 槽常被声明为 (1,)。命中该类报错时，把 1 元素张量数值
        等价地 reshape 成 0-dim 重试一次（仅此一类报错触发，广播语义不变）。"""
        tensors = self._package_golden_tensors()
        kwargs = self._package_golden_torch_kwargs()
        try:
            return gf(*tensors, **kwargs)
        except RuntimeError as err:
            if "expected to be 0 dim" not in str(err):
                raise
            import torch

            changed = False

            def _as_scalar(t):
                nonlocal changed
                if isinstance(t, torch.Tensor) and t.numel() == 1 and t.dim() > 0:
                    changed = True
                    return t.reshape(())
                if isinstance(t, (list, tuple)):
                    return type(t)(_as_scalar(x) for x in t)
                return t

            adapted = tuple(_as_scalar(t) for t in tensors)
            if not changed:
                raise
            return gf(*adapted, **kwargs)

    def _call_custom_golden(self, gf):
        plan = self._ctx.get_param_plan()
        args, extra_attrs = plan.build_args(self._ctx.tensors, self._ctx.scalars, self._ctx.attributes)
        kwargs = {
            "short_soc_version": self._switch.short_soc_version,
            "testcase_name": self._ctx.testcase_name,
            "tensor_dtypes": self._ctx.tensor_dtypes,
            "tensor_formats": self._ctx.tensor_formats,
            "scalar_dtypes": self._ctx.scalar_dtypes,
            "use_numpy": not self._ctx.is_torch_dtype_support(),
        }
        kwargs.update(batch_relation_kwargs(self._ctx))
        kwargs.update(extra_attrs)
        results = gf(*args, **kwargs)
        return results

    def _named_values(self):
        """Build a name→value pool (input tensors + scalars + attrs) for class-form bind_by_name.

        Pool is built in C header parameter order (from op_api_info.params OrderedDict),
        preserving tensor/scalar interleaving — critical for *args / positional fallback.
        """
        pool = {}
        op_api_info = OpApiInfoKeeper().info_of(self._ctx.api_name)
        tensors = self._package_golden_tensors()  # input tensors only (pure outputs skipped)
        if op_api_info is not None:
            tensor_names = op_api_info.tensors
            pure_output_names = {tensor_names[i] for i in self._ctx.pure_output_indexes if i < len(tensor_names)}
            tensor_queue = list(tensors)
            scalar_queue = list(self._ctx.flat_scalars or ())
            for name, info in op_api_info.params.items():
                if name in pure_output_names:
                    continue
                ptype = info["type"]
                if ptype in ("aclTensor*", "aclTensorList*"):
                    if tensor_queue:
                        pool[name] = tensor_queue.pop(0)
                elif ptype in ("aclScalar*", "aclScalarList*"):
                    if scalar_queue:
                        pool[name] = scalar_queue.pop(0)
                else:
                    attrs = self._ctx.attributes or {}
                    if name in attrs:
                        pool[name] = attrs[name]
            # 显式 ValueError（非 assert）：-O 下 assert 被剥除会静默失去防护；
            # 异常经 Invoke.golden 收敛为 GOLDEN_FAILURE 占位，用例 FAIL 可见。
            if tensor_queue:
                raise ValueError(
                    f"[{self._ctx.api_name}] _named_values: {len(tensor_queue)} tensors not consumed by params "
                    f"(possible TensorList nesting or pure_output_indexes mismatch)"
                )
            if scalar_queue:
                raise ValueError(
                    f"[{self._ctx.api_name}] _named_values: {len(scalar_queue)} scalars not consumed by params "
                    f"(possible ScalarList nesting or scalar count mismatch)"
                )
        pool.update(self._ctx.attributes or {})
        return pool

    def _invoke_class(self, cls):
        # Promote-context wrap is now applied at the outer _invoke_golden dispatch.
        return instantiate_and_call(cls, self._named_values())

    def _invoke_golden(self, golden_func):
        """通路执行闭包（str 已由 Invoke.golden 解析）：Promote 包裹 + class/torch/custom 分发。

        Promote-context wrap around the ENTIRE dispatch (class / torch / custom)
        — custom goldens also need promoted bfloat16/float16 inputs.
        """
        with self._golden_mode():
            if isinstance(golden_func, type):
                return self._invoke_class(golden_func)
            if framework_of(golden_func) == "torch":
                return self._call_torch_api(golden_func)
            return self._call_custom_golden(golden_func)

    def _generate_golden(self):
        golden_func = self._import_golden_funcs()
        if golden_func is None:
            logging.warning(f"Golden data generation method for [{self._ctx.api_name}] is not registered!")
        return golden_flatten(Invoke.golden(self._ctx, golden_func, self._switch, self._invoke_golden))

    def _import_golden_funcs(self):
        """resolve 链：spec/插件 golden → ACLNN_GOLDEN 映射 → torch 同名自动导入。"""
        return SpecResolver(self._ctx.api_name, "aclnn", self._switch.plugin_path).resolve_golden(
            self._builtin_golden_fallback
        )

    def _builtin_golden_fallback(self):
        golden_func = ACLNN_GOLDEN.get(self._ctx.api_name)
        if golden_func:
            return golden_func
        snake_api_name = camel_to_snake(self._ctx.api_name[5:])
        return self._auto_import_from_torch(snake_api_name)

    def _find_torch_api(self, torch_module, snake_name: str):
        # find torch.snake_name
        try:
            torch_api = getattr(torch_module, snake_name, None)
        except RuntimeError:
            # older torch version, 1.11.0 for exp,
            # will throw RuntimeError when absent.
            torch_api = None
        if torch_api and isinstance(torch_api, Callable):
            logging.debug(
                f"Using function [{torch_module.__name__}.{snake_name}] "
                f"for api {self._ctx.api_name} to generate golden."
            )
            return torch_api
        if snake_name.startswith("inplace_"):
            return self._find_torch_api(torch_module, snake_name[len("inplace_") :])
        if snake_name.endswith("_scalar"):
            return self._find_torch_api(torch_module, snake_name[: -len("_scalar")])
        if snake_name.endswith("_tensor"):
            return self._find_torch_api(torch_module, snake_name[: -len("_tensor")])
        if snake_name.endswith("_v2"):
            return self._find_torch_api(torch_module, snake_name[: -len("_v2")])
        if snake_name.endswith("s"):
            return self._find_torch_api(torch_module, snake_name[: -len("s")])
        if not snake_name.startswith("_"):  # _add_relu
            return self._find_torch_api(torch_module, "_" + snake_name)
        return None

    def _auto_import_from_torch(self, snake_name: str):
        import torch

        # torch api first
        torch_api = self._find_torch_api(torch, snake_name)
        if torch_api is not None:
            return torch_api
        torch_nn_api = self._find_torch_api(torch.nn.functional, snake_name)
        if torch_nn_api is not None:
            return torch_nn_api
        return self._find_torch_api(torch.ops.aten, snake_name)

    @contextlib.contextmanager
    def _golden_mode(self):
        if golden_mode(self._ctx, self._switch) != "Promote":
            yield
        else:
            yield from self._promote_dtype()

    def _promote_dtype(self):
        """Promote low-precision input tensors+scalars to higher precision for golden (true value), restore after.

        ctx.tensors may be numpy or torch (decided by input_generation) — promoted in-place
        via .to() (torch) or .astype() (numpy), type preserved.
        Strided torch views (storage larger than the logical view, non-zero offset or
        non-contiguous — e.g. FRACTAL_NZ logical views over ceil-aligned fractal storage)
        are promoted whole-storage elementwise with the original view geometry rebuilt,
        so bytes beyond the logical view (fractal tail padding) survive the cast.
        Pure output slots (ctx.pure_output_indexes) are NOT promoted: they are
        result containers whose dtype is the CSV/API-declared output dtype, not
        operands of the golden computation.
        If source dtype is non-torch-native but target is (cross-boundary), raises
        NotImplementedError (numpy<->torch conversion not yet implemented).
        Bare generator (no @contextlib.contextmanager) — _golden_mode uses yield from."""
        import torch

        ctx = self._ctx
        t_dtypes = list(ctx.flat_tensor_dtypes)
        s_dtypes = list(ctx.flat_scalar_dtypes or ())
        pure_outputs = set(ctx.pure_output_indexes)
        t_need = [d in DTYPE_PROMOTE_MAP and i not in pure_outputs for i, d in enumerate(t_dtypes)]
        s_need = [d in DTYPE_PROMOTE_MAP for d in s_dtypes]
        if not (any(t_need) or any(s_need)):
            yield
            return
        # scenario-2 guard
        guard_dtypes = [d for i, d in enumerate(t_dtypes) if i not in pure_outputs] + s_dtypes
        for d in guard_dtypes:
            if d in DTYPE_PROMOTE_MAP:
                tgt = DTYPE_PROMOTE_MAP[d]
                if not is_torch_native_dtype(d) and is_torch_native_dtype(tgt):
                    raise NotImplementedError(
                        f"ACLNN promote crosses torch native boundary {d}->{tgt}: "
                        f"numpy<->torch conversion not implemented"
                    )
        # backup nested attrs (flat caches go through invalidate, not backup)
        bak = (ctx.tensors, ctx.tensor_dtypes, ctx.scalars, ctx.scalar_dtypes)
        try:
            # promote tensors: flat → .to/.astype → re-nest
            flat_t = list(deep_flatten(ctx.tensors))
            new_t_dtypes = list(t_dtypes)
            for i, t in enumerate(flat_t):
                if t_need[i] and t is not None:
                    tgt = DTYPE_PROMOTE_MAP[t_dtypes[i]]
                    if isinstance(t, torch.Tensor):
                        flat_t[i] = _promote_tensor(t, str_to_torch_dtype(tgt))
                    else:
                        flat_t[i] = t.astype(tgt)
                    new_t_dtypes[i] = tgt
            dist = ctx.tensor_list_dist
            ctx.tensors = tuple(nest_by_distribution(flat_t, dist))
            ctx.tensor_dtypes = tuple(nest_by_distribution(new_t_dtypes, dist))
            # promote scalars
            flat_s = list(deep_flatten(ctx.scalars)) if ctx.scalars else []
            new_s_dtypes = list(s_dtypes)
            for i, s in enumerate(flat_s):
                if i < len(s_need) and s_need[i] and s is not None:
                    tgt = DTYPE_PROMOTE_MAP[s_dtypes[i]]
                    flat_s[i] = s.to(str_to_torch_dtype(tgt)) if isinstance(s, torch.Tensor) else s.astype(tgt)
                    new_s_dtypes[i] = tgt
            sdist = ctx.scalar_list_dist
            ctx.scalars = tuple(nest_by_distribution(flat_s, sdist))
            ctx.scalar_dtypes = tuple(nest_by_distribution(new_s_dtypes, sdist))
            ctx.invalidate_flat_cache("tensors", "tensor_dtypes", "scalars", "scalar_dtypes")
            yield
        finally:
            ctx.tensors, ctx.tensor_dtypes, ctx.scalars, ctx.scalar_dtypes = bak
            ctx.invalidate_flat_cache("tensors", "tensor_dtypes", "scalars", "scalar_dtypes")

    def _package_golden_torch_kwargs(self):
        DEL_KWARG = ("impl_mode", "cube_math_type")
        # torch api both support keepdims & keepdim.
        # but dim / dims diffs.
        RENAME_KWARG = {
            "keep_dim": "keepdim",
            "keep_dims": "keepdims",
            "keepDims": "keepdims",
            "keepDim": "keepdim",
        }
        kwargs_camel = copy.deepcopy(self._ctx.attributes)
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self._ctx.api_name)
        for idx, s in enumerate(self._ctx.flat_scalars or ()):
            s_name = op_api_info.scalars[idx]
            kwargs_camel.update({s_name: s})
        # camel -> snake
        kwargs_snake = {}
        for k, v in kwargs_camel.items():
            k_snake = camel_to_snake(k)
            kwargs_snake.update({k_snake: v})
        for k, v in RENAME_KWARG.items():
            if k in kwargs_snake:
                kwargs_snake[v] = kwargs_snake[k]
                del kwargs_snake[k]
        for ele in DEL_KWARG:
            if ele in kwargs_snake:
                del kwargs_snake[ele]
        # dtype from aclDType to torchDtype automatically
        if "dtype" in kwargs_snake and isinstance(kwargs_snake["dtype"], int):
            kwargs_snake["dtype"] = acl_to_torch_dtype([kwargs_snake["dtype"]])[0]
        if "dim" in kwargs_snake and isinstance(kwargs_snake["dim"], list):
            kwargs_snake["dim"] = tuple(kwargs_snake["dim"])
        if "dims" in kwargs_snake and isinstance(kwargs_snake["dims"], list):
            kwargs_snake["dims"] = tuple(kwargs_snake["dims"])
        return kwargs_snake

    def _package_golden_tensors(self) -> list:
        golden_tensors = []
        case_tensors = self._ctx.tensors
        real_idx = 0
        for ct in case_tensors:
            # pure output 位只跳过装载，索引照常推进
            if real_idx not in self._ctx.pure_output_indexes:
                golden_tensors.append(ct)
            real_idx += len(ct) if isinstance(ct, (list, tuple)) else 1
        return golden_tensors
