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
import gc
import logging
from typing import Sequence

import numpy

try:
    from collections.abc import Callable
except ImportError:
    from collections.abc import Callable

# Third-party Packages
from ....utilities import (
    DTYPE_PROMOTE_MAP,
    acl_to_torch_dtype,
    bind_by_name,
    camel_to_snake,
    framework_of,
    get_global_storage,
    is_torch_native_dtype,
    resolve_callable_str,
    str_to_torch_dtype,
)
from ....utilities.container_utils import apply_as_list, deep_flatten
from ...aclnn import OpApiInfo, OpApiInfoKeeper
from ...deterministic import batch_relation_kwargs
from ...plugin_loader import get_plugin_function
from ...testcase_manager import TestcaseAclnn

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


class GoldenGenerator:
    def __init__(self, context: TestcaseAclnn):
        self._ctx = context
        self._switch = get_global_storage()

    def gen(self):
        if self._switch.golden_mode == "Disable":
            golden_arrays = ("SUPPRESSED",)
        elif self._ctx.manual_golden_binaries:
            golden_arrays = self._load_golden_from_file()
        else:
            golden_arrays = self._generate_golden()

        # golden_arrays now is flatten.
        # maybe torch or numpy arrays.
        self._ctx.golden_tensors = golden_arrays[: len(self._ctx.output_tensor_indexes)]

    def _load_golden_from_file(self):
        # TODO
        logging.warning("Using manually configured golden data")

    def _call_torch_api(self, gf):
        """ACLNN torch API golden:packaged tensors + torch-convention kwargs → gf(*tensors, **kwargs)。"""
        tensors = self._package_golden_tensors()
        kwargs = self._package_golden_torch_kwargs()
        return gf(*tensors, **kwargs)

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
            scalar_queue = list(self._ctx.flatten_scalars or ())
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
            assert not tensor_queue, (
                f"_named_values: {len(tensor_queue)} tensors not consumed by params "
                f"(possible TensorList nesting or pure_output_indexes mismatch)"
            )
            assert not scalar_queue, (
                f"_named_values: {len(scalar_queue)} scalars not consumed by params "
                f"(possible ScalarList nesting or scalar count mismatch)"
            )
        pool.update(self._ctx.attributes or {})
        return pool

    def _invoke_class(self, cls):
        pool = self._named_values()
        if cls.__init__ is object.__init__:
            inst = cls()
        else:
            ia, ik = bind_by_name(cls.__init__, pool)
            inst = cls(*ia, **ik)
        ca, ck = bind_by_name(inst.__call__, pool)
        # Promote-context wrap is now applied at the outer _invoke_golden dispatch.
        return inst(*ca, **ck)

    def _invoke_golden(self, golden_func):
        if isinstance(golden_func, str):
            golden_func = resolve_callable_str(golden_func)
        # Promote-context wrap around the ENTIRE dispatch (class / torch / custom)
        # — custom goldens also need promoted bfloat16/float16 inputs.
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
            golden_arrays = ["UNSUPPORTED"]
        else:
            # noinspection PyBroadException
            try:
                golden_results = self._invoke_golden(golden_func)
            except TimeoutError:
                logging.exception("Golden generation timeout")
                golden_results = ["GOLDEN_TIMEOUT"]
            except Exception as e:
                logging.exception(f"Golden generation failure: {e}")
                golden_results = ["GOLDEN_FAILURE"]
            finally:
                gc.collect()
            golden_arrays = self._golden_reshape(golden_results)
        return golden_arrays

    def _import_golden_funcs(self):
        golden_func = get_plugin_function(self._ctx.api_name, "golden", "aclnn", self._switch.plugin_path)
        if golden_func:
            return golden_func
        golden_func = ACLNN_GOLDEN.get(self._ctx.api_name)
        if golden_func:
            return golden_func
        snake_api_name = camel_to_snake(self._ctx.api_name[5:])
        func = self._auto_import_from_torch(snake_api_name)
        # TD: skip searching next time. optimize later.
        return func

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
        mode = getattr(self._ctx, "golden_mode_override", None) or self._switch.golden_mode
        if mode != "Promote":
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
        If source dtype is non-torch-native but target is (cross-boundary), raises
        NotImplementedError (numpy<->torch conversion not yet implemented).
        Bare generator (no @contextlib.contextmanager) — _golden_mode uses yield from."""
        import torch

        ctx = self._ctx
        t_dtypes = list(ctx.flat_tensor_dtypes)
        s_dtypes = list(ctx.flat_scalar_dtypes or ())
        t_need = [d in DTYPE_PROMOTE_MAP for d in t_dtypes]
        s_need = [d in DTYPE_PROMOTE_MAP for d in s_dtypes]
        if not (any(t_need) or any(s_need)):
            yield
            return
        # scenario-2 guard
        for d in t_dtypes + s_dtypes:
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
                        tgt_dt = str_to_torch_dtype(tgt)
                        storage_numel = t.untyped_storage().nbytes() // t.element_size()
                        if storage_numel > t.numel() or t.storage_offset() != 0 or not t.is_contiguous():
                            # Strided view (e.g. a FRACTAL_NZ logical view over a larger
                            # physical storage): promote the WHOLE storage elementwise and
                            # rebuild the same view geometry, so bytes beyond the logical
                            # view (fractal tail padding) survive the cast.
                            whole = t.as_strided((storage_numel,), (1,), 0)
                            promoted = whole.to(tgt_dt)
                            flat_t[i] = promoted.as_strided(tuple(t.shape), t.stride(), t.storage_offset())
                        else:
                            flat_t[i] = t.to(tgt_dt)
                    else:
                        flat_t[i] = t.astype(tgt)
                    new_t_dtypes[i] = tgt
            dist = ctx.tensor_list_dist
            ctx.tensors = tuple(apply_as_list(flat_t, dist))
            ctx.tensor_dtypes = tuple(apply_as_list(new_t_dtypes, dist))
            # promote scalars
            flat_s = list(deep_flatten(ctx.scalars)) if ctx.scalars else []
            new_s_dtypes = list(s_dtypes)
            for i, s in enumerate(flat_s):
                if i < len(s_need) and s_need[i] and s is not None:
                    tgt = DTYPE_PROMOTE_MAP[s_dtypes[i]]
                    flat_s[i] = s.to(str_to_torch_dtype(tgt)) if isinstance(s, torch.Tensor) else s.astype(tgt)
                    new_s_dtypes[i] = tgt
            sdist = ctx.scalar_list_dist
            ctx.scalars = tuple(apply_as_list(flat_s, sdist))
            ctx.scalar_dtypes = tuple(apply_as_list(new_s_dtypes, sdist))
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
        for idx, s in enumerate(self._ctx.flatten_scalars or ()):
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
            if real_idx in self._ctx.pure_output_indexes:
                continue
            golden_tensors.append(ct)
            real_idx += len(ct) if isinstance(ct, (list, tuple)) else 1
        return golden_tensors

    @staticmethod
    def _golden_reshape(golden_results):
        golden_arrays = []
        if not isinstance(golden_results, Sequence):
            golden_results = [golden_results]
        for golden_result in golden_results:
            if isinstance(golden_result, numpy.generic):
                golden_arrays.append(numpy.array((golden_result,)))
            else:
                golden_arrays.append(golden_result)
        return golden_arrays
