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
data generator

Module constraints (do not violate in future changes):
- numpy-only: do NOT import torch/tensorflow (or any other framework) here, not
  even inside functions. TTK must work in environments with only numpy
  installed (e.g. tensorflow-only e2e without torch); a framework RNG branch
  here once broke such environments and was removed.
- _sample_distinct_positions must stay O(count) in time and memory regardless
  of `total`. Replacing it with numpy.random.choice(total, count,
  replace=False) regresses to O(total) (a full permutation: ~61MB temp for an
  8M-element tensor, ~320MB for 40M) and, combined with a large per_count,
  re-introduces both the memory spike and the collision-driven retry storm.
  Benchmark reference: 8M tensor, 20k positions — choice ~167ms vs
  unique+randint ~4ms; 40M fp32 generate end-to-end 77.8s -> 0.18s.
"""

__all__ = ["RandomData", "constant_array"]


# Standard Packages
import math
from typing import Union

import numpy

# Third-party Packages
from .container import get
from .dtypes import get_dtype_range, is_float_dtype, resolve_numpy_dtype, resolve_numpy_dtypes
from .math import is_negative_zero, is_positive_zero
from .shape import shape_numel

DEFAULT_LOW = -2
DEFAULT_HIGH = 2
CHUNK_ELEMS = 4_000_000


def constant_array(dtype, shape, init_value=1):
    """Create a numpy array filled with a fixed value, handling complex32 specially.

    complex32 is stored as float16 with an extra trailing dim [2] (real, imag).
    For complex32: real part = init_value, imag part = 0.
    """
    if "complex32" in str(dtype):
        c_shape = list(shape) + [1]
        real = numpy.full(c_shape, init_value, dtype=numpy.float16)
        imag = numpy.zeros(c_shape, dtype=numpy.float16)
        return numpy.concatenate((real, imag), axis=-1)
    return numpy.full(shape, init_value, dtype=dtype)


def _is_native_integer(dtype) -> bool:
    """是否为 numpy 原生整型(int8..int64 / uint8..uint64)。

    自定义窄类型(int4/uint1 等)不在此列——它们没有原生 RNG，仍走 float64 路径。
    """
    try:
        return numpy.issubdtype(numpy.dtype(dtype), numpy.integer)
    except TypeError:
        return False


class RandomData:
    def __init__(self, dtype: str, shape: Union[list, tuple], data_range: Union[list, tuple]):
        self._dtype = resolve_numpy_dtypes([dtype])[0]
        self._shape = list(shape)
        self._data_range = self._replace_none_in_data_range(data_range)

    @property
    def data_range(self):
        return self._data_range

    def generate(self, distribution: str = "uniform"):
        if self._dtype == "uint1":
            return self._generate_uint1(distribution)
        if "complex" in str(self._dtype):
            return self._generate_complex(distribution)
        if "e8m0" in str(self._dtype):
            return self._generate_e8m0()
        return self._random(self._dtype, self._shape, distribution=distribution)

    def _generate_uint1(self, distribution: str):
        """uint1 packs one bit per element into a uint8 bitstream (via uint8 random)."""
        np_uint8 = self._random("uint8", self._shape, distribution=distribution)
        np_bool = np_uint8.astype("bool", copy=False)
        return numpy.packbits(np_bool)

    def _generate_complex(self, distribution: str):
        """complex32 interleaves float16 (real, imag) pairs in a trailing dim [2];
        complex64/complex128 build native complex arrays from their component dtype."""
        if self._dtype == "complex32":
            shape = list(self._shape) + [1]
            real = self._random("float16", shape, distribution=distribution)
            imag = self._random("float16", shape, distribution=distribution, is_complex_imag=True)
            return numpy.concatenate((real, imag), axis=-1)
        bits = int(str(self._dtype)[7:])
        fp_dtype = f"float{bits // 2}"
        real = self._random(fp_dtype, self._shape, distribution=distribution)
        imag = self._random(fp_dtype, self._shape, distribution=distribution, is_complex_imag=True)
        return numpy.array(real + imag * 1j)

    def _generate_e8m0(self):
        """float8_e8m0 only represents positive scale powers; generate a positive range."""
        low = float(self._data_range[0])
        high = float(self._data_range[1])
        if not numpy.isfinite(low) or low <= 0:
            low = 2**-3
        if not numpy.isfinite(high) or high <= low:
            high = 2**7
        f32 = self._gen_uniform_data(low, high, "float32", self._shape)
        return f32.astype(resolve_numpy_dtype("float8_e8m0"))

    @staticmethod
    def _replace_none_in_data_range(data_range: Union[list, tuple]) -> list:
        low, high = get(data_range, 0), get(data_range, 1)
        if low is None:
            low = DEFAULT_LOW
        if high is None:
            high = DEFAULT_HIGH
        return [low, high, *data_range[2:]]

    @staticmethod
    def _digitize_inf_nan(r, dtype: str):
        # Integer endpoints carry no inf/nan and must bypass float64 entirely:
        # int64/uint64 bounds beyond 2^53 are not exactly representable there,
        # and the rounded bound leaks into the RNG range / constant fill.
        # Float endpoints and non-finite values keep the digitize semantics below.
        if isinstance(r, (int, numpy.integer)):
            return int(r)
        tmp = numpy.array([r], dtype="float64")
        if numpy.isinf(tmp):
            dtype_range = get_dtype_range(dtype)
            return dtype_range[0] if numpy.isneginf(tmp) else dtype_range[1]
        if numpy.isnan(tmp):
            return 0
        return tmp[0]

    def _must_contain_values(self, dtype, is_complex_imag: bool) -> tuple:
        # border value will always be included
        replace_list = list(set(self._data_range))
        if _is_native_integer(dtype):
            # 整型端点不经 float64：INT64_MAX 经 float64 会舍入到 2^63 再溢出成 INT64_MIN，
            # 声明的区间端点会拿到符号相反的值。
            info = numpy.iinfo(dtype)
            clamped = [
                min(max(int(x), int(info.min)), int(info.max)) for x in replace_list if numpy.isfinite(numpy.float64(x))
            ]
            replace_np_array = numpy.array(clamped, dtype=dtype)
        else:
            candidates = numpy.array(replace_list, dtype="float64")
            if not is_float_dtype(dtype):
                # int/bool cannot represent inf/nan: digitize to dtype bounds / 0,
                # or the astype below yields undefined values (which rc>1 mixing
                # would then inject into the output).
                candidates = numpy.array([self._digitize_inf_nan(x, dtype) for x in candidates], dtype="float64")
            replace_np_array = candidates.astype(dtype)
        # dedupe again after the dtype cast: distinct declared values can
        # collapse into one in a narrower dtype (e.g. 3.5 vs 3.6 in bfloat16)
        replace_list = list(set(replace_np_array))
        if is_float_dtype(dtype):
            # handle -float(0) & float(0): set() keeps only one signed zero
            # (0.0 == -0.0), so restore the other when both were declared
            positive_zero = [is_positive_zero(x) for x in self._data_range]
            negative_zero = [is_negative_zero(x) for x in self._data_range]
            if any(positive_zero) and any(negative_zero):
                replace_list[replace_list.index(0)] = 0
                replace_list.append(-float(0))
        if is_complex_imag:
            # remove inf/nan
            replace_list = [x for x in replace_list if numpy.isfinite(x)]
        return tuple(replace_list)

    @staticmethod
    def _spans_dtype_range(dtype, low, high) -> bool:
        """True when BOTH endpoints reach (>=99% of) their dtype bounds AND straddle zero.

        The range form that makes linear uniform waste the dtype's dynamic
        range, switching _random's uniform family to log-uniform sampling.
        Each side is checked separately: an asymmetric range like [-1, finfo.max]
        must stay on the bounded path, because the log-uniform sampler ignores
        the declared bounds and would emit values outside input_data_ranges'
        closed-interval contract.
        """
        if not is_float_dtype(dtype):
            return False
        try:
            dtype_min, dtype_max = get_dtype_range(dtype)
        except Exception:
            return False
        if dtype_min >= 0 or dtype_max <= 0:
            return False
        return low < 0 < high and low <= dtype_min * 0.99 and high >= dtype_max * 0.99

    @staticmethod
    def _gen_exponential_data(dtype, shape) -> numpy.ndarray:
        """Generate log-uniform full-range samples into a pre-allocated typed buffer chunk by chunk.

        Values are sign(uniform) * 10^randint, capped below dtype max so the cast
        never overflows to inf. Small path (<= 2*CHUNK_ELEMS) is bitwise identical
        to a single full-size call. The chunked path interleaves the sign and
        exponent draws per chunk (a single full-size call draws all signs first),
        so under a fixed seed values above the threshold differ from one full-size
        call — the log-uniform distribution itself is unchanged.
        """
        try:
            info = numpy.finfo(dtype)
            max_v = float(info.max)
            tiny = float(info.tiny)
        except (TypeError, ValueError):
            _, max_v = get_dtype_range(dtype)
            s = str(dtype)
            if "float16" in s:
                tiny = float(numpy.finfo(numpy.float16).tiny)
            elif "bfloat" in s:
                tiny = float(numpy.finfo(numpy.float32).tiny)
            else:
                tiny = float(numpy.finfo(numpy.float64).tiny)
        low_exp = int(numpy.log10(tiny)) + 1
        high_exp = int(numpy.log10(max_v)) + 1
        if low_exp >= high_exp:
            return RandomData._gen_uniform_data(-1.0, 1.0, dtype, shape)
        elem_count = shape_numel(shape)
        if elem_count <= 2 * CHUNK_ELEMS:
            arr = numpy.random.uniform(low=-1.0, high=1.0, size=shape)
            arr_exp = numpy.random.randint(low=low_exp, high=high_exp, size=shape)
            arr_pow = numpy.power(10.0, arr_exp.astype(numpy.float64))
            result = numpy.multiply(arr, arr_pow)
            return result.astype(dtype, copy=False)
        out = numpy.empty(shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, elem_count, CHUNK_ELEMS):
            end = min(start + CHUNK_ELEMS, elem_count)
            arr = numpy.random.uniform(low=-1.0, high=1.0, size=end - start)
            arr_exp = numpy.random.randint(low=low_exp, high=high_exp, size=end - start)
            flat[start:end] = numpy.multiply(arr, numpy.power(10.0, arr_exp.astype(numpy.float64)))
        return out

    @staticmethod
    def _gen_normal_data(gen, dtype, shape):
        """Generate truncated-normal samples into a pre-allocated typed buffer chunk by chunk.

        gen.rvs consumes one numpy RandomState stream element by element, so the chunked
        output is bitwise identical to a single full-size rvs(...).astype(dtype) call.
        """
        elem_count = shape_numel(shape)
        if elem_count <= 2 * CHUNK_ELEMS:
            return gen.rvs(shape).astype(dtype, copy=False)
        out = numpy.empty(shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, elem_count, CHUNK_ELEMS):
            end = min(start + CHUNK_ELEMS, elem_count)
            flat[start:end] = gen.rvs(end - start)
        return out

    @staticmethod
    def _gen_uniform_data(low, high, dtype, shape):
        """Generate uniform samples into a pre-allocated typed buffer chunk by chunk.

        numpy.random.uniform consumes the RandomState stream element by element
        (low + (high-low) * next_double), so the chunked output is bitwise identical
        to a single full-size uniform(...).astype(dtype) call.
        """
        elem_count = shape_numel(shape)
        if elem_count <= 2 * CHUNK_ELEMS:
            return numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
        out = numpy.empty(shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, elem_count, CHUNK_ELEMS):
            end = min(start + CHUNK_ELEMS, elem_count)
            flat[start:end] = numpy.random.uniform(low, high, end - start)
        return out

    @staticmethod
    def _gen_integer_data(low, high, dtype, shape):
        """原生整型直接用整型 RNG 生成，不经 float64。

        其余 dtype 仍走 _gen_uniform_data(float64 再 astype)——那条路对 ≤32 位整型无损
        (2^32 < 2^53)、对浮点也无损(往窄了转)，且能覆盖 bfloat16/float8/int4 这些 numpy
        无原生 RNG 的类型。唯独 64 位整型的值域超出 float64 的 53 位尾数：
          * > 2^53 的值只能落在 2 的幂格点上——一个奇数都生成不出来；
          * INT64_MAX 舍入到 2^63，astype(int64) 后**静默溢出**成 INT64_MIN
            (只有一条 RuntimeWarning)，声明的区间端点拿到的是符号相反的值。
        """
        info = numpy.iinfo(dtype)
        lo = max(int(math.floor(low)), int(info.min))
        hi = min(int(math.ceil(high)), int(info.max))
        lo = min(lo, hi)
        # 种子取自**已播种的全局 legacy RNG**，而不是 default_rng() 的 OS 熵：
        # 后者让原生整型输入无视 np.random.seed，同一用例每次跑数据都不同
        # （batch_seed 的可复现承诺对整型形同虚设，免上传的配方也无从复算）。
        generator = numpy.random.default_rng(numpy.random.randint(0, 2**32))
        return generator.integers(lo, hi, size=shape, endpoint=True, dtype=dtype)

    def _random(
        self, dtype, shape: Union[list, tuple], is_complex_imag: bool = False, distribution: str = "uniform"
    ) -> numpy.ndarray:
        """Scalar-engine sampling: one decision level per orthogonal concern.

        L0 range form:      constant (single value / both-nan) → constant fill
        L1 distribution:    normal → truncated normal; otherwise the uniform family
        L2 uniform family:  dtype overflow clamp (float64) → range form:
                            spans dtype range → log-uniform, else linear uniform
        Tail:               must-contain border values mixed in (_mix_must_contain_values)

        The float64 full-range shadowing is emergent, not ordering-dependent:
        the /2 clamp leaves max_abs at 50% of dtype max, failing the 0.99
        full-range threshold on its own.
        """
        low, high = self._data_range[:2]
        if self._is_constant_range(low, high):
            array = self._fill_constant(low, dtype, shape, is_complex_imag)
        else:
            low = self._digitize_inf_nan(low, dtype)
            high = self._digitize_inf_nan(high, dtype)
            if distribution == "normal":
                array = self._gen_truncated_normal(low, high, dtype, shape)
            elif dtype == "hifloat4":
                dtype = "float4_e1m2"
                array = numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
            elif dtype in ("float64", "double"):
                finfo = numpy.finfo(numpy.float64)
                low = max(low, finfo.min / 2)
                high = min(high, finfo.max / 2)
                array = numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
            elif distribution == "uniform" and self._spans_dtype_range(dtype, low, high):
                array = self._gen_exponential_data(dtype, shape)
            elif _is_native_integer(dtype):
                array = self._gen_integer_data(low, high, dtype, shape)
            else:
                low, high = self._clamp_overflow(low, high, dtype)
                if distribution == "uniform" and self._spans_dtype_range(dtype, low, high):
                    array = self._gen_exponential_data(dtype, shape)
                else:
                    array = self._gen_uniform_data(low, high, dtype, shape)
        return self._mix_must_contain_values(array, dtype, shape, is_complex_imag)

    @staticmethod
    def _is_constant_range(low, high) -> bool:
        """Constant-fill condition: the CSV 固定值 convention (min == max) or both endpoints nan.

        Note -0.0 == 0.0 counts: fills -0.0 (sign kept), mixing can re-add +0.0.
        A single nan endpoint is NOT constant — the sampling path digitizes it to 0.
        """
        return low == high or all(numpy.isnan(numpy.array([low, high], dtype="float64")))

    def _fill_constant(self, value, dtype, shape, is_complex_imag: bool) -> numpy.ndarray:
        fill_val = 0 if is_complex_imag and not numpy.isfinite(value) else value
        if not is_float_dtype(dtype):
            # int/bool cannot represent inf/nan: digitize like the sampling
            # path does (inf -> dtype bound, nan -> 0), or numpy.full casts
            # them to undefined values with only a RuntimeWarning.
            fill_val = self._digitize_inf_nan(fill_val, dtype)
        return numpy.full(shape, fill_value=fill_val, dtype=dtype)

    @staticmethod
    def _clamp_overflow(low, high, dtype):
        """float64 uniform overflows computing high-low at ±finfo.max; halve the bounds.

        This clamp is also what shadows the full-value-range log-uniform path for
        float64 (max_abs drops to 50% of dtype max, below the 0.99 threshold).
        """
        if dtype in ("float64", "double"):
            finfo = numpy.finfo(numpy.float64)
            return max(low, finfo.min / 2), min(high, finfo.max / 2)
        return low, high

    def _gen_truncated_normal(self, low, high, dtype, shape) -> numpy.ndarray:
        from scipy.stats import truncnorm

        mean = (high + low) / 2
        sigma = (high - mean) / 3
        gen = truncnorm((low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma)
        return self._gen_normal_data(gen, dtype, shape)

    @staticmethod
    def _sample_distinct_positions(total: int, count: int) -> numpy.ndarray:
        """Draw `count` distinct random positions from [0, total) in O(count) time/memory.

        PERFORMANCE CONTRACT — keep this O(count), never O(total):
        do not "simplify" this back to numpy.random.choice(total, count,
        replace=False): choice internally builds a full permutation of `total`
        (~61MB temp for 8M elements, ~320MB for 40M) where this stays at KB
        scale; see the module docstring for the benchmark reference.

        Collision-driven retries converge in ~count^2/2total expected rounds,
        which is negligible while per_count stays capped (2048) — keep the cap
        if you touch the per_count formula, otherwise the retry loop degenerates
        at high sampling density (k/n of 25% once measured 70s+).
        """
        positions = numpy.unique(numpy.random.randint(0, total, size=count))
        while positions.size < count:
            extra = numpy.random.randint(0, total, size=count - positions.size)
            positions = numpy.unique(numpy.concatenate((positions, extra)))
        # unique returns sorted positions; shuffle so that slicing per value does
        # not concentrate each must-contain value in one region of the tensor.
        return numpy.random.permutation(positions)

    def _mix_must_contain_values(
        self, np_array: numpy.ndarray, dtype, shape: Union[list, tuple], is_complex_imag: bool
    ) -> numpy.ndarray:
        replace_list = self._must_contain_values(dtype, is_complex_imag)
        replace_count = len(replace_list)
        if replace_count > 1 and np_array.size > 0:  # more than one distinct must-contain value
            if np_array.size <= replace_count:
                # replace_list 先落成 np_array.dtype：直接 concatenate typed 数组与
                # Python float 列表会被 numpy 提升成 float64（如 bf16 输入 + range(-0,+0)
                # 的小 tensor 被静默改成 double）。
                candidate = numpy.concatenate((np_array.reshape([-1]), numpy.array(replace_list, dtype=np_array.dtype)))
                idx = numpy.random.permutation(candidate.size)
                # use copy() to discard view of `candidate`
                np_array = candidate[idx][: np_array.size].reshape(shape).copy()
            else:
                per_count = (
                    1 if np_array.size <= 4 * replace_count else min(int(0.25 * np_array.size / replace_count), 2048)
                )
                replace_idx = self._sample_distinct_positions(np_array.size, per_count * replace_count)
                values = numpy.repeat(numpy.array(replace_list, dtype=np_array.dtype), per_count)
                np_array.flat[replace_idx] = values
        return np_array
