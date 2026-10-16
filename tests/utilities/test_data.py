# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""data 模块公开面测试：RandomData（构造/generate/data_range）+ constant_array。

内部契约（分块逐位一致、必现值混入、分布策略）均经 generate() 入口或
同 seed 对拍锁定，不直接依赖私有实现细节的等价性重放。
"""

import numpy as np
import pytest

from ttk.utilities.data import CHUNK_ELEMS, RandomData


def _missing_en_dtypes():
    try:
        import en_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


@pytest.mark.parametrize(
    ("dtype", "shape", "data_range", "distribution"),
    [
        ("float32", (2,), (-0.0, 0.0), "uniform"),
        ("bfloat16", (2,), (-0.0, 0.0), "uniform"),
        ("float16", (2,), (-0.0, 0.0), "uniform"),
        ("bfloat16", (100,), (-0.0, 0.0), "uniform"),
        ("bfloat16", (0,), (-0.0, 0.0), "uniform"),
        ("float32", (2 * CHUNK_ELEMS + 8,), (-1.0, 1.0), "uniform"),
        ("bfloat16", (2 * CHUNK_ELEMS + 8,), (-1.0, 1.0), "uniform"),
        ("float32", (2 * CHUNK_ELEMS + 8,), (-1.0, 1.0), "normal"),
        ("bfloat16", (2 * CHUNK_ELEMS + 8,), (-1.0, 1.0), "normal"),
        ("bool", (40000,), (None, None), "uniform"),
        ("bool", (40000,), (0, 1), "uniform"),
        ("bool", (3, 4, 5), (None, None), "uniform"),
        ("bool", (0,), (None, None), "uniform"),
        ("bool", (8,), (True, True), "uniform"),
        ("bool", (8,), (False, False), "uniform"),
    ],
    ids=[
        "concat-mix-f32",
        "concat-mix-bf16",
        "concat-mix-f16",
        "flat-mix-bf16",
        "empty-guard-bf16",
        "chunked-uni-f32",
        "chunked-uni-bf16",
        "chunked-norm-f32",
        "chunked-norm-bf16",
        "bool-default-range",
        "bool-explicit-01",
        "bool-multidim",
        "bool-zero-axis",
        "bool-const-true",
        "bool-const-false",
    ],
)
def test_generate_dtype_preserved_across_paths(dtype, shape, data_range, distribution):
    """generate 全路径 dtype 保持声明值（防 float64 提升/降级）：

    concat-mix = size<=rc 混入（2053554 修复路径）；flat-mix = 按位覆写混入；
    empty-guard = size=0 跳过混入；chunked-* = 分块采样（uniform/normal）；
    bool-* = uniform 覆盖 0/1 时走离散域抽样：双值接近平衡、同 seed 可复现
    （连续 uniform astype(bool) 会全 True）。
    """
    arr = RandomData(dtype, shape, data_range).generate(distribution=distribution)
    assert str(arr.dtype) == dtype

    if dtype == "bool" and distribution == "uniform":
        np.random.seed(714)
        first = RandomData(dtype, shape, data_range).generate(distribution=distribution)
        np.random.seed(714)
        second = RandomData(dtype, shape, data_range).generate(distribution=distribution)
        assert str(first.dtype) == dtype
        assert np.array_equal(first, second)
        if first.size and np.unique(first).size == 2:
            frac_true = float(first.mean())
            assert 0.2 <= frac_true <= 0.8


@pytest.mark.parametrize(
    ("dtype", "max_val"),
    [("float32", 3.4028234663852886e38), ("float16", 65504.0)],
)
def test_full_value_range_covers_multiple_magnitudes(dtype, max_val):
    """全值域应走指数分布法，覆盖多个量级而非全是极大数。

    回归场景：uniform 在全值域下 97% 的值落在 1e37~1e38，
    小数几乎无法生成。改用指数分布后各量级概率均等。
    """
    np.random.seed(0)
    arr = RandomData(dtype, (5000,), (-max_val, max_val)).generate()
    assert str(arr.dtype) == dtype
    arr64 = arr.astype("float64")
    assert not np.any(np.isinf(arr64))
    assert not np.any(np.isnan(arr64))
    small = np.sum(np.abs(arr64) < 1e10)
    assert small > 500


def test_spans_dtype_range_threshold():
    """全值域判定阈值：上下界分别 ≥99% dtype 端点且跨零才触发；窄区间、int 全值域、非对称区间不触发。"""
    f32_max = 3.4028234663852886e38
    assert RandomData._spans_dtype_range("float32", -3.3895313892515355e38, 3.3895313892515355e38)  # ≈99.7%
    assert not RandomData._spans_dtype_range("float32", -1.0, 1.0)
    assert not RandomData._spans_dtype_range("float32", -100.0, 100.0)
    assert not RandomData._spans_dtype_range("int32", -2147483648, 2147483647)
    assert not RandomData._spans_dtype_range("float32", -1.0, f32_max)
    assert not RandomData._spans_dtype_range("float32", -f32_max, 1.0)


def test_asymmetric_full_range_stays_within_declared_bounds():
    """非对称全值域区间不得触发对称指数采样。

    旧判定只看较大端点绝对值，[-1, finfo.max] 误入全值域 log-uniform（忽略声明
    区间），seed=17 下 10000 样本有 1890 个低于 -1；修复后走受边界约束的均匀
    采样，全部落在闭区间内（端点经必现值混入保证取到）。
    """
    max_v = float(np.finfo("float32").max)
    np.random.seed(17)
    arr = RandomData("float32", (10000,), (-1.0, max_v)).generate()
    assert float(arr.min()) == -1.0
    assert float(arr.max()) <= max_v

    np.random.seed(17)
    mirror = RandomData("float32", (10000,), (-max_v, 1.0)).generate()
    assert float(mirror.min()) >= -max_v
    assert float(mirror.max()) == 1.0


@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "float64", "int32", "int64", "uint8", "bool"],
)
def test_chunked_bitwise_equal(dtype):
    """大 tensor 分块生成与单次全量逐位相同（uniform + normal，含非整除尾块 2C+1024）。"""
    n = 2 * CHUNK_ELEMS + 1024
    rd = RandomData(dtype, (n,), (-1.0, 1.0))

    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, n).astype(rd._dtype, copy=False)
    np.random.seed(42)
    chunked = RandomData._gen_uniform_data(-1.0, 1.0, rd._dtype, (n,))
    assert chunked.dtype == full.dtype
    assert np.array_equal(chunked, full)

    from scipy.stats import truncnorm

    low, high = rd._digitize_inf_nan(-1.0, rd._dtype), rd._digitize_inf_nan(1.0, rd._dtype)
    mean = (high + low) / 2
    sigma = (high - mean) / 3
    gen = truncnorm((low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma)
    np.random.seed(42)
    full = gen.rvs(n).astype(rd._dtype, copy=False)
    np.random.seed(42)
    chunked = rd._gen_normal_data(gen, rd._dtype, (n,))
    assert np.array_equal(chunked, full)


@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int32",
        "int64",
        "uint8",
        "int4",
        "float8_e5m2",
        "float8_e4m3fn",
        pytest.param(
            "float4_e2m1", marks=pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
        ),
        pytest.param("hifloat8", marks=pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")),
    ],
)
def test_chunked_direct_write_matches_astype(dtype):
    """resolve 后的各 dtype：np.empty 预分配 + 分块直写 cast 与整块 astype 逐位一致。"""
    from ttk.utilities.dtypes import resolve_numpy_dtypes

    resolved = resolve_numpy_dtypes([dtype])[0]
    src = np.random.uniform(-1.0, 1.0, 4096)
    chunk = 1000
    out = np.empty(4096, dtype=resolved)
    for start in range(0, 4096, chunk):
        end = min(start + chunk, 4096)
        out[start:end] = src[start:end]
    assert np.array_equal(out, src.astype(resolved, copy=False))


@pytest.mark.parametrize(
    "shape",
    [
        (2 * CHUNK_ELEMS + 1,),
        (3 * CHUNK_ELEMS - 1,),
        (3 * CHUNK_ELEMS,),
        (3, CHUNK_ELEMS // 2 + 17, 5),
    ],
    ids=["min-entry-2C+1", "max-tail-3C-1", "exact-multiple-3C", "multidim-ravel"],
)
def test_uniform_chunked_shapes_bitwise_equal(shape):
    """分块边界形状均与单次全量逐位相同：最小入口、最大尾块、整除无尾块、多维 ravel 写入顺序。"""
    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, shape).astype("float32", copy=False)
    np.random.seed(42)
    chunked = RandomData._gen_uniform_data(-1.0, 1.0, "float32", shape)
    assert chunked.shape == shape
    assert np.array_equal(chunked, full)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_large_float16_uniform_is_numpy_seeded(dtype):
    """>10M float16/bfloat16 走 numpy 分块路径：同 seed 逐位可复现（torch RNG 不受 numpy seed 控制，必失败）。"""
    n = 10_000_001  # 超过旧 torch 分支阈值
    rd = RandomData(dtype, (n,), (-1.0, 1.0))

    np.random.seed(42)
    first = rd.generate()
    np.random.seed(42)
    second = rd.generate()

    assert str(first.dtype) == dtype
    assert np.array_equal(first, second)
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0


def test_large_uniform_mix_expect_bitwise_reproducible():
    """>10M 混入走轻量采样路径：同 seed 整个 generate（含混入）逐位一致。"""
    n = 10_000_001
    rd = RandomData("float32", (n,), (-1.0, 1.0))

    np.random.seed(42)
    first = rd.generate()
    np.random.seed(42)
    second = rd.generate()

    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("shape", "data_range", "min_count"),
    [
        ((10_000_001,), (-1.0, 1.0, 3.5), 2048),  # 大 tensor：每值覆写封顶 2048
        ((3,), (-1.0, 1.0, 0.0), 1),  # 等容量：必现值全量出现
        ((4,), (-1.0, 1.0, 0.0), 1),  # 超容量控制
        ((2,), (-1.0, 1.0, 0.0), 0),  # 装不下：子集抽样
    ],
    ids=["large-capped-injection", "equal-capacity", "larger-control", "cannot-hold-all"],
)
def test_uniform_must_contain_values_present(shape, data_range, min_count):
    """data_range 端点与显式必现值必须混入生成结果；等容量覆写注入、装不下才子集抽样。"""
    np.random.seed(0)
    arr = RandomData("float32", shape, data_range).generate()
    assert arr.shape == shape
    for v in data_range:
        assert (arr == v).sum() >= min_count


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
def test_integer_reaches_dtype_bounds(dtype):
    """整型能取到自身值域的上下界。

    原实现所有 dtype 统一走 uniform(float64) 再 astype：对 ≤32 位整型无损
    （2^32 < 2^53），但 64 位整型值域超出 float64 的 53 位尾数——INT64_MAX 舍入到
    2^63 后 astype 静默溢出成 INT64_MIN，声明的区间端点拿到的是符号相反的值。
    """
    info = np.iinfo(dtype)
    arr = RandomData(dtype, (512,), (int(info.min), int(info.max))).generate()

    assert arr.dtype == np.dtype(dtype)
    assert (arr == info.max).any()
    assert (arr == info.min).any()


@pytest.mark.parametrize("high", [1 << 56, 1 << 60, 1 << 62, (1 << 63) - 1])
def test_int64_high_range_is_not_grid_quantized(high):
    """int64 高值域能生成任意值，而非落在 2 的幂格点上。

    float64 在 2^60 量级的最小间隔是 2^(60-52)=256，经它生成的值全是 256 的倍数，
    一个奇数都产不出——这会让"索引取大值"的用例实际只覆盖到格点上的少数取值。
    """
    arr = RandomData("int64", (2048,), (0, high)).generate()

    assert (arr % 2 == 1).any(), "高值域生成不出奇数，说明仍经 float64 量化"
    assert (arr == high).any(), "区间上端点未命中"


@pytest.mark.parametrize(
    ("dtype", "data_range", "expected"),
    [
        # 超 float64 精确范围（2^53）端点不得被舍入（expected=None 走端点断言）
        ("int64", (9007199254740993, 9007199254740996), None),
        ("uint64", (9223372036854775809, 9223372036854775812), None),
        ("int64", (9007199254740992, 9007199254740996), None),
        # 小数区间向内取整：生成与混入均不越出声明闭区间
        ("int32", (1.5, 2.5), {2}),
        ("int64", (1.5, 2.5), {2}),
        ("uint64", (0.5, 1.5), {1}),
        ("int32", (-2.5, -1.5), {-2}),
        ("int32", (-0.5, 0.5), {0}),
        ("int32", (0.5, 2.5), {1, 2}),
        # 显式额外必现值保持 int(x) 截断
        ("int32", (1, 2, 3.9), {1, 2, 3}),
        # 固定小数截断（(1.5, 1.5)→1）
        ("int32", (1.5, 1.5), {1}),
        # 区间内无可表示整数时明确拒绝
        ("int32", (1.2, 1.8), "reject"),
    ],
    ids=[
        "int64-2p53-plus-1",
        "uint64-2p63-plus-1",
        "int64-exact-control",
        "int32-pos-fraction",
        "int64-pos-fraction",
        "uint64-pos-fraction",
        "int32-neg-fraction",
        "int32-cross-zero-fraction",
        "int32-multi-int-subset",
        "extra-must-contain-truncation",
        "fixed-fraction-truncation",
        "empty-int-subset-reject",
    ],
)
def test_integer_range_contract(dtype, data_range, expected):
    """整数区间契约：2^53+ 端点不经 float64；非固定小数区间向内取整不越边界；
    显式额外必现值 int(x) 截断；固定小数截断；空整数集合拒绝。"""
    low, high = data_range[:2]
    if expected == "reject":
        with pytest.raises(ValueError, match="no representable"):
            RandomData(dtype, (128,), data_range).generate()
        return
    arr = RandomData(dtype, (8192,), data_range).generate()
    if expected is None:
        # 超 float64 精确整数范围（2^53）的整型端点不得被舍入：端点经 float64
        # 处理时 2^53+1 round-to-even 到 2^53，随机生成会产出低于声明下界的值。
        assert int(arr.min()) >= low
        assert int(arr.max()) <= high
        assert (arr == low).any()
        assert (arr == high).any()

        const = RandomData(dtype, (4,), (low, low)).generate()
        assert (const == low).all()
    else:
        assert set(arr.tolist()) == expected


def test_exponential_small_tensor_keeps_single_shot_path():
    """elem_count <= 2*CHUNK_ELEMS 时保持单次全量路径（逐位对齐改动前的实现）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS
    tiny = float(np.finfo("float32").tiny)
    max_v = float(np.finfo("float32").max)
    low_exp = int(np.log10(tiny)) + 1
    high_exp = int(np.log10(max_v)) + 1

    np.random.seed(42)
    arr = np.random.uniform(low=-1.0, high=1.0, size=(n,))
    arr_exp = np.random.randint(low=low_exp, high=high_exp, size=(n,))
    full = np.multiply(arr, np.power(10.0, arr_exp.astype(np.float64))).astype("float32", copy=False)

    np.random.seed(42)
    chunked = RandomData._gen_exponential_data("float32", (n,))
    assert chunked.dtype == full.dtype
    assert np.array_equal(chunked, full)


def test_exponential_chunked_actually_splits(monkeypatch):
    """大 tensor 必须真实走分块路径：uniform 调用次数 >= 2（防阈值被误改后静默退化为单发）。

    单发路径对每个全值域浮点 tensor 要开 uniform/randint/power/multiply 四个全尺寸
    float64|int64 临时量（32 倍元素数），dim0=2^30 时 34GB —— 容器内必 OOM，主进程被打掉
    后 rc 仍是 0，跑批静默截断。
    """
    import numpy

    from ttk.utilities.data import CHUNK_ELEMS

    calls = []
    real_uniform = numpy.random.uniform

    def counting_uniform(low, high, size):
        calls.append(size)
        return real_uniform(low, high, size)

    monkeypatch.setattr("ttk.utilities.data.numpy.random.uniform", counting_uniform)
    arr = RandomData._gen_exponential_data("float32", (2 * CHUNK_ELEMS + 1,))
    assert len(calls) >= 2
    assert max(calls) <= CHUNK_ELEMS  # 没有任何一次是全尺寸
    assert arr.dtype == numpy.dtype("float32")


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_exponential_chunked_dtype_and_magnitudes(dtype):
    """分块路径的输出 dtype/shape 正确，且仍覆盖多个数量级（不是退化成单一量级或全 0）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 1024
    arr = RandomData._gen_exponential_data(dtype, (n,))
    assert arr.dtype == np.dtype(dtype)
    assert arr.shape == (n,)

    nonzero = np.abs(arr[np.isfinite(arr) & (arr != 0)])
    assert nonzero.size > 0
    magnitudes = np.unique(np.floor(np.log10(nonzero.astype(np.float64))))
    assert magnitudes.size >= 3  # 指数分布应跨多个量级


def test_exponential_chunked_is_seed_reproducible():
    """同 seed 两次生成必须逐位相同（分块只改消费次序，不破坏确定性）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 7
    np.random.seed(20260917)
    first = RandomData._gen_exponential_data("float32", (n,))
    np.random.seed(20260917)
    second = RandomData._gen_exponential_data("float32", (n,))
    assert np.array_equal(first, second)
