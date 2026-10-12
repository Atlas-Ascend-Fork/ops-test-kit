# Kernel 流程 TestSpec 示例（numpy）

Kernel 流程在 CANN 底层运行，`golden` / `customize_inputs` 用 **numpy.ndarray**。参数名和顺序与算子 def.cpp 中的输入参数一致（不含输出）。

## golden 示例

### 简单逐元素算子

```python
import numpy

__spec__ = {"my_relu": "MyReluTestSpec"}

class MyReluTestSpec:
    def golden(x, **kwargs):
        '''Parameters follow my_relu_def.cpp without outputs.'''
        return [numpy.maximum(x, 0)]
```

### 带属性的算子（`*` 分隔）

```python
import numpy

class MyReduceTestSpec:
    def golden(x, *, axis=0, keepdims=False, **kwargs):
        '''
        Parameters follow my_reduce_def.cpp without outputs.

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name
        '''
        ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
        if "bfloat16" in str(ori_dtype).lower():
            x = x.astype("float32")
            return [numpy.sum(x, axis=axis, keepdims=keepdims).astype(numpy.dtype("bfloat16"), copy=False)]
        return [numpy.sum(x, axis=axis, keepdims=keepdims)]
```

### 多输出算子

```python
import numpy

class MySortTestSpec:
    def golden(x, descending=False, **kwargs):
        '''Returns [values, indices].'''
        values = numpy.sort(x, axis=-1)
        indices = numpy.argsort(x, axis=-1)
        if descending:
            values = numpy.flip(values, axis=-1)
            indices = numpy.flip(indices, axis=-1)
        return [values, indices]
```

## customize_inputs 示例

### 避免零值

```python
import numpy

class MyDivOpTestSpec:
    def golden(x, y, **kwargs):
        return [numpy.divide(x, y)]

    def customize_inputs(x, y, **kwargs):
        '''除法算子输入，避免零值。返回 (x, y)。'''
        x_data = numpy.random.uniform(0.1, 10.0, size=x.shape).astype(x.dtype)
        y_data = numpy.random.uniform(0.1, 10.0, size=y.shape).astype(y.dtype)
        return (x_data, y_data)
```

### 归约算子生成 axes

```python
import numpy

class ReduceMaxTestSpec:
    def golden(x, axes, **kwargs):
        return [numpy.max(x, axis=tuple(axes) if axes is not None else None, keepdims=False)]

    def customize_inputs(x, axes, **kwargs):
        '''axes 为 None（动态 shape）时生成全轴归约的 axes。返回 (x, axes_arr)。'''
        input_dtypes = kwargs.get('input_dtypes')
        axes_dtype = input_dtypes[0] if input_dtypes else numpy.int64
        if axes is None or (hasattr(axes, '__len__') and len(axes) == 0):
            axes_arr = numpy.array(tuple(range(len(x.shape))), dtype=axes_dtype)
        else:
            axes_arr = numpy.array(tuple(set(int(a) for a in axes)), dtype=axes_dtype)
        return (x, axes_arr)
```

### 索引算子生成关联输入

```python
import numpy

class GatherNdTestSpec:
    def golden(x, indices, **kwargs):
        # golden 实现
        ...

    def customize_inputs(x, indices, **kwargs):
        '''为 gather_nd 生成 x 和 indices。x 生成有序数据验证索引正确性。返回 (params, res_indices)。'''
        if str(x.dtype) != "bool":
            params = numpy.arange(0, x.size, 1, dtype=x.dtype).reshape(x.shape)
        else:
            params = numpy.random.choice(a=[False, True], size=x.shape, p=[0.5, 0.5])

        ranks = indices.shape[-1]
        res_indices = []
        for rank in range(ranks):
            idx = numpy.random.uniform(0, params.shape[rank], (1,)).astype(indices.dtype)
            res_indices.append(idx.item())
        for index in indices.shape[0:-1]:
            res_indices = res_indices * index
        res_indices = numpy.reshape(res_indices, indices.shape).astype(indices.dtype, copy=False)
        return (params, res_indices)
```

## tolerance

```python
class MyOpTestSpec:
    def golden(x, **kwargs):
        return [numpy.abs(x)]
    # 默认即 mix_tolerance（生态算子开源精度标准）；可省略或按需覆盖阈值
    tolerance = {"float32": {"standard": "mix_tolerance"}, "float16": {"standard": "mix_tolerance"}}
```

> `mix_tolerance` 可用 `rtol` / `atol` / `required_matched_ratio` / `max_abs_error_limit` 覆盖默认阈值表（`max_abs_error_limit` 配置后完全替代 ULP 动态上限，配置值即最终上限）；其余标准（`stat_rel_err`/`binary_equal`/`cross_check`/`quant`）见 SKILL.md tolerance 表。

## kwargs 字段

| 字段 | 说明 |
|------|------|
| `input_dtypes` | 输入 dtype 列表，如 `['float32', 'float32']` |
| `output_dtypes` | 输出 dtype 列表 |
| `input_ori_shapes` | 原始输入 shape |
| `output_ori_shapes` | 原始输出 shape |
| `input_formats` / `output_formats` | 张量格式，如 `['ND', 'NCHW']` |
| `input_ori_formats` / `output_ori_formats` | 原始张量格式 |
| `input_ranges` | 输入数据范围（customize_inputs 专用） |
| `full_soc_version` | SoC 版本全称 |
| `short_soc_version` | SoC 版本简称，如 `Ascend910B3` |
| `testcase_name` | 当前用例名称 |
| `golden_mode` | 本次调用的 golden 精度模式（`Enable`/`Disable`/`Promote`/`XPU`）。`Promote` 下输入已被框架抬一档精度（fp16/bf16→fp32、fp32→fp64），golden 可据此分支处理，避免把已抬精度再砍回来 |

## CSV 数据范围替代

如果只需要调整随机数据范围，可以不写 customize_inputs，直接在 CSV 中设置：

```csv
input_data_ranges,"((-1, 1), (0.1, 10))"
```

或使用 `--input-dist normal` 切换为正态分布。
