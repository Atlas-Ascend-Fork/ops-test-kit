# Kernel 通路示例用例

单文件 `cases.csv` 覆盖 Kernel 通路的主要功能面，每条用例的 `remark` 列即测试项。GEIR 模式复用本文件（GEIR 同样支持 shape-unknown 用例）。

```bash
K=examples/case_store/kernel
# 全量（动态/静态/二进制三模式一次跑齐，18 例）
python3 -m ttk kernel -i $K/cases.csv --plugin $K/golden.py -c -b release
```

## 基础精度链路（add / abs）

| 用例 | 测试项 |
|---|---|
| `add_fp16_broadcast` | fp16 标量广播，默认 mix_tolerance |
| `add_fp32_mustappear` | 值域区间定制 + 第 3 元必现值(0)；fp32 经 `AddTestSpec.tolerance` 走 stat_rel_err |
| `add_int32_boundary` | int32 上限边界值域 + 显式 precision_tolerances；整型自动 binary_equal |
| `add_int8_negative` | int8 负值域定点比对 |
| `add_bf16_bcast3d` | bfloat16 双侧广播（输出 rank=两输入维度并集） |
| `add_complex64` | complex64 双输入分别定域（第二个广播输入整体窄域控叠加幅度；实/虚部共用所在张量的值域） |
| `abs_int16_3d` / `abs_int64_2d` | 一元算子 dtype 补齐（int16 / int64） |

```bash
python3 -m ttk kernel -i $K/cases.csv --plugin $K/golden.py -c -b release \
    -t add_fp16_broadcast,add_fp32_mustappear,add_int32_boundary,add_int8_negative,add_bf16_bcast3d,add_complex64,abs_int16_3d,abs_int64_2d
```

## 结构特性

| 用例 | 测试项 |
|---|---|
| `concat_d_tensorlist_in` | TensorList 动态输入（3 张量变长拼接）+ 负维 concat_dim |
| `split_tensorlist_out` | TensorList 输出（12 张量）+ split_dim/num_split + 混合 dtype 输入 |
| `reduce_min_valuedepend` | ValueDepend 张量输入（axes 须在 attributes 指定取值）+ keep_dims |
| `zeros_like_constant` | 常量输出算子（输出恒 0，与输入值无关） |

```bash
python3 -m ttk kernel -i $K/cases.csv --plugin $K/golden.py -c -b release \
    -t concat_d_tensorlist_in,split_tensorlist_out,reduce_min_valuedepend,zeros_like_constant
```

## shape-unknown 输出

| 用例 | 测试项 |
|---|---|
| `non_zero_shape_unknown` | 输出 shape 由输入取值决定（`output_shape_unknown_indexes`）；CSV 声明 shape 仅为占位；shape 元数据协议槽不进精度比对（静态 kernel 不写该槽） |

```bash
python3 -m ttk kernel -i $K/cases.csv --plugin $K/golden.py -t non_zero_shape_unknown -c -b release
```

## 低比特 dtype 与量化判据

| 用例 | 测试项 |
|---|---|
| `pad_float8` | **float8_e5m2**（通用算子 Pad 零填充）；paddings 为 ValueDepend 张量输入（attributes 指定取值） |
| `pad_float4` | **float4_e2m1**（en_dtypes 承载，通用算子 Pad）；fp4 时尾轴 padFront 须偶数（2 元素/字节打包对齐） |
| `quant_int8` | fp16→int8 量化输出；int8 输出按 dtype 轴默认路由 `binary_equal`，如需 quant 判据（\|a-g\|≤1）可在 TestSpec `tolerance` 中声明 `{'int8': {'standard': 'quant'}}` |
| `requant_float8` | fp16→float8_e5m2 量化输出（dst_type=35），requant 判据（ULP≤1，仅 hifloat8 输出才自动路由，建议显式 `--compare requant`；默认 mixed 亦可通过本例） |

```bash
python3 -m ttk kernel -i $K/cases.csv --plugin $K/golden.py -c -b release \
    -t pad_float8,pad_float4,quant_int8
python3 -m ttk kernel -i $K/cases.csv --plugin $K/golden.py -t requant_float8 --compare requant -c -b release
```

## 重编译与 perf-only

| 用例 | 测试项 |
|---|---|
| `matmul_v3_optional_inputs` | 可选输入 bias/offset_w None 占位 + transpose_x1/x2 + `#enable_pad`；无内置 golden（perf-only）；多 tiling key 编译（~130 子目标，分钟级/模式），限额容器自动收敛编译并行度，必要时 `--pc 1` |

```bash
python3 -m ttk kernel -i $K/cases.csv -t matmul_v3_optional_inputs -c -b release
```

dtype 覆盖：float16 / float32 / bfloat16 / int8 / int16 / int32 / int64 / complex64 / float8_e5m2 / float4_e2m1。

用例编写与命令行细节见 `.claude/skills/ttk-how-write-case/` 与 `ttk-how-run-test`；判据详解见 `docs/Precision_Comparison.md`。
