# Dump 数据调试

通过 `--dump` 系列参数保存输入、输出、Golden 和远端 XPU 数据到磁盘，便于事后分析精度问题、复现失败用例、或离线比对。

## 适用场景

| 场景 | 命令组合 | 说明 |
|------|---------|------|
| 主动 Dump 全量数据 | `--dump full` | 保存输入+输出+Golden+XPU（无远端配置时自动跳过 XPU） |
| 选择性 Dump | `--dump in,golden` | 只保存指定类别 |
| XPU 输出 Dump | `--dump xpu --config ttk.conf.yaml` | 保存远端第三方输出 |
| 失败时自动 Dump | `--dump-on-fail` | 精度比对失败时自动补保存输入/输出/Golden（不含 `xpu` 槽） |
| 离线数据准备 | `--no-prof --dump in,golden` | 不上板，可恢复数据落盘到 manual-data 目录，详见[离线数据准备与导入](./Offline_Data_Prepare_and_Import.md) |

## 1. Dump 类别

| 参数值 | 保存内容 | 时机 |
|--------|---------|------|
| `in` | 输入 tensor（含 tiling data） | 输入生成后、设备执行前 |
| `out` | NPU 输出 tensor | 设备执行后、精度比对前 |
| `golden` | Golden tensor | Golden 生成后 |
| `full` | 以上全部（含 XPU） | 各阶段分别保存 |
| `xpu` | 远端 XPU 返回的第三方输出 | XPU dispatch 成功后 |

可组合指定，逗号分隔：`--dump in,golden`。`full` 代表全部四类（含 XPU）；
未配置远端 XPU（`ttk.conf.yaml` / `--config`）时 XPU 位自动跳过——有数据才
落盘，不因缺配置报错。

## 2. Dump 格式

通过 `--dump-format` 指定保存格式：

| 格式 | 参数值 | 说明 | 适用分析方式 |
|------|--------|------|-------------|
| 二进制 | `bin`（默认） | 原始字节，无 shape 信息 | 十六进制查看、按 dtype 手动解析 |
| NumPy | `npy` | `.npy` 文件，含 dtype 和 shape | `numpy.load()` 直接加载 |
| PyTorch | `pt` | `.pt` 文件，含 dtype 和 shape | `torch.load()` 直接加载 |
| 终端打印 | `print` | 直接输出到终端 | 快速目视检查 |

### 2.1 格式选择建议

- **npy**：最推荐，Python 生态通用，`numpy.load()` 即可用
- **pt**：需要 Torch 张量语义（含自定义 dtype）时使用
- **bin**：最小体积，但需额外信息（dtype/shape）才能解析
- **print**：仅用于小规模数据快速查看

### 2.2 特殊 dtype 处理

`complex32` 在 npy/pt 格式下按 `float16` 存储；`int4`/`float4` 在 bin 格式下为打包字节，npy 格式下自动解包；pt 格式下从 bin 字节载荷解包时回退写为 `.npy` 文件（4bit 解包产物 pt 不可承载）。

## 3. 文件命名与路径

### 3.1 输出路径

所有通路统一落盘到 dump 专属目录（不再散落工作目录根）：

```
{root_path}/dump/{case}/          # 默认
{NPU_DUMP_PATH}/{case}/           # 环境变量覆盖（所有通路一致）
```

- `{case}` 为用例子目录名：Kernel/ACLNN 模式下 CSV 的 `dump_file_prefix` 字段优先（E2E/GEIR 恒用用例名），目录名做特殊字符
  安全化，与 `--manual-data-dirs` store 目录同构
- 目录在首次实际写文件时才创建：未开启任何 dump 开关时不产生空目录
- 远端 XPU 输出例外：直接写入 `NPU_DUMP_PATH` 或 `root_path`（不进 `dump/{case}/`），
  以便直接归档为 `manual_xpu_binaries` 素材，见 3.2

```bash
export NPU_DUMP_PATH=/data/dump
python3 -m ttk e2e -i cases.csv --dump full
```

### 3.2 文件命名规则

统一命名协议（与 `--manual-data-dirs` store 文件名同构）：

```
{role}[_{segment}]_{index}_{dtype}.{ext}
```

| 通配 | 说明 | 示例 |
|------|------|------|
| `{role}` | 数据类别 | `input` / `output` / `golden` / `xpu` / `tiling_data` |
| `{segment}` | 形态/模式段（可选，见下表） | `dyn` / `eager` / `graph_cst` / ... |
| `{index}` | 槽位序号，**与 CSV 列槽位对齐**（占位槽跳过但不移位；例外：ACLNN 的 input 先剔除纯输出与 None 占位槽再编号，其后真实输入的序号会前移） | `0` / `1` |
| `{dtype}` | numpy dtype 名（bin 无自描述，进文件名）；bytes 无 dtype 时为 `none` | `float16` / `int32` |
| `{ext}` | 格式后缀 | `bin` / `npy` / `pt` |

各通路 segment 取值：

| 通路 | segment | 说明 |
|------|---------|------|
| Kernel | output: `dyn` / `cst` / `bin`；input: 仅 `dyn` | 三模式共用一份输入 |
| E2E | output: `eager` / `graph_cst` / `graph_dyn` / `graph_aclgraph` | 与 result 列前缀对应 |
| GEIR | output: `const` / `dynamic`；input 同 | 两编译模式 |
| ACLNN | 无 segment | — |
| XPU | 不适用本协议（远端输出直接写 `NPU_DUMP_PATH`/`root_path`，文件名 `{case}_xpu_golden_{index}.{ext}`，`{ext}` 仅 `npy`/`bin`：`npy` 格式写 `.npy`，其余格式（含 `pt`/`print`）写 `.bin`，见 3.1 与下方示例） | — |

示例（`add_01` 用例，npy 格式）：

```
dump/add_01/
├── input_dyn_0_float16.npy        # Kernel 输入（三模式共用）
├── tiling_data_dyn.npy            # Kernel 动态 tiling（无 dtype/idx 的特殊 role）
├── tiling_data_bin.npy            # Kernel binary（-b release）模式 tiling（const 模式无 tiling 落盘）
├── output_dyn_0_float16.npy       # Kernel 动态输出
├── output_cst_0_float16.npy       # Kernel 静态输出
├── output_eager_0_float32.npy     # E2E eager 输出
├── output_graph_cst_0_float32.npy # E2E 静态图输出
├── output_const_0_float32.npy     # GEIR const 输出
├── output_0_float32.npy           # ACLNN 输出（无 segment）
└── golden_0_float16.npy           # golden（各通路均无 segment）
```

远端 XPU 输出对齐归档回放契约（不进 `dump/{case}/`，见 3.1）：

```
add_01_xpu_golden_0.npy
add_01_xpu_golden_1.npy
```

多 provider 时按 provider 分目录（如 `torch/add_01_xpu_golden_0.npy`）。

## 4. 失败时自动 Dump

### 4.1 命令

```bash
python3 -m ttk kernel -i cases.csv --dump-on-fail
```

### 4.2 行为

- 精度比对**失败**时自动保存该用例的全部数据（输入+输出+Golden）
- 比对**通过**时不保存
- 不需要同时指定 `--dump`，`--dump-on-fail` 独立生效
- 文件名与常规 dump **完全相同**（无特殊前缀）：只补未开启的槽位，
  已通过 `--dump` 落盘的槽位不重复写

### 4.3 与 `--dump` 组合

可同时使用：`--dump full --dump-on-fail`。此时所有用例都保存正常数据，
失败用例仅补漏尚未开启的槽位（`full` 已含 in/out/golden 时无额外产物）。

## 5. 典型调试流程

### 5.1 单用例定位

```bash
# 1. 先跑一遍，用 --dump-on-fail 抓失败数据（-t 锁定单用例，独立日志目录自动生成）
python3 -m ttk kernel -i cases.csv -t add_01 --dump-on-fail

# 2. 用 npy 格式重新 dump，方便 Python 加载
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy

# 3. Python 中加载分析（统一目录 dump/{case}/）
python3 -c "
import numpy as np
out = np.load('dump/add_01/output_dyn_0_float16.npy')
golden = np.load('dump/add_01/golden_0_float16.npy')
diff = np.abs(out - golden)
print('max diff:', diff.max())
print('argmax:', np.unravel_index(diff.argmax(), diff.shape))
"
```

### 5.2 全量回归抓取

```bash
# 全量用例失败时自动 dump，npy 格式
python3 -m ttk kernel -i cases.csv --dump-on-fail --dump-format npy
```

### 5.3 固定随机种子复现

```bash
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy --seed 42
```

`--seed` 固定输入数据生成和 Golden 计算的随机性，确保可复现。

## 6. 参数约束

| 约束 | 说明 |
|------|------|
| `--manual-data-dirs`（不带 `--no-prof`） | load 方向：搜索已存用例数据；禁止 `--validate`，要求 `--golden-mode Enable`（Disable/Promote/XPU 均被拒绝），E2E 禁止 `--backend cpu`，Kernel 禁止 `--compile-only` |
| `--no-prof --dump in[,golden]` | dump 方向：可恢复数据落盘到（显式或默认）manual-data 目录；要求 `--dump-format` 为 bin/npy/pt；Kernel 需精确 `--dump in,golden` |
| `--dump xpu` | 需要远端 XPU 配置；未配置时自动跳过不报错（`--xpu-perf` 缺配置仍报错） |

## 7. 通路支持

所有通路（Kernel / ACLNN / GEIR / E2E）均支持 `--dump` 和 `--dump-on-fail`；
各通路均可显式使用 `--dump xpu` 保存成功返回的远端第三方输出。
