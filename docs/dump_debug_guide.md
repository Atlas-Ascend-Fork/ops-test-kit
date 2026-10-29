# Dump 数据调试

## 使用须知

`--dump` 系列参数把输入、输出、Golden 与远端 XPU 数据存盘，用于分析精度问题、复现失败用例或离线比对。

文档导览：

1. **[适用场景](#一适用场景)**
2. **[类别与格式](#二类别与格式)**
3. **[命名与路径](#三命名与路径)**
4. **[自动转储](#四自动转储)**
5. **[调试流程](#五调试流程)**
6. **[约束与支持](#六约束与支持)**

## 一、适用场景

| 场景 | 命令组合 | 说明 |
|------|---------|------|
| 主动 Dump 全量数据 | `--dump full` | 保存输入+输出+Golden+XPU（无远端配置时自动跳过 XPU） |
| 选择性 Dump | `--dump in,golden` | 只保存指定类别 |
| XPU 输出 Dump | `--dump xpu --config ttk.conf.yaml` | 保存远端第三方输出 |
| 失败时自动 Dump | `--dump-on-fail` | 精度失败时自动补保存输入/输出/Golden |
| 离线数据准备 | `--no-prof --dump in,golden` | 不上板，数据落盘到 manual-data 目录，详见[离线数据准备](./tips/offline_data_prepare_guide.md) |

## 二、类别与格式

### 1. Dump 类别

| 参数值 | 保存内容 | 时机 |
|--------|---------|------|
| `in` | 输入 tensor（含 tiling data） | 输入生成后、设备执行前 |
| `out` | NPU 输出 tensor | 设备执行后、精度比对前 |
| `golden` | Golden tensor | Golden 生成后 |
| `full` | 以上全部（含 XPU） | 各阶段分别保存 |
| `xpu` | 远端 XPU 返回的第三方输出 | XPU dispatch 成功后 |

可逗号组合（`--dump in,golden`）；未配置远端 XPU 时 `xpu` 位自动跳过。

### 2. Dump 格式

| 格式 | 参数值 | 适用分析方式 |
|------|--------|-------------|
| 二进制 | `bin`（默认） | 十六进制查看、按 dtype 手动解析 |
| NumPy | `npy` | `numpy.load()` 直接加载 |
| PyTorch | `pt` | `torch.load()` 直接加载 |
| 终端打印 | `print` | 快速目视检查 |

> [!TIP]
> 推荐 `npy`；需 Torch 张量语义用 `pt`；`bin` 最小但需 dtype/shape 才能解析；`print` 适合小数据。

特殊 dtype：`complex32` 在 npy/pt 下按 `float16` 存储；`int4`/`float4` 在 bin 下为打包字节、npy 下自动解包。

## 三、命名与路径

### 1. 输出路径

```
{root_path}/dump/{case}/          # 默认
{NPU_DUMP_PATH}/{case}/           # 环境变量覆盖（所有通路一致）
```

- `{case}` 为每个用例的子目录名，默认即用例名；Kernel/ACLNN 可在 CSV 用 `dump_file_prefix` 列另起名字（E2E/GEIR 无该列，恒用用例名）
- 用例名含特殊字符、以 `.`/`_` 开头结尾或超过 120 字符时，框架自动改用安全目录名（截断后附原名哈希，保证合法且不重名）
- 目录在首次写入 dump 文件时创建，未开启 `--dump` 不会产生空目录

```bash
export NPU_DUMP_PATH=/data/dump
python3 -m ttk e2e -i cases.csv --dump full
```

### 2. 文件命名规则

```
{role}[_{segment}]_{index}_{dtype}.{ext}
```

| 通配 | 说明 | 示例 |
|------|------|------|
| `{role}` | 数据类别 | `input` / `output` / `golden` / `xpu` / `tiling_data` |
| `{segment}` | 形态/模式段（可选，见下表） | `dyn` / `eager` / `graph_cst` / ... |
| `{index}` | 槽位序号，与 CSV 列槽位对齐 | `0` / `1` |
| `{dtype}` | numpy dtype 名（bin 进文件名）；bytes 无 dtype 时为 `none` | `float16` / `int32` |
| `{ext}` | 格式后缀 | `bin` / `npy` / `pt` |

bin 格式下 golden 与 xpu 文件名追加 `_shape_<shape-token>` 段（如 `golden_0_float32_shape_2x3.bin`），回放按文件名还原 shape；npy/pt 自带 shape，不带该段。

各通路 segment 取值：

| 通路 | segment |
|------|---------|
| Kernel | output：`dyn`/`cst`/`bin`；input：仅 `dyn` |
| E2E | output：`eager`/`graph_cst`/`graph_dyn`/`graph_aclgraph`（onnx 仅 `eager`） |
| GEIR | output：`const`/`dynamic`；input 同 |
| ACLNN | 无 segment |
| XPU | case 目录槽位：segment = provider（如 `xpu_torch_0_float16.bin`，bin 另带 `_shape_` 段，manual-data 回放消费） |

示例（`add_01` 用例，npy 格式）：

```
dump/add_01/
├── input_dyn_0_float16.npy        # Kernel 输入（三模式共用）
├── tiling_data_dyn.npy            # Kernel 动态 tiling
├── output_dyn_0_float16.npy       # Kernel 动态输出
├── output_cst_0_float16.npy       # Kernel 静态输出
├── output_eager_0_float32.npy     # E2E eager 输出
├── output_graph_cst_0_float32.npy # E2E 静态图输出
├── output_const_0_float32.npy     # GEIR const 输出
├── output_0_float32.npy           # ACLNN 输出（无 segment）
├── xpu_torch_0_float16.npy        # XPU 三方输出（provider 占 segment 位）
└── golden_0_float16.npy           # golden（一般无 segment）
```

XPU 槽位文件同样落盘在 `dump/{case}/`（segment 位为 provider），供 `--manual-data-dirs` 回放。

### 3. 从调试 dump 目录读回

`--manual-data-dirs` 可直接指向调试 dump 目录（`{root}/dump/`，即 `--dump` / `--dump-on-fail` 的产物）：

| 调试产物 | 读回处理 |
| --- | --- |
| `golden_{idx}_{dtype}[_shape_{dims}]`、`xpu_{provider}_{idx}_{dtype}[_shape_{dims}]` | 直接读（shape 段可选） |
| `input_dyn_{idx}_{dtype}`（kernel）、`input_{const,dynamic}_{idx}_{dtype}`（geir） | 去模式段映射到 input 槽；同槽多份要求字节一致 |
| `output[_{segment}]_{idx}_{dtype}`、`tiling_data[_{segment}]` | 忽略（读回会重新执行/编译产出） |
| `input_{eager,graph_*}_{idx}_{dtype}`（e2e dump-on-fail 补漏） | 拒绝（in-place API 可能已改写输入） |

限制：含 None 槽（空输入位）的用例无法回放；回放时 `dump_file_prefix` 须与导出时一致；目录不会自动清理旧文件，改 shape/dtype 后靠 dtype/字节校验发现不一致并报错。

## 四、自动转储

```bash
python3 -m ttk kernel -i cases.csv --dump-on-fail
```

- 精度比对**失败**时自动保存该用例的输入+输出+Golden，**通过**时不保存
- 独立生效，无需同时指定 `--dump`；文件名与常规 dump 完全相同，只补未落盘的槽位
- 可与 `--dump` 组合（`--dump full --dump-on-fail`）：正常数据全存，失败用例仅补漏

## 五、调试流程

### 1. 单用例定位

```bash
python3 -m ttk kernel -i cases.csv -t add_01 --dump-on-fail          # 失败时抓数据
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy  # npy 便于加载

python3 -c "
import numpy as np
out = np.load('dump/add_01/output_dyn_0_float16.npy')
golden = np.load('dump/add_01/golden_0_float16.npy')
diff = np.abs(out - golden)
print('max diff:', diff.max())
print('argmax:', np.unravel_index(diff.argmax(), diff.shape))
"
```

### 2. 全量回归抓取

```bash
python3 -m ttk kernel -i cases.csv --dump-on-fail --dump-format npy
```

### 3. 固定随机种子复现

```bash
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy --seed 42
```

## 六、约束与支持

| 约束 | 说明 |
|------|------|
| `--manual-data-dirs`（不带 `--no-prof`） | load 方向；禁止 `--validate`，要求 `--golden-mode Enable`，E2E 禁止 `--backend cpu`，Kernel 禁止 `--compile-only` |
| `--no-prof --dump in[,golden]` | dump 方向；要求 `--dump-format` 为 bin/npy/pt；Kernel 需精确 `--dump in,golden` |
| `--dump xpu` | 需远端 XPU 配置，未配置时自动跳过不报错 |

所有通路（Kernel / ACLNN / GEIR / E2E）均支持 `--dump` 与 `--dump-on-fail`；`--dump xpu` 各通路可用（E2E 的 onnx 除外：无远端分发通道，告警后跳过）。
