# E2E 算子测试指南

## 使用须知

`ttk e2e` 通过框架 API 端到端测试算子，内含 torch / tensorflow / onnx 三种 API（由 `api_name` 区分，`onnx.<OpType>` 即 onnx）。用例 CSV 的写法见 [E2E 用例编写](./e2e_case_writing_guide.md)。

> [!NOTE]
> 文档中出现的硬件厂商名仅作示例，TTK 通过配置驱动支持任意符合接口的硬件加速器。

文档导览：

1. **[环境准备](#一环境准备)**
2. **[精度测试](#二精度测试)**
3. **[性能测试](#三性能测试)**
4. **[多卡并行](#四多卡并行)**
5. **[调试](#五调试)**
6. **[常用场景示例](#六常用场景示例)**

## 一、环境准备

### 1. torch

- Python 3.8+，安装 PyTorch 2.0+
- NPU 后端需额外安装 [torch_npu](https://gitcode.com/Ascend/pytorch)，torch 与 torch_npu 必须版本配套（按官方配套矩阵选择；实测组合 `torch==2.12.0` + `torch-npu==2.12.0.post2`，CANN 9.3.0）。`pip install ".[e2e-npu]"` 已带版本上限约束
- 完成 CANN 包安装

> [!CAUTION]
> torch 与 torch_npu 版本不配套时 `import torch_npu` 直接崩溃（退出码 134），症状与修复见 [FAQ](../faq/faq_guide.md)。

```shell
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

### 2. tensorflow

- Python 3.8+，安装 TensorFlow 2.12+
- NPU 后端需额外编包安装 npu_device，编包参考 [cann/tensorflow](https://gitcode.com/cann/tensorflow)
- 完成 CANN 包安装并 source 环境变量（同上）

### 3. onnx

无需 torch/tensorflow：

- `pip install onnx`（onnx ≥ 1.13 且 opset ≤ 18，CANN 9.2.x 配套 onnx 1.13.x）
- CANN `atc`（被测腿 atc 编译 .om → aclmdl 执行）

失败码与缓存见 [FAQ 一本通](../faq/faq_guide.md)。

### 4. 自定义算子包（可选）

测试自定义算子时先完成编译部署，.run 包名为 `cann-ops-xxx-custom_linux-${arch}.run`（如 `cann-ops-math-custom_linux-x86_64.run`）：

```shell
bash cann-ops-xxx-custom_linux-${arch}.run                         # 默认路径安装
bash cann-ops-xxx-custom_linux-${arch}.run --install-path=${install_path}  # 指定路径安装
```

指定路径安装后需执行 `source ${install_path}/vendors/${vendor_name}/bin/set_env.bash` 使算子包生效。

> [!TIP]
> 未安装直接使用解压目录时，手动执行 `export ASCEND_CUSTOM_OPP_PATH="${算子包路径}"`。

### 5. 安装 TTK 并检查

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt

# NPU 后端
pip install ".[e2e-npu]"

# TF 后端
pip install ".[tf]"

# MLU 后端
pip install ".[e2e-xpu]"
```

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

## 二、精度测试

### 1. 后端选择

E2E 支持 NPU / CPU 后端，默认自动探测（无 NPU 环境回退到 CPU 并告警）；`--backend cpu` 强制 CPU。

| 后端 | 依赖 |
|------|------|
| NPU | torch 用例：torch + torch_npu；TF 用例：tensorflow + npu_device；onnx 用例：CANN（atc + libascendcl） |
| NPUSim | `--backend npusim` 仿真执行（详见 [NPUSim 仿真测试](../npusim/npusim_usage_guide.md)）；onnx 不支持，启动即报错 |
| CPU | torch 用例：torch；TF 用例：tensorflow（`--backend cpu` 强制）；onnx 用例：仅 Golden/输入生成（设备执行需 NPU） |

### 2. 执行命令

```shell
# 自动探测可用后端（NPU 优先，失败回退 CPU）
python3 -m ttk e2e -i torch_cases.csv

# 强制 CPU 后端（常用于 Golden 生成；选 CPU 可跑的用例）
python3 -m ttk e2e -i torch_cases.csv -t add_f32_alpha --backend cpu

# onnx（atc 编译 .om + aclmdl 执行）
python3 -m ttk e2e -i onnx_cases.csv
```

执行流程：

```
读取CSV → 生成输入张量 → 在CPU生成Golden → 在待测后端调用API → 精度比对 → 输出结果
```

> [!NOTE]
> 精度比对方法详见[精度比对方法](../reference/precision_comparison.md)，Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 三、性能测试

E2E 模式默认采集 Profiling 性能数据：

```shell
python3 -m ttk e2e -i torch_cases.csv                    # 默认执行（含 Profiling）
python3 -m ttk e2e -i torch_cases.csv --task-prof=false  # 关闭任务级 Profiling（仍执行被测算子）
python3 -m ttk e2e -i torch_cases.csv --run=5            # 执行次数（默认板端 3 次）
```

| 参数 | 缩写 | 支持 | 说明 | 默认值 |
|------|------|------|------|--------|
| `--run` | | 通用 | 执行次数 | 板端 3 次 |
| `--warmup` | | 通用 | Profiling 前预热 | 开启 |
| `--task-prof` | | 通用 | 任务级 Profiling 采集开关 | 开启 |
| `--no-prof` | | 通用 | dry-run：不执行被测算子，仅生成输入/Golden 并落盘（离线数据准备） | 关闭 |

编译类组合参数（需与其他开关搭配使用）：

| 参数 | 缩写 | 支持 | 说明 | 默认值 |
|------|------|------|------|--------|
| `-c` / `-d` | | torch / tf / onnx | 图模式静态 / 动态编译；tf 不支持 `-1` 动态槽；onnx：om 恒为静态图，`-d` 直接报错，`--eager -c` 双开单次执行、两列同值 | 关闭 |
| `--aclgraph` | | torch | 启用 ACLGraph 图捕获模式，算子编译模式随 `-b`；tf 不支持（告警后跳过）、onnx 直接报错 | 关闭 |
| `--binary` | `-b` | torch | 二进制复用（默认开启）：eager/GE 图/aclgraph 三种执行模式均复用预编译产物，`-b=false` 恢复在线编译 | 开启 |
| `--core-limit` | | torch | 限制核数上限：`N` 仅限 AI 核，`a,v` 分别限制（如 `2,32`）；eager 不支持 | 无 |
| `--super-kernel` | | torch | 被测算子走 SuperKernel 编译流程（功能验证），需搭配 `-c`/`-d`/`--aclgraph` | 关闭 |

onnx 性能为墙上时钟计时（`eager_device_perf_us`），kernel 明细为空，详见[性能测试](../performance_testing_guide.md)。

```shell
python3 -m ttk e2e -i torch_cases.csv --aclgraph          # 默认（二进制复用）
python3 -m ttk e2e -i torch_cases.csv --aclgraph -b=false # 恢复算子在线编译
```

> [!IMPORTANT]
> `--no-prof` 不是性能参数：只生成输入和 Golden、不执行被测通路，适用于离线数据准备，详见[离线数据准备与导入](../tips/offline_data_prepare_guide.md)。

## 四、多卡并行

```shell
python3 -m ttk e2e -i torch_cases.csv               # 使用全部可用 NPU 卡
python3 -m ttk e2e -i torch_cases.csv --dev=2       # 使用 2 张卡
python3 -m ttk e2e -i torch_cases.csv --pc=2        # 每张卡 2 个进程
python3 -m ttk e2e -i torch_cases.csv --devices=0   # 指定使用卡 0
```

## 五、调试

```shell
python3 -m ttk e2e -i torch_cases.csv -t add_f32_alpha   # 单用例（独立日志目录自动生成）
python3 -m ttk e2e -i torch_cases.csv --seed 42          # 固定随机种子（可复现）
python3 -m ttk e2e -i torch_cases.csv --validate         # 仅校验 CSV 用例格式（不下设备）
```

Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 六、常用场景示例

```shell
# torch.add 基础测试（自动选择可用后端）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha

# torch_npu.npu_gelu（使用 golden_api，CPU golden 用 F.gelu）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv -t npu_gelu_f16

# 强制 CPU 后端（选 CPU 可跑的用例）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha --backend cpu

# onnx全量示例
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv

# 输出结果
python3 -m ttk e2e -i torch_cases.csv -o results.csv
```

> [!NOTE]
> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../task_execution_guide.md)
