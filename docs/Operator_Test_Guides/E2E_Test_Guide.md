# E2E算子测试指南


---

> 文档中出现的硬件厂商名仅作示例，TTK 通过配置驱动支持任意符合接口的硬件加速器。

# 环境准备

## torch 测试环境准备

1、基本环境配置

- 建议Python 3.8+，安装PyTorch 2.0+
- NPU后端需额外安装 [torch_npu](https://gitcode.com/Ascend/pytorch)，**torch 与 torch_npu 必须主次版本配套**（经测试组合：`torch==2.12.0` + `torch-npu==2.12.0.post2`，CANN 9.3.0；其余组合见 torch_npu 官方配套矩阵）。`pip install ".[e2e-npu]"` 的 extras 已带版本上限约束，避免 pip 解析到配套矩阵之外的最新组合
- 完成CANN包安装

> **版本不配套的典型症状**：`import torch_npu` 直接崩溃（退出码 134），报
> `terminate called after throwing an instance of 'c10::ValueError' what(): Duplicated key
> 'pinned_reserve_segment_size_mb'`——报错栈不指向版本问题，实为 torch 与 torch_npu
> 的 libc10/libtorch_npu 注册冲突。按配套矩阵重装配套版本组合即可恢复。

```shell
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

2、自定义算子包（可选）

若测试自定义算子，请先完成编译部署。安装 .run 包时支持两种安装路径，包名格式为 `cann-ops-xxx-custom_linux-${arch}.run`（如 `cann-ops-math-custom_linux-x86_64.run`）：

**默认路径安装**

```shell
bash cann-ops-xxx-custom_linux-${arch}.run
```

需先完成上述 CANN 环境变量 source。安装到 `${ASCEND_OPP_PATH}/vendors/` 下。

**指定路径安装**

```shell
bash cann-ops-xxx-custom_linux-${arch}.run --install-path=${install_path}
```

安装到 `${install_path}/vendors/${vendor_name}/` 下，需执行 `source ${install_path}/vendors/${vendor_name}/bin/set_env.bash` 使算子包生效，`set_env.bash` 自动设置 `ASCEND_CUSTOM_OPP_PATH` 环境变量。

> 若未安装直接使用解压目录，可手动执行 `export ASCEND_CUSTOM_OPP_PATH="${算子包路径}"`。

3、TTK工具安装

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt

# NPU后端
pip install ".[e2e-npu]"

# MLU后端
pip install ".[e2e-xpu]"
```

4、检查环境

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

## TF 测试环境准备

1、基本环境配置

- 建议Python 3.8+，安装 TensorFlow 2.12+
- NPU后端需额外编包安装 npu_device，编包参考 [cann/tensorflow](https://gitcode.com/cann/tensorflow)
- 完成CANN包安装

```shell
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

2、自定义算子包（可选）

安装方式与 torch 测试相同，参见「torch 测试环境准备」第2步。

3、TTK工具安装

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt

# TF后端
pip install ".[tf]"

# 如需同时测试torch用例，加装 e2e-npu
# pip install ".[e2e-npu]"
```

4、检查环境

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

## onnx 测试环境准备

onnx 域（`api_name` 为 `onnx.<OpType>`）无需 torch/tensorflow/npu_device：被测腿经 atc 离线编译 .om 后由 aclmdl 执行，Golden 用 onnx ReferenceEvaluator。环境准备（onnx ≥ 1.13 且 opset ≤ 18 的版本约束、CANN atc）、用例约定（输出槽/属性/Variadic）、模式等价与失败码详见 [ONNX算子测试指南](./ONNX_Test_Guide.md)。

# 测试用例编写

CSV 字段说明详见 [E2E 用例编写](./E2E_Case_Writing.md)。

以 torch.add 为例：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
add_f32_alpha,torch.add,"((2,3,4),(2,3,4))","('float32','float32')",{'alpha':1.0}
```

更多示例参考 `examples/case_store/e2e/`。

# 精度测试

## 后端选择

E2E 模式由统一的 Backend 抽象层（`ttk.core_modules.common.backend`）驱动 NPU/CPU 后端，各后端共享同一套用例解析、输入生成与精度比对逻辑。CPU 后端通常用作 Golden 计算源。torch 用例按 `torch_npu` 可用性探测选择 NPU/CPU（探测失败回落 CPU 并告警）；TF 用例按 `npu_device` 是否存在选择 NPU/CPU。`--backend cpu` 强制 CPU 后端；三方标杆采集（`--golden-mode XPU` / `--xpu-perf`）仅经远端 endpoints，无本机执行后端。

| 后端 | 依赖 |
|------|------|
| NPU | torch 用例：torch + torch_npu；TF 用例：tensorflow + npu_device；onnx 用例：CANN（atc + libascendcl） |
| NPUSim | `--backend npusim` 仿真执行（camodel，详见 [NPUSim 仿真测试](../NPUSim/npusim_usage.md)） |
| CPU | torch 用例：torch；TF 用例：tensorflow（`--backend cpu` 强制）；onnx 用例：仅 Golden/输入生成（设备执行需 NPU） |

## 执行命令

```shell
# 自动探测可用后端（NPU 优先，失败回落 CPU）
python3 -m ttk e2e -i torch_cases.csv

# 强制 CPU 后端（常用于Golden生成；选 CPU 可跑的用例）
python3 -m ttk e2e -i torch_cases.csv -t add_f32_alpha --backend cpu
```

## 执行流程

E2E模式的执行流程如下：

```
读取CSV → 生成输入张量 → 在CPU生成Golden → 在待测后端调用API → 精度比对 → 输出结果
```

> Golden 先于设备执行生成（单卡静态用例）；动态 shape（`-1` 槽位）用例与 `--golden-mode XPU` 的 Golden 延迟到执行后装载。

> 精度比对方法详见[精度比对方法](../Precision_Comparison.md)，Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 性能测试

E2E 模式默认会采集Profiling性能数据：

```shell
# 默认执行（含Profiling）
python3 -m ttk e2e -i torch_cases.csv

# 关闭任务级Profiling采集（仍执行被测算子）
python3 -m ttk e2e -i torch_cases.csv --task-prof=false

# 设置执行次数（默认板端3次）
python3 -m ttk e2e -i torch_cases.csv --run=5
```

## 性能相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 执行次数 | 板端3次 |
| `--warmup` | Profiling前预热 | 开启 |
| `--task-prof` | 任务级Profiling采集开关 | 开启 |
| `--aclgraph` | 启用 ACLGraph（`torch.compile(dynamic=False)` + npugraph_ex backend），算子编译模式随 `-b` | 关闭 |
| `-b` / `--binary` | 二进制复用（默认开启，三种执行模式全生效）：eager 模式 `jitCompile=disable` 复用预编译 host 代码；`-c`/`-d` GE 图 `ge.jit_compile=0` 复用预编译 kernel；aclgraph 模式 `static_kernel_compile=False` 保留图捕获重放。`-b=false` 恢复在线编译。不能与 `--super-kernel` 显式同用；`--backend npusim` 下强制在线 | 开启 |
| `--no-prof` | 不执行被测通路，只生成输入/Golden 并落盘手动数据（离线数据准备，需搭配 `--dump in`） | 关闭 |
| `--core-limit` | 限制核数上限：`N` 仅限 AI 核，`a,v` 分别限制（如 `2,32`），未设置的一侧回落物理核数。GE 图模式映射 `ge.aicoreNum`；ACLGraph 模式经 `torch_npu.npu.npugraph_ex.scope.limit_core_num` 下发（每次图捕获均生效）；eager 不支持 | 无 |
| `--super-kernel` | 被测算子走 SuperKernel 编译流程（功能验证）：以 api_name 为 scope 标定，GE 图模式（-c/-d）与 aclgraph 模式均适用。需搭配 -c/-d/--aclgraph，不支持 --backend cpu | 关闭 |
```shell
# 默认（二进制复用）：eager/图/aclgraph 三种模式均复用预编译产物
python3 -m ttk e2e -i torch_cases.csv --aclgraph

# 恢复算子在线编译
python3 -m ttk e2e -i torch_cases.csv --aclgraph -b=false
# 等价写法：-b=no / -b=off / --binary=false
```

`-b` 在 E2E 中控制三种执行模式的算子编译（默认二进制复用）；`--fullgraph` 独立控制整图捕获。
`--super-kernel` 要求在线编译：显式 `-b --super-kernel` 报错，隐式默认二进制时自动回退在线编译并告警。
任何实际触发的算子在线编译一旦失败，即使后端回退产生了输出，结果仍记为 FAIL。

> 注意：`--no-prof` 不是性能参数：只生成输入和 Golden、不执行被测通路，适用于离线数据准备，详见[离线数据准备与导入](../Offline_Data_Prepare_and_Import.md)。

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk e2e -i torch_cases.csv

# 使用2张卡
python3 -m ttk e2e -i torch_cases.csv --dev=2

# 每张卡2个进程
python3 -m ttk e2e -i torch_cases.csv --pc=2

# 指定使用卡0
python3 -m ttk e2e -i torch_cases.csv --devices=0
```

# 调试

```shell
# 调试单个用例（单用例独立日志目录自动生成，无需参数）
python3 -m ttk e2e -i torch_cases.csv -t add_f32_alpha

# 固定随机种子（可复现）
python3 -m ttk e2e -i torch_cases.csv --seed 42

# 仅校验CSV用例格式（不下设备）
python3 -m ttk e2e -i torch_cases.csv --validate
```

Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 常用场景示例

```shell
# torch.add 基础测试（自动选择可用后端）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha

# torch_npu.npu_gelu（使用 golden_api，CPU golden 用 F.gelu）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv -t npu_gelu_f16

# 强制 CPU 后端（选 CPU 可跑的用例）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha --backend cpu

# 输出结果
python3 -m ttk e2e -i torch_cases.csv -o results.csv
```

> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../Task_Execution.md)
