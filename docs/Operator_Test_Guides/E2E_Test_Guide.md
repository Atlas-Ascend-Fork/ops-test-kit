# E2E算子测试指南


---

> 文档中出现的硬件厂商名仅作示例，TTK 通过配置驱动支持任意符合接口的硬件加速器。

# 环境准备

1、基本环境配置

- 建议Python 3.8+，安装PyTorch 2.0+
- NPU后端需额外安装 [torch_npu](https://gitcode.com/Ascend/pytorch)
- 完成CANN包安装

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

# 测试用例编写

CSV 字段说明详见 [E2E 用例编写](./E2E_Case_Writing.md)。

以 torch.add 为例：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
add_f32_01,torch.add,"((2,3,4),(2,3,4))","('float32','float32')",{'alpha':1.0}
```

更多示例参考 `examples/case_store/e2e/`。

# 精度测试

## 后端选择

E2E 模式由统一的 Backend 抽象层（`ttk.core_modules.framework_api.backends`）驱动 NPU/MLU/CPU 等后端，各后端共享同一套用例解析、输入生成与精度比对逻辑。CPU 后端通常用作 Golden 计算源。后端按配置 hardware segment（`yaml` 的 `frameworks.torch.<seg>`）自动选择可用后端，`--cpu` 强制 CPU 后端。

| 后端 | 依赖 |
|------|------|
| NPU | torch + torch_npu |
| MLU | torch (MLU) |
| CPU | torch |

## 执行命令

```shell
# 自动按配置 hardware segment 选择可用后端
python3 -m ttk e2e -i torch_ops.csv

# 强制 CPU 后端（常用于Golden生成；选 CPU 可跑的用例）
python3 -m ttk e2e -i torch_ops.csv -t add_f32_01 --cpu
```

## 执行流程

E2E模式的执行流程如下：

```
读取CSV → 生成输入张量 → 在待测后端调用API → 在CPU调用Golden API → 精度比对 → 输出结果
```

> 精度比对方法详见[精度比对方法](../Precision_Comparison.md)，Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 性能测试

E2E 模式默认会采集Profiling性能数据：

```shell
# 默认执行（含Profiling）
python3 -m ttk e2e -i torch_ops.csv

# 禁用Profiling
python3 -m ttk e2e -i torch_ops.csv --no-prof

# 设置执行次数（默认板端3次）
python3 -m ttk e2e -i torch_ops.csv --run=5
```

## 性能相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 执行次数 | 板端3次 |
| `--warmup` | Profiling前预热 | 开启 |
| `--aclgraph` | 启用 ACLGraph，默认开启算子在线编译，允许复用编译缓存 | 关闭 |
| `-b` / `--binary` | `--aclgraph -b` 或 `--aclgraph -b=release` 复用已有算子二进制，关闭算子在线编译并保留图捕获和重放；`-b=false` 恢复在线编译。启用时需搭配 `--aclgraph`，不能与 `--super-kernel` 同用 | 关闭 |
| `--no-prof` | 禁用Profiling采集 | 关闭 |
| `--core-limit` | 限制核数上限：`N` 仅限 AI 核，`a,v` 分别限制（如 `2,32`），未设置的一侧回落物理核数。GE 图模式映射 `ge.aicoreNum`；ACLGraph 模式通过 `scope.limit_core_num` 限制捕获前的预处理算子，并在每次图捕获中记录核数限制；eager 不支持 | 无 |
| `--super-kernel` | 被测算子走 SuperKernel 编译流程（功能验证）：以 api_name 为 scope 标定，GE 图模式（-c/-d）与 aclgraph 模式均适用。需搭配 -c/-d/--aclgraph，不支持 --cpu | 关闭 |

```shell
# ACLGraph 默认开启算子在线编译
python3 -m ttk e2e -i torch_ops.csv --aclgraph

# 复用已有算子二进制，保留 ACLGraph 捕获和重放
python3 -m ttk e2e -i torch_ops.csv --aclgraph -b
# 等价写法：--aclgraph -b=release 或 --aclgraph --binary
```

`-b` 在 E2E 中仅控制 ACLGraph 的算子在线编译；`--fullgraph` 仍独立控制整图捕获。
普通 GE 图模式（`-c`/`-d`）未指定 `--aclgraph` 时不能启用此选项。
任何实际触发的算子在线编译一旦失败，即使后端回退产生了输出，结果仍记为 FAIL。

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk e2e -i torch_ops.csv

# 使用2张卡
python3 -m ttk e2e -i torch_ops.csv --dev=2

# 每张卡2个进程
python3 -m ttk e2e -i torch_ops.csv --pc=2

# 指定使用卡0
python3 -m ttk e2e -i torch_ops.csv --device-whitelist=0
```

# 调试

```shell
# 调试单个用例
python3 -m ttk e2e -i torch_ops.csv -t add_f32_01 --single-log

# 固定随机种子（可复现）
python3 -m ttk e2e -i torch_ops.csv --seed 42

# 仅校验CSV用例格式（不下设备）
python3 -m ttk e2e -i torch_ops.csv --validate
```

Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 常用场景示例

```shell
# torch.add 基础测试（自动选择可用后端）
python3 -m ttk e2e -i examples/case_store/e2e/torch_ops.csv -t add_f32_01

# torch_npu.npu_conv2d（使用golden_api，需设备支持该算子）
python3 -m ttk e2e -i examples/case_store/e2e/torch_ops.csv -t npu_conv2d_f16

# 强制 CPU 后端（选 CPU 可跑的用例）
python3 -m ttk e2e -i examples/case_store/e2e/torch_ops.csv -t add_f32_01 --cpu

# 输出结果
python3 -m ttk e2e -i torch_ops.csv -o results.csv

# Excel 多 sheet 用例（默认首个工作表；--sheet 指定工作表）
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.xlsx
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.xlsx --sheet T2
```

> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../Task_Execution.md)
