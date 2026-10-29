# ACLNN 算子测试指南

## 使用须知

`ttk aclnn` 通过 aclnn\* C API 测试算子。用例 CSV 的写法见 [ACLNN 用例编写](./aclnn_case_writing_guide.md)。

文档导览：

1. **[环境准备](#一环境准备)**
2. **[精度测试](#二精度测试)**
3. **[性能测试](#三性能测试)**
4. **[多卡并行](#四多卡并行)**
5. **[调试](#五调试)**
6. **[常用场景示例](#六常用场景示例)**

## 一、环境准备

### 1. 基本环境

- Python 3.8+
- 完成 CANN 包安装并 source 环境变量

```shell
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

### 2. 自定义算子包（可选）

测试自定义算子时先完成编译部署，.run 包名为 `cann-ops-xxx-custom_linux-${arch}.run`（如 `cann-ops-math-custom_linux-x86_64.run`）：

**默认路径安装**

```shell
bash cann-ops-xxx-custom_linux-${arch}.run
```

需先 source 上述 CANN 环境变量。安装到 `${ASCEND_OPP_PATH}/vendors/` 下。

**指定路径安装**

```shell
bash cann-ops-xxx-custom_linux-${arch}.run --install-path=${install_path}
```

安装到 `${install_path}/vendors/${vendor_name}/` 下，需执行 `source ${install_path}/vendors/${vendor_name}/bin/set_env.bash` 使算子包生效。

> [!TIP]
> 未安装直接使用解压目录时，手动执行 `export ASCEND_CUSTOM_OPP_PATH="${算子包路径}"`。

### 3. 安装 TTK 并检查

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

> [!IMPORTANT]
> ACLNN 通路的输入生成与 Golden 计算依赖 `torch`（不在 `requirements.txt` 中，`pip3 install torch` 安装；使用 `npu_preprocess` 钩子或多卡 HCCL 用例时还需 `torch_npu`）。

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

## 二、精度测试

```shell
# 基本执行
python3 -m ttk aclnn -i cases.csv

# 指定设备
python3 -m ttk aclnn -i cases.csv --devices=0

# 使用余弦相似度比对
python3 -m ttk aclnn -i cases.csv --compare cosine

# 输出结果
python3 -m ttk aclnn -i cases.csv -o results.csv
```

执行流程：

```
读取CSV → 生成输入张量/标量 → 生成 Golden（CPU）→ 调用 aclnn* C API → 精度比对 → 输出结果
```

> [!NOTE]
> 精度比对方法详见[精度比对方法](../reference/precision_comparison.md)，Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 三、性能测试

ACLNN 模式默认采集 Profiling 性能数据：

```shell
python3 -m ttk aclnn -i cases.csv                  # 默认执行（含 Profiling）
python3 -m ttk aclnn -i cases.csv --task-prof=false    # 关闭任务级 Profiling（仍执行被测算子）
python3 -m ttk aclnn -i cases.csv --run=5          # 执行次数（默认板端 3 次）
```

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--run` | | 执行次数 | 板端 3 次 |
| `--warmup` | | Profiling 前预热 | 开启 |
| `--task-prof` | | 任务级 Profiling 采集开关 | 开启 |
| `--no-prof` | | dry-run：不执行被测算子，仅生成输入/Golden 并落盘（离线数据准备） | 关闭 |

ACLNN 无模式专属选项，其余参数（用例筛选/设备并行/精度控制/结果输出）见[任务执行](../task_execution_guide.md)。

## 四、多卡并行

```shell
python3 -m ttk aclnn -i cases.csv               # 使用全部可用 NPU 卡
python3 -m ttk aclnn -i cases.csv --dev=2       # 使用 2 张卡
python3 -m ttk aclnn -i cases.csv --devices=0   # 指定使用卡 0
```

## 五、调试

```shell
python3 -m ttk aclnn -i cases.csv --seed 42     # 固定随机种子（可复现）
```

Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 六、常用场景示例

```shell
# 运行全部示例用例（配套 golden 插件）
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py

# 运行单个用例（-t 按用例名过滤）
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast

# 使用自定义 Golden 插件
python3 -m ttk aclnn -i my_cases.csv --plugin /path/to/my_golden.py
```

> [!NOTE]
> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../task_execution_guide.md)
