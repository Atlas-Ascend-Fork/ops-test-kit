# ACLNN算子测试指南


---

# 环境准备

1、基本环境配置

- 建议Python 3.8+
- 完成CANN包安装（包含aclnn头文件和动态库）
- 确保环境变量已source

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
```

4、检查环境

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

# 测试用例编写

CSV 字段说明详见 [ACLNN 用例编写](./ACLNN_Case_Writing.md)。

以 aclnnCat 为例：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes
cat_tensorlist_fp32,aclnnCat,"(((3,3),(3,2)),(3,5),)","(('float32','float32'),'float32')",{'dim': -1},"(1,)"
```

更多示例参考 `examples/case_store/aclnn/`（单文件 `cases.csv` 覆盖 dtype 与基本测试场景，配套 `golden.py` golden 插件，详见目录内 README）。

# 精度测试

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

## 执行流程

ACLNN模式的执行流程如下：

```
读取CSV → 生成输入张量/标量 → 生成Golden（CPU） → 调用aclnn* C API → 精度比对 → 输出结果
```

> Golden 先于设备执行生成；多卡（HCCL）用例的 Golden 延迟到执行后。

> 精度比对方法详见[精度比对方法](../Precision_Comparison.md)，Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 性能测试

ACLNN 模式默认会采集Profiling性能数据：

```shell
# 默认执行（含Profiling）
python3 -m ttk aclnn -i cases.csv

# 关闭任务级Profiling采集（仍执行被测算子）
python3 -m ttk aclnn -i cases.csv --task-prof=false

# 设置执行次数（默认板端3次）
python3 -m ttk aclnn -i cases.csv --run=5
```

## 性能相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 执行次数 | 板端3次 |
| `--warmup` | Profiling前预热 | 开启 |
| `--task-prof` | 任务级Profiling采集开关 | 开启 |

> 注意：`--no-prof` 不是性能参数：只生成输入和 Golden、不执行被测通路，适用于离线数据准备，详见[离线数据准备与导入](../Offline_Data_Prepare_and_Import.md)。

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk aclnn -i cases.csv

# 使用2张卡
python3 -m ttk aclnn -i cases.csv --dev=2

# 指定使用卡0
python3 -m ttk aclnn -i cases.csv --devices=0
```

# 调试

```shell
# 固定随机种子
python3 -m ttk aclnn -i cases.csv --seed 42
```

Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 常用场景示例

```shell
# 运行全部示例用例（25 例；golden 插件承接 torch 同名映射不适配的 API）
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py

# 运行单个用例（-t 按用例名过滤）
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast

# 使用自定义Golden插件
python3 -m ttk aclnn -i my_cases.csv --plugin /path/to/my_golden.py
```

> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../Task_Execution.md)
