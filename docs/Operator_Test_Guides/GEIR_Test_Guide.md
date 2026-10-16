# GEIR算子测试指南


---

# 环境准备

1、基本环境配置

- 建议Python 3.8+，安装PyTorch（Golden计算使用）
- 完成CANN包安装，确保环境变量已source

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

> GEIR 通路生成 C++ 源码后调用 `g++` 编译，需系统工具链支持（可用 `g++ --version` 检查，缺失时通过系统包管理器安装，如 `yum install gcc-c++` 或 `apt install g++`）。

4、检查环境

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

> **GEIR 复用 Kernel 的 CSV 用例与 TestSpec 插件**：GEIR 模式以 `op_name` 为注册名，可直接复用 Kernel 的 CSV 用例与 Golden/插件资产（标准字段共 31 列 = Kernel 的 30 列 + GEIR 专属 `dyn_input_shapes`；另可按需追加 `attributes1`～`attributes9`，全量 40 列；可选列可省略——示例 `cases.csv` 实际仅用 14 列基础子集），无需为 GEIR 单独编写用例或 golden。需要精确控制图编译动态维度时，可选在末尾追加 `dyn_input_shapes` 列。

# 测试用例编写

GEIR 模式CSV字段说明详见 [GEIR用例编写](./GEIR_Case_Writing.md)。

以 `add.csv` 为例（与 Kernel 指南同一份文件）：

```csv
testcase_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries,manual_xpu_binaries
add_01,add,"((128, 1024), (1, 1024))","('float32', 'float32')","('ND',)","((128, 1024),)","('float32',)","('ND',)","((128, 1024), (1, 1024))","('ND',)","((128, 1024),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),(),()
```

# 精度测试

执行精度测试时，使用 `geir` 子命令：

```shell
# 基本执行（默认静态 shape 编译）
python3 -m ttk geir -i add.csv

# 指定设备
python3 -m ttk geir -i add.csv --devices=0

# 输出结果
python3 -m ttk geir -i add.csv -o results.csv
```

## 执行流程

GEIR 模式的完整执行流程如下：

```
读取CSV → 生成输入数据 → 生成Golden（CPU） → 构建GE图并图编译（含Tiling） → NPU执行图 → 精度比对 → 输出结果
```

与 Kernel 模式相比，GEIR 将算子组装为 GE 计算图后整体编译执行，覆盖 GE 图构建、图编译、算子调度等引擎层通路，验证算子在图模式下的行为。

## 编译模式

| 参数 | 编译模式 | 说明 | 默认值 |
|------|---------|------|--------|
| `-c` | 静态 shape 编译 | 用固定shape编译后执行 | **开启** |
| `-d` | 动态 shape 编译 | 用动态shape编译，经tiling得到实际shape后执行 | 关闭 |
| `-b` | 二进制模式 | `ge.jit_compile=0`，复用预编译的发布内核 | 关闭 |

> **注意**：GEIR 默认开启静态编译（`-c`），与 Kernel 默认开启动态编译（`-d`）相反。如需在 GEIR 下做动态 shape 编译，需显式指定 `-d`。

```shell
# 静态 shape 编译（默认）
python3 -m ttk geir -i add.csv -c

# 动态 shape 编译（需显式开启）
python3 -m ttk geir -i add.csv -d

# 静态 + 动态同时启用
python3 -m ttk geir -i add.csv -c -d

# 关闭静态，仅动态
python3 -m ttk geir -i add.csv -c=false -d

# 二进制模式（复用发布内核）
python3 -m ttk geir -i add.csv -b
```

> 精度比对方法详见[精度比对方法](../Precision_Comparison.md)，Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 性能测试

GEIR 模式默认会采集Profiling性能数据：

```shell
# 默认执行（含Profiling，每用例图执行 1 次）
python3 -m ttk geir -i add.csv

# 设置图执行轮数（仅在 --dl 1 确定性校验下生效，默认 3 轮）
python3 -m ttk geir -i add.csv --run=5 --dl=1
```

## 性能相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 图执行轮数，仅在 `--dl 1`（确定性校验）时生效；默认模式下每用例执行 1 次 | 3 |
| `--task-prof` | 任务级 Profiling 开关 | 开启 |

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk geir -i add.csv

# 使用2张卡
python3 -m ttk geir -i add.csv --dev=2

# 每张卡2个进程
python3 -m ttk geir -i add.csv --pc=2

# 指定使用卡0和卡1
python3 -m ttk geir -i add.csv --devices=0,1
```

# 调试

```shell
# 调试单个用例（单用例独立日志目录自动生成，无需参数）
python3 -m ttk geir -i add.csv -t add_01

# 固定随机种子（可复现）
python3 -m ttk geir -i add.csv --seed 42

# 仅校验CSV用例格式（不下设备）
python3 -m ttk geir -i add.csv --validate
```

Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 常用场景示例

```shell
# 用 Kernel 的 CSV 跑 GEIR 图模式
python3 -m ttk geir -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast

# 动态 shape 编译
python3 -m ttk geir -i examples/case_store/kernel/cases.csv -t matmul_v3_optional_inputs -d

# 使用自定义Golden插件（与 Kernel 共用同一份插件）
python3 -m ttk geir -i add.csv --plugin /path/to/my_golden.py

# GEIR 通路示例用例（dyn_input_shapes + 历史缺陷回归）
python3 -m ttk geir -i examples/case_store/geir/cases.csv --plugin examples/case_store/geir/golden.py -c -d

# 重跑精度失败的用例
python3 -m ttk geir -i add.csv --rerun=precision_status
```

> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../Task_Execution.md)
