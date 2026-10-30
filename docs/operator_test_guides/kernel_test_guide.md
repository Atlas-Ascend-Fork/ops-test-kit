# Kernel 算子测试指南

## 使用须知

`ttk kernel` 完成算子内核的编译（或二进制匹配）与 NPU 执行，并做精度比对。执行前需备好 CANN 环境与算子包（自定义算子）。用例 CSV 的写法见 [Kernel 用例编写](./kernel_case_writing_guide.md)。

文档导览：

1. **[环境准备](#一环境准备)**
2. **[精度测试](#二精度测试)**
3. **[性能测试](#三性能测试)**
4. **[多卡并行](#四多卡并行)**
5. **[调试](#五调试)**
6. **[常用场景示例](#六常用场景示例)**

## 一、环境准备

### 1. 基本环境

- Python 3.8+，安装 PyTorch（Golden 计算使用）
- 完成 CANN 包安装并 source 环境变量
- 依赖系统工具链 `make`（AscendC 构建链，可用 `make --version` 检查）

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

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

## 二、精度测试

```shell
# 基本执行（默认动态 shape 编译）
python3 -m ttk kernel -i add.csv

# 指定设备
python3 -m ttk kernel -i add.csv --devices=0

# 输出结果
python3 -m ttk kernel -i add.csv -o results.csv
```

执行流程：

```
读取CSV → 编译/匹配内核并 tiling → 生成输入数据 → 生成 Golden（CPU）→ NPU 执行 → 精度比对 → 输出结果
```

### 1. 编译模式

| 参数 | 编译模式 | 说明 |
|------|---------|------|
| `-d`（默认） | 动态 shape 编译 | 用动态 shape 编译，经 tiling 得到实际 shape 后执行 |
| `-c` | 静态 shape 编译 | 用固定 shape 编译后直接执行 |
| `-b release` | 二进制模式 | 使用预编译的发布内核 |

```shell
python3 -m ttk kernel -i add.csv -d            # 动态 shape 编译（默认）
python3 -m ttk kernel -i add.csv -d false      # 关闭动态 shape 编译
python3 -m ttk kernel -i add.csv -c            # 静态 shape 编译
python3 -m ttk kernel -i add.csv -b release    # 二进制模式（使用发布内核）
```

### 2. 仅编译

```shell
python3 -m ttk kernel -i add.csv --co
```

### 3. 编译控制参数

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--dynamic` | `-d` | 动态 shape 编译；`-d false` 关闭 | **开启** |
| `--const` | `-c` | 静态 shape 编译 | 关闭 |
| `--binary` | `-b` | 二进制模式；`-b release` 使用发布内核 | 关闭 |
| `--compile-only` | `--co` | 仅编译不执行 | 关闭 |
| `--no-prof` | | dry-run：不执行被测通路，仅生成输入与 Golden；与 `--dump in,golden` 组合用于离线数据准备 | 关闭 |
| `--compile-opts` | | 编译选项（KEY=VALUE 格式，可多次指定），详解见 [Kernel 编译选项使用技巧](../tips/kernel_compile_options.md) | 无 |
| `--tiling-run` | `--tr` | Tiling 运行次数 | 3 |
| `--reuse-hbm` | | 重复执行间复用同一块 HBM（配合默认 3 次执行使能 L2 Cache） | 关闭 |
| `--reserve-hbm` | | 预留 HBM 内存（MB） | 无 |
| `--clear-atomic` | | 强制在算子执行前清零输出和 workspace | 关闭 |
| `--clear-ub` / `--clear-l1` / `--clear-l0` | | 执行前将 UB / L1 / L0 填充为指定值（默认清零） | 关闭 |
| `--simt-ub` / `--simt-stack-dcu` / `--simt-stack-dvg` | | SIMT 模式 UB / DCU 栈 / DVG 栈大小 | 无 |
| `--limit` | `-l` | 单用例输入+输出总量 HBM 上限（GB），超过则跳过 | 30 |
| `--force-block-dim` | | 强制指定 block_dim | 无 |
| `--core-limit` | | 限制 tiling 可见核数（`N` 限 AI 核，`a,v` 分别限 AI / Vector 核，如 `2,32`）；与 `--force-block-dim` 不同，仅设上限 | 无 |

示例：

```shell
# 编译选项
python3 -m ttk kernel -i cases.csv --compile-opts op_debug_config=oom,dump_cce --compile-opts enable_deterministic_mode=1

# 透传编译宏到 ccec 命令行（空格分隔，可引号包裹）
python3 -m ttk kernel -i cases.csv --compile-opts "macros=-DFOO=1 -DBAR=2"

# 多次执行间复用HBM（使能L2 Cache）
python3 -m ttk kernel -i cases.csv --reuse-hbm
```

> [!NOTE]
> 精度比对方法详见[精度比对方法](../reference/precision_comparison.md)，Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 三、性能测试

Kernel 模式默认采集 Profiling 性能数据：

```shell
python3 -m ttk kernel -i add.csv                 # 默认执行（含 Profiling）
python3 -m ttk kernel -i add.csv --task-prof=false   # 关闭任务级 Profiling（仍执行被测算子）
python3 -m ttk kernel -i add.csv --run=5         # 执行次数（默认板端 3 次）
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 执行次数 | 板端 3 次 |
| `--warmup` | Profiling 前预热 | 开启 |
| `--task-prof` | 任务级 Profiling 采集开关 | 开启 |
| `--npu-timeout` | NPU 执行超时（ms） | 无限制 |

> [!IMPORTANT]
> `--no-prof` 不是性能参数：只生成输入和 Golden、不执行被测通路，适用于离线数据准备，详见[离线数据准备与导入](../tips/offline_data_prepare_guide.md)。

## 四、多卡并行

```shell
python3 -m ttk kernel -i add.csv               # 使用全部可用 NPU 卡
python3 -m ttk kernel -i add.csv --dev=2       # 使用 2 张卡
python3 -m ttk kernel -i add.csv --pc=2        # 每张卡 2 个进程
python3 -m ttk kernel -i add.csv --devices=0,1 # 指定使用卡 0 和卡 1
```

## 五、调试

```shell
python3 -m ttk kernel -i add.csv -t add_01        # 单用例（独立日志目录自动生成）
python3 -m ttk kernel -i add.csv --seed 42        # 固定随机种子（可复现）
python3 -m ttk kernel -i add.csv --validate       # 仅校验 CSV 用例格式（不下设备）
```

Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 六、常用场景示例

```shell
# 运行 MatMulV3（可选输入 bias/offset_w 以 None 占位 + enable_pad 编译选项）
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t matmul_v3_optional_inputs

# 运行 Split 算子（带常量输入）
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t split_tensorlist_out -c

# 运行 ConcatD 算子（TensorList）
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t concat_d_tensorlist_in

# 重跑精度失败的用例
python3 -m ttk kernel -i add.csv --rerun=precision_status

# 使用自定义 Golden 插件
python3 -m ttk kernel -i add.csv --plugin /path/to/my_golden.py

# Excel 多 sheet 用例
python3 -m ttk kernel -i cases.xlsx --sheet T2
```

> [!NOTE]
> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../task_execution_guide.md)
