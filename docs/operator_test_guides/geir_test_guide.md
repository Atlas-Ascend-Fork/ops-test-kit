# GEIR 算子测试指南

## 使用须知

`ttk geir` 把算子组装为 GE 计算图后整体编译执行，覆盖 GE 图构建、图编译、算子调度等引擎层通路。用例 CSV 字段与 Kernel 相同、另加 `dyn_input_shapes` 列，写法见 [GEIR 用例编写](./geir_case_writing_guide.md)。

> [!TIP]
> GEIR 复用 Kernel 的 CSV 用例与 TestSpec 插件（注册名同 `op_name`），无需单独编写；需要精确控制图编译动态维度时，追加 `dyn_input_shapes` 列即可。

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
- 依赖系统工具链 `g++`（GE 图编译，可用 `g++ --version` 检查，缺失时 `yum install gcc-c++` 或 `apt install g++`）

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
# 基本执行（默认静态 shape 编译）
python3 -m ttk geir -i add.csv

# 指定设备
python3 -m ttk geir -i add.csv --devices=0

# 输出结果
python3 -m ttk geir -i add.csv -o results.csv
```

执行流程：

```
读取CSV → 生成输入数据 → 生成 Golden（CPU）→ 构建 GE 图并图编译（含 tiling）→ NPU 执行图 → 精度比对 → 输出结果
```

### 1. 编译控制参数

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--const` | `-c` | 静态 shape 编译 | **开启** |
| `--dynamic` | `-d` | 动态 shape 编译；`dyn_input_shapes` 仅在本模式生效 | 关闭 |
| `--binary` | `-b` | 二进制模式：复用预编译的发布内核 | 关闭 |
| `--xpu-perf` | | 采集 XPU 远程性能（需远端 XPU 配置） | 关闭 |
| `--rerun` | | 重跑失败项（可用列：`dyn_perf_us`/`cst_perf_us`/`precision_status`） | 无 |

> [!IMPORTANT]
> GEIR 默认开启静态编译（`-c`），与 Kernel 默认开启动态编译（`-d`）相反。`dyn_input_shapes` 仅在 `-d` 下生效，需显式指定。

```shell
python3 -m ttk geir -i add.csv            # 静态 shape 编译（默认）
python3 -m ttk geir -i add.csv -d         # 动态 shape 编译（需显式开启）
python3 -m ttk geir -i add.csv -c -d      # 静态 + 动态同时启用
python3 -m ttk geir -i add.csv -c=false -d    # 关闭静态，仅动态
python3 -m ttk geir -i add.csv -b         # 二进制模式（复用发布内核）
```

> [!NOTE]
> 精度比对方法详见[精度比对方法](../reference/precision_comparison.md)，Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 三、性能测试

GEIR 模式默认采集 Profiling 性能数据：

```shell
python3 -m ttk geir -i add.csv       # 默认执行（图执行 3 轮，首轮不进性能窗口）
python3 -m ttk geir -i add.csv --run=5   # 图执行轮数
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 图执行轮数（含义与 kernel/aclnn 相同）：`> 1` 时首轮（图编译/装载冷启动）不进性能窗口，其余 `run - 1` 轮 Task Duration 跨轮取 median；`1` 采单次冷启动；`--dl 1` 时多轮兼做轮间 MD5 一致性检查 | 3 |
| `--task-prof` | 任务级 Profiling 开关 | 开启 |

## 四、多卡并行

```shell
python3 -m ttk geir -i add.csv               # 使用全部可用 NPU 卡
python3 -m ttk geir -i add.csv --dev=2       # 使用 2 张卡
python3 -m ttk geir -i add.csv --pc=2        # 每张卡 2 个进程
python3 -m ttk geir -i add.csv --devices=0,1 # 指定使用卡 0 和卡 1
```

## 五、调试

```shell
python3 -m ttk geir -i add.csv -t add_01        # 单用例（独立日志目录自动生成）
python3 -m ttk geir -i add.csv --seed 42        # 固定随机种子（可复现）
python3 -m ttk geir -i add.csv --validate       # 仅校验 CSV 用例格式（不下设备）
```

Dump 调试详见[Dump 数据调试](../dump_debug_guide.md)。

## 六、常用场景示例

```shell
# 用 Kernel 的 CSV 跑 GEIR 图模式
python3 -m ttk geir -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast

# 动态 shape 编译
python3 -m ttk geir -i examples/case_store/kernel/cases.csv -t matmul_v3_optional_inputs -d

# 使用自定义 Golden 插件（与 Kernel 共用同一份插件）
python3 -m ttk geir -i add.csv --plugin /path/to/my_golden.py

# GEIR 通路示例用例（dyn_input_shapes + 历史缺陷回归）
python3 -m ttk geir -i examples/case_store/geir/cases.csv --plugin examples/case_store/geir/golden.py -c -d

# 重跑精度失败的用例
python3 -m ttk geir -i add.csv --rerun=precision_status
```

> [!NOTE]
> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../task_execution_guide.md)
