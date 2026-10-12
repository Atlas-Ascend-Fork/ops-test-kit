# GEIR 模式专用参数

## 执行方式

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| **在线编译**（默认） | `ge.jit_compile=1`，GE 图编译时在线编译 kernel | 验证算子全链路（GE 图编译 + kernel 编译 + 执行） |
| **二进制复用**（`-b release`） | `ge.jit_compile=0`，直接使用 CANN 预编译好的二进制 kernel | 验证发布态 kernel 的精度和性能 |

## 命令示例

```shell
# 在线编译+执行（默认）
python3 -m ttk geir -i cases.csv

# 指定单个用例
python3 -m ttk geir -i cases.csv -t case_name

# 二进制复用模式（使用 CANN 预编译 kernel）
python3 -m ttk geir -i cases.csv -b release

# 显式关闭二进制复用（等价于默认在线编译）
python3 -m ttk geir -i cases.csv -b false

# 加载 golden 插件
python3 -m ttk geir -i cases.csv --plugin /path/to/golden.py

# 保留生成的 C++ 源码和编译产物（默认即保留，无需额外参数）
python3 -m ttk geir -i cases.csv
```

## CSV 用例

GEIR 模式复用 Kernel 模式的 CSV 和 golden 资产，无需额外编写。模式由 CLI 子命令（`ttk geir`）决定，与 CSV 内容无关。

## 编译控制

| 参数 | 缩写 | 说明 | 默认 |
|------|------|------|------|
| `--binary` | `-b` | 二进制复用；`-b release` 使用 CANN 预编译 kernel，`-b false` 关闭 | 关闭（在线编译） |
| `--xpu-perf` | | 采集 XPU 远程性能；需远程 XPU 配置（`ttk.conf.yaml` 或 `--config`） | 关闭 |

> **注意**：GEIR **不支持** `--backend npusim` 仿真（仿真仅 kernel/aclnn/e2e 三模式支持）。

## 调试

C++ 源码和编译后的可执行文件默认保留在 `geir/` 目录下，输入 bin 文件执行后自动清理。

## 生成产物

| 目录/文件 | 说明 |
|-----------|------|
| `geir/ops/<op_name>.cpp` | 每个算子生成的 C++ 测试程序源码（按算子缓存复用） |
| `geir/const/<testcase_name>.json` | 静态 shape（默认）模式的图配置 |
| `geir/dynamic/<testcase_name>.json` | 动态 shape 模式（`-d`）的图配置 |
| `geir/const/binary/`、`geir/dynamic/binary/` | 二进制复用（`-b`）组合下对应模式的图配置目录 |

C++ 源码按算子名生成并缓存，图配置以用例名命名；执行后输入 bin 文件自动清理，C++ 源码和可执行文件默认保留。

## 结果 CSV

结果 CSV 额外包含以下列：

| 列名 | 说明 |
|------|------|
| `precision` | 最终精度结果（取通过的模式） |
| `cst_precision` / `cst_bin_precision` | 静态 shape 模式精度（列名随 `-b` 切换） |
| `dyn_precision` / `dyn_bin_precision` | 动态 shape 模式精度（列名随 `-b` 切换） |
| `cst_perf_us` / `cst_bin_perf_us` | 静态 shape 模式耗时（列名随 `-b` 切换） |
| `dyn_perf_us` / `dyn_bin_perf_us` | 动态 shape 模式耗时（列名随 `-b` 切换） |

## 与 Kernel 模式的对应关系

| Kernel 模式 | GEIR 模式 | GE 配置 |
|-------------|-----------|---------|
| 动态/静态 shape 编译（默认） | 在线编译（默认） | `ge.jit_compile=1` |
| 二进制执行（`-b release`） | 二进制复用（`-b release`） | `ge.jit_compile=0` |

## 环境变量

GE 日志默认写入 `~/ascend/log/debug/plog/`。如需更详细日志：

```shell
# DEBUG 级别日志
ASCEND_GLOBAL_LOG_LEVEL=0 python3 -m ttk geir -i cases.csv
```
