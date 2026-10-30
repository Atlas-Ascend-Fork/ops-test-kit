# Kernel 编译选项（--compile-opts）

## 使用须知

`--compile-opts KEY=VALUE`（可多次指定）是 Kernel 模式向编译链路传参的入口，用于控制编译宏、调试开关、编译产物与构建行为。用法：

```shell
python3 -m ttk kernel -i cases.csv --compile-opts KEY1=VALUE1 --compile-opts KEY2=VALUE2
```

所有键值先收进编译选项字典，再沿两条路径消费：一条透传给 CANN 的 `build_config`（引擎配置，仅 CANN 注册过的键生效），一条把 `macros` 键的值直接拼进 ccec 编译命令行。

文档导览：

1. **[编译宏 macros](#一编译宏macros)**
2. **[调试选项](#二调试选项)**
3. **[编译产物](#三编译产物)**
4. **[构建开关](#四构建开关)**
5. **[注意事项](#五注意事项)**

## 一、编译宏 macros

`macros` 把 `-D` 宏或任意编译器参数追加到 ccec 命令行，两条编译路径（在线编译与手动 ccec 编译）都覆盖：

```shell
# 定义编译宏
python3 -m ttk kernel -i cases.csv --compile-opts "macros=-DFOO=1 -DBAR=2"

# 宏值含空格时用引号包裹
python3 -m ttk kernel -i cases.csv --compile-opts 'macros=-DCUSTOM_NAME="my add op"'
```

值按 shell 语义切分，可一次传多个宏；配合算子源码里的 `#ifdef` 分支即可切换实现。确认宏是否生效的办法：在算子源码中给输出加一个只随宏开启的偏移，带宏/不带宏各跑一次，前者精度比对从 PASS 翻转为 FAIL 即说明宏已进编译命令行。

## 二、调试选项

### 1. op_debug_config

逗号分隔的调试开关组合，也是 `oom` 溢出检测的开关：

| 子项 | 效果 | 示例 |
|------|------|------|
| `ccec_O0` | 编译加 `-O0`（关闭优化） | `--compile-opts op_debug_config=ccec_O0` |
| `ccec_g` | 编译加 `-g`（保留源码级调试信息） | `--compile-opts op_debug_config=ccec_g` |
| `oom` | 使能溢出检测 | `--compile-opts op_debug_config=oom` |
| `dump_cce` | 保留生成的 cce 中间文件（TBE 栈） | `--compile-opts op_debug_config=dump_cce` |
| `printf` | 使能内核 printf 调试输出 | `--compile-opts op_debug_config=printf` |

组合使用：

```shell
# 源码级调试 + 溢出检测
python3 -m ttk kernel -i cases.csv --compile-opts op_debug_config=ccec_O0,ccec_g,oom
```

### 2. tbe_debug_level

`tbe_debug_level=2` 开启全量源码级调试，等价于 `ccec_O0,ccec_g`，Ascend950 平台下 TTK 还会自动补全配套编译参数，使断点能定位到源码行：

```shell
python3 -m ttk kernel -i cases.csv --compile-opts tbe_debug_level=2
```

### 3. enable_deterministic_mode

值为 `1` 时，内置算子的源码实现会映射为 `-DDETERMINISTIC_MODE=1` 宏：

```shell
python3 -m ttk kernel -i cases.csv --compile-opts enable_deterministic_mode=1
```

注意该键的值以字符串透传，部分 CANN 版本在 tiling 侧对字符串值不兼容，传任何值都会导致 tiling 进程崩溃、用例报 FAIL。如无明确需要不要使用；确定性比对请改用 `--deterministic-level`。

## 三、编译产物

以下键控制编译中间产物的保留与输出，仅在 TBE 栈消费；AscendC 2.0 算子（如内置 add）的编译由 asc 栈完成，不读取这些键。

| 键 | 作用 | 产物位置 |
|----|------|---------|
| `save_temp_cce_file` | 编译后保留生成的 `.cce` 源码 | `kernel_meta/` 下与内核同名 |
| `dump_cce_code` | 生成的代码打印到日志 | 用例日志（`log/kernel/<用例名>/ttk.log`） |
| `dump_ir` | 导出编译中间 IR | 实现依赖 CANN 版本，部分版本无消费点 |

```shell
# 保留生成的 cce 源码，便于阅读生成的算子代码
python3 -m ttk kernel -i cases.csv --compile-opts save_temp_cce_file=1
```

## 四、构建开关

其余键均为 CANN `build_config` 的注册项（数量随版本变化，完整列表见 CANN 安装目录 `python/site-packages/asc_op_compile_base/common/buildcfg/default_buildcfg.py`）。常用：

| 键 | 示例 | 作用 |
|----|------|------|
| `enable_op_prebuild` | `--compile-opts enable_op_prebuild=1` | 算子预构建模式，算子 py 读到后跳过现场编译 |
| `enable_L1_fusion` | `--compile-opts enable_L1_fusion=1` | 开启 L1 融合 |
| `enable_auto_atomic` | `--compile-opts enable_auto_atomic=1` | 自动原子加优化 |
| `enable_vector_core` | `--compile-opts enable_vector_core=1` | 使用向量核编译 |
| `pipeline_opt` | `--compile-opts pipeline_opt=1` | 流水线优化 |
| `vector_fp_ceiling` | `--compile-opts vector_fp_ceiling=2` | 向量浮点上限 |

## 五、注意事项

1. **值一律按字符串透传，非空即"开"**：`build_config` 消费端多把值当布尔用，`save_temp_cce_file=0` 实际仍是开启状态。想关闭某个键，就不要传它。
2. **未知键会被静默丢弃**：不在 CANN 注册表内的键只产生告警 `build_config cannot set attribute '<key>'`，用例日志中可见，不会报错。
3. **TBE 与 AscendC 栈消费的键不同**：`save_temp_cce_file`/`dump_cce_code`/`dump_ir` 只在 TBE 栈生效；AscendC 栈的编译命令组装只读 `op_debug_config`。给两类算子传编译宏请统一使用 `macros`。
4. **`--compile-opts` 属于全局编译控制**：对本次运行的全部用例生效，逐用例的编译控制不在该参数的职责范围内。

常见报错与检查项见 [FAQ 一本通](../faq/faq_guide.md)。
