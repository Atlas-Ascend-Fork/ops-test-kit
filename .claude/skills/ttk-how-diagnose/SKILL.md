---
name: ttk-how-diagnose
description: 诊断 TTK 测试失败（跑完后结果不对/报错/超时排障）。只要用户提到测试跑不过、结果不对、精度差、编译挂了、执行超时、OOM、仿真报错（--backend npusim）、需要 dump 数据或排障，就必须使用此 skill。即使用户只是说"精度差一点"、"编译报错怎么解决"、"跑不过了"、"挂了/被杀了"、"仿真跑出来不对"、"无卡跑不了"、"SIM_RESULT_MISSING"等口语化表述，也应触发。首次正常运行用 ttk-how-run-test；定义/编写 golden 等资产用 ttk-how-write-plugin。
---

# 诊断 TTK 测试失败

## Step 1: 环境检查

```shell
python3 -m ttk -v        # 版本号
python3 -m ttk info       # 设备信息（NPU卡数、CANN版本）
```

TTK 启动时自动检测并 source CANN 环境（`setenv.bash`），通常无需手动设置。仅当 CANN 装在非标路径时设置 `ASCEND_CUSTOM_PATH`（详见 `ttk-how-run-test` Step 1）。若自动 source 失败，常见报错：`ImportError: libhccl.so`、`AttributeError: 'NoneType' object has no attribute 'acl_init'`。

## Step 2: 错误分类

| 类别 | 关键词 | 诊断方向 |
|------|--------|---------|
| **环境** | `not found`、`ImportError`、`ASCEND_HOME_PATH` | Step 1 环境检查 |
| **CSV 解析** | `parse`、`field`、`dtype`、`shape` | CSV 格式与字段 |
| **编译** | `compile`、`tiling`、`build`、`op not found` | 算子名/属性/shape |
| **执行** | `timeout`、`OOM`、`HBM`、`runtime` | 内存/超时参数 |
| **精度** | `precision`、`cosine`、`close`、`compare` | Step 4 精度调试 |
| **插件** | `plugin`、`register`、`golden` | 注册名/插件路径 |
| **E2E onnx** | `ATC_`、`OM_`、`ONNX_` | ONNX 通路错误表（见 `references/error-patterns.md`） |
| **仿真** | `npusim`、`sim`、`SIM_`、`wrapper`、`cannsim` | NPUSim 仿真排障（见 `references/npusim-errors.md`） |

## Step 3: 锁定单个用例调试

每个用例自动生成独立日志（`log/<mode>/<用例名>/ttk.log`），并行运行不会互相覆盖；用 `-t` 锁定单个失败用例可以隔离输出。`--dump-on-fail` 在精度失败时自动保存输入/输出/Golden 数据。

```shell
# 单用例 + 失败时dump（日志自动落 log/<mode>/<用例名>/ttk.log）
python3 -m ttk kernel -i cases.csv -t case_name --dump-on-fail

# 指定 dump 内容和格式
python3 -m ttk kernel -i cases.csv -t case_name --dump in,out,golden --dump-format npy
```

### 按类别的诊断命令

**环境问题**：

```shell
python3 -m ttk info                                    # 设备状态
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast  # 内置示例验证
```

**执行超时/OOM**：

```shell
python3 -m ttk kernel -i cases.csv --proc-timeout=300  # 加大超时
python3 -m ttk kernel -i cases.csv -l 10              # 跳过输入输出超过 10GB 的用例
python3 -m ttk kernel -i cases.csv --reserve-hbm=512   # 预留 HBM 512MB
```

**多卡问题**：

```shell
python3 -m ttk info                        # 检查设备
python3 -m ttk kernel -i cases.csv --devices=^2,^3  # 排除设备 2、3
python3 -m ttk kernel -i cases.csv --pc=1  # 减少进程数
```

**可复现性**：

```shell
python3 -m ttk kernel -i cases.csv --seed 42  # 固定种子（固定输入数据生成）
```

**重跑与校验**：

```shell
# 只重跑上次结果里精度失败的用例（-i 须为含 precision_status 列的结果 CSV，不能是原始用例 CSV）
python3 -m ttk kernel -i result.csv --rerun precision_status
python3 -m ttk kernel -i cases.csv --validate                 # 仅校验 CSV（不跑设备）
```

**NPUSim 仿真问题**：仿真报错与真机不同，先确认环境含 950-ops，再看 wrapper 产物：

```shell
python3 -m ttk kernel --backend npusim -i cases.csv -t case_name --sim-cores 0  # 单用例重跑
ls <sim_output>/<case>/wrapper_error.json     # wrapper 异常（内容非空即有错）
ls <sim_output>/<case>/record_out/cannsim_*/cannsim.log   # record 日志
```

完整仿真排障（`SIM_RESULT_MISSING`、`OPTILING_FAILURE`、`ACLNN_EXECUTE_FAILED`、卡死等）见 `references/npusim-errors.md`。

**插件不生效**：确认 `--plugin` 已传、类名/`__spec__` 注册名与 CSV 一致、插件无语法错。详见 `references/error-patterns.md` 插件错误表。

## Step 4: 精度调试流程

### 4.1 选择比对方法

| 场景 | 方法 | 参数 |
|------|------|------|
| 浮点常规（默认） | 混合容差（生态算子开源精度标准） | `--compare mixed`（默认） |
| 统计误差均值/最大值 | 统计相对误差（社区标准） | `--compare stat_rel_err` |
| 逐点 isclose | 数值近似 | `--compare close` |
| 大规模向量整体趋势 | 余弦相似度 | `--compare cosine` |
| 整型/精确结果 | 二进制精确 | `--compare binary` |
| hifloat8 类型 | 重量化 | `--compare requant`（自动） |
| 三方交叉校验 | 三方交叉校验 | `--compare cross_check`（需 `third_party`） |

> 默认未设 `--compare` 时，按 `Spec.tolerance` 逐输出路由（需 `--plugin`），否则 `mixed`。

### 4.2 Dump 数据分析

```shell
# 失败时自动 dump
python3 -m ttk kernel -i cases.csv -t case_name --dump-on-fail

# 指定 dump 内容
python3 -m ttk kernel -i cases.csv -t case_name --dump in,out,golden --dump-format npy
```

### 4.3 调整容差

容差首选 `Spec.tolerance`（TestSpec 中按 dtype 声明，详见 `ttk-how-write-plugin`）；无 plugin 时默认 `mixed` 走 dtype 阈值表，CSV `precision_tolerances` 仅对 `--compare close`/`cosine` 生效、`absolute_precision` 仅对 `close` 生效：

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

> 完整优先级：`--compare`（CLI）> `Spec.tolerance`（TestSpec）> 方法默认阈值。CSV 字段为 legacy，`precision_tolerances` 由 `close`/`cosine` 读取、`absolute_precision` 仅 `close` 读取（见 `references/precision-debug.md`）。

### 4.4 常见精度原因

| 现象 | 修复 |
|------|------|
| 全部 NaN / 大误差集中 | 调整 `input_data_ranges`、缩小输入范围（防溢出/下溢） |
| 数值接近但比对失败 | 浮点精度差异，放大容差或换 `--compare` |

> 完整现象-原因-修复表见 `references/precision-debug.md`。

## 常见错误模式速查

详细错误模式和修复见 `references/`：
- 错误模式表：`references/error-patterns.md`
- 精度调试详解：`references/precision-debug.md`
- NPUSim 仿真排障：`references/npusim-errors.md`

完整文档：`docs/faq/faq_guide.md`（路径相对于 ops-test-kit 仓库根目录）

## 相关 Skill

- CSV 格式有误 → `ttk-how-write-case`
- 重新运行测试 → `ttk-how-run-test`
- Golden 需要自定义 → `ttk-how-write-plugin`
