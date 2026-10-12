---
name: ttk-prof2case
description: 将 Ascend profiling 报告转换成 TTK 用例。只要用户提到 profiling 报告转用例、kernel_details.csv、从 profiling/调优报告生成测试用例、真实业务 shape 生成用例、msprof / torch_npu profiler 导出数据转 CSV 用例、算子调用记录变测试，就必须使用此 skill。支持四种目标模式（aclnn/kernel/geir/e2e），转换成哪种由用户提示词决定；值层字段留空并生成待补值报告。
---

# Profiling 报告 → TTK 用例

将 profiling 报告（`kernel_details.csv`，torch_npu profiler / msprof 导出，通常在 `ASCEND_PROFILER_OUTPUT/` 下）转换为 TTK 测试用例。

## Step 0: 确定目标模式（由用户提示词决定）

**用户要求转成什么模式，就用什么模式**；未明说时按关键词判断，仍不确定时询问用户：

| 用户提示关键词 | 模式 | `--mode` | 说明 |
|---|---|---|---|
| 算子层 / kernel / 编译执行 / Ascend C / 算子内核 | Kernel | `kernel` | kernel I/O 原样转换，op_name 由算子类型派生 |
| GE 图 / GEIR / 图编译 / 单算子图 | GEIR | `geir` | 同 kernel + `dyn_input_shapes` 列（留空） |
| aclnn / 引擎 API / 引擎层 | ACLNN（默认） | `aclnn` | 按 aclnn 头文件签名对齐张量 |
| e2e / 端到端 / torch / 框架 API / 框架层 | E2E | `e2e` | aclnn→torch 映射 + torch 签名对齐 |

## 转换原则（必须向用户遵守）

| 信息层 | 来源 | 策略 |
|---|---|---|
| 签名层（op/api name / shapes / dtypes / formats） | profiling 各列 | 原样填充，shape 一律不裁剪 |
| 结构层（output_tensor_indexes / None 占位 / TensorList 嵌套 / op_name） | 签名库（aclnn 头文件 / 算子信息库 / torch 签名，可用时） | 推断填充 |
| **值层（attributes 值 / data_ranges / strides / ValueDepend 输入值等）** | profiling 中**不存在** | **一律留空**，不得编造 |
| 管理层（testcase_name / remark） | 脚本生成 | 自动编号，remark 记录来源行号/原始 Name/耗时 |

- **默认逐行转换、不去重**（一个 profile 行 = 一个用例）；`--dedup` 可按唯一签名压缩
- 通信 kernel（COMMUNICATION / Hccl / hcom）、无签名信息行、snake_case 非 aclnn 内核自动跳过并计入报告
- 模式特有行为：
  - **kernel/geir**：GE 内部常量（perm/begin/end/axis）是真实算子输入，**保留**在 input_shapes 中（与 aclnn 模式丢弃不同）；无 SOC/算子信息库时 op_name 为派生名且跳过校验预检
  - **aclnn**：未解析 API（头文件查不到）仍生成用例但标记待处理
  - **e2e**：aclnn→torch 映射三路来源——内置 `ACLNN_TO_TORCH` 表（含 torch_npu `npu_*` 引擎算子映射，如 MOE/MX量化/融合注意力/采样）、`--op-libs` 派生的 `torch.ops.cann_ops_transformer.*` 映射、`torch.ops.aten.*` 等价 API；仍无映射的 API（WeightNz 权重布局等引擎级通路）跳过并列入报告；近似映射（语义/默认值可能不同）单独列出

## Step 1: 运行转换

```shell
python3 .claude/skills/ttk-prof2case/scripts/prof2case.py -i <kernel_details.csv> -o <outdir>/ --mode <aclnn|kernel|geir|e2e> [--dedup] [--no-selfcheck] [--op-libs <算子库根1>:<算子库根2>] [--audit]
```

可从任意目录以绝对路径调用（脚本自动把 ops-test-kit 仓根注入 `sys.path` 以导入 `ttk` 签名库；不可用时报告会说明）。

产出两个文件：
- `<outdir>/cases_<mode>.csv` —— 对应模式用例
- `<outdir>/conversion_report.md` —— 转换报告（跳过/警告/待补值清单/校验预检）

**算子库（`--op-libs`，推荐）**：profiling 含自定义算子（MOE/量化/ROPE/融合注意力等）而本机 CANN 未装对应算子包时，把算子库源码仓根目录传进来（如 ops-nn / ops-transformer 的 checkout，冒号分隔可多个）：
- aclnn/e2e 模式自动递归收集 `**/op_api` 头文件目录并入 `TTK_OP_API_HEADER_PATH`（源码头文件优先于已安装版本），未装包的自定义算子也能完成签名对齐
- e2e 模式额外扫描 `torch_extension` 包装器：同算子目录内声明的 aclnn API 自动派生 `torch.ops.cann_ops_transformer.<op>` 映射，并注册其 schema（无需编译安装扩展包即可离线对齐与自检）

**全量审计（`--audit`，可不带 `-i`）**：对 `--op-libs` 库内全部 aclnn API 审计可转换性（aclnn 头文件解析率 + e2e 映射覆盖率三路来源分计 + 无映射清单），回答"这套算子库都能转吗"——profiling 只覆盖出现过的算子，全量结论以审计为准。

e2e 映射三路来源（按优先级）：内置 `ACLNN_TO_TORCH` 表（含 torch_npu `npu_*` 引擎算子，经 ge-converter/cann-recipes 网络用法核实）→ `--op-libs` 派生的 `torch.ops.*` → 同名自动解析（精确名/WeightNz 剥离/版本递减，均过签名校验，报告列清单）。Grad/Backward、Metadata、MC2 通信融合等无 torch 公开接口的属预期跳过。

## Step 2: 阅读转换报告

按顺序检查 `conversion_report.md` 各节，逐节决策：

| 报告章节 | 含义 | 处理 |
|---|---|---|
| 值层空缺提醒 | 固定提醒：attributes 等留空的含义（按模式不同） | 向用户转述 |
| 跳过统计 | 通信 kernel 等不可转换行（e2e 另含无映射行） | 无需处理 |
| 未解析 API（仅 aclnn） | 头文件中无此 API，用例必然校验失败 | 建议用户确认算子包/`--op-libs` 算子库源码仓/`TTK_OP_API_HEADER_PATH`，或删除对应用例 |
| 未映射 API（仅 e2e） | 无 torch 映射已跳过（Grad/Metadata/MC2 通信融合等无 torch 公开接口，属预期） | 需要覆盖时扩展 `ACLNN_TO_TORCH` 表或经 `--op-libs` 派生后重跑 |
| 同名自动解析映射（仅 e2e） | 按名称自动匹配并过签名校验的映射（含版本/WeightNz 剥离） | 抽查核对，尤其带版本剥离注记的 |
| 算子库全量算子审计（--audit） | 库内全部 API 的 aclnn 解析率 + e2e 覆盖率（三路来源分计）+ 无映射清单 | 全量覆盖结论以此为准；缺口按清单处理 |
| 近似映射（仅 e2e） | 语义/默认值可能不同 | 抽查核对 |
| 疑似 ValueDepend 输入（仅 kernel/geir） | perm/begin/end 等输入值需在 attributes 指定 | 向用户列出，由用户补值 |
| 需人工补 attributes 的 API/算子 | 必选参数空值导致校验失败/编译失败 | **逐 API 向用户列出参数清单，由用户补值**；用户明确要求时代填合理值并标注"非 profiling 来源" |
| 对齐警告 | 张量与签名对齐的启发式风险点 | 抽查对应用例，确认 shapes 落位合理 |
| TTK 校验预检 | 按签名外推的校验失败统计 | 预期失败（缺值/未解析/无映射/派生名）无需处理，其余需排查 |

## Step 3: 处理值层空缺（核心原则：不编造）

值层拿不到的字段**保持空**，必须向用户显式汇报，格式：

```
以下 N 个用例的值层字段留空，需人工补齐后才能通过校验/复现业务语义：
- aclnnConvolution（31 例）: 需填 attributes {stride, padding, dilation, transposed, outputPadding, groups, cubeMathType}
- transpose（kernel 模式，252 例）: perm 输入值需在 attributes 指定，否则随机生成
- ...
其余 M 个用例可直接运行（标量为随机值，输入数据默认 [-2,2] 随机）。
```

补值方式参考 `ttk-how-write-case`（各模式字段说明）。用户未要求时不代填。

## Step 4: 验证

```shell
python3 -m ttk <mode> -i <outdir>/cases_<mode>.csv --validate
python3 -m ttk list -i <outdir>/cases_<mode>.csv
```

- 校验失败与报告"待补值/未解析/无映射"清单对应的，属预期，等用户补值
- 其余失败（格式错误、dtype 不支持等）需修复后重跑

## Step 5: 交接

- 运行测试 → `ttk-how-run-test`
- 算子无内置 golden / 需要精度标准 → `ttk-how-write-plugin`
- 需要真实业务数据（而非随机数据）→ 提示用户走 dump 数据导入路线（`manual_tensor_binaries`，见 docs/Offline_Data_Prepare_and_Import.md）

## 相关 Skill

- 用例字段格式 → `ttk-how-write-case`
- 运行测试 → `ttk-how-run-test`
- 失败排障 → `ttk-how-diagnose`
