# 映射规则与对齐算法

`prof2case.py` 的确定性转换规则。修改脚本前先读本文件。

目标模式由 `--mode` 指定（skill 使用时由用户提示词决定，见 SKILL.md Step 0）：**aclnn**（默认）/ **kernel** / **geir** / **e2e**。§1-3、§6 为各模式公共规则；§4-5、§7-8 按模式分节；§9 为算子库接入（`--op-libs`，作用于 aclnn/e2e）。

## 1. 输入格式

torch_npu profiler / msprof 导出的 `kernel_details.csv`，关键列：

| 列 | 用途 |
|---|---|
| `Name` | `{aclnnApi}_{kernelName}_{opType}` 三段式 |
| `Type` | 算子类型（最后一段） |
| `Accelerator Core` | AI_VECTOR_CORE / AI_CORE / MIX_AIC / MIX_AIV / AI_CPU / COMMUNICATION |
| `Input/Output Shapes` / `Data Types` / `Formats` | 分号分隔的多张量；`""` 空段 = 标量；`DT_UNDEFINED`/`NULL` = 占位符；输入侧「空 shape 段 + 空 dtype 段」在 aclnn/e2e 模式保留为标量（dtype 置 None 并告警，kernel/geir 仍丢弃——该 0-d 项是 GE 物化常量，入表会破坏算子输入数） |
| `Model ID` / `Task ID` / `Duration(us)` | 进 remark 追溯 |

解析容错：多重引号（`"""shape"""`）剥离、表头与单元格尾部 tab/空白（header key 归一化，`Duration(us)\t` 等不再静默取空）、`N/A`、BOM（utf-8-sig）。

## 2. 名字解析

- **aclnn/e2e 模式**（解析 aclnn API）：
  - `Name` 以 `aclnn` 开头：先剥尾部 `_{Type}`，再按 `_` 切段，取能在头文件命中的**最长前缀**（`aclnnInplaceCopy_CastAiCore_Cast` → `aclnnInplaceCopy`）
  - 非 aclnn 前缀且 Type 为 CamelCase：候选 `aclnn + Type`（`FusedInferAttentionScore` → `aclnnFusedInferAttentionScore`）
  - aclnn 模式下未命中头文件 → 未解析，按 kernel 张量原序生成 + 报告；e2e 模式下继续查映射表
- **kernel/geir 模式**（派生 op_name）：`Type` 驼峰转下划线，候选取 `camel_to_snake` 与缩写感知变体二者（`LayerNormV4` → `layer_norm_v4`；缩写词保留如 `FFNToAttention` → `ffn_to_attention`，符合官方算子构建命名规则，见 asc-devkit `naming_conversion_table.md`）；算子信息库可用时对候选取 `opInterface.value` 并核对输入输出数量，不可用时按官方规则取缩写变体为派生名（校验预检跳过，报告说明）
- Type 为 snake_case / 小写开头（`_collect_welford_kernel`、`bf16_hybrid_attention`）：torch/自定义内核，所有模式均跳过

## 3. dtype / format 映射

| profiling | TTK |
|---|---|
| `INT8..INT64` / `UINT8..UINT64` | `int8..int64` / `uint8..uint64` |
| `FLOAT` / `DOUBLE` / `FLOAT16` | `float32` / `float64` / `float16` |
| `DT_BF16` | `bfloat16` |
| `BOOL` | `bool` |
| `DT_FLOAT8_E4M3FN` / `E5M2` / `E8M0` | `float8_e4m3fn` / `float8_e5m2` / `float8_e8m0` |
| `DT_UNDEFINED` / 未知值 | 该张量丢弃（DT_UNDEFINED 是可选占位；未知 dtype 触发用例级警告） |

format（`ND`/`NCHW`/`NCL`/`NCDHW`）原样透传；全 ND 时省略列（TTK 默认广播 `('ND',)`）。Formats 列缺失或空段默认 `ND`（张量保留，丢弃只由 dtype 决定；列缺失触发解析警告）。

shape：`"1330"` → `(1330,)`；`""` → `()`（标量）；`"1,3,1024,1024"` → `(1,3,1024,1024)`。

## 4. 张量对齐算法

### 4a. kernel / geir 模式：不对齐

kernel I/O **原样转换**：`input_shapes/dtypes/formats` ← 全部输入张量（含 perm/begin/end/axis 等 GE 内部常量——它们是真实算子输入，与 aclnn 模式的丢弃行为相反）；`output_*` ← 全部输出张量。geir 模式额外输出 `dyn_input_shapes` 列（**留空**，profiling 无动态 shape 信息）。

### 4b. aclnn 模式：头文件签名对齐

利用头文件 `aclnnXXGetWorkspaceSize` 签名（参数名 + 类型），按以下顺序：

1. **参数分类**：输出参数 = 名为 `out`/`output`、以 `Out`/`Output`/`OutOptional`/`OutputOptional`/`Ref` 结尾、或 `softmaxLse` 等特例；一个都没有时**末位张量参数兜底为输出**（对齐 TTK 的 `(-1,)` 规则，如 GeluV2 的 `y`）
2. **输出对齐**：profile 输出张量左对齐到输出参数（签名序）；**剩余输出右对齐到尾部空闲参数**（覆盖不带 Out 命名的多输出 API，如 BatchNorm 的 `saveMean`/`saveInvstd`、GroupedMatmulSwigluQuantWeightNzV2 的 `outputScale`），仍放不下的丢弃并告警；profile 输出 shape 为空段时保留 0-d 并告警提示复核
3. **输入对齐**（profile 非标量输入）：
   - 首参为 `self` 且数量相等：1-D 张量优先给 `self`（addmm/baddbmm 的 bias 广播模式）
   - 首参为 `self` 且输入不足：右对齐（累加输入被 kernel 省略）
   - **中置 TensorList + 尾随固定参数**（`aclnnIndexPutImpl` 的 values）：kernel 输入按签名序物化（列表原地展开），列表后的固定参数从 kernel 输入**尾部右对齐**预留，中段归列表；首位 `*Ref` 参数的 self 缓冲若同时出现在 kernel 首输入（与 kernel 首输出同 shape/dtype/format）则先行丢弃（已由输出对齐覆盖，记 remark `selfbuf-drop`），成功右对齐记 remark `tail-align:参数名`。多列表或输入不足无法右对齐时回退左对齐并告警（`post-list fixed params left-aligned, needs spot check`）
   - 其他：非 bool 张量左对齐；bool 张量优先匹配名字含 `mask` 的参数；**输入盈余时非 TensorList 槽位跳过 aux 常量**（aux 是最可能的盈余来源，防止挤占参数位导致整体错位，如 FastLayerNorm 的 axis 挤掉 weight），TensorList 槽位保留 aux（列表元素常为小 int 张量，如 aclnnIndex 的 indices）；数量匹配/不足时按 kernel 顺序原样对齐
4. **aux 回填**：被跳过的 aux 回填仍空闲的输入参数（合法小张量输入由此落位）
5. **TensorList 吸收**：剩余非 aux 输入追加到第一个 `aclTensorList*` 参数（Cat/GroupedMatmul 的多张量合并）
6. **aux 丢弃**：回填后仍未落位的 aux 丢弃，记入 remark `aux-drop`
7. **缺口补 None**：未匹配参数填 None 占位（TTK 语义：nullptr 可选输入）；未匹配告警在所有回填（aux/标量溢出/剩余输出）之后统一统计，避免误报与重复告警
8. `*Ref` 参数强制并入 output_tensor_indexes（inplace 语义）

### 4c. e2e 模式：torch 签名对齐

aclnn API 经 `ACLNN_TO_TORCH` 映射表转 torch API（映射来源见 §9；无映射 → 跳过 + 报告），再按 torch 签名（`FrameworkApiInfoKeeper`）对齐：

1. **张量参数分类**：`out` 参数排除（不生成 out= 变体，输出经返回值比对）；**必选** `out` 参数例外——TTK 校验强制要求 `output_tensor_indexes`，kernel 输出平铺为多个 out 槽位追加在输入之后（TensorList out 一输出一槽位，TTK 按参数名把索引槽位路由成 `out=`）；var-positional 参数（`tensor_distribution < 0`，如 `torch.cat` 的 `tensors`）为 TensorList 参数
2. **inplace Tensor 方法**（`torch.Tensor.xxx_`）：`self` 槽位取 kernel 输出（GE 对 copy_ 只报 src、对 masked_fill_ 报 self——按"非 aux 实输入是否覆盖**固定（非列表）张量参数**"判定；列表参数造成的缺口说明 kernel 已报 self，如 `index_put_`），`output_tensor_indexes` 留空由 TTK 自动填 `(0,)`
3. **输入对齐**：非标量 kernel 输入左对齐到固定张量参数；盈余时跳过 aux 常量并**回填**空闲参数（广播标量源等）；**中置列表参数 + 其后固定参数**（`index_put_` 的 values）：kernel 输入按签名序物化（列表原地展开），列表后的固定参数从 kernel 输入**尾部右对齐**预留、中段归列表（记 remark `tail-align:参数名`），多列表或输入不足时回退左对齐并告警；TensorList 参数吸收剩余**全部**输入（aux 小 int 张量保留——列表成员常为小 int 张量，如 `index_put_` 的 indices，与 aclnn 模式 TensorList 槽规则一致；嵌套表达），槽位按**签名序**落位（TTK 按位置映射张量参数，中置列表参数不后置到末尾），中间缺口输出 None 占位
4. **标量/输出**：shape `()` 的 kernel 输入对应 Number 参数（值层留空，必选项列入报告），每处记 remark `scalar-drop:K((),...)` 并列入对齐警告；kernel 输出不再入表（返回值语义）

## 5. 标量处理（仅 aclnn）

- kernel 输入中 shape 为 `()` 的张量 → 按位对到 `aclScalar*` 参数，**dtype 取自 profiling**（如 Fill 的 value）；dtype 段为空（导出缺值）时标量仍保留、dtype 置 None 占位并告警
- 其余标量参数按参数名启发式猜 dtype（alpha/eps/scale → float32；dim/k/numHeads/*Size/*Type → int64；descending/outFlag → bool），猜不出 → None 占位 + 警告
- **标量值一律不填**（TTK 随机生成）

## 6. 跳过规则

| 条件 | 原因 | 适用模式 |
|---|---|---|
| `Accelerator Core == COMMUNICATION` 或 Name 以 `Hccl`/`hcom`/`hccl`/`HCCL` 开头 | 通信 kernel，无 aclnn API（AI core 上跑的 hccl 辅助内核无 COMMUNICATION 标记，靠前缀兜底） | 全部 |
| 输入输出 Shapes 均为 `N/A` | 无签名信息 | 全部 |
| 解析后张量全为 DT_UNDEFINED | 全占位 | 全部 |
| Type 为 snake_case 且无 aclnn 前缀 | torch/自定义内核 | 全部 |
| aclnn API 不在映射来源（内置表 / 算子库派生 / 同名自动解析）中 | 无 torch 等价 API | 仅 e2e |
| 对齐后无张量参数（全标量/常量，如 arange） | 无法构成张量用例 | 仅 e2e |

## 7. 输出 CSV 列（按模式）

```
aclnn:  testcase_name, api_name, tensor_view_shapes, tensor_dtypes, tensor_formats,
        scalar_dtypes, output_tensor_indexes, output_inplace_indexes, remark
kernel: testcase_name, op_name, input_shapes, input_dtypes, input_formats,
        output_shapes, output_dtypes, output_formats, attributes, remark
geir:   同 kernel + dyn_input_shapes（remark 前）
e2e:    testcase_name, api_name, tensor_view_shapes, tensor_dtypes, tensor_formats,
        attributes, output_tensor_indexes, remark
```

- `testcase_name`：`{前缀}_{全局序号:05d}`，前缀 = aclnn api / op_name / torch API 末段（去尾部 `_`），同前缀递增
- `remark`：`src=rowN;name=原始Name;type=Type;dur=Xus;model=M;task=T[;approx-map:...（仅e2e）][;aux-drop:...][;xN=次数(仅--dedup)]`
- 不输出的列 = 值层留空（attributes / data ranges / strides / offsets / storage_shapes / dyn_input_shapes）

## 8. 自检（校验预检）

脚本内置：对每个唯一签名取 1 例，经 TTK 真实解析器（`UniversalTestcaseFactory` + 各模式用例类：`TestcaseAclnn`/`TestcaseKernel`/`GeirTestcase`/`TestcaseE2e`）校验，失败原因按签名外推计数。签名/算子信息库不可用时（无 CANN/torch 环境或无 SOC）跳过自检并在报告说明。预期失败与报告清单对应：`PARAM_COUNT_MISMATCH`/`MISSING_REQUIRED_ATTR`（缺值）、`OP_API_NOT_FOUND`/`API_PARSE_FAIL`（未解析/无签名）、`SOC_NOT_SUPPORT`（op_name 派生名未在算子信息库命中）。

## 9. 算子库接入（`--op-libs`，aclnn/e2e 模式）

自定义算子（ops-nn / ops-transformer 等）未装进本机 CANN 时，把算子库源码仓根目录传给 `--op-libs`（冒号分隔可多个），脚本在构造签名库**之前**完成扫描：

### 9a. aclnn 头文件收集（aclnn/e2e 共用）

- 递归收集「名为 `op_api` 且直接含 `.h/.hpp`」的目录，以及任何直接含 `aclnn*.h` 的目录；`tests`/`docs`/`examples`/`build`/`.git` 等目录剪枝（`examples/` 内是示例算子，非可交付 API，收集会虚增 `--audit` 库存）
- 覆盖两类布局：ops-nn `<category>/<op>/op_api/` 与 ops-transformer `<module>/<op>/op_host/op_api/`
- 收集结果并入 `TTK_OP_API_HEADER_PATH`（保留用户已有值）→ `OpApiInfoKeeper` 解析；**源码头文件优先于已安装版本**（TTK 语义）；无 CANN 环境时仅凭算子库也能做签名对齐与自检

### 9b. torch_extension 派生映射（仅 e2e）

- 扫描 `**/torch_extension/*.py` 中的 `OpBuilder` 子类：AST 提取 `super().__init__("op名")` 的注册名与 `def schema()` 的字面量 schema（字符串或列表）
- 命名空间从 `<root>/torch_extension/<pkg>/op_builder/builder.py` 目录名探测（缺省 `cann_ops_transformer`）
- 同一算子目录内：`op_api` 头文件声明的 aclnn API ↔ torch 注册名配对——snake 名完全相等优先，其次互为前缀（词边界），单 torch 算子目录兜底全配（Grad/Backward API 除外）；跨目录重复声明保留首个并记报告
- 派生映射形如 `aclnnXxx → torch.ops.<ns>.<op>`，与内置 `ACLNN_TO_TORCH` 冲突时**内置表优先**（人工核实过的映射）
- schema 按 torch 原生文法解析（`Tensor`/`Tensor?`/`Tensor[]`/`(a!)` 别名/`*` 后关键字参数/默认值），经 `FrameworkApiInfoKeeper.register` 注册——**无需编译安装扩展包**即可离线完成对齐与自检

### 9c. e2e 映射来源与优先级

映射按优先级依次尝试（前一来源命中即不再查下一层）：

1. **内置表 `ACLNN_TO_TORCH`**：标准 torch 映射 + torch_npu 引擎算子映射（经 torch_npu ge-converter / 插件文档 / cann-recipes 网络用法核实）：MOE 族、MX 量化族、FlashAttention 族（torch 侧名为 `npu_fusion_attention`）、`npu_fused_infer_attention_score(_v2)`、Incre/Prompt/Sparse FA、NSA 族、ROPE/rotary 族、采样族、WeightNz 量化矩阵乘族（同一 torch API，NZ 布局由权重张量 4-D shape 携带）与 `torch.ops.aten.index.Tensor`（GE Index→高级索引）
2. **算子库派生**（`--op-libs`）：同目录/同名配对的 `torch.ops.<ns>.<op>`
3. **同名自动解析**（`auto_resolve_torch_api`，签名校验通过才接受）：
   - 精确名：`torch_npu.npu_<snake>` / `torch.<snake>` / `torch.nn.functional.<snake>`（snake 取 camel_to_snake 与缩略词感知变体二者——FFN/KV 等大写缩略词切分不同，另支持 `_grad`→`_backward`）
   - WeightNz 后缀剥离：`aclnnAddmmWeightNz → torch.addmm`（同一 torch 算子，权重 NZ 布局）
   - 版本后缀递减：`aclnnEluV2 → F.elu`、`aclnnFusedInferAttentionScoreV4 → npu_fused_infer_attention_score_v2`（引擎迭代版本，语义需核对）
   - WeightNz 与版本后缀可组合剥离：`aclnnXxxWeightNzV2 → XxxWeightNz → Xxx`（逐层累积注记）
   - 每个候选都经 `torch_signature_compatible`（必选 torch 张量参数数 ≤ aclnn 张量参数总数）校验；任一签名不可解析则不映射。被接受的自动映射带 `auto-map` remark 位并列入报告「同名自动解析映射」清单

**不映射**（预期跳过，报告说明）：Grad/Backward 族（torch 无公开 backward API，经前向 golden 覆盖）、`*Metadata` 元数据预处理、Inner/Inplace 内部变体、MC2 通信融合族（需分布式执行环境）、Foreach 优化器族、 fused 优化器（Apply*）。

### 9d. e2e 参数填充顺序（`E2E_PARAM_FILL_ORDER`）

部分量化/GMM 算子的 torch 签名序与 kernel I/O 物化序不一致（经 cann-recipes 网络调用用法核实）：
`npu_quant_matmul`（kernel 序 x1,x2,权重scale,per-token scale → 跳过 offset 槽位）、`npu_grouped_matmul`（x,weight,scale,group_list,per_token_scale）、`npu_grouped_matmul_swiglu_quant_v2`（x,x_scale,group_list,weight,weight_scale）。表内 API 按「参数名」落位：kernel 输入按填充序赋给参数，**中间未填参数输出 None 占位**（TTK 按槽位顺序映射到张量参数），列表参数包装为单元素列表；**此路径不做 aux 过滤**（kernel 序经人工核实，小 int 张量是业务输入——如 ≤7 专家的 `group_list` 为 int64 且 numel ≤ 8，若套用 aux 启发式会被误删并致后续张量左移错位）；表外 API 走位置左对齐。

### 9e. 全量算子审计（`--audit`）

对 `--op-libs` 库内**全部** aclnn API（不依赖 profiling 是否出现过）审计可转换性并写入报告：aclnn 模式头文件解析率 + e2e 映射覆盖率（三路来源分计）+ 无映射清单 + 自动解析映射清单。可不带 `-i`（纯审计）。
