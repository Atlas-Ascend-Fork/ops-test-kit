# 离线数据准备与导入

两阶段执行把输入生成、可选的 CPU Golden 生成与目标设备执行拆开，适用于
准备机和执行机不是同一台服务器的场景，也支持只准备 input。

```text
dump input：    CSV -> input 生成/修正 -> 只保存 input
dump 完整：CSV -> input 生成/修正 -> CPU Golden -> 保存两者
load：          CSV -> 恢复 input -> 恢复/生成 Golden（静态单卡先于设备执行；
                动态 shape/多卡用例延迟到执行后）-> NPU API -> compare
```

数据流向由两个独立开关决定，没有独立的"模式"：

- `--manual-data-dirs DIR` 只是数据目录指定：load 方向的搜索空间；
  `--no-prof` 落盘时作为（单个）输出目录；
- `--no-prof` 只是"不上板"标记：关闭设备执行，dump 门控开启时
  （`--dump in[,golden]`）把生成数据落盘到（显式或默认）目录；
- 全流程执行 + 目录为空时，首跑正常生成并按 dump 门控落盘（`--dump in,golden`），
  后续运行自动转为 load——不需要先单独跑一次准备命令。

支持 `ttk e2e`、`ttk aclnn`、`ttk kernel` 和 `ttk geir`。两个阶段都以同一份 CSV 为用例来源，
按 `testcase_name` 匹配数据；框架不会根据数据目录反向生成用例。

## 1. 阶段选择

| 场景 | 参数 | Input | Golden | 设备执行 | Compare |
| --- | --- | --- | --- | --- | --- |
| 普通执行 | 不传数据目录 | 生成 | 生成 | 执行 | 执行 |
| dump input | `--no-prof --dump in` | 生成并保存 | 不执行 | 跳过 | 跳过 |
| dump in,golden | `--no-prof --dump in,golden` | 生成并保存 | 生成并保存 | 跳过 | 跳过 |
| load（目录有该用例） | `--manual-data-dirs DIR...` | 从文件恢复 | 存在时恢复；不存在时生成（静态单卡在执行前完成；动态 shape/多卡延迟到执行后） | 执行 | 执行 |
| 首跑落盘（目录为空） | `--manual-data-dirs DIR --dump in,golden` | 生成并保存 | 生成并保存 | 执行 | 执行 |

load 时目录非空但用例未命中会直接报错（防止"以为在跑存量数据实际现场生成"）；
目录整体为空才按普通执行生成并落盘。落什么由 `--dump` 门控自然决定：
不上板就没有 output 可落，不存在"必须/禁止 out"的人工约束。

E2E、ACLNN 和 GEIR 支持 input-only 和完整 dump。input-only 会替换整个同名 case
目录，不会把旧 Golden 与新 input 混用。不支持 Golden-only dump。
`--dump-format` 支持 `bin`、`npy`、`pt`，默认值为 `bin`。

### Kernel 的 `--no-prof` 与 `--co`

Kernel 单独使用 `--no-prof` 时保持原有 dry-run 行为：完成 input、Golden 和 workspace
生成，但关闭 dynamic、const、binary 执行。只有精确组合
`--no-prof --dump in,golden` 才向 manual-data 目录落盘。
Kernel 不支持 input-only 落盘，因为其 output 分配可能在设备执行前依赖
Golden 或动态 output 信息。

`--co/--compile-only` 在编译或 tiling 后、input/Golden 生成前返回，不能产生两阶段数据，
因此不能与 Kernel 数据目录组合。Kernel dump 仍执行当前模式所需的编译或 binary
匹配和 tiling，只在设备锁和目标 Kernel 执行前返回。

## 2. 快速开始

以下命令中的落盘（dump）和加载（load）必须使用兼容的 CSV 与 assets。

### 2.1 E2E

```bash
# dump（落盘）
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# load（加载）
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

如果只准备 input，不希望执行 CPU Golden 回调：

```bash
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/input_result.csv
```

E2E 落盘可以增加 `--backend cpu` 强制 CPU backend；load 是设备阶段，不能使用 `--backend cpu`。

### 2.2 ACLNN

```bash
# dump（落盘）
python3 -m ttk aclnn \
  -i /path/to/aclnn_cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  --plat Ascend950 \
  -o /path/to/prepare_result.csv

# load（加载）
python3 -m ttk aclnn \
  -i /path/to/aclnn_cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

ACLNN 同样支持 `--no-prof --dump in`。该数据被 load 时，静态单卡用例会先
基于恢复的 input 生成缺失的 Golden，再执行目标 API；只有动态 shape（-1 槽）
等无法预先确定输出的用例才在执行后生成/装载。

ACLNN 落盘不调用主 API，也不查询设备数量和编译 warmup 辅助 Kernel。CSV 与 API
元信息解析仍依赖 CANN/OPP；准备机无法探测 SoC 时应显式传 `--plat`。

### 2.3 Kernel

两个阶段必须选择相同的 dynamic、const 或 release binary 模式。以下示例使用 release：

```bash
# dump：编译/binary匹配和tiling后保存数据，不执行目标Kernel
python3 -m ttk kernel \
  -i /path/to/kernel_cases.csv \
  --plugin /path/to/kernel_assets \
  --plat Ascend910_9362 \
  -d=false -c=false -b=release \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# load：恢复数据后执行release Kernel和compare
python3 -m ttk kernel \
  -i /path/to/kernel_cases.csv \
  --plugin /path/to/kernel_assets \
  --plat Ascend910_9362 \
  -d=false -c=false -b=release \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

框架没有内置 Kernel input/Golden 时，assets 必须为 CSV 的 `op_name` 注册
`__input__["kernel"]` 和 `__golden__["kernel"]`。只有 E2E TestSpec 的 assets 不能替代
raw Kernel 回调。

### 2.4 GEIR

数据集（输入+golden）与图模式（const/dynamic × binary）无关：`--no-prof` 单遍
生成并落盘，跳过 GE 图编译与执行；load 方向存储数据逐模式恢复，golden 装载一次
全程复用（存储 golden 缺席时按存储输入现算）。shape-unknown 输出的 uint64 元数据
golden 槽随数据集一并落盘/恢复。

```bash
# dump：生成 input/golden 并保存，跳过 GE 图编译与执行
python3 -m ttk geir \
  -i /path/to/geir_cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# load：恢复数据后图编译、执行与 compare 照常
python3 -m ttk geir \
  -i /path/to/geir_cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

`--golden-mode XPU` 与 `--no-prof` 互斥（golden 由三方执行产出，无可制备）。

## 3. 数据目录

### 3.1 dump 输出目录

显式传入 `--manual-data-dirs DIR` 时，落盘写入该目录。省略时按以下顺序确定：

1. 只有一个目录 plugin：`<plugin>/manual_data`；
2. 只有一个 `.py` plugin：`<plugin父目录>/manual_data`；
3. 没有 plugin：`<当前工作目录>/manual_data`。

dump 只允许一个输出目录。多个 plugin 路径时必须显式指定输出目录。

### 3.2 load 搜索目录

```bash
python3 -m ttk e2e -i cases.csv \
  --manual-data-dirs /data/current /data/archive
```

TTK 按参数顺序为每个 case 查找目录。第一个同名目录一旦命中便立即使用；该目录损坏时
直接失败，不会跳到后续目录，也不会回退随机生成。

### 3.3 case 目录名

只含 `[A-Za-z0-9_.-]`、首尾不是 `.`/`_` 且不超过 120 字符的 `testcase_name` 直接作为目录名。
其他名称会安全化（非法字符替换为 `_`）后截断到 96 字符，并用 `-` 追加完整原名 SHA-256 的前 12 位。dump/load 使用完全相同的原始名称
时会得到相同目录；这不是模糊匹配，也不能从目录名反推原始长名称。

同名 case 重新 dump 时，旧目录先失效；这也防止 input-only dump 遗留旧
Golden。新数据在隐藏临时目录中写入并回读校验，
全部成功后才发布；失败不会留下可 load 的半成品。

### 3.4 从调试 dump 目录读回

`--manual-data-dirs` 可以直接指向调试 dump 目录（`{root}/dump/`，即 `--dump`
/ `--dump-on-fail` 的产物）。扫描规则：

| 调试产物 | 读回处理 |
| --- | --- |
| `golden_{idx}_{dtype}`、`xpu_{provider}_{idx}_{dtype}` | 直接读（与协议同名） |
| `input_dyn_{idx}_{dtype}`（kernel）、`input_{const,dynamic}_{idx}_{dtype}`（geir） | 去掉模式段映射到 input 槽；同槽多份要求字节一致 |
| `output[_{segment}]_{idx}_{dtype}`、`tiling_data[_{segment}]` | 忽略（读回会重新执行/编译产出） |
| `input_{eager,graph_*}_{idx}_{dtype}`（e2e dump-on-fail 执行后补漏） | 拒绝（in-place API 可能已改写输入） |

已知边界：

- 调试 dump 对 None 槽不落 `*_none.bin` 标记，含 None 槽的用例会报槽位索引
  不连续；kernel/aclnn 的 `dump_file_prefix` 与 `testcase_name` 不同时目录名
  对不上；
- 调试目录不整目录替换，CSV 改 shape/dtype 后的陈旧文件由既有的
  槽位/dtype/字节校验拦截。

## 4. 文件协议

完整 case 可以包含以下文件；input-only case 不含任何 `golden_*` 文件：

```text
<manual-data-dir>/<case-directory>/
├── input_0_bfloat16.bin
├── input_1_int32.bin
├── input_2_none.bin
├── scalar_0_float32.bin
└── golden_0_bfloat16.bin
```

所有数据文件（含 bin Golden）使用统一文件名，不编码 shape：

```text
<input|scalar|golden>_<从0开始的扁平索引>_<numpy-dtype|none>.<bin|npy|pt>
```

bin Golden 的 reshape 真值 = load 侧传入的设备 output / CSV output shape
（`--manual-data-dirs` 回放时由 TTK 自动传入）；元素总数不符会在字节数预检中
报错。**存量兼容**：早期版本落盘的 bin Golden 文件名带
`__shape_<shape-token>` 段（如 `golden_0_float32__shape_2x3.bin`），load 时
仍解析并与设备 output shape 核对（同 numel 错误 shape 报错），无需重新落盘。

### 4.1 槽位和 `None`

- input、scalar、Golden 的索引必须分别从 0 连续；
- optional `None` 使用同格式的零字节 `*_none.<format>`，不能省略；
- Golden 的 `*_none` marker 表示抑制该 output slot 的比较；设备 API 仍可物化该输出，
  load 会把 Golden 恢复为 `None`，由 custom/default compare 标记为 suppressed；
- `*_none.npy` 和 `*_none.pt` 是 TTK marker，不是 NumPy/PyTorch 原生文件；
- 零元素 Tensor 仍用真实 dtype 文件名，例如 `input_0_float32.bin`，与 `None` 不同；
- input 保存最终 backing storage，view shape/stride/offset 与 TensorList/ScalarList 分组由
  当前 CSV 重建。

### 4.2 格式

| 格式 | 保存方式 | shape 来源 |
| --- | --- | --- |
| `bin` | 连续原始字节 | input/scalar 来自 CSV；Golden 来自 load 侧设备/CSV shape（存量文件名 shape 段仍核对） |
| `npy` | NumPy 数组 | 文件内 shape；custom dtype 使用等宽 `voidN`，逻辑 dtype 由文件名恢复 |
| `pt` | CPU Torch 数据 | 文件内 shape；Torch 不支持的 dtype 在同一 pt 内保存 raw bytes 和 shape |

dump 每次只写一种格式。若交付目录同时存在多种完整副本，load 按
`bin > npy > pt` 选择一整套数据，不会跨格式拼接。高优先级格式只要出现但不完整或损坏，
当前 case 直接失败，不回退低优先级副本。

目录内不允许 JSON、日志、checksum、子目录、符号链接或其他 sidecar。每个非 None 文件
写完后立即按 dtype、shape 和完整字节回读；校验失败时不发布 case。

## 5. load 行为

load 命中数据后始终跳过：

- 随机输入生成和 attributes 数据覆盖；
- input plugin；

完整数据目录中存在 Golden 时，load 跳过 CPU Golden API 或 Golden plugin。
input-only 数据目录中不存在任何 Golden 时，静态单卡用例会在设备执行前基于
恢复的 input 调用当前 Golden 链路，再执行设备 API 并 compare；动态 shape/多卡
用例的 Golden 生成/装载延迟到设备执行后。部分 Golden 文件缺失不会被
当成 input-only，而是直接报数据不完整。

load 仍执行 CSV/ParamPlan、API 或 Kernel 解析、必要的编译/tiling、wrapper、设备执行、
compare 和结果落表。依赖 Spec、wrapper、tolerance、pre-compare 或 custom compare 时，
仍应传入对应 `--plugin` 和 `PYTHONPATH`。

dump 在 Golden 回调前复制最终 tensor backing storage和 ACLNN scalar。即使自定义
Golden 原地修改输入，保存的仍是目标 API 在普通执行中应看到的数据。

全流程首跑落盘（目录为空 + `--dump in,golden`）同理：input 在设备执行前快照，
golden 落当次实际产物（判据要求 Promote 时即升精度标杆），文件名 dtype 自描述，
回放按文件名装载。

### Compare

两阶段执行不替换 compare：

- E2E、ACLNN和Kernel的pre-compare、自定义compare保持各自正常优先级；
- 没有自定义hook时，ACLNN/Kernel的tolerance和CSV精度字段保持原解析路径；
- 使用 CSV `rtol/ptol/atol` 进行 close 比较时仍需显式传 `--compare close`；
- load 可以调整精度标准，无需重新落盘。

custom compare 不应依赖落盘进程中 input/Golden plugin 写入的模块全局变量。跨进程或
跨服务器 load 不会恢复该状态；必要信息应由 Golden、CSV 或可重建配置提供。
若判定需要输入侧数据，compare应显式声明仅关键字参数`compare_context`。direct和
load都可从其`input_tensors`、ACLNN `input_scalars`（Kernel为空tuple）、CSV `attributes`和原始
`csv_fields`读取当前用例状态；仅声明`**kwargs`的hook不会收到该上下文。

## 6. 历史字段说明

ACLNN 的 `manual_tensor_binaries/manual_golden_binaries` 是另一组历史字段，
直接按 CSV 内文件路径读取，不接入本目录协议。

## 7. 参数约束

- `-i/--input` 和 `-o/--output` 每次各接收一个 CSV；
- `--plugin` 可在一个字符串中用逗号分隔多个搜索路径；
- `--manual-data-dirs` 在 dump 方向最多一个（输出目录），在 load 方向可有多个（搜索空间）；
- dump 方向（`--no-prof` 落盘）要求 `--dump in` 开启；golden 落盘时禁止
  `--golden-mode Disable`；`--dump-format print` 不可恢复；Kernel 额外要求
  精确 `--dump in,golden`，并禁止 `--compile-only`。output/full dump 不再报错——
  不上板自然无 output 产物，`--dump-on-fail`/`--validate` 同理不构成冲突；
- load 方向禁止 `--no-prof`（该组合走 dump 方向）、`--validate`，要求
  `--golden-mode Enable`（Disable/Promote/XPU 均被拒绝），E2E 禁止 `--backend cpu`，
  Kernel 禁止 `--compile-only`；
- `--seed` 只影响落盘方向的随机输入。load 读取文件，不用 seed 重新生成或匹配数据；
- 两个阶段可用相同 `-t/--testcase` 只处理指定 case。

## 8. 交付验收

至少执行以下检查：

| 测试项 | 验收点 |
| --- | --- |
| direct 基线 | 同一 CSV、plugin、compare 配置可普通执行 |
| 三种格式 | `bin/npy/pt` 分别 dump 和 load |
| 两种 dump | E2E/ACLNN input-only 和完整落盘都不执行设备 |
| 缺少 Golden | load input-only 数据时按当前 Golden 链路生成 Golden（静态单卡先于设备执行）并 compare |
| 目录迁移 | 完整复制数据根到另一绝对路径后 load |
| shape | 修改 device output 为同 numel 错 shape 时，新 bin/npy/pt 都明确失败 |
| compare | custom compare 和 CSV close 容差在 load 中生效 |
| 格式优先级 | 整套选择 `bin > npy > pt`，损坏高优先级不回退 |
| 失败保护 | case、槽位、None marker、dtype、shape 或字节错误明确失败 |

dump 成功时精度列为 `MANUAL_DATA_PREPARED`：ACLNN/E2E 的 precision 为
`MANUAL_DATA_PREPARED`、perf 为 `PASS`；Kernel 三精度槽（dyn/cst/bin）均为
`MANUAL_DATA_PREPARED`、perf 为 `PASS`；GEIR 通路 precision/precision_status 均为
`MANUAL_DATA_PREPARED`。全流程首跑落盘则为正常精度结果。
完成目录迁移后，应核对 direct 与 load 的 case 集合、设备执行状态、精度状态和自定义
compare 结果。

## 9. 已知边界

本协议不使用 manifest 或 sidecar，因此不能自动校验：

- 同名 case 是否仍对应同一 API、attributes、wrapper 或 plugin 版本；
- input bin 在 shape 改变但 dtype 和总元素数不变时是否来自旧 CSV；
- 文件同大小替换或位翻转，因为没有 checksum；
- E2E、ACLNN、Kernel 的同名兼容目录是否被混用；
- seed、生成环境和协议版本历史。

修改 API、attributes、input shape/view、wrapper 或 assets 后应重新落盘。不同入口、
命令或数据版本应使用不同 manual-data 根目录，不要手工重命名 case 内文件。

常见错误：

| 提示 | 检查项 |
| --- | --- |
| `manual-data case ... was not found` | CSV 的完整 `testcase_name` 和搜索根目录 |
| `slot count ... != CSV` | 文件是否漏传、索引是否连续、CSV 是否变化 |
| `filename dtype ... != CSV storage dtype` | dump/load 的 tensor/scalar dtype |
| `saved shape ... != device output shape` | 设备输出 shape 是否回归，是否使用旧 CSV/assets |
| `byte size ... != expected` | 文件是否截断，dtype/shape 是否变化 |
| `unexpected file in manual-data case` | 目录里的确无关文件（日志、JSON、子目录）；调试 dump 的 output/tiling 已按 §3.4 忽略，仍报错即为未知命名 |
