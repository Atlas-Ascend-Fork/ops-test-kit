# E2E 模式专用参数

## 专属选项

| 参数 | 说明 | 默认 |
|------|------|------|
| `--backend cpu` | 强制 CPU 执行（跳过 NPU 自动探测）；onnx仅用于 golden/输入生成，设备执行需 NPU | 关闭 |
| `--no-prof` | 仅准备输入/golden，不执行主 API | 关闭 |
| `--eager` | eager（单算子直调）执行模式；`--eager=false` 跳过 eager、精度主判据锚定图模式（需搭配 -c/-d/--aclgraph） | 开启 |
| `--fullgraph` | Graph 模式（torch.compile）；0=禁用，1=启用 | 0 |
| `--aclgraph` | aclgraph 模式（`torch.compile(dynamic=False)` + npugraph_ex backend），算子编译模式随 `-b` | 关闭 |
| `-b`/`--binary` | 二进制复用（默认开启，三种模式全生效）：eager `jitCompile=disable`、`-c`/`-d` GE 图 `ge.jit_compile=0`、aclgraph `static_kernel_compile=False`；`-b=false` 恢复在线编译。不能与 `--super-kernel` 显式同用；npusim 下强制在线 | 开启 |
| `-d`/`--dynamic` | 动态图 | 默认关闭 |
| `-c`/`--const` | 静态图 | 默认关闭 |
| `--core-limit` | 限制核数上限：`N` 仅限 AI 核，`a,v` 分别限制。图模式经 torchair `ge_config.aicore_num` 下发（未设置侧回落物理核数）；aclgraph 模式经 `torch_npu.npu.npugraph_ex.scope.limit_core_num` 下发（每次图捕获均生效）；eager 不支持 | 无 |
| `--super-kernel` | 被测算子走 SuperKernel 编译流程（功能验证）：以 api_name 为 scope 标定（GE 图模式 -c/-d 与 aclgraph 模式均适用）。需搭配 -c/-d/--aclgraph，不支持 --backend cpu；要求在线编译（显式 `-b` 同用报错，隐式默认二进制自动回退在线并告警） | 关闭 |

> **仿真**：E2E 支持 `--backend npusim` 无卡仿真，建议仅用 **eager**（graph/aclgraph/fullgraph 未验证、可能失败），且仅支持 aclnn 算子——非 aclnn 的 legacy 自定义算子（如 `torch_npu.npu_conv2d`）会被 torch_npu 拒绝；仿真要求算子在线编译，`-b` 强制关闭。详见 `npusim-params.md`。

图模式在线编译一旦实际触发并失败（后端可能回退到已有二进制），用例仍标记为失败，并在日志中报告 `KERNEL_COMPILE_FAILURE` 和编译产物目录；即使回退产物通过精度比较，也不能作为 SuperKernel/在线编译验证通过。允许复用已有的有效编译缓存。

## 后端依赖

| 后端 | 依赖 |
|------|------|
| NPU | `torch` + `torch_npu`（默认自动探测）；onnx 不需要 torch——numpy 直通（atc 编译 .om + aclmdl 执行） |
| CPU | `torch`（`--backend cpu` 强制）；onnx CPU 仅用于 golden（ReferenceEvaluator）与输入生成 |

## 执行流程

```
读取 CSV → 生成输入张量 → 在待测后端调用 API → 在 CPU 调用 Golden API → 精度比对 → 输出结果
```

onnx（`onnx.<OpType>`）：TTK 组模单算子 ONNX 模型 → atc 编译 .om（`--precision_mode=must_keep_origin_dtype`，缓存于 `<root>/onnx_cache/<op_type>/`，进程级会话 LRU 复用）→ aclmdl 执行；golden 走 ReferenceEvaluator 跑同一模型产物。

## API 类型

| api_name 示例 | 说明 |
|-------------|------|
| `torch.add` | 函数调用 |
| `torch.nn.functional.relu` | 模块函数 |
| `torch.Tensor.relu_` | Tensor 方法 |
| `onnx.Add` | ONNX 算子（`onnx.<OpType>`；环境 onnx 包 schema 即权威，opset 须 ≤ 18） |

## ONNX 模式约束

- **模式等价**：eager ≡ 静态图（om 恒为静态图）。`--eager -c` 双开时单次执行，EAGER 与 GRAPH CST 两列同值（失败也镜像）
- **快速失败**：`-d` 报 `ONNX_DYNAMIC_GRAPH_UNSUPPORTED`；`--aclgraph` 报 `ONNX_ACLGRAPH_UNSUPPORTED`；`--backend cpu` 执行报 `ONNX_CPU_EXECUTION_UNSUPPORTED`
- **启动期整批拒绝**：`--backend npusim` 报 `ONNX_NPUSIM_UNSUPPORTED`（om 执行需真机）；`--golden-mode XPU` / `--xpu-perf` / `--compare cross_check` 报 `ONNX_XPU_UNSUPPORTED`（三方远端无 onnx 分发通道）；`--dump xpu` 告警跳过；`--golden-mode NPU` 退化同 Enable（Golden 恒为 ReferenceEvaluator）
- **混表限制**：同一 CSV 表内不得混用 torch/tf/onnx API；onnx 进程预载 numpy + onnx，与 torch/tf 互斥
- **perf**：墙钟计时（kernel 明细为空，已知限制）
- **离线数据/dump**：两阶段（`--no-prof --dump in,golden` / `--manual-data-dirs`）与 `--dump`/`--dump-on-fail` 均可用
- 用例编写（输出槽/属性/Optional 省略等）见 `ttk-how-write-case/references/e2e-case.md` 的 ONNX 约定

## golden_api

当被测 API 与 Golden 计算 API 不同时，通过 CSV 的 `golden_api` 字段指定：
- NPU 调用 `api_name`（如 `torch_npu.npu_conv2d`）
- CPU 调用 `golden_api`（如 `torch.nn.functional.conv2d`）
- onnx：`golden_api` 也可写 `onnx.<OpType>`，经 ReferenceEvaluator 跑同一 model_builder 产物（同构参照）；无 golden_api 且无插件时同 API 分发 ReferenceEvaluator
