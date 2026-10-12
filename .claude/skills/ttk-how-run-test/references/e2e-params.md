# E2E 模式专用参数

## 专属选项

| 参数 | 说明 | 默认 |
|------|------|------|
| `--backend cpu` | 强制 CPU 执行（跳过 NPU 自动探测） | 关闭 |
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
| NPU | `torch` + `torch_npu`（默认自动探测） |
| CPU | `torch`（`--backend cpu` 强制） |

## 执行流程

```
读取 CSV → 生成输入张量 → 在待测后端调用 API → 在 CPU 调用 Golden API → 精度比对 → 输出结果
```

## API 类型

| api_name 示例 | 说明 |
|-------------|------|
| `torch.add` | 函数调用 |
| `torch.nn.functional.relu` | 模块函数 |
| `torch.Tensor.relu_` | Tensor 方法 |

## golden_api

当被测 API 与 Golden 计算 API 不同时，通过 CSV 的 `golden_api` 字段指定：
- NPU 调用 `api_name`（如 `torch_npu.npu_conv2d`）
- CPU 调用 `golden_api`（如 `torch.nn.functional.conv2d`）
