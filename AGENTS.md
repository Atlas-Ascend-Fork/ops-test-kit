# AGENTS.md - TTK 算子测试指南

## 定位

面向算子开发者的 TTK 使用指南。适用于 Claude Code、OpenCode 等 CLI 类 AI 编程助手。TTK 当前面向昇腾 NPU 单算子测试，支持 Kernel 编译执行、GEIR GE 图编译执行、ACLNN API 测试、E2E 框架 API 对比四种测试模式（E2E 内含 torch/tensorflow/onnx 三种 API：`api_name` 为 `onnx.<OpType>` 时走 atc 编译 .om + aclmdl 执行的 onnx，详见 [E2E算子测试指南](./docs/operator_test_guides/e2e_test_guide.md)）。

## 架构层级

TTK 的四种测试模式对应昇腾技术栈的不同层级，越往上覆盖通路越全：

```
┌─────────────────────────────────────────┐
│  E2E · ONNX · Torch · TensorFlow · ...  │  ← ✅ 应用框架层 E2E（端到端，torch / tensorflow / onnx）
├─────────────────────────────────────────┤
│  GEIR · ACLNN                           │  ← ✅ 引擎层 GEIR（GE图编译+执行）/ ACLNN（引擎 API）
├─────────────────────────────────────────┤
│  Kernel · AiCore · AiCpu                │  ← ✅ 算子层 Kernel（算子内核，覆盖编译+执行）
└─────────────────────────────────────────┘
```

不同层级使用不同的测试用例结构，**测试模式由 CLI 子命令唯一决定**（模式选错会导致用例解析失败；如何按 CSV 内容选择子命令详见各 skill）。输入支持 CSV、Excel(.xlsx/.xlsm) 与 zip（打包多个 csv 一次运行），详见 [用例生成](./docs/test_case_generation_guide.md)。

## 使用方式

执行 TTK 命令需要在本仓库（ops-test-kit）目录下运行：
- Kernel 模式：`python3 -m ttk kernel -i cases.csv`
- GEIR 模式：`python3 -m ttk geir -i cases.csv`
- ACLNN 模式：`python3 -m ttk aclnn -i cases.csv`
- E2E 模式：`python3 -m ttk e2e -i cases.csv`（默认自动探测 NPU；`--backend cpu` 强制 CPU；`api_name` 为 `onnx.<OpType>` 的用例走 onnx：atc 编译 .om 后经 aclmdl 执行，om 恒为静态图）

> 输入支持 CSV、Excel(.xlsx/.xlsm) 与 zip（打包多个 csv）；Excel 默认读首个工作表，用 `--sheet` 指定，如 `ttk kernel -i cases.xlsx --sheet T2`。详见 [任务执行](./docs/task_execution_guide.md)。

验证 NPU 环境：`python3 -m ttk info`

## 技能索引

| 任务 | 技能 | 触发场景 |
|------|------|---------|
| 运行测试 | ttk-how-run-test | 运行ttk、执行测试、查看设备、ttk info/list |
| 编写用例 | ttk-how-write-case | 写用例、生成CSV、CSV字段、TensorList、动态shape |
| 诊断失败 | ttk-how-diagnose | 测试失败、精度不通过、编译报错、执行超时、排障 |
| 编写插件 | ttk-how-write-plugin | TestSpec、__spec__、算子测试规范编写 |
| Profiling转用例 | ttk-prof2case | profiling报告、kernel_details.csv、真实业务shape转用例、msprof导出；支持 kernel/aclnn/e2e/geir 四种目标模式（由用户提示词决定） |

> 读取 `.claude/skills/{技能名}/SKILL.md` 获取详细工作流程。

## 关键路径

- 用例示例：`examples/case_store/`（E2E 按框架分文件：`torch_cases.csv` / `tf_cases.csv` / `onnx_cases.csv`，同一 CSV 不得混放不同框架 API）
- 详细文档：`docs/`
- 插件示例：本仓库 `examples/case_store/kernel/golden.py`、`examples/case_store/aclnn/golden.py`、`examples/plugin/`；外部算子仓库的插件位于其 `tests/assets/` 目录（`golden.py` / `input.py`）

## Commit 规范

- commit message 中不要添加 Co-Authored-By 行
