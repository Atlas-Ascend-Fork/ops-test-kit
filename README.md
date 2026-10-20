# 🚀 TTK —— 算子测试工具

TTK（ops Test Tool Kit）是[CANN](https://hiascend.com/software/cann)算子库提供的全链路、自动化、批量化算子测试框架，帮助开发者快速完成算子批量功能验证、性能评估以及Golden值比对，提升算子开发质量和效率。

> 文档中出现的硬件厂商名仅作示例，TTK 通过配置驱动支持任意符合接口的硬件加速器。

* **全栈测试通路**：Kernel（AscendC）/ GEIR（GE图）/ ACLNN（aclnn* C API）/ E2E（torch/torch_npu），覆盖算子到框架各层
* **多设备 + 仿真**：真实设备支持 NPU / MLU / CPU；仿真模式基于 CPU 模拟 NPU 行为，无需真实硬件即可开发调试
* **多卡并行**：多 NPU 设备并行测试
* **多种精度对比**：混合容差（生态算子开源精度标准，默认）、统计相对误差、余弦相似度、二进制精确、重量化、三方交叉校验
* **可扩展插件**：自定义 Golden / 输入生成，Kernel 与 ACLNN/E2E 独立命名空间

[English Documentation](./README-EN.md)

## 🏗️ 架构层级与测试覆盖

```
┌──────────────────────────────────────────────────┐
│  应用框架层  ✅ Torch · TensorFlow · ...          │  ← E2E（端到端，覆盖全链路）
├──────────────────────────────────────────────────┤
│  引擎层     ✅ GEIR · ✅ ACLNN                   │  ← GEIR（GE图编译+执行）/ ACLNN（引擎 API）
├──────────────────────────────────────────────────┤
│  算子层     ✅ AiCore · AiCpu                     │  ← Kernel（算子内核，覆盖编译+执行）
└──────────────────────────────────────────────────┘
```

## 🛠️ 环境准备

- 按照昇腾社区安装CANN环境：[官方链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_quick.html)
- Python版本建议选择3.8及以上
- Kernel 通路依赖系统工具链 `make`（AscendC 构建链，可用 `make --version` 检查；缺失时 `make: not found` 会埋在编译日志深处并伴随大量 `ld.lld: cannot open ./kernel_meta/*.o`，新版 TTK 已前置拦截并提示安装）
- GEIR 通路依赖系统工具链 `g++`（GE 图编译，可用 `g++ --version` 检查）
- ACLNN 通路额外依赖 `torch`（输入生成/Golden计算），E2E 通路额外依赖 `torch`/`torch_npu`（均不在 `requirements.txt` 中，需按本地 CANN 版本配套安装，参见 [torch_npu 安装指引](https://gitee.com/ascend/pytorch)）。**torch 与 torch_npu 必须主次版本配套**：`pip install ".[e2e-npu]"` 已带上限约束防止解析到配套矩阵之外的组合；经测试组合 `torch==2.12.0` + `torch-npu==2.12.0.post2`（CANN 9.3.0）。若 `import torch_npu` 崩溃退出码 134 并报 `Duplicated key 'pinned_reserve_segment_size_mb'`，即版本不配套，按 [torch_npu 配套矩阵](https://gitee.com/ascend/pytorch) 重装配套版本
- 自带示例用例（`examples/case_store/`）包含 fp8/fp4 等前沿 dtype 覆盖用例，其通过依赖本地 CANN 版本的算子支持面；不支持时报 `E80008`/`EZ1001`/`EZ3002` 等 dtype 不支持类错误，属预期行为而非环境装错，详见[示例用例说明](./examples/case_store/README.md)

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

## 🚀 快速开始

通过 CSV 文件定义测试用例（字段说明详见[用例生成](./docs/Test_Case_Generation.md)），一条命令批量运行：

```shell
# 查看设备信息
python3 -m ttk info

# Kernel：编译 + 执行 + 精度比对（--plugin 提供部分用例的 golden）
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# GEIR：GE 图编译 + 执行
python3 -m ttk geir -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# ACLNN：aclnn* C API
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py

# E2E：框架端到端（--backend cpu 强制 CPU 后端）
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv
```

运行后终端输出每个用例的精度状态和整体通过率，加 `-o` 可输出结果 CSV。参数帮助执行 `python3 -m ttk kernel --help`。

## 📚 使用文档

### 通用

- [用例生成](./docs/Test_Case_Generation.md) — CSV 格式、字段说明、数据类型
- [任务执行](./docs/Task_Execution.md) — 命令行参数、设备并行、高阶场景
- [精度比对方法](./docs/Precision_Comparison.md) — 比对公式、容差配置、方法选择

### 测试通路

四条测试通路对应昇腾技术栈不同层级，详见各通路指南。

| 通路 | 覆盖层级 | 指南 |
|------|---------|------|
| Kernel | 算子层（AiCore/AiCpu），覆盖编译+执行 | [Kernel算子测试指南](./docs/Operator_Test_Guides/Kernel_Test_Guide.md) |
| GEIR | 引擎层，GE 图编译+执行 | [GEIR算子测试指南](./docs/Operator_Test_Guides/GEIR_Test_Guide.md) |
| ACLNN | 引擎层，aclnn\* C API | [ACLNN算子测试指南](./docs/Operator_Test_Guides/ACLNN_Test_Guide.md) |
| E2E | 应用框架层（torch/torch_npu），端到端 | [E2E算子测试指南](./docs/Operator_Test_Guides/E2E_Test_Guide.md) |

### 高阶使用场景

离线数据准备、XPU 三方交叉校验、确定性计算、Dump 调试、NPUSim 仿真等多参数组合场景，详见[任务执行 - 高阶使用场景](./docs/Task_Execution.md#高阶使用场景)。

### 工具适配

- [mssanitizer 适配方法](./docs/Operator_Test_Guides/mssanitizer_guide.md) — 内存/竞争/同步检测
- [msopprof 适配方法](./docs/Operator_Test_Guides/msopprof_guide.md) — 算子性能 Profiling
- [msdebug 适配方法](./docs/Operator_Test_Guides/msdebug_guide.md) — 算子源码级调试

### FAQ

- [FAQ一本通](./docs/FAQ/faq_guide.md) — 常见问题与自定位

## 🤖 AI 辅助

TTK 配备 Agent Skills，为 CLI 类 AI 编程助手（Claude Code、OpenCode 等）提供 TTK 使用指导。详见 [AGENTS.md](./AGENTS.md)。

> **注意**：执行 `python3 -m ttk` 命令需要在 ops-test-kit 目录下运行。

## 💬 相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

## 🤝 联系我们

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-test-kit/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-test-kit/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-test-kit/wiki)获取技术文章，如系列化教程、优秀实践等。
