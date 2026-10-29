<div align="center">

# TTK（算子测试工具）

[![repo](https://img.shields.io/badge/docs-项目文档-green.svg?style=flat)](docs)
[![license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![contributing](https://img.shields.io/badge/贡献指南-teal)](CONTRIBUTING.md)
[![SIG](https://img.shields.io/badge/SIG-ops--basic-yellow)](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

TTK（ops Test Tool Kit）是 [CANN](https://hiascend.com/software/cann) 算子库提供的批量算子测试框架，支持算子内核（Kernel / GEIR）、引擎接口（ACLNN）与框架 API（E2E，含 torch / tensorflow / onnx）四个层级的算子功能验证、性能评估与 Golden 值比对。

[English Documentation](./README-EN.md)

</div>

> 文档中出现的硬件厂商名仅作示例，TTK 通过配置驱动支持任意符合接口的硬件加速器。

## 🚀 概述

- 测试通路：Kernel、GEIR、ACLNN、E2E（torch / tensorflow / onnx）四个层级（见下图）
- 多设备与多卡：NPU / MLU / CPU，支持多卡并行
- 无卡仿真：NPUSim 在 CPU 上模拟 NPU 行为
- 精度比对：混合容差（默认）、统计相对误差、余弦相似度、二进制精确、重量化、三方交叉校验
- 插件扩展：自定义 Golden 与输入生成

```
┌─────────────────────────────────────────┐
│  E2E · ONNX · Torch · TensorFlow · ...  │  ← ✅ 应用框架层 E2E（端到端，torch / tensorflow / onnx）
├─────────────────────────────────────────┤
│  GEIR · ACLNN                           │  ← ✅ 引擎层 GEIR（GE图编译+执行）/ ACLNN（引擎 API）
├─────────────────────────────────────────┤
│  Kernel · AiCore · AiCpu                │  ← ✅ 算子层 Kernel（算子内核，覆盖编译+执行）
└─────────────────────────────────────────┘
```

## 🛠️ 环境准备

- 安装 CANN 环境：[昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_quick.html)
- Python 3.8+
- 工具链：Kernel 需 `make`、GEIR 需 `g++`、ACLNN/E2E 需 `torch`/`torch_npu`；自定义算子需算子包。详见各通路指南
- E2E onnx 无需 torch/tensorflow，依赖 `onnx`/`ml-dtypes` 包与 CANN `atc`，详见 [E2E 算子测试指南](./docs/operator_test_guides/e2e_test_guide.md)

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

## 🚀 快速开始

在仓库目录下，通过 CSV 定义测试用例（格式见[用例生成](./docs/test_case_generation_guide.md)），一条命令批量运行：

```shell
# 查看设备信息
python3 -m ttk info

# Kernel：编译 + 执行 + 精度比对
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# GEIR：GE 图编译 + 执行
python3 -m ttk geir -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# ACLNN：aclnn* C API
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py

# E2E：框架端到端
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv

# E2E-onnx：ONNX 单算子（atc 编译 .om + aclmdl 执行，无需 torch/tensorflow）
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv
```

## 📖 文档

### 📚 通用文档

<div align="center">

| 文档 | 说明 |
|------|------|
| [用例生成](./docs/test_case_generation_guide.md) | CSV / Excel 用例格式与字段说明 |
| [任务执行](./docs/task_execution_guide.md) | 命令行参数、设备并行、高阶使用场景 |
| [精度比对方法](./docs/reference/precision_comparison.md) | 比对公式、容差配置、方法选择 |

</div>

### 🧭 测试通路指南

<div align="center">

| 指南 | 说明 |
|------|------|
| [Kernel 算子测试指南](./docs/operator_test_guides/kernel_test_guide.md) | 算子层（AiCore/AiCpu），覆盖编译+执行 |
| [GEIR 算子测试指南](./docs/operator_test_guides/geir_test_guide.md) | 引擎层，GE 图编译+执行 |
| [ACLNN 算子测试指南](./docs/operator_test_guides/aclnn_test_guide.md) | 引擎层，aclnn\* C API |
| [E2E 算子测试指南](./docs/operator_test_guides/e2e_test_guide.md) | 应用框架层，端到端（torch / tensorflow / onnx） |

</div>

### 🧰 工具适配

<div align="center">

| 文档 | 说明 |
|------|------|
| [mssanitizer 适配方法](./docs/tools/mssanitizer_guide.md) | 内存/竞争/同步检测 |
| [msopprof 适配方法](./docs/tools/msopprof_guide.md) | 算子性能 Profiling |
| [msdebug 适配方法](./docs/tools/msdebug_guide.md) | 算子源码级调试 |

</div>

### 📌 其他

<div align="center">

| 文档 | 说明 |
|------|------|
| [FAQ 一本通](./docs/faq/faq_guide.md) | 常见问题与自定位 |
| [AGENTS](./AGENTS.md) | CLI 类 AI 编程助手的 TTK 使用指南 |

</div>

## 💬 相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

## 🤝 联系我们

- **问题反馈**：通过 GitCode [【Issues】](https://gitcode.com/cann/ops-test-kit/issues) 提交问题。
- **社区互动**：通过 GitCode [【讨论】](https://gitcode.com/cann/ops-test-kit/discussions) 参与交流。
- **技术专栏**：通过 GitCode [【Wiki】](https://gitcode.com/cann/ops-test-kit/wiki) 获取技术文章，如系列化教程、优秀实践等。
