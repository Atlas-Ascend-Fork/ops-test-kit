<div align="center">

# TTK (Operator Test Tool Kit)

[![repo](https://img.shields.io/badge/docs-Documentation-green.svg?style=flat)](docs)
[![license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![contributing](https://img.shields.io/badge/Contributing-teal)](CONTRIBUTING.md)
[![SIG](https://img.shields.io/badge/SIG-ops--basic-yellow)](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

TTK (ops Test Tool Kit) is a batch operator testing framework in the [CANN](https://hiascend.com/software/cann) operator library. It supports functional verification, performance evaluation, and Golden value comparison at four levels: operator kernel (Kernel / GEIR), engine API (ACLNN), and framework API (E2E, with torch / tensorflow / onnx domains).

[中文文档](./README.md)

</div>

> Hardware vendor names appearing in this document are for illustration only; TTK is configuration-driven and supports any hardware accelerator that conforms to the interface.

## 🚀 Overview

- Test paths: Kernel, GEIR, ACLNN, E2E (torch / tensorflow / onnx domains) across four stack layers (see below)
- Devices: NPU / MLU / CPU, with multi-card parallelism
- NPUSim simulation: mimics NPU behavior on CPU without hardware
- Comparison methods: mixed tolerance (default), statistical relative error, cosine similarity, binary exact, requantization, cross-check
- Plugins: custom Golden and input generation

```
┌─────────────────────────────────────────┐
│  E2E · ONNX · Torch · TensorFlow · ...  │  ← ✅ E2E (end-to-end) at the framework layer: torch / tensorflow / onnx domains
├─────────────────────────────────────────┤
│  GEIR · ACLNN                           │  ← ✅ GEIR (GE graph compile + execute) / ACLNN (engine API) at the engine layer
├─────────────────────────────────────────┤
│  Kernel · AiCore · AiCpu                │  ← ✅ Kernel (operator kernel, compile + execute) at the operator layer
└─────────────────────────────────────────┘
```

## 🛠️ Installation

- Install CANN: [Ascend Community](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_quick.html)
- Python 3.8+
- Toolchains: `make` (Kernel), `g++` (GEIR), `torch`/`torch_npu` (ACLNN/E2E); custom operators need the operator package. See the per-path guides
- The E2E onnx domain needs no torch/tensorflow: it requires the `onnx` package and the CANN `atc` compiler — see the [E2E Operator Test Guide](./docs/operator_test_guides/e2e_test_guide.md)

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

## 🚀 Quick Start

Run from the repository directory. Define test cases in a CSV file (see [Test Case Generation](./docs/test_case_generation_guide.md)) and run them in one command:

```shell
# Show device info
python3 -m ttk info

# Kernel: compile + execute + precision comparison
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# GEIR: GE graph compile + execute
python3 -m ttk geir -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# ACLNN: aclnn* C API
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py

# E2E: framework end-to-end
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv

# E2E onnx domain: single ONNX op (atc-compiled .om + aclmdl execution, no torch/tensorflow needed)
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv
```

## 📖 Documentation

### 📚 Common

<div align="center">

| Document | Description |
|----------|-------------|
| [Test Case Generation](./docs/test_case_generation_guide.md) | CSV / Excel case format and field reference |
| [Task Execution](./docs/task_execution_guide.md) | CLI parameters, device parallelism, advanced scenarios |
| [Precision Comparison](./docs/reference/precision_comparison.md) (Chinese) | Comparison formulas, tolerance config, method selection |

</div>

### 🧭 Test Path Guides

<div align="center">

| Guide | Description |
|-------|-------------|
| [Kernel Operator Test Guide](./docs/operator_test_guides/kernel_test_guide.md) | Operator layer (AiCore/AiCpu), compile + execute |
| [GEIR Operator Test Guide](./docs/operator_test_guides/geir_test_guide.md) (Chinese) | Engine layer, GE graph compile + execute |
| [ACLNN Operator Test Guide](./docs/operator_test_guides/aclnn_test_guide.md) | Engine layer, aclnn\* C API |
| [E2E Operator Test Guide](./docs/operator_test_guides/e2e_test_guide.md) | Framework layer, end-to-end (torch / tensorflow / onnx domains) |

</div>

### 🧰 Tool Integration

<div align="center">

| Document | Description |
|----------|-------------|
| [mssanitizer Integration](./docs/tools/mssanitizer_guide.md) (Chinese) | Memory/race/sync detection |
| [msopprof Integration](./docs/tools/msopprof_guide.md) (Chinese) | Operator performance profiling |
| [msdebug Integration](./docs/tools/msdebug_guide.md) (Chinese) | Operator source-level debugging |

</div>

### 📌 Others

<div align="center">

| Document | Description |
|----------|-------------|
| [FAQ](./docs/faq/faq_guide.md) | Common issues and self-diagnosis |
| [AGENTS](./AGENTS.md) | TTK usage guide for CLI-based AI coding assistants |

</div>

## 💬 Related Information

- [Contributing Guide](CONTRIBUTING.md)
- [Security Statement](SECURITY.md)
- [License](LICENSE)
- [SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

## 🤝 Contact Us

- **Issue Tracking**: [GitCode Issues](https://gitcode.com/cann/ops-test-kit/issues)
- **Community Discussion**: [GitCode Discussions](https://gitcode.com/cann/ops-test-kit/discussions)
- **Technical Articles**: [GitCode Wiki](https://gitcode.com/cann/ops-test-kit/wiki)
