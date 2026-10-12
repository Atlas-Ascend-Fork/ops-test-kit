# TTK —— Operator Test Framework

TTK (ops Test Tool Kit) is a full-pipeline, automated, batch operator testing framework provided by the [CANN](https://hiascend.com/software/cann) operator library. It helps developers quickly perform batch operator functional verification, performance evaluation, and Golden value comparison, improving operator development quality and efficiency.

> Hardware vendor names appearing in this document are for illustration only; TTK is configuration-driven and supports any hardware accelerator that conforms to the interface.

* **Full-stack test paths**: Kernel (AscendC) / GEIR (GE graph) / ACLNN (aclnn\* C API) / E2E (torch/torch_npu) — covering operator to framework layers
* **Multi-device + simulation**: Real devices support NPU / MLU / CPU; simulation mode mimics NPU behavior on CPU, enabling development and debugging without real hardware
* **Multi-card parallel**: Multi-NPU parallel testing
* **Multiple comparison methods**: Mixed tolerance (ecosystem open-source standard, default), statistical relative error, cosine similarity, binary exact, requantization, cross-check
* **Extensible plugins**: Custom Golden / input generation, separate namespaces for Kernel and ACLNN/E2E

[中文文档](./README.md)

## Architecture Layers & Test Coverage

```
┌──────────────────────────────────────────────────┐
│  Framework Layer  ✅ Torch · TensorFlow · ...    │  ← E2E (end-to-end, full pipeline)
├──────────────────────────────────────────────────┤
│  Engine Layer     ✅ GEIR · ✅ ACLNN             │  ← GEIR (GE graph compile + execute) / ACLNN (engine API)
├──────────────────────────────────────────────────┤
│  Operator Layer   ✅ AiCore · AiCpu              │  ← Kernel (operator kernel, compile + execute)
└──────────────────────────────────────────────────┘
```

## Installation

- Install CANN from the Ascend community: [Official Link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_quick.html)
- Python 3.8+ recommended

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

## Quick Start

Define test cases via CSV files (see [Test Case Generation](./docs/Test_Case_Generation.md)) and run them in batch:

```shell
# Show device info
python3 -m ttk info

# Kernel: compile + execute + precision comparison (--plugin provides golden for some cases)
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# GEIR: GE graph compile + execute
python3 -m ttk geir -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py

# ACLNN: aclnn* C API
python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py

# E2E: framework end-to-end (--backend cpu forces CPU backend)
python3 -m ttk e2e -i examples/case_store/e2e/torch_cases.csv
```

Per-case precision status and overall pass rate are printed to the terminal. Add `-o` to output a result CSV. Run `python3 -m ttk kernel --help` for parameters.

## Documentation

### Common

- [Test Case Generation (CSV Format)](./docs/Test_Case_Generation.md) — CSV format, field reference, data types
- [Task Execution](./docs/Task_Execution.md) — CLI parameters, case filtering, device parallelism, advanced scenarios
- [Precision Comparison](./docs/Precision_Comparison.md) (Chinese) — comparison formulas, tolerance config, method selection

### Test Paths

Four test paths map to different layers of the Ascend stack.

| Path | Layer | Guide |
|------|-------|-------|
| Kernel | Operator (AiCore/AiCpu), compile + execute | [Kernel Operator Test Guide](./docs/Operator_Test_Guides/Kernel_Test_Guide.md) |
| GEIR | Engine, GE graph compile + execute | [GEIR Operator Test Guide](./docs/Operator_Test_Guides/GEIR_Test_Guide.md) (Chinese) |
| ACLNN | Engine, aclnn\* C API | [ACLNN Operator Test Guide](./docs/Operator_Test_Guides/ACLNN_Test_Guide.md) |
| E2E | Framework (torch/torch_npu), end-to-end | [E2E Operator Test Guide](./docs/Operator_Test_Guides/E2E_Test_Guide.md) |

### Advanced Scenarios

Offline data prepare, XPU cross-check, deterministic compute, dump debug, NPUSim simulation — see [Task Execution - Advanced Scenarios](./docs/Task_Execution.md#高阶使用场景).

### Tool Adaptation

- [mssanitizer Adaptation](./docs/Operator_Test_Guides/mssanitizer_guide.md) (Chinese) — memory/race/sync detection
- [msopprof Adaptation](./docs/Operator_Test_Guides/msopprof_guide.md) (Chinese) — operator performance profiling
- [msdebug Adaptation](./docs/Operator_Test_Guides/msdebug_guide.md) (Chinese) — operator source-level debugging

### FAQ

- [FAQ](./docs/FAQ/faq_guide.md) — common issues and self-diagnosis

## AI Assistant

TTK ships with Agent Skills that provide TTK usage guidance for CLI-based AI coding assistants (Claude Code, OpenCode, etc.). See [AGENTS.md](./AGENTS.md) for details.

> **Note**: `python3 -m ttk` commands must be run from the ops-test-kit directory.

## Related Information

- [Contributing Guide](CONTRIBUTING.md)
- [Security Statement](SECURITY.md)
- [License](LICENSE)
- [SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

## Contact Us

- **Issue Tracking**: [GitCode Issues](https://gitcode.com/cann/ops-test-kit/issues)
- **Community Discussion**: [GitCode Discussions](https://gitcode.com/cann/ops-test-kit/discussions)
- **Technical Articles**: [GitCode Wiki](https://gitcode.com/cann/ops-test-kit/wiki)
