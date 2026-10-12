### 目的

- 单文件覆盖通路功能面，用于用例书写借鉴（用例清单与运行方式：Kernel 见 `kernel/README.md`，其余通路见下方说明）。
- 作为门槛，TTK任何改动必须执行这里的用例。

> Kernel 通路：`kernel/cases.csv`（18 条用例，`remark` 列即测试项；`kernel/golden.py` 为配套 golden 插件，`--plugin` 加载）。
> GEIR 通路：`geir/cases.csv`（13 条用例，`remark` 列即测试项；GEIR 专属 `dyn_input_shapes` 图编译动态 shape（含 `-2` 未知秩）+ TensorList split/foreach/scatter 家族 + int4 低比特 + 历史缺陷回归；`geir/golden.py` 为配套 golden 插件，与 Kernel 共用同一套 TestSpec 契约）。
> ACLNN 通路：`aclnn/cases.csv`（25 条用例，dtype 与基本测试场景全覆盖，`remark` 列即测试项；`aclnn/golden.py` 为配套 golden 插件，`--plugin` 加载）。
> E2E 通路：按框架分文件——`e2e/tf_cases.csv`（TensorFlow，13 条）与 `e2e/torch_cases.csv`（torch/torch_npu，24 条，dtype 全覆盖含 fp8 族），`remark` 列即测试项；含 TF 有状态算子/量化 dtype/value-depend 等 TF 专属链路用例（详见 `e2e/README.md`）。
> 各通路每个用例的「验证特性 + 关键列」详见对应 Case Writing 指南：
> [Kernel](../../docs/Operator_Test_Guides/Kernel_Case_Writing.md) ·
> [GEIR](../../docs/Operator_Test_Guides/GEIR_Case_Writing.md) ·
> [ACLNN](../../docs/Operator_Test_Guides/ACLNN_Case_Writing.md) ·
> [E2E](../../docs/Operator_Test_Guides/E2E_Case_Writing.md)。
