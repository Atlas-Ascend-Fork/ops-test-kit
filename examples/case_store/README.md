### 用途

- 各通路的用例书写范例（清单与运行方式见各通路 README）
- TTK 回归门槛：任何改动都需跑通这里的用例

> Kernel 通路：`kernel/cases.csv`（18 条用例，`remark` 列即测试项；`kernel/golden.py` 为配套 golden 插件，`--plugin` 加载）。
> GEIR 通路：`geir/cases.csv`（13 条用例，`remark` 列即测试项；GEIR 专属 `dyn_input_shapes` 图编译动态 shape（含 `-2` 未知秩）+ TensorList split/foreach/scatter 家族 + int4 低比特 + 历史缺陷回归；`geir/golden.py` 为配套 golden 插件，与 Kernel 共用同一套 TestSpec 契约）。
> ACLNN 通路：`aclnn/cases.csv`（25 条用例，dtype 与基本测试场景全覆盖，`remark` 列即测试项；`aclnn/golden.py` 为配套 golden 插件，`--plugin` 加载）。
> E2E 通路：按框架分文件——`e2e/tf_cases.csv`（TensorFlow，13 条）、`e2e/torch_cases.csv`（torch/torch_npu，24 条，dtype 全覆盖含 fp8 族）与 `e2e/onnx_cases.csv`（ONNX，7 条，atc 编译 .om + aclmdl 执行的引擎层链路，详见 [E2E 算子测试指南](../../docs/operator_test_guides/e2e_test_guide.md)），`remark` 列即测试项；含 TF 有状态算子/量化 dtype/value-depend 等 TF 专属链路用例（详见 `e2e/README.md`）。
> 各通路每个用例的「验证特性 + 关键列」详见对应 Case Writing 指南：
> [Kernel](../../docs/operator_test_guides/kernel_case_writing_guide.md) ·
> [GEIR](../../docs/operator_test_guides/geir_case_writing_guide.md) ·
> [ACLNN](../../docs/operator_test_guides/aclnn_case_writing_guide.md) ·
> [E2E](../../docs/operator_test_guides/e2e_case_writing_guide.md)。

### CANN 版本配套说明

示例用例分「基线场景」与「较新 dtype/算子」两类。后一类用例是否通过取决于本地 CANN 的算子支持面；不支持时报错签名与「环境装错」不同，可据此区分：

| 通路 | 用例 | 验证特性 | 本地 CANN 不支持时的错误签名 |
|------|------|---------|------------------------------|
| Kernel | `abs_int16_3d` | abs int16 | `E80008: the parameter[x]'s dtype should be one of [...], but actually is [int16]` |
| Kernel | `pad_float8` / `pad_float4` | pad float8_e5m2 / float4_e2m1 | `E80008: ... but actually is [float8_e5m2 / float4_e2m1]` |
| Kernel | `requant_float8` | AscendQuantV2 → float8 | `OPTILING_FAILURE: Parameter dst_type of has incorrect dtype DT_FLOAT8_E5M2. It should be DT_INT8, DT_INT4.` |
| Kernel/GEIR | `matmul_v3_optional_inputs` | MatMulV3 可选输入 | Kernel 通路为 perf-only 无 golden（Comparison 列 UNSUPPORTED，状态记 FAIL，不是 CANN 支持面的问题）；GEIR 通路报 `E19999 Cannot find Node custom infer_data` |
| GEIR | `pad_float8` / `pad_float4` / `quant_int8` / `requant_float8` | Pad / AscendQuantV2 | `EZ3002 Unsupported_Operator`（其中 float4 用例还需 `pip3 install en-dtypes`，缺失时先报 `EN_DTYPES IS NEEDED TO SUPPORT FLOAT4_E2M1 DTYPE`） |
| E2E | `pad_fp8_e4m3` / `pad_fp8_e5m2` / `pad_fp8_e8m0` | torch.nn.functional.pad fp8 族 | `EZ1001: Tensor self not implemented for DT_FLOAT8_*` |
| E2E | `causal_conv1d_fn_basic` | torch.ops.cann_ops_transformer | `EZ1013: JSON configuration file of operator aclnnCausalConv1dFn_* cannot be found`（需部署配套 cann-ops-transformer 自定义算子包；加载 C++ 扩展还需系统 `ninja`） |
| E2E | `add_bool` | bool add（logical or 语义，binary_equal 字节级比对） | 部分版本线 NPU 侧按算术加实现（1+1 字节为 0x02 而非 0x01），比对失败，日志显示 `Expected: True(0x01) Actual: True(0x02)`——这是设备侧 bool 表示缺陷，不是用例或环境问题 |

遇到上表签名，说明本地 CANN 不支持对应 dtype/算子（或缺算子包），属于预期内的失败；升级 CANN 或装包后重跑。基线用例各版本均应通过，失败先查环境。
