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

### CANN 版本配套说明

示例用例按「基线场景 + 前沿 dtype/算子覆盖」组织。**前沿用例的通过依赖本地 CANN 版本的算子支持面**——在暂不支持该 dtype/算子的版本线上会失败，且错误签名与「环境装错」明显不同，可据此区分：

| 通路 | 用例 | 验证特性 | 本地 CANN 不支持时的错误签名 |
|------|------|---------|------------------------------|
| Kernel | `abs_int16_3d` | abs int16 | `E80008: the parameter[x]'s dtype should be one of [...], but actually is [int16]` |
| Kernel | `pad_float8` / `pad_float4` | pad float8_e5m2 / float4_e2m1 | `E80008: ... but actually is [float8_e5m2 / float4_e2m1]` |
| Kernel | `requant_float8` | AscendQuantV2 → float8 | `OPTILING_FAILURE: Parameter dst_type of has incorrect dtype DT_FLOAT8_E5M2. It should be DT_INT8, DT_INT4.` |
| Kernel/GEIR | `matmul_v3_optional_inputs` | MatMulV3 可选输入 | Kernel 通路为 perf-only 无 golden（Comparison 列 UNSUPPORTED，状态记 FAIL，非支持面问题）；GEIR 通路报 `E19999 Cannot find Node custom infer_data` |
| GEIR | `pad_float8` / `pad_float4` / `quant_int8` / `requant_float8` | Pad / AscendQuantV2 | `EZ3002 Unsupported_Operator`（其中 float4 用例还需 `pip3 install en-dtypes`，缺失时先报 `EN_DTYPES IS NEEDED TO SUPPORT FLOAT4_E2M1 DTYPE`） |
| E2E | `pad_fp8_e4m3` / `pad_fp8_e5m2` / `pad_fp8_e8m0` | torch.nn.functional.pad fp8 族 | `EZ1001: Tensor self not implemented for DT_FLOAT8_*` |
| E2E | `causal_conv1d_fn_basic` | torch.ops.cann_ops_transformer | `EZ1013: JSON configuration file of operator aclnnCausalConv1dFn_* cannot be found`（需部署配套 cann-ops-transformer 自定义算子包；加载 C++ 扩展还需系统 `ninja`） |
| E2E | `add_bool` | bool add（logical or 语义，binary_equal 字节级比对） | 部分版本线 NPU 侧按算术加实现（1+1 字节为 0x02 而非 0x01），比对判 FAIL 且日志呈 `Expected: True(0x01) Actual: True(0x02)`——为设备侧 bool 规范表示缺陷，非用例或环境问题 |

遇到上表签名时说明本地 CANN 线尚不支持对应 dtype/算子（或缺少配套自定义算子包），**属预期失败而非环境配置错误**；升级 CANN 或部署对应算子包后重跑即可。基线用例（常见 dtype 与常规场景）在各版本线均应通过——若基线用例失败，请优先排查环境安装。
