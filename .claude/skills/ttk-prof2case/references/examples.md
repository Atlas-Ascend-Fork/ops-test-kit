# 转换示例（before → after）

取自真实报告的典型行，展示脚本输出与需人工介入的点。目标模式由用户提示词决定（`--mode`，默认 aclnn），§0 为四模式对照，§1-9 为 aclnn 模式示例。

## 0. 同一行 profiling 的四种模式输出

profiling 行：
```
Name=aclnnAdd_AddAiCore_Add, Type=Add,
Input="4718592;4718592" DT_BF16;DT_BF16 NCL;NCL, Output="1,4096,1152" DT_BF16
```

| 模式 | 生成（关键列） |
|---|---|
| `--mode aclnn` | `aclnnAdd, "((4718592,), (4718592,), (1, 4096, 1152))", "(\`bfloat16\`×3)", scalar=('float32',), out=(2,)` — 签名含 out 槽位与 alpha 标量 |
| `--mode kernel` | `add, input=((4718592,), (4718592,)), output=((1, 4096, 1152),)` — kernel I/O 原样，op_name=add |
| `--mode geir` | 同 kernel + `dyn_input_shapes` 留空 |
| `--mode e2e` | `torch.add, "((4718592,), (4718592,),)"` — 输出经返回值比对，不含 out 槽位 |

各模式差异：kernel/geir 保留 GE 内部常量为算子输入（原样）；aclnn 丢弃常量并对齐头文件签名；e2e 只保留 torch 签名的输入张量。

## 1. 简单算子（直接可跑）

profiling 行：
```
Name=aclnnAdd_AddAiCore_Add, Type=Add,
Input Shapes="2,3;2,3", Input Data Types=FLOAT;FLOAT, Formats=ND;ND,
Output Shapes="2,3", Output Data Types=FLOAT
```

生成用例（与 `examples/case_store/aclnn/cases.csv` 结构一致）：
```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,scalar_dtypes,output_tensor_indexes
aclnnAdd_00001,aclnnAdd,"((2, 3), (2, 3), (2, 3))","('float32', 'float32', 'float32')","('float32',)","(2,)"
```

alpha 标量 dtype 按启发式取 float32，值由 TTK 随机生成，无需人工干预。

## 2. Inplace + dtype 转换（Cast 型 InplaceCopy）

profiling 行：
```
Name=aclnnInplaceCopy_CastAiCore_Cast, Type=Cast,
Input="1330" INT64, Output="1330" INT32
```

签名 `aclnnInplaceCopy(selfRef, src)`，输出张量对齐到 `selfRef`：
```csv
aclnnInplaceCopy_00001,aclnnInplaceCopy,"((1330,), (1330,))","('int32', 'int64')",,,"(0,)"
```
`selfRef` 自动进 output_tensor_indexes（inplace），无需人工干预。

## 3. GE 内部常量剔除（Transpose 型 InplaceCopy）

profiling 行：
```
Name=aclnnInplaceCopy_TransposeAiCore_Transpose, Type=Transpose,
Input="128,128,1152;3" DT_BF16;INT64, Output="1152,128,128" DT_BF16
```
第二个输入 (3,) int64 是 perm 常量，签名只有 2 个张量参数：
```csv
aclnnInplaceCopy_00002,aclnnInplaceCopy,"((1152, 128, 128), (128, 128, 1152))","('bfloat16', 'bfloat16')",,,"(0,)"
```
perm 被丢弃（remark 记 `aux-drop:K((3,),int64,ND)`）。真实 perm 值 profiling 不可见，转出的用例按连续排布（0,1,2 交换）处理，布局可能与真实调用不同，属语义近似，抽查核对即可。

## 4. 含 C 参数（需补值，B 档）

profiling 行：
```
Name=aclnnConvolution_Conv2dWithFlag_Conv2DV2, Type=Conv2DV2,
Input="1,3,1024,1024;1152,3,16,16;1152" DT_BF16×3, Output="1,1152,64,64"
```

生成：
```csv
aclnnConvolution_00001,aclnnConvolution,"((1, 3, 1024, 1024), (1152, 3, 16, 16), (1152,), (1, 1152, 64, 64))","('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')","('NCHW', 'NCHW', 'ND', 'NCHW')",,"(3,)"
```
attributes 空 → 校验失败（PARAM_COUNT_MISMATCH）。报告列出需补：
`{'stride':[1,1], 'padding':[0,0], 'dilation':[1,1], 'transposed':False, 'outputPadding':[0,0], 'groups':1, 'cubeMathType':0}`
（stride/padding 等真实值 profiling 不可见，需按业务推断或用户确认）

## 5. GE 内部常量剔除（Slice 型 InplaceCopy）

profiling 行：
```
Name=aclnnInplaceCopy_SliceAiCore_Slice, Type=Slice,
Input="4096,12,72;3;3" DT_BF16;INT64;INT64, Output="4096,4,72"
```
签名 `aclnnInplaceCopy(selfRef, src)`——begin/end 在 kernel 里物化成的 (3,) int64 张量被 aux 剔除。生成：
```csv
aclnnInplaceCopy_00016,aclnnInplaceCopy,"((4096, 4, 72), (4096, 12, 72))","('bfloat16', 'bfloat16')","('NCHW', 'NCHW')",,"(0,)",,...
```
begin/end 被丢弃（remark 记两次 `aux-drop:K((3,),int64,ND)`）。用例**可直接运行**（InplaceCopy 无 C 参数，拷贝语义与切片值无关）；它复现的是切片后张量的拷贝，切片窗口本身不可从 profiling 恢复。

## 6. 输入盈余时的 aux 跳过（LayerNormV4 型 FastLayerNorm）

profiling 行：
```
Name=aclnnFastLayerNorm_LayerNormV4_LayerNormV4, Type=LayerNormV4,
Input="1,1024,1152;1;1152;1152" FLOAT;INT32;FLOAT;FLOAT, Output="1,1024,1152;1,1024,1;1,1024,1"
```
第二个输入 (1,) int32 是 axis 常量（真实 weight/bias 是 (1152,) float）。输入盈余（4 输入 > 3 输入参数）时非 TensorList 槽位跳过 aux，axis 不再挤掉 weight：
```csv
aclnnFastLayerNorm_00001,aclnnFastLayerNorm,"((1, 1024, 1152), (1152,), (1152,), (1, 1024, 1152), (1, 1024, 1), (1, 1024, 1))","('float32', 'float32', 'float32', 'float32', 'float32', 'float32')","('NCL', 'ND', 'ND', 'ND', 'ND', 'ND')",,"(3, 4, 5)",,...
```
axis 记 `aux-drop`。需补 attributes `{normalizedShape, eps}`（weight 形状暗示 normalizedShape=[1152]，值待人工确认）。

## 7. TensorList 吸收（Cat）

profiling 行：
```
Name=aclnnCat_ConcatD_ConcatD, Input="1,1024,1152" ×4 (DT_BF16,NCL), Output="1,4096,1152"
```
签名 `aclnnCat(self TensorList, dim, out)`——4 个输入合并进 TensorList：
```csv
aclnnCat_00001,aclnnCat,"(((1, 1024, 1152), (1, 1024, 1152), (1, 1024, 1152), (1, 1024, 1152)), (1, 4096, 1152))","(('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16'), 'bfloat16')","(('NCL', 'NCL', 'NCL', 'NCL'), 'NCL')",,"(1,)"
```
需补 `{'dim': -1 或 1}`（NCL 布局下拼 dim=1，具体值不可恢复）。

## 8. 复杂融合算子（FusedInferAttentionScore）

q/k/v/mask 对齐到 query/key/value/attenMask（bool→mask 参数优先规则），其余可选参数 None 占位：
```csv
aclnnFusedInferAttentionScore_00001,aclnnFusedInferAttentionScore,
"((1, 4096, 4, 96), ((1, 4096, 4, 96),), ((1, 4096, 4, 96),), None, (1, 1, 4096, 4096), None, ..., (1, 4096, 4, 96), ())",
"('bfloat16', ('bfloat16',), ('bfloat16',), None, 'bool', None, ..., 'bfloat16', 'float32')",
...,,"(15, 16)"
```
需补 14 个 C 参数（numHeads=4、scaleValue、inputLayout='BSH' 等）——其中部分可从头数/shape 推断（numHeads=4 可从 head 维看出），但均属人工决策。softmaxLse 的空 shape（profile 输出末段为空）保留 0-d，报告有「empty output shape in profile」告警，需人工补真实 shape。

## 9. 未解析 API（算子库接入）

```
Name=QkvRmsNormRopeCacheV2, Type=QkvRmsNormRopeCacheV2（本机头文件无 aclnnQkvRmsNormRopeCacheV2）
```
按 kernel 张量原序生成用例 + API_NOT_RESOLVED 警告；校验必失败。处置：安装对应算子包，或 `--op-libs` 指向算子库源码仓（自动递归收集 `**/op_api` 头文件，源码优先于已装版本）后重跑，或删除对应用例。

## 10. 算子库派生 e2e 映射（--op-libs）

profiling 行：
```
Name=UndGenQkvRmsNormRopeCache, Type=UndGenQkvRmsNormRopeCache,
Input="5576,10,128;128;128;440000,128;20117,128,1,128;20117,128,1,128;9676;3,9676" DT_BF16×3;FLOAT32;DT_BF16×2;INT64×2,
Output="4100,10,128;..." DT_BF16...
```

`--op-libs /path/to/ops-transformer` 时：ops-transformer `posembedding/und_gen_qkv_rms_norm_rope_cache/` 目录的 `op_api` 头声明 `aclnnUndGenQkvRmsNormRopeCache`，`torch_extension/` 注册 `und_gen_qkv_rms_norm_rope_cache`——派生映射 + schema 注册后按 12 个张量参数对齐：
```csv
und_gen_qkv_rms_norm_rope_cache_00001,torch.ops.cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache,"((5576, 10, 128), (128,), (128,), (440000, 128), (20117, 128, 1, 128), (20117, 128, 1, 128), (9676,), (3, 9676), (4100, 10, 128), (128,), (128,), (9676,))","('bfloat16', ...)",...
```
无需编译安装 cann_ops_transformer 扩展包即可完成对齐与校验预检（schema 离线注册）；实际执行仍需包可用。

## 11. torch_npu 引擎算子映射（e2e）

profiling 行：
```
Name=MoeGatingTopK, Type=MoeGatingTopK, Input="1394,128;128" FLOAT32;FLOAT32
```
内置映射 `aclnnMoeGatingTopK → torch_npu.npu_moe_gating_top_k`（经 ge-converter 核实）：
```csv
npu_moe_gating_top_k_00001,torch_npu.npu_moe_gating_top_k,"((1394, 128), (128,))","('float32', 'float32')",,,,src=row1334;name=MoeGatingTopK;type=MoeGatingTopK;...;approx-map:k 必填
```
`k` 必填标量列入报告待补值。WeightNz 族（`aclnnQuantMatmulWeightNz`/`aclnnGroupedMatmulWeightNz`/`aclnnGroupedMatmulSwigluQuantWeightNzV2`）映射到同一 torch API（NZ 布局由权重张量 4-D shape 携带），并按 `E2E_PARAM_FILL_ORDER` 填充序落位：

```csv
npu_quant_matmul_00001,torch_npu.npu_quant_matmul,"((1394, 2048), (160, 128, 16, 32), (32, 5120, 2), None, (1394, 32, 2))","('float8_e4m3fn', 'float8_e4m3fn', 'float8_e8m0', None, 'float8_e8m0')","('ND', 'FRACTAL_NZ', 'ND', None, 'ND')",,,src=...;approx-map:...权重为 NZ 布局...
```

x1/x2(NZ 4-D)/scale(权重)/pertoken_scale 与 cann-recipes 网络调用用法一致（offset 槽位 None 占位）。Grad/Backward、Metadata、MC2 通信融合族无 torch 公开接口，属预期跳过。

## 12. 全量算子审计（--audit，可不带 -i）

```shell
python3 prof2case.py --mode e2e --op-libs /path/to/ops-nn:/path/to/ops-transformer --audit -o out/
```

不依赖 profiling，对库内全部 aclnn API 输出可转换性结论：

```
- 库内 aclnn API 总数: 614
- aclnn 模式可解析（头文件签名对齐）: 614（100%）
- e2e 映射覆盖: 383/614（62%）= 内置表 115 + 算子库派生 65 + 同名自动解析 203
- e2e 无映射: 231（清单如下，Grad/Metadata/版本滞后/引擎级通路等；examples/ 示例算子已剪枝）
```
