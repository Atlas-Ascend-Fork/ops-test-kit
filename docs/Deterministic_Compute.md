# 确定性计算与batch一致性

`--deterministic-level`（缩写 `--dl`）控制一次 TTK 执行的确定性等级。level 3 的第一阶段执行
普通精度校验 + 框架内 batch 一致性比对（按声明样本切片哈希，跨用例以 api+seed+切片长度对齐分组）；
不同 seed 关系集合的样本级比对（FA 整批 vs 分批）由 dump 驱动的第二阶段单独完成。

## 等级与优先级

| 等级 | 执行含义 | 结果判据 |
|------|----------|----------|
| `0` | 不启用确定性选项 | 普通精度 |
| `1` | 同一用例的确定性执行 | 通路原有的同用例确定性检查 |
| `2` | 强一致性等级；仅 aclnn/e2e（torch/tf 域）将等级值透传运行时确定性接口（kernel/geir 当前无差异化行为；onnx 域 aclmdl 执行为同步语义，无独立确定性接口） | 通路原有的强确定性检查 |
| `3` | 确定性执行；若完整声明 batch relation，则进入 batch 一致性比对 | 第一阶段普通精度 + 逐样本输出哈希分组比对；第二阶段（可选）raw dump slice 比对 |

可以在 CSV 的 `attributes` 中为单个用例写
`'batch_deterministic_level': 3`。当命令行 `--dl` 为 `0`（默认）时，它与该用例指定
`--dl=3` 等价；非零命令行值始终优先，例如 `--dl=1` 会覆盖 CSV 的 level 3。该属性只决定
确定性执行等级，不会自动构成 batch relation（注：kernel/geir 通路的 level 1 多轮检查读
命令行值，CSV 属性声明的 level 1/2 在这两条通路不触发同用例确定性检查）。

只有 `batch_axis`、`batch_slice_info`、`batch_seed` 三个字段都存在且可解析时，才构成 batch
relation。三者都没有或只填了其中一部分时，level 3 仍正常执行确定性/精度流程，但不进入
batch 一致性比对，也不会因为缺字段失败；执行前会按用例输出一次 warning，列出缺少的字段并
提示继续普通 level-3 执行。prepare/validate 不执行主算子，不输出该执行提示。

## 框架侧片段替换（level 3 输入一致性保证）

batch relation 完整时，四通路输入生成期由框架执行片段替换
（`common/testcase/batch_replace.py`）：

- 整张量先按 `seed + 输入槽 idx` 播种（需给定 `--seed`，跨用例同 shape 同值），再按
  `batch_seed/batch_slice_info/batch_axis` 逐样本区段以独立 `sample seed` 重生成
  （片段替换本身不依赖 `--seed`，仅要求 batch relation 完整）；
- `batch_axis` 是**用户声明的逻辑 shape** 上的轴（aclnn/e2e 视图 shape、kernel/GEIR
  `input_ori_shapes`）；kernel/GEIR 在 ori 态替换后再做 format 变换；
- 同 `(sample_seed, 区段 shape, dtype, data_range, distribution)` 的样本跨用例逐位
  一致——同 `batch_consistency_id` 组内输出哈希比对因此成立（构造性保证）；
- 边界：manual 输入 / 动态 `-1` 槽 / pure_output 槽跳过；aclnn/e2e 的 view 与 storage
  布局错位槽（fp8 存储展平、非连续 stride、非零 offset）warning 跳过；多卡用例的 rank
  输入生成同样经 `generate_np_storages`（rank 模板深拷贝保留 batch 字段，参与片段替换）；
  input/npu_preprocess 插件 kwargs 附带 `batch_applied` 标记（kernel/GEIR 与 aclnn 的
  input 插件及 aclnn/e2e 的 npu_preprocess hook——kernel/GEIR 无该 hook；e2e input
  插件收到 batch_* 字段但不含该标记）。

## 第一阶段比对（框架内，四通路）

worker 比对后，dl=3 且 bcid 非空的用例按**声明的 relation 样本切片**切 output 0
（每个样本把 `batch_slice_info` 的 (start, stop, step) 直接落在 `batch_axis` 声明的轴上），
逐样本 MD5 → `batch_summary`（随 ReturnStructure 回传；样本 key = 各轴 seed 元组 +
各轴切片长度元组）。**只 hash 声明位置**——每个用例声明了几个样本，就 hash 几次；
未声明位置不参与；切片粒度（整 batch / 单 token / 多轴）由声明区间本身决定，
框架不做布局感知。主进程 cleanup 按 key 跨用例分组（key = api_name + seed 元组 +
长度元组，不含声明顺序/位置；多 API 混在同一 CSV 时不串比），同 key 片段 hash
集合基数判定一致/不一致（PASS/FAIL 汇总日志，非结果列判据）；同 seed 样本跨
用例换序（reorder）自动按 key 对齐比较。四通路结果 CSV 均含
`batch_consistency_id` 列（对账用）。

## 第二阶段（dump 驱动，可选）

不同 seed 的跨 relation 集合样本级比对（如 FA 整批一次跑 vs 分批多次跑，两组
声明 seed 不同）由 `scripts/compare_batch_consistency.py` 完成：`--dump out
--dump-format bin` 且设置 `NPU_DUMP_PATH` 执行后，用结果 CSV 运行该脚本比对输出 dump。

> 注意：该脚本要求 dump 文件为平铺命名（`NPU_DUMP_PATH/<用例名或 dump_file_prefix>_output_0.bin`）；
> `--dump` 落盘为 `<case>/output[_<segment>]_0_<dtype>.bin` 目录布局，使用前需按脚本约定整理 dump 文件。
