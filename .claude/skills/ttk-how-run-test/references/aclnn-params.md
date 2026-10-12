# ACLNN 模式专用参数

ACLNN 子命令复用全部通用参数（含共享组的 `--no-prof` 手工数据准备与 `--xpu-perf` XPU 性能采集），无模式专属选项。

> **仿真**：ACLNN 支持 `--backend npusim` 无卡仿真，见 `npusim-params.md`。

## 执行流程

```
读取 CSV → 生成输入张量/标量 → 调用 aclnn* C API → 生成 Golden（CPU） → 精度比对 → 输出结果
```

## 注意事项

- 依赖 CANN 包含 aclnn 头文件和动态库
- `api_name` 必须与 CANN 注册的 aclnn 函数名完全匹配
- 支持 TensorList 输入/输出（通过 `tensor_view_shapes` 嵌套结构表示）
- 原地操作通过 `output_inplace_indexes` 指定
- 标量参数通过 `scalar_dtypes` 指定数据类型
- 精度比对走通用 `--compare`（默认 Spec.tolerance 路由，否则 `mixed`；支持 `mixed`/`stat_rel_err`/`cross_check` 等）
