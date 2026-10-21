# GEIR 用例编写

适用于 `python3 -m ttk geir`，使用 `GeirTestcase`，继承自 Kernel 的 `TestcaseKernel`。CSV 列复用 Kernel 的字段（标准字段共 31 列 = Kernel 的 30 列 + GEIR 专属 `dyn_input_shapes`，另可按需追加 `attributes1`～`attributes9`，全量 40 列；可选列可省略——示例 `cases.csv` 仅用 14 列基础子集）；需要精确指定图编译动态维度时，可选在末尾追加 GEIR 专属的 `dyn_input_shapes` 列。

Kernel 标准示例列及可选字段说明详见 [Kernel用例编写](./Kernel_Case_Writing.md)。

## GEIR 专属字段

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `dyn_input_shapes` | SHAPELIKE_DYN_NESTED | 否 | `None` | 动态编译时输入张量的图编译 desc_shape。动态 shape 编译模式下，未指定时框架自动将所有维度替换为 `-1`；指定后按给定值构建图的 shape 描述。`-1` 表示未知维度，`-2` 表示未知秩（rank）；单输入的秩未知可写为 `((-2,),)`。如 `"((128, -1), (1, 1024))"` 表示第一个输入第 0 维固定 128、第 1 维动态，第二个输入固定。 |

## dyn_input_shapes 与 input_shapes 的关系

| 字段 | 作用 | 示例 |
|------|------|------|
| `input_shapes` | 实际执行时的数据 shape | `"((128, 1024), (1, 1024))"` |
| `dyn_input_shapes` | 图编译时的 desc_shape，`-1` 表示未知维度，`-2` 表示未知秩（rank） | `"((128, -1), (1, 1024))"` |

未指定 `dyn_input_shapes` 时，动态模式下等价于将 `input_shapes` 的全部维度替换为 `-1`（与 Kernel 通路仅替换正数维度的行为不同）。

## 参考用例

GEIR 复用 Kernel 的 CSV 用例（字段与验证场景见 [Kernel 参考用例](./Kernel_Case_Writing.md)），额外用 `dyn_input_shapes` 列控制图编译时的动态维度。本通路独立示例：

| 文件 | 验证特性 | 关键列 |
|------|---------|--------|
| `geir/cases.csv` | `dyn_input_shapes` 全语义（缺省全 `-1` / 尾维固定 / 逐维混合 / `-2` 未知秩 / 嵌套 TensorList 逐元素 / 可选输入槽位 None）+ TensorList split/foreach/scatter 家族（非原地 DYNAMIC_OUTPUT 声明 dtype 落位、原地 desc 继承）+ int4 半字节打包 + 历史缺陷回归（非有限浮点属性、Type 型属性、ValueDepend 值域定值）；配套 `geir/golden.py` 插件与 `geir/README.md` 运行方式 | `dyn_input_shapes` |

> 直接使用 `examples/case_store/kernel/` 下的 CSV 跑 GEIR 无需改动；如需精确控制某输入维度的动态性，在末尾追加 `dyn_input_shapes` 列即可。
>
> 已知限制：`concat_d`（DYNAMIC_INPUT 聚合）在 `-d` 动态编译下报 `EZ1001`（GE 侧 shape 校验），仅 `-c` 可用；其余 TensorList 算子全模式可用。
